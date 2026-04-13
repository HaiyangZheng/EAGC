"""Open-source training entrypoint for the SimGCD baseline."""

import argparse
import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
import model.vision_transformer as vits_dino
import model.vision_transformer2 as vits_dinov2
from model.model import (
    ContrastiveLearningViewGenerator,
    DINOHead,
    DistillLoss,
    SupConLoss,
    get_params_groups,
    info_nce_logits,
)
from util.cluster_and_log_utils import log_accs_from_preds
from util.general_utils import AverageMeter, init_experiment


# ----------------------
# BACKBONE
# ----------------------

def build_backbone(args):
    # Keep the original backbone construction logic unchanged while exposing
    # the existing switch between DINO and DINOv2 backbones.
    if args.backbone_type == 'dino':
        return vits_dino.__dict__['vit_base']()
    elif args.backbone_type == 'dinov2':
        return vits_dinov2.vit_base(
            patch_size=args.dinov2_patch_size,
            num_register_tokens=args.dinov2_num_register_tokens,
        )
    else:
        raise ValueError(f"Unsupported backbone_type: {args.backbone_type}")


def load_backbone_ckpt(model, ckpt_path, args):
    # DINO and DINOv2 checkpoints store weights under different key layouts,
    # so the state dict is normalized before loading.
    state = torch.load(ckpt_path, map_location='cpu')

    if args.backbone_type == 'dino':
        if isinstance(state, dict) and 'teacher' in state:
            state_dict = {
                k.replace("backbone.", ""): v
                for k, v in state['teacher'].items()
            }
        else:
            state_dict = state
        msg = model.load_state_dict(state_dict, strict=True)

    elif args.backbone_type == 'dinov2':
        if isinstance(state, dict) and 'model' in state:
            state_dict = state['model']
        else:
            state_dict = state
        msg = model.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(f"Unsupported backbone_type: {args.backbone_type}")

    return model


# ----------------------
# UTILS
# ----------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ----------------------
# CONCEPTOR
# ----------------------
@torch.no_grad()
def compute_known_energy_ratio(z, known_subspace, eps=1e-12):
    """
    E_old(z) = (z^T S_known z) / ||z||^2 ∈ [0,1]
    z: [B, D] feature vectors.
    known_subspace: [D, D] conceptor-defined known-class subspace.
    """
    subspace = known_subspace.to(dtype=z.dtype, device=z.device)
    projected = z @ subspace
    num = (z * projected).sum(dim=-1)
    den = (z * z).sum(dim=-1).clamp_min(eps)
    return (num / den).clamp(0.0, 1.0)


@torch.no_grad()
def compute_labeled_energy_reference(backbone, loader, known_subspace):
    """Estimate the mean known-subspace energy ratio on labeled data."""
    vals = []
    backbone.eval()
    device = next(backbone.parameters()).device
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        z = backbone(images)
        e = compute_known_energy_ratio(z, known_subspace)
        vals.append(e)

    return torch.cat(vals).mean().item()


def compute_conceptor(data, aperture=4):
    """Compute a conceptor matrix from feature activations."""
    I = torch.eye(data.size(0), device=data.device, dtype=data.dtype)
    R = (data @ data.T) / data.size(1)
    C = R @ torch.linalg.inv(R + (aperture**-2) * I)
    return C


@torch.no_grad()
def compute_known_subspace(backbone, labeled_eval_loader, args, aperture=4):
    """Build the known-class subspace from labeled feature activations."""

    device = next(backbone.parameters()).device
    backbone.eval()

    feats = []
    for images, _, _ in labeled_eval_loader:
        images = images.to(device, non_blocking=True)
        z = backbone(images)
        feats.append(z)

    # compute_conceptor expects features in [D, N] layout.
    Z = torch.cat(feats, dim=0).T

    known_subspace = compute_conceptor(Z, aperture=aperture)
    return known_subspace


class GradientCoordinator:
    """Compose projection and alignment terms on backbone feature gradients."""

    def __init__(self, backbone, aga_weight):
        self.enabled = True

        self.known_subspace = None
        self.adaptive_projection_weight = None

        # Batch-local alignment targets are refreshed before each forward pass.
        self._reference_features = None
        self._labeled_mask = None
        self._aga_active = False

        self.aga_weight = aga_weight
        self.handle = backbone.register_forward_hook(self._fwd_hook)

    def set_conceptor(self, C):
        """Update the conceptor used by the gradient projector."""
        self.known_subspace = C.detach()

    def set_projection_weight(self, adaptive_projection_weight):
        """Set batch-wise projection strengths."""
        self.adaptive_projection_weight = adaptive_projection_weight.detach()

    def set_aga_alignment(self, labeled_mask, reference_features, scaler_scale: float = 1.0):
        """Register the alignment target for the next backward pass."""
        self._aga_active = False

        if reference_features is None or labeled_mask is None:
            return
        N = int(labeled_mask.sum().item())
        D = reference_features.shape[-1]

        if N <= 0:
            return

        self._reference_features = reference_features.detach()
        self._labeled_mask = labeled_mask.detach()
        self._aga_weight = (float(self.aga_weight) / (N * D)) * float(scaler_scale)
        self._aga_active = True

    def _fwd_hook(self, module, inputs, output):
        if (not torch.is_tensor(output)) or (not output.requires_grad) or (not self.enabled):
            return

        delta = None
        if self._aga_active and (self._reference_features is not None):
            reference_features = self._reference_features.to(dtype=output.dtype, device=output.device)
            delta = (output - reference_features).detach()

        def _bwd_hook(grad):
            if not self.enabled:
                return grad
            g = grad

            def _project(g):
                known_subspace = self.known_subspace
                adaptive_projection_weight = self.adaptive_projection_weight
                if (known_subspace is None) or (adaptive_projection_weight is None):
                    return g
                known_subspace = known_subspace.to(dtype=g.dtype, device=g.device)
                projection_weight = adaptive_projection_weight.to(dtype=g.dtype, device=g.device)
                return g - projection_weight.unsqueeze(1) * (g @ known_subspace)

            def _apply_aga(g):
                if (delta is None) or (self._labeled_mask is None):
                    return g
                labeled_mask = self._labeled_mask.to(dtype=g.dtype, device=g.device).unsqueeze(1)
                return g + self._aga_weight * labeled_mask * delta.to(dtype=g.dtype, device=g.device)

            g = _project(g)
            g = _apply_aga(g)
            return g

        output.register_hook(_bwd_hook)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

# ----------------------
# HEADS
# ----------------------

class MLP_Head(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, nlayers=3, hidden_dim=2048):
        super().__init__()

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            layers = [nn.Linear(input_dim, output_dim)]

        elif nlayers != 0:
            layers = [nn.Linear(input_dim, hidden_dim)]
            layers.append(nn.ReLU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x_proj = nn.ReLU()(x_proj)
        return x_proj


def init_seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------
# REFERENCE MODEL
# ----------------------
def train_reference_model(backbone, projector, reference_loader, args):
    """Train the reference model on labeled data only."""
    args.logger.info("Starting reference model training...")

    params_groups = get_params_groups(backbone) + get_params_groups(projector)
    optimizer = SGD(
        params_groups,
        lr=args.reference_lr,
        momentum=args.reference_momentum,
        weight_decay=args.reference_weight_decay
    )

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.reference_epochs,
        eta_min=args.reference_lr * 1e-3,
    )

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.reference_epochs):
        loss_record = AverageMeter()
        backbone.train()
        projector.train()
        for batch_idx, batch in enumerate(reference_loader):
            images, class_labels, _ = batch

            images = torch.cat(images, dim=0).cuda(non_blocking=True)
            class_labels = class_labels.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                image_feat = backbone(images)
                _, student_out = projector(image_feat)

                sup_logits = student_out / 0.1
                sup_labels = torch.cat([class_labels for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                loss = cls_loss

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()

            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq_reference == 0:
                args.logger.info('Reference Epoch: [{}][{}/{}]\t Loss {:.5f}\t cls_loss: {:.4f}'
                            .format(epoch, batch_idx, len(reference_loader), loss.item(), cls_loss.item()))

        args.logger.info('Reference Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        exp_lr_scheduler.step()

    args.logger.info("Reference model training completed!")
    return backbone


# ----------------------
# TRAIN
# ----------------------
def train(backbone, projector, train_loader, test_loader, unlabelled_train_loader, args, reference_backbone, gradient_coordinator, labeled_energy_ratio_ema):
    params_backbone = get_params_groups(backbone)
    params_projector = get_params_groups(projector)
    optimizer = SGD(
        [
            {"params": params_backbone[0]['params'], "lr": args.lr_backbone},
            {"params": params_backbone[1]['params'], "lr": args.lr_backbone, "weight_decay": 0.},
            {"params": params_projector[0]['params'], "lr": args.lr},
            {"params": params_projector[1]['params'], "lr": args.lr, "weight_decay": 0.},
        ],
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    labeled_energy_ratio_running = labeled_energy_ratio_ema
    mu_momentum = 0.95
    eps_mu = 1e-6

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        backbone.train()
        projector.train()
        reference_backbone.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            mask_lab_2v = mask_lab.repeat(2)          # [B_tot]

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                with torch.no_grad():
                    reference_features = reference_backbone(images).detach()

                scale_val = float(fp16_scaler.get_scale()) if fp16_scaler is not None else 1.0
                gradient_coordinator.set_aga_alignment(mask_lab_2v, reference_features, scaler_scale=scale_val)

                feats = backbone(images)
                student_proj, student_out = projector(feats)
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            # Compute the sample-adaptive EEP projection weight.
            with torch.no_grad():
                known_energy_ratio = compute_known_energy_ratio(feats, gradient_coordinator.known_subspace)

                mask_lab_2v = mask_lab.repeat(args.n_views)
                if mask_lab_2v.any():
                    batch_labeled_energy_ratio = known_energy_ratio[mask_lab_2v].mean().item()
                    labeled_energy_ratio_running = (
                        mu_momentum * labeled_energy_ratio_running
                        + (1 - mu_momentum) * batch_labeled_energy_ratio
                    )

                reference_energy_ratio = labeled_energy_ratio_running
                ratio = (1.0 - known_energy_ratio / (reference_energy_ratio + eps_mu)).clamp(0.0, 1.0)

                base_projection_weight = args.eep_weight
                # Labeled samples are excluded from projection.
                adaptive_projection_weight = base_projection_weight * ratio * (~mask_lab_2v).float()
                gradient_coordinator.set_projection_weight(adaptive_projection_weight)

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()


            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t{}'
                                .format(epoch, batch_idx, len(train_loader), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(backbone, projector, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)

        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        exp_lr_scheduler.step()

        # Save the best checkpoint according to all-class training accuracy.
        if epoch == 0:
            best_all_acc = all_acc
            best_old_acc = old_acc
            best_new_acc = new_acc
            best_epoch = epoch
        else:
            if all_acc > best_all_acc:
                best_all_acc = all_acc
                best_old_acc = old_acc
                best_new_acc = new_acc
                best_epoch = epoch

                if args.save_best:
                    os.makedirs(args.save_path, exist_ok=True)
                    best_dir = os.path.join(args.save_path, "best")
                    os.makedirs(best_dir, exist_ok=True)

                    save_name = f"SimGCD_{args.dataset_name}_best_seed{args.seed}.pth"
                    save_path = os.path.join(best_dir, save_name)

                    save_dict = {
                        "epoch": epoch,
                        "backbone_state_dict": backbone.state_dict(),
                        "projector_state_dict": projector.state_dict(),
                        "best_all_acc": best_all_acc,
                        "best_old_acc": best_old_acc,
                        "best_new_acc": best_new_acc,
                    }
                    torch.save(save_dict, save_path)
                    args.logger.info(f"Model saved to: {save_path}")
                    
                args.logger.info(f"[Best ACC Updated] Epoch {epoch}: "
                                 f"All={best_all_acc:.4f}, Old={best_old_acc:.4f}, New={best_new_acc:.4f}")

# ----------------------
# EVAL
# ----------------------
def test(backbone, projector, test_loader, epoch, save_name, args):

    backbone.eval()
    projector.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feats = backbone(images)
            _, logits = projector(feats)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


# ----------------------
# ARGS
# ----------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default='')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    parser.add_argument('--cub_root', type=str, default='')
    parser.add_argument('--osr_split_dir', type=str, default='')
    parser.add_argument('--dino_pretrain_path', type=str, default='')
    parser.add_argument('--cars_root', type=str, default='')
    parser.add_argument('--pets_root', type=str, default='')
    parser.add_argument('--cifar_100_root', type=str, default='')
    parser.add_argument('--aircraft_root', type=str, default='')
    parser.add_argument('--imagenet_root', type=str, default='')
    parser.add_argument('--herbarium_dataroot', type=str, default='')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--reference_epochs', default=100, type=int,
                        help='Number of reference model training epochs')
    parser.add_argument('--reference_batch_size', default=128, type=int)
    parser.add_argument('--reference_lr', type=float, default=0.1,
                        help='Learning rate for reference model training')
    parser.add_argument('--reference_momentum', type=float, default=0.9,
                        help='Momentum for reference model training')
    parser.add_argument('--reference_weight_decay', type=float, default=1e-4,
                        help='Weight decay for reference model training')
    parser.add_argument('--reference_print_freq', dest='print_freq_reference', default=10, type=int,
                        help='Print frequency for reference model training')
    parser.add_argument('--projection_head_nlayers', dest='projection_head_nlayers', default=1, type=int)

    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--reference_model_path', type=str, default='')
    parser.add_argument('--save_best', action='store_true', default=False,
                        help='If set, save the best model during training.')


    parser.add_argument('--lr_backbone',   type=float, default=0.1)
    parser.add_argument("--train_reference_model", dest='use_reference_model_training', action="store_true")

    parser.add_argument('--aga_weight', type=float, default=0.1, help='')
    parser.add_argument('--aperture', type=float, default=4.0)
    parser.add_argument('--eep_weight', type=float, default=0.3)

    parser.add_argument('--estk', action='store_true', default=False)
    parser.add_argument('--ested_k', default=1, type=int)
    
    parser.add_argument('--no_save_reference_model', action='store_false', dest='save_reference_model')
    parser.add_argument('--save_reference_model', action='store_true', dest='save_reference_model', default=True)
    parser.add_argument('--grad_from_block_reference', type=int, default=11)

    parser.add_argument('--backbone_type', type=str, default='dino',
                        choices=['dino', 'dinov2'])
    parser.add_argument('--dinov2_patch_size', type=int, default=14)
    parser.add_argument('--dinov2_num_register_tokens', type=int, default=4)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    init_seed_torch(args.seed)
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = build_backbone(args)
    backbone = load_backbone_ckpt(backbone, args.dino_pretrain_path, args)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))    
        
    args.image_size = 224
    args.feat_dim = backbone.embed_dim if hasattr(backbone, "embed_dim") else 768
    args.num_mlp_layers = 3
    if not args.estk:
        args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
        print("GT: k=",args.mlp_out_dim)
    else:
        args.mlp_out_dim = args.ested_k
        print("ESTK: k=",args.mlp_out_dim)


    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_dataset = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args,
                                                                                         labeledset=True)
    args.logger.info(f"num_labeled_classes {args.num_labeled_classes}")
    args.logger.info(f"num_unlabeled_classes {args.num_unlabeled_classes}")
    args.logger.info(f"Train dataset size: {len(train_dataset)}")
    args.logger.info(f"Unlabelled_train_examples_test size: {len(unlabelled_train_examples_test)}")
    args.logger.info(f"Labelled dataset size: {len(train_dataset.labelled_dataset)}")
    args.logger.info(f"Unlabelled dataset size: {len(train_dataset.unlabelled_dataset)}")
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    reference_loader = DataLoader(
        labelled_dataset,
        num_workers=args.num_workers,
        batch_size=args.reference_batch_size,
        shuffle=True,
    )
    
    # ----------------------
    # REFERENCE MODEL
    # ----------------------

    # If a reference-model checkpoint exists, reuse it; otherwise train it.
    if os.path.exists(args.reference_model_path) and os.path.isfile(args.reference_model_path):
        args.use_reference_model_training = False
    else:
        args.use_reference_model_training = True

    if args.use_reference_model_training:
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)

        for m in backbone.parameters():
            m.requires_grad = False

        # Only unfreeze blocks from grad_from_block_reference onward.
        for name, m in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block_reference:
                    m.requires_grad = True
                    print(name)
        
        backbone = backbone.to(device)
        projector = projector.to(device)
        backbone = train_reference_model(backbone, projector, reference_loader, args)

        # Save the reference backbone weights to CPU without moving the live model.
        if args.save_reference_model:
            os.makedirs(args.save_path, exist_ok=True)
            reference_dir = os.path.join(args.save_path, "pretrain")
            os.makedirs(reference_dir, exist_ok=True)
            
            reference_backbone_path = os.path.join(
                reference_dir,
                f"SimGCD_block{args.grad_from_block_reference}_seed{args.seed}.pth"
            )
    
            torch.save({k: v.detach().cpu() for k, v in backbone.state_dict().items()}, reference_backbone_path)
            print(f"Saved backbone (reference model) -> {reference_backbone_path}")


    # ----------------------
    # TRAIN
    # ----------------------
    reference_backbone = build_backbone(args)

    if not args.use_reference_model_training:
        state_dict = torch.load(args.reference_model_path, map_location='cpu')
        backbone.load_state_dict(state_dict)

        reference_backbone.load_state_dict(state_dict)
        for m in reference_backbone.parameters():
            m.requires_grad = False
        reference_backbone = reference_backbone.to(device)
    else:
        reference_backbone.load_state_dict(backbone.state_dict())
        for m in reference_backbone.parameters():
            m.requires_grad = False
        reference_backbone = reference_backbone.to(device)


    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    mlp_head = MLP_Head(input_dim=args.feat_dim, output_dim=args.feat_dim, nlayers=args.projection_head_nlayers)

    for m in backbone.parameters():
        m.requires_grad = False
    # Only unfreeze blocks from grad_from_block onward.
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
                print(name)


    backbone = backbone.to(device)
    projector = nn.Sequential(mlp_head, projector).to(device)

    # ----------------------
    # CONCEPTOR INIT
    # ----------------------
    labeled_eval_ds = copy.deepcopy(train_dataset.labelled_dataset)
    labeled_eval_ds.transform = test_transform
    labeled_eval_loader = DataLoader(
        labeled_eval_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )

    gradient_coordinator = GradientCoordinator(backbone, args.aga_weight)

    known_subspace = compute_known_subspace(
        backbone=reference_backbone,
        labeled_eval_loader=labeled_eval_loader,
        args=args,
        aperture=args.aperture,
    )
    if known_subspace is None:
        known_subspace = torch.zeros(args.feat_dim, args.feat_dim, device=next(backbone.parameters()).device)
    gradient_coordinator.set_conceptor(known_subspace)

    labeled_energy_ratio_ema = compute_labeled_energy_reference(
        reference_backbone,
        labeled_eval_loader,
        known_subspace,
    )

    train(
        backbone,
        projector,
        train_loader,
        None,
        test_loader_unlabelled,
        args,
        reference_backbone,
        gradient_coordinator,
        labeled_energy_ratio_ema,
    )
