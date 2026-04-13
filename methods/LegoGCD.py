"""Open-source training entrypoint for the LegoGCD baseline."""

import argparse
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from tqdm import tqdm
import torch.nn.functional as F

from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
# from config import exp_root
from model.model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, initial_qhat, update_qhat, causal_inference, WeightedEntropyLoss 

import copy
import model.vision_transformer as vits

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# seed = 1
# np.random.seed(seed)
# random.seed(seed)  # Python random module.
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


# ----------------------
# CONCEPTOR
# ----------------------
@torch.no_grad()
def compute_known_energy_ratio(z, known_subspace, eps=1e-12):
    """
    e_old = (z^T C z) / ||z||^2 ∈ [0,1]
    z: [B, D] feature vectors.
    known_subspace: [D, D] conceptor matrix.
    """
    known_subspace = known_subspace.to(dtype=z.dtype, device=z.device)
    Cz = z @ known_subspace          # [B, D]
    num = (z * Cz).sum(dim=-1)       # [B]
    den = (z * z).sum(dim=-1).clamp_min(eps)
    return (num / den).clamp(0.0, 1.0)

@torch.no_grad()
def compute_labeled_energy_reference(backbone, loader, known_subspace):
    vals = []
    backbone.eval()
    # mlp_head.eval()
    device = next(backbone.parameters()).device
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        # z = mlp_head(backbone(images))
        z = backbone(images)
        e = compute_known_energy_ratio(z, known_subspace)  # [B]
        vals.append(e)

    return torch.cat(vals).mean().item()

def compute_conceptor(data, aperture=4):
    I = torch.eye(data.size(0), device=data.device, dtype=data.dtype)
    R = (data @ data.T) / data.size(1)
    C = R @ torch.linalg.inv(R + (aperture**-2) * I)
    return C

@torch.no_grad()
def compute_known_subspace(backbone, labeled_eval_loader, args, aperture=4):
    # Build a soft old-class subspace from labeled features for the later
    # energy-aware projection step.

    device = next(backbone.parameters()).device
    backbone.eval()
    # mlp_head.eval()

    feats = []
    for images, _, _ in labeled_eval_loader:
        images = images.to(device, non_blocking=True)

        z = backbone(images)   # [b_old, D]
        # z = mlp_head(z)        # [b_old, D]                                   
        feats.append(z)

    Z = torch.cat(feats, dim=0).T

    known_subspace = compute_conceptor(Z, aperture=aperture)  # [D, D]
    return known_subspace


class GradientCoordinator:
    def __init__(self, backbone, aga_weight):
        self.enabled = True

        self.known_subspace = None
        self.adaptive_projection_weight = None

        self._reference_features = None
        self._labeled_mask = None
        self._aga_active = False

        self.aga_weight = aga_weight               
        self.handle = backbone.register_forward_hook(self._fwd_hook)

    def set_conceptor(self, C):
        self.known_subspace = C.detach()

    def set_projection_weight(self, adaptive_projection_weight):
        self.adaptive_projection_weight = adaptive_projection_weight.detach()

    def set_aga_alignment(self, labeled_mask, reference_features, scaler_scale: float = 1.0):
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

            def _align(g):
                if (delta is None) or (self._labeled_mask is None):
                    return g
                labeled_mask = self._labeled_mask.to(dtype=g.dtype, device=g.device).unsqueeze(1)
                return g + self._aga_weight * labeled_mask * delta.to(dtype=g.dtype, device=g.device)

            g = _project(g)
            g = _align(g)
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

def train(backbone, projector, train_loader, test_loader, unlabelled_train_loader, args, reference_backbone, gradient_coordinator, labeled_energy_ratio_ema):      
    # params_groups = get_params_groups(student)
    # optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
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
  
    qhat = initial_qhat(class_num=args.num_labeled_classes+args.num_unlabeled_classes)
    
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        # student.train()
        backbone.train()
        projector.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            mask_lab_2v = mask_lab.repeat(2)          # [B_tot]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                #------------------------------Conceptor------------------------------#
                with torch.no_grad():
                    reference_features = reference_backbone(images).detach()
                
                scale_val = float(fp16_scaler.get_scale()) if fp16_scaler is not None else 1.0
                gradient_coordinator.set_aga_alignment(mask_lab_2v, reference_features, scaler_scale=scale_val)
                #------------------------------Conceptor------------------------------#

                # student_proj, student_out = student(images)
                feats = backbone(images)
                student_proj, student_out = projector(feats)
                teacher_out = student_out.detach()
                
                #* clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                #* clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                #* represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                #* representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)
                
                #* Soft loss
                unsup_logits = torch.softmax(student_out/0.1, dim=-1)
                max_probs_, idx = torch.max(unsup_logits, dim=-1)
                mask_all = max_probs_.ge(args.thr)

                mask_lab_p = torch.cat([mask_lab, mask_lab], dim=0)
                labels_p = torch.cat([class_labels, class_labels], dim=0)
                mask_p_true = (mask_lab_p == False) & (mask_all == True)
                mask_p = (mask_lab_p == False) & (mask_all == True) & (labels_p == idx)

                mask_old = torch.zeros_like(mask_p_true)
                mask_old[(idx.lt(args.num_labeled_classes)) & (mask_p_true == True)] = 1
                mask_old_num = torch.sum(mask_old.int()).item()

                #* Unlabeled True
                mask_condidate_unlabeled = (mask_lab_p == False) & (mask_all == True)
                mask_p_unlabeled_true = (mask_lab_p == False) & (mask_all == True) &(labels_p == idx)
                mask_p_unlabeled_true_old = (mask_lab_p == False) & (mask_all == True) &(labels_p == idx) & (idx.lt(args.num_labeled_classes))
                mask_p_unlabeled_true_novel = (mask_lab_p == False) & (mask_all == True) &(labels_p == idx) & (idx.ge(args.num_labeled_classes))
                
                #* Unlabeled True old ro novel                                                                                            
                Unlabeled_true_to_old = torch.sum(mask_p_unlabeled_true_old.int()).item()
                Unlabeled_true_to_novel = torch.sum(mask_p_unlabeled_true_novel.int()).item()
                                                          
                mask_p_unlabeled_wrong = (mask_lab_p == False) & (mask_all == True) &(labels_p != idx)

                Unlabeled_condiate = torch.sum(mask_condidate_unlabeled.int()).item()
                Unlabeled_true = torch.sum(mask_p_unlabeled_true.int()).item()
                Unlabeled_wrong = torch.sum(mask_p_unlabeled_wrong.int()).item()
            
                #* Unlabeled Wrong
                idx_ = labels_p[mask_p_unlabeled_wrong]
                idx_pre = idx[mask_p_unlabeled_wrong]
                
                idx_num_old = idx_.lt(args.num_labeled_classes)
                idx_num_novel = idx_.ge(args.num_labeled_classes)
                
                idx_num_novel_to_old = idx_.ge(args.num_labeled_classes) & idx_pre.lt(args.num_labeled_classes)
                idx_num_novel_to_novel = idx_.ge(args.num_labeled_classes) & idx_pre.ge(args.num_labeled_classes)

                num_wrong_old = torch.sum(idx_num_old.int()).item()
                num_wrong_novel = torch.sum(idx_num_novel.int()).item()
                
                num_wrong_novel_to_old = torch.sum(idx_num_novel_to_old.int()).item()
                num_wrong_novel_to_novel = torch.sum(idx_num_novel_to_novel.int()).item()

                
                pseudo_label = torch.softmax((student_out/0.05), dim=-1)
                delta_logits = torch.log(qhat)
                #logits_u = student_out/0.05
                logits_u = student_out/0.05 + 0.4*delta_logits
                log_pred = F.log_softmax(logits_u, dim=-1)
                nll_loss = torch.sum(-pseudo_label*log_pred, dim=1)*mask_old
                
                qhat = update_qhat(torch.softmax(student_out.detach(), dim=-1), qhat, momentum=args.qhat_m)
                
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'nll_loss: {nll_loss.mean().item():.4f} '
               
                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                loss += 2*nll_loss.mean()

            with torch.no_grad():
                e_all = compute_known_energy_ratio(feats, gradient_coordinator.known_subspace)  # [B*views]

                mask_lab_2v = mask_lab.repeat(args.n_views)     # [B*views]
                if mask_lab_2v.any():
                    mu_batch = e_all[mask_lab_2v].mean().item()
                    labeled_energy_ratio_running = mu_momentum * labeled_energy_ratio_running + (1 - mu_momentum) * mu_batch

                mu_ref = labeled_energy_ratio_running
                ratio = (1.0 - e_all / (mu_ref + eps_mu)).clamp(0.0, 1.0)  # [B*views]

                tau_base = args.eep_weight
                tau_b = tau_base * ratio * (~mask_lab_2v).float()          # [B*views]
                gradient_coordinator.set_projection_weight(tau_b)
                
            # Train acc
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
                # args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                #             .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

                args.logger.info('Epoch: [{}][{}/{}]\t{}'
                                .format(epoch, batch_idx, len(train_loader), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(backbone, projector, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
        
        # Step schedule
        exp_lr_scheduler.step()

        # Save the best model according to all_acc.
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

                    save_name = f"LegoGCD_{args.dataset_name}_best_seed{args.seed}.pth"
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
        # --------------------------------------------------------


def test(backbone, projector, test_loader, epoch, save_name, args):

    # model.eval()
    backbone.eval()
    projector.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            # _, logits = model(images)
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
    parser.add_argument('--thr', type=float, default=0.7)
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

    parser.add_argument('--masked-qhat', action='store_true', default=False,
                    help='update qhat with instances passing a threshold')
    parser.add_argument('--qhat_m', default=0.999, type=float,
                    help='momentum for updating q_hat')
    parser.add_argument('--e_cutoff', default=-5.4, type=int)
    parser.add_argument('--use_marginal_loss', default=False)
    parser.add_argument('--tau', default=0.4, type=float)

    parser.add_argument('--cub_root', type=str, default='')
    parser.add_argument('--osr_split_dir', type=str, default='')
    parser.add_argument('--dino_pretrain_path', type=str, default='')
    parser.add_argument('--cars_root', type=str, default='')
    parser.add_argument('--pets_root', type=str, default='')
    parser.add_argument('--cifar_100_root', type=str, default='')
    parser.add_argument('--aircraft_root', type=str, default='')
    parser.add_argument('--imagenet_root', type=str, default='')

    parser.add_argument("--cineca", action="store_true", help="use cineca gpu")

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--reference_epochs', default=100, type=int, help='Number of reference model training epochs')
    parser.add_argument('--reference_batch_size', default=128, type=int)
    parser.add_argument('--reference_lr', type=float, default=0.1, help='Learning rate for reference model training')
    parser.add_argument('--reference_momentum', type=float, default=0.9, help='Momentum for reference model training')
    parser.add_argument('--reference_weight_decay', type=float, default=1e-4, help='Weight decay for reference model training')
    parser.add_argument('--reference_print_freq', default=10, type=int, help='Print frequency for reference model training')
    parser.add_argument('--projection_head_nlayers', default=1, type=int)

    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--reference_model_path', type=str, default='')
    parser.add_argument('--save_best', action='store_true', default=False,
                        help='If set, save the best model during training.')

    parser.add_argument('--lr_backbone',   type=float, default=0.1)
    parser.add_argument('--aga_weight', type=float, default=0.1, help='')
    parser.add_argument('--aperture', type=float, default=4.0)
    parser.add_argument('--eep_weight', type=float, default=0.3)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    init_seed_torch(args.seed)
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['legogcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    backbone = vits.__dict__['vit_base']()
    state_dict = torch.load(args.dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    # args.mlp_out_dim = args.num_labeled_classes
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes


    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, train_labelled, labelled_dataset = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args, 
                                                                                         labeledset=True)
    # train_unlabeled = unlabelled_train_examples_test
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
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    reference_backbone = vits.__dict__['vit_base']()
    state_dict = torch.load(args.reference_model_path, map_location='cpu')
    
    reference_backbone.load_state_dict(state_dict)
    print("Load reference model weights from:", args.reference_model_path)
    for m in reference_backbone.parameters():
        m.requires_grad = False

    reference_backbone = reference_backbone.to(device)
    print("Load reference model weights from:", args.reference_model_path)

    backbone.load_state_dict(state_dict)

    for m in backbone.parameters():
        m.requires_grad = False
    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
                print(name)

    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    mlp_head = MLP_Head(input_dim=args.feat_dim, output_dim=args.feat_dim, nlayers=args.projection_head_nlayers)

    # model = nn.Sequential(backbone, projector).to(device)
    backbone = backbone.to(device)
    projector = nn.Sequential(mlp_head, projector).to(device)
    # ----------------------
    # TRAIN
    # ----------------------
    #------------------------------Conceptor------------------------------#
    labeled_eval_ds = copy.deepcopy(train_dataset.labelled_dataset)
    labeled_eval_ds.transform = test_transform
    labeled_eval_loader = DataLoader(
        labeled_eval_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )

    gradient_coordinator = GradientCoordinator(backbone, args.aga_weight)

    known_subspace = compute_known_subspace(backbone=reference_backbone,
                                            labeled_eval_loader=labeled_eval_loader, args=args,
                                            aperture=args.aperture)
    if known_subspace is None:
        known_subspace = torch.zeros(args.feat_dim, args.feat_dim, device=next(backbone.parameters()).device)
    gradient_coordinator.set_conceptor(known_subspace)

    labeled_energy_ratio_ema = compute_labeled_energy_reference(reference_backbone, labeled_eval_loader, known_subspace)
    #------------------------------Conceptor------------------------------#

    train(backbone, projector, train_loader, test_loader_labelled, test_loader_unlabelled, args, reference_backbone, gradient_coordinator, labeled_energy_ratio_ema)
