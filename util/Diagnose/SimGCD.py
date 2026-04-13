import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds

from model.model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
import model.vision_transformer as vits

import re
import random

import copy

@torch.no_grad()
def _l2_normalize_rows(x, eps=1e-12):
    # x: [N, D]
    return x / (x.norm(dim=1, keepdim=True) + eps)

@torch.no_grad()
def compute_soc(model, labeled_loader, unlabeled_loader, args, k=16):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    feats_old, feats_new = [], []

    # ---- 1) labeled -> old feats ----
    for images, labels, _ in labeled_loader:
        images = images.to(device, non_blocking=True)
        z = model(images).detach().float().cpu()
        feats_old.append(z)

    # ---- 2) unlabeled -> novel feats ----
    train_class_set = set(range(len(args.train_classes)))
    for images, labels, _ in unlabeled_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.cpu().numpy()
        z = model(images).detach().float().cpu()
        for i, y in enumerate(labels):
            if y not in train_class_set:
                feats_new.append(z[i])

    result = {"soc": float('nan')}

    if len(feats_old) == 0 or len(feats_new) == 0:
        print("[SOC] Warning: no valid old/new features found.")
        return result

    Z_old = torch.cat(feats_old, dim=0)    # [N_old, D]
    Z_new = torch.stack(feats_new, dim=0)  # [N_new, D]

    # Normalize feature rows.
    Z_old = _l2_normalize_rows(Z_old)
    Z_new = _l2_normalize_rows(Z_new)

    # SOC computation.
    mu_old = Z_old.mean(dim=0, keepdim=True)
    Z_old_c = Z_old - mu_old
    Z_new_c = Z_new - mu_old

    try:
        _, _, Vhc = torch.linalg.svd(Z_old_c, full_matrices=False)
        Vc = Vhc.T
    except RuntimeError:
        _, _, Vhc = torch.linalg.svd(Z_old_c.cpu(), full_matrices=False)
        Vc = Vhc.T

    Dc, rc = Vc.shape
    den_c = (Z_new_c ** 2).sum().item()

    k_eff = min(int(k), Dc, rc)
    if k_eff > 0 and den_c > 0:
        Vk = Vc[:, :k_eff]
        Z_new_proj_c = Z_new_c @ (Vk @ Vk.T)
        num_c = (Z_new_proj_c ** 2).sum().item()
        result["soc"] = num_c / den_c

    if was_training:
        model.train()
    return result

def _flatten_grads(grad_list):
    """Flatten a list of gradients into one vector and skip None entries."""
    vec = []
    for g in grad_list:
        if g is None:
            continue
        vec.append(g.contiguous().view(-1))
    if len(vec) == 0:
        return None
    return torch.cat(vec)

def _pick_params_for_gdc(model):
    """
    Use only parameters from the last trainable backbone block for GDC.
    Assumes model = nn.Sequential(backbone, projector).
    Print selection info only once.
    """
    if not (isinstance(model, nn.Sequential) and len(model) > 0):
        if not hasattr(_pick_params_for_gdc, "_warned_not_seq"):
            print("[GDC] model is not nn.Sequential, fallback to all trainable params.")
            _pick_params_for_gdc._warned_not_seq = True
        return [p for _, p in model.named_parameters() if p.requires_grad]

    backbone = model[0]

    block_ids = []
    for n, _ in backbone.named_parameters():
        if 'blocks.' in n:
            try:
                block_ids.append(int(n.split('blocks.')[1].split('.')[0]))
            except Exception:
                pass
        elif 'block.' in n:
            try:
                block_ids.append(int(n.split('block.')[1].split('.')[0]))
            except Exception:
                pass

    if len(block_ids) == 0:
        if not hasattr(_pick_params_for_gdc, "_warned_no_block"):
            print("[GDC] No backbone block found, fallback to all trainable params.")
            _pick_params_for_gdc._warned_no_block = True
        return [p for _, p in model.named_parameters() if p.requires_grad]

    last_block_id = max(block_ids)

    picked = []
    for n, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        if f'blocks.{last_block_id}.' in n or f'block.{last_block_id}.' in n:
            picked.append(p)

    if len(picked) == 0:
        if not hasattr(_pick_params_for_gdc, "_warned_no_trainable_last"):
            print(f"[GDC] No trainable params found in last block {last_block_id}, fallback to all trainable params.")
            _pick_params_for_gdc._warned_no_trainable_last = True
        return [p for _, p in model.named_parameters() if p.requires_grad]

    if not hasattr(_pick_params_for_gdc, "_printed_success"):
        print(f"[GDC] Using backbone last block {last_block_id} ({len(picked)} tensors).")
        _pick_params_for_gdc._printed_success = True

    return picked

@torch.no_grad()
def _cosine(a, b, eps=1e-12):
    return torch.clamp(torch.dot(a, b) / (a.norm() * b.norm() + eps), -1.0, 1.0)


def compute_gdc(loss_sup, loss_unsup, model):
    """
    Compute GDC on the last backbone block:
      gL  = grad(loss_sup)
      gLU = grad(loss_sup + loss_unsup)
      GDC = 1 - cos(gL, gLU)
    """
    params = _pick_params_for_gdc(model)
    if len(params) == 0:
        return float('nan')

    gL = torch.autograd.grad(
        outputs=loss_sup,
        inputs=params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    gL = _flatten_grads(gL)
    if gL is None or gL.numel() == 0 or float(gL.norm()) == 0.0:
        return float('nan')

    total = loss_sup + loss_unsup
    gLU = torch.autograd.grad(
        outputs=total,
        inputs=params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    gLU = _flatten_grads(gLU)
    if gLU is None or gLU.numel() == 0 or float(gLU.norm()) == 0.0:
        return float('nan')

    cos_sim = _cosine(gL, gLU)
    gdc = (1.0 - cos_sim).item()
    return gdc

def init_seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract_losses_from_pstr(pstr):
    """
    Extract sub-loss names and values from the progress string.
    Input format example: "cls_loss: 0.5678 cluster_loss: 0.1234 ..."
    Returns a dict mapping loss name (e.g., "cls_loss") to float value.
    """
    loss_dict = {}
    # Match "loss_name: float_value" patterns.
    pattern = r'(\w+_loss): ([0-9]+\.[0-9]+)'
    # Collect all matches.
    matches = re.findall(pattern, pstr)
    # Store matches into the dictionary.
    for loss_name, loss_value in matches:
        loss_dict[loss_name] = float(loss_value)
    return loss_dict

# def train(student, train_loader, test_loader, unlabelled_train_loader, args):
def train(student, train_loader, test_loader, unlabelled_train_loader, labeled_eval_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
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

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0

    gdc_trace = []
    soc_trace = []

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            global_step = epoch * len(train_loader) + batch_idx
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            # for i, label in enumerate(class_labels):
            #     if mask_lab[i]:
            #         print("Test: label labeled", label)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)
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

                # loss = 0
                # loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                # loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

                # Split total loss into supervised and unsupervised parts.
                loss_sup   = args.sup_weight      * (cls_loss + sup_con_loss)
                loss_unsup = (1 - args.sup_weight) * (cluster_loss + contrastive_loss)

                # Compute GDC at the configured sampling frequency.
                gdc_value = None
                if (batch_idx % args.gdc_every == 0) and mask_lab.any():
                    gdc_value = compute_gdc(loss_sup, loss_unsup, student)

                # Merge losses for normal backward/update.
                loss = loss_sup + loss_unsup

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

            # Compute SOC after each configured training step.
            soc_value = None
            if batch_idx % args.soc_every == 0:
                soc_dict = compute_soc(
                    student[0],              # backbone
                    labeled_eval_loader,     # old
                    unlabelled_train_loader, # novel
                    args
                )
                soc_value = soc_dict["soc"]

            if (gdc_value is not None) and np.isfinite(gdc_value):
                gdc_trace.append(float(gdc_value))
            if (soc_value is not None) and np.isfinite(soc_value):
                soc_trace.append(float(soc_value))

            def _fmt_diag(x):
                try:
                    return f"{x:.6f}"
                except Exception:
                    return "nan"

            args.logger.info(
                f"[DIAG] "
                f"gdc={_fmt_diag(gdc_value)} "
                f"soc={_fmt_diag(soc_value)}"
            )

            if (global_step + 1) >= args.max_step:
                mean_gdc = float(np.mean(gdc_trace)) if len(gdc_trace) > 0 else float('nan')
                mean_soc = float(np.mean(soc_trace)) if len(soc_trace) > 0 else float('nan')
                args.logger.info(
                    f"[DIAG_SUMMARY] "
                    f"mean_gdc={mean_gdc:.6f} "
                    f"mean_soc={mean_soc:.6f} "
                )
                return


            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))


        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)


        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
        # Step schedule
        exp_lr_scheduler.step()

def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
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
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    parser.add_argument('--cub_root', type=str, default='')
    parser.add_argument('--osr_split_dir', type=str, default='')
    parser.add_argument('--dino_pretrain_path', type=str, default='')
    parser.add_argument('--cars_root', type=str, default='')
    parser.add_argument('--pets_root', type=str, default='')
    parser.add_argument('--cifar_100_root', type=str, default='')
    parser.add_argument('--aircraft_root', type=str, default='')

    parser.add_argument('--seed', type=int, default=0)


    # Diagnostic sampling frequency.
    parser.add_argument('--gdc_every', type=int, default=1,
                        help='Compute GDC every N batches.')
    parser.add_argument('--soc_every', type=int, default=1,
                        help='Compute SOC every N batches.')
    parser.add_argument('--max_step', type=int, default=200,
                        help='Stop after N steps and report diagnostic means.')

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

    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    backbone = vits.__dict__['vit_base']()
    state_dict = torch.load(args.dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, _ = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    args.logger.info(f"num_labeled_classes {args.num_labeled_classes}")
    args.logger.info(f"num_labeled_classes {args.num_unlabeled_classes}")
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
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)
    labeled_eval_ds = copy.deepcopy(train_dataset.labelled_dataset)
    labeled_eval_ds.transform = test_transform
    labeled_eval_loader = DataLoader(
        labeled_eval_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    # train(model, train_loader, None, test_loader_unlabelled, args)
    train(model, train_loader, None, test_loader_unlabelled, labeled_eval_loader, args)
