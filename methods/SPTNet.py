"""Open-source training entrypoint for the SPTNet baseline."""

import argparse
import os

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import model.vision_transformer as vits
from model.prompters import PadPrompter, PatchPrompter

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import str2bool, get_params_groups, finetune_params, freeze, unfreeze, cosine_lr
from util.cluster_and_log_utils import log_accs_from_preds
from model.model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator

import random
# from config import clip_pretrain_path, dino_pretrain_path

def init_seed_torch(seed=1):
    # The public release keeps the original deterministic setup so reported
    # runs can still be reproduced as closely as possible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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


parser = argparse.ArgumentParser(description='SPTNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

parser.add_argument('--warmup_model_dir', type=str, default=None)
parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
parser.add_argument('--prop_train_labels', type=float, default=0.5)
parser.add_argument('--use_ssb_splits', action='store_true', default=True)

parser.add_argument('--grad_from_block', type=int, default=11)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr2', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--transform', type=str, default='imagenet')
parser.add_argument('--sup_weight', type=float, default=0.35)
parser.add_argument('--n_views', default=2, type=int)
parser.add_argument('--lamb', type=float, default=0.1, help='The balance factor.')
parser.add_argument('--model', type=str, default='dino')

parser.add_argument('--freq_rep_learn', type=int)
parser.add_argument('--prompt_size', type=int)
parser.add_argument('--prompt_type', type=str, default='patch')
parser.add_argument('--pretrained_model_path', type=str)

parser.add_argument('--memax_weight', type=float, default=2)
parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

parser.add_argument('--fp16', action='store_true', default=True)
parser.add_argument('--eval_freq', default=1, type=int)

parser.add_argument('--cub_root', type=str, default='')
parser.add_argument('--osr_split_dir', type=str, default='')
parser.add_argument('--dino_pretrain_path', type=str, default='')
parser.add_argument('--cars_root', type=str, default='')
parser.add_argument('--pets_root', type=str, default='')
parser.add_argument('--cifar_100_root', type=str, default='')
parser.add_argument('--aircraft_root', type=str, default='')
parser.add_argument('--imagenet_root', type=str, default='')

parser.add_argument('--exp_name', default=None, type=str)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pretrain_nlayers', default=1, type=int)

parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--save_best', action='store_true', default=False,
                    help='If set, save the best model during training.')

# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
device = torch.device('cuda')
args = get_class_splits(args)


def construct_gcd_loss(prompter, backbone, projector, images, class_labels, mask_lab, cluster_criterion, epoch, args):
    if prompter is None:
        feats = backbone(images)
        student_proj, student_out = projector(feats)
    else:
        feats = backbone(prompter(images))
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
    contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

    # representation learning, sup
    student_proj1 = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
    student_proj1 = F.normalize(student_proj1, dim=-1)
    sup_con_labels = class_labels[mask_lab]
    sup_con_loss = SupConLoss()(student_proj1, labels=sup_con_labels)

    loss = 0
    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
    loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

    return loss, feats, student_out


def train(prompter, backbone, projector, train_loader, optimizer, optimizer_cls, exp_lr_scheduler, exp_lr_scheduler_cls, cluster_criterion, epoch, args):

    prompter.train()
    backbone.train()
    projector.train()

    num_batches_per_epoch = len(train_loader)
    switch_to_cls = False

    for batch_idx, batch in enumerate(tqdm(train_loader)):

        if (batch_idx + 1) % args.freq_rep_learn == 0: # train classifier
            switch_to_cls = not switch_to_cls

        step = num_batches_per_epoch * epoch + batch_idx
        exp_lr_scheduler(step)

        images, class_labels, uq_idxs, mask_lab = batch
        mask_lab = mask_lab[:, 0]

        class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()

        if switch_to_cls: # train classifier
            freeze(prompter)
            args.grad_from_block = 11
            finetune_params(backbone, args)

            with torch.cuda.amp.autocast(args.fp16_scaler is not None):
                images = torch.cat([images[0].cuda(non_blocking=True), prompter(images[0].cuda(non_blocking=True)).detach()], dim=0)
                loss, feats, outs = construct_gcd_loss(None, backbone, projector, images, class_labels, mask_lab, cluster_criterion, epoch, args)

            optimizer_cls.zero_grad()

            if args.fp16_scaler is None:
                loss.backward()
                optimizer_cls.step()

            else:
                args.fp16_scaler.scale(loss).backward()
                args.fp16_scaler.step(optimizer_cls)
                args.fp16_scaler.update()

        else: # train prompter
            unfreeze(prompter)
            args.grad_from_block = 20 # large enough
            finetune_params(backbone, args)

            with torch.cuda.amp.autocast(args.fp16_scaler is not None):
                images = torch.cat(images, dim=0).cuda(non_blocking=True)
                loss, feats, outs = construct_gcd_loss(prompter, backbone, projector, images, class_labels, mask_lab, cluster_criterion, epoch, args)

            optimizer.zero_grad()

            if args.fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                args.fp16_scaler.scale(loss).backward()
                args.fp16_scaler.step(optimizer)
                args.fp16_scaler.update()

    exp_lr_scheduler_cls.step()


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
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    init_seed_torch(args.seed)

    print(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # Hyper-paramters
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.proj_dim = 256
    args.num_mlp_layers = 3
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_ctgs = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # BASE MODEL
    # ----------------------
    backbone = vits.__dict__['vit_base']().to(device)
    args.patch_size = 16
        
    if args.prompt_type == 'patch':
        args.prompt_size = 1
        prompter = PatchPrompter(args)

    elif args.prompt_type == 'all':
        args.prompt_size = 30
        prompter1 = PadPrompter(args)
        args.prompt_size = 1
        prompter2 = PatchPrompter(args)
        prompter = nn.Sequential(prompter1, prompter2)

    print(args)

    finetune_params(backbone, args) # HOW MUCH OF BASE MODEL TO FINETUNE

    # ----------------------
    # CLS HEAD
    # ----------------------
    # projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    
    # classifier = nn.Sequential(backbone, projector).cuda()
    # model = nn.Sequential(prompter, classifier).cuda()

    ckpt = torch.load(args.pretrained_model_path, map_location='cpu')
    mlp_head = MLP_Head(input_dim=args.feat_dim, output_dim=args.feat_dim, nlayers=args.pretrain_nlayers)
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    projector = nn.Sequential(mlp_head, projector)

    backbone.load_state_dict(ckpt['backbone_state_dict'])
    projector.load_state_dict(ckpt['projector_state_dict'])

    classifier = nn.Sequential(backbone, projector).cuda()
    model = nn.Sequential(prompter, classifier).cuda()

    # ----------------------
    # OPTIMIZATION
    # ----------------------
    optimizer = SGD(get_params_groups(prompter), lr=args.lr, momentum=args.momentum, weight_decay=0)
    optimizer_cls = SGD(get_params_groups(classifier), lr=args.lr2, momentum=args.momentum, weight_decay=args.weight_decay)
    
    args.fp16_scaler = None
    if args.fp16:
        args.fp16_scaler = torch.cuda.amp.GradScaler()

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # CONTRASTIVE TRANSFORM
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    # DATASETS
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, _ = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # ------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    total_steps = len(train_loader) * args.epochs
    exp_lr_scheduler = cosine_lr(optimizer, args.lr, 1000, total_steps)
    exp_lr_scheduler_cls = lr_scheduler.CosineAnnealingLR(
            optimizer_cls,
            T_max=args.epochs,
            eta_min=args.lr2 * 0.1,
        )
    
    # ----------------------
    # TRAIN
    # ----------------------

    for epoch in range(args.epochs):
        print("Epoch: " + str(epoch))

        train(prompter, backbone, projector, train_loader, optimizer, optimizer_cls, exp_lr_scheduler, exp_lr_scheduler_cls, cluster_criterion, epoch, args)
    
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                # Testing on labelled examples
                # all_acc, old_acc, new_acc = test(model, test_loader_labelled, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
                all_acc, old_acc, new_acc = test(model, test_loader_unlabelled, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
                print(f"Train ACC: All {all_acc:.4f}  Old {old_acc:.4f}  New {new_acc:.4f}")

                # ----------- 保存最佳模型 (based on all_acc) -----------
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
                            # 构造保存路径
                            os.makedirs(args.save_path, exist_ok=True)
                            best_dir = os.path.join(args.save_path, "best")
                            os.makedirs(best_dir, exist_ok=True)

                            save_name = f"SPTNet_{args.dataset_name}_best_seed{args.seed}.pth"
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
                            print(f"Model saved to: {save_path}")
                            
                        print(f"[Best ACC Updated] Epoch {epoch}: "
                                        f"All={best_all_acc:.4f}, Old={best_old_acc:.4f}, New={best_new_acc:.4f}")
                # --------------------------------------------------------
