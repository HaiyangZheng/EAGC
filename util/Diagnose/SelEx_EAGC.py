import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from util.general_utils import AverageMeter

import model.vision_transformer as vits
from model.model import get_params_groups

from util.general_utils_selex import init_experiment, get_mean_lr, str2bool

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

from torch.nn import functional as F
import torch.nn as nn

from util.cluster_and_log_utils import log_accs_from_preds

# from matplotlib import pyplot as plt
from util.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

from kmeans_pytorch import kmeans

# TODO: Debug
import warnings
import math
import random
import copy
warnings.filterwarnings("ignore")
import re


@torch.no_grad()
def _l2_normalize_rows(x, eps=1e-12):
    return x / (x.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def compute_soc(model, labeled_loader, unlabeled_loader, args, k=16):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    feats_old, feats_new = [], []

    # labeled -> old feats
    for images, labels, _ in labeled_loader:
        images = images.to(device, non_blocking=True)
        z = model(images).detach().float().cpu()
        feats_old.append(z)

    # unlabeled -> novel feats
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
        if was_training:
            model.train()
        return result

    Z_old = torch.cat(feats_old, dim=0)
    Z_new = torch.stack(feats_new, dim=0)

    Z_old = _l2_normalize_rows(Z_old)
    Z_new = _l2_normalize_rows(Z_new)

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
    Assumes model = nn.Sequential(backbone, projection_head).
    """
    if not (isinstance(model, torch.nn.Sequential) and len(model) > 0):
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


def extract_losses_from_pstr(pstr):
    loss_dict = {}
    pattern = r'(\w+_loss): ([0-9]+\.[0-9]+)'
    matches = re.findall(pattern, pstr)
    for loss_name, loss_value in matches:
        loss_dict[loss_name] = float(loss_value)
    return loss_dict

#------------------------------Conceptor------------------------------#
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
    vals = []
    backbone.eval()
    device = next(backbone.parameters()).device
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
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
    # The known-class conceptor summarizes the reference subspace that EEP tries
    # to suppress for likely novel unlabeled samples.

    device = next(backbone.parameters()).device
    backbone.eval()

    feats = []
    for images, _, _ in labeled_eval_loader:
        images = images.to(device, non_blocking=True)

        z = backbone(images)
        feats.append(z)

    Z = torch.cat(feats, dim=0).T

    known_subspace = compute_conceptor(Z, aperture=aperture)
    return known_subspace


class GradientCoordinator:
    def __init__(self, backbone, aga_weight):
        # The hook keeps the original SelEx objective untouched and only
        # injects AGA / EEP corrections into feature gradients.
        self.enabled = True

        self.known_subspace = None
        self.adaptive_projection_weight = None

        self._reference_features = None # Reference features, shape [B, D]
        self._labeled_mask = None       # Labeled mask, shape [B] bool
        self._aga_active = False

        self.aga_weight = aga_weight
        self.handle = backbone.register_forward_hook(self._fwd_hook)

    def set_conceptor(self, C):
        self.known_subspace = C.detach()

    def set_projection_weight(self, projection_weight):
        self.adaptive_projection_weight = projection_weight.detach()

    def set_aga_alignment(self, mask_lab, reference_features, scaler_scale: float = 1.0):
        self._aga_active = False

        if reference_features is None or mask_lab is None:
            return
        N = int(mask_lab.sum().item()) # number of feats
        D = reference_features.shape[-1]               # shape of feats

        if N <= 0:
            return

        self._reference_features = reference_features.detach()
        self._labeled_mask = mask_lab.detach()
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
                projection_weight = self.adaptive_projection_weight
                if (known_subspace is None) or (projection_weight is None):
                    return g
                known_subspace = known_subspace.to(dtype=g.dtype, device=g.device)
                projection_weight = projection_weight.to(dtype=g.dtype, device=g.device)
                return g - projection_weight.unsqueeze(1) * (g @ known_subspace)

            def _align(g):
                if (delta is None) or (self._labeled_mask is None):
                    return g
                m = self._labeled_mask.to(dtype=g.dtype, device=g.device).unsqueeze(1)  # [B,1]
                return g + self._aga_weight * m * delta.to(dtype=g.dtype, device=g.device)

            g = _project(g)
            g = _align(g)
            return g

        output.register_hook(_bwd_hook)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
#------------------------------Conceptor------------------------------#



class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits

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

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity,smoothing = 0.5):
        target_smooth = F.one_hot(target,input.size(1)).float()*(1-smoothing) +smoothing*similarity#F.one_hot(similarity,input.size(1)).float()#s1/input.size(0)#coef# / self.num_classes
        return torch.nn.CrossEntropyLoss()(input, target_smooth)




class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None,is_code=False):#, smoothing=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if is_code:
            dist = torch.cdist(anchor_feature, contrast_feature)
            dist=-dist/(dist.sum(dim=1)+1e-10)
        else:
            dist = -torch.cdist(anchor_feature, contrast_feature)

        anchor_dot_contrast = torch.div(dist, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, confusion_factor, args,is_code=False):

    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    if is_code:
        dist=torch.cdist(features, features,p=2)
        similarity_matrix =-dist/(dist.sum(dim=1)+1e-10)
    else:
        similarity_matrix=-torch.cdist(features, features)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)


    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    pos_confs= confusion_factor[labels.bool()].view(confusion_factor.shape[0], -1)


    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg_confs= confusion_factor[~labels.bool()].view(confusion_factor.shape[0], -1)


    logits = torch.cat([positives, negatives], dim=1)
    log_confs = torch.cat([pos_confs, neg_confs], dim=1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels, log_confs


def train_reference_model(backbone, projector, reference_loader, args):
    """
    Train the reference model on labeled data.
    """
    print("Starting reference model training...")
    
    # Optimizer and scheduler.
    params_groups = get_params_groups(backbone) + get_params_groups(projector)
    optimizer = SGD(
        params_groups, 
        lr=args.reference_lr, 
        momentum=args.reference_momentum, 
        weight_decay=args.reference_weight_decay
    )
    
    # Learning-rate scheduler.
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.reference_epochs,
        eta_min=args.reference_lr * 1e-3,
    )
    
    # FP16 support.
    fp16_scaler = None
    # if args.fp16:
    #     fp16_scaler = torch.cuda.amp.GradScaler()
    
    # Reference training loop.
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
                
                # Supervised classification loss.
                sup_logits = student_out / 0.1
                # sup_logits = student_out
                # sup_logits = student_out / 0.5
                sup_labels = torch.cat([class_labels for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                
                loss = cls_loss
            
            # Parameter update.
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            
            # Logging.
            if batch_idx % args.reference_print_freq == 0:
                print('Reference Epoch: [{}][{}/{}]\t Loss {:.5f}\t cls_loss: {:.4f}'
                            .format(epoch, batch_idx, len(reference_loader), loss.item(), cls_loss.item()))
        
        print('Reference Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        # Update learning rate.
        exp_lr_scheduler.step()
        
    print("Reference model training completed!")
    return backbone



def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader,
          merge_train_loader, labeled_eval_loader, args, reference_backbone, gradient_coordinator, labeled_energy_ratio_ema):
    # optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
    #                 weight_decay=args.weight_decay)

    optimizer = SGD([
        {"params": model.parameters(), "lr": args.lr_backbone},          # Backbone
        {"params": list(projection_head.parameters()),
         "lr": args.lr},                                                 # Head
    ],
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()
    best_epoch_lab, best_epoch_comb, best_epoch = 0, 0, 0
    strategy=args.strategy
    cluster_momentum=args.cluster_momentum

    best_stats = []
    Total_loss=[]
    Contrastive_loss=[]

    accuracy_old=[]
    accuracy_new=[]
    accuracy_all=[]

    old_acc_test, best_test_acc_lab = 0, 0
    unsupervised_smoothing = args.unsupervised_smoothing
    train_report_interval = args.train_report_interval
    prototype_extraction_interval=args.prototype_extraction_interval
    Distance=args.distance

    labeled_energy_ratio_running = labeled_energy_ratio_ema
    mu_momentum = 0.95
    eps_mu = 1e-6

    # ----------- Best ACC tracking -----------
    best_all_acc = -1.0
    best_old_acc = -1.0
    best_new_acc = -1.0
    best_epoch = -1
    best_stats = [0.0, 0.0, 0.0]
    # -----------------------------------------
    gdc_trace = []
    soc_trace = []
    global_step = 0


    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        loss_cons_record = AverageMeter()

        with torch.no_grad():
            if epoch%prototype_extraction_interval==0:
                uq_index, all_preds, cluster_protos_list, preds_ind_list,  metrics= \
                    extract_labeled_protos(model,  merge_train_loader, args=args)
                for i in range(len(preds_ind_list)):
                    preds_ind_list[i] = preds_ind_list[i].to(device).long()
                    cluster_protos_list[i] = cluster_protos_list[i].to(device)
                    if Distance == 'cosine':
                        cluster_protos_list[i]=cluster_protos_list[i]/torch.norm(cluster_protos_list[i],dim=1).unsqueeze(1)

                cluster_distances_list=[]
                cluster_radius_list=[]
                for i in range(len(preds_ind_list)):
                    if Distance=='euclidean':
                        cluster_distances = torch.cdist(cluster_protos_list[i], cluster_protos_list[i])
                    else:
                        cluster_distances = torch.matmul(cluster_protos_list[i], cluster_protos_list[i].T)

                    cluster_distances_list.append(cluster_distances.clone())
                    cluster_radius = \
                    (cluster_distances + torch.eye(cluster_distances.shape[0]).to(device) * cluster_distances.max()).min(dim=1)[0] / 2
                    cluster_radius_list.append(cluster_radius.clone())



        projection_head.train()
        model.train()
        reference_backbone.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)

            mask_lab_2v = mask_lab.repeat(2)          # [B_tot]

            # Extract features with base model
            #------------------------------Conceptor------------------------------#
            with torch.no_grad():
                features_backbone_pretrain = reference_backbone(images).detach()
            
            gradient_coordinator.set_aga_alignment(mask_lab_2v, features_backbone_pretrain)
            #------------------------------Conceptor------------------------------#
            features_backbone = model(images)

            features_1, features_2 = features_backbone.detach().chunk(2)
            all_features = torch.cat([features_1, features_2], dim=0)

            features = projection_head(features_backbone)
            features = torch.nn.functional.normalize(features, dim=-1)

            with torch.no_grad():

                confusion_factor=0
                if Distance == 'euclidean':
                    pair_dist = torch.cdist(all_features, all_features)
                else:
                    normalized_feats = all_features/torch.norm(all_features.unsqueeze(1))
                    pair_dist = torch.matmul(normalized_feats, normalized_feats.T)

                n_labeled = args.num_labeled_classes
                if args.estk:
                    n_unlabeled = args.ested_k - args.num_labeled_classes
                    print("Use estedk:", args.ested_k)
                else:
                    n_unlabeled = args.num_unlabeled_classes

                for i in range(len(preds_ind_list)):
                    cluster_labels=(preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]).clone()
                    cluster_indexer = F.one_hot(cluster_labels.long(), n_labeled +n_unlabeled ).float().T
                    n_labeled = max(int(n_labeled / 2), 1)
                    n_unlabeled = max(int(n_unlabeled / 2), 1)

                    cluster_indexer = torch.cat([cluster_indexer,cluster_indexer],dim=1)
                    n_samples = torch.sum(cluster_indexer, dim=1).unsqueeze(1)
                    n_samples[n_samples == 0] = 1

                    if Distance=='euclidean':
                        distance = torch.cdist(all_features, cluster_protos_list[i].float())
                    else:
                        normalized_feats = all_features / torch.norm(all_features.unsqueeze(1))
                        distance = torch.matmul(normalized_feats, cluster_protos_list[i].float().T)

                    cluster_radius_list[i]= (cluster_indexer*distance.T).sum(dim=1)/n_samples.squeeze()\
                                            *(1-cluster_momentum)+cluster_radius_list[i]* cluster_momentum

                    cluster_labels = torch.cat([cluster_labels,cluster_labels])
                    if Distance=='euclidean':
                        if strategy=='zero_one':
                            confusion_factor+=(pair_dist>2*cluster_radius_list[i][cluster_labels]).float()/2 ** i
                        elif strategy=='pair_dist':
                            confusion_factor += pair_dist / 2 ** i
                        elif strategy=='pair_cluster':
                            confusion_factor += distance[:,cluster_labels] / 2 ** i
                        else:
                            pass

                    else:
                        if strategy=='zero_one':
                            confusion_factor += (pair_dist< distance[:,cluster_labels]/2).float()/ 2 ** i
                        elif strategy == 'pair_dist':
                            confusion_factor += -pair_dist / 2 ** i
                        elif strategy=='pair_cluster':
                            confusion_factor += -distance[:,cluster_labels] / 2 ** i

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features
            confusion_factor = (confusion_factor - confusion_factor.min()) / (
                        confusion_factor.max() - confusion_factor.min() + 0.0000001)
            confusion_factor = confusion_factor / confusion_factor.sum(dim=1)
            torch.cuda.empty_cache()

            torch.cuda.empty_cache()
            contrastive_logits, contrastive_labels, similarity= info_nce_logits(features=con_feats, confusion_factor=confusion_factor, args=args)
            contrastive_loss = LabelSmoothingLoss()(contrastive_logits, contrastive_labels, similarity, unsupervised_smoothing)

            f1n, f2n = features.chunk(2)
            semisup_con_feats = torch.cat([f1n.unsqueeze(1), f2n.unsqueeze(1)], dim=1)
            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            
            dimension=semisup_con_feats.shape[-1]
            for i in range(len(preds_ind_list)):
                sup_con_loss += sup_con_crit(semisup_con_feats[:, :, :int(dimension/2**(i+1))],
                                             labels=preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]) / 2**(i+1)

            # Total loss
            loss_unsup = (1 - args.sup_con_weight) * contrastive_loss
            loss_sup   = args.sup_con_weight * sup_con_loss / 2
            loss       = loss_sup + loss_unsup

            # pseudo_supcon_loss = 0
            # dimension=semisup_con_feats.shape[-1]
            # for i in range(len(preds_ind_list)):
            #     pseudo_supcon_loss += sup_con_crit(semisup_con_feats[:, :, :int(dimension/2**(i+1))],
            #                                  labels=preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]) / 2**(i+1)

            # # Total loss
            # loss_unsup = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * pseudo_supcon_loss / 2
            # loss_sup   = args.sup_con_weight * sup_con_loss / 2
            # loss       = loss_sup + loss_unsup


            #######################GDC
            gdc_value = None
            if (batch_idx % args.gdc_every == 0) and mask_lab.any():
                student_for_gdc = torch.nn.Sequential(model, projection_head)
                gdc_value = compute_gdc(loss_sup, loss_unsup, student_for_gdc)
                
            pstr = ''
            pstr += f'sup_loss: {loss_sup.item():.4f} '
            pstr += f'unsup_loss: {loss_unsup.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            #######################GDC

            # ====== Per-sample tau: tau_i = tau_base * clamp(1 - e_i / mu_ref, 0, 1) ======
            with torch.no_grad():
                e_all = compute_known_energy_ratio(features_backbone, gradient_coordinator.known_subspace)  # [B*views]

                # Update EMA of mu if labeled samples exist in this batch.
                mask_lab_2v = mask_lab.repeat(args.n_views)     # [B*views]
                if mask_lab_2v.any():
                    mu_batch = e_all[mask_lab_2v].mean().item()
                    labeled_energy_ratio_running = mu_momentum * labeled_energy_ratio_running + (1 - mu_momentum) * mu_batch

                mu_ref = labeled_energy_ratio_running
                ratio = (1.0 - e_all / (mu_ref + eps_mu)).clamp(0.0, 1.0)  # [B*views]

                tau_base = args.eep_weight
                # Force no projection for labeled samples via (~mask_lab_2v).
                tau_b = tau_base * ratio * (~mask_lab_2v).float()          # [B*views]
                gradient_coordinator.set_projection_weight(tau_b)

            # ====== End per-sample tau ======

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_cons_record.update(loss_unsup.item(), class_labels.size(0))
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t loss {loss.item():.5f}\t {pstr}')
            
            soc_value = None
            if batch_idx % args.soc_every == 0:
                soc_dict = compute_soc(
                    model,
                    labeled_eval_loader,
                    unlabelled_train_loader,
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

            print(
                f"[DIAG] "
                f"gdc={_fmt_diag(gdc_value)} "
                f"soc={_fmt_diag(soc_value)}"
            )

            if (global_step + 1) >= args.max_step:
                mean_gdc = float(np.mean(gdc_trace)) if len(gdc_trace) > 0 else float('nan')
                mean_soc = float(np.mean(soc_trace)) if len(soc_trace) > 0 else float('nan')
                print(
                    f"[DIAG_SUMMARY] "
                    f"mean_gdc={mean_gdc:.6f} "
                    f"mean_soc={mean_soc:.6f} "
                )
                return

            global_step += 1


        print('Epoch: {} Avg Loss: {:.3f} | Constrastive: {:.3f} '.format(epoch, loss_record.avg, loss_cons_record.avg))
        Total_loss.append(loss_record.avg)
        Contrastive_loss.append(loss_cons_record.avg)


        with torch.no_grad():
            if (epoch+1) % train_report_interval == 0:
                print('Testing on unlabelled examples in the training data...')
                all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled',args=args)
            else:
                all_acc, old_acc, new_acc = metrics["all_acc"], metrics["old_acc"], metrics["new_acc"]
        

        # ----------- Save best model (based on all_acc) -----------
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
                    # Build checkpoint path.
                    os.makedirs(args.save_path, exist_ok=True)
                    best_dir = os.path.join(args.save_path, "best")
                    os.makedirs(best_dir, exist_ok=True)

                    save_name = f"SelEx_{args.dataset_name}_best_seed{args.seed}.pth"
                    save_path = os.path.join(best_dir, save_name)

                    save_dict = {
                        "epoch": epoch,
                        "backbone_state_dict": model.state_dict(),
                        "projector_state_dict": projection_head.state_dict(),
                        "best_all_acc": best_all_acc,
                        "best_old_acc": best_old_acc,
                        "best_new_acc": best_new_acc,
                    }

                    torch.save(save_dict, save_path)
                    print(f"Model saved to: {save_path}")
                    
                print(f"[Best ACC Updated] Epoch {epoch}: "
                                 f"All={best_all_acc:.4f}, Old={best_old_acc:.4f}, New={best_new_acc:.4f}")

        # --------------------------------------------------------


            # print('Testing on disjoint test set...')
            # all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader, epoch=epoch, save_name='Test ACC', args=args)

            # accuracy_old.append(old_acc)
            # accuracy_new.append(new_acc)
            # accuracy_all.append(all_acc)


        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        # if (epoch+1) % train_report_interval == 0:
        #     print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))


        # Step schedule
        exp_lr_scheduler.step()

        # torch.save(model.state_dict(), args.model_path)
        # torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')

        # if old_acc_test > best_test_acc_lab:

        #     print(f'Best ACC on new Classes on disjoint test set: {new_acc_test:.4f}...')
        #     #print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     best_stats =[all_acc,old_acc,new_acc]
        #     # torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
        #     print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

        #     # torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
        #     # torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_label_head_best.pt')

        #     best_test_acc_lab = old_acc_test
        #     best_epoch = epoch

        # print('Best Train Epochs:  {}'.format(best_epoch))



    # print('############# Final Reports #############')
    # print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(best_stats[0], best_stats[1], best_stats[2]))
    # print('Best Train Epochs:  {} '.format(best_epoch))
    # plot_a_loss(np.arange(0,int(args.epochs)),np.array(Total_loss), "Total")
    # plot_a_loss(np.arange(0,int(args.epochs)),np.array(Contrastive_loss), "Contrastive")


    # plot_acc(np.arange(0,int(args.epochs)),accuracy_old, accuracy_new, accuracy_all, "features")



# def plot_a_loss(x, y, name):

#     SMALL_SIZE = 14
#     MEDIUM_SIZE = 18
#     BIGGER_SIZE = 20
#     plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
#     plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
#     plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
#     plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#     plt.figure(figsize=(10, 10))
#     plt.title(name+" Loss")
#     plt.grid(color='gray', linestyle='--', linewidth=0.5)
#     plt.plot(x, y, 'cornflowerblue',  linewidth=3)
#     plt.xlim(0, x.shape[0]-1)
#     plt.locator_params(axis='y', nbins=10)
#     plt.xlabel("epochs",)
#     plt.ylabel(" Loss")
#     plt.savefig("Plots/"+name+"_Loss.png")


# def plot_acc(x, old, new, all, name):
#     SMALL_SIZE = 14
#     MEDIUM_SIZE = 18
#     BIGGER_SIZE = 20
#     plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
#     plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
#     plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
#     plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#     plt.figure(figsize=(10, 10))
#     plt.title("Accuracy using "+name+" for clustering")
#     plt.grid(color='gray', linestyle='--', linewidth=0.5)

#     plt.plot(x, old, color= 'cornflowerblue', label='Old', linewidth=3)
#     plt.plot(x, new, color='orangered', label='New', linewidth=3)
#     plt.plot(x, all, color='limegreen', label='All', linewidth=3)
#     plt.xlabel("epochs")
#     plt.ylabel("Accuray")
#     plt.xlim(0,x.shape[0]-1)
#     plt.locator_params(axis='both', nbins=10)
#     plt.legend()
#     plt.savefig("Plots/Accuracy_"+name+".png")


def extract_labeled_protos(model, train_loader, args):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    ids=np.array([])
    mask_cls=np.array([])
    metrics=dict()

    for batch_idx, (images, label, uq_idx, mask_lab_) in enumerate(tqdm(train_loader)):

        images = images[0].cuda()
        label, mask_lab_ = label.to(device), mask_lab_.to(device).bool()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        ids=np.append(ids,uq_idx.cpu().numpy())
        mask = np.append(mask, mask_lab_.cpu().bool().numpy())
        mask_cls = np.append(mask_cls,np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
    mask = mask.astype(bool)
    mask_cls = mask_cls.astype(bool)
    # -----------------------
    # K-MEANS
    # -----------------------
    # print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = targets[mask]  # Get labelled targets
    u_targets = targets[~mask]  # Get unlabelled targets
    n_samples =len(targets)

    if args.unbalanced: cluster_size=None
    else: 
        if args.estk:
            cluster_size=math.ceil(n_samples /(args.ested_k))
            print("Use estedk:", args.ested_k)
        else:
            cluster_size=math.ceil(n_samples /(args.num_labeled_classes + args.num_unlabeled_classes))

    if args.estk:
        kmeanssem = SemiSupKMeans(k=args.ested_k, tolerance=1e-4,
                                max_iterations=10, init='k-means++',
                                n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                                mode=None, protos=None,cluster_size=cluster_size)
        print("Use estedk:", args.ested_k)
    else:
        kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                                max_iterations=10, init='k-means++',
                                n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                                mode=None, protos=None,cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeanssem.labels_
    mask_cls=mask_cls[~mask]
    preds = all_preds.cpu().numpy()[~mask]
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls,
                                                    eval_funcs=args.eval_funcs,
                                                    save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
    metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
    prototype_higher=[]
    prototypes = kmeanssem.cluster_centers_
    prototype_higher.append(prototypes.clone())
    n_labeled=args.num_labeled_classes
    
    if args.estk:
        n_novel= args.ested_k-args.num_labeled_classes
        print("Use estedk:", args.ested_k)
    else:
        n_novel= args.num_unlabeled_classes

    label_proto = prototypes.cpu().numpy()[:args.num_labeled_classes,:]
    preds_higher=[]

    preds_higher.append(all_preds.clone())
    print('Hierarchy clustering')
    mask_known=(all_preds<args.num_labeled_classes).cpu().numpy()
    l_feats = all_feats[mask_known]  # Get labelled set
    u_feats = all_feats[~mask_known]
    l_feats, u_feats= (torch.from_numpy(x).to(device) for  x in (l_feats, u_feats))

    while n_labeled>1:
        n_labeled=max(int(n_labeled/2),1)
        n_novel=max(int(n_novel/2),1)

        kmeans_l = KMeans(n_clusters=n_labeled, random_state=0).fit(label_proto)
        preds_labels = torch.from_numpy(kmeans_l.labels_).to(device)
        level_l_targets=preds_labels[all_preds[mask_known]]
        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil( n_samples / (n_labeled+n_novel))
        kmeans_higher =SemiSupKMeans(k=n_labeled+n_novel, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)
        kmeans_higher.fit_mix(u_feats, l_feats, level_l_targets)
        preds_level = kmeans_higher.labels_
        prototypes_level = kmeans_higher.cluster_centers_
        prototype_higher.append(prototypes_level.clone())
        preds_higher.append(preds_level.to(device).clone())

    return ids,all_preds, prototype_higher,preds_higher, metrics
def test_kmeans(model, test_loader, epoch,
                 save_name, args,projection_head=None, Use_GPU=True):

    model.eval()
    all_feats = []

    targets = np.array([])
    mask = np.array([])

    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.cuda()
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask.astype(bool)
    all_feats = np.concatenate(all_feats)
    # -----------------------
    # EVALUATE
    # -----------------------

    if Use_GPU:
        if args.estk:
            preds, prototypes = kmeans(X=torch.from_numpy(all_feats).to(device), num_clusters=args.ested_k,
                                        distance='euclidean', device=device)
            print("Use estedk:", args.ested_k)
        else:
            preds, prototypes = kmeans(X=torch.from_numpy(all_feats).to(device), num_clusters=args.num_unlabeled_classes+args.num_labeled_classes,
                                        distance='euclidean', device=device)

        preds, prototypes = preds.cpu().numpy(), prototypes.cpu().numpy()
    else:
        if args.estk:
            kmeanss = KMeans(n_clusters=args.ested_k, random_state=0).fit(
                all_feats)
            print("Use estedk:", args.ested_k)
        else:
            kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
                all_feats)
        preds = kmeanss.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)


    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--grad_from_block', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default='')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    parser.add_argument('--strategy', type=str, default='zero_one')
    parser.add_argument('--cluster_momentum', type=float, default=1)

    parser.add_argument('--unsupervised_smoothing', type=float, default=1)
    parser.add_argument('--distance', type=str, default='euclidean',
                        help='options: euclidean, cosine')

    parser.add_argument('--train_report_interval', default=200, type=int)
    parser.add_argument('--prototype_extraction_interval', default=1, type=int)

    parser.add_argument('--gpu_clustering', type=str2bool, default=True)
    parser.add_argument('--unbalanced', type=str2bool, default=False)

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--report', type=str2bool, default=True)


    parser.add_argument('--cub_root', type=str, default='')
    parser.add_argument('--osr_split_dir', type=str, default='')
    parser.add_argument('--dino_pretrain_path', type=str, default='')
    parser.add_argument('--cub_model_best', type=str, default='')
    parser.add_argument('--cars_root', type=str, default='')
    parser.add_argument('--pets_root', type=str, default='')
    parser.add_argument('--cifar_100_root', type=str, default='')
    parser.add_argument('--aircraft_root', type=str, default='')
    parser.add_argument('--imagenet_root', type=str, default='')
    parser.add_argument('--herbarium_dataroot', type=str, default='')

    parser.add_argument('--exp_name', default=None, type=str)

    parser.add_argument('--reference_epochs', default=100, type=int, help='Number of reference model training epochs')
    parser.add_argument('--reference_batch_size', default=128, type=int)
    parser.add_argument('--reference_lr', type=float, default=0.1, help='Learning rate for reference model training')
    parser.add_argument('--reference_momentum', type=float, default=0.9, help='Momentum for reference model training')
    parser.add_argument('--reference_weight_decay', type=float, default=1e-4, help='Weight decay for reference model training')
    parser.add_argument('--reference_print_freq', default=10, type=int, help='Print frequency for reference model training')
    parser.add_argument('--projection_head_nlayers', default=1, type=int)

    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--reference_model_path', type=str, default='')
    parser.add_argument('--train_reference_model', action='store_true')
    parser.add_argument('--save_reference_model', action='store_true', dest='save_reference_model', default=True)
    parser.add_argument('--no_save_reference_model', action='store_false', dest='save_reference_model')
    parser.add_argument('--save_best', action='store_true', default=False,
                        help='If set, save the best model during training.')

    parser.add_argument('--lr_backbone',   type=float, default=0.1)
    parser.add_argument('--aga_weight', type=float, default=1.0, help='')

    parser.add_argument('--aperture', type=float, default=4.0)
    parser.add_argument('--eep_weight', type=float, default=0.3)
    parser.add_argument('--estk', action='store_true', default=False)
    parser.add_argument('--ested_k', default=1, type=int)
    parser.add_argument('--hidden_dim', default=2048, type=int)


    parser.add_argument('--gdc_every', type=int, default=1,
                        help='Compute GDC every N batches.')
    parser.add_argument('--soc_every', type=int, default=1,
                        help='Compute SOC every N batches.')
    parser.add_argument('--max_step', type=int, default=200,
                        help='Stop after N steps and report diagnostic means.')
    parser.add_argument('--print_freq', default=1, type=int)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    init_seed_torch(args.seed)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875

        pretrain_path = args.dino_pretrain_path
        model = vits.__dict__['vit_base']()
        torch.cuda.empty_cache()
        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)
        # if dino_v1:
            # pretrain_path = dino_pretrain_path
            # model = vits.__dict__['vit_base']()
            # torch.cuda.empty_cache()
            # state_dict = torch.load(pretrain_path, map_location='cpu')
            # model.load_state_dict(state_dict)
            # state_dict = torch.load(pretrain_path, map_location='cpu')['teacher']
            # dict_keys=list(state_dict.keys())
            # for key in dict_keys:
            #     newkey= key.replace("backbone.",'')
            #     state_dict[newkey]=state_dict[key]
            #     del state_dict[key]
            # model.load_state_dict(state_dict)
        # else:
        #     pretrain_path = dino_pretrain_path2
        #     model = vits.__dict__['vit_base']()
        #     torch.cuda.empty_cache()
        #     state_dict = torch.load(pretrain_path, map_location='cpu')
        #     model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir+'model_best.pt', map_location='cpu'), strict=False)
        # model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        # for m in model.parameters():
        #     m.requires_grad = False

        # # Only finetune layers from block 'args.grad_from_block' onwards
        # max_block=0
        # for name, m in model.named_parameters():
        #     if 'block' in name:
        #         block_num = int(name.split('.')[1])
        #         if block_num>max_block:
        #             max_block=block_num

        #         if block_num >= args.grad_from_block:
        #             m.requires_grad = True

    else:
        raise NotImplementedError
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
    print(f"num_labeled_classes {args.num_labeled_classes}")
    print(f"num_labeled_classes {args.num_unlabeled_classes}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Unlabelled_train_examples_test size: {len(unlabelled_train_examples_test)}")
    print(f"Labelled dataset size: {len(train_dataset.labelled_dataset)}")
    print(f"Unlabelled dataset size: {len(train_dataset.unlabelled_dataset)}")
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / (unlabelled_len+label_len) for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    merge_train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)
    reference_loader = DataLoader(labelled_dataset, num_workers=args.num_workers, batch_size=args.reference_batch_size, shuffle=True)

    # ----------------------
    # REFERENCE MODEL
    # ----------------------
    # If --train_reference_model is explicitly set, keep it as-is.
    has_reference_checkpoint = bool(args.reference_model_path) and os.path.isfile(args.reference_model_path)
    if args.train_reference_model:
        if has_reference_checkpoint:
            print("`--train_reference_model` is set: ignore existing reference checkpoint and retrain.")
    else:
        args.train_reference_model = not has_reference_checkpoint

    if args.train_reference_model:
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
                    print(name)

        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)

        model = model.to(device)
        projector = projector.to(device)

        model = train_reference_model(model, projector, reference_loader, args)
        if args.save_reference_model:
            os.makedirs(args.save_path, exist_ok=True)
            reference_dir = os.path.join(args.save_path, "reference")
            os.makedirs(reference_dir, exist_ok=True)

            reference_backbone_path = os.path.join(
                reference_dir,
                f"SelEx_block{args.grad_from_block}_seed{args.seed}.pth"
            )
        
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, reference_backbone_path)
            print(f"Saved reference backbone -> {reference_backbone_path}")

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    reference_backbone = vits.__dict__['vit_base']()

    if not args.train_reference_model:
        state_dict = torch.load(args.reference_model_path, map_location='cpu')
        model.load_state_dict(state_dict)

        reference_backbone.load_state_dict(state_dict)
        for m in reference_backbone.parameters():
            m.requires_grad = False
        reference_backbone = reference_backbone.to(device)
    else:
        reference_backbone.load_state_dict(model.state_dict())
        for m in reference_backbone.parameters():
            m.requires_grad = False
        reference_backbone = reference_backbone.to(device)

    for m in model.parameters():
        m.requires_grad = False
        # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
                print(name)

    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    if args.warmup_model_dir is not None:
        print(f'Loading projection head weights from {args.warmup_model_dir}')
        projection_head.load_state_dict(torch.load(args.warmup_model_dir + 'model_proj_head_best.pt', map_location='cpu'), strict=False)

    model = model.to(device)
    projection_head.to(device)

    #------------------------------Conceptor------------------------------#
    labeled_eval_ds = copy.deepcopy(train_dataset.labelled_dataset)
    labeled_eval_ds.transform = test_transform
    labeled_eval_loader = DataLoader(
        labeled_eval_ds, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )

    gradient_coordinator = GradientCoordinator(model, args.aga_weight)

    known_subspace = compute_known_subspace(backbone=reference_backbone, 
                                        labeled_eval_loader=labeled_eval_loader, args=args,
                                        aperture=args.aperture)
    if known_subspace is None:
        known_subspace = torch.zeros(args.feat_dim, args.feat_dim, device=next(reference_backbone.parameters()).device)
    gradient_coordinator.set_conceptor(known_subspace)

    labeled_energy_ratio_ema = compute_labeled_energy_reference(reference_backbone, labeled_eval_loader, known_subspace)
    #------------------------------Conceptor------------------------------#


    # ----------------------
    # TRAIN
    # ----------------------
    # if not os.path.exists('Plots'):
    #     os.mkdir('Plots')
    train(
        projection_head,
        model,
        train_loader,
        test_loader_labelled,
        test_loader_unlabelled,
        merge_train_loader,
        labeled_eval_loader,
        args,
        reference_backbone,
        gradient_coordinator,
        labeled_energy_ratio_ema
    )
    torch.cuda.empty_cache()
    if args.report:
        print("Reports for the best checkpoint:")
        os.system("CUDA_VISIBLE_DEVICES="+str(args.gpu_id)+" python ../clustering/extract_features.py --dataset "+args.dataset_name+
                  " --warmup_model_dir "+ args.model_path.replace('(','\(').replace(')','\)').replace('|','\|'))
        os.system("CUDA_VISIBLE_DEVICES="+str(args.gpu_id)+" python ../clustering/k_means.py --dataset "+args.dataset_name+
                  " --unbalanced "+str(int(args.unbalanced)))
        print("Reports for the last checkpoint:")
        os.system("CUDA_VISIBLE_DEVICES="+str(args.gpu_id)+" python ../clustering/extract_features.py --dataset "+args.dataset_name+
                  " --warmup_model_dir "+ args.model_path.replace('(','\(').replace(')','\)').replace('|','\|')+
                  "  --use_best_model 0")
        os.system("CUDA_VISIBLE_DEVICES="+str(args.gpu_id)+" python ../clustering/k_means.py --dataset "+args.dataset_name+
                  " --unbalanced "+str(int(args.unbalanced)))

    torch.cuda.empty_cache()
    print(args.model_path)
