#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

SAVE_ROOT="${SAVE_ROOT:-$ROOT_DIR/outputs/diagnose/eagc_selex_cub}"

CONFIG_VALUES=$(python -c "
from config1 import exp_root, osr_split_dir, dino_pretrain_path, cub_root
print(f'--exp_root={exp_root}')
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--cub_root={cub_root}')
" | tr '\n' ' ')

python -m util.Diagnose.SelEx_EAGC \
    --dataset_name cub \
    --batch_size 128 \
    --epochs 200 \
    --base_model vit_dino \
    --num_workers 4 \
    --use_ssb_splits True \
    --sup_con_weight 0.35 \
    --weight_decay 5e-5 \
    --contrast_unlabel_only False \
    --transform imagenet \
    --seed 0 \
    --eval_funcs v2 \
    --unsupervised_smoothing 1.0 \
    --grad_from_block 10 \
    --exp_name SelEx_eagc_cub_seed0 \
    --lr_backbone 0.1 \
    --lr 0.1 \
    --projection_head_nlayers 2 \
    --reference_epochs 30 \
    --reference_batch_size 32 \
    --reference_lr 0.02 \
    --save_path "$SAVE_ROOT" \
    --reference_model_path "$SAVE_ROOT/pretrain/SelEx_block10_seed0.pth" \
    $CONFIG_VALUES \
    --aga_weight 0.7 \
    --aperture 2.0 \
    --eep_weight 0.5

