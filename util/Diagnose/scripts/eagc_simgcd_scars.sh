#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

SAVE_ROOT="${SAVE_ROOT:-$ROOT_DIR/outputs/diagnose/eagc_simgcd_scars}"

CONFIG_VALUES=$(python -c "
from config1 import exp_root, osr_split_dir, dino_pretrain_path, cars_root
print(f'--exp_root={exp_root}')
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--cars_root={cars_root}')
" | tr '\n' ' ')

python -m util.Diagnose.SimGCD_EAGC \
    --dataset_name 'scars' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --seed 0 \
    --exp_name "simgcd_eagc_scars_seed0" \
    --lr_backbone 0.1 \
    --lr 0.1 \
    --reference_epochs 30 \
    --reference_batch_size 32 \
    --reference_lr 0.02 \
    --projection_head_nlayers 1 \
    --save_path "$SAVE_ROOT" \
    --reference_model_path "$SAVE_ROOT/pretrain/SimGCD_block11_seed0.pth" \
    $CONFIG_VALUES \
    --aga_weight 0.7 \
    --aperture 2.0 \
    --eep_weight 0.5 \
    --print_freq 1
