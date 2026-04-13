#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_VALUES=$(python -c "
from config1 import exp_root, osr_split_dir, dino_pretrain_path, cars_root
print(f'--exp_root={exp_root}')
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--cars_root={cars_root}')
" | tr '\n' ' ')

python -m util.Diagnose.SimGCD \
    --dataset_name scars \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform imagenet \
    --lr 0.1 \
    --eval_funcs v2 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name simgcd_scars_seed0 \
    --seed 0 \
    --gdc_every 1 \
    --soc_every 1 \
    $CONFIG_VALUES