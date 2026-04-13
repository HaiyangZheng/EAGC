#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_VALUES=$(python -c "
from config1 import exp_root, osr_split_dir, dino_pretrain_path, aircraft_root
print(f'--exp_root={exp_root}')
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--aircraft_root={aircraft_root}')
" | tr '\n' ' ')

python -m util.Diagnose.SelEx \
    --dataset_name aircraft \
    --batch_size 128 \
    --grad_from_block 10 \
    --epochs 200 \
    --base_model vit_dino \
    --num_workers 4 \
    --use_ssb_splits True \
    --sup_con_weight 0.35 \
    --weight_decay 5e-5 \
    --contrast_unlabel_only False \
    --transform imagenet \
    --lr 0.1 \
    --eval_funcs v2 \
    --unsupervised_smoothing 0.5 \
    --exp_name SelEx_aircraft_seed0 \
    --seed 0 \
    $CONFIG_VALUES