#!/bin/bash

python generate_stats.py \
  --src "/ibex/project/c2261/datasets/omat/train/rattled-300-subsampled" \
  --type pretrain_omat \
  --dest /ibex/project/c2261/datasets/omat/train/omat_stats.json \
  --num-workers 50 \
  --batch-size 500
