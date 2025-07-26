"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import OMATConfig
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import (
    EarlyStoppingConfig,
    PrimaryMetricConfig,
    RLPConfig,
    WarmupCosRLPConfig,
)

STATS: dict[str, dict[str, NC]] = {
    "y": NC(mean=0.0, std=30.74026854956568),
    "force": NC(mean=0.0, std=4.445453643798828),
}


def jmp_l_omat_config_(
    config: OMATConfig, base_path: Path, args: argparse.Namespace = None
):
    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=5.0e-6,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.1,
    # )

    config.cutoff = 12.0
    config.max_neighbors = 30
    # config.backbone.max_radius = config.cutoff
    # config.backbone.max_neighbors = config.max_neighbors

    # Set data config
    if args.large:
        config.batch_size = 18
    else:
        config.batch_size = 18

    # Set up dataset
    config.train_dataset = DC.omat_config(base_path, "train", args=args, max_samples=2_000_000)
    config.val_dataset = DC.omat_config(base_path, "val", args=args, max_samples=2000)#, max_samples=2_500)
    config.test_dataset = DC.omat_config(base_path, "val", args=args)

    # OMAT specific settings
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

    config.trainer.val_check_interval = 0.05 #0.25

    # Gradient forces
    config.model_type = "energy_forces"
    # config.gradient_forces = True
    config.gradient_forces = False
    config.backbone.regress_forces = True
    config.backbone.direct_forces = True

    config.trainer.inference_mode = False
    config.trainer.precision = "16-mixed"

    # Set up normalization
    config.normalization = {
        "y": STATS["y"],
        "force": STATS["force"],
    }

    # We use more conservative early stopping for rMD17
    #   (we essentially copy Allegro here).
    config.trainer.max_epochs = args.epochs
    config.trainer.max_time = "07:00:00:00"
    # config.early_stopping = EarlyStoppingConfig(
    #     patience=1000,
    #     min_delta=1.0e-8,
    #     min_lr=1.0e-10,
    # )