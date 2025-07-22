"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings
import argparse
from pathlib import Path
from typing import assert_never

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from jmp.datasets.finetune.base import LmdbDataset as FinetuneLmdbDataset
from jmp.datasets.pretrain_lmdb import PretrainDatasetConfig, PretrainLmdbDataset
from jmp.datasets.pretrain_aselmdb import PretrainAseDbDataset


def _gather_energy_force_stats(dataset: Dataset[BaseData], num_workers: int, batch_size: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda data_list: [
            (data.y.item(), data.force.numpy()) for data in data_list
        ],
        shuffle=False,
    )

    energy_list = []
    force_values = []

    for batch in tqdm(loader, total=len(loader)):
        for energy, force in batch:
            energy_list.append(energy)
            force_values.append(force.reshape(-1, 3))

    energy_array = np.array(energy_list)
    force_array = np.concatenate(force_values, axis=0)

    energy_mean = np.mean(energy_array)
    energy_std = np.std(energy_array)

    force_mean = np.mean(force_array, axis=0)
    force_std = np.std(force_array, axis=0)

    return energy_mean, energy_std, force_mean, force_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Path to dataset directory or LMDB file")
    parser.add_argument("--dest", type=Path, required=False, help="Output .npz file path")
    parser.add_argument("--type", type=str, choices=["pretrain", "pretrain_omat", "finetune"], required=True, help="Dataset type")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    src: Path = args.src
    dest: Path = args.dest or src / "energy_force_stats.npz"
    dataset_type: str = args.type

    assert src.exists(), f"{src} does not exist"
    assert dest.suffix == ".npz", f"{dest} must be a .npz file"
    assert not dest.exists(), f"{dest} already exists"

    match dataset_type:
        case "pretrain":
            dataset = PretrainLmdbDataset(PretrainDatasetConfig(src=src))
        case "pretrain_omat":
            dataset = PretrainAseDbDataset(PretrainDatasetConfig(src=src, args=argparse.Namespace(number_of_samples=False, seed=0)))
        case "finetune":
            dataset = FinetuneLmdbDataset(src=src, args=argparse.Namespace(number_of_samples=False, seed=0))
        case _:
            assert_never(dataset_type)

    energy_mean, energy_std, force_mean, force_std = _gather_energy_force_stats(
        dataset, args.num_workers, args.batch_size
    )

    print("Energy mean:", energy_mean)
    print("Energy std:", energy_std)
    print("Force mean:", force_mean)
    print("Force std:", force_std)

    # np.savez(dest,
    #          energy_mean=energy_mean,
    #          energy_std=energy_std,
    #          force_mean=force_mean,
    #          force_std=force_std)


if __name__ == "__main__":
    main()
