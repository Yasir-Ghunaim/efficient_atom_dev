"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
from functools import cache
from pathlib import Path

import multiprocess as mp
import numpy as np
import torch
from jmp.datasets.pretrain_lmdb import PretrainDatasetConfig, PretrainLmdbDataset
from jmp.datasets.pretrain_aselmdb import PretrainAseDbDataset
from torch_scatter import scatter
from torch.utils.data import DataLoader

from tqdm import tqdm


def _compute_mean_std(args: argparse.Namespace):
    dataset = PretrainAseDbDataset(
        PretrainDatasetConfig(
            src=args.src,
            args=argparse.Namespace(seed=0),
            lin_ref=args.linref_path
        )
    )

    def collate_fn(batch):
        energies, natoms, forces = [], [], []
        for data in batch:
            energies.append(data.y)
            natoms.append(data.natoms)
            forces.append(data.force.numpy().reshape(-1, 3))  # shape: (num_atoms, 3)
        return np.array(energies), np.array(natoms), np.concatenate(forces, axis=0)

    
    dataloader = DataLoader(
        dataset,
        batch_size=100,  # adjust based on memory
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    all_energies = []
    all_natoms = []
    all_forces = []

    for energies, natoms, forces in tqdm(dataloader, total=len(dataloader)):
        all_energies.append(energies)
        all_natoms.append(natoms)
        all_forces.append(forces)

    energies = np.concatenate(all_energies)
    num_atoms = np.concatenate(all_natoms)
    all_forces = np.concatenate(all_forces)

    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    avg_num_atoms = np.mean(num_atoms)

    force_mean = np.mean(all_forces, axis=0)
    force_std = np.std(all_forces, axis=0)
    force_std_scalar = np.mean(force_std)

    print(
        f"energy_mean: {energy_mean}, energy_std: {energy_std}, average number of atoms: {avg_num_atoms}"
    )
    print(f"force_mean: {force_mean.tolist()}")
    print(f"force_std: {force_std.tolist()} (scalar: {force_std_scalar:.6f})")

    stats = {
        "energy_mean": float(energy_mean),
        "energy_std": float(energy_std),
        "avg_num_atoms": float(avg_num_atoms),
        "force_mean": force_mean.tolist(),
        "force_std": force_std.tolist(),
        "force_std_scalar": float(force_std_scalar),
    }

    with open(args.out_path, "w") as f:
        json.dump(stats, f, indent=2)


def _linref(args: argparse.Namespace):
    dataset = PretrainAseDbDataset(
        PretrainDatasetConfig(
            src=args.src,
            args=argparse.Namespace(seed=0)
        )
    )
    
    def collate_fn(batch):
        # Custom collate since we just need atomic number counts and energies
        X_list, y_list = [], []
        for data in batch:
            x = scatter(
                torch.ones(data.atomic_numbers.shape[0]),
                data.atomic_numbers.long(),
                dim_size=95,
            ).long().numpy()
            y = data.y
            X_list.append(x)
            y_list.append(y)
        return np.stack(X_list), np.array(y_list)
    
    dataloader = DataLoader(
        dataset,
        batch_size=100,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    X_list = []
    y_list = []

    for X_batch, y_batch in tqdm(dataloader, total=len(dataloader)):
        X_list.append(X_batch)
        y_list.append(y_batch)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    np.savez_compressed(args.out_path, coeff=coeff)
    print(f"Saved linear reference coefficients to {args.out_path}")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand")

    compute_mean_std_parser = subparsers.add_parser("compute_mean_std")
    compute_mean_std_parser.add_argument("--src", type=Path, required=True)
    compute_mean_std_parser.add_argument("--out_path", type=Path, required=True)
    compute_mean_std_parser.add_argument("--linref_path", type=Path, required=True)
    compute_mean_std_parser.add_argument("--num_workers", type=int, default=32)
    compute_mean_std_parser.set_defaults(fn=_compute_mean_std)

    linref_parser = subparsers.add_parser("linref")
    linref_parser.add_argument("--src", type=Path, required=True)
    linref_parser.add_argument("--out_path", type=Path, required=True)
    linref_parser.add_argument("--num_workers", type=int, default=32)
    linref_parser.set_defaults(fn=_linref)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
