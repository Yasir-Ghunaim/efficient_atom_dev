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
from tqdm import tqdm


def _compute_mean_std(args: argparse.Namespace):
    @cache
    def dataset():
        return PretrainLmdbDataset(
            PretrainDatasetConfig(src=args.src, lin_ref=args.linref_path)
        )

    def extract_data(idx):
        data = dataset()[idx]
        y = data.y
        na = data.natoms
        f = data.force.numpy().reshape(-1, 3)  # shape: (num_atoms, 3)
        return (y, na, f)

    pool = mp.Pool(args.num_workers)
    indices = range(len(dataset()))

    outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))

    energies = [y for y, na, f in outputs]
    num_atoms = [na for y, na, f in outputs]
    forces = [f for y, na, f in outputs]

    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    avg_num_atoms = np.mean(num_atoms)

    all_forces = np.concatenate(forces, axis=0)  # shape: (total_atoms, 3)
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
    @cache
    def dataset():
        return PretrainAseDbDataset(
            PretrainDatasetConfig(
                src=args.src, 
                args=argparse.Namespace(seed=0)
                )
            )
        # return PretrainLmdbDataset(PretrainDatasetConfig(src=args.src))

    def extract_data(idx):
        data = dataset()[idx]
        x = (
            scatter(
                torch.ones(data.atomic_numbers.shape[0]),
                data.atomic_numbers.long(),
                dim_size=95,
            )
            .long()
            .numpy()
        )
        y = data.y
        return (x, y)

    pool = mp.Pool(args.num_workers)
    indices = range(len(dataset()))

    outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))

    features = [x[0] for x in outputs]
    targets = [x[1] for x in outputs]

    X = np.vstack(features)
    y = targets

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
    linref_parser.add_argument("--num_workers", type=int, default=100)
    linref_parser.set_defaults(fn=_linref)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
