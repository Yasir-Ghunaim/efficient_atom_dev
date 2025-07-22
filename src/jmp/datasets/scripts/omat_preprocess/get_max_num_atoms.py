import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data.data import BaseData

from jmp.datasets.pretrain_aselmdb import PretrainAseDbDataset
from jmp.datasets.pretrain_lmdb import PretrainDatasetConfig


def find_max_atomic_number(dataset, num_workers, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda data_list: [data.atomic_numbers for data in data_list],
        shuffle=False,
    )

    max_atomic_number = 0

    for batch in tqdm(loader, desc="Scanning atomic numbers"):
        for atomic_numbers in batch:
            max_atomic_number = max(max_atomic_number, atomic_numbers.max().item())

    return max_atomic_number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    dataset = PretrainAseDbDataset(
        PretrainDatasetConfig(
            src=args.src,
            args=argparse.Namespace(seed=0),
        )
    )

    max_Z = find_max_atomic_number(dataset, args.num_workers, args.batch_size)
    print(f"Maximum atomic number in dataset: {max_Z}")


if __name__ == "__main__":
    main()
