import torch
from torch.utils.data import Dataset
from jmp.fairchem.core.datasets.ase_datasets import AseDBDataset
from collections.abc import Callable, Mapping

# from jmp.datasets.utils import get_molecule_df
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from functools import cache

import numpy as np
from jmp.datasets.pretrain_lmdb import PretrainDatasetConfig

class PretrainAseDbDataset(Dataset[BaseData]):

    @property
    def atoms_metadata(self) -> np.ndarray:
        if (
            metadata := next(
                (
                    self.metadata[k]
                    for k in ["natoms", "num_nodes"]
                    if k in self.metadata
                ),
                None,
            )
        ) is None:
            raise ValueError(
                f"Could not find atoms metadata key in loaded metadata.\n"
                f"Available keys: {list(self.metadata.keys())}"
            )
        return metadata

    @property
    @cache
    def metadata(self) -> Mapping[str, np.ndarray]:
        metadata_path = getattr(self, "metadata_path", None)
        if not metadata_path or not metadata_path.is_file():
            metadata_path = self.config.metadata_path

        if metadata_path and metadata_path.is_file():
            return np.load(metadata_path, allow_pickle=True)

        raise ValueError(f"Could not find atoms metadata in {metadata_path=}.")

    def __init__(self, config: PretrainDatasetConfig):
        self.config = config
        self.path = str(config.src)
        self.seed = config.args.seed
        self.max_samples = config.max_samples
        self.is_train = config.is_train
        # breakpoint()

        # config_kwargs = {} 

        # self.dataset = AseDBDataset(config=dict(src=self.path, **config_kwargs))
        # Wrap AseDBDataset
        self.dataset = AseDBDataset(config=dict(
            src=self.path,
            a2g_args={"r_energy": True, "r_forces": True, "r_stress": False},  # adjust if needed
            transforms={},  # or specify if you have any
        ))

        self.total_len = len(self.dataset)
        self.shuffled_indices = list(range(min(self.total_len, self.max_samples)))

        # Optional: load molecule_df for filtering or grouping
        # self.molecule_df = None
        # if self.is_train and hasattr(config.args, 'extract_features') and config.args.extract_features:
        #     self.molecule_df = get_molecule_df(Path(self.path))

    def __len__(self):
        return len(self.shuffled_indices)

    def __getitem__(self, idx):
        true_idx = self.shuffled_indices[idx]
        data: Data = self.dataset[true_idx]  # Already a torch_geometric.data.Data

        # Add required metadata
        data.sid = int(true_idx)
        data.fid = data.fid if hasattr(data, "fid") else 0
        data.lmdb_idx = true_idx
            


        # Rename data.energy to data.y exists
        if getattr(data, 'y', None) is None and hasattr(data, 'energy'):
            data.y = data.energy
            # delattr(data, "energy")

        # Rename data.forces to data.force exists
        if getattr(data, 'force', None) is None and hasattr(data, 'forces'):
            data.force = data.forces
            # delattr(data, "forces")

        # if self.molecule_df is not None:
        #     row = self.molecule_df[(self.molecule_df['sid'] == data.sid) & (self.molecule_df['fid'] == data.fid)]
        #     if not row.empty:
        #         data.molecule_name = row.iloc[0]['Molecule']
        return data
