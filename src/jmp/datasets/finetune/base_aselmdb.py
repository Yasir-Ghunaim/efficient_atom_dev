import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from jmp.fairchem.core.datasets.ase_datasets import AseDBDataset
from contextlib import ContextDecorator
from functools import cached_property

import numpy as np
from functools import cache
from pathlib import Path
from typing import Mapping, Callable
from argparse import Namespace
from typing_extensions import TypeVar, override

from jmp.datasets.sampling_utils import apply_random_sampling

T = TypeVar("T", infer_variance=True)

class FinetuneAseLMDBDataset(Dataset[T], ContextDecorator):
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.metadata["natoms"][indices]

    @cached_property
    def metadata(self) -> Mapping[str, np.ndarray]:
        if self.metadata_path and self.metadata_path.is_file():
            return np.load(self.metadata_path, allow_pickle=True)
        raise ValueError(f"Could not find atoms metadata in {self.metadata_path=}")

    def data_transform(self, data: Data) -> Data:
        return data

    def __init__(
        self,
        src: str | Path,
        metadata_path: str | Path | None = None,
        args: Namespace = Namespace(),
        is_train: bool = False,
        max_samples: int | None = None,
        lin_ref_path: Path | None = None
    ):
        super().__init__()
        self.path = Path(src)
        self.metadata_path = Path(metadata_path) if metadata_path else self.path / "metadata.npz"
        self.args = args
        self.is_train = is_train
        self.max_samples = max_samples
        self.lin_ref_path = lin_ref_path

        self.dataset = AseDBDataset(config=dict(
            src=str(self.path),
            a2g_args={"r_energy": True, "r_forces": True, "r_stress": False},
            transforms={},
        ))

        self.total_len = len(self.dataset)

        if self.max_samples is not None and self.max_samples > self.total_len:
            self.max_samples = self.total_len

        # if self.extract_features:
        #     self.shuffled_indices = apply_random_sampling(self.total_len, self.max_samples, args.seed)
        # else:
        if self.max_samples is not None:
            self.shuffled_indices = apply_random_sampling(self.total_len, self.max_samples, args.seed)
        else: 
            self.shuffled_indices = list(range(self.total_len))

        self.lin_ref = None
        if self.lin_ref_path is not None:
            coeff = np.load(self.lin_ref_path, allow_pickle=True)["coeff"]
            try:
                self.lin_ref = torch.nn.Parameter(
                    torch.tensor(coeff), requires_grad=False
                )
            except BaseException:
                self.lin_ref = torch.nn.Parameter(
                    torch.tensor(coeff[0]), requires_grad=False
                )


    def __len__(self) -> int:
        return len(self.shuffled_indices)

    def __getitem__(self, idx: int):
        true_idx = self.shuffled_indices[idx]
        data: Data = self.dataset[true_idx]

        # Standardize naming
        data.sid = int(true_idx)
        data.fid = getattr(data, "fid", 0)
        data.lmdb_idx = true_idx

        if getattr(data, "y", None) is None and hasattr(data, "energy"):
            data.y = data.energy
        if getattr(data, "force", None) is None and hasattr(data, "forces"):
            data.force = data.forces


        if self.lin_ref is not None:
            lin_energy = sum(self.lin_ref[data.atomic_numbers.long()])
            data.y -= lin_energy

        # if self.molecule_df is not None:
        #     row = self.molecule_df[(self.molecule_df["sid"] == data.sid) & (self.molecule_df["fid"] == data.fid)]
        #     if not row.empty:
        #         data.molecule_name = row.iloc[0]["Molecule"]

        return data

    # @override
    # def __getitem__(self, idx: int):
    #     # Figure out which db this should be indexed from.
    #     db_idx = bisect.bisect(self.keylen_cumulative, idx)
    #     # Extract index of element within that db.
    #     el_idx = idx
    #     if db_idx != 0:
    #         el_idx = idx - self.keylen_cumulative[db_idx - 1]
    #     assert el_idx >= 0, f"{el_idx=} is not a valid index"

    #     # Return features.
    #     key = f"{self.keys[db_idx][el_idx]}".encode("ascii")
    #     env = self.envs[db_idx]
    #     data_object_pickled = env.begin().get(key, default=None)
    #     if data_object_pickled is None:
    #         raise KeyError(
    #             f"Key {key=} not found in {env=}. {el_idx=} {db_idx=} {idx=}"
    #         )

    #     data_object = _pyg2_data_transform(pickle.loads(cast(Any, data_object_pickled)))
    #     data_object.id = f"{db_idx}_{el_idx}"
    #     return data_object

    @classmethod
    def pre_data_transform(cls, data: Data) -> Data:
        if not hasattr(data, "tags"):
            data.tags = torch.full_like(data.atomic_numbers, 2)
        if not hasattr(data, "natoms"):
            data.natoms = data.num_nodes
    