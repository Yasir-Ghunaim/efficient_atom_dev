import lmdb
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import cast
from multiprocessing import Pool, cpu_count
from jmp.fairchem.core.datasets.ase_datasets import AseDBDataset

import torch_geometric
from torch_geometric.data.data import BaseData, Data

from collections import Counter
from ase.data import chemical_symbols

def generate_molecule_name(record):
    atomic_numbers = record.atomic_numbers.tolist()
    symbol_counts = Counter(chemical_symbols[z] for z in atomic_numbers)
    # Format like: B2Li2
    name = ''.join(f"{elem}{count if count > 1 else ''}" for elem, count in sorted(symbol_counts.items()))
    return name

# Function to generate molecule ID
def generate_molecule_id(metadata):
    bulk_symbols = metadata.get("bulk_symbols", "Unknown")
    # ads_symbols = metadata.get("ads_symbols", "None")
    return bulk_symbols

# Function to transform PyG data
def pyg2_data_transform(data: BaseData):
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        data = Data(**{k: v for k, v in data.__dict__.items() if v is not None})
        data = cast(BaseData, data)
    return data

# Function to process a single LMDB file
def process_aselmdb(aselmdb_file, output_file, position, start_sid):
    dataset = AseDBDataset(config={"src": str(aselmdb_file)})
    data = []

    # Process each entry in the LMDB with its own progress bar
    for idx in tqdm(range(len(dataset)), desc=f"Processing {aselmdb_file.name}", unit="record", position=position, leave=False):
        try:
            record = dataset[idx]
            sid = start_sid + idx
            fid = 0
            molecule_symbol = generate_molecule_name(record)
            data.append([molecule_symbol, fid, sid])
        except Exception as e:
            print(f"Error processing key {key.decode('utf-8')} in {lmdb_file.name}: {e}")

    # Save the processed data to a text file
    df = pd.DataFrame(data, columns=["Molecule", "fid", "sid"])
    df.to_csv(output_file, index=False, sep=",", header=True)
    print(f"Saved: {output_file}")

    return len(dataset)

# Function to process all LMDB files
def process_all_aselmdbs(lmdb_dir, output_dir):
    aselmdb_files = sorted(
        [f for f in Path(lmdb_dir).glob("*.aselmdb") if not f.name.endswith("-lock")],
        key=lambda x: int(x.stem.split("_")[-1])  # Sort by numeric part of filename
    )
    # output_files = [Path(output_dir) / ("data_log." + f.name.replace(".aselmdb", ".txt")) for f in aselmdb_files]
    output_files = [Path(output_dir) / ("data_log." + f.name.replace("db_", "").replace(".aselmdb", ".txt")) for f in aselmdb_files]

    
    # Use multiprocessing Pool with tqdm for each worker
    # with Pool(num_workers) as pool:
    #     pool.map(
    #         process_lmdb,
    #         [(lmdb_file, output_file, i) for i, (lmdb_file, output_file) in enumerate(zip(aselmdb_files, output_files))]
    #     )

    start_sid = 0
    for i, (aselmdb_file, output_file) in enumerate(zip(aselmdb_files, output_files)):
        n = process_aselmdb(aselmdb_file, output_file, i, start_sid)
        start_sid += n  # accumulate sid


if __name__ == "__main__":
    # TODO: Update this path
    # metadata_file = "path/to/oc22_metadata.pkl"

    # Path to the OMAT lmdb training files
    # TODO: Update this path
    lmdb_dir = "/ibex/project/c2261/datasets/omat/rattled-300-subsampled/"

    # Load metadata
    # with open(metadata_file, "rb") as f:
    #     dataset_metadata = pickle.load(f)

    # Run the processing
    output_dir = lmdb_dir
    process_all_aselmdbs(lmdb_dir, output_dir)#, num_workers=100)