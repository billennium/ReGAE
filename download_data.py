from torch_geometric.datasets import KarateClub, TUDataset
from tqdm.auto import tqdm
import os
import shutil

allowed_datasets = [
    "MUTAG", "PROTEINS", "REDDIT-BINARY", "NCI109", "ENZYMES", "NCI109",
    "FRANKENSTEIN", "AIDS", "MOLT-4",
    "MSRC_9", "IMDB-BINARY", "IMDB-MULTI",# "COLLAB"
]


for dataset_name in tqdm(allowed_datasets):
    TUDataset(root="datasets", name=dataset_name)
    raw_dir = f"datasets/{dataset_name}/raw"
    processed_dir = f"datasets/{dataset_name}/processed"
    target_dir = f"datasets/{dataset_name}"

    for filename in os.listdir(raw_dir):
        os.rename(os.path.join(raw_dir, filename), os.path.join(target_dir, filename))

    shutil.rmtree(processed_dir)
    os.rmdir(raw_dir)