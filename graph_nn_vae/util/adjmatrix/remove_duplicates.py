import torch
import numpy as np
from tqdm import tqdm


def remove_duplicates(graphs: list, labels: list = None):
    adjency_matrixes = [el[0] for el in graphs]
    hashes = [
        hash(el.int().numpy().tostring()) for el in tqdm(adjency_matrixes, "hashes")
    ]
    _, indices = np.unique(hashes, return_index=True)

    if labels is None:
        return [graphs[i] for i in indices], None
    else:
        return [graphs[i] for i in indices], [labels[i] for i in indices]
