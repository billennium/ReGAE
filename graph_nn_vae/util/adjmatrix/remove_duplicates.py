import numpy as np


def remove_duplicates(graphs: list):
    hashes = [hash(el.astype(int).tostring()) for el in graphs]
    _, indices = np.unique(hashes, return_index=True)
    return [graphs[i] for i in indices]
