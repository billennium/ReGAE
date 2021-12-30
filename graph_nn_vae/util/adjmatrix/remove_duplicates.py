import numpy as np


def remove_duplicates(graphs: list, labels: list = None):
    hashes = [hash(el.astype(int).tostring()) for el in graphs]
    _, indices = np.unique(hashes, return_index=True)

    if labels is None:
        return [graphs[i] for i in indices], None
    else:
        return [graphs[i] for i in indices], [labels[i] for i in indices]
