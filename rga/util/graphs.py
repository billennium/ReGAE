from typing import List

import numpy as np


def max_number_of_nodes_in_graphs(graphs: List[np.array]) -> int:
    max_number_of_nodes = 0
    for graph in graphs:
        if graph.shape[0] > max_number_of_nodes:
            max_number_of_nodes = graph.shape[0]
    return max_number_of_nodes
