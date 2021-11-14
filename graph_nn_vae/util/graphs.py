from typing import List

import networkx as nx


def max_number_of_nodes_in_graphs(graphs: List[nx.Graph]) -> int:
    max_number_of_nodes = 0
    for graph in graphs:
        if graph.number_of_nodes() > max_number_of_nodes:
            max_number_of_nodes = graph.number_of_nodes()
    return max_number_of_nodes
