import numpy as np
import networkx as nx


def bfs_ordering(adjacency_matrix: np.array) -> np.array:
    graph = nx.from_numpy_matrix(adjacency_matrix)

    node_order_sorted_degree = np.array(sorted(
        graph.degree, key=lambda x: x[1], reverse=True
    ))[:, 0]

    graph = change_graph_order(graph, node_order_sorted_degree)

    node_order_bfs = []
    for i in range(len(graph)):
        if i not in node_order_bfs:
            node_order_bfs.extend(nx.bfs_tree(graph, i))        
    
    graph = change_graph_order(graph, node_order_bfs)
 
    return nx.to_numpy_array(graph)
    
def change_graph_order(graph: nx.Graph, node_order: list) -> np.array:
    adjacency_matrix = nx.to_numpy_array(graph)
    adjacency_matrix = adjacency_matrix[np.ix_(node_order, node_order)]
    return nx.from_numpy_matrix(adjacency_matrix)
