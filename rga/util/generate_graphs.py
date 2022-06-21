from typing import List, Tuple

import torch
from torch.functional import Tensor

from rga import util
from rga.util import adjmatrix


def convert_model_output_to_diag_block(
    model_output,
) -> List[Tuple[Tensor, Tensor, int]]:
    diag_block_graphs = []
    for (graphs, masks), _ in model_output:
        for i in range(len(graphs)):
            graph = graphs[i]
            mask = masks[i]

            graph_without_padding = (
                torch.sigmoid(remove_block_padding(graph)).round().int()
            )

            block_count = mask.shape[0]
            block_size = mask.shape[1]
            mask_without_padding = torch.sigmoid(remove_block_padding(mask))

            num_nodes_upper_limit = (
                adjmatrix.block_count_to_num_block_diagonals(block_count) * block_size
            )
            if mask_without_padding.shape[0] == 34:
                mask_without_padding = mask_without_padding
                pass
            adj_matrix_mask = adjmatrix.diagonal_block_to_adj_matrix_representation(
                mask_without_padding, num_nodes_upper_limit
            )
            num_nodes = get_num_nodes(adj_matrix_mask)

            diag_block_graphs.append(
                (graph_without_padding, mask_without_padding, num_nodes)
            )
    return diag_block_graphs


def get_num_nodes(mask):
    for i in range(1, mask.shape[0]):
        if torch.diagonal(mask, -(mask.shape[0] - i)).mean() < 0.5:
            break
    return i


def remove_block_padding(graph):
    mask = graph.flatten(start_dim=1).isinf().all(dim=1)
    return graph[~mask]


def diag_block_graphs_to_tril_adj_matrices(
    data: List[Tuple[Tensor, Tensor, int]]
) -> List[Tensor]:
    adj_matrices = []
    for (graph, _, num_nodes) in data:
        graph = util.to_dense_if_not(graph)
        graph = graph.clamp(min=0)
        adj_matrix = adjmatrix.diagonal_block_to_adj_matrix_representation(
            graph, num_nodes
        )
        adj_matrix = adj_matrix[:, :, 0]
        adj_matrix = torch.tril(adj_matrix, -1)
        adj_matrix = adj_matrix.int()

        adj_matrices.append(adj_matrix)
    return adj_matrices
