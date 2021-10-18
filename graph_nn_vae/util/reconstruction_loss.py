from typing import Callable

import torch
from torch import Tensor


def get_reconstruction_loss(
    adjacency_matrices_batch: Tensor,
    reconstructed_graph_diagonals: Tensor,
    loss_function: Callable,
) -> Tensor:
    input_concatenated_diagonals = []
    for adjacency_matrix in adjacency_matrices_batch:
        diagonals = [
            torch.diagonal(adjacency_matrix, offset=-i).transpose(1, 0)
            for i in reversed(range(adjacency_matrix.shape[0]))
        ]
        for i in reversed(range(len(diagonals))):
            if torch.count_nonzero(diagonals[i]) > 1:
                diagonals[i + 1] = diagonals[i + 1].fill_(-1.0)
                break
        concatenated_diagonals = torch.cat(diagonals, dim=0)
        input_concatenated_diagonals.append(concatenated_diagonals)
    input_batch_reshaped = torch.stack(input_concatenated_diagonals)

    reconstructed_diagonals_length = reconstructed_graph_diagonals.shape[1]
    input_pad_length = reconstructed_diagonals_length - input_batch_reshaped.shape[1]
    input_batch_reshaped = torch.nn.functional.pad(
        input_batch_reshaped, (0, 0, 0, input_pad_length)
    )

    return loss_function(reconstructed_graph_diagonals, input_batch_reshaped)
