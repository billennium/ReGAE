import torch


def adj_matrix_to_diagonal_representation(
    adj_matrix: torch.Tensor,
    num_nodes: int,
    max_num_nodes_padding: int = None,
    padding_value: float = -1.0,
) -> torch.Tensor:
    """
    The adjacency matrix has now a shape like [y, x, edge_size].
    For example, skipping the edge_size dimension:
     x 0 1 2 3
    y
    0  0 0 0 0
    1  1 0 0 0
    2  1 0 0 0
    3  0 1 1 0
    """
    diagonals = []
    for diagonal_offset in reversed(range(1, num_nodes)):
        diagonal = torch.diagonal(
            adj_matrix,
            offset=diagonal_offset - adj_matrix.shape[0],
        ).transpose(1, 0)
        diagonals.append(diagonal)
    # concat over the not-edge-shape dimension, that is, the second highest
    diagonal_dimension = len(diagonals[0].shape) - 2
    concatenated_diagonals = torch.cat(diagonals, dim=diagonal_dimension)

    if max_num_nodes_padding is not None and max_num_nodes_padding > num_nodes:
        diagonal_padding_nodes = max_num_nodes_padding - 1
        max_concatenated_diagonals_length = int(
            diagonal_padding_nodes * (1 + diagonal_padding_nodes) / 2
        )
        pad_length = max_concatenated_diagonals_length - concatenated_diagonals.shape[0]
        concatenated_diagonals = torch.nn.functional.pad(
            concatenated_diagonals, (0, 0, 0, pad_length), value=padding_value
        )

    return concatenated_diagonals
