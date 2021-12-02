import torch


def adj_matrix_to_diagonal_block_representation(
    adj_matrix: torch.Tensor, num_nodes: int, block_size: int
) -> torch.Tensor:
    """
    The adjacency matrix has a shape like [y, x, edge_size].
    For example, skipping the edge_size dimension:
     x 0 1 2 3
    y
    0  0 0 0 0
    1  1 0 0 0
    2  1 0 0 0
    3  0 1 1 0

    The diagonal block 2x2 representation will split the graph into blocks:

    0 0   0 0
    1 0   0 0

    1 0   0 0
    0 1   1 0

    And then stack them as-is, in a two dimensional way, in a new "block" dimension,
    in the "diagonal" order, omitting the blocks that lay entirely in the upper rigth triangle:

    0 0
    0 0 - omitted

    --> 1 0  0 0  0 0
        0 1  1 0  1 0

    The dimensions of the resulting Tensor: [ block_idx  : edge_y_idx : edge_x_idx : edge      ]
    The size of the dimensions:             [ num_blocks : block_size : block_size : edge_size ]
    """

    adj_matrix = adj_matrix[:num_nodes, :num_nodes, :]

    pad_diff = adj_matrix.shape[0] % block_size
    if pad_diff != 0:
        pad_diff = block_size - pad_diff
        adj_matrix = torch.nn.functional.pad(
            adj_matrix, (0, 0, 0, pad_diff, pad_diff, 0)
        )

    # if block_size == 1:
    #     block_adj_matrix = adj_matrix[:, :, None, None, :]
    # else:
    blocks_per_dim = int(adj_matrix.shape[0] / block_size)
    block_rows = []
    for block_y in range(blocks_per_dim):
        block_y_start = block_y * block_size
        block_y_end = (block_y + 1) * block_size
        block_row = []
        for block_x in range(blocks_per_dim):
            block_x_start = block_x * block_size
            block_x_end = (block_x + 1) * block_size
            block = adj_matrix[block_y_start:block_y_end, block_x_start:block_x_end, :]
            block_row.append(block)
        block_row = torch.stack(block_row)
        block_rows.append(block_row)
    block_adj_matrix = torch.stack(block_rows)

    # now the dimensions are [block_y : block_x : edge_in_block_y : edge_in_block_x : edge]

    diagonals = []
    for diagonal_offset in range(block_adj_matrix.shape[0]):
        diagonal = torch.diagonal(
            block_adj_matrix,
            offset=diagonal_offset - (block_adj_matrix.shape[0] - 1),
        ).moveaxis(-1, 0)
        diagonals.append(diagonal)
    concatenated_diagonals = torch.cat(diagonals, dim=0)

    return concatenated_diagonals
