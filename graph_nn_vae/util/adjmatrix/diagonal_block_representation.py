import torch


def calculate_num_blocks(num_nodes: int, block_size: int):
    return torch.ceil((num_nodes - 1) / block_size)


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

    adj_matrix = adj_matrix[1:num_nodes, : num_nodes - 1, :]

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


def diagonal_block_to_adj_matrix_representation(
    diagonal_block_graph: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    The diagonal block representatuin has a shape like [ block_idx  : edge_y_idx : edge_x_idx : edge      ]
                                                       [ num_blocks : block_size : block_size : edge_size ]

    For a block size of 4 a diagonal representation like:

    1 0  0 0  0 0
    0 1  1 0  1 0

    Translates to:
     x 0 1 2 3
    y
    0  0 0 0 0
    1  1 0 0 0
    2  1 0 0 0
    3  0 1 1 0

    to a shape [y, x, edge_size]
    """

    num_blocks = diagonal_block_graph.shape[0]
    block_size = diagonal_block_graph.shape[1]
    edge_size = diagonal_block_graph.shape[-1]
    num_columns = divide_integer_round_up(num_nodes - 1, block_size)

    num_unpadded_blocks = int((num_columns + 1) / 2 * num_columns)
    diagonal_block_graph = diagonal_block_graph[:num_unpadded_blocks]
    num_blocks = num_unpadded_blocks

    num_missing_blocks_for_full_matrix = num_columns * num_columns - num_blocks
    diagonal_block_graph = torch.nn.functional.pad(
        diagonal_block_graph, (0, 0, 0, 0, 0, 0, 0, num_missing_blocks_for_full_matrix)
    )

    # create indices mapping the block diagonal to columns of a matrix
    columns = [[] for _ in range(num_columns)]
    num_blocks_in_diagonal = 1
    diagonal_offset = 0
    max_idx = 0
    while diagonal_offset < num_blocks:
        for idx_in_diagonal in range(num_blocks_in_diagonal):
            max_idx = diagonal_offset + idx_in_diagonal
            columns[idx_in_diagonal].append(max_idx)
        diagonal_offset += num_blocks_in_diagonal
        num_blocks_in_diagonal += 1

    for i in range(num_columns):
        columns[i].extend([max_idx + 1 for _ in range(i)])

    bottom_up_column_indices = torch.tensor(columns)
    bottom_up_column_indices = torch.rot90(bottom_up_column_indices)
    flattened_indices = bottom_up_column_indices.flatten()

    # recreate a "blocky" adjacency matrix
    flattened_rows_graph = diagonal_block_graph[flattened_indices]
    block_adj_matrix = flattened_rows_graph.view(
        num_columns,
        num_columns,
        block_size,
        block_size,
        edge_size,
    )

    # concatenate the blocks into a flatter matrix
    block_rows = [row[0] for row in torch.split(block_adj_matrix, 1, dim=0)]
    columnar_block_adj_matrix = torch.cat(block_rows, dim=1)
    columns = [column[0] for column in torch.split(columnar_block_adj_matrix, 1, dim=0)]
    adj_matrix = torch.cat(columns, dim=1)

    # adjust the shape
    pad_diff = num_nodes - adj_matrix.shape[0]
    if pad_diff > 0:
        adj_matrix = torch.nn.functional.pad(
            adj_matrix, (0, 0, 0, pad_diff, pad_diff, 0)
        )
    elif pad_diff < 0:
        adj_matrix = adj_matrix[-pad_diff:, :pad_diff, :]

    return adj_matrix


def divide_integer_round_up(dividend, divisor) -> int:
    return int((dividend + divisor - 1) / divisor)
