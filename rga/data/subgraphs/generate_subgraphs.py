import torch


def generate_subgraphs(
    graph,
    mask,
    num_nodes: int,
    num_blocks: int,
    block_size: int,
    new_size: int,
    stride: int = 1,
    probability: float = 1.0,
):
    if new_size > num_blocks:
        return [graph], [torch.ones(graph.shape)], [num_nodes]

    candidates = torch.arange(0, num_blocks - new_size + 1, stride).int()
    if (num_blocks - new_size) not in candidates:
        candidates = torch.cat([candidates, torch.IntTensor([num_blocks - new_size])])
    if probability < 1:
        candidates = candidates[torch.rand(len(candidates)) < probability]

    if len(candidates) == 0:
        return ([], [], [])

    graph_diagonals = []
    mask_diagonals = []
    index = 0
    for diag_len in range(1, num_blocks + 1):
        if diag_len > num_blocks - new_size:
            graph_diagonals.append(graph[index : index + diag_len])
            mask_diagonals.append(mask[index : index + diag_len])
        index = index + diag_len

    subgraphs = []
    subgraphs_masks = []
    graph_num_nodes = []

    for k in candidates:
        reduced_graph = torch.cat(
            [graph_diagonals[i][k : k + i + 1] for i in range(new_size)]
        )
        reduced_mask = torch.cat(
            [mask_diagonals[i][k : k + i + 1] for i in range(new_size)]
        )
        subgraphs.append(reduced_graph)
        subgraphs_masks.append(reduced_mask)
        reduced_graph_size = block_size * (new_size - 1) + sum(reduced_mask[-1][-1]) + 1
        graph_num_nodes.append(reduced_graph_size.int())

    return (subgraphs, subgraphs_masks, graph_num_nodes)
