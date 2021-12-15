import torch


def remove_duplicates(graphs: list, labels: list = None):
    adjency_matrixes = [el[0] for el in graphs]

    _, indices = unique(
        torch.nn.utils.rnn.pad_sequence(
            adjency_matrixes, batch_first=True, padding_value=-1.0
        ),
        dim=0,
    )
    if labels is None:
        return [graphs[i] for i in indices], None
    else:
        return [graphs[i] for i in indices], [labels[i] for i in indices]


def unique(x, dim: int = None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)
