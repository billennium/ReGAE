from typing import List

from torch import nn


def sequential_from_layer_sizes(
    input_size: int, output_size: int, hidden_sizes: List[int]
) -> nn.Sequential:
    layer_sizes = hidden_sizes + [output_size]
    layers = [nn.Linear(input_size, layer_sizes[0])]
    for i in range(len(layer_sizes)):
        if i == len(layer_sizes) - 1:
            break
        layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    return nn.Sequential(*layers)
