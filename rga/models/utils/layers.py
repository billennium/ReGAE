from typing import Callable, List

from torch import nn


def sequential_from_layer_sizes(
    input_size: int,
    output_size: int,
    hidden_sizes: List[int],
    activation_function: nn.Module = nn.ReLU,
    output_function: nn.Module = None,
    dropout: float = 0,
) -> nn.Sequential:
    layer_sizes = hidden_sizes + [output_size]
    layers = [nn.Linear(input_size, layer_sizes[0])]
    for i in range(len(layer_sizes)):
        if i == len(layer_sizes) - 1:
            break
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(activation_function())
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    if output_function is not None:
        layers.append(output_function())

    return nn.Sequential(*layers)


def parse_layer_sizes_list(s: str) -> List[int]:
    if s is None or s == "":
        return []
    if isinstance(s, str):
        if "," in s:
            return [int(v) for v in s.split(",")]
        if "|" in s:
            return [int(v) for v in s.split("|")]
        if ":" in s:
            return [int(v) for v in s.split(":")]
    return [int(s)]
