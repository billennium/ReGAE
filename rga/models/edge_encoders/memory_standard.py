from typing import List
from argparse import ArgumentParser, ArgumentError

import torch
from torch import nn, Tensor

from rga.models.utils.getters import get_activation_function
from rga.models.utils.calc import weighted_average
from rga.models.utils.layers import (
    sequential_from_layer_sizes,
    parse_layer_sizes_list,
)


class MemoryEdgeEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        block_size: int,
        encoder_hidden_layer_sizes: List[int],
        encoder_activation_function: str,
        **kwargs,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        self.block_size = block_size

        input_size = 2 * embedding_size + edge_size * (block_size * block_size)

        activation_f = get_activation_function(encoder_activation_function)
        self.nn = sequential_from_layer_sizes(
            input_size,
            embedding_size * 3,
            encoder_hidden_layer_sizes,
            activation_f,
        )

    def forward(
        self, diagonal_x: Tensor, embedding_l: Tensor, embedding_r: Tensor
    ) -> Tensor:
        x = torch.cat(
            (embedding_l, embedding_r, diagonal_x.flatten(start_dim=2)), dim=-1
        )
        nn_output = self.nn(x)

        (embedding, mem_overwrite_ratio, embedding_ratio,) = torch.split(
            nn_output,
            [
                self.embedding_size,
                self.embedding_size,
                self.embedding_size,
            ],
            dim=-1,
        )

        weighted_prev_embedding = weighted_average(
            embedding_l, embedding_r, embedding_ratio
        )
        new_embedding = weighted_average(
            weighted_prev_embedding, embedding, mem_overwrite_ratio
        )
        return new_embedding

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--encoder_hidden_layer_sizes",
            dest="encoder_hidden_layer_sizes",
            default=[256],
            type=parse_layer_sizes_list,
            metavar="DECODER_H_SIZES",
            help="list of the sizes of the decoder's hidden layers",
        )
        parser.add_argument(
            "--encoder_activation_function",
            dest="encoder_activation_function",
            default="ReLU",
            type=str,
            metavar="ACTIVATION_F_NAME",
            help="name of the activation function of hidden layers",
        )
        return parent_parser
