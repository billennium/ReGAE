from typing import List
from argparse import ArgumentParser, ArgumentError

import torch
from torch import nn, Tensor

from graph_nn_vae.models.utils.getters import get_activation_function
from graph_nn_vae.models.utils.calc import weighted_average
from graph_nn_vae.models.utils.layers import (
    sequential_from_layer_sizes,
    parse_layer_sizes_list,
)


class ConvolutionalEdgeEncoder(nn.Module):
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

        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.conv = nn.Sequential(
            nn.Conv2d(edge_size, 3, 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
        )

        conv_out_shape = self.conv(torch.empty(1, 1, block_size, block_size)).size(-1)
        print(f"{conv_out_shape = }")
        linear_input_size = 2 * embedding_size + conv_out_shape

        activation_f = get_activation_function(encoder_activation_function)
        self.linear = sequential_from_layer_sizes(
            linear_input_size,
            embedding_size * 3,
            encoder_hidden_layer_sizes,
            activation_f,
        )

    def forward(
        self, diagonal_x: Tensor, embedding_l: Tensor, embedding_r: Tensor
    ) -> Tensor:
        was_reshaped = False
        if diagonal_x.ndim == 5:
            was_reshaped = True
            batch_size = diagonal_x.shape[0]
            blocks_in_batch = diagonal_x.shape[1]
            diagonal_x = diagonal_x.reshape(
                batch_size * blocks_in_batch, *diagonal_x.shape[2:]
            )

        diagonal_x = torch.movedim(diagonal_x, -1, 1)
        diagonal_features = self.conv(diagonal_x)

        if was_reshaped:
            diagonal_features = diagonal_features.view(
                batch_size, blocks_in_batch, *diagonal_features.shape[1:]
            )

        x = torch.cat((embedding_l, embedding_r, diagonal_features), dim=-1)
        nn_output = self.linear(x)

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


def conv_shape(x, k=1, p=0, s=1, d=1):
    return int((x + 2 * p - d * (k - 1) - 1) / s + 1)
