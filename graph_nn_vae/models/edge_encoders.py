from typing import List
from argparse import ArgumentParser

import torch
from torch import nn, Tensor

from graph_nn_vae.models.utils.getters import get_activation_function
from graph_nn_vae.models.utils.calc import weighted_average
from graph_nn_vae.models.utils.layers import (
    sequential_from_layer_sizes,
    parse_layer_sizes_list,
)


class MemoryEdgeEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        encoder_hidden_layer_sizes: List[int],
        encoder_activation_function: str,
        **kwargs,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        input_size = 2 * embedding_size + edge_size

        activation_f = get_activation_function(encoder_activation_function)
        self.nn = sequential_from_layer_sizes(
            input_size,
            embedding_size * 3,
            encoder_hidden_layer_sizes,
            activation_f,
        )

    def forward(
        self, embedding_l: Tensor, embedding_r: Tensor, diagonal_x: Tensor
    ) -> Tensor:
        x = torch.cat((embedding_l, embedding_r, diagonal_x), dim=-1)
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

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
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
        return parser


class GRULikeEdgeEncoder(nn.Module):
    def __init__(self, embedding_size: int, edge_size: int, **kwargs):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        self.input_size = 2 * embedding_size + edge_size
        super().__init__()
        self.r = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ELU(),
            nn.Linear(512, embedding_size * 2),
            nn.Sigmoid(),
        )
        self.z = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ELU(),
            nn.Linear(512, embedding_size * 2),
            nn.Sigmoid(),
        )
        self.h = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ELU(), nn.Linear(512, embedding_size)
        )

    def forward(
        self, embedding_l: Tensor, embedding_r: Tensor, diagonal_x: Tensor
    ) -> Tensor:
        x = torch.cat((embedding_l, embedding_r, diagonal_x), dim=-1)
        embeddings = torch.cat((embedding_l, embedding_r), dim=-1)

        z = self.z(x)
        embedding_ratio, mem_overwrite_ratio = torch.split(
            z, (self.embedding_size, self.embedding_size), dim=-1
        )
        weighted_prev_emb = weighted_average(embedding_l, embedding_r, embedding_ratio)

        r = self.r(x)
        altered_embeddings = r * embeddings

        altered_prev_embs_with_diag_input = torch.cat(
            (altered_embeddings, diagonal_x), dim=-1
        )
        h = self.h(altered_prev_embs_with_diag_input)

        y = weighted_average(weighted_prev_emb, h, mem_overwrite_ratio)
        return y
