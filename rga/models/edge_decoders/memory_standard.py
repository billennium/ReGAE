from typing import List, Tuple
from argparse import ArgumentError, ArgumentParser

import torch
from torch import nn, Tensor

from rga.models.utils.getters import get_activation_function
from rga.models.utils.calc import weighted_average
from rga.models.utils.layers import (
    sequential_from_layer_sizes,
    parse_layer_sizes_list,
)


class MemoryEdgeDecoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        block_size: int,
        decoder_hidden_layer_sizes: List[int],
        decoder_activation_function: str,
        **kwargs,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        self.block_size = block_size

        nn_input_size = embedding_size * 2
        graph_end_mask_size = 1
        self.edge_with_mask_block_size = block_size ** 2 * (
            graph_end_mask_size + edge_size
        )
        nn_output_size = embedding_size * 4 + self.edge_with_mask_block_size

        activation_f = get_activation_function(decoder_activation_function)
        self.nn = sequential_from_layer_sizes(
            nn_input_size,
            nn_output_size,
            decoder_hidden_layer_sizes,
            activation_f,
        )

        # self.kernel_count = 50
        # self.kernel_size = 1
        # self.rate = 0.00

        # self.kernels = nn.ModuleList()
        # for _ in range(self.kernel_count):
        #     self.kernels.append(
        #         sequential_from_layer_sizes(
        #             nn_input_size,
        #             self.kernel_size,
        #             [int((nn_input_size + self.kernel_size) / 2)],
        #             dropout=0.1,
        #         )
        #     )

        # self.aggregator = sequential_from_layer_sizes(
        #     nn_input_size + self.kernel_count * self.kernel_size,
        #     nn_input_size,
        #     hidden_sizes=[
        #         int(nn_input_size + self.kernel_count * self.kernel_size / 2)
        #     ],
        # )
        # V3
        # ===================================================================
        self.rate = 0.005
        self.kernel_count = 10
        # self.kernel_size = 32
        self.kernels = nn.ModuleList()
        for _ in range(self.kernel_count):
            self.kernels.append(
                sequential_from_layer_sizes(
                    nn_input_size,
                    nn_input_size + 1,
                    [nn_input_size],
                    dropout=0.1,
                )
            )

        # self.aggregator = sequential_from_layer_sizes(
        #     nn_input_size,
        #     self.kernel_count,
        #     [int(embedding_size / 2)],
        #     dropout=0.1,
        # )
        # ===================================================================

    def forward(
        self, embedding_l: Tensor, embedding_r: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        prev_doubled_embeddings = torch.cat((embedding_l, embedding_r), dim=-1)
        nn_output = self.nn(prev_doubled_embeddings)

        (
            decoded_edges_with_mask,
            doubled_embeddings,
            mem_overwrite_ratio,
        ) = torch.split(
            nn_output,
            [
                self.edge_with_mask_block_size,
                self.embedding_size * 2,
                self.embedding_size * 2,
            ],
            dim=2,
        )

        decoded_edges_with_mask = decoded_edges_with_mask.view(
            *decoded_edges_with_mask.shape[:-1],
            self.block_size,
            self.block_size,
            self.edge_size + 1,
        )

        doubled_embeddings = weighted_average(
            doubled_embeddings, prev_doubled_embeddings, mem_overwrite_ratio
        )

        # # O tutaj!
        # if doubled_embeddings.shape[1] != 1:
        #     kernel_results = torch.cat(
        #         [kernel(doubled_embeddings).mean(dim=1) for kernel in self.kernels],
        #         dim=-1,
        #     )[:, None, :]
        #     kernel_results = kernel_results.expand(-1, doubled_embeddings.shape[1], -1)

        #     kernel_changes = self.aggregator(
        #         torch.cat((doubled_embeddings, kernel_results), dim=-1)
        #     )
        #     doubled_embeddings = (
        #         doubled_embeddings * (1 - self.rate) + self.rate * kernel_changes
        #     )

        # V3
        # ===================================================================
        if doubled_embeddings.shape[1] != 1:
            kernel_results = torch.stack(
                [kernel(doubled_embeddings) for kernel in self.kernels],
                dim=1,
            )

            attention = torch.sigmoid(kernel_results[:, :, :, -1:])
            kernel_changes = kernel_results[:, :, :, :-1]

            attention = attention / attention.sum(dim=2)[:, :, None, :]

            changes = torch.matmul(
                kernel_changes.transpose(-2, -1), attention
            )  # CZY TO MA SENS?!

            aggreagted_changes = (
                torch.mean(changes, dim=1)
                .transpose(-2, -1)
                .expand(-1, doubled_embeddings.shape[1], -1)
            )  # TODO agregator bazujacy na embeddingu - czego on chce

            # graph_kernel = self.aggregator(doubled_embeddings)[:, :, :, None]
            # changes = changes.permute(0, 3, 2, 1).expand(
            #     -1, doubled_embeddings.shape[1], -1, -1
            # )

            # aggreagted_changes = torch.matmul(changes, graph_kernel)[..., 0]

            doubled_embeddings = (
                doubled_embeddings * (1 - self.rate) + aggreagted_changes * self.rate
            )

        # ===================================================================

        new_embedding_l, new_embedding_r = torch.split(
            doubled_embeddings,
            [self.embedding_size, self.embedding_size],
            dim=-1,
        )

        return decoded_edges_with_mask, new_embedding_l, new_embedding_r

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--decoder_hidden_layer_sizes",
            dest="decoder_hidden_layer_sizes",
            default=[256],
            type=parse_layer_sizes_list,
            metavar="DECODER_H_SIZES",
            help="list of the sizes of the decoder's hidden layers",
        )
        parser.add_argument(
            "--decoder_activation_function",
            dest="decoder_activation_function",
            default="ReLU",
            type=str,
            metavar="ACTIVATION_F_NAME",
            help="name of the activation function of hidden layers",
        )
        return parent_parser


class ZeroFillingMemoryEdgeDecoder(MemoryEdgeDecoder):
    def __init__(
        self,
        embedding_size: int,
        edge_decoder_filling_nn_layer_sizes: List[int],
        edge_decoder_filling_nn_activation_function: str,
        **kwargs,
    ):
        super().__init__(embedding_size, **kwargs)

        self.input_embedding_filling_nn = sequential_from_layer_sizes(
            embedding_size,
            embedding_size * 2,
            edge_decoder_filling_nn_layer_sizes,
            edge_decoder_filling_nn_activation_function,
        )

    def forward(
        self, embedding_l: Tensor, embedding_r: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if torch.count_nonzero(embedding_r) == 0:
            embedding_r = self.create_missing_embedding(embedding_l)
        elif torch.count_nonzero(embedding_l) == 0:
            embedding_l = self.create_missing_embedding(embedding_r)

        super().forward(embedding_l, embedding_r)

    def create_missing_embedding(self, other_embedding: Tensor) -> Tensor:
        filling_nn_output = self.input_embedding_filling_nn(other_embedding)
        new_embedding, weight = torch.split(
            filling_nn_output, (self.embedding_size, self.embedding_size), dim=-1
        )
        return weighted_average(new_embedding, other_embedding, weight)

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = MemoryEdgeDecoder.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--edge_decoder_filling_nn_layer_sizes",
            dest="edge_decoder_filling_nn_layer_sizes",
            default=[],
            type=parse_layer_sizes_list,
            metavar="EDGE_DECODER_FILL_H_SIZES",
            help="list of the hidden layer sizes of the edge decoder's input embedding filling nn",
        )
        parser.add_argument(
            "--edge_decoder_filling_nn_activation_function",
            dest="edge_decoder_filling_nn_activation_function",
            default="ELU",
            type=str,
            metavar="ACTIVATION_F_NAME",
            help="name of the activation function of the edge decoderr's input embedding filling nn",
        )
        return parent_parser
