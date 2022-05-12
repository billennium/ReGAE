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

        # V1
        # ===================================================================
        # self.kernel_count = 5
        # self.kernel_size = 32
        # self.kernels = nn.ModuleList()
        # for _ in range(self.kernel_count):
        #     self.kernels.append(
        #         sequential_from_layer_sizes(
        #             embedding_size,
        #             self.kernel_size,
        #             [int((embedding_size + self.kernel_size) / 2)],
        #             dropout=0.1,
        #         )
        #     )

        # self.aggregator = sequential_from_layer_sizes(
        #     embedding_size + self.kernel_count * self.kernel_size,
        #     embedding_size,
        #     hidden_sizes=[
        #         int(embedding_size + self.kernel_count * self.kernel_size / 2)
        #     ],
        # )
        # ===================================================================

        # V2
        # ===================================================================
        # self.kernel_count = 5
        # self.kernel_size = 32
        # self.attention_mechanizm = sequential_from_layer_sizes(
        #     embedding_size * 2 + 1,
        #     1,
        #     hidden_sizes=[int(embedding_size)],  #
        #     output_function=nn.Sigmoid,
        #     dropout=0.1,
        # )

        # self.aggregator = sequential_from_layer_sizes(
        #     embedding_size + self.kernel_count * self.kernel_size,
        #     embedding_size,
        #     hidden_sizes=[
        #         int(embedding_size + self.kernel_count * self.kernel_size / 2)
        #     ],
        # )
        # ===================================================================

        # V3
        # ===================================================================
        self.rate = 0.005
        self.kernel_count = 10
        # self.kernel_size = 32
        self.kernels = nn.ModuleList()
        for _ in range(self.kernel_count):
            self.kernels.append(
                sequential_from_layer_sizes(
                    embedding_size,
                    embedding_size + 1,
                    [embedding_size],
                    dropout=0.1,
                )
            )

        # self.aggregator = sequential_from_layer_sizes(
        #     embedding_size,
        #     self.kernel_count,
        #     [int(embedding_size / 2)],
        #     dropout=0.1,
        # )

        # ===================================================================

    def forward(
        self,
        diagonal_x: Tensor,
        embedding_l: Tensor,
        embedding_r: Tensor,
        subgraph_size: int,
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

        # V3
        # ===================================================================
        if new_embedding.shape[1] != 1:
            kernel_results = torch.stack(
                [kernel(new_embedding) for kernel in self.kernels],
                dim=1,
            )

            attention = torch.sigmoid(kernel_results[:, :, :, -1:])
            kernel_changes = kernel_results[:, :, :, :-1]

            attention = attention / attention.sum(dim=2)[:, :, None, :]

            changes = torch.matmul(
                kernel_changes.transpose(-2, -1), attention
            )  # CZY TO MA SENS?!

            # graph_kernel = self.aggregator(new_embedding)[:, :, :, None]
            # changes = changes.permute(0, 3, 2, 1).expand(
            #     -1, new_embedding.shape[1], -1, -1
            # )

            # aggreagted_changes = torch.matmul(changes, graph_kernel)[..., 0]

            aggreagted_changes = (
                torch.mean(changes, dim=1)
                .transpose(-2, -1)
                .expand(-1, new_embedding.shape[1], -1)
            )  # TODO agregator bazujacy na embeddingu - czego on chce

            new_embedding = (
                new_embedding * (1 - self.rate) + aggreagted_changes * self.rate
            )

        # ===================================================================

        # V2
        # ===================================================================
        # if (new_embedding.shape[1] != 1) and (self.rate != 0):
        #     pairs = torch.zeros(
        #         (
        #             new_embedding.shape[0],
        #             new_embedding.shape[1],
        #             new_embedding.shape[1],
        #             new_embedding.shape[2] * 2 + 1,
        #         ),
        #         device=new_embedding.device,
        #     )

        #     for cur_graph_i, cur_graph in enumerate(new_embedding):
        #         for cur_a, cur_embed_a in enumerate(cur_graph):
        #             for cur_b, cur_embed_b in enumerate(cur_graph):
        #                 pairs[cur_graph_i, cur_a, cur_b] = torch.cat(
        #                     [
        #                         cur_embed_a,
        #                         cur_embed_b,
        #                         torch.clamp(
        #                             torch.tensor(
        #                                 [cur_a - cur_b], device=new_embedding.device
        #                             ),
        #                             min=-subgraph_size,
        #                             max=subgraph_size,
        #                         )
        #                         / subgraph_size,
        #                     ]
        #                 )

        #     attention = self.attention_mechanizm(pairs)[:, :, :, 0]
        #     # for i in range(attention.size(0)):
        #     #     attention[i, :, :].fill_diagonal_(0)
        #     attention = attention / attention.sum(dim=-1)[:, :, None]

        #     changes = torch.matmul(attention, new_embedding)  # CZY TO MA SENS?!

        #     new_embedding = new_embedding * (1 - self.rate) + changes * self.rate

        # ===================================================================

        # V1
        # ===================================================================
        # # # O tutaj!
        # if new_embedding.shape[1] != 1:
        #     kernel_results = torch.cat(
        #         [kernel(new_embedding).mean(dim=1) for kernel in self.kernels],
        #         dim=-1,
        #     )[:, None, :]
        #     kernel_results = kernel_results.expand(-1, new_embedding.shape[1], -1)

        #     kernel_changes = self.aggregator(
        #         torch.cat((new_embedding, kernel_results), dim=-1)
        #     )
        #     new_embedding = new_embedding * (1 - self.rate) + self.rate * kernel_changes
        # ===================================================================

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
        parser.add_argument(
            "--kernel_count",
            dest="kernel_count",
            default="10",
            type=int,
        )
        parser.add_argument(
            "--attention_rate",
            dest="attention_rate",
            default="0.005",
            type=float,
        )
        return parent_parser
