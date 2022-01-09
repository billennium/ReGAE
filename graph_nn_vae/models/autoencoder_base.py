from argparse import ArgumentParser
from typing import Callable, List, Tuple
import math

import torch
from torch import Tensor

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.autoencoder_components import GraphEncoder, GraphDecoder
from graph_nn_vae.models.edge_encoders.memory_standard import MemoryEdgeEncoder
from graph_nn_vae.models.edge_decoders.memory_standard import MemoryEdgeDecoder
from graph_nn_vae.models.utils.getters import get_loss

from graph_nn_vae.util.adjmatrix.diagonal_block_representation import (
    calculate_num_blocks,
)
from graph_nn_vae.models.utils.calc import torch_bincount


class GraphAutoencoder(BaseModel):
    def __init__(
        self,
        loss_function: str,
        recall_to_precision_bias: float,
        mask_loss_function: str = None,
        mask_loss_weight=None,
        diagonal_embeddings_loss_weight: int = 0,
        weight_power_level: float = 1,
        **kwargs,
    ):
        super(GraphAutoencoder, self).__init__(loss_function=loss_function, **kwargs)
        self.mask_loss_function = get_loss(mask_loss_function, mask_loss_weight)

        self.edge_1_loss_function = get_loss(loss_function, recall_to_precision_bias)
        self.edge_0_loss_function = get_loss(
            loss_function, (1 - recall_to_precision_bias)
        )
        self.diagonal_embeddings_loss_weight = diagonal_embeddings_loss_weight
        self.weight_power_level = weight_power_level

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        y_pred, diagonal_embeddings_norm = self(batch)

        y_edge, y_mask, y_pred_edge, y_pred_mask = self.adjust_y_to_prediction(
            batch, y_pred
        )

        loss_reconstruction = self.calc_reconstruction_loss(
            y_edge, y_mask, y_pred_edge, y_pred_mask, batch[2]
        )
        loss_embeddings = (
            diagonal_embeddings_norm * self.diagonal_embeddings_loss_weight
        )
        loss = loss_reconstruction + loss_embeddings

        shared_metric_state = {}
        for metric in metrics:
            metric.update(
                edges_predicted=y_pred_edge,
                edges_target=y_edge,
                mask_predicted=y_pred_mask,
                mask_target=y_mask,
                num_nodes=batch[2],
                loss_reconstruction=loss_reconstruction,
                loss_embeddings=loss_embeddings,
                shared_metric_state=shared_metric_state,
            )

        return loss

    def calc_reconstruction_loss(
        self, y_edge, y_mask, y_pred_edge, y_pred_mask, num_nodes
    ) -> Tensor:
        block_size = y_edge.shape[2] if len(y_edge.shape) == 5 else 1
        if block_size != 1:
            num_blocks = calculate_num_blocks(num_nodes, block_size)
        else:
            num_blocks = num_nodes

        graph_counts_per_size = torch_bincount(num_blocks)

        losses_edge_1 = torch.zeros(1, device=y_edge.device)
        losses_edge_0 = torch.zeros(1, device=y_edge.device)
        losses_mask = torch.zeros(1, device=y_edge.device)
        weights_edge_0 = torch.zeros(1, device=y_edge.device)
        weights_edge_1 = torch.zeros(1, device=y_edge.device)
        weights_mask = torch.zeros(1, device=y_edge.device)

        for size, count in enumerate(graph_counts_per_size):
            if not count:
                continue

            size_mask = num_blocks == size
            y_pred_edge_per_size = y_pred_edge[size_mask]
            y_pred_mask_per_size = y_pred_mask[size_mask]
            y_edge_per_size = y_edge[size_mask]
            y_mask_per_size = y_mask[size_mask]

            mask = y_pred_mask_per_size > float("-inf")
            (
                y_pred_edge_l_per_size,
                y_pred_mask_l_per_size,
                y_edge_l_per_size,
                y_mask_l_per_size,
            ) = (
                y_pred_edge_per_size[mask],
                y_pred_mask_per_size[mask],
                y_edge_per_size[mask],
                y_mask_per_size[mask],
            )
            y_edge_l_per_size = torch.clamp(y_edge_l_per_size, min=0)
            y_mask_l_per_size = torch.clamp(y_mask_l_per_size, min=0)

            y_edge_1_mask_per_size = y_edge_l_per_size.data == 1
            y_edge_l_1_per_size = y_edge_l_per_size[y_edge_1_mask_per_size]
            y_edge_l_0_per_size = y_edge_l_per_size[~y_edge_1_mask_per_size]
            y_pred_edge_l_1_per_size = y_pred_edge_l_per_size[y_edge_1_mask_per_size]
            y_pred_edge_l_0_per_size = y_pred_edge_l_per_size[~y_edge_1_mask_per_size]

            loss_edge_1_per_size = (
                self.edge_1_loss_function(y_pred_edge_l_1_per_size, y_edge_l_1_per_size)
                if len(y_edge_l_1_per_size > 0)
                else 0.0
            )
            loss_edge_0_per_size = (
                self.edge_0_loss_function(y_pred_edge_l_0_per_size, y_edge_l_0_per_size)
                if len(y_edge_l_0_per_size > 0)
                else 0.0
            )

            loss_mask_per_size = self.mask_loss_function(
                y_pred_mask_l_per_size, y_mask_l_per_size
            )

            wieght = pow(size * block_size, 2 - self.weight_power_level) * count
            if len(y_edge_l_per_size):
                weight_edge_0 = wieght * (
                    len(y_edge_l_0_per_size) / len(y_edge_l_per_size)
                )
                weight_edge_1 = wieght * (
                    1 - len(y_edge_l_0_per_size) / len(y_edge_l_per_size)
                )
            else:
                print("Warning ,,y_edge_l_per_size`` has len 0")
                weight_edge_0 = 0
                weight_edge_1 = 0

            weight_mask = wieght

            weights_edge_0 += weight_edge_0
            weights_edge_1 += weight_edge_1
            weights_mask += wieght
            losses_edge_0 += loss_edge_0_per_size * weight_edge_0
            losses_edge_1 += loss_edge_1_per_size * weight_edge_1
            losses_mask += loss_mask_per_size * weight_mask

        return (
            ((losses_edge_0 / weights_edge_0) if weight_edge_0 else 0)
            + ((losses_edge_1 / weights_edge_1) if weights_edge_1 else 0)
            + ((losses_mask / weights_mask) if weights_mask else 0)
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser=parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--recall_to_precision_bias",
            dest="recall_to_precision_bias",
            default=0.5,
            type=float,
            metavar="BIAS_VALUE",
            help="""
                Weight of loss function for the graph's adjacency matrix 0s in proportion to the weight for 1s.
                Affects the precision to recall balance, with higher values favoring better recall at the
                cost of worse precision. [0.0 - 1.0]
                """,
        )
        parser.add_argument(
            "--weight_power_level",
            dest="weight_power_level",
            default=1,
            type=float,
            metavar="WEIGHT_POWER_LEVEL",
            help="normalization weight per edge, level of power (0 each edges has the same weight, 1 proprtional to 1/N, 2, proportional to 1/(N^2))",
        )
        parser.add_argument(
            "--mask_loss_function",
            dest="mask_loss_function",
            default=None,
            type=str,
            metavar="MASK_LOSS_F_NAME",
            help="name of loss function for the graph mask",
        )
        parser.add_argument(
            "--mask_loss_weight",
            dest="mask_loss_weight",
            default=1.0,
            type=float,
            metavar="MASK_LOSS_WEIGHT",
            help="weight of loss function for the graph mask",
        )
        parser.add_argument(
            "--diagonal_embeddings_loss_weight",
            dest="diagonal_embeddings_weight",
            default=0.2,
            type=float,
            metavar="DIAGONAL_EMBEDDINGS_LOSS_WEIGHT",
            help="weight of loss function for the graph diagonal embeddings norm",
        )
        return parent_parser


class RecurrentGraphAutoencoder(GraphAutoencoder):
    model_name = "RecurrentGraphAutoencoder"

    graph_encoder_class = GraphEncoder
    edge_encoder_class = MemoryEdgeEncoder
    graph_decoder_class = GraphDecoder
    edge_decoder_class = MemoryEdgeDecoder

    def __init__(self, **kwargs):
        super(RecurrentGraphAutoencoder, self).__init__(**kwargs)
        self.encoder = self.graph_encoder_class(self.edge_encoder_class, **kwargs)
        self.decoder = self.graph_decoder_class(
            edge_decoder_class=self.edge_decoder_class,
            **kwargs,
        )

    def forward(self, batch: Tensor) -> Tensor:
        num_nodes_batch = batch[2]
        max_num_nodes_in_graph_batch = max(num_nodes_batch)

        graph_embeddings = self.encoder(batch)

        reconstructed_graph_diagonals, diagonal_embeddings_norm = self.decoder(
            graph_encoding_batch=graph_embeddings,
            max_number_of_nodes=max_num_nodes_in_graph_batch,
        )

        return reconstructed_graph_diagonals, diagonal_embeddings_norm

    # override
    def adjust_y_to_prediction(self, batch, y_predicted) -> Tuple[Tensor, Tensor]:
        diagonal_repr_graphs = batch[0]
        graph_masks = batch[1]
        predicted_graphs = y_predicted[0]
        predicted_graph_masks = y_predicted[1]
        diagonal_repr_graphs, predicted_graphs = equalize_dim_by_padding(
            diagonal_repr_graphs, predicted_graphs, 1, 0.0, float("-inf")
        )
        graph_masks, predicted_graph_masks = equalize_dim_by_padding(
            graph_masks, predicted_graph_masks, 1, 0.0, float("-inf")
        )
        return (
            diagonal_repr_graphs,
            graph_masks,
            predicted_graphs,
            predicted_graph_masks,
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = GraphAutoencoder.add_model_specific_args(parent_parser=parent_parser)
        parser = cls.graph_encoder_class.add_model_specific_args(parent_parser=parser)
        parser = cls.graph_decoder_class.add_model_specific_args(parent_parser=parser)
        parser = cls.edge_encoder_class.add_model_specific_args(parent_parser=parser)
        parser = cls.edge_decoder_class.add_model_specific_args(parent_parser=parser)
        return parser


def equalize_dim_by_padding(
    t1: Tensor, t2: Tensor, dim: int, padding_value_1, padding_value_2
) -> Tuple[Tensor, Tensor]:
    diff = t1.shape[dim] - t2.shape[dim]
    if diff < 0:
        padded_t = t1
        padding_value = padding_value_1
    else:
        padded_t = t2
        padding_value = padding_value_2

    if padded_t.type() == torch.bool:
        padding_value = (
            True if padding_value == 1.0 or padding_value == float("inf") else False
        )

    pad_dims = [0] * ((padded_t.ndim - dim) * 2 - 1) + [abs(diff)]
    padded_t = torch.nn.functional.pad(padded_t, pad_dims, value=padding_value)
    return (padded_t, t2) if diff < 0 else (t1, padded_t)
