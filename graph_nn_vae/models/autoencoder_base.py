from argparse import ArgumentParser
from typing import Callable, List, Tuple

import torch
from torch import Tensor

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.autoencoder_components import GraphEncoder, GraphDecoder
from graph_nn_vae.models.edge_encoders.memory_standard import MemoryEdgeEncoder
from graph_nn_vae.models.edge_decoders.memory_standard import MemoryEdgeDecoder
from graph_nn_vae.models.utils.getters import get_loss

from graph_nn_vae.util.adjmatrix.diagonal_block_representation import (
    diagonal_block_to_adj_matrix_representation,
)
from graph_nn_vae.util.adjmatrix.diagonal_representation import (
    adj_matrix_to_diagonal_representation,
)


class GraphAutoencoder(BaseModel):
    is_with_graph_mask = False

    def __init__(
        self,
        loss_function: str,
        edge_1_loss_weight: float,
        edge_0_loss_weight: float,
        mask_loss_function: str = None,
        mask_loss_weight=None,
        diagonal_embeddings_loss_weight: int = 0,
        weight_loss_positive_edges: float = 1.0,
        **kwargs,
    ):
        super(GraphAutoencoder, self).__init__(loss_function=loss_function, **kwargs)
        self.mask_loss_function = get_loss(mask_loss_function, mask_loss_weight)
        self.edge_1_loss_function = get_loss(loss_function, edge_1_loss_weight * 0.5)
        self.edge_0_loss_function = get_loss(loss_function, edge_0_loss_weight * 0.5)
        self.diagonal_embeddings_loss_weight = diagonal_embeddings_loss_weight
        self.weight_loss_positive_edges = weight_loss_positive_edges

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        y_pred, diagonal_embeddings_norm = self(batch)

        y_edge, y_mask, y_pred_edge, y_pred_mask = self.adjust_y_to_prediction(
            batch, y_pred
        )

        loss_reconstruction = self.calc_reconstruction_loss(
            y_edge, y_mask, y_pred_edge, y_pred_mask
        )
        loss_embeddings = (
            diagonal_embeddings_norm * self.diagonal_embeddings_loss_weight
        )
        loss = loss_reconstruction + loss_embeddings

        for metric in metrics:
            metric(
                edges_predicted=y_pred_edge,
                edges_target=y_edge,
                mask_predicted=y_pred_mask,
                mask_target=y_mask,
                num_nodes=batch[2],
                loss_reconstruction=loss_reconstruction,
                loss_embeddings=loss_embeddings,
            )

        return loss

    def calc_reconstruction_loss(
        self, y_edge, y_mask, y_pred_edge, y_pred_mask
    ) -> Tensor:
        mask = y_pred_mask > float("-inf")
        y_pred_edge_l, y_pred_mask_l, y_edge_l, y_mask_l = (
            y_pred_edge[mask],
            y_pred_mask[mask],
            y_edge[mask],
            y_mask[mask],
        )
        y_edge_l = torch.clamp(y_edge_l, min=0)
        y_mask_l = torch.clamp(y_mask_l, min=0)

        y_edge_1_mask = y_edge_l.data == 1
        y_edge_l_1 = y_edge_l[y_edge_1_mask]
        y_edge_l_0 = y_edge_l[~y_edge_1_mask]
        y_pred_edge_l_1 = y_pred_edge_l[y_edge_1_mask]
        y_pred_edge_l_0 = y_pred_edge_l[~y_edge_1_mask]

        loss_edge_1 = (
            self.edge_1_loss_function(y_pred_edge_l_1, y_edge_l_1)
            if len(y_edge_l_1 > 0)
            else 0.0
        )
        loss_edge_0 = (
            self.edge_1_loss_function(y_pred_edge_l_0, y_edge_l_0)
            if len(y_edge_l_0 > 0)
            else 0.0
        )
        loss_mask = self.mask_loss_function(y_pred_mask_l, y_mask_l)
        return loss_edge_1 + loss_edge_0 + loss_mask

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser=parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--edge_1_loss_weight",
            dest="edge_1_loss_weight",
            default=1.0,
            type=float,
            metavar="MASK_LOSS_WEIGHT",
            help="weight of loss function for the graph's adjacency matrix 1s",
        )
        parser.add_argument(
            "--edge_0_loss_weight",
            dest="edge_0_loss_weight",
            default=1.0,
            type=float,
            metavar="MASK_LOSS_WEIGHT",
            help="weight of loss function for the graph's adjacency matrix 0s",
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
