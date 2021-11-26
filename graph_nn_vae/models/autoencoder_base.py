from argparse import ArgumentParser
from typing import Callable, List, Tuple

import torch
from torch import Tensor

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.autoencoder_components import GraphEncoder, GraphDecoder
from graph_nn_vae.models.utils.getters import get_loss


class GraphAutoencoder(BaseModel):
    is_with_graph_mask = False

    def __init__(self, mask_loss_function: str = None, mask_loss_weight=None, **kwargs):
        super(GraphAutoencoder, self).__init__(**kwargs)
        if mask_loss_function is not None:
            self.is_with_graph_mask = True
            self.edge_loss_function = self.loss_function
            self.mask_loss_function = get_loss(mask_loss_function, mask_loss_weight)

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        if not self.is_with_graph_mask:
            return super().step(batch, metrics)

        y_pred = self(batch)
        y_edge, y_mask, y_pred_edge, y_pred_mask = self.adjust_y_to_prediction(
            batch, y_pred
        )

        mask = y_pred_mask > float("-inf")
        y_pred_edge_l, y_pred_mask_l, y_edge_l, y_mask_l = (
            y_pred_edge[mask],
            y_pred_mask[mask],
            y_edge[mask],
            y_mask[mask],
        )

        loss_edges = self.edge_loss_function(y_pred_edge_l, y_edge_l.data)
        loss_mask = self.mask_loss_function(y_pred_mask_l, y_mask_l)
        loss = loss_edges + loss_mask

        for metric in metrics:
            metric(
                edges_predicted=y_pred_edge,
                edges_target=y_edge,
                mask_predicted=y_pred_mask,
                mask_target=y_mask,
            )

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = BaseModel.add_model_specific_args(parent_parser=parser)
        parser.add_argument(
            "--mask_loss_function",
            dest="mask_loss_function",
            default=None,
            type=str,
            metavar="MASK_LOSS_F_NAME",
            help="name of loss function for the graph mask, if used",
        )
        return parser


class RecurrentGraphAutoencoder(GraphAutoencoder):
    model_name = ""

    def __init__(self, max_number_of_nodes: int, **kwargs):
        self.max_number_of_nodes = max_number_of_nodes
        super(RecurrentGraphAutoencoder, self).__init__(**kwargs)
        self.encoder = GraphEncoder(**kwargs)
        self.decoder = GraphDecoder(max_number_of_nodes=max_number_of_nodes, **kwargs)

    def forward(self, batch: Tensor) -> Tensor:
        num_nodes_batch = batch[2]
        max_num_nodes_in_graph_batch = max(num_nodes_batch)
        if max_num_nodes_in_graph_batch > self.max_number_of_nodes:
            raise ValueError(
                "the max number of nodes of the requested reconstructed graphs cannot "
                + "be lower than the number of nodes of the biggest input graph"
            )
        graph_embdeddings = self.encoder(batch)
        reconstructed_graph_diagonals = self.decoder(graph_embdeddings)
        return reconstructed_graph_diagonals

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

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = GraphAutoencoder.add_model_specific_args(parent_parser=parser)
        parser = GraphEncoder.add_model_specific_args(parent_parser=parser)
        parser = GraphDecoder.add_model_specific_args(parent_parser=parser)
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

    padded_t = torch.nn.functional.pad(
        padded_t,
        [0] * (dim * 2 + 1) + [abs(diff)],
        value=padding_value,
    )
    return (padded_t, t2) if diff < 0 else (t1, padded_t)
