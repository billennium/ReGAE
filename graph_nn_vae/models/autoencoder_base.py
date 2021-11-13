from argparse import ArgumentParser
from typing import Tuple

import torch
from torch import Tensor

from graph_nn_vae import util
from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.autoencoder_components import GraphEncoder, GraphDecoder
from graph_nn_vae import util


class RecurrentGraphAutoencoder(BaseModel):
    model_name = ""

    def __init__(self, max_number_of_nodes: int, **kwargs):
        self.max_number_of_nodes = max_number_of_nodes
        super(RecurrentGraphAutoencoder, self).__init__(**kwargs)
        self.encoder = GraphEncoder(**kwargs)
        self.decoder = GraphDecoder(max_number_of_nodes=max_number_of_nodes, **kwargs)

    def forward(self, batch: Tensor) -> Tensor:
        num_nodes_batch = batch[1]
        max_num_nodes_in_graph_batch = max(num_nodes_batch)
        if max_num_nodes_in_graph_batch > self.max_number_of_nodes:
            raise ValueError(
                "the max number of nodes of the requested reconstructed graphs cannot "
                + "be lower than the number of nodes of the biggest input graph"
            )
        graph_embdeddings = self.encoder(batch)
        reconstructed_graph_diagonals = self.decoder(graph_embdeddings)
        return reconstructed_graph_diagonals

    def adjust_y_to_prediction(self, batch, y_predicted) -> Tuple[Tensor, Tensor]:
        diagonal_repr_graphs = batch[0]
        diagonal_repr_len = diagonal_repr_graphs.shape[1]
        y_predicted_len = y_predicted.shape[1]
        if diagonal_repr_len > y_predicted_len:
            y_predicted = torch.nn.functional.pad(
                y_predicted,
                (0, 0, 0, diagonal_repr_len - y_predicted_len),
                value=-1.0,
            )
        elif y_predicted_len > diagonal_repr_len:
            diagonal_repr_graphs = torch.nn.functional.pad(
                diagonal_repr_graphs,
                (0, 0, 0, y_predicted_len - diagonal_repr_len),
                value=-1.0,
            )
        return diagonal_repr_graphs, y_predicted

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = GraphEncoder.add_model_specific_args(parent_parser=parser)
        parser = GraphDecoder.add_model_specific_args(parent_parser=parser)
        return parser
