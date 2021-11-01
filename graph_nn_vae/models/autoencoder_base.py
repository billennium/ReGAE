from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
import pytorch_lightning as pl

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

    def forward(self, adjacency_matrices_batch: Tensor) -> Tensor:
        num_nodes_in_graphs = adjacency_matrices_batch.shape[1]
        if num_nodes_in_graphs > self.max_number_of_nodes:
            raise ValueError(
                "the max number of nodes of the requested reconstructed graphs cannot "
                + "be lower than the number of nodes of the biggest input graph"
            )
        graph_embdeddings = self.encoder(adjacency_matrices_batch)
        reconstructed_graph_diagonals = self.decoder(graph_embdeddings)
        return reconstructed_graph_diagonals

    def step(self, batch: Tensor) -> Tensor:
        reconstructed_graph_diagonals = self.forward(batch)
        loss = util.get_reconstruction_loss(
            batch,
            reconstructed_graph_diagonals,
            torch.nn.MSELoss(),
        )
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = GraphEncoder.add_model_specific_args(parent_parser=parser)
        parser = GraphDecoder.add_model_specific_args(parent_parser=parser)
        return parser
