from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
import pytorch_lightning as pl

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.autoencoder_components import GraphEncoder, GraphDecoder


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

    def _get_reconstruction_loss(
        self, adjacency_matrices_batch: Tensor, reconstructed_graph_diagonals: Tensor
    ) -> Tensor:
        input_concatenated_diagonals = []
        for adjacency_matrix in adjacency_matrices_batch:
            diagonals = [
                torch.diagonal(adjacency_matrix, offset=-i).transpose(1, 0)
                for i in reversed(range(adjacency_matrix.shape[0]))
            ]
            for i in reversed(range(len(diagonals))):
                if torch.count_nonzero(diagonals[i]) > 1:
                    diagonals[i + 1] = diagonals[i + 1].fill_(-1.0)
                    break
            concatenated_diagonals = torch.cat(diagonals, dim=0)
            input_concatenated_diagonals.append(concatenated_diagonals)
        input_batch_reshaped = torch.stack(input_concatenated_diagonals)

        reconstructed_diagonals_length = reconstructed_graph_diagonals.shape[1]
        input_pad_length = (
            reconstructed_diagonals_length - input_batch_reshaped.shape[1]
        )
        input_batch_reshaped = torch.nn.functional.pad(
            input_batch_reshaped, (0, 0, 0, input_pad_length)
        )

        loss_f = torch.nn.MSELoss()
        loss = loss_f(reconstructed_graph_diagonals, input_batch_reshaped)
        return loss

    def step(self, batch: Tensor) -> Tensor:
        reconstructed_graph_diagonals = self.forward(batch)
        loss = self._get_reconstruction_loss(batch, reconstructed_graph_diagonals)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = GraphEncoder.add_model_specific_args(parent_parser=parser)
        parser = GraphDecoder.add_model_specific_args(parent_parser=parser)
        return parser
