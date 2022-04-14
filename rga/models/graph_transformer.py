from argparse import ArgumentParser
from typing import Callable, List, Tuple
import math

import torch
from torch import Tensor

from rga.models.autoencoder_base import RecursiveGraphAutoencoder
from rga.models.autoencoder_components import GraphEncoder, GraphDecoder
from rga.models.edge_encoders.memory_standard import MemoryEdgeEncoder
from rga.models.edge_decoders.memory_standard import MemoryEdgeDecoder
from rga.models.utils.getters import get_loss

from rga.util.adjmatrix.diagonal_block_representation import (
    calculate_num_blocks,
)
from rga.models.utils.calc import torch_bincount


class RecursiveGraphTransformer(RecursiveGraphAutoencoder):
    model_name = "RecursiveGraphTransformer"

    graph_encoder_class = GraphEncoder
    edge_encoder_class = MemoryEdgeEncoder
    graph_decoder_class = GraphDecoder
    edge_decoder_class = MemoryEdgeDecoder

    def __init__(self, **kwargs):
        super(RecursiveGraphAutoencoder, self).__init__(**kwargs)
        self.encoder = self.graph_encoder_class(self.edge_encoder_class, **kwargs)
        self.decoder = self.graph_decoder_class(
            edge_decoder_class=self.edge_decoder_class,
            **kwargs,
        )

    # TODO do przemyślenia, czy nie powinno byc defaultowe
    def forward(self, batch: Tensor) -> Tensor:
        batch_input_graphs = batch[0]
        batch_output_graphs = batch[1]

        num_nodes_batch = batch_output_graphs[2]
        max_num_nodes_in_graph_batch = max(num_nodes_batch)

        graph_embeddings = self.encoder(batch_input_graphs)

        reconstructed_graph_diagonals, diagonal_embeddings_norm = self.decoder(
            graph_encoding_batch=graph_embeddings,
            max_number_of_nodes=max_num_nodes_in_graph_batch,
        )

        return reconstructed_graph_diagonals, diagonal_embeddings_norm

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        y_pred, diagonal_embeddings_norm = self(batch)

        y_edge, y_mask, y_pred_edge, y_pred_mask = self.adjust_y_to_prediction(
            batch[1], y_pred
        )

        loss_reconstruction = self.calc_reconstruction_loss(
            y_edge, y_mask, y_pred_edge, y_pred_mask, batch[1][2]
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
                num_nodes=batch[1][2],
                loss_reconstruction=loss_reconstruction,
                loss_embeddings=loss_embeddings,
                shared_metric_state=shared_metric_state,
            )

        return loss
