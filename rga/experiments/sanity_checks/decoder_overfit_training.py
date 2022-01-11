from argparse import ArgumentParser
from typing import Tuple
import networkx as nx
import numpy as np

import torch
from torch import Tensor

from rga.data import SyntheticGraphsDataModule
from rga.experiments.experiment import Experiment
from rga.models.autoencoder_components import GraphDecoder
from rga import util
from rga.util import adjmatrix, split_dataset_train_val_test
from rga.data.data_module import BaseDataModule
from rga.data.synthetic_graphs_create import create_synthetic_graphs


class OverfitDecoder(GraphDecoder):
    def __init__(self, embedding_size: int, **kwargs):
        super().__init__(embedding_size=embedding_size, **kwargs)
        self.input_adapter_layer = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embedding_size),
        )

    def step(self, batch: Tensor) -> Tensor:
        graph_codes = batch[2]
        graph_embeddings = self.input_adapter_layer(graph_codes[:, None])
        y_predicted = self(graph_embeddings)
        y, y_predicted = self.adjust_y_to_prediction(batch, y_predicted)
        loss = self.loss_function(y_predicted, y)
        # for metric in self.metrics:
        #     metric(y_hat, y)
        return loss

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
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GraphDecoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            batch_size=32,
            learning_rate=0.0005,
            gradient_clip_val=0.01,
            max_epochs=100000,
            check_val_every_n_epoch=100,
            embedding_size=64,
            max_number_of_nodes=17,
            num_dataset_graph_permutations=1,
        )
        return parser


class CodedGraphsDataModule(SyntheticGraphsDataModule):
    data_name = "synthethic"
    i = 0.0

    def i_incr(self) -> int:
        v = self.i
        self.i += 1
        return v

    def add_codes_to_dataset(self, dataset):
        return [(g, n, torch.tensor(self.i_incr())) for g, n in dataset]

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
        i = 0
        self.train_dataset = self.add_codes_to_dataset(self.train_dataset)
        self.val_dataset = self.add_codes_to_dataset(self.val_dataset)
        self.test_dataset = self.add_codes_to_dataset(self.test_dataset)


if __name__ == "__main__":
    Experiment(OverfitDecoder, CodedGraphsDataModule).run()
