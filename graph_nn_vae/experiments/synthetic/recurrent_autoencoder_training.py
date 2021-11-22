import argparse
import networkx as nx
from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import SyntheticGraphsDataModule
from graph_nn_vae.models.autoencoder_base import RecurrentGraphAutoencoder

from graph_nn_vae.data.data_module import BaseDataModule
import networkx as nx
from graph_nn_vae.util import adjmatrix, split_dataset_train_val_test
import torch
import numpy as np


class GraphAutoencoder(RecurrentGraphAutoencoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecurrentGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            embedding_size=128,
            batch_size=32,
            learning_rate=0.0005,
            max_number_of_nodes=16,
            gradient_clip_val=0.01,
            max_epochs=10000,
            check_val_every_n_epoch=20,
            encoder_hidden_layer_sizes=[512],
            decoder_hidden_layer_sizes=[512],
            # metrics=[
            #     "Accuracy",
            #     "PositivePrecision",
            #     "PositiveRecall",
            #     "NegativePrecision",
            #     "NegativeRecall",
            # ],
            metric_update_interval=20,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=10
        )
        return parser


if __name__ == "__main__":
    Experiment(GraphAutoencoder, SyntheticGraphsDataModule).run()
