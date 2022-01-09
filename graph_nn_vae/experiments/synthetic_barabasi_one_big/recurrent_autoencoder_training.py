from argparse import ArgumentParser
from typing import List
import networkx as nx
import numpy as np

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.experiments.decorators import add_graphloader_args
from graph_nn_vae.data import (
    BaseGraphLoader,
    DiagonalRepresentationGraphDataModule,
)
from graph_nn_vae.models.autoencoder_base import RecurrentGraphAutoencoder


class ExperimentModel(RecurrentGraphAutoencoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecurrentGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCEWithLogits",
            mask_loss_function="BCEWithLogits",
            mask_loss_weight=0.5,
            diagonal_embeddings_loss_weight=0.2,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="NoSched",
            # lr_scheduler_metric="max_graph_size/train_avg",
            # lr_scheduler_params={"factor": 0.9},
            learning_rate=0.001,
            gradient_clip_val=1.0,
            batch_size=1,
            embedding_size=256,
            block_size=5,
            encoder_hidden_layer_sizes=[2048],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[2048],
            decoder_activation_function="ELU",
            subgraph_scheduler_name="none",
            metrics=[
                "EdgePrecision",
                "EdgeRecall",
                "MaskPrecision",
                "MaskRecall",
                "MaxGraphSize",
            ],
            max_epochs=100000,
            check_val_every_n_epoch=30,
            metric_update_interval=2,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=1,
        )
        return parser


class OneBigBarabasiGraphLoader(BaseGraphLoader):
    data_name = "one_big_barabasi"

    def __init__(self, barabasi_size, **kwargs):
        self.barabasi_size = barabasi_size
        super().__init__(**kwargs)

    def load_graphs(self) -> List[nx.Graph]:
        return {
            "graphs": [
                nx.to_numpy_array(
                    nx.barabasi_albert_graph(self.barabasi_size, 4), dtype=np.float32
                )
            ]
        }

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseGraphLoader.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--barabasi_size",
            dest="barabasi_size",
            default=100,
            type=int,
            help="size of one barabasi graph",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = OneBigBarabasiGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
