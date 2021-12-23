from argparse import ArgumentParser
from typing import List
import networkx as nx
import numpy as np

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import (
    GraphLoaderBase,
    DiagonalBlockRepresentationGraphDataModule,
)
from graph_nn_vae.models.autoencoder_base import RecurrentGraphAutoencoder
from graph_nn_vae.models.autoencoder_components import (
    GraphDecoder,
)
from graph_nn_vae.models.edge_decoders.memory_standard import (
    MemoryEdgeDecoder,
    ZeroFillingMemoryEdgeDecoder,
)
from graph_nn_vae.models.edge_decoders.single_input_embedding import (
    MeanSingleInputMemoryEdgeDecoder,
    RandomSingleInputMemoryEdgeDecoder,
    WeightingSingleInputMemoryEdgeDecoder,
)


class GraphAutoencoder(RecurrentGraphAutoencoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        RecurrentGraphAutoencoder.graph_decoder_class = GraphDecoder
        RecurrentGraphAutoencoder.edge_decoder_class = MemoryEdgeDecoder

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
            subgraph_scheduler_name="no_graph_scheduler",
            # subgraph_stride=0.5,
            # minimal_subgraph_size=10,
            # subgraph_scheduler_params={
            #     "subgraph_size_initial": 0.005,
            #     "metrics_treshold": 0.6,
            #     "step": 0.05,
            # },
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


class OneBigBarabasiGraphLoader(GraphLoaderBase):
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
        parser = GraphLoaderBase.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--barabasi_size",
            dest="barabasi_size",
            default=100,
            type=int,
            help="size of one barabasi graph",
        )
        return parser


if __name__ == "__main__":
    Experiment(
        GraphAutoencoder,
        DiagonalBlockRepresentationGraphDataModule,
        OneBigBarabasiGraphLoader,
    ).run()
