from argparse import ArgumentParser
from typing import List
import networkx as nx

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import (
    GraphLoaderBase,
    SmoothLearningStepGraphDataModule,
)
from graph_nn_vae.models.autoencoder_base import RecurrentGraphAutoencoder
from graph_nn_vae.models.autoencoder_components import (
    BorderFillingGraphDecoder,
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
        RecurrentGraphAutoencoder.graph_decoder_class = BorderFillingGraphDecoder
        RecurrentGraphAutoencoder.edge_decoder_class = MemoryEdgeDecoder

        parser = RecurrentGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCEWithLogits",
            mask_loss_function="BCEWithLogits",
            mask_loss_weight=0.5,
            diagonal_embeddings_loss_weight=0.2,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="FactorDecreasingOnMetricChange",
            lr_scheduler_metric="max_graph_size/train_avg",
            lr_scheduler_params={"factor": 0.9},
            learning_rate=0.001,
            gradient_clip_val=0.7,
            batch_size=1,
            embedding_size=256,
            encoder_hidden_layer_sizes=[2048],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[2048],
            decoder_activation_function="ELU",
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
            subgraph_stride=6,
            subgraph_scheduler_name="edge_metrics_based",
            subgraph_scheduler_params={"metrics_treshold": 0.6, "step": 10},
        )
        return parser


class OneBigBarabasiGraphLoader(GraphLoaderBase):
    def load_graphs(self) -> List[nx.Graph]:
        return [nx.barabasi_albert_graph(100, 4)]


if __name__ == "__main__":
    Experiment(
        GraphAutoencoder, SmoothLearningStepGraphDataModule, OneBigBarabasiGraphLoader
    ).run()
