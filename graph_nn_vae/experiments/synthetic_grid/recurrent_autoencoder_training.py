from argparse import ArgumentParser
from typing import List
import networkx as nx

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import (
    GraphLoaderBase,
    SyntheticGraphLoader,
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
            lr_scheduler_params={"factor": 0.85},
            learning_rate=0.0005,
            gradient_clip_val=0.7,
            batch_size=4,
            embedding_size=256,
            encoder_hidden_layer_sizes=[1024, 768],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[768, 1024],
            decoder_activation_function="ELU",
            metrics=[
                "EdgePrecision",
                "EdgeRecall",
                "MaskPrecision",
                "MaskRecall",
                "MaxGraphSize",
            ],
            max_epochs=100000,
            check_val_every_n_epoch=5,
            metric_update_interval=1,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=1,
            minimal_subgraph_size=10,
            subgraph_stride=0.5,
            subgraph_scheduler_name="edge_metrics_based",
            subgraph_scheduler_params={
                "subgraph_size_initial": 0.025,
                "metrics_treshold": 0.6,
                "step": 0.025,
            },
            workers=0,
            # gpus=1,
            # precision=16,
            graph_type="grid",
        )
        return parser


if __name__ == "__main__":
    Experiment(
        GraphAutoencoder, SmoothLearningStepGraphDataModule, SyntheticGraphLoader
    ).run()
