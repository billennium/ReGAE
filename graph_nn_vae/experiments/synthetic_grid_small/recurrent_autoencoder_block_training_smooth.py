from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import (
    DiagonalBlockRepresentationGraphDataModule,
    SyntheticGraphLoader,
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
            # lr_scheduler_name="NoSched",
            # lr_scheduler_metric="loss/train_avg",
            lr_scheduler_name="FactorDecreasingOnMetricChange",
            lr_scheduler_metric="max_graph_size/train_avg",
            lr_scheduler_params={"factor": 0.9},
            learning_rate=0.001,
            gradient_clip_val=1.0,
            batch_size=32,
            embedding_size=128,
            encoder_hidden_layer_sizes=[1024],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[1024],
            decoder_activation_function="ELU",
            block_size=3,
            metrics=[
                "EdgePrecision",
                "EdgeRecall",
                "MaskPrecision",
                "MaskRecall",
                "MaxGraphSize",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=1,
            metric_update_interval=1,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=1,
            graph_type="grid_small",
            minimal_subgraph_size=3,
            subgraph_stride=0.5,
            subgraph_scheduler_name="edge_metrics_based",
            subgraph_scheduler_params={
                "subgraph_size_initial": 0.005,
                "metrics_treshold": 0.6,
                "step": 0.25,
            },
        )
        return parser


if __name__ == "__main__":
    Experiment(
        GraphAutoencoder,
        DiagonalBlockRepresentationGraphDataModule,
        SyntheticGraphLoader,
    ).run()
