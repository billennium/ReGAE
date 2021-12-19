from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import (
    DiagonalRepresentationGraphDataModule,
    SmoothLearningStepGraphDataModule,
    SyntheticGraphLoader,
)
from graph_nn_vae.models.autoencoder_base import RecurrentGraphAutoencoder
from graph_nn_vae.models.vae import RecurrentGraphVAE
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


class GraphAutoencoder(RecurrentGraphVAE):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        # If using lr_schedulers that base their calculations on steps/epochs remember
        # that the scheduling occurs at the frequency of the `check_val_every_n_epoch`
        # interval. Thus, their calculations are skewed if it's higher than 1.
        #
        # To fix the intervals recalculate the values like this (MultiStepLR example):
        # val_and_lr_update_interval = 20
        # lr_milestones = [400, 800, 1200]
        # lr_milestones = [v / val_and_lr_update_interval for v in lr_milestones]

        RecurrentGraphAutoencoder.graph_decoder_class = GraphDecoder
        RecurrentGraphAutoencoder.edge_decoder_class = MemoryEdgeDecoder

        parser = RecurrentGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCEWithLogits",
            mask_loss_function="BCEWithLogits",
            mask_loss_weight=0.5,
            diagonal_embeddings_loss_weight=0.2,
            kld_loss_weight=0.00001,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            # lr_scheduler_name="NoSched",
            lr_scheduler_name="SingleTimeChangeOnMetricTreshold",
            lr_scheduler_params={"treshold": 0.8, "lr_change": 0.0003},
            lr_scheduler_metric="edge_recall/train_avg",
            learning_rate=0.001,
            gradient_clip_val=1.0,
            batch_size=32,
            embedding_size=128,
            encoder_hidden_layer_sizes=[1024],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[1024],
            decoder_activation_function="ELU",
            metrics=[
                "EdgePrecision",
                "EdgeRecall",
                "MaskPrecision",
                "MaskRecall",
                "MeanReconstructionLoss",
                "MeanEmbeddingsLoss",
                "MeanKLDLoss",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=20,
            metric_update_interval=20,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=10,
            graph_type="grid_small",
        )
        return parser


if __name__ == "__main__":
    Experiment(
        GraphAutoencoder, DiagonalRepresentationGraphDataModule, SyntheticGraphLoader
    ).run()
