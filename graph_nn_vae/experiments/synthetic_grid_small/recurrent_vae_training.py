from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.experiments.decorators import add_graphloader_args
from graph_nn_vae.data import (
    DiagonalRepresentationGraphDataModule,
    SyntheticGraphLoader,
)
from graph_nn_vae.models.vae import RecurrentGraphVAE


class ExperimentModel(RecurrentGraphVAE):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecurrentGraphVAE.add_model_specific_args(parent_parser)
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
            num_dataset_graph_permutations=10,
            graph_type="grid_small",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = SyntheticGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
