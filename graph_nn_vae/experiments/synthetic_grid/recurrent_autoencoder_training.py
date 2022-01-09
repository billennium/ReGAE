from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.experiments.decorators import add_graphloader_args
from graph_nn_vae.data import (
    DiagonalRepresentationGraphDataModule,
    SyntheticGraphLoader,
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
            # lr_scheduler_params={"factor": 0.85},
            learning_rate=0.0005,
            gradient_clip_val=1.0,
            batch_size=64,
            embedding_size=512,
            block_size=15,
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
            num_dataset_graph_permutations=10,
            minimal_subgraph_size=10,
            subgraph_stride=0.5,
            subgraph_scheduler_name="none",
            # subgraph_scheduler_params={
            #     "subgraph_size_initial": 0.025,
            #     "metrics_treshold": 0.6,
            #     "step": 0.025,
            # },
            graph_type="grid",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = SyntheticGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
