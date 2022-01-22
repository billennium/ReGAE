from argparse import ArgumentParser

from rga.experiments.experiment import Experiment
from rga.experiments.decorators import add_graphloader_args
from rga.data import (
    DiagonalRepresentationGraphDataModule,
    RealGraphLoader,
)
from rga.models.autoencoder_base import RecursiveGraphAutoencoder


class ExperimentModel(RecursiveGraphAutoencoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecursiveGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCEWithLogits",
            mask_loss_function="BCEWithLogits",
            mask_loss_weight=0.5,
            diagonal_embeddings_loss_weight=0.2,
            recall_to_precision_bias=0.03,
            gradient_clip_val=0.5,
            optimizer="AdamWAMSGrad",
            learning_rate=0.0003,
            lr_monitor=True,
            lr_scheduler_name="FactorDecreasingOnMetricChange",
            lr_scheduler_metric="max_graph_size/train_avg",
            lr_scheduler_params={"factor": 0.9},
            minimal_subgraph_size=4,
            subgraph_stride=0.5,
            subgraph_scheduler_name="edge_metrics_based",
            subgraph_scheduler_params={
                "subgraph_size_initial": 0.05,
                "metrics_treshold": 0.1,
                "step": 0.1,
            },
            batch_size=2,
            accumulate_grad_batches=16,
            embedding_size=1564,
            block_size=64,
            encoder_hidden_layer_sizes=[4096],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[6144],
            decoder_activation_function="ELU",
            metrics=[
                "EdgeAccuracy",
                "EdgePrecision",
                "EdgeRecall",
                "EdgeF1",
                "MaskPrecision",
                "MaskRecall",
                "MaxGraphSize",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=1,
            metric_update_interval=3,
            early_stopping=False,
            bfs=True,
            enable_checkpointing=False,
            num_dataset_graph_permutations=1,
            dataset_name="REDDIT-MULTI-12K",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = RealGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
