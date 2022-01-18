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
            gradient_clip_val=1.0,
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
            embedding_size=1720,
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
            dataset_name="REDDIT-BINARY",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = RealGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()

# Statistic of set:  Full original dataset
#              Dataset size : 2000
#                    Labels : False
#            Min node count : 6
#        Average node count : 429.63
#            Max node count : 3782
#            Min edge count : 4.0
#        Average edge count : 497.75
#            Max edge count : 4071.0
#      Min filling fraction : 0.0
#  Average filling fraction : 0.02
#      Max filling fraction : 0.29

# dkrga guild -H ./guild_reddit run recursive_autoencoder:reddit_binary --force-flags \
#     gpus=",2" \
#     progress_bar_refresh_rate=2 \
#     datasets_dir="/rga/datasets" \
#     max_time='00:03:00:00' \
#     early_stopping="True" \
#     es_patience=5 \
#     decoder_hidden_layer_sizes="6144" \
#     lr='[0.0002,0.0003]' \
#     block_size='[64]'
