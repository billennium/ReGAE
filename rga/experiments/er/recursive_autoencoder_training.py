from argparse import ArgumentParser

from rga.experiments.experiment import Experiment
from rga.experiments.decorators import add_graphloader_args
from rga.data.paired_data_modules import (
    DiagonalRepresentationGraphPairedDataModule,
    PairedGraphLoader,
)
from rga.models.graph_transformer import RecursiveGraphTransformer
from rga.util.early_stopping import TimeBasedEarlyStopping


class ExperimentModel(RecursiveGraphTransformer):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecursiveGraphTransformer.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCEWithLogits",
            mask_loss_function="BCEWithLogits",
            mask_loss_weight=0.5,
            diagonal_embeddings_loss_weight=0.2,
            recall_to_precision_bias=0.5,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="none",
            lr_scheduler_metric="loss/train_avg",
            learning_rate=0.0005,
            gradient_clip_val=1.0,
            batch_size=64,
            embedding_size=160,
            block_size=4,
            encoder_hidden_layer_sizes=[1024, 768],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[2048],
            decoder_activation_function="ELU",
            metrics=[
                "EdgeAccuracy",
                "EdgePrecision",
                "EdgeRecall",
                "EdgeF1",
                "MaskPrecision",
                "MaskRecall",
                "MeanReconstructionLoss",
                "MeanEmbeddingsLoss",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=3,
            metric_update_interval=3,
            early_stopping=True,
            dataset_name="ER",
            dataset_suffixes=["_20", "_40", "_60"],
            datasets_dir="/home/amalkowski/phd/datasets",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphPairedDataModule):
    graphloader_class = PairedGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
