import time
import argparse
import random
import warnings
from typing import Type

import torch
import pytorch_lightning as pl

from graph_nn_vae.data import BaseDataModule
from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.autoencoder_components import GraphEncoder

warnings.filterwarnings(
    "ignore",
    message=r"The dataloader, (.*?) does not have many workers",
    category=UserWarning,
)


class Experiment:
    def __init__(
        self,
        model: Type[BaseModel],
        data_module: BaseDataModule,
        parser_default: dict = None,
    ):
        self.model = model
        self.data_module = data_module
        # self.early_stopping = ThresholdedEarlyStopping
        self.parser_default = parser_default if parser_default is not None else {}

    def run(self):
        parser = self.create_parser()
        args = parser.parse_args()

        if args.seed is not None:
            pl.seed_everything(args.seed)

        if args.fast_dev_run:
            args.batch_size_val = args.batch_size
            args.batch_size_test = args.batch_size

        data_module = self.data_module(**vars(args))
        model = self.model(
            **vars(args),
            loss_weight=data_module.loss_weight(),
        )
        # early_stopping = self.early_stopping(**vars(args))

        logger = self.create_logger(logger_name=args.logger_name)
        trainer = pl.Trainer.from_argparse_args(args, logger=logger)
        if args.checkpoint_monitor:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor=args.checkpoint_monitor,
                save_top_k=args.checkpoint_top_k,
                mode=args.checkpoint_mode,
            )
            trainer.callbacks.append(checkpoint_callback)

        # if args.thresholded_early_stopping:
        #     trainer.callbacks.append(early_stopping)
        trainer.logger.log_hyperparams(args)

        start = time.time()
        trainer.fit(model, datamodule=data_module)
        end = time.time()

        if not args.no_evaluate:
            if args.checkpoint_monitor:
                trainer.test(ckpt_path=checkpoint_callback.best_model_path)
            else:
                trainer.test(ckpt_path="best")

        print("Elapsed time:", "%.2f" % (end - start))

    def create_logger(self, logger_name: str = "tb") -> pl.loggers.LightningLoggerBase:
        if logger_name == "tb":
            return pl.loggers.TensorBoardLogger(
                save_dir="tb_logs",
                name=self.data_module.data_name,
            )
        else:
            raise RuntimeError(f"unknown logger name: {logger_name}")

    def create_parser(self):
        parser = argparse.ArgumentParser(add_help=True)
        parser = self.add_trainer_parser(parser)
        parser = self.add_experiment_parser(parser)
        parser = self.data_module.add_model_specific_args(parser)
        parser = self.model.add_model_specific_args(parser)
        # parser = self.early_stopping.add_callback_specific_args(parser)
        parser.set_defaults(
            progress_bar_refresh_rate=2,
            **self.parser_default,
        )
        parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        return parser

    def add_trainer_parser(self, parser: argparse.ArgumentParser):
        parser = pl.Trainer.add_argparse_args(parser)
        parser.set_defaults(
            deterministic=True,
            max_epochs=100,
        )
        return parser

    def add_experiment_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--no-evaluate",
            dest="no_evaluate",
            action="store_true",
            help="do not evaluate model on validation set",
        )
        parser.add_argument(
            "--seed",
            dest="seed",
            type=int,
            default=random.randrange(1 << 32 - 1),
            help="seed for model training.",
        )
        parser.add_argument(
            "--logger-name",
            dest="logger_name",
            type=str,
            choices=["tb", "mlf"],
            default="tb",
            help="Logger name.",
        )
        parser.add_argument(
            "--checkpoint-monitor",
            dest="checkpoint_monitor",
            type=str,
            default="",
            help="Metric used for checkpointing",
        )
        parser.add_argument(
            "--checkpoint-top-k",
            dest="checkpoint_top_k",
            type=int,
            default=1,
            help="Save top k models",
        )
        parser.add_argument(
            "--checkpoint-mode",
            dest="checkpoint_mode",
            type=str,
            choices=["min", "max"],
            default="min",
            help="Mode for the checkpoint monitoring",
        )
        return parser
