import time
import argparse
import random
import warnings
from typing import Type

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from rga.data import BaseDataModule, data_module
from rga.models.base import BaseModel
from rga import util
from rga.util.early_stopping import (
    EarlyStoppingBase,
    ProgressiveSubgraphTrainingEarlyStopping,
)


warnings.filterwarnings(
    "ignore",
    message=r"The dataloader, (.*?) does not have many workers",
    category=UserWarning,
)
warnings.filterwarnings(
    # This warning may occur when running sanity validation checks. The cause is a val metric that
    # is getting registered for display in the status bar during the sanity checks,
    # but is not yet initialized during a proper validation step.
    "ignore",
    message=r"The ``compute`` method of metric (.*?) was called before the ``update``",
    category=UserWarning,
)


class Experiment:
    def __init__(
        self,
        model: Type[BaseModel],
        data_module: BaseDataModule,
        early_stopping: EarlyStoppingBase = ProgressiveSubgraphTrainingEarlyStopping,
        parser_default: dict = None,
    ):
        self.model = model
        self.data_module = data_module
        self.early_stopping = early_stopping
        self.parser_default = parser_default if parser_default is not None else {}

    def run(self):
        parser = self.create_parser()
        args = parser.parse_args()

        if args.seed is not None:
            pl.seed_everything(args.seed)

        torch.multiprocessing.set_sharing_strategy("file_system")

        if args.batch_size_val is None:
            args.batch_size_val = args.batch_size
        if args.batch_size_test is None:
            args.batch_size_test = args.batch_size

        self.data_module: BaseDataModule = self.data_module(**vars(args))

        if args.load_from_checkpoint_path is not None:
            model = self.model.load_from_checkpoint(
                checkpoint_path=args.load_from_checkpoint_path,
                hparams_file=args.load_with_hparams_path,
                **vars(args),
                data_module=self.data_module,
                num_train_dataloaders=self.data_module.num_train_dataloaders(),
                num_val_dataloaders=self.data_module.num_val_dataloaders(),
                num_test_dataloaders=self.data_module.num_test_dataloaders(),
            )
        else:
            model = self.model(
                **vars(args),
                data_module=self.data_module,
                num_train_dataloaders=self.data_module.num_train_dataloaders(),
                num_val_dataloaders=self.data_module.num_val_dataloaders(),
                num_test_dataloaders=self.data_module.num_test_dataloaders(),
            )

        logger = self.create_logger(logger_name=args.logger_name)
        trainer = pl.Trainer.from_argparse_args(args, logger=logger)

        if args.checkpoint_monitor:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor=args.checkpoint_monitor,
                save_top_k=args.checkpoint_top_k,
                mode=args.checkpoint_mode,
            )
            trainer.callbacks.append(checkpoint_callback)

        if args.lr_monitor:
            lr_monitor = LearningRateMonitor(logging_interval="step")
            trainer.callbacks.append(lr_monitor)

        steps_per_epoch = util.divide_int_round_up(
            len(self.data_module.train_dataset), self.data_module.batch_size
        )

        if args.early_stopping:
            preogressive_subgraph_training_enabled = False
            if (
                "subgraph_scheduler_name" in args
                and args.subgraph_scheduler_name not in (None, "none")
            ):
                preogressive_subgraph_training_enabled = True
            early_stopping = self.early_stopping(
                steps_per_epoch=steps_per_epoch,
                preogressive_subgraph_training_enabled=preogressive_subgraph_training_enabled,
                **vars(args),
            )
            trainer.callbacks.append(early_stopping)

        args.train_dataset_length = len(self.data_module.train_dataset)
        args.val_dataset_length = [len(d) for d in self.data_module.val_datasets]
        args.test_dataset_length = [len(d) for d in self.data_module.test_datasets]

        arg_dict = {
            k: v for (k, v) in vars(args).items() if not callable(v) and v is not None
        }
        trainer.logger.log_hyperparams(argparse.Namespace(**arg_dict))

        start = time.time()
        trainer.fit(model, datamodule=self.data_module)
        end = time.time()

        if not args.no_evaluate:
            if args.checkpoint_monitor:
                if checkpoint_callback.best_model_path == "":
                    print(f"No checkpoints saved; skipping test")
                else:
                    trainer.test(
                        ckpt_path=checkpoint_callback.best_model_path,
                        dataloaders=self.data_module,
                    )
            else:
                trainer.test(ckpt_path="best", dataloaders=self.data_module)

        if args.checkpoint_monitor:
            print(f"Best model path: {checkpoint_callback.best_model_path}")

        print("Elapsed time:", "%.2f" % (end - start))

    def create_logger(self, logger_name: str = "tb") -> pl.loggers.LightningLoggerBase:
        if logger_name == "tb":
            return pl.loggers.TensorBoardLogger(
                save_dir="tb_logs/" + self.model.model_name,
                name=self.data_module.data_name,
            )
        else:
            raise RuntimeError(f"unknown logger name: {logger_name}")

    def create_parser(self):
        parser = argparse.ArgumentParser(add_help=True)
        parser = self.add_trainer_parser(parser)
        parser = self.add_experiment_parser(parser)
        parser = self.data_module.add_model_specific_args(parser)
        parser = self.early_stopping.add_callback_specific_args(parser)
        parser = self.model.add_model_specific_args(parser)
        parser.set_defaults(
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

    @classmethod
    def add_experiment_parser(cls, parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--no_evaluate",
            dest="no_evaluate",
            type=bool,
            default=False,
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
            "--logger_name",
            dest="logger_name",
            type=str,
            choices=["tb", "mlf"],
            default="tb",
            help="Logger name.",
        )
        parser.add_argument(
            "--checkpoint_monitor",
            dest="checkpoint_monitor",
            type=str,
            default="loss/val",
            help="Metric used for checkpointing",
        )
        parser.add_argument(
            "--checkpoint_top_k",
            dest="checkpoint_top_k",
            type=int,
            default=1,
            help="Save top k models",
        )
        parser.add_argument(
            "--checkpoint_mode",
            dest="checkpoint_mode",
            type=str,
            choices=["min", "max"],
            default="min",
            help="Mode for the checkpoint monitoring",
        )
        parser.add_argument(
            "--early_stopping",
            dest="early_stopping",
            type=bool,
            default=False,
            help="Enable early stopping",
        )
        parser.add_argument(
            "--lr_monitor",
            dest="lr_monitor",
            type=bool,
            default=False,
            help="Enable learning rate monitor",
        )
        parser.add_argument(
            "--load_from_checkpoint_path",
            dest="load_from_checkpoint_path",
            type=str,
            default=None,
            help="Load and train the model from the specified checkpoint instead of starting from scratch",
        )
        parser.add_argument(
            "--load_with_hparams_path",
            dest="load_with_hparams_path",
            type=str,
            default=None,
            help="If loading from checkpoint, optionally specify the hparams file",
        )
        return parent_parser
