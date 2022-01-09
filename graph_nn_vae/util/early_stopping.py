from argparse import ArgumentParser
from typing import Tuple
import time
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from graph_nn_vae import util


class EarlyStoppingBase(EarlyStopping):
    """
    Note that the patience number reflects the patience in regard to number of callback calls.
    If `check_val_every_n_epoch` is higher than 1, the number of epochs of the patience will be
    `check_val_every_n_epoch` * <patience_arg>.
    """

    def __init__(
        self,
        es_metric: str,
        es_patience: int,
        steps_per_epoch: int = None,
        min_delta: float = 0.0,
        verbose: bool = False,
        es_mode: str = "min",
        strict: bool = True,
        **kwargs,
    ):
        es_patience = self._scale_patience_to_num_steps(es_patience, steps_per_epoch)
        super().__init__(
            es_metric,
            min_delta,
            es_patience,
            verbose,
            es_mode,
            strict,
            check_finite=True,
        )

    def _scale_patience_to_num_steps(self, patience: int, steps_per_epoch: int) -> int:
        if steps_per_epoch is None:
            return patience
        mult = util.divide_int_round_up(100, steps_per_epoch)
        return patience * mult

    def _run_early_stopping_check(self, trainer: pl.Trainer) -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run
            or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                logs
            )
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        should_stop, reason = self._evaluate_custom_stopping_criteria(
            trainer, should_stop, reason
        )

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)

    def _evaluate_custom_stopping_criteria(
        self, trainer, should_stop: bool, reason: str
    ) -> Tuple[bool, str]:
        # override to define additional behaviour
        return should_stop, reason

    @classmethod
    def add_callback_specific_args(cls, parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--es_metric",
            dest="es_metric",
            default="loss/val",
            type=str,
            help="""Target metric of the early stopping.""",
        )
        parser.add_argument(
            "--es_mode",
            dest="es_mode",
            default="min",
            type=str,
            help="""one of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when
            the quantity monitored has stopped decreasing and in ``'max'`` mode it will stop
            when the quantity monitored has stopped increasing.""",
        )
        parser.add_argument(
            "--es_patience",
            dest="es_patience",
            default=10,
            type=int,
            metavar="PATIENCE",
            help="patience for the early stopping in epochs (*check_val_every_n_epoch)",
        )
        return parent_parser


class ProgressiveSubgraphTrainingEarlyStopping(EarlyStoppingBase):
    def __init__(
        self,
        preogressive_subgraph_training_enabled: bool,
        es_graph_size_metric: str = "max_graph_size/train_avg",
        es_graph_size_change_patience: int = 10,
        steps_per_epoch: int = None,
        **kwargs,
    ):
        super().__init__(
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )
        es_graph_size_change_patience = self._scale_patience_to_num_steps(
            es_graph_size_change_patience, steps_per_epoch
        )
        self.curr_biggest_graph = 0
        self.same_graph_size_wait_count = 0
        self.graph_size_change_patience = es_graph_size_change_patience
        self.preogressive_subgraph_training_enabled = (
            preogressive_subgraph_training_enabled
        )
        self.monitored_graph_size_metric = es_graph_size_metric

    def _evaluate_custom_stopping_criteria(
        self, trainer, should_stop: bool, reason: str
    ) -> Tuple[bool, str]:
        logs = trainer.callback_metrics
        if self.preogressive_subgraph_training_enabled:
            new_biggest_graph = logs[self.monitored_graph_size_metric].squeeze()
            if self.curr_biggest_graph == new_biggest_graph:
                self.same_graph_size_wait_count += 1
            else:
                self.curr_biggest_graph = new_biggest_graph
                self.same_graph_size_wait_count = 0
                torch_inf = torch.tensor(np.Inf)
                self.best_score = (
                    torch_inf if self.monitor_op == torch.lt else -torch_inf
                )
                self.wait_count = 0

            if (
                should_stop
                and self.same_graph_size_wait_count < self.graph_size_change_patience
            ):
                should_stop = False

        return should_stop, reason

    @classmethod
    def add_callback_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = EarlyStoppingBase.add_callback_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--es_graph_size_change_patience",
            dest="es_graph_size_change_patience",
            default=10,
            type=int,
            metavar="PATIENCE",
            help="patience for the early stopping for the graph size metric, if used, in epochs (*check_val_every_n_epoch)",
        )
        return parent_parser


class TimeBasedEarlyStopping(EarlyStoppingBase):
    """
    Stops when the specified time elapses, in addition to when the specified metric stops improving.
    Starts the counter at class instance init.
    """

    def __init__(self, es_time: int, **kwargs):
        super().__init__(**kwargs)
        self.time_seconds = es_time
        self.start_time = time.time()

    def _evaluate_custom_stopping_criteria(
        self, trainer, should_stop: bool, reason: str
    ) -> Tuple[bool, str]:
        elapsed_seconds = time.time() - self.start_time
        if elapsed_seconds > self.time_seconds:
            return True, (
                f"Elapsed time has crossed the specified max run time treshold of {self.time_seconds} seconds."
            )

        return should_stop, reason

    @classmethod
    def add_callback_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = EarlyStoppingBase.add_callback_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--es_time",
            dest="es_time",
            default=3600,
            type=int,
            metavar="NUM_SECONDS",
            help="Time, in seconds, after which the early stopping will trigger.",
        )
        return parent_parser
