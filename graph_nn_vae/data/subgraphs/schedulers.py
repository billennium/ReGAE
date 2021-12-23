from typing import Callable, Tuple, Type
from argparse import ArgumentParser

import pytorch_lightning as pl


def get_subgraph_size_scheduler(name: str):
    return {
        "linear": LinearSubgraphSizeScheduler,
        "step": StepSubgraphSizeScheduler,
        "edge_metrics_based": EdgeMetricsBasedSubgraphSizeScheduler,
        "no_graph_scheduler": None,
    }[name]


class SubgraphSizeScheduler:
    _get_current_epoch: Callable

    def __init__(
        self, subgraph_scheduler_params: dict, epoch_num_source=None, **kwargs
    ):
        """
        Args:
            epoch_num_source: Either a pl.Trainer or a function that returns the current epoch number.
            subgraph_scheduler_params: a dict of params specific to a SubgrapghSizeScheduler implementation.
        """
        self.params = subgraph_scheduler_params
        if epoch_num_source is not None:
            self.set_epoch_num_source(epoch_num_source)

    def set_epoch_num_source(self, epoch_num_source):
        if isinstance(epoch_num_source, pl.Trainer):
            self.trainer = epoch_num_source
            self._get_current_epoch = self._get_current_epoch_from_trainer
        elif callable(epoch_num_source):
            self._get_current_epoch = epoch_num_source
        else:
            raise TypeError(
                f"epoch_num_source must be either a pl.Trainer or a Callable; got: {type(epoch_num_source)}"
            )

    def _get_current_epoch_from_trainer(self):
        return self.trainer.current_epoch

    def get_current_subgraph_size(self):
        raise NotImplementedError

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        parser.add_argument(
            "--subgraph_scheduler_params",
            dest="subgraph_scheduler_params",
            default={},
            type=dict,
            help="parameters for selected subgraph size scheduler",
        )
        return parser


class LinearSubgraphSizeScheduler(SubgraphSizeScheduler):
    """
    Scales the graph subgraphs linearly.
    Requires a float `speed` subgraph_scheduler_params param.
    """

    def __init__(self, subgraph_scheduler_params, **kwargs):
        if "speed" not in subgraph_scheduler_params:
            subgraph_scheduler_params["speed"] = 0.01
        super().__init__(subgraph_scheduler_params=subgraph_scheduler_params, **kwargs)

    def get_current_subgraph_size(self) -> float:
        return max(min(float(self.trainer.current_epoch * self.params["speed"]), 1), 0)


class StepSubgraphSizeScheduler(SubgraphSizeScheduler):
    """
    Scales the graph subgraphs by steps.
    Requires a int `step_length` and int `step_size` subgraph_scheduler_params params.
    """

    def __init__(self, subgraph_scheduler_params, **kwargs):
        if "step_length" not in subgraph_scheduler_params:
            subgraph_scheduler_params["step_length"] = 1000
        if "step_size" not in subgraph_scheduler_params:
            subgraph_scheduler_params["step_size"] = 0.05
        super().__init__(subgraph_scheduler_params=subgraph_scheduler_params, **kwargs)

    def get_current_subgraph_size(self):
        return max(
            min(
                float(self.trainer.current_epoch / self.params["step_length"])
                * self.params["step_size"],
                1.0,
            ),
            0,
        )


class EdgeMetricsBasedSubgraphSizeScheduler(SubgraphSizeScheduler):
    """
    Scales the graph subgraphs based on edge metrics
    Requires a int `step` and float `metrics_treshold` subgraph_scheduler_params params.
    """

    def __init__(
        self, subgraph_scheduler_params, data_module, metric_update_interval, **kwargs
    ):
        if "step" not in subgraph_scheduler_params:
            subgraph_scheduler_params["step"] = 0.05
        if "metrics_treshold" not in subgraph_scheduler_params:
            subgraph_scheduler_params["metrics_treshold"] = 0.5
        if "subgraph_size_initial" not in subgraph_scheduler_params:
            subgraph_scheduler_params["subgraph_size_initial"] = 0.2
        super().__init__(subgraph_scheduler_params=subgraph_scheduler_params, **kwargs)
        self.data_module = data_module
        self.size = subgraph_scheduler_params["subgraph_size_initial"]
        self.metric_update_interval = metric_update_interval
        self.last_epoch_changed = -metric_update_interval - 1

    def get_current_subgraph_size(self) -> float:
        if self.size >= 1:
            return self.size

        if (
            self.last_epoch_changed + self.metric_update_interval
            < self.data_module.trainer.current_epoch
            and (
                self.data_module.current_metrics.get("edge_recall/train_avg", 0)
                > self.params["metrics_treshold"]
            )
            and (
                self.data_module.current_metrics.get("edge_precision/train_avg", 0)
                > self.params["metrics_treshold"]
            )
        ):
            self.size = self.size + self.params["step"]
            self.last_epoch_changed = self.data_module.trainer.current_epoch

        return max(min(self.size, 1), 0)
