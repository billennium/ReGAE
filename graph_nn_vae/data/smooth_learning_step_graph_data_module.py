from typing import Callable, Tuple, Type
from operator import itemgetter
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from torch.functional import Tensor

from graph_nn_vae.data.synthetic_graphs_module import SyntheticGraphsDataModule


class SmoothLearningStepGraphDataModule(SyntheticGraphsDataModule):
    data_name = "subgraphs"
    max_num_nodes_in_train_dataset: int
    is_scheduling_initialized = False

    def __init__(
        self,
        subgraph_scheduler_name: str,
        subgraph_depth: int = 10,
        subgraph_depth_step: int = 1,
        subgraph_stride: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        kwargs["data_module"] = self

        self.subgraph_size_scheduler = self.get_subgraph_size_scheduler(
            subgraph_scheduler_name
        )(**kwargs)
        self.collate_fn_train = self.collate_graph_batch_training
        self.depth = subgraph_depth
        self.stride = subgraph_stride
        self.depth_step = subgraph_depth_step
        self.current_metrics = {}

    def get_subgraph_size_scheduler(self, name: str):
        return {
            "linear": LinearSubgrapghSizeScheduler,
            "step": StepSubgrapghSizeScheduler,
            "edge_metrics_based": EdgeMetricsBasedSubgrapghSizeScheduler,
        }[name]

    def train_dataloader(self, **kwargs):
        if not self.is_scheduling_initialized:
            self.subgraph_size_scheduler.set_epoch_num_source(self.trainer)
            max_num_nodes_in_train = max(self.train_dataset, key=itemgetter(2))[2]
            self.subgraph_size_monitor = SteppingGraphSizeMonitor(
                self.subgraph_size_scheduler.get_current_subgraph_size,
                max_num_nodes_in_train,
            )
            self.trainer.callbacks.append(self.subgraph_size_monitor)

            self.current_metric_monitor = MetricMonitor(self)
            self.trainer.callbacks.append(self.current_metric_monitor)

            self.is_logging_initialized = True
        return super().train_dataloader(**kwargs)

    def collate_graph_batch_training(self, batch):
        # As part of the collation graph diag_repr are padded with 0.0 and the graph masks
        # are padded with 1.0 to represent the end of the graphs.
        # print(step.TRAINING_EPOCH)

        graphs = [g[0] for g in batch]
        graph_masks = [g[1] for g in batch]
        num_nodes = [g[2] for g in batch]

        scheduled_subgraph_size = (
            self.subgraph_size_scheduler.get_current_subgraph_size()
        )
        max_num_nodes_in_batch = max(num_nodes)
        if scheduled_subgraph_size < max_num_nodes_in_batch:
            target_subgraph_size = min(scheduled_subgraph_size, max_num_nodes_in_batch)
            graphs, graph_masks, num_nodes = self.generate_subgraphs_for_batch(
                graphs, num_nodes, target_subgraph_size
            )

        graphs = torch.nn.utils.rnn.pad_sequence(
            graphs, batch_first=True, padding_value=0.0
        )
        graph_masks = torch.nn.utils.rnn.pad_sequence(
            graph_masks, batch_first=True, padding_value=0.0
        )
        num_nodes = torch.tensor(num_nodes)

        return graphs, graph_masks, num_nodes

    def generate_subgraphs_for_batch(
        self, graphs: Tensor, num_nodes: Tensor, target_max_subgraph_size: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        splitted_graphs = []
        splitted_graph_masks = []
        splitted_graphs_sizes = []

        for graph, graph_size in zip(graphs, num_nodes):
            for subgraph_size_offset in (
                range(0, min(target_max_subgraph_size - 1, self.depth), self.depth_step)
                if target_max_subgraph_size < graph_size
                else [0]
            ):
                subgrpahs, subgraph_masks, subgraph_sizes = self.generate_subgraphs(
                    graph,
                    graph_size,
                    new_size=target_max_subgraph_size - subgraph_size_offset,
                    stride=self.stride,
                    probability=1.0 / (subgraph_size_offset + 1),
                )
                splitted_graphs.extend(subgrpahs)
                splitted_graph_masks.extend(subgraph_masks)
                splitted_graphs_sizes.extend(subgraph_sizes)

        return splitted_graphs, splitted_graph_masks, splitted_graphs_sizes

    def generate_subgraphs(
        self,
        graph,
        graph_size: int,
        new_size: int,
        stride: int = 1,
        probability: float = 1.0,
    ):
        if new_size > graph_size:
            return [graph], [torch.ones(graph.shape)], [graph_size]

        candidates = torch.arange(0, graph_size - new_size + 1, stride).int()

        if probability < 1:
            candidates = candidates[torch.rand(len(candidates)) < probability]

        if len(candidates) == 0:
            return ([], [], [])

        graph_diagonals = []
        index = 0
        for diag_len in range(1, graph_size):
            if diag_len > graph_size - new_size:
                graph_diagonals.append(graph[index : index + diag_len])
            index = index + diag_len

        subgraphs = []
        subgraphs_masks = []
        graph_sizes = []

        for k in candidates:
            reduced_graph = torch.cat(
                [graph_diagonals[i][k : k + i + 1] for i in range(new_size - 1)]
            )
            subgraphs.append(reduced_graph)
            subgraphs_masks.append(torch.ones(reduced_graph.shape))
            graph_sizes.append(new_size)

        return (subgraphs, subgraphs_masks, graph_sizes)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = SyntheticGraphsDataModule.add_model_specific_args(parent_parser)
        parser = SubgrapghSizeScheduler.add_model_specific_args(parser)
        parser.add_argument(
            "--subgraph_scheduler_name",
            dest="subgraph_scheduler_name",
            default="linear",
            type=str,
            help="name of maximum subgraph size scheduler",
        )
        parser.add_argument(
            "--subgraph_depth",
            dest="subgraph_depth",
            default=10,
            type=int,
            help="depth of looking from maximal subgraph size to smaller subgraphs",
        )
        parser.add_argument(
            "--subgraph_depth_step",
            dest="subgraph_depth_step",
            default=1,
            type=int,
            help="step of looking from maximal subgraph size to smaller subgraphs",
        )
        parser.add_argument(
            "--subgraph_stride",
            dest="subgraph_stride",
            default=1,
            type=int,
            help="stride between subgraphs",
        )
        return parser


class SubgrapghSizeScheduler:
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


class LinearSubgrapghSizeScheduler(SubgrapghSizeScheduler):
    """
    Scales the graph subgraphs linearly.
    Requires a float `speed` subgraph_scheduler_params param.
    """

    def __init__(self, subgraph_scheduler_params, **kwargs):
        if "speed" not in subgraph_scheduler_params:
            subgraph_scheduler_params["speed"] = 1.0
        super().__init__(subgraph_scheduler_params=subgraph_scheduler_params, **kwargs)

    def get_current_subgraph_size(self):
        return int(self.trainer.current_epoch * self.params["speed"]) + 2


class StepSubgrapghSizeScheduler(SubgrapghSizeScheduler):
    """
    Scales the graph subgraphs by steps.
    Requires a int `step_length` and int `step_size` subgraph_scheduler_params params.
    """

    def __init__(self, subgraph_scheduler_params, **kwargs):
        if "step_length" not in subgraph_scheduler_params:
            subgraph_scheduler_params["step_length"] = 1000
        if "step_size" not in subgraph_scheduler_params:
            subgraph_scheduler_params["step_size"] = 5
        super().__init__(subgraph_scheduler_params=subgraph_scheduler_params, **kwargs)

    def get_current_subgraph_size(self):
        return (
            int(self.trainer.current_epoch / self.params["step_length"])
            * self.params["step_size"]
            + 5
        )


class EdgeMetricsBasedSubgrapghSizeScheduler(SubgrapghSizeScheduler):
    """
    Scales the graph subgraphs based on edge metrics
    Requires a int `step` and float `metrics_treshold` subgraph_scheduler_params params.
    """

    def __init__(
        self, subgraph_scheduler_params, data_module, metric_update_interval, **kwargs
    ):
        if "step" not in subgraph_scheduler_params:
            subgraph_scheduler_params["step"] = 5
        if "metrics_treshold" not in subgraph_scheduler_params:
            subgraph_scheduler_params["metrics_treshold"] = 0.5
        super().__init__(subgraph_scheduler_params=subgraph_scheduler_params, **kwargs)
        self.data_module = data_module
        self.size = 2
        self.metric_update_interval = metric_update_interval
        self.last_epoch_changed = -metric_update_interval - 1

    def get_current_subgraph_size(self):
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

        return self.size


# callback_metrics


class SteppingGraphSizeMonitor(Callback):
    def __init__(
        self, get_current_subgraph_size_fn: Callable, max_num_nodes_in_dataset: int = 0
    ):
        self._get_current_subgraph_size_fn = get_current_subgraph_size_fn
        self._max_num_nodes_in_dataset = max_num_nodes_in_dataset

    def set_max_num_nodes_in_dataset(self, num_nodes: int) -> None:
        self._max_num_nodes_in_dataset = num_nodes

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        scheduled_max_size = self._get_current_subgraph_size_fn()
        curr_max_num_nodes_in_subgraph = min(
            scheduled_max_size, self._max_num_nodes_in_dataset
        )
        trainer.logger.log_metrics(
            {"max_subgraph_size": curr_max_num_nodes_in_subgraph},
            step=trainer.global_step,
        )

    @staticmethod
    def _should_log(trainer) -> bool:
        # should_log = ((trainer.global_step + 1) % trainer.log_every_n_steps == 0 or trainer.should_stop)
        return True


class MetricMonitor(Callback):
    def __init__(
        self,
        data_module: SmoothLearningStepGraphDataModule,
    ):
        self.data_module = data_module

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        self.data_module.current_metrics = {
            k: v.cpu().numpy() for (k, v) in trainer.callback_metrics.items()
        }
