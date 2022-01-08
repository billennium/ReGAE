from typing import List
import torch
import torchmetrics

from graph_nn_vae.util.adjmatrix.diagonal_block_representation import (
    calculate_num_blocks,
)
from graph_nn_vae.models.utils.calc import torch_bincount


class EdgeMetric(torchmetrics.Metric):
    label = "edge_metric"
    metric_class = None
    weight_power = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric = self.metric_class(num_classes=2, average=None, **kwargs)

        self.add_state("metrics", default=[], dist_reduce_fx="cat")
        self.add_state("weights", default=[], dist_reduce_fx="cat")
        self.metric_type_label = self.get_generic_metric_label()

    def get_generic_metric_label(self) -> str:
        """
        Returns a label with just the type of metric.
        ex. edge_torchmetrics.Precision
        """
        return "edge_" + self.metric_class.__name__

    def update(
        self,
        edges_predicted: torch.Tensor,
        edges_target: torch.Tensor,
        num_nodes: torch.Tensor,
        shared_metric_state: dict = None,
        **kwargs,
    ):
        if (
            isinstance(shared_metric_state, dict)
            and self.metric_type_label in shared_metric_state
        ):
            self.metric.reset()
            self.metrics = shared_metric_state[self.metric_type_label]
            self.calc_weights_only(edges_predicted, num_nodes)
            shared_metric_state[self.label + "_weights"] = self.weights
            return

        edges_predicted = torch.sigmoid(edges_predicted).round().int()
        edges_target = torch.clamp(edges_target.int(), min=0)

        block_size = edges_predicted.shape[2] if len(edges_predicted.shape) == 5 else 1
        if block_size != 1:
            num_blocks = calculate_num_blocks(num_nodes, block_size)
        else:
            num_blocks = num_nodes

        graph_counts_per_size = torch_bincount(num_blocks)

        for index, count in enumerate(graph_counts_per_size):
            if count != 0:
                mask = num_blocks == index
                predicted = edges_predicted[mask].flatten()
                target = edges_target[mask].flatten()

                current_metric = self.metric(predicted, target)[1]
                self.metric.reset()
                if current_metric != current_metric:
                    current_metric = torch.zeros(1, device=predicted.device)[0]

                self.metrics.append(current_metric)

                self.weights.append(
                    pow(index * block_size, 2 - self.weight_power) * count
                )

        if isinstance(shared_metric_state, dict):
            shared_metric_state[self.metric_type_label] = self.metrics
            shared_metric_state[self.label + "_weights"] = self.weights

    def calc_weights_only(self, edges_predicted, num_nodes):
        edges_predicted = torch.sigmoid(edges_predicted).round().int()

        block_size = edges_predicted.shape[2] if len(edges_predicted.shape) == 5 else 1
        if block_size != 1:
            num_blocks = calculate_num_blocks(num_nodes, block_size)
        else:
            num_blocks = num_nodes

        graph_counts_per_size = torch_bincount(num_blocks)

        for index, count in enumerate(graph_counts_per_size):
            if count != 0:
                self.weights.append(
                    pow(index * block_size, 2 - self.weight_power) * count
                )

    def compute(self) -> torch.Tensor:
        metrics = (
            torch.stack(self.metrics)
            if isinstance(self.metrics, list)
            else self.metrics
        )
        weights = (
            torch.stack(self.weights)
            if isinstance(self.weights, list)
            else self.weights
        )
        return (metrics * weights).sum() / weights.sum()


class EdgePrecision(EdgeMetric):
    label = "edge_precision"
    metric_class = torchmetrics.Precision


class EdgeRecall(EdgePrecision):
    label = "edge_recall"
    metric_class = torchmetrics.Recall


class EdgePrecisionNonWeighted(EdgeMetric):
    weight_power = 0
    label = "edge_precision_non_weighted"
    metric_class = torchmetrics.Precision


class EdgeRecallNonWeighted(EdgePrecision):
    weight_power = 0
    label = "edge_recall_non_weighted"
    metric_class = torchmetrics.Recall


class EdgePrecisionSquareWeighted(EdgeMetric):
    weight_power = 2
    label = "edge_precision_square_weighted"
    metric_class = torchmetrics.Precision


class EdgeRecallSquareWeighted(EdgePrecision):
    weight_power = 2
    label = "edge_recall_square_weighted"
    metric_class = torchmetrics.Recall


class MaskPrecision(torchmetrics.Precision):
    label = "mask_precision"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(
        self,
        mask_predicted: torch.Tensor,
        mask_target: torch.Tensor,
        **kwargs,
    ):
        mask_predicted = torch.sigmoid(mask_predicted).round().int().flatten()
        mask_target = mask_target.int().flatten()

        super().update(mask_predicted, mask_target)

    def compute(self) -> torch.Tensor:
        return super().compute()[1]


class MaskRecall(torchmetrics.Recall):
    label = "mask_recall"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(
        self,
        mask_predicted: torch.Tensor,
        mask_target: torch.Tensor,
        **kwargs,
    ):
        mask_predicted = torch.sigmoid(mask_predicted).round().int().flatten()
        mask_target = mask_target.int().flatten()

        super().update(mask_predicted, mask_target)

    def compute(self) -> torch.Tensor:
        return super().compute()[1]
