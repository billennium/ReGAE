from typing import List, Tuple
import torch
from torch.functional import Tensor
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

        new_metrics, new_weights = calc_egde_metric(
            edges_predicted, edges_target, num_nodes, self.metric, self.weight_power
        )
        self.metrics.extend(new_metrics)
        self.weights.extend(new_weights)

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


def calc_egde_metric(
    edges_predicted: torch.Tensor,
    edges_target: torch.Tensor,
    num_nodes: torch.Tensor,
    metric: torchmetrics.Metric,
    weight_power: float,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Returns per graph metric and weight values calculated based on the metric passed.
    """
    edges_predicted = torch.sigmoid(edges_predicted).round().int()
    edges_target = torch.clamp(edges_target.int(), min=0)

    block_size = edges_predicted.shape[2] if len(edges_predicted.shape) == 5 else 1
    if block_size != 1:
        num_blocks = calculate_num_blocks(num_nodes, block_size)
    else:
        num_blocks = num_nodes

    graph_counts_per_size = torch_bincount(num_blocks)

    metric_values = []
    weights = []

    for index, count in enumerate(graph_counts_per_size):
        if count != 0:
            mask = num_blocks == index
            predicted = edges_predicted[mask].flatten()
            target = edges_target[mask].flatten()

            current_metric = metric(predicted, target)[1]
            metric.reset()
            if current_metric != current_metric:
                current_metric = torch.zeros(1, device=predicted.device)[0]

            metric_values.append(current_metric)

            weights.append(pow(index * block_size, 2 - weight_power) * count)

    return metric_values, weights


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


class EdgeF1(torchmetrics.Metric):
    """
    Measures the F_1 score of edge classification.
    https://en.wikipedia.org/wiki/F-score
    """

    label = "edge_f1"
    weight_power = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("precision_metrics", default=[], dist_reduce_fx="cat")
        self.add_state("precision_weights", default=[], dist_reduce_fx="cat")
        self.add_state("recall_metrics", default=[], dist_reduce_fx="cat")
        self.add_state("recall_weights", default=[], dist_reduce_fx="cat")

        self.precision_metric = torchmetrics.Precision(num_classes=2, average=None)
        self.recall_metric = torchmetrics.Recall(num_classes=2, average=None)
        self.shared_precision_metrics_label = (
            "edge_" + self.precision_metric.__class__.__name__
        )
        self.shared_recall_metrics_label = (
            "edge_" + self.recall_metric.__class__.__name__
        )
        self.shared_precision_weights_label = EdgePrecision.label + "_weights"
        self.shared_recall_weights_label = EdgeRecall.label + "_weights"

    def update(
        self,
        edges_predicted: torch.Tensor,
        edges_target: torch.Tensor,
        num_nodes: torch.Tensor,
        shared_metric_state: dict = None,
        **kwargs,
    ):
        self.precision_metrics, self.precision_weights = self._update_metric(
            self.shared_precision_metrics_label,
            self.shared_precision_weights_label,
            self.precision_metric,
            self.precision_metrics,
            self.precision_weights,
            edges_predicted,
            edges_target,
            num_nodes,
            shared_metric_state,
        )
        self.recall_metrics, self.recall_weights = self._update_metric(
            self.shared_recall_metrics_label,
            self.shared_recall_weights_label,
            self.recall_metric,
            self.recall_metrics,
            self.recall_weights,
            edges_predicted,
            edges_target,
            num_nodes,
            shared_metric_state,
        )

    def _update_metric(
        self,
        shared_metrics_label: str,
        shared_weights_label: str,
        metric: torchmetrics.Metric,
        metrics: list,
        weights: list,
        edges_predicted: torch.Tensor,
        edges_target: torch.Tensor,
        num_nodes: torch.Tensor,
        shared_metric_state: dict = None,
    ) -> Tuple[list, list]:
        if (
            isinstance(shared_metric_state, dict)
            and shared_metrics_label in shared_metric_state
            and shared_weights_label in shared_metric_state
        ):
            metrics = shared_metric_state[shared_metrics_label]
            weights = shared_metric_state[shared_weights_label]
        else:
            new_prec_metrics, new_prec_weights = calc_egde_metric(
                edges_predicted,
                edges_target,
                num_nodes,
                metric,
                self.weight_power,
            )
            metrics.extend(new_prec_metrics)
            weights.extend(new_prec_weights)
            if isinstance(shared_metric_state, dict):
                shared_metric_state[shared_metrics_label] = metrics
        return metrics, weights

    def compute(self) -> torch.Tensor:
        precision = self._compute_metric(self.precision_metrics, self.precision_weights)
        recall = self._compute_metric(self.recall_metrics, self.recall_weights)
        if precision <= 0 or recall <= 0:
            return torch.tensor([0.0], device=self.device)
        return 2 / (1 / precision + 1 / recall)

    def _compute_metric(self, metrics: list, weights: list) -> Tensor:
        stacked_metrics = torch.stack(metrics) if isinstance(metrics, list) else metrics
        stacked_weights = torch.stack(weights) if isinstance(weights, list) else weights
        return (stacked_metrics * stacked_weights).sum() / stacked_weights.sum()


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
