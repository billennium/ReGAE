import torch
import torchmetrics

from rga.util import adjmatrix


def metric(
    target,
    predicted,
    func,
    weight_power=1,
):
    metric_func = func(num_classes=2, average=None)

    metrics = []
    weights = []
    for target_graph, predicted_graph in zip(target, predicted):
        target_graph_size = adjmatrix.block_count_to_num_block_diagonals(
            target_graph.shape[0]
        )

        target_graph = target_graph.flatten()
        predicted_graph = predicted_graph.flatten()

        if target_graph.shape[0] != predicted_graph.shape[0]:
            new_size = max([target_graph.shape[0], predicted_graph.shape[0]])
            target_graph = torch.nn.functional.pad(
                target_graph, [0, new_size - target_graph.shape[0]]
            )
            predicted_graph = torch.nn.functional.pad(
                predicted_graph, [0, new_size - predicted_graph.shape[0]]
            )

        precision_value = metric_func(predicted_graph, target_graph)[1]
        if precision_value != precision_value:
            precision_value = 0

        metrics.append(precision_value)

        weight = target_graph_size ** (weight_power)
        if weight_power == 2:
            weight /= 2
        weights.append(torch.tensor(weight))

    metrics = torch.stack(metrics)
    weights = torch.stack(weights)
    return (metrics * weights).sum() / weights.sum()


def f1(precision, recall):
    if precision <= 0 or recall <= 0:
        return torch.tensor([0.0])
    return 2 / (1 / precision + 1 / recall)


def mean_size_error(target, predicted):
    target_sizes = torch.tensor([el.shape[0] for el in target])
    predicted_sizes = torch.tensor([el.shape[0] for el in predicted])
    return ((target_sizes - predicted_sizes).abs() / target_sizes).mean()


def size_accuracy(target, predicted):
    target_sizes = torch.tensor([el.shape[0] for el in target])
    predicted_sizes = torch.tensor([el.shape[0] for el in predicted])
    return (target_sizes == predicted_sizes).sum() / len(target)


def num_node_error_on_size_error(target, predicted):
    target_sizes = torch.tensor([el.shape[0] for el in target])
    predicted_sizes = torch.tensor([el.shape[0] for el in predicted])
    indices_size_error = target_sizes != predicted_sizes
    target_sizes = target_sizes[indices_size_error]
    predicted_sizes = predicted_sizes[indices_size_error]
    size_differences = (target_sizes - predicted_sizes).abs()
    return size_differences.float().mean()


def mean_size_error_on_size_error(target, predicted):
    target_sizes = torch.tensor([el.shape[0] for el in target])
    predicted_sizes = torch.tensor([el.shape[0] for el in predicted])
    indices_size_error = target_sizes != predicted_sizes
    target_sizes = target_sizes[indices_size_error]
    predicted_sizes = predicted_sizes[indices_size_error]
    size_differences = (target_sizes - predicted_sizes).abs()
    return (size_differences / target_sizes).mean()


def calculate_metrics(target, predictions):
    precision_values = []
    recall_values = []
    accuracy_values = []
    for weight_power in [0, 1, 2]:
        precision_values.append(
            metric(
                target, predictions, torchmetrics.Precision, weight_power=weight_power
            )
        )
        recall_values.append(
            metric(target, predictions, torchmetrics.Recall, weight_power=weight_power)
        )
        accuracy_values.append(
            metric(
                target, predictions, torchmetrics.Accuracy, weight_power=weight_power
            )
        )

    return {
        "Accuracy_w0": accuracy_values[0],
        "Precision_w0": precision_values[0],
        "Recall_w0": recall_values[0],
        "F1_w0": f1(precision_values[0], recall_values[0]),
        "Accuracy_w1": accuracy_values[1],
        "Precision_w1": precision_values[1],
        "Recall_w1": recall_values[1],
        "F1_w1": f1(precision_values[1], recall_values[1]),
        "Accuracy_w2": accuracy_values[2],
        "Precision_w2": precision_values[2],
        "Recall_w2": recall_values[2],
        "F1_w2": f1(precision_values[2], recall_values[2]),
        "Size accuracy": size_accuracy(target, predictions),
        "Mean size error": mean_size_error(target, predictions),
        "Num node error on size error": num_node_error_on_size_error(
            target, predictions
        ),
        "Mean size error on size error": mean_size_error_on_size_error(
            target, predictions
        ),
    }
