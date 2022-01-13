import torch
import torchmetrics

from rga.util import adjmatrix


def metric(
    target,
    predicted,
    func,
    power=1,
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
            # continue
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

        weights.append(torch.tensor(pow(target_graph_size, 2 - power)))

    metrics = torch.stack(metrics)
    weights = torch.stack(weights)
    return (metrics * weights).sum() / weights.sum()


def f1(precision, recall):
    if precision <= 0 or recall <= 0:
        return torch.tensor([0.0])
    return 2 / (1 / precision + 1 / recall)


def average_num_node_error(target, predicted):
    target_sizes = torch.tensor([el.shape[0] for el in target])
    predicted_sizes = torch.tensor([el.shape[0] for el in predicted])
    return ((target_sizes - predicted_sizes).abs() / target_sizes).mean()


def fraction_graph_size_errors(target, predicted):
    target_sizes = torch.tensor([el.shape[0] for el in target])
    predicted_sizes = torch.tensor([el.shape[0] for el in predicted])
    return (target_sizes != predicted_sizes).sum() / len(target)


def average_num_node_difference_on_size_error(target, predicted):
    target_sizes = torch.tensor([el.shape[0] for el in target])
    predicted_sizes = torch.tensor([el.shape[0] for el in predicted])
    indices_size_error = target_sizes != predicted_sizes
    target_sizes = target_sizes[indices_size_error]
    predicted_sizes = predicted_sizes[indices_size_error]
    size_differences = (target_sizes - predicted_sizes).abs()
    return size_differences.float().mean()


def average_size_difference_on_size_error(target, predicted):
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
    for power in [0, 1, 2]:
        precision_values.append(
            metric(target, predictions, torchmetrics.Precision, power=power)
        )
        recall_values.append(
            metric(target, predictions, torchmetrics.Recall, power=power)
        )
    accuracy_value = metric(target, predictions, torchmetrics.Accuracy, power=power)

    return {
        "accuracy": accuracy_value,
        "precision_d0": precision_values[0],
        "recall_d0": recall_values[0],
        "f1_d0": f1(precision_values[0], recall_values[0]),
        "precision_d1": precision_values[1],
        "recall_d1": recall_values[1],
        "f1_d1": f1(precision_values[1], recall_values[1]),
        "precision_d2": precision_values[2],
        "recall_d2": recall_values[2],
        "f1_d2": f1(precision_values[2], recall_values[2]),
        "average_num_node_error": average_num_node_error(target, predictions),
        "fraction_graph_size_errors": fraction_graph_size_errors(target, predictions),
        "average_num_node_difference_on_size_error": average_num_node_difference_on_size_error(
            target, predictions
        ),
        "average_size_difference_on_size_error": average_size_difference_on_size_error(
            target, predictions
        ),
    }
