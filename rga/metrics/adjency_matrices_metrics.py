import torch
import torchmetrics


def metric(
    target,
    predicted,
    func,
    d=1,
):
    metric_func = func(num_classes=2, average=None)

    metrics = []
    weights = []
    for target_graph, predictied_graph in zip(target, predicted):
        target_graph_size = target_graph.shape[0]

        target_graph = target_graph.flatten()
        predictied_graph = predictied_graph.flatten()

        if target_graph.shape[0] != predictied_graph.shape[0]:
            new_size = max([target_graph.shape[0], predictied_graph.shape[0]])
            target_graph = torch.nn.functional.pad(
                target_graph, [0, new_size - target_graph.shape[0]]
            )
            predictied_graph = torch.nn.functional.pad(
                predictied_graph, [0, new_size - predictied_graph.shape[0]]
            )

        precision_value = metric_func(target_graph, predictied_graph)[1]
        if precision_value != precision_value:
            precision_value = 0

        metrics.append(precision_value)

        weights.append(torch.tensor(pow(target_graph_size, 2 - d)))

    metrics = torch.stack(metrics)
    weights = torch.stack(weights)
    return (metrics * weights).sum() / weights.sum()


def f1(precision, recall):
    if precision <= 0 or recall <= 0:
        return torch.tensor([0.0])
    return 2 / (1 / precision + 1 / recall)


def average_num_node_mistake(target, predicted):
    target_sizes = torch.tensor([el.shape[0] for el in target])
    predicted_sizes = torch.tensor([el.shape[0] for el in predicted])
    return ((target_sizes - predicted_sizes).abs() / target_sizes).mean()


def calculate_metrics(target, predictions):
    precision_values = []
    recall_values = []
    for d in [0, 1, 2]:
        precision_values.append(
            metric(target, predictions, torchmetrics.Precision, d=d)
        )
        recall_values.append(metric(target, predictions, torchmetrics.Recall, d=1))

    return {
        "precision_d0": precision_values[0],
        "recall_d0": recall_values[0],
        "f1_d0": f1(precision_values[0], recall_values[0]),
        "precision_d1": precision_values[1],
        "recall_d1": recall_values[1],
        "f1_d1": f1(precision_values[1], recall_values[1]),
        "precision_d2": precision_values[2],
        "recall_d2": recall_values[2],
        "f1_d2": f1(precision_values[2], recall_values[2]),
        "average_num_node_mistake": average_num_node_mistake(target, predictions),
    }
