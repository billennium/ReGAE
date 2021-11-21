import torch
import torchmetrics


class EdgeAccuracy(torchmetrics.Accuracy):
    label = "edge_accuracy"

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def update(self, edges_predicted: torch.Tensor, edges_target: torch.Tensor):
        # TODO add optional accept of mask and consider it instead of -1.0
        edges_predicted = torch.sigmoid(edges_predicted)
        edges_target = edges_target.int()
        mask = edges_target != -1
        edges_predicted = torch.round(edges_predicted).int()
        super().update(edges_predicted[mask] + 1, edges_target[mask] + 1)
