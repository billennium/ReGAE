import torch
import torchmetrics


class EdgeAccuracy(torchmetrics.Accuracy):
    label = "edge_accuracy"

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def update(self, prediction_batch: torch.Tensor, target_batch: torch.Tensor):
        target_batch = target_batch.int()
        mask = target_batch != -1
        prediction_batch = torch.round(prediction_batch).int()
        super().update(prediction_batch[mask] + 1, target_batch[mask] + 1)
