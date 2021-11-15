import torch
import torchmetrics

class PositivePrecision(torchmetrics.Precision):
    label = "edge_precision_1"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(self, prediction_batch: torch.Tensor, target_batch: torch.Tensor):
        target_batch = target_batch.int().clamp(min=0).flatten()
        prediction_batch = torch.round(prediction_batch).int().clamp(min=0).flatten()

        super().update(prediction_batch, target_batch)

    def compute(self) -> torch.Tensor:
        return super().compute()[1] 


class PositiveRecall(torchmetrics.Recall):
    label = "edge_recall_1"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(self, prediction_batch: torch.Tensor, target_batch: torch.Tensor):
        target_batch = target_batch.int().clamp(min=0).flatten()
        prediction_batch = torch.round(prediction_batch).int().clamp(min=0).flatten()

        super().update(prediction_batch, target_batch)

    def compute(self) -> torch.Tensor:
        return super().compute()[1] 

class NegativePrecision(torchmetrics.Precision):
    label = "edge_precision_-1"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(self, prediction_batch: torch.Tensor, target_batch: torch.Tensor):
        target_batch = target_batch.int().clamp(max=0).flatten()*-1
        prediction_batch = torch.round(prediction_batch).int().clamp(max=0).flatten()*-1

        super().update(prediction_batch, target_batch)

    def compute(self) -> torch.Tensor:
        return super().compute()[1] 


class NegativeRecall(torchmetrics.Recall):
    label = "edge_recall_-1"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(self, prediction_batch: torch.Tensor, target_batch: torch.Tensor):
        target_batch = target_batch.int().clamp(max=0).flatten()*-1
        prediction_batch = torch.round(prediction_batch).int().clamp(max=0).flatten()*-1

        super().update(prediction_batch, target_batch)

    def compute(self) -> torch.Tensor:
        return super().compute()[1] 
