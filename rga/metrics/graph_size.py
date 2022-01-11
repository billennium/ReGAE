import torch
import torchmetrics
from torchmetrics.utilities.imports import _LIGHTNING_AVAILABLE


class MaxGraphSize(torchmetrics.MaxMetric):
    label = "max_graph_size"

    def __init__(self, **kwargs):
        super().__init__(nan_strategy="error", **kwargs)

    def update(
        self,
        num_nodes: torch.Tensor,
        **kwargs,
    ):
        max_num_nodes = torch.max(num_nodes)
        super().update(max_num_nodes)

    def reset(self) -> None:
        """Overloaded so that the metric persists between epochs."""
        self._update_called = False
        self._forward_cache = None
        # lower lightning versions requires this implicitly to log metric objects correctly in self.log
        if not _LIGHTNING_AVAILABLE or self._LIGHTNING_GREATER_EQUAL_1_3:
            self._computed = None
        # reset internal states
        self._cache = None
        self._is_synced = False

    @property
    def is_differentiable(self) -> bool:
        return True
