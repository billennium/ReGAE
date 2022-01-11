import torch


class FactorDecreasingOnMetricChange(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Multiplies the lr by `factor` on every change of a specified metric.

    The inheritance from ReduceLROnPlateau is a quite hacky solution to the problem of acquiring an outside metric.
    In Pytorch Lightning the ReduceLROnPlateau scheduler is treated differently from the rest, and PL checks
    explicitly by `isinstance(scheduler, ReduceLROnPlateau)`. Thus, by inheriting from it and using none of its
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        factor: float = 0.9,
        min_lr: float = 0.0,
        verbose=False,
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer: torch.optim.Optimizer = optimizer

        self.factor = factor
        self.verbose = verbose

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self._was_first_metric_read = False
        self._prev_metric_value = None

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if not self._was_first_metric_read:
            self._prev_metric_value = current
            self._was_first_metric_read = True

        if self._prev_metric_value != current:
            self._update_lr()
            self._prev_metric_value = current

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _update_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            param_group["lr"] = new_lr
            if self.verbose:
                print(f"Reducing learning rate of group {i} to {new_lr:.4e}.")


class SingleTimeChangeOnMetricTreshold(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Sets the lr to a specific value when a metric crosses given treshold
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_change: float,
        treshold: float,
        verbose: bool = False,
    ):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer: torch.optim.Optimizer = optimizer
        self.verbose = verbose

        self.treshold = treshold
        self.is_treshold_crossed = False

        if isinstance(lr_change, list) or isinstance(lr_change, tuple):
            if len(lr_change) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(lr_change)
                    )
                )
            self.lr_changes = list(lr_change)
        else:
            self.lr_changes = [lr_change] * len(optimizer.param_groups)

    def step(self, metrics, epoch=None):
        if not self.is_treshold_crossed:
            current = float(metrics)
            if current >= self.treshold:
                self._update_lr()
                self.is_treshold_crossed = True

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _update_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self.lr_changes[i]
            param_group["lr"] = new_lr
            if self.verbose:
                print(f"Changing learning rate of group {i} to {new_lr:.4e}.")
