import torch


class NoSched(torch.optim.lr_scheduler._LRScheduler):
    """
    Does not decay the lr whatsoever.
    """

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(NoSched, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs
