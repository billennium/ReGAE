from typing import Callable

from pytorch_lightning.callbacks.base import Callback


class SteppingGraphSizeMonitor(Callback):
    def __init__(
        self, get_current_subgraph_size_fn: Callable, max_num_nodes_in_dataset: int = 0
    ):
        self._get_current_subgraph_size_fn = get_current_subgraph_size_fn
        self._max_num_nodes_in_dataset = max_num_nodes_in_dataset

    def set_max_num_nodes_in_dataset(self, num_nodes: int) -> None:
        self._max_num_nodes_in_dataset = num_nodes

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        trainer.logger.log_metrics(
            {"subgraph_size": self._get_current_subgraph_size_fn()},
            step=trainer.global_step,
        )

    @staticmethod
    def _should_log(trainer) -> bool:
        # should_log = ((trainer.global_step + 1) % trainer.log_every_n_steps == 0 or trainer.should_stop)
        return True


class MetricMonitor(Callback):
    def __init__(
        self,
        data_module,
    ):
        self.data_module = data_module

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        self.data_module.current_metrics = {
            k: v.cpu().numpy() for (k, v) in trainer.callback_metrics.items()
        }
