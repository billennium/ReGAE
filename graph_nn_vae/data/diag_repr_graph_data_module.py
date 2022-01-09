from typing import List, Tuple
from argparse import ArgumentError, ArgumentParser
from operator import itemgetter

import torch
from torch.functional import Tensor
from torch.utils import data

from graph_nn_vae.data.adj_matrix_data_module import (
    AdjMatrixDataModule,
)
from graph_nn_vae.util.adjmatrix.diagonal_block_representation import (
    adj_matrix_to_diagonal_block_representation,
    calculate_num_blocks,
)
from graph_nn_vae.util.callbacks import MetricMonitor, SteppingGraphSizeMonitor
from graph_nn_vae.data.subgraphs import (
    get_subgraph_size_scheduler,
    generate_subgraphs,
)


class DiagonalRepresentationGraphDataModule(AdjMatrixDataModule):
    is_scheduling_initialized = False

    def __init__(
        self,
        block_size: int,
        subgraph_scheduler_name: str,
        subgraph_scheduler_params: dict,
        **kwargs
    ):
        self.block_size = block_size

        super().__init__(**kwargs)

        self.collate_fn_train = self.collate_graph_batch
        self.collate_fn_val = self.collate_graph_batch
        self.collate_fn_test = self.collate_graph_batch

        self.subgraph_size_scheduler = get_subgraph_size_scheduler(
            subgraph_scheduler_name
        )

        if self.subgraph_size_scheduler is not None:
            self.prepare_module_to_smooth_learning(subgraph_scheduler_params, **kwargs)

    def prepare_module_to_smooth_learning(
        self,
        scheduler_params: dict,
        subgraph_stride: float,
        minimal_subgraph_size: int,
        **kwargs
    ):
        kwargs["data_module"] = self
        self.subgraph_size_scheduler = self.subgraph_size_scheduler(
            scheduler_params, **kwargs
        )
        self.subgraph_stride = max(min(1, subgraph_stride), 0)
        self.minimal_subgraph_size = minimal_subgraph_size
        self.current_metrics = {}

        self.current_training_dataloader = None
        self.current_training_dataset_lvl = -1

    def init_scheduler(self):
        self.subgraph_size_scheduler.set_epoch_num_source(self.trainer)
        max_num_nodes_in_train = self.get_max_num_nodes_in_dataset(self.train_dataset)
        self.subgraph_size_monitor = SteppingGraphSizeMonitor(
            self.subgraph_size_scheduler.get_current_subgraph_size,
            max_num_nodes_in_train,
        )
        self.trainer.callbacks.append(self.subgraph_size_monitor)

        self.current_metric_monitor = MetricMonitor(self)
        self.trainer.callbacks.append(self.current_metric_monitor)

        self.is_logging_initialized = True
        self.is_scheduling_initialized = True

    def get_max_num_nodes_in_dataset(self, dataset):
        return max(dataset, key=itemgetter(2))[2]

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
        self.train_dataset = self.adjust_batch_representation(self.train_dataset)
        self.val_datasets = [
            self.adjust_batch_representation(d) for d in self.val_datasets
        ]
        self.test_datasets = [
            self.adjust_batch_representation(d) for d in self.test_datasets
        ]

    def adjust_batch_representation(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:

        diag_block_represented_batch = []
        for graph_info_set in batch:
            graph_info = graph_info_set[0] if self.use_labels else graph_info_set
            matrix = graph_info[0]
            num_nodes = graph_info[1]
            diag_block_graph = adj_matrix_to_diagonal_block_representation(
                matrix, num_nodes, self.block_size, pad_value=-1
            )
            adj_matrix_mask = torch.tril(
                torch.ones((num_nodes, num_nodes)), diagonal=-1
            )[:, :, None]
            diag_block_mask = adj_matrix_to_diagonal_block_representation(
                adj_matrix_mask, num_nodes, self.block_size
            )

            processed_example = (diag_block_graph, diag_block_mask, num_nodes)
            if self.use_labels:
                processed_example = (processed_example, graph_info_set[1])

            diag_block_represented_batch.append(processed_example)

        return diag_block_represented_batch

    def train_dataloader(self, **kwargs):
        if self.subgraph_size_scheduler is not None:
            return self.train_dataloader_subgraphs(**kwargs)

        return super().train_dataloader(**kwargs)

    def train_dataloader_subgraphs(self, **kwargs):
        if not self.is_scheduling_initialized:
            self.init_scheduler()

        scheduled_subgraph_size = (
            self.subgraph_size_scheduler.get_current_subgraph_size()
        )

        if (scheduled_subgraph_size > self.current_training_dataset_lvl) and (
            scheduled_subgraph_size < 1
        ):
            graphs = [g[0] for g in self.train_dataset]
            graph_masks = [g[1] for g in self.train_dataset]
            num_nodes = [g[2] for g in self.train_dataset]

            graphs, graph_masks, num_nodes = self.generate_subgraphs_for_batch(
                graphs, graph_masks, num_nodes, scheduled_subgraph_size
            )
            current_training_dataset = list(zip(graphs, graph_masks, num_nodes))

            self.current_training_dataset_lvl = scheduled_subgraph_size
            self.current_training_dataloader = data.DataLoader(
                current_training_dataset,
                batch_size=self.batch_size,
                num_workers=self.workers,
                pin_memory=True,
                collate_fn=self.collate_fn_train,
                **kwargs,
            )
        elif scheduled_subgraph_size >= 1 and self.current_training_dataset_lvl < 1:
            self.current_training_dataset_lvl = 1
            self.current_training_dataloader = data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.workers,
                pin_memory=True,
                collate_fn=self.collate_fn_train,
                **kwargs,
            )

        return self.current_training_dataloader

    def generate_subgraphs_for_batch(
        self,
        graphs: Tensor,
        graph_masks: Tensor,
        num_nodes: Tensor,
        target_subgraph_size: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        splitted_graphs = []
        splitted_graph_masks = []
        splitted_graphs_sizes = []

        for graph, mask, num_nodes in zip(graphs, graph_masks, num_nodes):
            num_blocks = calculate_num_blocks(torch.tensor(num_nodes), self.block_size)
            if num_blocks > self.minimal_subgraph_size:
                current_subgraph_size = max(
                    int(target_subgraph_size * num_blocks), self.minimal_subgraph_size
                )
                stride = int(current_subgraph_size * self.subgraph_stride)

                subgrpahs, subgraph_masks, subgraph_sizes = generate_subgraphs(
                    graph,
                    mask,
                    num_nodes,
                    num_blocks,
                    self.block_size,
                    new_size=current_subgraph_size,
                    stride=stride,
                    probability=1.0,
                )

                splitted_graphs.extend(subgrpahs)
                splitted_graph_masks.extend(subgraph_masks)
                splitted_graphs_sizes.extend(subgraph_sizes)
            else:
                splitted_graphs.append(graph)
                splitted_graph_masks.append(torch.ones(graph.shape))
                splitted_graphs_sizes.append(num_nodes)

        return splitted_graphs, splitted_graph_masks, splitted_graphs_sizes

    def collate_graph_batch(self, batch):
        # As part of the collation graph diag_repr and masks are padded. The graph masks 0.0 paddings
        # represent the end of the graphs.

        graphs = torch.nn.utils.rnn.pad_sequence(
            [g[0][0] if self.use_labels else g[0] for g in batch],
            batch_first=True,
            padding_value=0.0,
        )
        graph_masks = torch.nn.utils.rnn.pad_sequence(
            [g[0][1] if self.use_labels else g[1] for g in batch],
            batch_first=True,
            padding_value=0.0,
        )
        num_nodes = torch.tensor([g[0][2] if self.use_labels else g[2] for g in batch])

        if self.use_labels:
            labels = torch.LongTensor([g[1] for g in batch])
            return (graphs, graph_masks, num_nodes, labels)

        else:
            return (graphs, graph_masks, num_nodes)

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = AdjMatrixDataModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--subgraph_scheduler_name",
            dest="subgraph_scheduler_name",
            default=None,
            type=str,
            help="name of maximum subgraph size scheduler",
        )
        parser.add_argument(
            "--subgraph_scheduler_params",
            dest="subgraph_scheduler_params",
            default={},
            type=dict,
            help="parameters for selected subgraph size scheduler",
        )
        parser.add_argument(
            "--subgraph_depth",
            dest="subgraph_depth",
            default=10,
            type=int,
            help="depth of looking from maximal subgraph size to smaller subgraphs",
        )
        parser.add_argument(
            "--subgraph_depth_step",
            dest="subgraph_depth_step",
            default=1,
            type=int,
            help="step of looking from maximal subgraph size to smaller subgraphs",
        )
        parser.add_argument(
            "--subgraph_stride",
            dest="subgraph_stride",
            default=1,
            type=int,
            help="stride between subgraphs",
        )
        parser.add_argument(
            "--minimal_subgraph_size",
            dest="minimal_subgraph_size",
            default=10,
            type=int,
            help="minimal subgraph size",
        )
        parser.set_defaults(
            reload_dataloaders_every_n_epochs=1
        )  # TODO only if we have scheduler

        try:  # may collide with an autoencoder module, but that's fine
            parser.add_argument(
                "--block_size",
                dest="block_size",
                default=1,
                type=int,
                help="size (width or height) of a block of adjacency matrix edges",
            )
        except ArgumentError:
            pass

        return parent_parser
