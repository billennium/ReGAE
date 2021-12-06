import torch
import numpy as np

from functools import partial

from graph_nn_vae.data.synthetic_graphs_module import SyntheticGraphsDataModule

import graph_nn_vae.util.training_monitor as training_monitor


class SmoothLearningStepGraphDataModule(SyntheticGraphsDataModule):
    data_name = "subgraphs"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.saved_metric = -1

    def train_dataloader(self, **kwargs):
        dl = super().train_dataloader(**kwargs)
        dl.collate_fn = self.collate_graph_batch_training
        return dl

    def log_size(self, size):
        if training_monitor.TRAINING_EPOCH > self.saved_metric:
            self.logger_engine.log_metrics(  # TODO doesen't work :<
                {"max_subgraph_size/train": size},
                step=training_monitor.TRAINING_EPOCH
                # * np.ceil(len(self.train_dataset) / self.batch_size),
            )
            # self.logger_engine.experiment.add_scalar(
            #     "max_subgraph_size/train",
            #     size,
            #     step.TRAINING_EPOCH
            #     * np.ceil(len(self.train_dataset) / self.batch_size),
            # )
            # print("Epoch: ", step.TRAINING_EPOCH, "max graph:", size)
            self.saved_metric = self.saved_metric + 1

    def epoch_size_scheduler(self, max_size):  # TODO make more generic
        size = min(int(training_monitor.TRAINING_EPOCH / 3) + 2, max_size)
        return size

    def split_graphs(self, graphs, num_nodes):
        splitted_graphs = []
        splitted_graph_masks = []
        splitted_graphs_sizes = []
        n = self.epoch_size_scheduler(max(num_nodes))

        self.log_size(n)

        for graph, graph_size in zip(graphs, num_nodes):  # TODO beta version
            if n < graph_size:
                tmp = range(min(n - 1, 10))
            else:
                tmp = [0]

            for i in tmp:
                subgrpahs, subgraph_masks, subgraph_sizes = self.generate_subgraphs(
                    graph,
                    graph_size,
                    n=n - i,
                    stride=1,
                    probability=1.0 / pow(i + 1, 2),
                )
                splitted_graphs.extend(subgrpahs)
                splitted_graph_masks.extend(subgraph_masks)
                splitted_graphs_sizes.extend(subgraph_sizes)

        return splitted_graphs, splitted_graph_masks, splitted_graphs_sizes

    def generate_subgraphs(
        self, graph, graph_size: int, n: int, stride: int = 1, probability: float = 1.0
    ):
        if n > graph_size:
            return [graph], [torch.ones(graph.shape)], [graph_size]

        candidates = torch.arange(0, graph_size - n + 1, stride).int()

        if probability < 1:
            candidates = candidates[torch.rand(len(candidates)) < probability]

        if len(candidates) == 0:
            return ([], [], [])

        graph_diagonals = []
        index = 0
        for diag_len in range(1, graph_size):
            if diag_len > graph_size - n:
                graph_diagonals.append(graph[index : index + diag_len])
            index = index + diag_len

        subgraphs = []
        subgraphs_masks = []
        graph_sizes = []

        for k in candidates:
            reduced_graph = torch.cat(
                [graph_diagonals[i][k : k + i + 1] for i in range(n - 1)]
            )
            subgraphs.append(reduced_graph)
            subgraphs_masks.append(torch.ones(reduced_graph.shape))
            graph_sizes.append(n)

        return (subgraphs, subgraphs_masks, graph_sizes)

    def collate_graph_batch_training(self, batch):
        # As part of the collation graph diag_repr are padded with 0.0 and the graph masks
        # are padded with 1.0 to represent the end of the graphs.
        # print(step.TRAINING_EPOCH)

        graphs = [g[0] for g in batch]
        graph_masks = [g[1] for g in batch]
        num_nodes = [g[2] for g in batch]

        graphs, graph_masks, num_nodes = self.split_graphs(graphs, num_nodes)

        graphs = torch.nn.utils.rnn.pad_sequence(
            graphs, batch_first=True, padding_value=0.0
        )
        graph_masks = torch.nn.utils.rnn.pad_sequence(
            graph_masks, batch_first=True, padding_value=0.0
        )
        num_nodes = torch.tensor(num_nodes)

        return graphs, graph_masks, num_nodes
