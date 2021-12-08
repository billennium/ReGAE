import torch
from argparse import ArgumentParser

from graph_nn_vae.data.synthetic_graphs_module import SyntheticGraphsDataModule


class SmoothLearningStepGraphDataModule(SyntheticGraphsDataModule):
    data_name = "subgraphs"

    def __init__(
        self,
        subgraph_scheduler_name: str = "uniform",
        subgraph_scheduler_params: dict = None,
        subgraph_depth: int = 10,
        subgraph_depth_step: int = 1,
        subgraph_stride: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.collate_fn_train = self.collate_graph_batch_training
        self.scheduler = self.get_scheduler(subgraph_scheduler_name)
        self.scheduler_params = (
            subgraph_scheduler_params if subgraph_scheduler_params is not None else {}
        )
        self.depth = subgraph_depth
        self.stride = subgraph_stride
        self.depth_step = subgraph_depth_step

    def log_size(self, size: int):
        pass
        # if self.trainer.current_epoch > self.saved_metric:
        #     # self.trainer.logger.log_metrics(  # TODO doesn't work :<
        #     #     {"max_subgraph_size/train": size},
        #     #     step=self.trainer.current_epoch
        #     #     # * np.ceil(len(self.train_dataset) / self.batch_size),
        #     # )
        #     # self.trainer.lightning_module.log(
        #     #     "max_subgraph_size/train", size, on_step=False, on_epoch=True
        #     # )

        #     # self.logger_engine.experiment.add_scalar(
        #     #     "max_subgraph_size/train",
        #     #     size,
        #     #     step.TRAINING_EPOCH
        #     #     * np.ceil(len(self.train_dataset) / self.batch_size),
        #     # )
        #     # print("Epoch: ", self.trainer.current_epoch, "max graph:", size)
        #     self.saved_metric = self.saved_metric + 1

    def get_scheduler(self, name: str):
        return {"uniform": self.uniform_epoch_size_scheduler}[name]

    def uniform_epoch_size_scheduler(self, max_size: int, speed: float = 1):
        return min(int(self.trainer.current_epoch * speed) + 2, max_size)

    def generate_subgraphs_for_batch(self, graphs, num_nodes):
        splitted_graphs = []
        splitted_graph_masks = []
        splitted_graphs_sizes = []

        current_max_size = self.scheduler(
            max_size=max(num_nodes), **self.scheduler_params
        )

        self.log_size(current_max_size)

        for graph, graph_size in zip(graphs, num_nodes):
            for i in (
                range(0, min(current_max_size - 1, self.depth), self.depth_step)
                if current_max_size < graph_size
                else [0]
            ):
                subgrpahs, subgraph_masks, subgraph_sizes = self.generate_subgraphs(
                    graph,
                    graph_size,
                    new_size=current_max_size - i,
                    stride=self.stride,
                    probability=1.0 / (i + 1),
                )
                splitted_graphs.extend(subgrpahs)
                splitted_graph_masks.extend(subgraph_masks)
                splitted_graphs_sizes.extend(subgraph_sizes)

        return splitted_graphs, splitted_graph_masks, splitted_graphs_sizes

    def generate_subgraphs(
        self,
        graph,
        graph_size: int,
        new_size: int,
        stride: int = 1,
        probability: float = 1.0,
    ):
        if new_size > graph_size:
            return [graph], [torch.ones(graph.shape)], [graph_size]

        candidates = torch.arange(0, graph_size - new_size + 1, stride).int()

        if probability < 1:
            candidates = candidates[torch.rand(len(candidates)) < probability]

        if len(candidates) == 0:
            return ([], [], [])

        graph_diagonals = []
        index = 0
        for diag_len in range(1, graph_size):
            if diag_len > graph_size - new_size:
                graph_diagonals.append(graph[index : index + diag_len])
            index = index + diag_len

        subgraphs = []
        subgraphs_masks = []
        graph_sizes = []

        for k in candidates:
            reduced_graph = torch.cat(
                [graph_diagonals[i][k : k + i + 1] for i in range(new_size - 1)]
            )
            subgraphs.append(reduced_graph)
            subgraphs_masks.append(torch.ones(reduced_graph.shape))
            graph_sizes.append(new_size)

        return (subgraphs, subgraphs_masks, graph_sizes)

    def collate_graph_batch_training(self, batch):
        # As part of the collation graph diag_repr are padded with 0.0 and the graph masks
        # are padded with 1.0 to represent the end of the graphs.
        # print(step.TRAINING_EPOCH)

        graphs = [g[0] for g in batch]
        graph_masks = [g[1] for g in batch]
        num_nodes = [g[2] for g in batch]

        graphs, graph_masks, num_nodes = self.generate_subgraphs_for_batch(
            graphs, num_nodes
        )

        graphs = torch.nn.utils.rnn.pad_sequence(
            graphs, batch_first=True, padding_value=0.0
        )
        graph_masks = torch.nn.utils.rnn.pad_sequence(
            graph_masks, batch_first=True, padding_value=0.0
        )
        num_nodes = torch.tensor(num_nodes)

        return graphs, graph_masks, num_nodes

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = SyntheticGraphsDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--subgraph_scheduler_name",
            dest="subgraph_scheduler_name",
            default="uniform",
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
        return parser
