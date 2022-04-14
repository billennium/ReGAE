from typing import List, Tuple, Dict, Type
from tqdm.auto import tqdm
from argparse import ArgumentError, ArgumentParser

import json
import torch

from rga import util
from rga.data.data_module import BaseDataModule
from rga.data.graph_loaders import BaseGraphLoader
from rga.util import adjmatrix, split_dataset_train_val_test, errors
from rga.data.util.print_dataset_statistics import print_dataset_statistics
from rga.util.adjmatrix.diagonal_block_representation import (
    adj_matrix_to_diagonal_block_representation,
)

from typing import List, Tuple, Dict
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import tqdm

import networkx as nx
import numpy as np

from rga.data.data_module import BaseDataModule
from rga.util.adjmatrix import filter_out_big_graphs


class PairedGraphLoader(BaseGraphLoader):
    data_name = "paired"

    def __init__(
        self,
        datasets_dir: str = "",
        dataset_name: str = "",
        dataset_suffixes: str = "",
        **kwargs,
    ):
        self.dataset_dir = Path(datasets_dir)
        self.dataset_name = dataset_name
        self.dataset_suffixes = dataset_suffixes
        self.data_name = dataset_name
        super().__init__(**kwargs)

    def load_graphs(self) -> Dict:

        input_graphs = []
        output_graphs = []

        for suffix in self.dataset_suffixes:
            print(
                f"Loading graphs from {self.dataset_dir / Path(self.dataset_name+ suffix)}"
            )
            input_graphs.extend(
                np.load(
                    self.dataset_dir
                    / Path(self.dataset_name + suffix)
                    / Path("input_adj.npy")
                )
            )
            output_graphs.extend(
                np.load(
                    self.dataset_dir
                    / Path(self.dataset_name + suffix)
                    / Path("target_adj.npy")
                )
            )

        return {
            "graphs": [
                (input_graph, output_graph)
                for (input_graph, output_graph) in zip(input_graphs, output_graphs)
            ],
            "labels": None,
        }

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = BaseGraphLoader.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--dataset_suffixes",
            dest="dataset_suffixes",
            type=list,
            help="dataset_suffixes",
        )
        parser.add_argument(
            "--datasets_dir",
            dest="datasets_dir",
            default="datasets",
            type=str,
            help="dir to folder of datasets (imdb, reddit, collab)",
        )
        parser.add_argument(
            "--dataset_name",
            dest="dataset_name",
            default="",
            type=str,
            help="name of dataset (IMDB_BINARY, IMDB_MULTI, COLLAB, REDDIT-BINARY, REDDIT-MULTI-5K, REDDIT-MULTI-12K)",
        )
        return parent_parser


class AdjMatrixPairedDataModule(BaseDataModule):
    graphloader_class: Type[BaseGraphLoader] = None  # override in experiment

    def __init__(
        self,
        train_val_test_split: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initialize_graphloader(**kwargs)
        self.train_val_test_split = train_val_test_split

        self.prepare_data()

    def initialize_graphloader(self, **kwargs):
        if self.graphloader_class is None:
            raise errors.MisconfigurationException(
                "the graphloader_class attribute of the DataModule was not specified"
            )
        self.graphloader: BaseGraphLoader = self.graphloader_class(**kwargs)
        self.data_name = self.graphloader.data_name

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        graph_data = self.create_graphs()

        print_dataset_statistics(
            [el[0] for el in graph_data],
            "Full original dataset (inputs)",
            False,
        )
        print_dataset_statistics(
            [el[1] for el in graph_data],
            "Full original dataset (outputs)",
            False,
        )

        train_graphs, val_graphs, test_graphs = split_dataset_train_val_test(
            graph_data, self.train_val_test_split
        )

        if len(val_graphs) == 0 or len(train_graphs) == 0:
            val_graphs = graph_data
            train_graphs = graph_data
            test_graphs = graph_data

        self.train_dataset = train_graphs
        self.val_datasets = [val_graphs]
        self.test_datasets = [test_graphs]

        print_dataset_statistics(
            [el[0] for el in self.train_dataset], "Train dataset (inputs)", False
        )
        print_dataset_statistics(
            [el[1] for el in self.train_dataset], "Train dataset (outputs)", False
        )

        # for i, d in enumerate(self.val_datasets):
        #     print_dataset_statistics(d, f"Validation dataset {i}", False)

        # for i, d in enumerate(self.test_datasets):
        #     print_dataset_statistics(d, f"Test dataset {i}", False)

        self.train_dataset = self.prepare_dataset_for_autoencoder(
            self.train_dataset,
            dataset_name="train",
        )
        self.val_datasets = [
            self.prepare_dataset_for_autoencoder(d, dataset_name=f"val {i}")
            for i, d in enumerate(self.val_datasets)
        ]
        self.test_datasets = [
            self.prepare_dataset_for_autoencoder(d, dataset_name=f"test {i}")
            for i, d in enumerate(self.test_datasets)
        ]

    def create_graphs(self) -> Dict:
        data = self.graphloader.load_graphs()
        return data["graphs"]

    def prepare_dataset_for_autoencoder(
        self,
        graph_data,
        dataset_name: str = "",
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Reorders adj matrices with bfs, removes duplicates, minimizes the matrices and appends num_nodes info.
        """
        graphs = graph_data

        adj_matrices = []

        for index, (input_adj_matrix, output_adj_matrix) in enumerate(
            tqdm(graphs, desc=f"preparing dataset {dataset_name} for autoencoder")
        ):
            torch_input_adj_matrix = adjmatrix.minimize_adj_matrix(input_adj_matrix)
            torch_output_adj_matrix = adjmatrix.minimize_adj_matrix(output_adj_matrix)

            del input_adj_matrix
            del output_adj_matrix

            adj_matrices.append(
                (
                    (
                        util.to_sparse_if_not(torch_input_adj_matrix),
                        torch_input_adj_matrix.shape[0],
                    ),
                    (
                        util.to_sparse_if_not(torch_output_adj_matrix),
                        torch_output_adj_matrix.shape[0],
                    ),
                )
            )

        return adj_matrices

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_val_test_split",
            dest="train_val_test_split",
            default=[0.7, 0.15, 0.15],
            metavar="JSON_LIST",
            type=json.loads,
            help="list of 3 floats specifying the dataset train/val/test split",
        )
        return parent_parser


class DiagonalRepresentationGraphPairedDataModule(AdjMatrixPairedDataModule):
    def __init__(
        self,
        block_size: int,
        **kwargs,
    ):
        self.block_size = block_size

        super().__init__(**kwargs)

        self.collate_fn_train = self.collate_graph_batch
        self.collate_fn_val = self.collate_graph_batch
        self.collate_fn_test = self.collate_graph_batch

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
        self.train_dataset = self.adjust_batch_representation(self.train_dataset)
        self.val_datasets = [
            self.adjust_batch_representation(d) for d in self.val_datasets
        ]
        self.test_datasets = [
            self.adjust_batch_representation(d) for d in self.test_datasets
        ]

    def adjust_batch_representation(self, batch):

        diag_block_represented_batch = []
        for graph_info_set in batch:
            graph_info = graph_info_set

            processed_example = []
            for i in [0, 1]:
                matrix = graph_info[i][0]
                num_nodes = graph_info[i][1]
                diag_block_graph = adj_matrix_to_diagonal_block_representation(
                    util.to_dense_if_not(matrix),
                    num_nodes,
                    self.block_size,
                    pad_value=-1,
                )
                adj_matrix_mask = torch.tril(
                    torch.ones((num_nodes, num_nodes)), diagonal=-1
                )[:, :, None]
                diag_block_mask = adj_matrix_to_diagonal_block_representation(
                    adj_matrix_mask, num_nodes, self.block_size
                )

                processed_example.append(
                    (
                        util.to_sparse_if_not(diag_block_graph),
                        diag_block_mask,
                        num_nodes,
                    )
                )

            diag_block_represented_batch.append(tuple(processed_example))

        return diag_block_represented_batch

    def collate_graph_batch(self, batch):
        # As part of the collation graph diag_repr and masks are padded. The graph masks 0.0 paddings
        # represent the end of the graphs.

        results = []
        for i in [0, 1]:
            graphs = torch.nn.utils.rnn.pad_sequence(
                [util.to_dense_if_not(g[i][0]) for g in batch],
                batch_first=True,
                padding_value=0.0,
            )
            graph_masks = torch.nn.utils.rnn.pad_sequence(
                [g[i][1] for g in batch],
                batch_first=True,
                padding_value=0.0,
            )
            num_nodes = torch.tensor([g[i][2] for g in batch])
            results.append((graphs, graph_masks, num_nodes))
        return results

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = AdjMatrixPairedDataModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
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
