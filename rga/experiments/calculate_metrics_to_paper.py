import gc
import torch
import random
import pickle
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import pytorch_lightning as pl
from rga.data.diag_repr_graph_data_module import (
    DiagonalRepresentationGraphDataModule,
)
import math

from rga.util.load_model import load_hparams, load_model
from rga.models.autoencoder_base import RecursiveGraphAutoencoder
from rga.util.adjmatrix.diagonal_block_representation import (
    diagonal_block_to_adj_matrix_representation,
)
from rga.metrics.adjency_matrices_metrics import calculate_metrics

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def find_model_params(path, checkpoint_name):
    return {
        "params_path": str(Path(path) / Path("hparams.yaml")),
        "model_path": str(Path(path) / Path("checkpoints") / Path(checkpoint_name)),
    }


def process_model(params_path, model_path):
    target, predictions = generate_adj_matrixes(params_path, model_path)
    for k, v in calculate_metrics(target, predictions).items():
        print(k, ":", v)


def generate_adj_matrixes(params_path, model_path):
    hparams = load_hparams(params_path)
    model = load_model(params_path, model_path, RecursiveGraphAutoencoder)
    data = DiagonalRepresentationGraphDataModule(**hparams)
    predictor = pl.Trainer(gpus=1)
    test_results = predictor.predict(model, dataloaders=data.test_dataloader())
    diag_block_predictions = transfer_model_results_to_diag_block(test_results)

    prediction = diag_block_graphs_to_adj_matrices(diag_block_predictions)
    target = diag_block_graphs_to_adj_matrices(data.test_datasets[0])

    return target, prediction


def load_testset(dataset_path):
    with open(dataset_path, "rb") as input:
        return pickle.load(input)[2]


def get_num_nodes(mask):
    for i in range(1, mask.shape[0]):
        if torch.diagonal(mask, -(mask.shape[0] - i)).mean() < 0.5:
            break
    return i + 1


def remove_block_padding(graph):
    mask = graph.flatten(start_dim=1).isinf().all(dim=1)
    return graph[~mask]


def block_count_to_graph_size_in_blocks(block_count):
    return int((math.sqrt(block_count * 8 + 1) - 1) / 2)


def diag_block_graphs_to_adj_matrices(data):
    adj_matrices = []
    for (graph, _, num_nodes) in data:
        graph = graph.clamp(min=0)
        adj_matrice = diagonal_block_to_adj_matrix_representation(graph, num_nodes)
        adj_matrice = adj_matrice[:, :, 0]
        adj_matrice = adj_matrice + adj_matrice.T
        adj_matrice.fill_diagonal_(0)  # TODO czy to jest konieczne!
        adj_matrice = adj_matrice.int()

        adj_matrices.append(adj_matrice)

    return adj_matrices


def transfer_model_results_to_diag_block(model_output):
    diag_block_graphs = []

    for (graphs, masks), _ in model_output:
        for i in range(len(graphs)):
            graph = graphs[i]
            mask = masks[i]

            graph_without_padding = (
                torch.sigmoid(remove_block_padding(graph)).round().int()
            )

            block_count = mask.shape[0]
            block_size = mask.shape[1]
            mask_without_padding = torch.sigmoid(remove_block_padding(mask))

            num_nodes = get_num_nodes(
                diagonal_block_to_adj_matrix_representation(
                    mask_without_padding,
                    block_count_to_graph_size_in_blocks(block_count) * block_size,
                )
            )

            diag_block_graphs.append(
                (graph_without_padding, mask_without_padding, num_nodes)
            )

    return diag_block_graphs


process_model(
    **find_model_params(
        "tb_logs/RecurrentGraphAutoencoder/version_6", "epoch=379-step=759-v1.ckpt"
    )
)
