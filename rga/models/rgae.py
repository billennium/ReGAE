import torch
from typing import List

from rga.util import load_model
from rga.util.adjmatrix import diagonal_block_representation
from rga.models.autoencoder_base import RecursiveGraphAutoencoder
from rga.util.generate_graphs import (
    convert_model_output_to_diag_block,
    diag_block_graphs_to_tril_adj_matrices,
)


class RGAE:
        """
        Easier to use wrapper around the somewhat more complex RGAE internals. Among others, this wrapper converts the 
        input adjacency matrices to the models native format on-line, which may be quite inefficient. For proper, large
        scale training, the training dataloaders should pass graphs in the `diagonal` format.
        """
    def __init__(self, path_hparams: str, path_ckpt: str):
        self.hparams = load_model.load_hparams(path_hparams)
        self.engine = load_model.load_model(
            path_hparams, path_ckpt, RecursiveGraphAutoencoder
        )

    def encode(self, adj_matrices: List[torch.FloatTensor]) -> torch.FloatTensor:
        """
                Encode graphs in adjacency matrix format into embeddings.

        Parameters
        ----------
        adj_matrices : List[torch.FloatTensor]
            List of graphs. Each graph shape (N, N) where N is node count. N>2

        Returns
        -------
        torch.FloatTensor
            Tensor with embedded graphs. Shape (B, E) where B is the number of embedded graphs and E is the embedding size (based on the loaded model).
        """
        adj_matrices_in_block_representation = []
        for el in adj_matrices:
            adj_matrices_in_block_representation.append(
                diagonal_block_representation.adj_matrix_to_diagonal_block_representation(
                    el.float()[:, :, None].clone(),
                    num_nodes=el.shape[0],
                    block_size=self.hparams["block_size"],
                    pad_value=-1,
                )
            )

        adj_matrices_in_block_representation = [
            torch.nn.utils.rnn.pad_sequence(
                adj_matrices_in_block_representation,
                batch_first=True,
                padding_value=0.0,
            ),
            [],
            torch.Tensor([el.shape[0] for el in adj_matrices]),
        ]

        embeds = self.engine.encoder.forward(adj_matrices_in_block_representation)
        return embeds

    def decode(
        self, embeds: torch.FloatTensor, max_graph_size: int = 999
    ) -> List[torch.FloatTensor]:
        """
        Parameters
        ----------
        embeds : torch.FloatTensor
            Tensor with graph embeddings. Shape (B, E) where B is the number of embedded graphs and E is the embedding size (based on the loaded model).
        max_graph_size : int
            Graph size limit (in blocks) that ensures that the resulting graph will be finite.
            Infinitely large graphs may occur if the network for some reason does not decide to end the graph, 
            which should not happen with a properly trained network.
            Default 999

        Returns
        -------
        adj_matrices: List(torch.FloatTensor)
            List of reconstructed graphs. Each graph shape (N, N) where N is node count.
        """

        reconstructed_graphs = self.engine.decoder.forward(
            embeds, max_number_of_nodes=torch.FloatTensor([max_graph_size])
        )

        adj_matrices = diag_block_graphs_to_tril_adj_matrices(
            convert_model_output_to_diag_block([reconstructed_graphs])
        )

        for i in range(len(adj_matrices)):
            adj_matrices[i] = adj_matrices[i] + adj_matrices[i].T

        return adj_matrices
