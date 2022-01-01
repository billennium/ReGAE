from argparse import ArgumentParser, ArgumentError
from typing import Callable, List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.utils.calc import weighted_average
from graph_nn_vae.models.utils.layers import (
    parse_layer_sizes_list,
    sequential_from_layer_sizes,
)
from graph_nn_vae.models.utils.getters import get_activation_function
from graph_nn_vae.util.adjmatrix.diagonal_block_representation import (
    calculate_num_blocks,
)


class GraphEncoder(BaseModel):
    def __init__(
        self,
        edge_encoder_class: nn.Module,
        embedding_size: int,
        edge_size: int,
        block_size: int,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        self.block_size = block_size
        super(GraphEncoder, self).__init__(**kwargs)
        self.edge_encoder = edge_encoder_class(
            embedding_size, edge_size, block_size, **kwargs
        )

    def forward(self, input_batch: Tensor) -> Tensor:
        """
        :param input_batch:
            Batch consisting of a tuple of diagonally represented graphs, their masks (unused here) and their number of nodes.

            The graphs shall be represented in the "diagonal form" with a shape like [summed_diagonal_length+padding, edge_size].
            For example, skipping the edge_size dimension for a graph of adjacency matrix:
             x 0 1 2 3
            y
            0  0 0 0 0
            1  1 0 0 0
            2  1 0 0 0
            3  0 1 1 0
                |
                V
            011101 + -1 padding

        :return:
            Graph embedding Tensor of dimensions [batch_size x embedding_size]
        """
        diagonal_repr_graphs_batch = input_batch[0]
        diagonal_repr_graphs_batch.requires_grad = True
        num_nodes_batch = input_batch[2]
        num_blocks_batch = calculate_num_blocks(num_nodes_batch, self.block_size)

        sorted_num_blocks_batch, ordered_indices = num_blocks_batch.sort(
            descending=True
        )
        _, indices_in_original_batch_order = ordered_indices.sort()

        diagonal_repr_graphs_batch = diagonal_repr_graphs_batch[ordered_indices]

        max_num_blocks = sorted_num_blocks_batch[0]
        num_diagonals = max_num_blocks
        first_diag_length = num_diagonals
        diag_right_pos = int((1 + num_diagonals) * num_diagonals / 2)

        graph_counts_per_size = self.torch_bincount(num_blocks_batch)

        # Embedding batch is represented in the shape: [graph_idx, embedding_idx, embedding]
        # Starting with `0` for no graphs yet. Will get filled approprately in the recurrent loop.
        prev_embedding = torch.zeros(
            (0, max_num_blocks + 1, self.embedding_size),
            requires_grad=True,
            device=diagonal_repr_graphs_batch.device,
        )

        for diagonal_offset in range(max_num_blocks):
            # Some graphs from the input batch may have been too small for the previous diagonal.
            # Check if they should be added now and init their embeddings.
            graphs_to_add_in_curr_diag = graph_counts_per_size[
                max_num_blocks - diagonal_offset
            ]
            if graphs_to_add_in_curr_diag != 0:
                new_graph_init_tokens = torch.zeros(
                    (
                        graphs_to_add_in_curr_diag,
                        max_num_blocks + 1 - diagonal_offset,
                        self.embedding_size,
                    ),
                    requires_grad=True,
                    device=diagonal_repr_graphs_batch.device,
                )
                prev_embedding = torch.cat([prev_embedding, new_graph_init_tokens])

            diag_length = first_diag_length - diagonal_offset
            diag_left_pos = diag_right_pos - diag_length
            current_diagonal = diagonal_repr_graphs_batch[
                : prev_embedding.shape[0], diag_left_pos:diag_right_pos, :
            ]

            embeddings_left = prev_embedding[:, :-1, :]
            embeddings_right = prev_embedding[:, 1:, :]

            new_embedding = self.edge_encoder(
                current_diagonal, embeddings_left, embeddings_right
            )
            prev_embedding = new_embedding

            diag_right_pos = diag_left_pos

        # Reorder back to the original batch order and skip the no longer needed second dimension.
        return prev_embedding[indices_in_original_batch_order, 0, :]

    def torch_bincount(self, t: Tensor) -> Tensor:
        """
        torch.bincount() when used on CUDA may lead to nondeterministic gradients. From testing, this isn't an issue in our use case.
        """
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        t = torch.bincount(t)
        torch.use_deterministic_algorithms(was_deterministic)
        return t

    def step(self, batch: Tensor) -> Tensor:
        embeddings = self(batch)
        diagonal_repr_graphs_batch = batch[0]
        num_nodes_batch = batch[1]
        num_edges = torch.zeros(
            (len(diagonal_repr_graphs_batch), self.embedding_size),
            device=diagonal_repr_graphs_batch.device,
        )
        for i, diagonal_repr_graph in enumerate(diagonal_repr_graphs_batch):
            num_nodes = num_nodes_batch[i].item()
            non_padded_diagonal_length = int(((1 + num_nodes) * num_nodes) / 2)
            diagonal_repr_graph = diagonal_repr_graph[:non_padded_diagonal_length]
            num_edges[i, 0] = torch.sum(diagonal_repr_graph)
        return F.mse_loss(embeddings, num_edges)

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        try:  # these may collide with an upper autoencoder, but that's fine
            parent_parser = BaseModel.add_model_specific_args(
                parent_parser=parent_parser
            )
        except ArgumentError:
            pass
        parser = parent_parser.add_argument_group(cls.__name__)
        try:  # these may collide with an encoder module, but that's fine
            parser = BaseModel.add_model_specific_args(parent_parser=parser)
            parser.add_argument(
                "--embedding-size",
                dest="embedding_size",
                default=32,
                type=int,
                metavar="EMB_SIZE",
                help="size of the encoder output graph embedding",
            )
            parser.add_argument(
                "--edge-size",
                dest="edge_size",
                default=1,
                type=int,
                metavar="EDGE_SIZE",
                help="number of dimensions of a graph's edge",
            )
        except ArgumentError:
            pass
        try:  # may collide with a data module, but that's fine
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


class GraphDecoder(BaseModel):
    fill_border_embeddings_fn: Callable
    fill_border_separate_sides: bool

    def __init__(
        self,
        edge_decoder_class: nn.Module,
        embedding_size: int,
        edge_size: int,
        block_size: int,
        graph_decoder_border_embedding_fill: str,
        graph_decoder_filling_nn_layer_sizes: List[int],
        graph_decoder_filling_nn_activation_function: str,
        **kwargs,
    ):
        if embedding_size % 2 != 0:
            raise ValueError(
                "graph decoder's input graph embedding size must be divisible by 2"
            )
        self.internal_embedding_size = int(embedding_size / 2)
        self.edge_size = edge_size
        self.block_size = block_size
        super().__init__(**kwargs)

        self.edge_decoder = edge_decoder_class(
            embedding_size=self.internal_embedding_size,
            edge_size=edge_size,
            block_size=block_size,
            **kwargs,
        )

        self.set_fill_border_embeddings_fn(
            graph_decoder_border_embedding_fill,
            graph_decoder_filling_nn_layer_sizes,
            graph_decoder_filling_nn_activation_function,
        )

    def forward(
        self, graph_encoding_batch: Tensor, max_number_of_nodes: int
    ) -> Tuple[Tensor, Tensor]:
        """
        :param graph_encoding_batch: batch of graph encodings (products of an encoder) of dimensions [batch_size, embedding_size]
        :return: graph adjacency matrices tensor of dimensions [batch_size, num_nodes, num_nodes, edge_size]
        """
        decoded_diagonals_with_masks = []
        # The working embeddings batch has this shape: [graph_idx x embdedding_idx x embedding]
        prev_doubled_embeddings = graph_encoding_batch[:, None]
        prev_embeddings_l, prev_embeddings_r = torch.split(
            prev_doubled_embeddings,
            (self.internal_embedding_size, self.internal_embedding_size),
            dim=-1,
        )

        original_indices = torch.IntTensor(list(range(graph_encoding_batch.shape[0])))
        indices_of_finished_graphs = []

        diagonal_embedding_squares = torch.zeros(
            [1], device=graph_encoding_batch.device
        )
        mask_state = None
        max_num_blocks = int(calculate_num_blocks(max_number_of_nodes, self.block_size))

        for _ in range(max_num_blocks):
            (
                decoded_edges_with_mask,
                new_embedding_l,
                new_embedding_r,
            ) = self.edge_decoder(prev_embeddings_l, prev_embeddings_r)

            masks = decoded_edges_with_mask[..., 0]
            # just here, not part of the output - used for checking if the graphs are finished in the loop
            masks = torch.sigmoid(masks)

            decoded_edges_with_mask_padded = decoded_edges_with_mask
            for i in sorted(indices_of_finished_graphs):
                decoded_edges_with_mask_padded = torch.cat(
                    [
                        decoded_edges_with_mask_padded[:i],
                        torch.full(
                            (1, *decoded_edges_with_mask_padded.shape[1:]),
                            fill_value=float("-inf"),
                            device=decoded_edges_with_mask_padded.device,
                        ),
                        decoded_edges_with_mask_padded[i:],
                    ],
                )
            decoded_diagonals_with_masks.append(decoded_edges_with_mask_padded)

            indices_graphs_finished, mask_state = find_finished_masks(masks, mask_state)

            indices_of_finished_graphs.extend(
                original_indices[indices_graphs_finished].tolist()
            )
            original_indices = original_indices[~indices_graphs_finished]

            if any(indices_graphs_finished):
                finished_emb_l = new_embedding_l[indices_graphs_finished]
                finished_emb_r = new_embedding_r[indices_graphs_finished]
                finished_doubled_embeddings = torch.cat(
                    (finished_emb_l, finished_emb_r), dim=-1
                )
                diagonal_embedding_squares += (
                    finished_doubled_embeddings.flatten().square().sum()
                )

            new_embedding_l = new_embedding_l[~indices_graphs_finished]
            new_embedding_r = new_embedding_r[~indices_graphs_finished]

            if new_embedding_l.shape[0] == 0:
                break

            prev_embeddings_l = prev_embeddings_l[~indices_graphs_finished]
            prev_embeddings_r = prev_embeddings_r[~indices_graphs_finished]

            prev_embeddings_l, prev_embeddings_r = self.fill_border_embeddings_fn(
                prev_embeddings_l, prev_embeddings_r, new_embedding_l, new_embedding_r
            )

        concatenated_diagonals_with_masks = torch.cat(
            decoded_diagonals_with_masks, dim=1
        )

        if new_embedding_l.shape[0] > 0:
            unfinished_doubled_embeddings = torch.cat(
                (new_embedding_l, new_embedding_r), dim=-1
            )
            diagonal_embedding_squares += (
                unfinished_doubled_embeddings.flatten().square().sum()
            )

        masks, concatenated_diagonals = torch.split(
            concatenated_diagonals_with_masks, (1, self.edge_size), dim=-1
        )

        diagonal_embeddings_norm = diagonal_embedding_squares.sqrt()

        return (concatenated_diagonals, masks), diagonal_embeddings_norm

    def set_fill_border_embeddings_fn(
        self,
        name: str,
        graph_decoder_filling_nn_layer_sizes: List[int],
        graph_decoder_filling_nn_activation_function: str,
    ) -> None:
        if name == "pad":
            self.fill_border_embeddings_fn = self.pad_missing_embeddings
            return

        activation_f = get_activation_function(
            graph_decoder_filling_nn_activation_function
        )
        if name == "separate_sides_nn":
            self.border_embedding_nn_l = sequential_from_layer_sizes(
                self.internal_embedding_size * 2,
                self.internal_embedding_size * 2,
                graph_decoder_filling_nn_layer_sizes,
                activation_f,
            )
            self.border_embedding_nn_r = sequential_from_layer_sizes(
                self.internal_embedding_size * 2,
                self.internal_embedding_size * 2,
                graph_decoder_filling_nn_layer_sizes,
                activation_f,
            )
            self.fill_border_separate_sides = True
            self.fill_border_embeddings_fn = self.nn_fill_missing_embeddings
            return
        if name == "single_nn":
            self.border_embedding_nn = sequential_from_layer_sizes(
                self.internal_embedding_size * 2,
                self.internal_embedding_size * 2,
                graph_decoder_filling_nn_layer_sizes,
                activation_f,
            )
            self.fill_border_separate_sides = False
            self.fill_border_embeddings_fn = self.nn_fill_missing_embeddings
            return

        raise ArgumentError(
            "unknown `graph_decoder_border_embedding_fill` argument value: " + name
        )

    def pad_missing_embeddings(
        self, prev_embeddings_l, prev_embeddings_r, new_embedding_l, new_embedding_r
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns new left and right embeddings with the graph border padding embedding filled.
        The edge encoder can't generate an embedding "incoming" from the borders.

        This version simply adds zeroes to both sides.
        """
        new_embeddings_l = torch.nn.functional.pad(new_embedding_l, (0, 0, 1, 0))
        new_embeddings_r = torch.nn.functional.pad(new_embedding_r, (0, 0, 0, 1))
        return new_embeddings_l, new_embeddings_r

    def nn_fill_missing_embeddings(
        self, prev_embeddings_l, prev_embeddings_r, new_embedding_l, new_embedding_r
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate missing border embeddings from a nn.
        """
        prev_left_border_embedding_l = prev_embeddings_l[:, 0]
        prev_left_border_embedding_r = prev_embeddings_r[:, 0]
        prev_right_border_embedding_l = prev_embeddings_l[:, -1]
        prev_right_border_embedding_r = prev_embeddings_r[:, -1]

        if self.fill_border_separate_sides:
            new_left_border_embedding = self.generate_nn_border_embedding(
                prev_left_border_embedding_l,
                prev_left_border_embedding_r,
                self.border_embedding_nn_l,
            )
            new_right_border_embedding = self.generate_nn_border_embedding(
                prev_right_border_embedding_r,
                prev_right_border_embedding_l,
                self.border_embedding_nn_r,
            )
        else:
            new_left_border_embedding = self.generate_nn_border_embedding(
                prev_left_border_embedding_l,
                prev_left_border_embedding_r,
                self.border_embedding_nn,
            )
            new_right_border_embedding = self.generate_nn_border_embedding(
                prev_right_border_embedding_r,
                prev_right_border_embedding_l,
                self.border_embedding_nn,
            )

        new_embeddings_l = torch.cat(
            (new_embedding_l, new_left_border_embedding[:, None]), dim=1
        )
        new_embeddings_r = torch.cat(
            (new_right_border_embedding[:, None], new_embedding_r), dim=1
        )
        return new_embeddings_l, new_embeddings_r

    def generate_nn_border_embedding(
        self,
        prev_outer_border_embedding,
        prev_inner_border_embedding,
        border_embedding_nn: nn.Module,
    ) -> Tensor:
        prev_border_embedding = torch.cat(
            (prev_outer_border_embedding, prev_inner_border_embedding), dim=-1
        )
        filling_nn_ouptut = border_embedding_nn(prev_border_embedding)
        new_border_embedding, weight = torch.split(
            filling_nn_ouptut,
            (self.internal_embedding_size, self.internal_embedding_size),
            dim=-1,
        )
        new_border_embedding = weighted_average(
            new_border_embedding, prev_outer_border_embedding, weight
        )
        return new_border_embedding

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        try:  # these may collide with an upper autoencoder, but that's fine
            parent_parser = BaseModel.add_model_specific_args(
                parent_parser=parent_parser
            )
        except ArgumentError:
            pass
        parser = parent_parser.add_argument_group(cls.__name__)
        try:  # these may collide with an encoder module, but that's fine
            parser.add_argument(
                "--embedding-size",
                dest="embedding_size",
                default=32,
                type=int,
                metavar="EMB_SIZE",
                help="size of the encoder output graph embedding",
            )
            parser.add_argument(
                "--edge-size",
                dest="edge_size",
                default=1,
                type=int,
                metavar="EDGE_SIZE",
                help="number of dimensions of a graph's edge",
            )
            try:  # may collide with a data module, but that's fine
                parser.add_argument(
                    "--block_size",
                    dest="block_size",
                    default=1,
                    type=int,
                    help="size (width or height) of a block of adjacency matrix edges",
                )
            except ArgumentError:
                pass
            parser.add_argument(
                "--graph_decoder_border_embedding_fill",
                dest="graph_decoder_border_embedding_fill",
                default="separate_sides_nn",
                type=str,
                metavar="METHOD_NAME",
                help="the name of the type of the adjacency matrix border embedding filling method. Available: 'pad', 'separate_sides_nn', 'single_nn'",
            )
            parser.add_argument(
                "--graph_decoder_filling_nn_layer_sizes",
                dest="graph_decoder_filling_nn_layer_sizes",
                default=[256],
                type=parse_layer_sizes_list,
                metavar="EDGE_DECODER_FILL_H_SIZES",
                help="list of the hidden layer sizes of the edge decoder's input embedding filling nn. Applies only to 'separate_sides_nn', 'single_nn'",
            )
            parser.add_argument(
                "--graph_decoder_filling_nn_activation_function",
                dest="graph_decoder_filling_nn_activation_function",
                default="ELU",
                type=str,
                metavar="ACTIVATION_F_NAME",
                help="name of the activation function of the edge decoderr's input embedding filling nn. Applies only to 'separate_sides_nn', 'single_nn'",
            )
        except ArgumentError:
            pass
        return parent_parser


def find_finished_masks(
    masks: Tensor, prev_mask_state: Tensor
) -> Tuple[List[int], Tensor]:
    num_graphs = masks.shape[0]
    num_mask_blocks = masks.shape[1]
    block_size = masks.shape[2]
    num_diagonals_in_block = 2 * block_size - 1

    # The prev_mask_state is a Tensor containing weighted means of the previous masks,
    # but only the ones relevant, i.e. the means of the diagonals after the previous center.
    if prev_mask_state is None:
        # create a state of mean-neutral 0.5s
        num_diagonals_from_prev_mask_relevant_in_curr_mask = int(
            num_diagonals_in_block / 2
        )
        neutral_means = torch.zeros(
            (num_graphs, num_diagonals_from_prev_mask_relevant_in_curr_mask),
            device=masks.device,
        )
        prev_mask_state = neutral_means

    prev_means = prev_mask_state
    absolute_diag_offset = block_size * (num_mask_blocks - 1)

    # calculate block diagonal means
    curr_diag_means = torch.zeros(
        (num_graphs, num_diagonals_in_block),
        device=masks.device,
    )
    center_diag_offset = int(num_diagonals_in_block / 2)
    reduced_dims = tuple(range(1, masks.ndim - 1))
    for diag_offset in range(num_diagonals_in_block):
        t_diag_offset = diag_offset - center_diag_offset
        diag = torch.diagonal(masks, offset=t_diag_offset, dim1=2, dim2=3)
        diag_mean = torch.mean(diag, dim=reduced_dims)
        curr_absolute_diag_offset = absolute_diag_offset + diag_offset
        diag_len = diag.shape[2]
        diag_weight = diag_len * num_mask_blocks / (curr_absolute_diag_offset + 1)
        curr_diag_means[:, diag_offset] = diag_mean * diag_weight

    curr_diag_means[:, :center_diag_offset] += prev_means
    indices_graph_diags_finished = curr_diag_means[:, : center_diag_offset + 1] <= 0.5
    indices_graphs_finished = indices_graph_diags_finished.sum(dim=1) > 0

    curr_mask_state = curr_diag_means[:, center_diag_offset + 1 :]
    curr_mask_state = curr_mask_state[~indices_graphs_finished]

    return (indices_graphs_finished, curr_mask_state)
