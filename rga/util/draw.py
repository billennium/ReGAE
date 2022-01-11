from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from rga.util import adjmatrix


def draw_diag_repr_graph(g, num_nodes: int, name: str):
    g = (
        adjmatrix.diagonal_block_to_adj_matrix_representation(g, num_nodes)
        .cpu()
        .numpy()
    )[:num_nodes, :num_nodes, 0]
    g = np.tril(g, -1)
    g = nx.from_numpy_matrix(g)
    draw_graph(g, name)


def draw_graph(G: nx.Graph, name: str):
    plt.switch_backend("agg")
    plt.axis("off")

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=35, pos=pos)

    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("figures/graph_" + name + ".png", dpi=200)
    plt.close()

    plt.switch_backend("agg")
    G_deg = nx.degree_histogram(G)
    G_deg = np.array(G_deg)
    # plt.plot(range(len(G_deg)), G_deg, 'r', linewidth = 2)
    plt.bar(np.arange(len(G_deg))[G_deg > 0], G_deg[G_deg > 0])
    plt.savefig("figures/degree_view_" + name + ".png", dpi=200)
    plt.close()
