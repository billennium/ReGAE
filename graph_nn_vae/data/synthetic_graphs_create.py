import networkx as nx

# loads or generates graphs of type specified by the config.graph_type
# returns a graph list and an appropriate max_prev_nodes for nn training
def create_synthetic_graphs(graph_type: str) -> list[nx.Graph]:
    graph_generator_func = {
        "ladder": ladder_graphs,
        "ladder_small": ladder_small_graphs,
        "tree": tree_graphs,
        "caveman": caveman_graphs,
        "caveman_small": caveman_small_graphs,
        "community": community_graphs,
        "grid": grid_graphs,
        "grid_small": grid_small_graphs,
        "grid_medium": grid_medium_graphs,
        "grid_big": grid_big_graphs,
        "barabasi": barabasi_graphs,
        "barabasi_small": barabasi_small_graphs,
    }.get(graph_type, None)
    if graph_generator_func is None:
        raise ValueError(f'unknown graph type: "{graph_type}"')

    return graph_generator_func()


def ladder_graphs() -> list[nx.Graph]:
    graphs = []
    for num_nodes in range(100, 201):
        graphs.append(nx.ladder_graph(num_nodes))
    return graphs


def ladder_small_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(2, 11):
        graphs.append(nx.ladder_graph(i))
    return graphs


def tree_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(2, 5):
        for j in range(3, 5):
            graphs.append(nx.balanced_tree(i, j))
    return graphs


def caveman_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(2, 3):
        for j in range(6, 11):
            graphs.append(nx.relaxed_caveman_graph(i, j, p=0.3))
            # TODO: check this out
            # graphs.append(caveman_special(i,j, p_edge=0.3))
    return graphs


def caveman_small_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(2, 3):
        for j in range(30, 81):
            graphs.append(nx.relaxed_caveman_graph(i, j, p=0.8))
            # TODO: check this out
            # graphs.append(caveman_special(i,j, p_edge=0.3))
    return graphs


def community_graphs() -> list[nx.Graph]:
    # TODO:
    raise NotImplementedError("community type graphs are yet to be implemented")


def grid_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(10, 20):
        for j in range(10, 20):
            graphs.append(nx.grid_2d_graph(i, j))
    return graphs


def grid_big_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(36, 46):
        for j in range(36, 46):
            graphs.append(nx.grid_2d_graph(i, j))
    return graphs


def grid_medium_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(2, 9):
        for j in range(2, 9):
            graphs.append(nx.grid_2d_graph(i, j))
    return graphs


def grid_small_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(2, 5):
        for j in range(2, 5):
            graphs.append(nx.grid_2d_graph(i, j))
    return graphs


def barabasi_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(100, 200):
        for j in range(4, 5):
            for k in range(5):
                graphs.append(nx.barabasi_albert_graph(i, j))
    return graphs


def barabasi_small_graphs() -> list[nx.Graph]:
    graphs = []
    for i in range(4, 21):
        for j in range(3, 4):
            for k in range(10):
                graphs.append(nx.barabasi_albert_graph(i, j))
    return graphs
