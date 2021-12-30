import numpy as np
from collections import OrderedDict


def print_dataset_statistics(graphs_data, name, use_labels=False):
    if use_labels:
        graphs = [el[0] for el in graphs_data]
        labels = [el[1] for el in graphs_data]
    else:
        graphs = graphs_data

    graph_information = {"node count": [], "edge count": [], "filling fraction": []}
    for graph in graphs:
        edge_count = graph.sum() / 2
        node_count = graph.shape[0]
        graph_information["node count"].append(node_count)
        graph_information["edge count"].append(edge_count)
        graph_information["filling fraction"].append(
            (edge_count * 2) / ((node_count * node_count - node_count))
        )

    print("Statistic of set: ", name)

    stats = OrderedDict()

    stats["Dataset size"] = len(graphs_data)
    stats["Labels"] = use_labels

    for stat_name, stat_value in graph_information.items():
        for foo_name, foo in [("Min", np.min), ("Average", np.mean), ("Max", np.max)]:
            stats[(foo_name + " " + stat_name)] = np.round(foo(stat_value), 2)

    if use_labels:
        labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(labels, counts):
            stats['Label "' + str(label) + '" count'] = count

    for name, value in stats.items():
        print(name.rjust(25), ":", value)

    print("-" * 64)
