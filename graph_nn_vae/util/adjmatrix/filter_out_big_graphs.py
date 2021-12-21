def filter_out_big_graphs(graphs, labels, max_size):
    res_graphs = []
    res_labels = [] if labels is not None else None

    for el in zip(graphs, labels) if labels is not None else graphs:
        graph = el[0] if labels is not None else el
        label = el[1] if labels is not None else None
        if graph.shape[0] < max_size:
            res_graphs.append(graph)
            if labels is not None:
                res_labels.append(label)

    return res_graphs, res_labels
