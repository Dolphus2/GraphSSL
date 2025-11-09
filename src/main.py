if __name__ == '__main__':
    from torch_geometric.datasets import OGB_MAG
    root_path = ""
    transform = ["to_undirected"] # insert preprocessing steps that should be applied to the data. It is common to include reverse edges.
    preprocess = "transe" # specify how to obtain initial embeddings for nodes ("transe", "metapath2vec") are some options.
    dataset = OGB_MAG(root=root_path, preprocess=preprocess, transform=transform)

    node_type = "paper" # target node type
    data_inductive = to_inductive(dataset.clone(), node_type)

    # ... specify dataloader ...