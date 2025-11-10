from typing import Tuple
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG

def load_mag_hetero(root: str = "../data/ogb"):
    """
    Returns (data, y, split_idx, train_idx, valid_idx, test_idx)
    - data: object with x_dict, edge_index_dict, y_dict
    - y:    labels tensor for paper nodes (shape [N_paper])
    - split_idx: raw OGB split dict (for reference)
    """
    transform = T.ToUndirected(merge=True)
    ds = OGB_MAG(root, preprocess='metapath2vec', transform=transform)

    data = ds[0]
    y = data["paper"].y.view(-1)

    train_input_nodes = ('paper', data['paper'].train_mask)
    test_input_nodes = ('paper', data['paper'].test_mask)
    val_input_nodes = ('paper', data['paper'].val_mask)
    return ds, y, train_input_nodes, val_input_nodes, test_input_nodes

if __name__ == "__main__":
    dataset, y, train_input_nodes, val_input_nodes, test_input_nodes = load_mag_hetero()
    data = dataset[0]
    num_classes = dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(data)