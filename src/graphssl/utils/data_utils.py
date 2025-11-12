"""
Data loading utilities for OGB_MAG dataset
"""
import os
import logging
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def load_ogb_mag(root_path: str, preprocess: str = "metapath2vec") -> HeteroData:
    """
    Load the OGB_MAG dataset.
    
    Args:
        root_path: Path to save the dataset
        preprocess: Preprocessing method for node embeddings ("metapath2vec" or "transe")
    
    Returns:
        HeteroData object containing the graph
    """
    logger.info(f"Loading OGB_MAG dataset from {root_path}")
    logger.info(f"Using preprocessing method: {preprocess}")
    
    # Load dataset with preprocessing
    transform = T.ToUndirected(merge=True)
    dataset = OGB_MAG(root=root_path, preprocess=preprocess, transform=transform)
    data = dataset[0]
    
    logger.info("Dataset loaded successfully!")
    logger.debug(f"Node types: {data.node_types}")
    logger.debug(f"Edge types: {data.edge_types}")
    
    # Log statistics for paper nodes (target)
    logger.info("Paper node statistics:")
    logger.info(f"  Number of papers: {data['paper'].num_nodes}")
    logger.info(f"  Feature dimension: {data['paper'].x.shape[1]}")
    logger.info(f"  Number of venues (classes): {data['paper'].y.max().item() + 1}")
    logger.debug(f"  Train samples: {data['paper'].train_mask.sum().item()}")
    logger.debug(f"  Val samples: {data['paper'].val_mask.sum().item()}")
    logger.debug(f"  Test samples: {data['paper'].test_mask.sum().item()}")
    
    return data


def create_neighbor_loaders(
    data: HeteroData,
    num_neighbors: list = [15, 10],
    batch_size: int = 1024,
    num_workers: int = 4,
    target_node_type: str = "paper"
) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
    """
    Create train, validation, and test NeighborLoaders for heterogeneous graphs.
    
    Args:
        data: HeteroData object
        num_neighbors: Number of neighbors to sample at each layer [layer1, layer2, ...]
        batch_size: Batch size for loading
        num_workers: Number of worker processes for data loading
        target_node_type: The node type we're making predictions for
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.debug(f"Creating NeighborLoaders with:")
    logger.debug(f"  Neighbors per layer: {num_neighbors}")
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug(f"  Target node type: {target_node_type}")
    
    # Create inductive train dataset
    data_inductive = to_inductive(data.clone(), target_node_type)

    # Inductive training loader - only sample from training nodes with test and val nodes removed
    # Note: data_inductive already has only train nodes, so use its train_mask
    inductive_train_loader = NeighborLoader(
        data_inductive,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, data_inductive[target_node_type].train_mask),
        num_workers=num_workers,
        shuffle=True,
    )

    # Transductive training loader - only sample from training nodes
    transductive_train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, data[target_node_type].train_mask),
        num_workers=num_workers,
        shuffle=True,
    )
    
    # Validation loader
    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, data[target_node_type].val_mask),
        num_workers=num_workers,
        shuffle=False,
    )
    
    # Test loader
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, data[target_node_type].test_mask),
        num_workers=num_workers,
        shuffle=False,
    )
    
    logger.info("Loaders created successfully!")
    logger.debug(f"  Inductive Train batches: ~{len(inductive_train_loader)}")
    logger.debug(f"  Train batches: ~{len(transductive_train_loader)}")
    logger.debug(f"  Val batches: ~{len(val_loader)}")
    logger.debug(f"  Test batches: ~{len(test_loader)}")
    
    return inductive_train_loader, transductive_train_loader, val_loader, test_loader


def get_dataset_info(data: HeteroData, target_node_type: str = "paper") -> Dict:
    """
    Extract key information from the dataset.
    
    Args:
        data: HeteroData object
        target_node_type: The node type we're making predictions for
    
    Returns:
        Dictionary containing dataset information
    """
    info = {
        "node_types": data.node_types,
        "edge_types": data.edge_types,
        "num_classes": int(data[target_node_type].y.max().item() + 1),
        "num_features": {},
        "num_nodes": {},
    }
    
    # Get feature dimensions for each node type
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            info["num_features"][node_type] = data[node_type].x.shape[1]
        info["num_nodes"][node_type] = data[node_type].num_nodes
    
    return info

def to_inductive(data: HeteroData, node_type: str) -> HeteroData:
    """
    A function that removes all val/test node features and edges between train nodes and val/test nodes.

    """
    train_mask = data[node_type].train_mask
    train_mask_idxs = torch.where(train_mask)[0]
    N_train = len(train_mask_idxs)

    # define new edge index
    new_paper_idxs = torch.full((len(train_mask),), -1, device=train_mask.device)
    new_paper_idxs[train_mask] = torch.arange(N_train, device=train_mask.device)

    # restrict node_type to only include train split
    data[node_type].x = data[node_type].x[train_mask]
    data[node_type].y = data[node_type].y[train_mask]
    data[node_type].year = data[node_type].year[train_mask]
    data[node_type].train_mask = torch.ones((N_train), dtype=torch.bool, device=train_mask.device)
    data[node_type].val_mask = torch.zeros((N_train), dtype=torch.bool, device=train_mask.device)
    data[node_type].test_mask = torch.zeros((N_train), dtype=torch.bool, device=train_mask.device)
    # From here there is no way to recover the val and test sets. Indexes have been reset. 
    # This is not necessary. val_mask and test_mask could have been adjusted accordingly.

    # find edges with node_type as either source or destination
    edge_types = list(data.edge_index_dict.keys())
    edge_type_mask = [(e[0] == node_type, e[-1] == node_type) for e in edge_types]

    edge_index_dict = data.edge_index_dict
    
    for i, edge_type in enumerate(edge_types):
        if not any(edge_type_mask[i]):
            continue
        
        edge_index = edge_index_dict[edge_type]
        src_mask = torch.ones((edge_index.size(1)), dtype=bool)
        dst_mask = torch.ones((edge_index.size(1)), dtype=bool)

        # mask paper nodes in edge index not part of train
        if edge_type[0] == node_type:
            src_mask = new_paper_idxs[edge_index[0]] != -1

        if edge_type[-1] == node_type:
            dst_mask = new_paper_idxs[edge_index[1]] != -1
        
        edge_mask = src_mask & dst_mask
        filtered_edge_index = edge_index[:, edge_mask]
        
        if edge_type[0] == node_type:
            filtered_edge_index[0] = new_paper_idxs[filtered_edge_index[0]]
        
        if edge_type[-1] == node_type:
            filtered_edge_index[1] = new_paper_idxs[filtered_edge_index[1]]
        
        data[edge_type]['edge_index'] = filtered_edge_index

    return data
