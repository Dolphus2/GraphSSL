"""
Data loading utilities for OGB_MAG dataset
"""
import os
import torch
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from typing import Tuple, Dict


def load_ogb_mag(root_path: str, preprocess: str = "metapath2vec") -> HeteroData:
    """
    Load the OGB_MAG dataset.
    
    Args:
        root_path: Path to save the dataset
        preprocess: Preprocessing method for node embeddings ("metapath2vec" or "transe")
    
    Returns:
        HeteroData object containing the graph
    """
    print(f"Loading OGB_MAG dataset from {root_path}")
    print(f"Using preprocessing method: {preprocess}")
    
    # Load dataset with preprocessing
    dataset = OGB_MAG(root=root_path, preprocess=preprocess)
    data = dataset[0]
    
    print(f"\nDataset loaded successfully!")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    
    # Print statistics for paper nodes (target)
    print(f"\nPaper node statistics:")
    print(f"  Number of papers: {data['paper'].num_nodes}")
    print(f"  Feature dimension: {data['paper'].x.shape[1]}")
    print(f"  Number of venues (classes): {data['paper'].y.max().item() + 1}")
    print(f"  Train samples: {data['paper'].train_mask.sum().item()}")
    print(f"  Val samples: {data['paper'].val_mask.sum().item()}")
    print(f"  Test samples: {data['paper'].test_mask.sum().item()}")
    
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
    print(f"\nCreating NeighborLoaders with:")
    print(f"  Neighbors per layer: {num_neighbors}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target node type: {target_node_type}")
    
    # Training loader - only sample from training nodes
    train_loader = NeighborLoader(
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
    
    print(f"Loaders created successfully!")
    print(f"  Train batches: ~{len(train_loader)}")
    print(f"  Val batches: ~{len(val_loader)}")
    print(f"  Test batches: ~{len(test_loader)}")
    
    return train_loader, val_loader, test_loader


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
