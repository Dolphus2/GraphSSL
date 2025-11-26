"""
Data loading utilities for OGB_MAG dataset
"""
import os
import logging
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
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
    num_neighbors: list = [30]*2,
    batch_size: int = 1024,
    num_workers: int = 4,
    target_node_type: str = "paper"
) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader, NeighborLoader]:
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

    # We need global ids
    global_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, torch.arange(data[target_node_type].num_nodes)),
        num_workers=num_workers,
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
    logger.debug(f"  Global batches: ~{len(global_loader)}")
    logger.debug(f"  Val batches: ~{len(val_loader)}")
    logger.debug(f"  Test batches: ~{len(test_loader)}")
    
    return inductive_train_loader, val_loader, test_loader, global_loader


def create_link_loaders(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    num_neighbors: list = [15, 10],
    batch_size: int = 1024,
    neg_sampling_ratio: float = 1.0,
    num_workers: int = 4,
    split_edges: bool = True,
    seed: int = 42,
    target_node_type: str = "paper"
) -> Tuple[LinkNeighborLoader, LinkNeighborLoader, LinkNeighborLoader, NeighborLoader, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create train, validation, and test LinkNeighborLoaders for link prediction.
    
    Args:
        data: HeteroData object
        target_edge_type: Edge type tuple (src_type, relation, dst_type) for link prediction
        num_neighbors: Number of neighbors to sample at each layer [layer1, layer2, ...]
        batch_size: Batch size for loading (number of edges per batch)
        neg_sampling_ratio: Ratio of negative to positive samples (e.g., 1.0 = 1:1 ratio)
        num_workers: Number of worker processes for data loading
        split_edges: Whether to split edges into train/val/test (80/10/10 split)
        seed: Random seed for reproducible edge splitting
        target_node_type: Target node type for global_loader (default: "paper")
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, global_loader, edge_splits)
        where:
        - train_loader, val_loader, test_loader are LinkNeighborLoaders
        - global_loader is a NeighborLoader (ensures all nodes are included)
        - edge_splits is (train_edge_index, val_edge_index, test_edge_index)
    """
    logger.debug(f"Creating LinkNeighborLoaders with:")
    logger.debug(f"  Target edge type: {target_edge_type}")
    logger.debug(f"  Neighbors per layer: {num_neighbors}")
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug(f"  Negative sampling ratio: {neg_sampling_ratio}")
    
    # Get edge indices for target edge type
    edge_index = data[target_edge_type].edge_index
    num_edges = edge_index.size(1)
    
    # Split edges into train/val/test if requested
    if split_edges:
        train_edge_index, val_edge_index, test_edge_index = create_edge_splits(
            edge_index=edge_index,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed
        )
        logger.info(f"  Edge splits (seed={seed}): train={train_edge_index.size(1)}, "
                   f"val={val_edge_index.size(1)}, test={test_edge_index.size(1)}")
    else:
        # Use all edges for all splits (not recommended for evaluation)
        train_edge_index = edge_index
        val_edge_index = edge_index
        test_edge_index = edge_index
        logger.warning("Using all edges for train/val/test - this will cause data leakage!")
    
    # Create data copies for each split to avoid edge leakage
    # For proper evaluation, we should use only train edges in the graph
    data_train = data.clone()
    
    # Train loader with negative sampling
    train_loader = LinkNeighborLoader(
        data_train,
        num_neighbors=num_neighbors,
        edge_label_index=(target_edge_type, train_edge_index),
        edge_label=torch.ones(train_edge_index.size(1)),
        neg_sampling_ratio=neg_sampling_ratio,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # We need global ids - use NeighborLoader to ensure ALL nodes are included
    # This is important for downstream evaluation which needs embeddings for all nodes
    global_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, torch.arange(data[target_node_type].num_nodes)),
        num_workers=num_workers,
    )
    
    # Validation loader with negative sampling
    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=(target_edge_type, val_edge_index),
        edge_label=torch.ones(val_edge_index.size(1)),
        neg_sampling_ratio=neg_sampling_ratio,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    # Test loader with negative sampling
    test_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=(target_edge_type, test_edge_index),
        edge_label=torch.ones(test_edge_index.size(1)),
        neg_sampling_ratio=neg_sampling_ratio,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    logger.info("Link prediction loaders created successfully!")
    logger.debug(f"  Train batches: ~{len(train_loader)}")
    logger.debug(f"  Global batches: ~{len(global_loader)}")
    logger.debug(f"  Val batches: ~{len(val_loader)}")
    logger.debug(f"  Test batches: ~{len(test_loader)}")
    
    # Return loaders and edge splits for downstream evaluation
    edge_splits = (train_edge_index, val_edge_index, test_edge_index)
    return train_loader, val_loader, test_loader, global_loader, edge_splits


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


def create_edge_splits(
    edge_index: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split edge indices into train/val/test sets.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        train_ratio: Ratio of edges for training (default: 0.8)
        val_ratio: Ratio of edges for validation (default: 0.1)
        seed: Random seed for reproducible splitting
    
    Returns:
        Tuple of (train_edge_index, val_edge_index, test_edge_index)
    """
    num_edges = edge_index.size(1)
    
    # Shuffle edges with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=generator)
    edge_index_shuffled = edge_index[:, perm]
    
    # Calculate split sizes
    train_size = int(train_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    
    # Split edges
    train_edge_index = edge_index_shuffled[:, :train_size]
    val_edge_index = edge_index_shuffled[:, train_size:train_size + val_size]
    test_edge_index = edge_index_shuffled[:, train_size + val_size:]
    
    logger.debug(f"Split {num_edges} edges (seed={seed}): "
                f"train={train_edge_index.size(1)}, "
                f"val={val_edge_index.size(1)}, "
                f"test={test_edge_index.size(1)}")
    
    return train_edge_index, val_edge_index, test_edge_index


def subsample_dataset(
    data: HeteroData,
    target_node_type: str = "paper",
    max_nodes: int = 5000,
    seed: int = 42
) -> HeteroData:
    """
    Subsample a heterogeneous graph dataset for faster testing.
    
    This function randomly samples a subset of nodes and keeps only edges
    between sampled nodes. Useful for quick testing on large datasets.
    
    Args:
        data: HeteroData object to subsample
        target_node_type: The main node type to subsample (e.g., "paper")
        max_nodes: Maximum number of nodes to keep for target_node_type
        seed: Random seed for reproducible subsampling
    
    Returns:
        Subsampled HeteroData object
    """
    logger.info(f"Subsampling dataset: keeping max {max_nodes} {target_node_type} nodes")
    logger.info(f"Original {target_node_type} nodes: {data[target_node_type].num_nodes}")
    
    num_nodes = data[target_node_type].num_nodes
    if num_nodes <= max_nodes:
        logger.info(f"Dataset already small enough, skipping subsampling")
        return data
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Sample node indices
    perm = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    sampled_indices = perm[:max_nodes].sort()[0]  # Sort to maintain some structure
    
    # Create mapping from old indices to new indices
    node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=sampled_indices.device)
    node_mapping[sampled_indices] = torch.arange(max_nodes, device=sampled_indices.device)
    
    # Create new data object
    data_sampled = data.clone()
    
    # Subsample target node type
    data_sampled[target_node_type].x = data[target_node_type].x[sampled_indices]
    if hasattr(data[target_node_type], 'y') and data[target_node_type].y is not None:
        data_sampled[target_node_type].y = data[target_node_type].y[sampled_indices]
    if hasattr(data[target_node_type], 'train_mask') and data[target_node_type].train_mask is not None:
        data_sampled[target_node_type].train_mask = data[target_node_type].train_mask[sampled_indices]
    if hasattr(data[target_node_type], 'val_mask') and data[target_node_type].val_mask is not None:
        data_sampled[target_node_type].val_mask = data[target_node_type].val_mask[sampled_indices]
    if hasattr(data[target_node_type], 'test_mask') and data[target_node_type].test_mask is not None:
        data_sampled[target_node_type].test_mask = data[target_node_type].test_mask[sampled_indices]
    if hasattr(data[target_node_type], 'year') and data[target_node_type].year is not None:
        data_sampled[target_node_type].year = data[target_node_type].year[sampled_indices]
    
    # Filter edges to only include sampled nodes
    edge_types = list(data.edge_index_dict.keys())
    for edge_type in edge_types:
        edge_index = data[edge_type].edge_index
        
        # Check if this edge type involves the target node type
        if edge_type[0] == target_node_type or edge_type[-1] == target_node_type:
            # Ensure node_mapping is on the same device as edge_index
            node_mapping_on_device = node_mapping.to(edge_index.device)
            
            # Create masks for valid edges
            if edge_type[0] == target_node_type:
                src_mask = node_mapping_on_device[edge_index[0]] != -1
            else:
                src_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            
            if edge_type[-1] == target_node_type:
                dst_mask = node_mapping_on_device[edge_index[1]] != -1
            else:
                dst_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            
            edge_mask = src_mask & dst_mask
            filtered_edge_index = edge_index[:, edge_mask]
            
            # Remap node indices
            if edge_type[0] == target_node_type:
                filtered_edge_index[0] = node_mapping_on_device[filtered_edge_index[0]]
            if edge_type[-1] == target_node_type:
                filtered_edge_index[1] = node_mapping_on_device[filtered_edge_index[1]]
            
            data_sampled[edge_type].edge_index = filtered_edge_index
        # For edge types not involving target_node_type, keep all edges
        # (This is a simplification - in practice you might want to subsample other node types too)
    
    logger.info(f"Subsampled {target_node_type} nodes: {data_sampled[target_node_type].num_nodes}")
    logger.info(f"Train samples: {data_sampled[target_node_type].train_mask.sum().item()}")
    logger.info(f"Val samples: {data_sampled[target_node_type].val_mask.sum().item()}")
    logger.info(f"Test samples: {data_sampled[target_node_type].test_mask.sum().item()}")
    
    return data_sampled


def to_inductive(data: HeteroData, node_type: str) -> HeteroData:
    """
    Convert a heterogeneous graph into an inductive view by dropping validation/test
    nodes and any edges touching them. The original node indices are remapped so the
    returned graph only contains training nodes, making it safe for inductive loaders.
    Note: This mutates the provided `data` object (val/test masks become empty).
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
