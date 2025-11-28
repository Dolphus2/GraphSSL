"""
Data loading utilities for OGB_MAG dataset
"""
import os
import logging
from pathlib import Path
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.data import HeteroData
from typing import Tuple, Dict
from graphssl.utils.training_utils import extract_embeddings

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
    
    # Load dataset with preprocessing (without ToUndirected - will be applied after edge splits)
    dataset = OGB_MAG(root=root_path, preprocess=preprocess)
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
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    full_data: HeteroData,
    num_neighbors: list = [30]*2,
    batch_size: int = 1024,
    num_workers: int = 4,
    target_node_type: str = "paper"
) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader, NeighborLoader]:
    """
    Create train, validation, and test NeighborLoaders for heterogeneous graphs.
    
    Args:
        train_data: Train split HeteroData object
        val_data: Validation split HeteroData object
        test_data: Test split HeteroData object
        full_data: Full HeteroData object (for global loader)
        num_neighbors: Number of neighbors to sample at each layer [layer1, layer2, ...]
        batch_size: Batch size for loading
        num_workers: Number of worker processes for data loading
        target_node_type: The node type we're making predictions for
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, global_loader)
    """
    logger.debug(f"Creating NeighborLoaders with:")
    logger.debug(f"  Neighbors per layer: {num_neighbors}")
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug(f"  Target node type: {target_node_type}")

    train_loader = NeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, train_data[target_node_type].train_mask),
        num_workers=num_workers,
        shuffle=True,
    )
    
    val_loader = NeighborLoader(
        val_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, val_data[target_node_type].val_mask),
        num_workers=num_workers,
        shuffle=False,
    )
    
    test_loader = NeighborLoader(
        test_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, test_data[target_node_type].test_mask),
        num_workers=num_workers,
        shuffle=False,
    )

    global_loader = NeighborLoader(
        full_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, torch.arange(full_data[target_node_type].num_nodes)),
        num_workers=num_workers,
    )
    
    logger.info("NeighborLoaders created successfully!")
    logger.debug(f"  Train batches: ~{len(train_loader)}")
    logger.debug(f"  Val batches: ~{len(val_loader)}")
    logger.debug(f"  Test batches: ~{len(test_loader)}")
    logger.debug(f"  Global batches: ~{len(global_loader)}")
    
    return train_loader, val_loader, test_loader, global_loader


def create_link_loaders(
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    train_edge_index: torch.Tensor,
    val_edge_index: torch.Tensor,
    test_edge_index: torch.Tensor,
    target_edge_type: Tuple[str, str, str],
    num_neighbors: list = [15, 10],
    batch_size: int = 1024,
    neg_sampling_ratio: float = 1.0,
    num_workers: int = 4
) -> Tuple[LinkNeighborLoader, LinkNeighborLoader, LinkNeighborLoader]:
    """
    Create train, validation, and test LinkNeighborLoaders for link prediction.
    
    Args:
        train_data: Train split HeteroData object
        val_data: Validation split HeteroData object
        test_data: Test split HeteroData object
        train_edge_index: Train edge indices
        val_edge_index: Validation edge indices
        test_edge_index: Test edge indices
        target_edge_type: Edge type tuple (src_type, relation, dst_type) for link prediction
        num_neighbors: Number of neighbors to sample at each layer
        batch_size: Batch size for loading (number of edges per batch)
        neg_sampling_ratio: Ratio of negative to positive samples
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.debug(f"Creating LinkNeighborLoaders with:")
    logger.debug(f"  Target edge type: {target_edge_type}")
    logger.debug(f"  Neighbors per layer: {num_neighbors}")
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug(f"  Negative sampling ratio: {neg_sampling_ratio}")
    logger.debug(f"  Train edges: {train_edge_index.size(1)}, Val edges: {val_edge_index.size(1)}, Test edges: {test_edge_index.size(1)}")
    
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        edge_label_index=(target_edge_type, train_edge_index),
        edge_label=torch.ones(train_edge_index.size(1)),
        neg_sampling_ratio=neg_sampling_ratio,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = LinkNeighborLoader(
        val_data,
        num_neighbors=num_neighbors,
        edge_label_index=(target_edge_type, val_edge_index),
        edge_label=torch.ones(val_edge_index.size(1)),
        neg_sampling_ratio=neg_sampling_ratio,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=num_neighbors,
        edge_label_index=(target_edge_type, test_edge_index),
        edge_label=torch.ones(test_edge_index.size(1)),
        neg_sampling_ratio=neg_sampling_ratio,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    logger.info("LinkNeighborLoaders created successfully!")
    logger.debug(f"  Train batches: ~{len(train_loader)}")
    logger.debug(f"  Val batches: ~{len(val_loader)}")
    logger.debug(f"  Test batches: ~{len(test_loader)}")
    
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


def get_reverse_edge_type(edge_type: Tuple[str, str, str]) -> Tuple[str, str, str] | None:
    """
    Determine the reverse edge type for a given edge type.
    
    Args:
        edge_type: Edge type tuple (src_type, relation, dst_type)
    
    Returns:
        Reverse edge type tuple, or None if src_type == dst_type (self-loop edges)
    """
    src_type, relation, dst_type = edge_type
    
    # If source and destination are the same, no reverse edge
    if src_type == dst_type:
        return None
    
    # Construct reverse edge type with "rev_" prefix
    rev_relation = f"rev_{relation}"
    return (dst_type, rev_relation, src_type)


def create_edge_splits(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    seed: int = 42,
    node_inductive: bool = False,
    target_node_type: str = "paper",
    dependent: bool = True,
    train_edge_msg_pass_prop: float = 0.7,
    val_edge_msg_pass_prop: float = 0.7,
    test_edge_msg_pass_prop: float = 0.7,
) -> Tuple[HeteroData, HeteroData, HeteroData, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split edge indices into train/val/test sets using RandomLinkSplit.
    
    Args:
        data: HeteroData object
        target_edge_type: Edge type tuple (src_type, relation, dst_type)
        seed: Random seed for reproducible splitting
        node_inductive: If True, apply inductive transformation (remove val/test nodes from train/val splits)
        target_node_type: Node type for inductive transformation (only used if node_inductive=True)
        dependent: If True, use dependent edge splits (edges incident to split nodes are controlled separately)
        train_edge_msg_pass_prop: Proportion of train-incident edges to keep for message passing (only used if dependent=True)
        val_edge_msg_pass_prop: Proportion of val-incident edges to keep for message passing (only used if dependent=True)
        test_edge_msg_pass_prop: Proportion of test-incident edges to keep for message passing (only used if dependent=True)
    
    Returns:
        Tuple of (train_data, val_data, test_data, train_edge_index, val_edge_index, test_edge_index)
    """
    torch.manual_seed(seed)
    
    if not dependent:
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=0.0,
            add_negative_train_samples=False,
            edge_types=target_edge_type,
            rev_edge_types=None,  # No reverse edges exist yet
        )
        train_data, val_data, test_data = transform(data.clone())
        # Independent edge splits: apply node inductive transformation if requested
        if node_inductive:
            logger.debug(f"Applying inductive transformation to edge splits (target_node_type={target_node_type})")
            train_data = to_inductive(train_data, target_node_type)
            val_data = val_to_inductive(val_data, target_node_type, seed=seed)
        
        train_edge_index = train_data[target_edge_type].edge_label_index
        val_edge_index = val_data[target_edge_type].edge_label_index
        test_edge_index = test_data[target_edge_type].edge_label_index
    else:
        # Dependent edge splits: apply to_inductive transformations and split edges
        logger.debug(f"Using dependent edge splits with message passing proportions: "
                    f"train={train_edge_msg_pass_prop}, val={val_edge_msg_pass_prop}, test={test_edge_msg_pass_prop}")
        
        train_data_inductive = to_inductive(data.clone(), target_node_type)
        
        val_data_inductive = val_to_inductive(data.clone(), target_node_type, seed=seed)
        
        # Test data keeps all nodes (no inductive transformation)
        test_data_inductive = data.clone()
        
        # Split edges incident to train nodes from train_data
        train_data_inductive, train_edge_index = split_edges(
            train_data_inductive,
            target_edge_type,
            split="train",
            edge_msg_pass_prop=train_edge_msg_pass_prop,
            seed=seed
        )
        
        # Split edges incident to val nodes from val_data
        val_data_inductive, val_edge_index = split_edges(
            val_data_inductive,
            target_edge_type,
            split="val",
            edge_msg_pass_prop=val_edge_msg_pass_prop,
            seed=seed
        )
        
        # Split edges incident to test nodes from test_data
        test_data_inductive, test_edge_index = split_edges(
            test_data_inductive,
            target_edge_type,
            split="test",
            edge_msg_pass_prop=test_edge_msg_pass_prop,
            seed=seed
        )
        
        train_data = train_data_inductive
        val_data = val_data_inductive
        test_data = test_data_inductive
    
    # Apply ToUndirected transform after edge splits to create reverse edges
    logger.debug("Applying ToUndirected transform to create reverse edges after splitting")
    to_undirected = T.ToUndirected(merge=True)
    train_data = to_undirected(train_data)
    val_data = to_undirected(val_data)
    test_data = to_undirected(test_data)
    
    logger.debug(f"Split edges (seed={seed}, dependent={dependent}, node_inductive={node_inductive}): "
                f"train={train_edge_index.size(1)}, "
                f"val={val_edge_index.size(1)}, "
                f"test={test_edge_index.size(1)}")
    
    return train_data, val_data, test_data, train_edge_index, val_edge_index, test_edge_index


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


def _remap_and_filter_edges(data: HeteroData, node_type: str, node_mapping: torch.Tensor) -> None:
    """Apply node remapping to edges and filter out invalid edges."""
    edge_types = [et for et in data.edge_index_dict.keys() 
                  if et[0] == node_type or et[-1] == node_type]
    
    for edge_type in edge_types:
        edge_index = data.edge_index_dict[edge_type]
        
        src_mask = node_mapping[edge_index[0]] != -1 if edge_type[0] == node_type else torch.ones(edge_index.size(1), dtype=torch.bool)
        dst_mask = node_mapping[edge_index[1]] != -1 if edge_type[-1] == node_type else torch.ones(edge_index.size(1), dtype=torch.bool)
        edge_mask = src_mask & dst_mask
        
        filtered_edge_index = edge_index[:, edge_mask]
        
        if edge_type[0] == node_type:
            filtered_edge_index[0] = node_mapping[filtered_edge_index[0]]
        if edge_type[-1] == node_type:
            filtered_edge_index[1] = node_mapping[filtered_edge_index[1]]
        
        data[edge_type].edge_index = filtered_edge_index
        
        if hasattr(data[edge_type], 'edge_label_index') and data[edge_type].edge_label_index is not None:
            edge_label_index = data[edge_type].edge_label_index
            
            label_src_mask = node_mapping[edge_label_index[0]] != -1 if edge_type[0] == node_type else torch.ones(edge_label_index.size(1), dtype=torch.bool)
            label_dst_mask = node_mapping[edge_label_index[1]] != -1 if edge_type[-1] == node_type else torch.ones(edge_label_index.size(1), dtype=torch.bool)
            label_edge_mask = label_src_mask & label_dst_mask
            
            filtered_edge_label_index = edge_label_index[:, label_edge_mask]
            
            if edge_type[0] == node_type:
                filtered_edge_label_index[0] = node_mapping[filtered_edge_label_index[0]]
            if edge_type[-1] == node_type:
                filtered_edge_label_index[1] = node_mapping[filtered_edge_label_index[1]]
            
            data[edge_type].edge_label_index = filtered_edge_label_index
            
            if hasattr(data[edge_type], 'edge_label') and data[edge_type].edge_label is not None:
                data[edge_type].edge_label = data[edge_type].edge_label[label_edge_mask]


def to_inductive(data: HeteroData, node_type: str) -> HeteroData:
    """Keep only train nodes, remap indices sequentially."""
    train_mask = data[node_type].train_mask.clone()
    N_train = train_mask.sum().item()
    
    node_mapping = torch.full((len(train_mask),), -1, device=train_mask.device)
    node_mapping[train_mask] = torch.arange(N_train, device=train_mask.device)
    
    data[node_type].x = data[node_type].x[train_mask]
    data[node_type].y = data[node_type].y[train_mask]
    data[node_type].year = data[node_type].year[train_mask]
    data[node_type].train_mask = torch.ones(N_train, dtype=torch.bool, device=train_mask.device)
    data[node_type].val_mask = torch.zeros(N_train, dtype=torch.bool, device=train_mask.device)
    data[node_type].test_mask = torch.zeros(N_train, dtype=torch.bool, device=train_mask.device)
    
    _remap_and_filter_edges(data, node_type, node_mapping)
    
    return data


def val_to_inductive(data: HeteroData, node_type: str, seed: int = 42) -> HeteroData:
    """Keep train+val nodes, remap indices randomly."""
    train_mask = data[node_type].train_mask.clone()
    val_mask = data[node_type].val_mask.clone()
    N_train = train_mask.sum().item()
    N_val = val_mask.sum().item()
    
    node_mapping = torch.full((len(train_mask),), -1, device=train_mask.device)
    perm = torch.randperm(N_train + N_val, generator=torch.Generator().manual_seed(seed), device=train_mask.device)
    node_mapping[train_mask] = perm[:N_train]
    node_mapping[val_mask] = perm[N_train:]
    
    combined_mask = train_mask | val_mask
    data[node_type].x = data[node_type].x[combined_mask]
    data[node_type].y = data[node_type].y[combined_mask]
    data[node_type].year = data[node_type].year[combined_mask]
    data[node_type].train_mask = torch.zeros(N_train + N_val, dtype=torch.bool, device=train_mask.device).scatter_(0, perm[:N_train], True)
    data[node_type].val_mask = torch.zeros(N_train + N_val, dtype=torch.bool, device=train_mask.device).scatter_(0, perm[N_train:], True)
    data[node_type].test_mask = torch.zeros(N_train + N_val, dtype=torch.bool, device=train_mask.device)
    
    _remap_and_filter_edges(data, node_type, node_mapping)
    
    return data


def split_edges(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    split: str,
    edge_msg_pass_prop: float,
    seed: int = 42
) -> Tuple[HeteroData, torch.Tensor]:
    """
    Split edges incident to train/val/test nodes for message passing control.
    
    This is a post-processing step that finds all edges incident to nodes in a specific
    split (train/val/test) and randomly removes (1 - edge_msg_pass_prop) of them from
    the graph for message passing, storing them separately.
    
    Args:
        data: HeteroData object
        target_edge_type: Edge type tuple (src_type, relation, dst_type)
        split: Which split to target ('train', 'val', or 'test')
        edge_msg_pass_prop: Proportion of edges to keep for message passing (0.0 to 1.0)
        seed: Random seed for reproducible splitting
    
    Returns:
        Tuple of (data, split_edge_index) where:
        - data: Modified HeteroData with edges removed
        - split_edge_index: Edges removed from data
    """
    torch.manual_seed(seed)
    
    # Get the source and destination node types from the edge type
    src_type, _, dst_type = target_edge_type
    mask_attr = f"{split}_mask"
    
    split_mask_src = getattr(data[src_type], mask_attr, None) if hasattr(data[src_type], mask_attr) else None
    split_mask_dst = getattr(data[dst_type], mask_attr, None) if hasattr(data[dst_type], mask_attr) else None
    
    edge_index = data[target_edge_type].edge_index
    device = edge_index.device
    
    # Find edges incident to nodes in the specified split
    # An edge is incident if either source or destination is in the split
    incident_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
    if split_mask_src is not None:
        incident_mask |= split_mask_src[edge_index[0]]
    if split_mask_dst is not None:
        incident_mask |= split_mask_dst[edge_index[1]]
    
    # Get indices of edges incident to split nodes
    incident_indices = incident_mask.nonzero(as_tuple=True)[0]
    num_incident = incident_indices.size(0)
    
    if num_incident == 0:
        logger.warning(f"No edges incident to {split} nodes found for {target_edge_type}")
        return data, torch.empty((2, 0), dtype=torch.long, device=device)
    
    # Randomly shuffle incident edges
    perm = torch.randperm(num_incident, generator=torch.Generator().manual_seed(seed), device=device)
    shuffled_incident_indices = incident_indices[perm]
    
    num_msg_pass = int(num_incident * edge_msg_pass_prop)
    num_split = num_incident - num_msg_pass

    # Split indices: first num_msg_pass stay, rest are removed
    split_indices = shuffled_incident_indices[num_msg_pass:]
    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
    keep_mask[split_indices] = False
    
    # Extract edges before removing them
    split_edge_index = edge_index[:, split_indices]
    
    # Update edge_index in data (remove split edges)
    data[target_edge_type].edge_index = edge_index[:, keep_mask]
    
    logger.debug(f"Split edges for {target_edge_type} (split={split}, seed={seed}, edge_msg_pass_prop={edge_msg_pass_prop}): "
                f"incident={num_incident}, msg_pass={num_msg_pass}, removed={num_split}")
    
    return data, split_edge_index


def extract_and_save_embeddings(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    val_loader: NeighborLoader,
    test_loader: NeighborLoader,
    global_loader: NeighborLoader,
    device: torch.device,
    target_node_type: str,
    embeddings_path: Path,
    logger: logging.Logger
) -> Dict:
    """
    Extract embeddings from model using loaders and save to disk.
    
    Args:
        model: Trained model for embedding extraction
        train_loader: Training NeighborLoader
        val_loader: Validation NeighborLoader  
        test_loader: Test NeighborLoader
        global_loader: Global NeighborLoader
        device: Device to use for extraction
        target_node_type: Target node type for extraction
        embeddings_path: Path to save embeddings
        logger: Logger instance
    
    Returns:
        Dictionary containing all embeddings and labels
    """
    
    logger.info("Extracting embeddings from model...")
    logger.info("Extracting train embeddings...")
    train_embeddings, train_labels = extract_embeddings(
        model, train_loader, device, target_node_type
    )
    logger.info("Extracting val embeddings...")
    val_embeddings, val_labels = extract_embeddings(
        model, val_loader, device, target_node_type
    )
    logger.info("Extracting test embeddings...")
    test_embeddings, test_labels = extract_embeddings(
        model, test_loader, device, target_node_type
    )
    logger.info("Extracting global embeddings...")
    global_embeddings, _ = extract_embeddings(
        model, global_loader, device, target_node_type
    )
    
    # Save embeddings
    embeddings_data = {
        'train_embeddings': train_embeddings,
        'global_embeddings': global_embeddings,
        'train_labels': train_labels,
        'val_embeddings': val_embeddings,
        'val_labels': val_labels,
        'test_embeddings': test_embeddings,
        'test_labels': test_labels
    }
    torch.save(embeddings_data, embeddings_path)
    logger.info(f"Embeddings saved to: {embeddings_path}")
    logger.info(f"  Train embeddings shape: {train_embeddings.shape}")
    logger.info(f"  Val embeddings shape: {val_embeddings.shape}")
    logger.info(f"  Test embeddings shape: {test_embeddings.shape}")
    logger.info(f"  Global embeddings shape: {global_embeddings.shape}")
    
    return embeddings_data
