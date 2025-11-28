"""
Downstream evaluation utilities for node property prediction and link prediction.
Evaluates the quality of learned embeddings by training simple classifier heads.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from graphssl.utils.data_utils import create_edge_splits, create_neighbor_loaders, create_link_loaders, extract_and_save_embeddings
from graphssl.utils.training_utils import extract_embeddings
from graphssl.utils.objective_utils import DownstreamNodeClassification, DownstreamLinkMulticlass, SupervisedLinkPrediction, EdgeDecoder
from graphssl.utils.downstream_models import MLPClassifier
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
from pathlib import Path
import wandb


logger = logging.getLogger(__name__)


def train_downstream_model(
    model: nn.Module,
    objective,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    verbose: bool = False,
    set_model_train: bool = True
) -> Dict[str, List[float]]:
    """
    Unified training function for downstream tasks using objectives.
    
    Args:
        model: The model (encoder or classifier)
        objective: TrainingObjective instance defining the task
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        verbose: Whether to print progress
        set_model_train: Whether to call model.train() (False for frozen encoders)
    
    Returns:
        Dictionary containing training history
    """
    best_val_metric = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training
        if set_model_train:
            model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
            else:
                batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through objective
            loss, batch_metrics = objective.step(model, batch, is_training=True)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            train_loss += batch_metrics['loss'] * batch_metrics['total']
            train_correct += batch_metrics['correct']
            train_total += batch_metrics['total']
        
        print(f"End of epoch {epoch+1}: train_total={train_total}, type={type(train_total)}, train_correct={train_correct}, train_loss={train_loss}")
        
        if train_total == 0:
            logger.warning(f"Epoch {epoch+1}: train_total is 0 - no samples processed in training")
            train_loss = 0.0
            train_acc = 0.0
        else:
            train_loss /= train_total
            train_acc = train_correct / train_total
        
        # Validation
        val_loss, val_acc = evaluate_downstream_model(
            model, objective, val_loader, device
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_acc > best_val_metric:
            best_val_metric = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
    
    return history


@torch.no_grad()
def evaluate_downstream_model(
    model: nn.Module,
    objective,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Unified evaluation function for downstream tasks using objectives.
    
    Args:
        model: The model (encoder or classifier)
        objective: TrainingObjective instance defining the task
        loader: Data loader
        device: Device to evaluate on
    
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    num_empty_batches = 0
    
    for batch in loader:
        num_batches += 1
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            # Handle cases where batch contains tensors and non-tensors (e.g., integers)
            batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
        else:
            batch = batch.to(device)
        
        # Forward pass through objective
        _, batch_metrics = objective.step(model, batch, is_training=False)
        
        # Track empty batches
        if batch_metrics['total'] == 0:
            num_empty_batches += 1
        
        # Accumulate metrics
        total_loss += batch_metrics['loss'] * batch_metrics['total']
        correct += batch_metrics['correct']
        total += batch_metrics['total']
    
    print(f"End of evaluation: train_total={total}, type={type(total)}, train_correct={correct}, train_loss={total_loss}")
    if total == 0:
        logger.warning(f"Evaluation: total is 0 - processed {num_batches} batches, {num_empty_batches} had no edges")
        return 0.0, 0.0
    
    if num_empty_batches > 0:
        logger.info(f"Evaluation: {num_empty_batches}/{num_batches} batches had no edges (papers with no field_of_study)")
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate_node_property_prediction(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    n_runs: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.5,
    batch_size: int = 1024,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate node property prediction by training multiple classifier heads.
    
    Args:
        train_embeddings: Training embeddings [N_train, embedding_dim]
        train_labels: Training labels [N_train]
        val_embeddings: Validation embeddings [N_val, embedding_dim]
        val_labels: Validation labels [N_val]
        test_embeddings: Test embeddings [N_test, embedding_dim]
        test_labels: Test labels [N_test]
        num_classes: Number of classes
        device: Device to train on
        n_runs: Number of independent runs for uncertainty estimation
        hidden_dim: Hidden dimension for MLP
        num_layers: Number of MLP layers
        dropout: Dropout rate
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay
        num_epochs: Maximum number of epochs per run
        early_stopping_patience: Patience for early stopping
        verbose: Whether to print progress
    
    Returns:
        Dictionary with mean and std of metrics across runs
    """
    logger.info("="*80)
    logger.info("Downstream Task: Node Property Prediction")
    logger.info("="*80)
    logger.info(f"Train samples: {train_embeddings.shape[0]}")
    logger.info(f"Val samples: {val_embeddings.shape[0]}")
    logger.info(f"Test samples: {test_embeddings.shape[0]}")
    logger.info(f"Embedding dim: {train_embeddings.shape[1]}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Running {n_runs} independent trials...")
    
    # Create datasets
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Store results from all runs
    test_accuracies = []
    test_losses = []
    
    for run in range(n_runs):
        if verbose:
            logger.info(f"\nRun {run+1}/{n_runs}")
        
        # Create classifier
        classifier = MLPClassifier(
            input_dim=train_embeddings.shape[1],
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            use_batchnorm=True
        ).to(device)
        
        # Create objective
        objective = DownstreamNodeClassification(
            classifier=classifier,
            num_classes=num_classes
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Train classifier
        history = train_downstream_model(
            model=classifier,
            objective=objective,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose
        )
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_downstream_model(
            classifier, objective, test_loader, device
        )
        
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        if verbose:
            logger.info(f"Run {run+1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Log to wandb
        wandb.log({
            f"downstream_node/run_{run+1}/test_loss": test_loss,
            f"downstream_node/run_{run+1}/test_acc": test_acc,
        })
    
    # Compute statistics
    results = {
        'test_acc_mean': np.mean(test_accuracies),
        'test_acc_std': np.std(test_accuracies),
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
        'test_accuracies': test_accuracies,
        'test_losses': test_losses
    }
    
    logger.info("\n" + "="*80)
    logger.info("Node Property Prediction Results")
    logger.info("="*80)
    logger.info(f"Test Accuracy: {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    logger.info(f"Test Loss: {results['test_loss_mean']:.4f} ± {results['test_loss_std']:.4f}")
    
    # Log summary to wandb
    wandb.log({
        "downstream_node/test_acc_mean": results['test_acc_mean'],
        "downstream_node/test_acc_std": results['test_acc_std'],
        "downstream_node/test_loss_mean": results['test_loss_mean'],
        "downstream_node/test_loss_std": results['test_loss_std'],
    })
    
    return results


def train_downstream_link_predictor(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    objective,
    device: torch.device,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Train a link prediction decoder with frozen encoder.
    Wrapper around train_downstream_model that handles encoder freezing.
    
    Args:
        model: The encoder model (frozen)
        decoder: The decoder module (trainable)
        train_loader: Training LinkNeighborLoader
        val_loader: Validation LinkNeighborLoader
        optimizer: Optimizer for decoder parameters
        objective: SupervisedLinkPrediction objective
        device: Device to train on
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing training history
    """
    # Freeze encoder parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Ensure decoder parameters are trainable
    for param in decoder.parameters():
        param.requires_grad = True
    
    model.eval()  # Keep encoder in eval mode
    decoder.train()  # Decoder should be trainable
    
    # Use unified training function (don't set model.train() since encoder is frozen)
    return train_downstream_model(
        model=model,
        objective=objective,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
        set_model_train=False  # Keep encoder in eval mode
    )


def evaluate_link_prediction(
    model: torch.nn.Module,
    train_data,
    val_data,
    test_data,
    train_edge_index: torch.Tensor,
    val_edge_index: torch.Tensor,
    test_edge_index: torch.Tensor,
    target_edge_type: Tuple[str, str, str],
    device: torch.device,
    n_runs: int = 10,
    num_neighbors: list = [15, 10],
    hidden_dim: int = 128,
    dropout: float = 0.5,
    batch_size: int = 1024,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    num_epochs: int = 100,
    neg_sampling_ratio: float = 1.0,
    early_stopping_patience: int = 10,
    num_workers: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate link prediction by training decoder heads with frozen encoder.
    Uses LinkNeighborLoaders and SupervisedLinkPrediction objective.
    
    Args:
        model: The trained encoder model (will be frozen)
        train_data: Training data split (HeteroData)
        val_data: Validation data split (HeteroData)
        test_data: Test data split (HeteroData)
        train_edge_index: Training positive edges [2, num_train_edges]
        val_edge_index: Validation positive edges [2, num_val_edges]
        test_edge_index: Test positive edges [2, num_test_edges]
        target_edge_type: Edge type tuple (src_type, relation, dst_type)
        device: Device to train on
        n_runs: Number of independent runs for uncertainty estimation
        num_neighbors: Number of neighbors to sample at each layer
        hidden_dim: Hidden dimension for decoder MLP
        dropout: Dropout rate for decoder
        batch_size: Batch size for training (number of edges per batch)
        lr: Learning rate for decoder
        weight_decay: Weight decay for decoder
        num_epochs: Maximum number of epochs per run
        neg_sampling_ratio: Ratio of negative to positive samples
        early_stopping_patience: Patience for early stopping
        num_workers: Number of worker processes for data loading
        verbose: Whether to print progress
    
    Returns:
        Dictionary with mean and std of metrics across runs
    """
    
    logger.info("="*80)
    logger.info("Downstream Task: Link Prediction")
    logger.info("="*80)
    logger.info(f"Train edges: {train_edge_index.shape[1]}")
    logger.info(f"Val edges: {val_edge_index.shape[1]}")
    logger.info(f"Test edges: {test_edge_index.shape[1]}")
    logger.info(f"Target edge type: {target_edge_type}")
    logger.info(f"Negative sampling ratio: {neg_sampling_ratio}")
    logger.info(f"Running {n_runs} independent trials...")
    
    # Create LinkNeighborLoaders
    logger.info("Creating LinkNeighborLoaders for downstream evaluation...")
    train_loader, val_loader, test_loader = create_link_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_edge_index=train_edge_index,
        val_edge_index=val_edge_index,
        test_edge_index=test_edge_index,
        target_edge_type=target_edge_type,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        neg_sampling_ratio=neg_sampling_ratio,
        num_workers=num_workers
    )
    
    # Store results from all runs
    test_accuracies = []
    test_losses = []
    
    # Move model to device
    model = model.to(device)
    
    for run in range(n_runs):
        if verbose:
            logger.info(f"\nRun {run+1}/{n_runs}")
        
        # Create decoder for this run
        decoder = EdgeDecoder(hidden_dim=hidden_dim, dropout=dropout).to(device)
        
        # Create objective with decoder
        objective = SupervisedLinkPrediction(
            target_edge_type=target_edge_type,
            decoder=decoder
        )
        
        # Create optimizer (only for decoder parameters)
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Train decoder with frozen encoder
        history = train_downstream_link_predictor(
            model=model,
            decoder=decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            objective=objective,
            device=device,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose
        )
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_downstream_model(
            model, objective, test_loader, device
        )
        
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        if verbose:
            logger.info(f"Run {run+1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Log to wandb
        wandb.log({
            f"downstream_link/run_{run+1}/test_loss": test_loss,
            f"downstream_link/run_{run+1}/test_acc": test_acc,
        })
    
    # Compute statistics
    results = {
        'test_acc_mean': np.mean(test_accuracies),
        'test_acc_std': np.std(test_accuracies),
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
        'test_accuracies': test_accuracies,
        'test_losses': test_losses
    }
    
    logger.info("\n" + "="*80)
    logger.info("Link Prediction Results")
    logger.info("="*80)
    logger.info(f"Test Accuracy: {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    logger.info(f"Test Loss: {results['test_loss_mean']:.4f} ± {results['test_loss_std']:.4f}")
    
    # Log summary to wandb
    wandb.log({
        "downstream_link/test_acc_mean": results['test_acc_mean'],
        "downstream_link/test_acc_std": results['test_acc_std'],
        "downstream_link/test_loss_mean": results['test_loss_mean'],
        "downstream_link/test_loss_std": results['test_loss_std'],
    })
    
    return results


def evaluate_link_prediction_multiclass(
    model: torch.nn.Module,
    train_data,
    val_data,
    test_data,
    train_embeddings: torch.Tensor,
    train_edge_index: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_edge_index: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_edge_index: torch.Tensor,
    target_edge_type: Tuple[str, str, str],
    device: torch.device,
    n_runs: int = 10,
    num_neighbors: list = [15, 10],
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.5,
    batch_size: int = 1024,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    num_workers: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate link prediction as multiclass classification.
    Projects source node embeddings to target embedding space.
    Predicts which target nodes (e.g., field_of_study) each source node (e.g., paper) links to.
    Only uses edges from train/val/test_edge_index, not edges from the graph used for message passing.
    
    Args:
        model: The trained encoder model (frozen)
        train_data: Training data split (HeteroData, contains edges used for message passing)
        val_data: Validation data split (HeteroData, contains edges used for message passing)
        test_data: Test data split (HeteroData, contains edges used for message passing)
        train_embeddings: Precomputed training embeddings for source nodes [N_train, hidden_dim]
        train_edge_index: Training edges to predict [2, num_train_edges] where [0] are source indices, [1] are target indices
        val_embeddings: Precomputed validation embeddings for source nodes [N_val, hidden_dim]
        val_edge_index: Validation edges to predict [2, num_val_edges]
        test_embeddings: Precomputed test embeddings for source nodes [N_test, hidden_dim]
        test_edge_index: Test edges to predict [2, num_test_edges]
        target_edge_type: Edge type tuple (src_type, relation, dst_type)
        device: Device to train on
        n_runs: Number of independent runs for uncertainty estimation
        num_neighbors: Number of neighbors to sample at each layer
        hidden_dim: Hidden dimension for decoder MLP
        num_layers: Number of MLP layers
        dropout: Dropout rate for decoder
        batch_size: Batch size for training
        lr: Learning rate for decoder
        weight_decay: Weight decay for decoder
        num_epochs: Maximum number of epochs per run
        early_stopping_patience: Patience for early stopping
        num_workers: Number of worker processes for data loading
        verbose: Whether to print progress
    
    Returns:
        Dictionary with mean and std of metrics across runs
    """
    
    logger.info("="*80)
    logger.info("Downstream Task: Link Prediction as Multiclass Classification")
    logger.info("="*80)
    logger.info(f"Target edge type: {target_edge_type}")
    logger.info(f"Running {n_runs} independent trials...")
    
    # Extract node types from edge type
    source_node_type = target_edge_type[0]
    target_node_type = target_edge_type[2]
    
    # Get edges that were used for message passing during model training
    # These edges should NOT be used as positive or negative examples
    train_msg_passing_edges = train_data[target_edge_type].edge_index
    val_msg_passing_edges = val_data[target_edge_type].edge_index
    test_msg_passing_edges = test_data[target_edge_type].edge_index
    
    logger.info(f"Message passing edges - Train: {train_msg_passing_edges.size(1)}, Val: {val_msg_passing_edges.size(1)}, Test: {test_msg_passing_edges.size(1)}")
    logger.info(f"Supervision edges - Train: {train_edge_index.size(1)}, Val: {val_edge_index.size(1)}, Test: {test_edge_index.size(1)}")
    
    # Extract embeddings for all target nodes (e.g., all field_of_study nodes)
    logger.info(f"Extracting embeddings for all {target_node_type} nodes...")
    model = model.to(device)
    model.eval()
    
    # Create loader for target nodes
    target_loader = NeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(target_node_type, torch.arange(train_data[target_node_type].num_nodes)),
        num_workers=num_workers,
        shuffle=False
    )
    
    # Extract target embeddings
    target_embeddings_list = []
    with torch.no_grad():
        for batch in target_loader:
            batch = batch.to(device)
            _, embeddings_dict = model(batch.x_dict, batch.edge_index_dict)
            target_batch_size = batch[target_node_type].batch_size
            target_embeddings_list.append(
                embeddings_dict[target_node_type][:target_batch_size].cpu()
            )
    
    target_embeddings = torch.cat(target_embeddings_list, dim=0)
    logger.info(f"Target embeddings shape: {target_embeddings.shape}")
    
    # Use precomputed source embeddings
    logger.info(f"Using precomputed source node embeddings...")
    logger.info(f"Train samples: {train_embeddings.shape[0]}")
    logger.info(f"Val samples: {val_embeddings.shape[0]}")
    logger.info(f"Test samples: {test_embeddings.shape[0]}")
    logger.info(f"Embedding dim: {train_embeddings.shape[1]}")
    logger.info(f"Number of target classes: {target_embeddings.shape[0]}")
    
    # Create custom dataset that returns edge indices per batch
    logger.info("Creating edge-based datasets...")
    
    class EdgeIndexDataset(torch.utils.data.Dataset):
        """
        Dataset that returns embeddings and edge indices for multi-label classification.
        Each batch contains all embeddings and the edges for those specific source nodes.
        """
        def __init__(self, embeddings, edge_index):
            """
            Args:
                embeddings: Node embeddings [num_nodes, hidden_dim]
                edge_index: Edge indices [2, num_edges] where [0] are source, [1] are target
            """
            self.embeddings = embeddings
            self.edge_index = edge_index
            self.num_nodes = embeddings.size(0)
            
        def __len__(self):
            return self.num_nodes
        
        def __getitem__(self, idx):
            # Return embedding and node index
            return self.embeddings[idx], idx
    
    def build_edge_dict(edge_index):
        """Build mapping from source node to list of target nodes."""
        edge_dict = {}
        for i in range(edge_index.size(1)):
            src = int(edge_index[0, i].item())
            tgt = int(edge_index[1, i].item())
            if src not in edge_dict:
                edge_dict[src] = []
            edge_dict[src].append(tgt)
        return edge_dict
    
    # Build supervision edge dicts (disjoint from message passing edges)
    train_edge_dict = build_edge_dict(train_edge_index)
    val_edge_dict = build_edge_dict(val_edge_index)
    test_edge_dict = build_edge_dict(test_edge_index)
    
    # Build message passing edge dicts
    train_msg_pass_edge_dict = build_edge_dict(train_msg_passing_edges)
    val_msg_pass_edge_dict = build_edge_dict(val_msg_passing_edges)
    test_msg_pass_edge_dict = build_edge_dict(test_msg_passing_edges)
    
    logger.info(f"Train nodes - supervision edges: {len(train_edge_dict)}, msg passing edges: {len(train_msg_pass_edge_dict)}")
    logger.info(f"Val nodes - supervision edges: {len(val_edge_dict)}, msg passing edges: {len(val_msg_pass_edge_dict)}")
    logger.info(f"Test nodes - supervision edges: {len(test_edge_dict)}, msg passing edges: {len(test_msg_pass_edge_dict)}")
    
    def collate_with_edges(batch, edge_dict, msg_pass_edge_dict):
        """
        Custom collate that includes supervision and message passing edge indices for the batch.
        """
        embeddings_list, indices_list = zip(*batch)
        embeddings = torch.stack(embeddings_list, dim=0)
        indices = torch.tensor(indices_list, dtype=torch.long)
        
        # Build supervision edge index for this batch
        batch_edge_sources = []
        batch_edge_targets = []
        
        for local_idx, global_idx in enumerate(indices):
            global_idx = int(global_idx.item())
            if global_idx in edge_dict:
                for target_idx in edge_dict[global_idx]:
                    batch_edge_sources.append(local_idx)
                    batch_edge_targets.append(target_idx)
        
        if len(batch_edge_sources) > 0:
            batch_edge_index = torch.tensor(
                [batch_edge_sources, batch_edge_targets],
                dtype=torch.long
            )
        else:
            # Empty edge index
            batch_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Build message passing edge index for this batch
        msg_pass_batch_edge_sources = []
        msg_pass_batch_edge_targets = []
        
        for local_idx, global_idx in enumerate(indices):
            global_idx = int(global_idx.item())
            if global_idx in msg_pass_edge_dict:
                for target_idx in msg_pass_edge_dict[global_idx]:
                    msg_pass_batch_edge_sources.append(local_idx)
                    msg_pass_batch_edge_targets.append(target_idx)
        
        if len(msg_pass_batch_edge_sources) > 0:
            msg_pass_batch_edge_index = torch.tensor(
                [msg_pass_batch_edge_sources, msg_pass_batch_edge_targets],
                dtype=torch.long
            )
        else:
            # Empty edge index
            msg_pass_batch_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return embeddings, batch_edge_index, msg_pass_batch_edge_index, len(indices)
    
    # Create datasets
    train_dataset = EdgeIndexDataset(train_embeddings, train_edge_index)
    val_dataset = EdgeIndexDataset(val_embeddings, val_edge_index)
    test_dataset = EdgeIndexDataset(test_embeddings, test_edge_index)
    
    # Create data loaders with custom collate functions
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_with_edges(batch, train_edge_dict, train_msg_pass_edge_dict)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_with_edges(batch, val_edge_dict, val_msg_pass_edge_dict)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_with_edges(batch, test_edge_dict, test_msg_pass_edge_dict)
    )
    
    # Store results from all runs
    test_accuracies = []
    test_losses = []
    
    for run in range(n_runs):
        if verbose:
            logger.info(f"\nRun {run+1}/{n_runs}")
        
        # Create decoder that projects source embeddings to target embedding space
        decoder = MLPClassifier(
            input_dim=train_embeddings.shape[1],
            hidden_dim=hidden_dim,
            output_dim=train_embeddings.shape[1],  # Project to same dimension
            num_layers=num_layers,
            dropout=dropout,
            use_batchnorm=True
        ).to(device)
        
        # Create objective
        objective = DownstreamLinkMulticlass(
            decoder=decoder,
            target_embeddings=target_embeddings,
            device=device
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Train decoder
        history = train_downstream_model(
            model=decoder,
            objective=objective,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose
        )
        
        # Evaluate on test set
        test_loss, test_f1 = evaluate_downstream_model(
            decoder, objective, test_loader, device
        )
        
        test_accuracies.append(test_f1)
        test_losses.append(test_loss)
        
        if verbose:
            logger.info(f"Run {run+1} - Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
        
        # Log to wandb
        wandb.log({
            f"downstream_link_multiclass/run_{run+1}/test_loss": test_loss,
            f"downstream_link_multiclass/run_{run+1}/test_f1": test_f1,
        })
    
    # Compute statistics
    results = {
        'test_f1_mean': np.mean(test_accuracies),
        'test_f1_std': np.std(test_accuracies),
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
        'test_f1_scores': test_accuracies,
        'test_losses': test_losses
    }
    
    logger.info("\n" + "="*80)
    logger.info("Link Prediction as Multiclass Results")
    logger.info("="*80)
    logger.info(f"Test F1: {results['test_f1_mean']:.4f} ± {results['test_f1_std']:.4f}")
    logger.info(f"Test Loss: {results['test_loss_mean']:.4f} ± {results['test_loss_std']:.4f}")
    
    # Log summary to wandb
    wandb.log({
        "downstream_link_multiclass/test_f1_mean": results['test_f1_mean'],
        "downstream_link_multiclass/test_f1_std": results['test_f1_std'],
        "downstream_link_multiclass/test_loss_mean": results['test_loss_mean'],
        "downstream_link_multiclass/test_loss_std": results['test_loss_std'],
    })
    
    return results


def run_downstream_evaluation(
    args,
    model,
    train_data,
    val_data,
    test_data,
    full_data,
    edge_splits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    results_path: Path,
    num_classes: int
) -> None:
    """
    Run optional downstream evaluation:
    - Node property prediction
    - Link prediction
    
    Args:
        args: Command-line arguments
        model: Trained model for embedding extraction
        train_data: Training data split (HeteroData)
        val_data: Validation data split (HeteroData)
        test_data: Test data split (HeteroData)
        full_data: Full dataset (HeteroData)
        edge_splits: Tuple of (train_edge_index, val_edge_index, test_edge_index) from Step 2
        device: Device to evaluate on
        results_path: Path to save results
        num_classes: Number of classes for node classification
    """
    # Ensure embeddings are available for downstream classification evaluation
    embeddings_path = results_path / "embeddings.pt"

    # Load or extract embeddings
    if embeddings_path.exists():
        # Load from saved file
        logger.info(f"Loading embeddings from: {embeddings_path}")
        embeddings_data = torch.load(embeddings_path)
        train_embeddings = embeddings_data['train_embeddings']
        train_labels = embeddings_data['train_labels']
        val_embeddings = embeddings_data['val_embeddings']
        val_labels = embeddings_data['val_labels']
        test_embeddings = embeddings_data['test_embeddings']
        test_labels = embeddings_data['test_labels']
        global_embeddings = embeddings_data['global_embeddings']
    else:
        # Create loaders and extract embeddings
        logger.info("Creating loaders for embedding extraction...")
        train_loader, val_loader, test_loader, global_loader = create_neighbor_loaders(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            full_data=full_data,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_node_type=args.target_node
        )
        
        embeddings_data = extract_and_save_embeddings(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            global_loader=global_loader,
            device=device,
            target_node_type=args.target_node,
            embeddings_path=embeddings_path,
            logger=logger
        )
        
        train_embeddings = embeddings_data['train_embeddings']
        train_labels = embeddings_data['train_labels']
        val_embeddings = embeddings_data['val_embeddings']
        val_labels = embeddings_data['val_labels']
        test_embeddings = embeddings_data['test_embeddings']
        test_labels = embeddings_data['test_labels']
        global_embeddings = embeddings_data['global_embeddings']

    # Ensure labels are available
    if train_labels is None or val_labels is None or test_labels is None:
        raise ValueError("Labels are required for downstream evaluation but were not found in embeddings")

    # Step 10a: Node property prediction
    if args.downstream_task in ["node", "both"]:
        node_results = evaluate_node_property_prediction(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            num_classes=num_classes,
            device=device,
            n_runs=args.downstream_n_runs,
            hidden_dim=args.downstream_hidden_dim,
            num_layers=args.downstream_num_layers,
            dropout=args.downstream_dropout,
            batch_size=args.downstream_batch_size,
            lr=args.downstream_lr,
            weight_decay=args.downstream_weight_decay,
            num_epochs=args.downstream_epochs,
            early_stopping_patience=args.downstream_patience,
            verbose=True,
        )

        # Save node prediction results
        node_results_path = results_path / "downstream_node_results.pt"
        torch.save(node_results, node_results_path)
        logger.info(f"Node property prediction results saved to: {node_results_path}")

    # Step 10b: Link prediction
    if args.downstream_task in ["link", "both"]:
        # Use edge splits from Step 2
        target_edge_type = tuple(args.target_edge_type.split(","))
        logger.info(f"Using edge splits from Step 2 for edge type: {target_edge_type}")
        train_edge_index, val_edge_index, test_edge_index = edge_splits
        
        # Apply test mode optimization if needed
        if args.test_mode:
            logger.info("Test mode enabled - using reduced settings for link prediction")
        
        link_results = evaluate_link_prediction(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_edge_index=train_edge_index,
            val_edge_index=val_edge_index,
            test_edge_index=test_edge_index,
            target_edge_type=target_edge_type,
            device=device,
            n_runs=args.downstream_n_runs,
            num_neighbors=args.num_neighbors,
            hidden_dim=args.downstream_hidden_dim,
            dropout=args.downstream_dropout,
            batch_size=args.downstream_batch_size,
            lr=args.downstream_lr,
            weight_decay=args.downstream_weight_decay,
            num_epochs=args.downstream_epochs,
            neg_sampling_ratio=args.neg_sampling_ratio,
            early_stopping_patience=args.downstream_patience,
            num_workers=args.num_workers,
            verbose=True
        )

        # Save link prediction results
        link_results_path = results_path / "downstream_link_results.pt"
        torch.save(link_results, link_results_path)
        logger.info(f"Link prediction results saved to: {link_results_path}")
    
    # Step 10c: Link prediction as multiclass (only for paper -> field_of_study)
    target_edge_type_tuple = tuple(args.target_edge_type.split(","))
    if (args.downstream_task in ["link", "both", "multiclass_link"] and 
        target_edge_type_tuple == ("paper", "has_topic", "field_of_study")):
        train_edge_index, val_edge_index, test_edge_index = edge_splits
        
        logger.info("="*80)
        logger.info("Running link prediction as multiclass classification...")
        logger.info("="*80)
        multiclass_results = evaluate_link_prediction_multiclass(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_embeddings=train_embeddings,
            train_edge_index=train_edge_index,
            val_embeddings=val_embeddings,
            val_edge_index=val_edge_index,
            test_embeddings=test_embeddings,
            test_edge_index=test_edge_index,
            target_edge_type=target_edge_type_tuple,
            device=device,
            n_runs=args.downstream_n_runs,
            num_neighbors=args.num_neighbors,
            hidden_dim=args.downstream_hidden_dim,
            num_layers=args.downstream_num_layers,
            dropout=args.downstream_dropout,
            batch_size=args.downstream_batch_size,
            lr=args.downstream_lr,
            weight_decay=args.downstream_weight_decay,
            num_epochs=args.downstream_epochs,
            early_stopping_patience=args.downstream_patience,
            num_workers=args.num_workers,
            verbose=True
        )
        
        # Save multiclass link prediction results
        multiclass_results_path = results_path / "downstream_link_multiclass_results.pt"
        torch.save(multiclass_results, multiclass_results_path)
        logger.info(f"Link prediction multiclass results saved to: {multiclass_results_path}")

