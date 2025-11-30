"""
Downstream evaluation utilities for node property prediction and link prediction.
Evaluates the quality of learned embeddings by training simple classifier heads.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from graphssl.utils.data_utils import create_edge_splits, create_neighbor_loaders, create_link_loaders, extract_and_save_embeddings, validate_edge_index_for_data, validate_edge_index_for_embeddings, create_index_mapping, remap_edges
from graphssl.utils.training_utils import extract_embeddings
from graphssl.utils.objective_utils import DownstreamNodeClassification, DownstreamLinkMulticlass, SupervisedLinkPrediction, EdgeDecoder
from graphssl.utils.downstream_models import MLPClassifier
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
from pathlib import Path
import wandb
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)

# I should really make things functional enough that I can use the train functions from training_utils,
# but I can't be bothered right now. 
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
    set_model_train: bool = True,
    disable_tqdm: bool = False
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
        disable_tqdm: Whether to disable tqdm progress bars
    
    Returns:
        Dictionary containing training history
    """
    best_val_metric = 0.0
    patience_counter = 0
    
    # Initialize history with dynamic metric names from objective
    metric_names = objective.get_metric_names()
    history = {f"train_{name}": [] for name in metric_names}
    history.update({f"val_{name}": [] for name in metric_names})
    
    for epoch in range(num_epochs):
        # Training
        if set_model_train:
            model.train()
        
        # Initialize metric accumulators
        total_metrics = {name: 0.0 for name in metric_names}
        total_metrics['correct'] = 0
        total_metrics['total'] = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False, disable=disable_tqdm):
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
            batch_size = batch_metrics.get('total', 1)
            for key, value in batch_metrics.items():
                if key == 'correct' or key == 'total':
                    total_metrics[key] += value
                elif key in total_metrics:
                    total_metrics[key] += value * batch_size
        
        # print(f"End of epoch {epoch+1}: train_total={total_metrics['total']}, train_correct={total_metrics['correct']}, train_loss={total_metrics.get('loss', 0.0)}")
        
        # Compute epoch averages
        train_metrics = {}
        if total_metrics['total'] == 0:
            logger.warning(f"Epoch {epoch+1}: train_total is 0 - no samples processed in training")
            for key in metric_names:
                train_metrics[key] = 0.0
        else:
            for key in metric_names:
                if key in total_metrics:
                    train_metrics[key] = total_metrics[key] / total_metrics['total']
            # Add accuracy if we tracked correct/total
            if 'correct' in total_metrics and total_metrics['total'] > 0:
                train_metrics['acc'] = total_metrics['correct'] / total_metrics['total']
        
        # Validation
        val_metrics = evaluate_downstream_model(
            model, objective, val_loader, device, disable_tqdm
        )
        
        # Store history (dynamically add new keys if they appear)
        for key, value in train_metrics.items():
            if f"train_{key}" not in history:
                history[f"train_{key}"] = []
            history[f"train_{key}"].append(value)
        for key, value in val_metrics.items():
            if f"val_{key}" not in history:
                history[f"val_{key}"] = []
            history[f"val_{key}"].append(value)
        
        # Early stopping (use 'acc' metric if available, otherwise use first metric)
        val_metric_for_best = val_metrics.get('acc', val_metrics.get(metric_names[0], 0.0))
        if val_metric_for_best > best_val_metric:
            best_val_metric = val_metric_for_best
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if verbose and (epoch + 1) % 10 == 0:
            train_str = " | ".join([f"Train {k.capitalize()}: {v:.4f}" for k, v in train_metrics.items()])
            val_str = " | ".join([f"Val {k.capitalize()}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(f"Epoch {epoch+1}/{num_epochs} - {train_str} | {val_str}")
    
    return history


@torch.no_grad()
def evaluate_downstream_model(
    model: nn.Module,
    objective,
    loader: DataLoader,
    device: torch.device,
    disable_tqdm: bool = False
) -> Dict[str, float]:
    """
    Unified evaluation function for downstream tasks using objectives.
    
    Args:
        model: The model (encoder or classifier)
        objective: TrainingObjective instance defining the task
        loader: Data loader
        device: Device to evaluate on
        disable_tqdm: Whether to disable tqdm progress bars
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize metric accumulators
    metric_names = objective.get_metric_names()
    total_metrics = {name: 0.0 for name in metric_names}
    total_metrics['correct'] = 0
    total_metrics['total'] = 0
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
        batch_size = batch_metrics.get('total', 1)
        for key, value in batch_metrics.items():
            if key == 'correct' or key == 'total':
                total_metrics[key] += value
            elif key in total_metrics:
                total_metrics[key] += value * batch_size
    
    # print(f"End of evaluation: total={total_metrics['total']}, correct={total_metrics['correct']}, loss={total_metrics.get('loss', 0.0)}")
    
    # Compute averages
    eval_metrics = {}
    if total_metrics['total'] == 0:
        logger.warning(f"Evaluation: total is 0 - processed {num_batches} batches, {num_empty_batches} had no edges")
        for key in metric_names:
            eval_metrics[key] = 0.0
    else:
        if num_empty_batches > 0:
            logger.info(f"Evaluation: {num_empty_batches}/{num_batches} batches had no edges (papers with no field_of_study)")
        
        for key in metric_names:
            if key in total_metrics:
                eval_metrics[key] = total_metrics[key] / total_metrics['total']
        
        # Add accuracy if we tracked correct/total
        if 'correct' in total_metrics and total_metrics['total'] > 0:
            eval_metrics['acc'] = total_metrics['correct'] / total_metrics['total']
    
    return eval_metrics


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
    verbose: bool = True,
    disable_tqdm: bool = False
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
            verbose=verbose,
            disable_tqdm=disable_tqdm
        )
        
        # Evaluate on test set
        test_metrics = evaluate_downstream_model(
            classifier, objective, test_loader, device, disable_tqdm
        )
        
        test_acc = test_metrics.get('acc', 0.0)
        test_loss = test_metrics.get('loss', 0.0)
        test_precision = test_metrics.get('precision', 0.0)
        test_recall = test_metrics.get('recall', 0.0)
        
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        if verbose:
            logger.info(f"Run {run+1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        # Log to wandb (including training history)
        wandb.log({
            f"downstream_node/run_{run+1}/test_loss": test_loss,
            f"downstream_node/run_{run+1}/test_acc": test_acc,
            f"downstream_node/run_{run+1}/test_precision": test_precision,
            f"downstream_node/run_{run+1}/test_recall": test_recall,
            f"downstream_node/run_{run+1}/training_history": history,
        })
    
    # Compute statistics
    results = {
        'test_acc_mean': np.mean(test_accuracies),
        'test_acc_std': np.std(test_accuracies),
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
        'test_accuracies': test_accuracies,
        'test_losses': test_losses,
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
    verbose: bool = False,
    disable_tqdm: bool = False
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
        set_model_train=False,  # Keep encoder in eval mode
        disable_tqdm=disable_tqdm
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
    verbose: bool = True,
    disable_tqdm: bool = False
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
            verbose=verbose,
            disable_tqdm=disable_tqdm
        )
        
        # Evaluate on test set
        test_metrics = evaluate_downstream_model(
            model, objective, test_loader, device, disable_tqdm
        )
        
        test_acc = test_metrics.get('acc', 0.0)
        test_loss = test_metrics.get('loss', 0.0)
        test_precision = test_metrics.get('precision', 0.0)
        test_recall = test_metrics.get('recall', 0.0)
        
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        if verbose:
            logger.info(f"Run {run+1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        # Log to wandb (including training history)
        wandb.log({
            f"downstream_link/run_{run+1}/test_loss": test_loss,
            f"downstream_link/run_{run+1}/test_acc": test_acc,
            f"downstream_link/run_{run+1}/test_precision": test_precision,
            f"downstream_link/run_{run+1}/test_recall": test_recall,
            f"downstream_link/run_{run+1}/training_history": history,
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
    train_msg_passing_edges: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_edge_index: torch.Tensor,
    val_msg_passing_edges: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_edge_index: torch.Tensor,
    test_msg_passing_edges: torch.Tensor,
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
    verbose: bool = True,
    disable_tqdm: bool = False
) -> Dict[str, Any]:
    """
    Evaluate link prediction as multi-label classification using one-hot encoded labels.
    Projects source node embeddings to target embedding space.
    
    Labels are encoded as:
      - 1 if there's a supervision edge (from train/val/test_edge_index)
      - -1 if there's a message passing edge (masked out in loss)
      - 0 otherwise (negative example)
    
    Args:
        model: The trained encoder model (frozen)
        train_data: Training data split (HeteroData, contains edges used for message passing)
        val_data: Validation data split (HeteroData, contains edges used for message passing)
        test_data: Test data split (HeteroData, contains edges used for message passing)
        train_embeddings: Precomputed training embeddings for source nodes [N_train, hidden_dim]
        train_edge_index: Training edges to predict [2, num_train_edges] where [0] are source, [1] are target
        train_msg_passing_edges: Training message passing edges (remapped) [2, num_edges]
        val_embeddings: Precomputed validation embeddings for source nodes [N_val, hidden_dim]
        val_edge_index: Validation edges to predict [2, num_val_edges]
        val_msg_passing_edges: Validation message passing edges (remapped) [2, num_edges]
        test_embeddings: Precomputed test embeddings for source nodes [N_test, hidden_dim]
        test_edge_index: Test edges to predict [2, num_test_edges]
        test_msg_passing_edges: Test message passing edges (remapped) [2, num_edges]
        target_edge_type: Edge type tuple (src_type, relation, dst_type)
        device: Device to train on
        n_runs: Number of independent runs for uncertainty estimation
        num_neighbors: Number of neighbors to sample at each layer (not used, kept for compatibility)
        hidden_dim: Hidden dimension for decoder MLP
        num_layers: Number of MLP layers
        dropout: Dropout rate for decoder
        batch_size: Batch size for training
        lr: Learning rate for decoder
        weight_decay: Weight decay for decoder
        num_epochs: Maximum number of epochs per run
        early_stopping_patience: Patience for early stopping
        num_workers: Number of worker processes for data loading (not used)
        verbose: Whether to print progress
        disable_tqdm: Whether to disable tqdm progress bars
    
    Returns:
        Dictionary with mean and std of metrics across runs
    """
    
    logger.info("="*80)
    logger.info("Downstream Task: Link Prediction as Multi-Label Classification")
    logger.info("="*80)
    logger.info(f"Target edge type: {target_edge_type}")
    logger.info(f"Running {n_runs} independent trials...")
    
    # Extract node types from edge type
    source_node_type = target_edge_type[0]
    target_node_type = target_edge_type[2]
    
    logger.info(f"Message passing edges (remapped) - Train: {train_msg_passing_edges.size(1)}, Val: {val_msg_passing_edges.size(1)}, Test: {test_msg_passing_edges.size(1)}")
    logger.info(f"Supervision edges (remapped) - Train: {train_edge_index.size(1)}, Val: {val_edge_index.size(1)}, Test: {test_edge_index.size(1)}")
    
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
    num_targets = target_embeddings.shape[0]
    logger.info(f"Target embeddings shape: {target_embeddings.shape}")
    logger.info(f"Number of target classes: {num_targets}")
    
    logger.info(f"Train supervision edges: {train_edge_index.size(1)}")
    logger.info(f"Train message passing edges: {train_msg_passing_edges.size(1)}")
    logger.info(f"Total possible edges: {train_embeddings.shape[0] * num_targets}")
    
    class SparseEdgeDataset(Dataset):
        def __init__(self, embeddings, edge_index, msg_pass_edge_index):
            """
            Dataset that returns embeddings with their corresponding edge lists.
            
            Args:
                embeddings: Node embeddings [num_nodes, hidden_dim]
                edge_index: Supervision edges [2, num_edges]
                msg_pass_edge_index: Message passing edges [2, num_edges]
            """
            self.embeddings = embeddings
            
            # Create lookup dictionaries with tensors
            def create_target_dict(edge_index):
                dict = {}
                if edge_index.size(1) > 0:
                    unique_sources = torch.unique(edge_index[0])
                    for src in unique_sources.tolist():
                        targets = edge_index[1, edge_index[0] == src]
                        dict[src] = targets
                return dict

            self.pos_dict = create_target_dict(edge_index)
            self.msg_pass_dict = create_target_dict(msg_pass_edge_index)
    
        def __len__(self):
            return len(self.embeddings)
        
        def __getitem__(self, idx):
            embedding = self.embeddings[idx]
            pos_targets = self.pos_dict.get(idx, torch.tensor([], dtype=torch.long))
            msg_pass_targets = self.msg_pass_dict.get(idx, torch.tensor([], dtype=torch.long))
            return embedding, pos_targets, msg_pass_targets

    class SparseEdgeDataset2(Dataset):
        def __init__(self, embeddings, edge_index, msg_pass_edge_index):
            self.embeddings = embeddings

            def build_groups(edge_index):
                # Sort by source
                src = edge_index[0]
                dst = edge_index[1]
                order = torch.argsort(src)
                src = src[order]
                dst = dst[order]

                # Build index pointers of length (num_nodes + 1)
                num_nodes = embeddings.size(0)
                counts = torch.bincount(src, minlength=num_nodes)
                ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
                ptr[1:] = torch.cumsum(counts, dim=0)

                return dst, ptr  # flat_targets, index_pointer

            self.pos_targets, self.pos_ptr = build_groups(edge_index)
            self.msg_targets, self.msg_ptr = build_groups(msg_pass_edge_index)

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            emb = self.embeddings[idx]

            # Retrieve slices through the pointer array
            p0, p1 = self.pos_ptr[idx], self.pos_ptr[idx + 1]
            pos = self.pos_targets[p0:p1]

            m0, m1 = self.msg_ptr[idx], self.msg_ptr[idx + 1]
            msg = self.msg_targets[m0:m1]

            return emb, pos, msg

    train_dataset = SparseEdgeDataset2(train_embeddings, train_edge_index, train_msg_passing_edges)
    val_dataset = SparseEdgeDataset2(val_embeddings, val_edge_index, val_msg_passing_edges)
    test_dataset = SparseEdgeDataset2(test_embeddings, test_edge_index, test_msg_passing_edges)
    
    # Custom collate function to handle sparse edge tensors
    def collate_sparse_edges(batch):
        embeddings = torch.stack([item[0] for item in batch])
        pos_targets = [item[1] for item in batch]  # List of tensors
        msg_pass_targets = [item[2] for item in batch]  # List of tensors
        return embeddings, pos_targets, msg_pass_targets
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_sparse_edges)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sparse_edges)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sparse_edges)
    
    # Store results from all runs
    test_f1_scores = []
    test_losses = []
    test_precisions = []
    test_recalls = []
    test_accuracies = []
    confusion_matrices = []
    
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
            verbose=verbose,
            disable_tqdm=disable_tqdm
        )
        
        # Evaluate on test set
        test_metrics = evaluate_downstream_model(
            decoder, objective, test_loader, device, disable_tqdm
        )
        
        # Extract metrics (acc is actually F1 in this case for multiclass)
        test_f1 = test_metrics.get('acc', 0.0)
        test_loss = test_metrics.get('loss', 0.0)
        test_precision = test_metrics.get('precision', 0.0)
        test_recall = test_metrics.get('recall', 0.0)
        
        # Compute confusion matrix and accuracy for this run
        all_preds = []
        all_labels = []
        decoder.eval()
        with torch.no_grad():
            for batch in test_loader:
                source_embeddings, pos_targets_list, msg_pass_targets_list = batch
                source_embeddings = source_embeddings.to(device)
                
                # Forward pass
                projected_embeddings = decoder(source_embeddings)
                logits = torch.matmul(projected_embeddings, target_embeddings.T)
                predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                
                # Create labels
                labels = np.zeros_like(predictions)
                for i, pos_targets in enumerate(pos_targets_list):
                    if len(pos_targets) > 0:
                        labels[i, pos_targets.cpu().numpy()] = 1
                
                all_preds.append(predictions)
                all_labels.append(labels)
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Compute accuracy
        test_accuracy = (all_preds == all_labels).mean()
        
        # Compute confusion matrix (flatten to binary classification)
        cm = confusion_matrix(all_labels.flatten(), all_preds.flatten(), labels=[0, 1])
        
        test_f1_scores.append(test_f1)
        test_losses.append(test_loss)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_accuracies.append(test_accuracy)
        confusion_matrices.append(cm)
        
        if verbose:
            logger.info(f"Run {run+1} - Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        # Log to wandb (including training history)
        wandb.log({
            f"downstream_link_multiclass/run_{run+1}/test_loss": test_loss,
            f"downstream_link_multiclass/run_{run+1}/test_f1": test_f1,
            f"downstream_link_multiclass/run_{run+1}/test_precision": test_precision,
            f"downstream_link_multiclass/run_{run+1}/test_recall": test_recall,
            f"downstream_link_multiclass/run_{run+1}/training_history": history,
        })
    
    # Compute statistics
    results = {
        'test_f1_mean': np.mean(test_f1_scores),
        'test_f1_std': np.std(test_f1_scores),
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
        'test_precision_mean': np.mean(test_precisions),
        'test_precision_std': np.std(test_precisions),
        'test_recall_mean': np.mean(test_recalls),
        'test_recall_std': np.std(test_recalls),
        'test_acc_mean': np.mean(test_accuracies),
        'test_acc_std': np.std(test_accuracies),
        'test_f1_scores': test_f1_scores,
        'test_losses': test_losses,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls,
        'test_accuracies': test_accuracies,
        'confusion_matrix': np.mean(confusion_matrices, axis=0),  # Average confusion matrix
        'confusion_matrices': confusion_matrices,  # All confusion matrices
    }
    
    logger.info("\n" + "="*80)
    logger.info("Link Prediction as Multiclass Results")
    logger.info("="*80)
    logger.info(f"Test Accuracy: {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    logger.info(f"Test F1: {results['test_f1_mean']:.4f} ± {results['test_f1_std']:.4f}")
    logger.info(f"Test Precision: {results['test_precision_mean']:.4f} ± {results['test_precision_std']:.4f}")
    logger.info(f"Test Recall: {results['test_recall_mean']:.4f} ± {results['test_recall_std']:.4f}")
    logger.info(f"Test Loss: {results['test_loss_mean']:.4f} ± {results['test_loss_std']:.4f}")
    
    # Log summary to wandb
    wandb.log({
        "downstream_link_multiclass/test_acc_mean": results['test_acc_mean'],
        "downstream_link_multiclass/test_acc_std": results['test_acc_std'],
        "downstream_link_multiclass/test_f1_mean": results['test_f1_mean'],
        "downstream_link_multiclass/test_f1_std": results['test_f1_std'],
        "downstream_link_multiclass/test_precision_mean": results['test_precision_mean'],
        "downstream_link_multiclass/test_precision_std": results['test_precision_std'],
        "downstream_link_multiclass/test_recall_mean": results['test_recall_mean'],
        "downstream_link_multiclass/test_recall_std": results['test_recall_std'],
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
        # Extract embeddings from datasets (no need to create loaders separately)
        logger.info("Extracting embeddings from datasets...")

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
            logger=logger,
            disable_tqdm=args.disable_tqdm
        )
        
        train_embeddings = embeddings_data['train_embeddings']
        train_labels = embeddings_data['train_labels']
        val_embeddings = embeddings_data['val_embeddings']
        val_labels = embeddings_data['val_labels']
        test_embeddings = embeddings_data['test_embeddings']
        test_labels = embeddings_data['test_labels']
        global_embeddings = embeddings_data['global_embeddings']

    # Debug: Print embedding shapes
    logger.debug(f"Train embeddings shape: {train_embeddings.shape}, Train labels shape: {train_labels.shape}")
    logger.debug(f"Val embeddings shape: {val_embeddings.shape}, Val labels shape: {val_labels.shape}")
    logger.debug(f"Test embeddings shape: {test_embeddings.shape}, Test labels shape: {test_labels.shape}")
    logger.debug(f"Global embeddings shape: {global_embeddings.shape}")
    
    train_edge_index, val_edge_index, test_edge_index = edge_splits

    # Use edge splits from Step 2
    target_edge_type = tuple(args.target_edge_type.split(","))
    logger.info(f"Using edge splits from Step 2 for edge type: {target_edge_type}")

    # Validate edge indices match the datasets
    validate_edge_index_for_data(train_edge_index, train_data, target_edge_type, "train")
    validate_edge_index_for_data(val_edge_index, val_data, target_edge_type, "val")
    validate_edge_index_for_data(test_edge_index, test_data, target_edge_type, "test")
    
    target_node_type = target_edge_type[2]
    source_node_type = target_edge_type[0]
    num_target_embeddings = full_data[target_node_type].num_nodes

    logger.info("Remapping edge indices to match masked embeddings...")
    
    # Create mappings for source nodes
    train_mask = train_data[source_node_type].train_mask
    val_mask = val_data[source_node_type].val_mask
    test_mask = test_data[source_node_type].test_mask
    
    train_mapping = create_index_mapping(train_mask)
    val_mapping = create_index_mapping(val_mask)
    test_mapping = create_index_mapping(test_mask)
    
    # Remap edge indices
    train_edge_index_remapped = remap_edges(train_edge_index, train_mapping)
    val_edge_index_remapped = remap_edges(val_edge_index, val_mapping)
    test_edge_index_remapped = remap_edges(test_edge_index, test_mapping)
    
    logger.info(f"Edge remapping complete:")
    logger.info(f"  Train: {train_edge_index.size(1)} -> {train_edge_index_remapped.size(1)} edges")
    logger.info(f"  Val: {val_edge_index.size(1)} -> {val_edge_index_remapped.size(1)} edges")
    logger.info(f"  Test: {test_edge_index.size(1)} -> {test_edge_index_remapped.size(1)} edges")

    # Also need to remap message passing edges
    train_msg_passing_edges = train_data[target_edge_type].edge_index
    val_msg_passing_edges = val_data[target_edge_type].edge_index
    test_msg_passing_edges = test_data[target_edge_type].edge_index
    
    train_msg_passing_edges_remapped = remap_edges(train_msg_passing_edges, train_mapping)
    val_msg_passing_edges_remapped = remap_edges(val_msg_passing_edges, val_mapping)
    test_msg_passing_edges_remapped = remap_edges(test_msg_passing_edges, test_mapping)
    
    logger.info(f"Message passing edge remapping:")
    logger.info(f"  Train: {train_msg_passing_edges.size(1)} -> {train_msg_passing_edges_remapped.size(1)} edges")
    logger.info(f"  Val: {val_msg_passing_edges.size(1)} -> {val_msg_passing_edges_remapped.size(1)} edges")
    logger.info(f"  Test: {test_msg_passing_edges.size(1)} -> {test_msg_passing_edges_remapped.size(1)} edges")
    
    logger.info("Validating remapped edge splits compatibility with masked embeddings...")
    validate_edge_index_for_embeddings(
        train_edge_index_remapped, 
        train_mask.sum().item(), 
        num_target_embeddings,
        target_edge_type, 
        "train"
    )
    validate_edge_index_for_embeddings(
        val_edge_index_remapped, 
        val_mask.sum().item(), 
        num_target_embeddings,
        target_edge_type, 
        "val"
    )
    validate_edge_index_for_embeddings(
        test_edge_index_remapped, 
        test_mask.sum().item(), 
        num_target_embeddings,
        target_edge_type, 
        "test"
    )
    logger.info("Edge splits validation passed!")

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
            num_epochs=args.downstream_node_epochs,
            early_stopping_patience=args.downstream_patience,
            verbose=True,
            disable_tqdm=args.disable_tqdm
        )

        # Save node prediction results
        node_results_path = results_path / "downstream_node_results.pt"
        torch.save(node_results, node_results_path)
        logger.info(f"Node property prediction results saved to: {node_results_path}")

    # Step 10b: Link prediction as multiclass (only for paper -> field_of_study)
    # Run this first since it's much faster than regular link prediction

    target_edge_type = tuple(args.target_edge_type.split(","))
    if (args.downstream_task in ["link", "both", "multiclass_link"] and 
        target_edge_type == ("paper", "has_topic", "field_of_study")):
        
        logger.info("="*80)
        logger.info("Running link prediction as multiclass classification...")
        logger.info("="*80)
    
        multiclass_results = evaluate_link_prediction_multiclass(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_embeddings=train_embeddings,
            train_edge_index=train_edge_index_remapped,
            train_msg_passing_edges=val_msg_passing_edges_remapped,
            val_embeddings=val_embeddings,
            val_edge_index=val_edge_index_remapped,
            val_msg_passing_edges=val_msg_passing_edges_remapped,
            test_embeddings=test_embeddings,
            test_edge_index=test_edge_index_remapped,
            test_msg_passing_edges=test_msg_passing_edges_remapped,
            target_edge_type=target_edge_type,
            device=device,
            n_runs=args.downstream_n_runs,
            num_neighbors=args.num_neighbors,
            hidden_dim=args.downstream_hidden_dim,
            num_layers=args.downstream_num_layers,
            dropout=args.downstream_dropout,
            batch_size=args.downstream_batch_size,
            lr=args.downstream_lr,
            weight_decay=args.downstream_weight_decay,
            num_epochs=args.downstream_node_epochs,
            early_stopping_patience=args.downstream_patience,
            num_workers=args.num_workers,
            verbose=True,
            disable_tqdm=args.disable_tqdm
        )
        
        # Save multiclass link prediction results
        multiclass_results_path = results_path / "downstream_link_multiclass_results.pt"
        torch.save(multiclass_results, multiclass_results_path)
        logger.info(f"Link prediction multiclass results saved to: {multiclass_results_path}")

    # Step 10c: Link prediction (regular - slower) This uses LinkNeighborLoader
    if args.downstream_task in ["link", "both"]:
        
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
            num_epochs=args.downstream_link_epochs,
            neg_sampling_ratio=args.neg_sampling_ratio,
            early_stopping_patience=args.downstream_patience,
            num_workers=args.num_workers,
            verbose=True,
            disable_tqdm=args.disable_tqdm
        )

        # Save link prediction results
        link_results_path = results_path / "downstream_link_results.pt"
        torch.save(link_results, link_results_path)
        logger.info(f"Link prediction results saved to: {link_results_path}")

