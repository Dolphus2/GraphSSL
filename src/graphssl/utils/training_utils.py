"""
Training and evaluation utilities for graph neural networks
"""
import logging
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from typing import Dict, Tuple, Optional, Any, Union
import time
from tqdm import tqdm
from pathlib import Path
import wandb
from .objective_utils import TrainingObjective

logger = logging.getLogger(__name__)


def _initialize_metrics(metric_names: list) -> Dict[str, float]:
    """Initialize metric accumulator dictionary.
    
    Args:
        metric_names: List of metric names to track
    
    Returns:
        Dictionary with initialized metric counters
    """
    metrics = {name: 0.0 for name in metric_names}
    metrics['correct'] = 0
    metrics['total'] = 0
    return metrics


def _accumulate_batch_metrics(
    total_metrics: Dict[str, float],
    batch_metrics: Dict[str, float],
    batch_size: float
) -> None:
    """Accumulate batch metrics into total metrics.
    
    Args:
        total_metrics: Dictionary of accumulated metrics (modified in place)
        batch_metrics: Metrics from current batch
        batch_size: Size of current batch
    """
    for key, value in batch_metrics.items():
        if key == 'correct' or key == 'total':
            total_metrics[key] += value
        elif key in total_metrics:
            total_metrics[key] += value * batch_size


def _compute_epoch_metrics(
    total_metrics: Dict[str, float],
    metric_names: list
) -> Dict[str, float]:
    """Compute epoch-averaged metrics from accumulated totals.
    
    Args:
        total_metrics: Dictionary of accumulated metrics
        metric_names: List of metric names to include
    
    Returns:
        Dictionary of averaged metrics
    """
    epoch_metrics = {}
    total_examples = total_metrics['total']
    
    for key in metric_names:
        if key in total_metrics:
            epoch_metrics[key] = total_metrics[key] / max(total_examples, 1)
    
    # Add accuracy if we tracked correct/total
    if 'correct' in total_metrics and total_metrics['total'] > 0:
        epoch_metrics['acc'] = total_metrics['correct'] / total_metrics['total']
    
    return epoch_metrics


def _log_training_metrics(
    running_metrics: Dict[str, float],
    num_logged: int,
    metric_names: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int
) -> None:
    """Log training metrics to wandb.
    
    Args:
        running_metrics: Dictionary of running metric totals
        num_logged: Number of batches accumulated
        metric_names: List of metric names to log
        optimizer: Optimizer (for learning rate)
        device: Device (for GPU memory)
        epoch: Current epoch number
        global_step: Global training step
    """
    log_dict = {
        f"train/{key}": running_metrics[key] / num_logged
        for key in metric_names if key in running_metrics
    }
    log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
    log_dict["train/epoch"] = epoch
    
    if device.type == "cuda":
        log_dict["sys/gpu_mem_mb"] = torch.cuda.memory_allocated(device) / 1024**2
    
    wandb.log(log_dict, step=global_step)


def train_epoch(
    model: torch.nn.Module,
    loader: Union[NeighborLoader, LinkNeighborLoader],
    optimizer: torch.optim.Optimizer,
    objective: TrainingObjective,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_interval: int = 20,
    disable_tqdm: bool = False
) -> Tuple[Dict[str, float], int]:
    """
    Train the model for one epoch using a specified training objective.
    
    Args:
        model: The model to train
        loader: Training data loader
        optimizer: Optimizer
        objective: Training objective defining the task (step function)
        device: Device to train on
        epoch: Epoch number
        global_step: Global training step counter
        log_interval: How often to log metrics
        disable_tqdm: Whether to disable tqdm progress bars
    
    Returns:
        Tuple of (epoch_metrics dict, updated global_step)
    """
    model.train()
    
    # Initialize metric accumulators
    objective.set_current_epoch(epoch)
    metric_names = objective.get_metric_names()
    total_metrics = _initialize_metrics(metric_names)
    running_metrics = _initialize_metrics(metric_names)
    num_logged = 0
    
    for batch in tqdm(loader, desc="Training", leave=False, disable=disable_tqdm):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Task-specific forward pass and loss computation
        loss, batch_metrics = objective.step(model, batch, is_training=True)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        batch_size = batch_metrics.get('total', 1)
        _accumulate_batch_metrics(total_metrics, batch_metrics, batch_size)
        _accumulate_batch_metrics(running_metrics, batch_metrics, batch_size)
        
        global_step += 1
        num_logged += 1
        
        # Periodic logging
        if global_step % log_interval == 0 and num_logged > 0:
            _log_training_metrics(
                running_metrics, num_logged, metric_names,
                optimizer, device, epoch, global_step
            )
            # Reset running metrics
            running_metrics = _initialize_metrics(metric_names)
            num_logged = 0
    
    # Compute epoch averages
    epoch_metrics = _compute_epoch_metrics(total_metrics, metric_names)
    
    return epoch_metrics, global_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Union[NeighborLoader, LinkNeighborLoader],
    objective: TrainingObjective,
    device: torch.device,
    disable_tqdm: bool = False
) -> Dict[str, float]:
    """
    Evaluate the model using a specified training objective.
    
    Args:
        model: The model to evaluate
        loader: Data loader (validation or test)
        objective: Training objective defining the task (step function)
        device: Device to evaluate on
        disable_tqdm: Whether to disable tqdm progress bars
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize metric accumulators
    metric_names = objective.get_metric_names()
    total_metrics = _initialize_metrics(metric_names)
    
    for batch in tqdm(loader, desc="Evaluating", leave=False, disable=disable_tqdm):
        batch = batch.to(device)
        
        # Task-specific forward pass and loss computation
        _, batch_metrics = objective.step(model, batch, is_training=False)
        
        # Accumulate metrics
        batch_size = batch_metrics.get('total', 1)
        _accumulate_batch_metrics(total_metrics, batch_metrics, batch_size)
    
    # Compute averages
    eval_metrics = _compute_epoch_metrics(total_metrics, metric_names)
    
    return eval_metrics


def train_model(
    model: torch.nn.Module,
    train_loader: Union[NeighborLoader, LinkNeighborLoader],
    val_loader: Union[NeighborLoader, LinkNeighborLoader],
    optimizer: torch.optim.Optimizer,
    objective: TrainingObjective,
    device: torch.device,
    num_epochs: int = 100,
    log_interval: int = 20,
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
    disable_tqdm: bool = False
) -> Dict:
    """
    Train the model with early stopping using a specified training objective.
    Model selection is always based on validation loss (lower is better).
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        objective: Training objective defining the task
        device: Device to train on
        num_epochs: Maximum number of epochs
        log_interval: How often to log metrics during training
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory to save model checkpoints (if None, saves to current directory)
        verbose: Whether to print progress
        disable_tqdm: Whether to disable tqdm progress bars
    
    Returns:
        Dictionary containing training history
    """
    
    # Setup checkpoint directory
    if checkpoint_dir is not None:
        best_model_path = Path(checkpoint_dir) / "best_model.pt"
    else:
        best_model_path = "best_model.pt"
    
    # Initialize history with dynamic metric names
    metric_names = objective.get_metric_names()
    history = {f"train_{name}": [] for name in metric_names}
    history.update({f"val_{name}": [] for name in metric_names})
    history["epoch_time"] = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    global_step = 0
    best_model_saved = False  # Track if we've saved at least one checkpoint
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Training objective: {objective.__class__.__name__}")
    logger.info(f"Tracking metrics: {metric_names}")
    logger.info(f"Using 'loss' for model selection (lower is better)")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, objective, device, epoch, global_step, log_interval, disable_tqdm
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, objective, device, disable_tqdm
        )
        
        epoch_time = time.time() - start_time
        
        # Store metrics
        for key, value in train_metrics.items():
            history[f"train_{key}"].append(value)
        for key, value in val_metrics.items():
            history[f"val_{key}"].append(value)
        history["epoch_time"].append(epoch_time)
        
        # Log to wandb
        wandb_log = {
            f"val/{key}": float(value) for key, value in val_metrics.items()
        }
        wandb_log.update({
            f"train/epoch_{key}": float(value) for key, value in train_metrics.items()
        })
        wandb_log["val/epoch"] = epoch
        wandb_log["epoch_time"] = epoch_time
        wandb.log(wandb_log, step=global_step)
        
        # Print progress
        if verbose:
            train_str = " | ".join([f"Train {k.capitalize()}: {v:.4f}" for k, v in train_metrics.items()])
            val_str = " | ".join([f"Val {k.capitalize()}: {v:.4f}" for k, v in val_metrics.items()])
            print(f"Epoch {epoch:3d}/{num_epochs} | {train_str} | {val_str} | Time: {epoch_time:.2f}s")
        
        # Early stopping and checkpointing based on validation loss
        current_val_loss = val_metrics.get('loss', float('inf'))
        is_better = current_val_loss < best_val_loss
        
        # Save checkpoint if improved or if this is the first epoch (fallback)
        if is_better or not best_model_saved:
            best_val_loss = current_val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }
            torch.save(checkpoint, best_model_path)
            best_model_saved = True
            logger.debug(f"Saved best model checkpoint to {best_model_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            # Also log accuracy if available
            if 'acc' in val_metrics:
                logger.info(f"Final validation accuracy: {val_metrics['acc']:.4f}")
            break
    
    # Load best model if it exists
    if best_model_saved and Path(best_model_path).exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {best_epoch}")
    else:
        logger.warning(f"No checkpoint found at {best_model_path}, using final model state")
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    # Also log best accuracy if it was tracked
    if f"val_acc" in history and len(history["val_acc"]) > 0:
        best_acc_idx = best_epoch - 1 if best_epoch > 0 else 0
        if best_acc_idx < len(history["val_acc"]):
            logger.info(f"Validation accuracy at best epoch: {history['val_acc'][best_acc_idx]:.4f}")
    
    return history


def test_model(
    model: torch.nn.Module,
    test_loader: Union[NeighborLoader, LinkNeighborLoader],
    objective: TrainingObjective,
    device: torch.device,
    disable_tqdm: bool = False
) -> Dict[str, float]:
    """
    Test the model on the test set using a specified training objective.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        objective: Training objective defining the task
        device: Device to evaluate on
        disable_tqdm: Whether to disable tqdm progress bars
    
    Returns:
        Dictionary of test metrics
    """
    logger.info("Testing model on test set...")
    
    test_metrics = evaluate(model, test_loader, objective, device, disable_tqdm)
    
    for key, value in test_metrics.items():
        logger.info(f"Test {key.capitalize()}: {value:.4f}")
    
    return test_metrics


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: NeighborLoader,
    device: torch.device,
    target_node_type: str = "paper",
    return_labels: bool = True,
    disable_tqdm: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract node embeddings from the trained model using NeighborLoader.
    
    Args:
        model: The trained model
        loader: NeighborLoader data loader
        device: Device to evaluate on
        target_node_type: Target node type for extraction
        return_labels: Whether to return labels alongside embeddings
        disable_tqdm: Whether to disable tqdm progress bars
    
    Returns:
        Tuple of (embeddings, labels) if return_labels=True, else (embeddings, None)
    """
    model.eval()
    
    embeddings_list = []
    labels_list = [] if return_labels else None
    
    for batch in tqdm(loader, desc="Extracting embeddings", leave=False, disable=disable_tqdm):
        batch = batch.to(device)
        
        # Forward pass
        out_dict, embeddings_dict = model(batch.x_dict, batch.edge_index_dict)
        
        # Extract only target nodes (not context nodes)
        batch_size = batch[target_node_type].batch_size
        embeddings = embeddings_dict[target_node_type][:batch_size]
        embeddings_list.append(embeddings.cpu())
        
        if return_labels and labels_list is not None and hasattr(batch[target_node_type], 'y'):
            y = batch[target_node_type].y[:batch_size]
            labels_list.append(y.cpu())
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0) if return_labels and labels_list else None
    
    return embeddings, labels
