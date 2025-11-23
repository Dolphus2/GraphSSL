"""
Training and evaluation utilities for graph neural networks
"""
import logging
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from typing import Dict, Tuple, Optional
import time
from tqdm import tqdm
from pathlib import Path
import wandb

logger = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_interval: int = 20,
    target_node_type: str = "paper"
) -> Tuple[float, float, int]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Epoch num
        target_node_type: Target node type for prediction
    
    Returns:
        Tuple of (average loss, accuracy, global step)
    """
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_examples = 0
    running_loss, running_acc = 0.0, 0.0
    num_logged = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out, _ = model(batch.x_dict, batch.edge_index_dict) # out is class logits because of final linear layer
        
        # Get target nodes (the ones in the current batch)
        batch_size = batch[target_node_type].batch_size
        out = out[:batch_size]  # Out has already been filtered in forward pass out[self.target_node_type]
        y = batch[target_node_type].y[:batch_size]
        
        # Compute loss
        loss = F.cross_entropy(out, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        batch_loss = loss.item()
        batch_correct = int((out.argmax(dim=-1) == y).sum())

        total_loss += batch_loss * batch_size
        total_correct += batch_correct
        total_examples += batch_size

        global_step += 1
        running_loss += batch_loss
        running_acc += batch_correct / max(batch_size, 1)
        num_logged += 1

        if global_step % log_interval == 0:
            avg_loss = running_loss / num_logged
            avg_acc = running_acc / num_logged
            running_loss, running_acc = 0.0, 0.0
            num_logged = 0

            # current learning rate (if no scheduler, this still works)
            current_lr = optimizer.param_groups[0]["lr"]

            # gpu memory (if on cuda)
            gpu_mem = None
            if device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated(device) / 1024**2

            log_dict = {
                "train/loss": avg_loss,
                "train/acc": avg_acc,
                "train/lr": current_lr,
                "train/epoch": epoch,
            }
            if gpu_mem is not None:
                log_dict["sys/gpu_mem_mb"] = gpu_mem

            wandb.log(log_dict, step=global_step)
    
    return total_loss / total_examples, total_correct / total_examples, global_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    device: torch.device,
    target_node_type: str = "paper"
) -> Tuple[float, float]:
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        loader: Data loader (validation or test)
        device: Device to evaluate on
        target_node_type: Target node type for prediction
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        
        # Forward pass
        out, _ = model(batch.x_dict, batch.edge_index_dict)
        
        # Get target nodes
        batch_size = batch[target_node_type].batch_size
        out = out[:batch_size]
        y = batch[target_node_type].y[:batch_size]
        
        # Compute loss
        loss = F.cross_entropy(out, y)
        
        # Track metrics
        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())
        total_examples += batch_size
    
    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    val_loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    log_interval: int = 20,
    target_node_type: str = "paper",
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Train the model with early stopping.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        target_node_type: Target node type for prediction
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory to save model checkpoints (if None, saves to current directory)
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing training history
    """
    
    # Setup checkpoint directory
    if checkpoint_dir is not None:
        best_model_path = Path(checkpoint_dir) / "best_model.pt"
    else:
        best_model_path = "best_model.pt"
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    global_step = 0
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, global_step, log_interval, target_node_type
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, device, target_node_type
        )
        
        epoch_time = time.time() - start_time
        
        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        wandb.log(
            {
                "val/acc": float(val_acc),
                "val/loss": float(val_loss),
                "train/epoch_loss": float(train_loss),
                "train/epoch_acc": float(train_acc),
                "val/epoch": epoch,
                "epoch_time": epoch_time
            },
            step=global_step
        )
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, best_model_path)
            logger.debug(f"Saved best model checkpoint to {best_model_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    return history


def test_model(
    model: torch.nn.Module,
    test_loader: NeighborLoader,
    device: torch.device,
    target_node_type: str = "paper"
) -> Tuple[float, float]:
    """
    Test the model on the test set.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        device: Device to evaluate on
        target_node_type: Target node type for prediction
    
    Returns:
        Tuple of (test loss, test accuracy)
    """
    logger.info("Testing model on test set...")
    
    test_loss, test_acc = evaluate(model, test_loader, device, target_node_type)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    return test_loss, test_acc


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: NeighborLoader,
    device: torch.device,
    target_node_type: str = "paper"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract node embeddings from the trained model.
    
    Args:
        model: The trained model
        loader: Data loader
        device: Device to evaluate on
        target_node_type: Target node type for prediction
    
    Returns:
        Tuple of (embeddings, labels)
    """
    model.eval()
    
    embeddings_list = []
    labels_list = []
    
    for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
        batch = batch.to(device)
        
        # Forward pass
        _, embeddings = model(batch.x_dict, batch.edge_index_dict)
        
        # Get target nodes
        batch_size = batch[target_node_type].batch_size
        embeddings = embeddings[:batch_size]
        y = batch[target_node_type].y[:batch_size]
        
        embeddings_list.append(embeddings.cpu())
        labels_list.append(y.cpu())
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return embeddings, labels
