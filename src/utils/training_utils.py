"""
Training and evaluation utilities for graph neural networks
"""
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from typing import Dict, Tuple
import time
from tqdm import tqdm


def train_epoch(
    model: torch.nn.Module,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_node_type: str = "paper"
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        target_node_type: Target node type for prediction
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out, _ = model(batch.x_dict, batch.edge_index_dict)
        
        # Get target nodes (the ones in the current batch)
        batch_size = batch[target_node_type].batch_size
        out = out[:batch_size]
        y = batch[target_node_type].y[:batch_size]
        
        # Compute loss
        loss = F.cross_entropy(out, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())
        total_examples += batch_size
    
    return total_loss / total_examples, total_correct / total_examples


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
    target_node_type: str = "paper",
    early_stopping_patience: int = 10,
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
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing training history
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*60}")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, target_node_type
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
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*60}\n")
    
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
    print(f"\n{'='*60}")
    print("Testing model on test set...")
    print(f"{'='*60}")
    
    test_loss, test_acc = evaluate(model, test_loader, device, target_node_type)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"{'='*60}\n")
    
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
