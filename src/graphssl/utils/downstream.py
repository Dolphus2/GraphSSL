"""
Downstream evaluation utilities for node property prediction and link prediction.
Evaluates the quality of learned embeddings by training simple classifier heads.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """
    Simple MLP classifier for downstream tasks.
    Takes fixed embeddings as input and outputs predictions.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batchnorm: bool = True
    ):
        """
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes or 1 for binary)
            num_layers: Number of hidden layers
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm and self.batch_norms is not None:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm and self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
        
        Returns:
            Output logits [batch_size, output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batchnorm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x


def train_downstream_classifier(
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    task_type: str = "classification",
    early_stopping_patience: int = 10,
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Train a downstream classifier on fixed embeddings.
    
    Args:
        classifier: The classifier model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        task_type: "classification" or "link_prediction"
        early_stopping_patience: Patience for early stopping
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing training history
    """
    classifier.train()
    
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
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out = classifier(batch_x)
            
            # Compute loss
            if task_type == "classification":
                loss = F.cross_entropy(out, batch_y)
                pred = out.argmax(dim=1)
            else:  # link_prediction (binary classification)
                loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch_y.float())
                pred = (out.squeeze() > 0).long()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item() * batch_x.size(0)
            train_correct += (pred == batch_y).sum().item()
            train_total += batch_x.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        val_loss, val_acc = evaluate_downstream_classifier(
            classifier, val_loader, device, task_type
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
def evaluate_downstream_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str = "classification"
) -> Tuple[float, float]:
    """
    Evaluate a downstream classifier.
    
    Args:
        classifier: The classifier model
        loader: Data loader
        device: Device to evaluate on
        task_type: "classification" or "link_prediction"
    
    Returns:
        Tuple of (loss, accuracy)
    """
    classifier.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        out = classifier(batch_x)
        
        # Compute loss and predictions
        if task_type == "classification":
            loss = F.cross_entropy(out, batch_y)
            pred = out.argmax(dim=1)
        else:  # link_prediction
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch_y.float())
            pred = (out.squeeze() > 0).long()
        
        # Accumulate metrics
        total_loss += loss.item() * batch_x.size(0)
        correct += (pred == batch_y).sum().item()
        total += batch_x.size(0)
    
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
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Train classifier
        history = train_downstream_classifier(
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            task_type="classification",
            early_stopping_patience=early_stopping_patience,
            verbose=verbose
        )
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_downstream_classifier(
            classifier, test_loader, device, task_type="classification"
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


def create_link_prediction_data(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    num_neg_samples: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create link prediction dataset from embeddings and edges.
    
    Args:
        embeddings: Node embeddings [N, embedding_dim]
        edge_index: Positive edges [2, num_edges]
        num_neg_samples: Number of negative samples per positive edge
    
    Returns:
        Tuple of (edge_features, edge_labels)
        edge_features: [num_samples, 2*embedding_dim] concatenated node embeddings
        edge_labels: [num_samples] binary labels (1 for positive, 0 for negative)
    """
    num_nodes = embeddings.shape[0]
    num_pos_edges = edge_index.shape[1]
    
    # Positive edge features (concatenate source and target embeddings)
    pos_src = embeddings[edge_index[0]]
    pos_dst = embeddings[edge_index[1]]
    pos_edge_features = torch.cat([pos_src, pos_dst], dim=1)
    pos_labels = torch.ones(num_pos_edges)
    
    # Generate negative edges (random pairs not in positive edges)
    num_neg_edges = num_pos_edges * num_neg_samples
    
    # Create set of positive edges for fast lookup
    pos_edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    
    neg_edges = []
    while len(neg_edges) < num_neg_edges:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        if src != dst and (src, dst) not in pos_edge_set:
            neg_edges.append((src, dst))
    
    neg_edge_index = torch.tensor(neg_edges).t()
    
    # Negative edge features
    neg_src = embeddings[neg_edge_index[0]]
    neg_dst = embeddings[neg_edge_index[1]]
    neg_edge_features = torch.cat([neg_src, neg_dst], dim=1)
    neg_labels = torch.zeros(num_neg_edges)
    
    # Combine positive and negative samples
    edge_features = torch.cat([pos_edge_features, neg_edge_features], dim=0)
    edge_labels = torch.cat([pos_labels, neg_labels], dim=0)
    
    # Shuffle
    perm = torch.randperm(edge_features.shape[0])
    edge_features = edge_features[perm]
    edge_labels = edge_labels[perm]
    
    return edge_features, edge_labels


def evaluate_link_prediction(
    train_embeddings: torch.Tensor,
    train_edge_index: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_edge_index: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_edge_index: torch.Tensor,
    device: torch.device,
    n_runs: int = 10,
    num_neg_samples: int = 1,
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
    Evaluate link prediction by training multiple classifier heads.
    
    Args:
        train_embeddings: Training node embeddings [N_train, embedding_dim]
        train_edge_index: Training positive edges [2, num_train_edges]
        val_embeddings: Validation node embeddings [N_val, embedding_dim]
        val_edge_index: Validation positive edges [2, num_val_edges]
        test_embeddings: Test node embeddings [N_test, embedding_dim]
        test_edge_index: Test positive edges [2, num_test_edges]
        device: Device to train on
        n_runs: Number of independent runs for uncertainty estimation
        num_neg_samples: Number of negative samples per positive edge
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
    logger.info("Downstream Task: Link Prediction")
    logger.info("="*80)
    logger.info(f"Train edges: {train_edge_index.shape[1]}")
    logger.info(f"Val edges: {val_edge_index.shape[1]}")
    logger.info(f"Test edges: {test_edge_index.shape[1]}")
    logger.info(f"Embedding dim: {train_embeddings.shape[1]}")
    logger.info(f"Negative sampling ratio: {num_neg_samples}")
    logger.info(f"Running {n_runs} independent trials...")
    
    # Store results from all runs
    test_accuracies = []
    test_losses = []
    
    for run in range(n_runs):
        if verbose:
            logger.info(f"\nRun {run+1}/{n_runs}")
        
        # Create link prediction datasets (with different negative samples each run)
        train_features, train_labels = create_link_prediction_data(
            train_embeddings, train_edge_index, num_neg_samples
        )
        val_features, val_labels = create_link_prediction_data(
            val_embeddings, val_edge_index, num_neg_samples
        )
        test_features, test_labels = create_link_prediction_data(
            test_embeddings, test_edge_index, num_neg_samples
        )
        
        # Create datasets and loaders
        train_dataset = TensorDataset(train_features, train_labels)
        val_dataset = TensorDataset(val_features, val_labels)
        test_dataset = TensorDataset(test_features, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create classifier (binary classification, output_dim=1)
        classifier = MLPClassifier(
            input_dim=train_features.shape[1],  # 2 * embedding_dim
            hidden_dim=hidden_dim,
            output_dim=1,  # Binary classification
            num_layers=num_layers,
            dropout=dropout,
            use_batchnorm=True
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Train classifier
        history = train_downstream_classifier(
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            task_type="link_prediction",
            early_stopping_patience=early_stopping_patience,
            verbose=verbose
        )
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_downstream_classifier(
            classifier, test_loader, device, task_type="link_prediction"
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
