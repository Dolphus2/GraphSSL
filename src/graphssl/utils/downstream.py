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
    num_neg_samples: int = 1,
    all_positive_edges: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create link prediction dataset from embeddings and edges。
    负采样时显式避开所有已知正边（train+val+test）。
    """
    num_nodes = embeddings.shape[0]
    num_pos_edges = edge_index.shape[1]

    pos_src = embeddings[edge_index[0]]
    pos_dst = embeddings[edge_index[1]]
    pos_edge_features = torch.cat([pos_src, pos_dst], dim=1)
    pos_labels = torch.ones(num_pos_edges)

    num_neg_edges = num_pos_edges * num_neg_samples

    # 构建禁止采样的正边集合
    pos_forbidden = all_positive_edges if all_positive_edges is not None else edge_index
    pos_edge_set = set(zip(pos_forbidden[0].tolist(), pos_forbidden[1].tolist()))

    neg_edges = []
    tries = 0
    max_tries = num_neg_edges * 20 + 1000  # 简单防御，避免死循环
    while len(neg_edges) < num_neg_edges and tries < max_tries:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        tries += 1
        if src == dst:
            continue
        if (src, dst) in pos_edge_set:
            continue
        neg_edges.append((src, dst))
    if len(neg_edges) < num_neg_edges:
        logger.warning(
            f"Requested {num_neg_edges} negative edges, but only sampled {len(neg_edges)} "
            f"before hitting max tries ({max_tries})."
        )

    neg_edge_index = torch.tensor(neg_edges).t() if neg_edges else torch.empty((2, 0), dtype=torch.long)

    neg_src = embeddings[neg_edge_index[0]]
    neg_dst = embeddings[neg_edge_index[1]]
    neg_edge_features = torch.cat([neg_src, neg_dst], dim=1) if neg_edges else torch.empty((0, embeddings.size(1) * 2))
    neg_labels = torch.zeros(len(neg_edges))

    edge_features = torch.cat([pos_edge_features, neg_edge_features], dim=0)
    edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

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

    # 所有正边（用于负采样屏蔽）
    all_positive_edges = torch.cat([train_edge_index, val_edge_index, test_edge_index], dim=1)
    
    for run in range(n_runs):
        if verbose:
            logger.info(f"\nRun {run+1}/{n_runs}")
        
        # Create link prediction datasets (with different negative samples each run)
        train_features, train_labels = create_link_prediction_data(
            train_embeddings, train_edge_index, num_neg_samples, all_positive_edges=all_positive_edges
        )
        val_features, val_labels = create_link_prediction_data(
            val_embeddings, val_edge_index, num_neg_samples, all_positive_edges=all_positive_edges
        )
        test_features, test_labels = create_link_prediction_data(
            test_embeddings, test_edge_index, num_neg_samples, all_positive_edges=all_positive_edges
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


def evaluate_paper_field_multilabel(
    paper_embeddings: torch.Tensor,
    paper_field_labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
    n_runs: int = 5,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.5,
    batch_size: int = 1024,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    num_epochs: int = 50,
    early_stopping_patience: int = 5,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    针对 paper→field_of_study 的特例：多标签分类（预测一篇论文关联的所有学科）。
    """
    num_fields = paper_field_labels.size(1)

    def _split(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return tensor[mask]

    train_emb = _split(paper_embeddings, train_mask)
    val_emb = _split(paper_embeddings, val_mask)
    test_emb = _split(paper_embeddings, test_mask)
    train_lab = _split(paper_field_labels, train_mask)
    val_lab = _split(paper_field_labels, val_mask)
    test_lab = _split(paper_field_labels, test_mask)

    logger.info("=" * 80)
    logger.info("Downstream Task: Paper→Field-of-Study (Multi-Label)")
    logger.info("=" * 80)
    logger.info(f"Train papers: {train_emb.shape[0]}, Val: {val_emb.shape[0]}, Test: {test_emb.shape[0]}")
    logger.info(f"Fields of study: {num_fields}")

    def _micro_f1(pred: torch.Tensor, label: torch.Tensor) -> float:
        pred_bin = (pred > threshold).int()
        tp = (pred_bin & label.int()).sum().item()
        fp = (pred_bin & (1 - label.int())).sum().item()
        fn = ((1 - pred_bin) & label.int()).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    results = {
        "test_f1": [],
        "test_loss": [],
    }

    for run in range(n_runs):
        if verbose:
            logger.info(f"\nRun {run + 1}/{n_runs}")

        classifier = MLPClassifier(
            input_dim=paper_embeddings.shape[1],
            hidden_dim=hidden_dim,
            output_dim=num_fields,
            num_layers=num_layers,
            dropout=dropout,
            use_batchnorm=True,
        ).to(device)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

        train_dataset = TensorDataset(train_emb, train_lab)
        val_dataset = TensorDataset(val_emb, val_lab)
        test_dataset = TensorDataset(test_emb, test_lab)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        best_val_f1 = 0.0
        patience = 0

        for epoch in range(num_epochs):
            classifier.train()
            total_loss = 0.0
            total_samples = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = classifier(x)
                loss = F.binary_cross_entropy_with_logits(logits, y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

            # 验证
            classifier.eval()
            with torch.no_grad():
                all_logits = []
                all_labels = []
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = classifier(x)
                    all_logits.append(logits.cpu())
                    all_labels.append(y.cpu())
                if all_logits:
                    logits_cat = torch.cat(all_logits, dim=0)
                    labels_cat = torch.cat(all_labels, dim=0)
                    val_f1 = _micro_f1(torch.sigmoid(logits_cat), labels_cat)
                else:
                    val_f1 = 0.0

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience = 0
                best_state = classifier.state_dict()
            else:
                patience += 1
            if patience >= early_stopping_patience:
                break

        classifier.load_state_dict(best_state)
        classifier.eval()
        with torch.no_grad():
            test_logits = []
            test_labels = []
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = classifier(x)
                test_logits.append(logits.cpu())
                test_labels.append(y.cpu())
            if test_logits:
                logits_cat = torch.cat(test_logits, dim=0)
                labels_cat = torch.cat(test_labels, dim=0)
                test_f1 = _micro_f1(torch.sigmoid(logits_cat), labels_cat)
                test_loss = F.binary_cross_entropy_with_logits(logits_cat, labels_cat.float()).item()
            else:
                test_f1 = 0.0
                test_loss = 0.0

        results["test_f1"].append(test_f1)
        results["test_loss"].append(test_loss)
        wandb.log({
            f"downstream_fos/run_{run+1}/test_f1": test_f1,
            f"downstream_fos/run_{run+1}/test_loss": test_loss,
        })

    results_out = {
        "test_f1_mean": float(np.mean(results["test_f1"])),
        "test_f1_std": float(np.std(results["test_f1"])),
        "test_loss_mean": float(np.mean(results["test_loss"])),
        "test_loss_std": float(np.std(results["test_loss"])),
        "test_f1_runs": results["test_f1"],
        "test_loss_runs": results["test_loss"],
    }
    logger.info("\n" + "=" * 80)
    logger.info("Paper→Field-of-Study Results")
    logger.info("=" * 80)
    logger.info(f"Test F1 (micro): {results_out['test_f1_mean']:.4f} ± {results_out['test_f1_std']:.4f}")
    logger.info(f"Test Loss: {results_out['test_loss_mean']:.4f} ± {results_out['test_loss_std']:.4f}")
    wandb.log({
        "downstream_fos/test_f1_mean": results_out["test_f1_mean"],
        "downstream_fos/test_f1_std": results_out["test_f1_std"],
        "downstream_fos/test_loss_mean": results_out["test_loss_mean"],
        "downstream_fos/test_loss_std": results_out["test_loss_std"],
    })
    return results_out
