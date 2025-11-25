"""
Downstream evaluation utilities for node property prediction and link prediction.
Evaluates the quality of learned embeddings by training simple classifier heads.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch_geometric.utils import negative_sampling
from graphssl.utils.data_utils import create_edge_splits
from graphssl.utils.training_utils import extract_embeddings
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
from pathlib import Path
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


class EdgeFeatureDataset(Dataset):
    """
    Dataset for link prediction that returns (edge_features, label).

    Edge features are computed on the fly from node embeddings:
    e.g. concat(z_src, z_dst).
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_labels: torch.Tensor,
    ):
        """
        Args:
            embeddings: Node embeddings of shape [N_nodes, D].
            edge_index: Edge index tensor of shape [2, E].
                        Each column is (src, dst) global node IDs.
            edge_labels: Edge labels tensor of shape [E], values in {0,1}.
        """
        assert edge_index.size(1) == edge_labels.size(0)
        self.embeddings = embeddings
        self.edge_index = edge_index
        self.edge_labels = edge_labels

    def __len__(self) -> int:
        """
        Returns:
            Total number of edges (positive + negative).
        """
        return self.edge_labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_feat: Tensor of shape [2 * D] (concatenated node embeddings).
            label:     Scalar tensor with value 0 or 1.
        """
        src = self.edge_index[0, idx]
        dst = self.edge_index[1, idx]
        y   = self.edge_labels[idx]

        z_src = self.embeddings[src]
        z_dst = self.embeddings[dst]
        edge_feat = torch.cat([z_src, z_dst], dim=-1)

        return edge_feat, y

def create_link_index_data(
    edge_index,
    num_nodes,
    num_neg_samples=1,
    device=None,
    max_edges=None,
    seed=42,
):
    """
    Create (positive + negative) edges for link prediction.
    When max_edges is given, only use a subset for fast local debugging.

    Args:
        edge_index:       [2, E] positive edges
        num_nodes:        number of nodes
        num_neg_samples:  negatives per positive
        max_edges:        if set, sample only this many positive edges
        seed:             reproducible sampling

    Returns:
        all_edges:  [2, N] sampled edges (pos + neg)
        all_labels: [N]   labels (1 for pos, 0 for neg)
    """
    if device is None:
        device = edge_index.device

    torch.manual_seed(seed)

    # -------------------------
    # STEP 1: Subsample POSITIVE edges if requested
    # -------------------------
    E = edge_index.size(1)

    if max_edges is not None and max_edges < E:
        perm = torch.randperm(E, device=device)[:max_edges]
        edge_index = edge_index[:, perm]
        num_pos = max_edges
    else:
        num_pos = E

    # -------------------------
    # STEP 2: Negative sampling
    # -------------------------
    num_neg = num_pos * num_neg_samples

    neg_edge = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method="sparse",
    ).to(device)

    # -------------------------
    # STEP 3: Labels and mixing
    # -------------------------
    pos_labels = torch.ones(num_pos, device=device)
    neg_labels = torch.zeros(num_neg, device=device)

    all_edges = torch.cat([edge_index, neg_edge], dim=1)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # Shuffle for safety
    perm = torch.randperm(all_edges.size(1), device=device)
    return all_edges[:, perm], all_labels[perm]

def evaluate_link_prediction(
    embeddings: torch.Tensor,
    train_edge_index: torch.Tensor,
    val_edge_index: torch.Tensor,
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
    max_edges: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate link prediction by training multiple classifier heads.
    
    Args:
        embeddings: Node embeddings [N_train, embedding_dim]
        train_edge_index: Training positive edges [2, num_train_edges]
        val_edge_index: Validation positive edges [2, num_val_edges]
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
    logger.info(f"Embedding dim: {embeddings.shape[1]}")
    logger.info(f"Negative sampling ratio: {num_neg_samples}")
    logger.info(f"Running {n_runs} independent trials...")
    
    # Store results from all runs
    test_accuracies = []
    test_losses = []
    num_nodes = embeddings.size(0)
    emb = embeddings.to(device)

    # Build pos+neg edges and labels for each split.
    train_edges, train_labels = create_link_index_data(
        train_edge_index, num_nodes, num_neg_samples, device=device, max_edges=max_edges
    )
    val_edges, val_labels = create_link_index_data(
        val_edge_index, num_nodes, num_neg_samples, device=device, max_edges=max_edges
    )
    test_edges, test_labels = create_link_index_data(
        test_edge_index, num_nodes, num_neg_samples, device=device, max_edges=max_edges
    )
    train_dataset = EdgeFeatureDataset(emb, train_edges, train_labels)
    val_dataset   = EdgeFeatureDataset(emb, val_edges, val_labels)
    test_dataset  = EdgeFeatureDataset(emb, test_edges, test_labels)

    # Create data loaders from datasets.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for run in range(n_runs):
        if verbose:
            logger.info(f"\nRun {run+1}/{n_runs}")

        # Create classifier (binary classification, output_dim=1)
        classifier = MLPClassifier(
            input_dim=emb.size(1) * 2,    # concat(z_src, z_dst)
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


def run_downstream_evaluation(
    args,
    data,
    device: torch.device,
    results_path: Path,
    num_classes: int
) -> None:
    """
    Run optional downstream evaluation:
    - Node property prediction
    - Link prediction
    """
    # Ensure embeddings are available for downstream evaluation
    embeddings_path = results_path / "embeddings.pt"

    # Load or extract embeddings - always define these variables in Step 10
    if embeddings_path.exists():
        # Load from saved file
        logger.info(f"Loading embeddings from: {embeddings_path}")
        embeddings_data = torch.load(embeddings_path)
        train_embeddings = embeddings_data['train_embeddings']
        global_embeddings = embeddings_data['global_embeddings']
        train_labels = embeddings_data['train_labels']
        val_embeddings = embeddings_data['val_embeddings']
        val_labels = embeddings_data['val_labels']
        test_embeddings = embeddings_data['test_embeddings']
        test_labels = embeddings_data['test_labels']
    else:
        raise ValueError("Embeddings are required for downstream evaluation but were not found.")

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
        # For link prediction, we need edge indices
        target_edge_type = tuple(args.target_edge_type.split(","))

        # Use edge splits from Step 2 if available, otherwise create new splits
        logger.info(
            f"Creating new edge splits for edge type: {target_edge_type}"
        )
        logger.warning(
            "Edge splits not available from Step 2 - creating new splits with same seed"
        )

        # Get full edge index and use create_edge_splits() function
        full_edge_index = data[target_edge_type].edge_index
        train_edge_index, val_edge_index, test_edge_index = create_edge_splits(
            edge_index=full_edge_index,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=args.seed,
        )
        
        link_results = evaluate_link_prediction(
            embeddings=global_embeddings,
            train_edge_index=train_edge_index,
            val_edge_index=val_edge_index,
            test_edge_index=test_edge_index,
            device=device,
            n_runs=args.downstream_n_runs,
            num_neg_samples=args.downstream_neg_samples,
            hidden_dim=args.downstream_hidden_dim,
            num_layers=args.downstream_num_layers,
            dropout=args.downstream_dropout,
            batch_size=args.downstream_batch_size,
            lr=args.downstream_lr,
            weight_decay=args.downstream_weight_decay,
            num_epochs=args.downstream_epochs,
            early_stopping_patience=args.downstream_patience,
            max_edges=args.downstream_max_edges,
            verbose=True
        )

        # Save link prediction results
        link_results_path = results_path / "downstream_link_results.pt"
        torch.save(link_results, link_results_path)
        logger.info(f"Link prediction results saved to: {link_results_path}")

