"""
Training objective classes for different graph learning tasks.
Each objective encapsulates task-specific forward pass, loss computation, and backward pass.
"""
import logging
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)
loss_functions = ['mse', 'sce', 'mer', 'tar']

class TrainingObjective(ABC):
    """
    Abstract base class for training objectives.
    Each objective defines how to compute loss and perform training/evaluation steps.
    """
    
    def __init__(self, target_node_type: str = "paper"):
        """
        Args:
            target_node_type: The node type to predict (for node-level tasks)
        """
        self.target_node_type = target_node_type
    
    @abstractmethod
    def step(
        self,
        model: torch.nn.Module,
        batch: Any,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one training/evaluation step.
        
        Args:
            model: The model
            batch: Batch data
            is_training: Whether in training mode
        
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        pass
    
    def get_metric_names(self) -> list:
        """Return list of metric names tracked by this objective."""
        return ["loss"]


class SupervisedNodeClassification(TrainingObjective):
    """
    Supervised node classification objective.
    Predicts node labels (e.g., venue prediction for papers).
    """
    
    def __init__(self, target_node_type: str = "paper"):
        super().__init__(target_node_type)
    
    def step(
        self,
        model: torch.nn.Module,
        batch: Any,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass and loss computation for node classification.
        
        Args:
            model: The model
            batch: HeteroData batch
            is_training: Whether in training mode
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Forward pass
        out_dict, _ = model(batch.x_dict, batch.edge_index_dict)
        
        # Get target nodes only (seed nodes in the batch)
        batch_size = batch[self.target_node_type].batch_size
        logits = out_dict[self.target_node_type][:batch_size]
        y = batch[self.target_node_type].y[:batch_size]
        
        # Compute loss
        loss = F.cross_entropy(logits, y)
        
        # Compute accuracy
        pred = logits.argmax(dim=-1)
        correct = int((pred == y).sum())
        accuracy = correct / batch_size
        
        metrics = {
            "loss": loss.item(),
            "acc": accuracy,
            "correct": correct,
            "total": batch_size
        }
        
        return loss, metrics
    
    def get_metric_names(self) -> list:
        return ["loss", "acc"]


class DownstreamNodeClassification(TrainingObjective):
    """
    Downstream node classification objective for fixed embeddings.
    Used for evaluating the quality of learned embeddings by training a classifier head.
    """
    
    def __init__(self, classifier: torch.nn.Module, num_classes: int):
        """
        Args:
            classifier: The classifier model (e.g., MLP)
            num_classes: Number of classes for classification
        """
        super().__init__(target_node_type="embeddings")
        self.classifier = classifier
        self.num_classes = num_classes
    
    def step(
        self,
        model: torch.nn.Module,  # Not used, kept for interface compatibility
        batch: Any,  # Tuple of (embeddings, labels)
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass and loss computation for downstream node classification.
        
        Args:
            model: Not used (interface compatibility)
            batch: Tuple of (embeddings, labels) from DataLoader
            is_training: Whether in training mode
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Unpack batch
        embeddings, labels = batch
        
        # Forward pass through classifier
        logits = self.classifier(embeddings)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute accuracy
        pred = logits.argmax(dim=-1)
        correct = int((pred == labels).sum())
        batch_size = embeddings.size(0)
        accuracy = correct / batch_size
        
        metrics = {
            "loss": loss.item(),
            "acc": accuracy,
            "correct": correct,
            "total": batch_size
        }
        
        return loss, metrics
    
    def get_metric_names(self) -> list:
        return ["loss", "acc"]


class DownstreamLinkMulticlass(TrainingObjective):
    """
    Downstream link prediction as multiclass classification.
    Used for tasks like predicting field_of_study for papers by learning
    to map paper embeddings to field_of_study embeddings.
    """
    
    def __init__(
        self,
        decoder: torch.nn.Module,
        target_embeddings: torch.Tensor,
        device: torch.device
    ):
        """
        Args:
            decoder: MLP that projects source embeddings to target embedding space
            target_embeddings: All target node embeddings [num_targets, hidden_dim]
            device: Device for computation
        """
        super().__init__(target_node_type="embeddings")
        self.decoder = decoder
        self.target_embeddings = target_embeddings.to(device)
        self.num_classes = target_embeddings.size(0)
        self.eps = 1e-5
        self.neg_weight = 1
    
    def step(
        self,
        model: torch.nn.Module,  # Not used, kept for interface compatibility
        batch: Any,  # Tuple of (source_embeddings, edge_index, msg_pass_edge_index, num_source_nodes)
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass and loss computation for multi-label link prediction.
        
        Args:
            model: Not used (interface compatibility)
            batch: Tuple of (source_embeddings, edge_index, msg_pass_edge_index, num_source_nodes)
                   where edge_index[0] are source indices (relative to batch),
                   edge_index[1] are target indices (global),
                   msg_pass_edge_index contains edges used for message passing (to be masked out)
            is_training: Whether in training mode
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        source_embeddings, edge_index, msg_pass_edge_index, num_source_nodes = batch
        batch_size = source_embeddings.size(0)
        
        # Decode source embeddings to target embedding space
        projected_embeddings = self.decoder(source_embeddings)
        
        # Compute logits by dot product with all target embeddings
        # [batch_size, hidden_dim] @ [hidden_dim, num_targets] = [batch_size, num_targets]
        logits = torch.matmul(projected_embeddings, self.target_embeddings.T)
        
        # Apply sigmoid to get probabilities for adjacency matrix
        Ahat = torch.sigmoid(logits)
        
        # Calculate pos_weight as (num_negative_edges / num_positive_edges)
        num_positive_edges = edge_index.size(1)
        num_negative_edges = batch_size * self.num_classes - num_positive_edges
        pos_weight = num_negative_edges / (num_positive_edges + self.eps)
        
        # Get source and target indices from edge_index
        src_idxs_local = edge_index[0]  # Source indices (batch-local)
        trg_idxs_local = edge_index[1]  # Target indices (global)
        
        # Get message passing edge indices (to be masked out in loss)
        msg_pass_src_idxs = msg_pass_edge_index[0] if msg_pass_edge_index.size(1) > 0 else None
        msg_pass_trg_idxs = msg_pass_edge_index[1] if msg_pass_edge_index.size(1) > 0 else None
        
        # Compute negative log-likelihood loss
        neg_likelihood = self.calculate_neg_log_likelihood(
            Ahat, src_idxs_local, trg_idxs_local, pos_weight,
            msg_pass_src_idxs, msg_pass_trg_idxs
        )
        loss = neg_likelihood.mean()
        
        # Compute metrics
        # For multi-label, we compute precision/recall at a threshold
        with torch.no_grad():
            predictions = (Ahat > 0.5).float()
            
            # Create multi-hot target matrix
            targets = torch.zeros(batch_size, self.num_classes, device=logits.device)
            if edge_index.size(1) > 0:
                targets[edge_index[0], edge_index[1]] = 1.0
            
            # True positives, false positives, false negatives
            tp = (predictions * targets).sum()
            fp = (predictions * (1 - targets)).sum()
            fn = ((1 - predictions) * targets).sum()
            
            # Precision and recall
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
        print(f" total is {int(targets.sum().item())}")
        metrics = {
            "loss": loss.item(),
            "f1": f1.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "correct": int(tp.item()),
            "total": int(targets.sum().item())
        }
        
        return loss, metrics
    
    def get_metric_names(self) -> list:
        return ["loss", "f1", "precision", "recall"]
    
    def calculate_neg_log_likelihood(
        self, 
        Ahat, 
        src_idxs_local, 
        trg_idxs_local, 
        pos_weight,
        msg_pass_src_idxs=None,
        msg_pass_trg_idxs=None
    ) -> torch.Tensor:
        """
        Calculate negative log-likelihood loss with optional masking of message passing edges.
        
        Args:
            Ahat: Predicted adjacency matrix probabilities [batch_size, num_targets]
            src_idxs_local: Source indices for positive (supervision) edges
            trg_idxs_local: Target indices for positive (supervision) edges
            pos_weight: Weight for positive edges
            msg_pass_src_idxs: Optional source indices for message passing edges to mask out
            msg_pass_trg_idxs: Optional target indices for message passing edges to mask out
        
        Returns:
            Negative log-likelihood loss tensor [batch_size, num_targets]
        """
        Ahat = 1 - Ahat  # assume all edges are negative
        Ahat[src_idxs_local, trg_idxs_local] = (Ahat[src_idxs_local, trg_idxs_local] - 1) * (-1)  # Invert positive edges
        neg_likelihood = (-torch.log(Ahat + self.eps))
        neg_likelihood *= self.neg_weight
        neg_likelihood[src_idxs_local, trg_idxs_local] *= (pos_weight * (1 / self.neg_weight))
        
        # Mask out message passing edges by setting their loss to 0
        if msg_pass_src_idxs is not None and msg_pass_trg_idxs is not None:
            neg_likelihood[msg_pass_src_idxs, msg_pass_trg_idxs] = 0.0
        
        return neg_likelihood

class SupervisedLinkPrediction(TrainingObjective):
    """
    Supervised link prediction objective.
    Predicts existence of edges between nodes.
    """
    
    def __init__(
        self,
        target_edge_type: Tuple[str, str, str],
        decoder: Optional[torch.nn.Module] = None
    ):
        """
        Args:
            target_edge_type: Edge type to predict (src_type, relation, dst_type)
            decoder: Optional edge decoder module (if None, uses dot product)
        """
        super().__init__(target_node_type=target_edge_type[0])
        self.target_edge_type = target_edge_type
        self.decoder = decoder
    
    def step(
        self,
        model: torch.nn.Module,
        batch: Any,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass and loss computation for link prediction.
        
        Args:
            model: The model (encoder)
            batch: HeteroData batch with edge_label_index and edge_label
            is_training: Whether in training mode
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get node embeddings from encoder
        out_dict, embeddings_dict = model(batch.x_dict, batch.edge_index_dict)
        
        # Get edge indices and labels for target edge type
        edge_label_index = batch[self.target_edge_type].edge_label_index
        edge_label = batch[self.target_edge_type].edge_label
        
        # Get source and destination node types
        src_type, _, dst_type = self.target_edge_type
        
        # Get embeddings for source and destination nodes
        src_embeddings = embeddings_dict[src_type][edge_label_index[0]]
        dst_embeddings = embeddings_dict[dst_type][edge_label_index[1]]
        
        # Decode edge scores
        if self.decoder is not None:
            edge_scores = self.decoder(src_embeddings, dst_embeddings).squeeze()
        else:
            # Default: dot product similarity (logits)
            edge_scores = (src_embeddings * dst_embeddings).sum(dim=-1)
        
        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(edge_scores, edge_label.float())
        
        # Compute accuracy
        pred = (edge_scores > 0).float()
        correct = int((pred == edge_label).sum())
        accuracy = correct / edge_label.size(0)
        
        metrics = {
            "loss": loss.item(),
            "acc": accuracy,
            "correct": correct,
            "total": edge_label.size(0)
        }
        
        return loss, metrics
    
    def get_metric_names(self) -> list:
        return ["loss", "acc"]


class SelfSupervisedNodeReconstruction(TrainingObjective):
    """
    Self-supervised node feature reconstruction objective.
    Reconstructs masked node features from graph structure.
    """
    
    def __init__(
        self,
        target_node_type: str = "paper",
        mask_ratio: float = 0.5, # a large mask ratio like in GraphMAE
        decoder: Optional[torch.nn.Module] = None,
        loss_fn = "mse"
    ):
        """
        Args:
            target_node_type: Node type to reconstruct features for
            mask_ratio: Ratio of features to mask
            decoder: Optional decoder module (if None, uses linear projection)
        """
        super().__init__(target_node_type)
        self.mask_ratio = mask_ratio
        self.decoder = decoder
        assert loss_fn in loss_functions
        self.loss_fn = loss_fn
    
    def step(
        self,
        model: torch.nn.Module,
        batch: Any,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass and loss computation for node feature reconstruction.
        
        Args:
            model: The model
            batch: HeteroData batch
            is_training: Whether in training mode
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Store original features
        original_features = batch[self.target_node_type].x.clone()
        
        # Mask features (only during training)
        if is_training:
            mask = torch.rand(original_features.size()) < self.mask_ratio # Check that this masks out a node completely
            mask = mask.to(original_features.device)
            batch.x_dict[self.target_node_type] = original_features.masked_fill(mask, 0.0)
        
        # Forward pass through encoder
        out_dict, embeddings_dict = model(batch.x_dict, batch.edge_index_dict)
        
        # Get target node embeddings
        batch_size = batch[self.target_node_type].batch_size
        embeddings = embeddings_dict[self.target_node_type][:batch_size]
        target_features = original_features[:batch_size]
        
        # Decode to reconstruct features
        if self.decoder is not None:
            reconstructed = self.decoder(embeddings)
        else:
            # Default: use embeddings directly (assumes same dimension)
            reconstructed = embeddings
        
        # Compute reconstruction loss
        if self.loss_fn == "sce":
            loss = sce_loss(reconstructed, target_features)
        else:
            # (MSE)
            loss = F.mse_loss(reconstructed, target_features)
        
        metrics = {
            "loss": loss.item(),
            self.loss_fn: loss.item()
        }
        
        return loss, metrics
    
    def get_metric_names(self) -> list:
        return ["loss", self.loss_fn]


class SelfSupervisedEdgeReconstruction(TrainingObjective):
    """
    Self-supervised edge reconstruction objective.
    Reconstructs masked edges from node embeddings (link prediction as pretext task).
    Supports multiple loss functions including HGMAE's combined loss (MER + TAR + PFP).
    """
    
    def __init__(
        self,
        target_edge_type: Tuple[str, str, str],
        negative_sampling_ratio: float = 1.0,
        decoder: Optional[torch.nn.Module] = None,
        loss_fn: str = "bce",
        mer_weight: float = 1.0,
        tar_weight: float = 1.0,
        pfp_weight: float = 1.0,
        tar_temperature: float = 0.5
    ):
        """
        Args:
            target_edge_type: Edge type to reconstruct (src_type, relation, dst_type)
            negative_sampling_ratio: Ratio of negative to positive samples
            decoder: Optional edge decoder module (if None, uses dot product)
            loss_fn: Loss function to use. Options:
                - "bce": Binary cross-entropy (default)
                - "mer": Masked Edge Reconstruction
                - "tar": Topology-Aware Reconstruction
                - "pfp": Preference-based Feature Propagation
                - "combined_loss": Combined loss (MER + TAR + PFP)
            mer_weight: Weight for MER loss in combined loss (default: 1.0)
            tar_weight: Weight for TAR loss in combined loss (default: 1.0)
            pfp_weight: Weight for PFP loss in combined loss (default: 1.0)
            tar_temperature: Temperature parameter for TAR loss (default: 0.5)
        """
        super().__init__(target_node_type=target_edge_type[0])
        self.target_edge_type = target_edge_type
        self.negative_sampling_ratio = negative_sampling_ratio
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.mer_weight = mer_weight
        self.tar_weight = tar_weight
        self.pfp_weight = pfp_weight
        self.tar_temperature = tar_temperature
        
        # Validate loss function
        valid_loss_fns = ["bce", "mer", "tar", "pfp", "combined_loss"]
        if loss_fn not in valid_loss_fns:
            raise ValueError(f"Invalid loss_fn '{loss_fn}'. Must be one of {valid_loss_fns}")
    
    def step(
        self,
        model: torch.nn.Module,
        batch: Any,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass and loss computation for edge reconstruction.
        
        Args:
            model: The model (encoder)
            batch: HeteroData batch with edge_label_index and edge_label
            is_training: Whether in training mode
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get node embeddings from encoder
        _, embeddings_dict = model(batch.x_dict, batch.edge_index_dict)
        
        # Get edge indices and labels
        # Note: LinkNeighborLoader should provide edge_label_index and edge_label
        edge_label_index = batch[self.target_edge_type].edge_label_index
        edge_label = batch[self.target_edge_type].edge_label
        
        # Get source and destination node types
        src_type, _, dst_type = self.target_edge_type
        
        # Get embeddings for source and destination nodes
        src_embeddings = embeddings_dict[src_type][edge_label_index[0]]
        dst_embeddings = embeddings_dict[dst_type][edge_label_index[1]]
        
        # Get original features for PFP loss
        src_features = batch.x_dict[src_type][edge_label_index[0]]
        dst_features = batch.x_dict[dst_type][edge_label_index[1]]
        
        # Decode edge scores
        if self.decoder is not None:
            edge_scores = self.decoder(src_embeddings, dst_embeddings).squeeze()
        else:
            # Default: dot product similarity
            edge_scores = (src_embeddings * dst_embeddings).sum(dim=-1)
        
        # Compute loss based on loss_fn
        metrics = {}
        
        if self.loss_fn == "bce":
            # Standard binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(edge_scores, edge_label.float())
            metrics["bce"] = loss.item()
            
        elif self.loss_fn == "mer":
            # Masked Edge Reconstruction loss
            loss = mer_loss(edge_scores, edge_label)
            metrics["mer"] = loss.item()
            
        elif self.loss_fn == "tar":
            # Topology-Aware Reconstruction loss
            loss = tar_loss(src_embeddings, dst_embeddings, edge_label, 
                          temperature=self.tar_temperature)
            metrics["tar"] = loss.item()
            
        elif self.loss_fn == "pfp":
            # Preference-based Feature Propagation loss
            loss = pfp_loss(src_features, dst_features, edge_label)
            metrics["pfp"] = loss.item()
            
        elif self.loss_fn == "combined_loss":
            # Combined loss (MER + TAR + PFP)
            mer = mer_loss(edge_scores, edge_label)
            tar = tar_loss(src_embeddings, dst_embeddings, edge_label,
                         temperature=self.tar_temperature)
            pfp = pfp_loss(src_features, dst_features, edge_label)
            
            loss = (self.mer_weight * mer + 
                   self.tar_weight * tar + 
                   self.pfp_weight * pfp)
            
            metrics["mer"] = mer.item()
            metrics["tar"] = tar.item()
            metrics["pfp"] = pfp.item()
            metrics["combined_loss_total"] = loss.item()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")
        
        # Compute accuracy
        pred = (edge_scores > 0).float()
        correct = int((pred == edge_label).sum())
        accuracy = correct / edge_label.size(0)
        
        metrics["loss"] = loss.item()
        metrics["acc"] = accuracy
        metrics["correct"] = correct
        metrics["total"] = edge_label.size(0)
        
        return loss, metrics
    
    def get_metric_names(self) -> list:
        if self.loss_fn == "combined_loss":
            return ["loss", "acc", "mer", "tar", "pfp", "combined_loss_total"]
        else:
            return ["loss", "acc", self.loss_fn]


class EdgeDecoder(torch.nn.Module):
    """
    MLP-based edge decoder for link prediction with variable layers and batch normalization.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batchnorm: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension for MLP layers
            num_layers: Number of hidden layers
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout
        
        # Build MLP layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if use_batchnorm else None
        
        # Input layer (concatenated src + dst embeddings)
        input_dim = hidden_dim * 2
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        if use_batchnorm and self.batch_norms is not None:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm and self.batch_norms is not None:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode edge scores from source and destination embeddings.
        
        Args:
            src_embeddings: Source node embeddings [num_edges, hidden_dim]
            dst_embeddings: Destination node embeddings [num_edges, hidden_dim]
        
        Returns:
            Edge scores [num_edges, 1]
        """
        # Concatenate source and destination embeddings
        x = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        
        # Pass through MLP layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batchnorm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return x


class FeatureDecoder(torch.nn.Module):
    """
    Simple MLP-based feature decoder for node feature reconstruction.
    """
    
    def __init__(self, hidden_dim: int, feature_dim: int, dropout: float = 0.5):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode features from embeddings.
        
        Args:
            embeddings: Node embeddings [num_nodes, hidden_dim]
        
        Returns:
            Reconstructed features [num_nodes, feature_dim]
        """
        return self.decoder(embeddings)

def sce_loss(x, y, alpha=3):
    """
    Scaled Cosine Error loss from GraphMAE.
    
    Args:
        x: Reconstructed features [num_nodes, feature_dim]
        y: Target features [num_nodes, feature_dim]
        alpha: Scaling factor (default: 3)
    
    Returns:
        Scalar loss value
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mer_loss(edge_scores, edge_labels, mask=None):
    """
    Masked Edge Reconstruction (MER) loss from HGMAE.
    
    Binary cross-entropy loss for edge reconstruction with optional masking.
    
    Args:
        edge_scores: Predicted edge scores [num_edges]
        edge_labels: Ground truth edge labels [num_edges]
        mask: Optional mask for selecting specific edges [num_edges]
    
    Returns:
        Scalar loss value
    """
    loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels.float(), reduction='none')
    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-8)
    else:
        loss = loss.mean()
    return loss


def tar_loss(src_embeddings, dst_embeddings, edge_labels, temperature=0.5):
    """
    Topology-Aware Reconstruction (TAR) loss from HGMAE.
    
    Contrastive loss that encourages connected nodes to have similar embeddings
    while pushing disconnected nodes apart.
    
    Args:
        src_embeddings: Source node embeddings [num_edges, hidden_dim]
        dst_embeddings: Destination node embeddings [num_edges, hidden_dim]
        edge_labels: Binary edge labels (1 for positive, 0 for negative) [num_edges]
        temperature: Temperature parameter for contrastive learning
    
    Returns:
        Scalar loss value
    """
    # Normalize embeddings
    src_embeddings = F.normalize(src_embeddings, p=2, dim=-1)
    dst_embeddings = F.normalize(dst_embeddings, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = (src_embeddings * dst_embeddings).sum(dim=-1) / temperature
    
    # Positive pairs: maximize similarity (minimize negative log-likelihood)
    # Negative pairs: minimize similarity (maximize negative log-likelihood)
    pos_mask = edge_labels == 1
    neg_mask = edge_labels == 0
    
    loss = 0.0
    if pos_mask.sum() > 0:
        pos_sim = similarity[pos_mask]
        # For positive pairs, we want high similarity
        pos_loss = -torch.log(torch.sigmoid(pos_sim) + 1e-8).mean()
        loss += pos_loss
    
    if neg_mask.sum() > 0:
        neg_sim = similarity[neg_mask]
        # For negative pairs, we want low similarity
        neg_loss = -torch.log(1 - torch.sigmoid(neg_sim) + 1e-8).mean()
        loss += neg_loss
    
    return loss


def pfp_loss(src_features, dst_features, edge_labels, reconstructed_src=None, reconstructed_dst=None):
    """
    Preference-based Feature Propagation (PFP) loss from HGMAE.
    
    Encourages the model to preserve feature similarity between connected nodes.
    If reconstructed features are provided, uses them; otherwise uses original features.
    
    Args:
        src_features: Source node features [num_edges, feature_dim]
        dst_features: Destination node features [num_edges, feature_dim]
        edge_labels: Binary edge labels (1 for positive, 0 for negative) [num_edges]
        reconstructed_src: Optional reconstructed source features [num_edges, feature_dim]
        reconstructed_dst: Optional reconstructed destination features [num_edges, feature_dim]
    
    Returns:
        Scalar loss value
    """
    # Use reconstructed features if available, otherwise use original
    if reconstructed_src is not None and reconstructed_dst is not None:
        src_to_use = reconstructed_src
        dst_to_use = reconstructed_dst
    else:
        src_to_use = src_features
        dst_to_use = dst_features
    
    # Normalize features
    src_norm = F.normalize(src_to_use, p=2, dim=-1)
    dst_norm = F.normalize(dst_to_use, p=2, dim=-1)
    
    # Compute feature similarity
    feature_sim = (src_norm * dst_norm).sum(dim=-1)
    
    # For positive edges, maximize feature similarity
    # For negative edges, this term should be small
    pos_mask = edge_labels == 1
    
    if pos_mask.sum() > 0:
        # For positive edges, minimize 1 - similarity (maximize similarity)
        pos_feature_sim = feature_sim[pos_mask]
        loss = (1 - pos_feature_sim).mean()
    else:
        # No positive edges - return zero tensor
        loss = torch.tensor(0.0, device=src_features.device, dtype=src_features.dtype)
    
    return loss
    