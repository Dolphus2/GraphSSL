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
    """
    
    def __init__(
        self,
        target_edge_type: Tuple[str, str, str],
        negative_sampling_ratio: float = 1.0,
        decoder: Optional[torch.nn.Module] = None
    ):
        """
        Args:
            target_edge_type: Edge type to reconstruct (src_type, relation, dst_type)
            negative_sampling_ratio: Ratio of negative to positive samples
            decoder: Optional edge decoder module (if None, uses dot product)
        """
        super().__init__(target_node_type=target_edge_type[0])
        self.target_edge_type = target_edge_type
        self.negative_sampling_ratio = negative_sampling_ratio
        self.decoder = decoder
    
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
        
        # Decode edge scores
        if self.decoder is not None:
            edge_scores = self.decoder(src_embeddings, dst_embeddings).squeeze()
        else:
            # Default: dot product similarity
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


class EdgeDecoder(torch.nn.Module):
    """
    Simple MLP-based edge decoder for link prediction.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1)
        )
    
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
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        return self.decoder(edge_features)


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
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss
    