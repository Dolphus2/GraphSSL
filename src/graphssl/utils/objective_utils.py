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
loss_functions = ['mse', 'sce']

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

    def set_current_epoch(self, current_epoch: int) -> None:
        self.current_epoch = current_epoch


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


class SelfSupervisedTARPFP(TrainingObjective):
    """
    Self-supervised objective combining:
      - TAR (Target Attribute Restoration) on node features
      - PFP (Positional Feature Prediction) on positional embeddings

    Behavior controlled by lambda_tar / lambda_pfp:
      - lambda_tar > 0, lambda_pfp == 0 -> TAR-only (GraphMAE-style)
      - lambda_tar == 0, lambda_pfp > 0 -> PFP-only
      - lambda_tar > 0, lambda_pfp > 0  -> TAR + PFP (multi-task)
    """

    def __init__(
        self,
        target_node_type: str = "paper",
        attr_decoder: Optional[torch.nn.Module] = None,
        pos_decoder: Optional[torch.nn.Module] = None,
        mask_ratio_range: Tuple[float, float] = (0.2, 0.5),
        leave_prob: float = 0.1,
        replace_prob: float = 0.1,
        mask_token: Optional[torch.Tensor] = None,
        lambda_tar: float = 1.0,
        lambda_pfp: float = 1.0,
        num_epochs: int = 100,
    ):
        super().__init__(target_node_type)
        self.attr_decoder = attr_decoder
        self.pos_decoder = pos_decoder

        self.min_mask_ratio = mask_ratio_range[0]
        self.max_mask_ratio = mask_ratio_range[1]
        self.leave_prob = leave_prob
        self.replace_prob = replace_prob
        self.mask_token = mask_token

        self.lambda_tar = float(lambda_tar)
        self.lambda_pfp = float(lambda_pfp)
        self.num_epochs = num_epochs
        # At least one of them must be > 0
        assert self.lambda_tar + self.lambda_pfp > 0.0, \
            "At least one of lambda_tar or lambda_pfp must be > 0."

        # leave + replace <= 1 enforced via masking
        assert 0.0 <= self.leave_prob <= 1.0
        assert 0.0 <= self.replace_prob <= 1.0
        assert self.leave_prob + self.replace_prob <= 1.0, \
            "leave_prob + replace_prob must be <= 1."

    def _get_mask_ratio(self) -> float:
        """
        Linear scheduling δ(m):
        δ(m) = min(MAX_pa,
                   MIN_pa + (MAX_pa - MIN_pa) * m / M)
        """
        m = self.current_epoch
        M = max(1, self.num_epochs)
        min_pa = self.min_mask_ratio
        max_pa = self.max_mask_ratio

        delta = min_pa + (max_pa - min_pa) * (m / M)
        # ensure we never exceed MAX_pa
        delta = min(delta, max_pa)
        return float(delta)

    def step(
        self,
        model: torch.nn.Module,
        batch: Any,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        node_type = self.target_node_type
        x_orig = batch[node_type].x.clone()  # [N, F_attr]
        device = x_orig.device
        N, F_attr = x_orig.size()

        # Flags: which losses are active for this step?
        use_tar = (self.lambda_tar > 0.0)
        use_pfp = (self.lambda_pfp > 0.0)

        # -----------------------------
        # 1) Build (possibly) masked features for TAR
        # -----------------------------
        mask_ratio = 0.0
        num_masked_nodes = 0
        masked_nodes = None

        if use_tar:
            mask_ratio = self._get_mask_ratio()
            num_mask = max(1, int(mask_ratio * N))

            perm = torch.randperm(N, device=device)
            masked_nodes = perm[:num_mask]  # indices in [0, N)

            # partition masked_nodes into leave / replace / true-mask
            num_leave = int(self.leave_prob * num_mask)
            num_replace = int(self.replace_prob * num_mask)
            num_true_mask = num_mask - num_leave - num_replace
            # num_true_mask >= 0 guaranteed by assert in __init__

            shuffled = masked_nodes[torch.randperm(num_mask, device=device)]
            # leave_nodes = shuffled[:num_leave]
            replace_nodes = shuffled[num_leave:num_leave + num_replace]
            true_mask_nodes = shuffled[num_leave + num_replace:]

            x_masked = x_orig.clone()

            # REPLACE: copy features from random nodes
            if num_replace > 0:
                rand_src = torch.randint(0, N, (num_replace,), device=device)
                x_masked[replace_nodes] = x_orig[rand_src]

            # TRUE MASK: zero or mask token
            if num_true_mask > 0:
                if self.mask_token is None:
                    x_masked[true_mask_nodes] = 0.0
                else:
                    x_masked[true_mask_nodes] = self.mask_token.to(device)

            # leave_nodes: do nothing
            batch.x_dict[node_type] = x_masked
            num_masked_nodes = int(num_mask)
        else:
            # EVAL mode or TAR disabled → no masking
            batch.x_dict[node_type] = x_orig
            mask_ratio = 0.0
            num_masked_nodes = 0
            masked_nodes = None

        # -----------------------------
        # 2) Forward encoder
        # -----------------------------
        out_dict, embeddings_dict = model(batch.x_dict, batch.edge_index_dict)

        embeddings = embeddings_dict[node_type]  # [N, D]
        batch_size = embeddings.size(0)
        h = embeddings[:batch_size]             # [B, D]
        attr_target = x_orig[:batch_size]       # [B, F_attr]

        # --------------------------------------------------
        # 3) TAR loss
        #    - Train: on masked nodes only
        #    - Eval:  on all nodes
        # --------------------------------------------------
        if use_tar:
            if is_training and (masked_nodes is not None):
                # Train: masked subset V^e
                masked_in_batch = masked_nodes[masked_nodes < batch_size]
                idx = masked_in_batch
            else:
                # Eval: no masking → use all nodes
                idx = torch.arange(batch_size, device=device)

            if idx.numel() > 0:
                if self.attr_decoder is not None:
                    attr_pred = self.attr_decoder(h)  # [B, F_attr]
                else:
                    assert h.size(1) == F_attr, \
                        "attr_decoder is None but embedding_dim != feature_dim."
                    attr_pred = h

                x_t = attr_target[idx]         # targets
                x_p = attr_pred[idx]           # predictions

                # SCE = mean_v (1 - cos(x_t, x_p))^alpha
                loss_tar = sce_loss(x_p, x_t)
            else:
                loss_tar = torch.zeros(1, device=device)
        else:
            loss_tar = torch.zeros(1, device=device)

        # --------------------------------------------------
        # 4) PFP loss (cosine-style) on positional features
        #    - Train + Eval: always on all nodes (if lambda_pfp > 0)
        # --------------------------------------------------
        if use_pfp:
            if not hasattr(batch[node_type], "pos"):
                raise AttributeError(
                    f"Batch for node type '{node_type}' has no '.pos' attribute required for PFP."
                )

            # TAR + PFP training: PFP should see original (unmasked) features
            batch.x_dict[node_type] = x_orig

            # Run encoder again (shared weights, different input)
            _, emb_clean = model(batch.x_dict, batch.edge_index_dict)
            h_pfp = emb_clean[node_type]               # [B, D]

            pos = batch[node_type].pos.to(device)          # [N, F_pos]
            pos_target = pos[:h_pfp.size(0)]               # align with batch dimension

            # Decode positional features
            if self.pos_decoder is not None:
                pos_pred = self.pos_decoder(h_pfp)         # [B, F_pos]
            else:
                assert h_pfp.size(1) == pos_target.size(1), \
                    "pos_decoder is None but embedding_dim != pos_dim."
                pos_pred = h_pfp

            loss_pfp = sce_loss(pos_pred, pos_target)
        else:
            loss_pfp = torch.zeros(1, device=device)

        # -----------------------------
        # 5) Combine losses
        # -----------------------------
        loss = self.lambda_tar * loss_tar + self.lambda_pfp * loss_pfp

        metrics = {
            "loss": float(loss.item()),
            "loss_tar": float(loss_tar.item()),
            "loss_pfp": float(loss_pfp.item()),
            "mask_ratio": float(mask_ratio),
            "num_masked_nodes": int(num_masked_nodes),
            "total": int(N),
        }
        return loss, metrics

    def get_metric_names(self) -> list:
        return ["loss", "loss_tar", "loss_pfp", "mask_ratio", "num_masked_nodes", "acc"]

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
