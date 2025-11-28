"""
Model architectures for heterogeneous graph learning
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List

logger = logging.getLogger(__name__)


class HomogeneousGraphSAGE(nn.Module):
    """
    Homogeneous GraphSAGE model (will be converted to heterogeneous).
    Structured with separate encoder (GraphSAGE layers) and decoder (classifier).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
        aggr: str = "mean",
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (number of classes)
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout
        
        # Encoder: GraphSAGE convolution layers (flat structure for to_hetero)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # Batch normalization layers
        if use_batchnorm:
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Decoder: Classification head
        self.decoder = Linear(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        """
        Encode nodes to embeddings using GraphSAGE layers.
        
        Args:
            x: Node features
            edge_index: Edge indices
        
        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batchnorm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
    
    def decode(self, embeddings):
        """
        Decode embeddings to predictions.
        
        Args:
            embeddings: Node embeddings
        
        Returns:
            Logits
        """
        return self.decoder(embeddings)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
        
        Returns:
            Tuple of (logits, embeddings)
        """
        # Encode: Generate embeddings
        embeddings = self.encode(x, edge_index)
        
        # Decode: Generate predictions
        logits = self.decode(embeddings)
        
        return logits, embeddings


class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE model for OGB_MAG.
    Converts homogeneous model to heterogeneous using to_hetero.
    """
    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 128,
        out_channels: int = 349,  # Number of venues in OGB_MAG
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
        target_node_type: str = "paper",
        aggr: str = "mean",
        aggr_rel: str = "sum"
    ):
        """
        Args:
            metadata: Graph metadata (node_types, edge_types)
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (number of classes)
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
            target_node_type: The node type to predict
        """
        super().__init__()
        
        self.target_node_type = target_node_type
        self.hidden_channels = hidden_channels
        
        # Create homogeneous model
        # -1 is a placeholder, will be inferred by to_hetero
        self.model = HomogeneousGraphSAGE(
            in_channels=-1,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            aggr=aggr
        )
        
        # Convert to heterogeneous model
        self.model = to_hetero(self.model, metadata, aggr=aggr_rel)
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass for heterogeneous graph.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
        
        Returns:
            Tuple of (out_dict, embeddings_dict) where both are dictionaries
            containing outputs/embeddings for all node types
        """
        out_dict, embeddings_dict = self.model(x_dict, edge_index_dict)
        return out_dict, embeddings_dict
    
    def inference(self, x_dict, edge_index_dict):
        """
        Inference mode - returns predictions.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
        
        Returns:
            Predicted class probabilities
        """
        logits, _ = self.forward(x_dict, edge_index_dict)
        return F.softmax(logits, dim=-1)


def create_model(
    data: HeteroData,
    hidden_channels: int = 128,
    num_layers: int = 2,
    dropout: float = 0.5,
    use_batchnorm: bool = True,
    target_node_type: str = "paper",
    aggr: str = "mean",
    aggr_rel: str = "sum"
) -> HeteroGraphSAGE:
    """
    Create a heterogeneous GraphSAGE model.
    
    Args:
        data: HeteroData object (used to extract metadata)
        hidden_channels: Hidden layer dimension
        num_layers: Number of GraphSAGE layers
        dropout: Dropout rate
        use_batchnorm: Whether to use batch normalization
        target_node_type: The node type to predict
    
    Returns:
        HeteroGraphSAGE model
    """
    # Get number of classes
    num_classes = int(data[target_node_type].y.max().item() + 1)
    
    # Create model
    model = HeteroGraphSAGE(
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        target_node_type=target_node_type,
        aggr=aggr,
        aggr_rel=aggr_rel
    )
    
    logger.info("Model created:")
    logger.info(f"  Hidden channels: {hidden_channels}")
    logger.info(f"  Number of layers: {num_layers}")
    logger.info(f"  Output classes: {num_classes}")
    logger.debug(f"  Dropout: {dropout}")
    logger.debug(f"  Batch normalization: {use_batchnorm}")
    logger.debug(f"  Target node type: {target_node_type}")
    
    # Count parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.debug(f"  Total parameters: {total_params:,}")
    # logger.debug(f"  Trainable parameters: {trainable_params:,}")
    
    return model
