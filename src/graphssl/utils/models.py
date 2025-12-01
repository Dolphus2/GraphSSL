"""
Downstream model architectures for evaluation tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EdgeDecoder(MLPClassifier):
    """
    MLP-based edge decoder for link prediction.
    Inherits from MLPClassifier and handles edge-specific concatenation of source and destination embeddings.
    """
    
    def forward(self, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
        
        # Pass through parent MLPClassifier
        return super().forward(x)
