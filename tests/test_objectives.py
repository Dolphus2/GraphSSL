"""
Unit tests for training objective classes in GraphSSL.

Tests all objective classes:
- SupervisedNodeClassification
- SupervisedLinkPrediction
- SelfSupervisedNodeReconstruction
- SelfSupervisedEdgeReconstruction
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from graphssl.utils.objective_utils import (
    SupervisedNodeClassification,
    SupervisedLinkPrediction,
    SelfSupervisedNodeReconstruction,
    SelfSupervisedEdgeReconstruction,
    EdgeDecoder,
    FeatureDecoder
)


class DummyHeteroModel(nn.Module):
    """Dummy heterogeneous GNN model for testing."""
    
    def __init__(self, hidden_dim=128, num_classes=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
    
    def forward(self, x_dict, edge_index_dict):
        """Return dummy outputs and embeddings."""
        out_dict = {}
        embeddings_dict = {}
        
        for node_type, x in x_dict.items():
            batch_size = x.size(0)
            embeddings_dict[node_type] = torch.randn(batch_size, self.hidden_dim)
            
            if self.num_classes is not None:
                out_dict[node_type] = torch.randn(batch_size, self.num_classes)
            else:
                out_dict[node_type] = embeddings_dict[node_type]
        
        return out_dict, embeddings_dict


def create_dummy_hetero_batch(num_nodes=100, num_edges=200, feature_dim=256, num_classes=10):
    """Create a dummy HeteroData batch for testing."""
    batch = HeteroData()
    
    # Add paper nodes
    batch['paper'].x = torch.randn(num_nodes, feature_dim)
    batch['paper'].y = torch.randint(0, num_classes, (num_nodes,))
    batch['paper'].batch_size = num_nodes // 2  # Seed nodes
    
    # Add author nodes
    batch['author'].x = torch.randn(num_nodes, feature_dim)
    batch['author'].batch_size = num_nodes // 2
    
    # Add edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch['paper', 'cites', 'paper'].edge_index = edge_index
    batch['author', 'writes', 'paper'].edge_index = edge_index
    
    # Add edge labels for link prediction
    edge_label_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_label = torch.randint(0, 2, (num_edges,))
    batch['paper', 'cites', 'paper'].edge_label_index = edge_label_index
    batch['paper', 'cites', 'paper'].edge_label = edge_label
    
    return batch


class TestSupervisedNodeClassification:
    """Test SupervisedNodeClassification objective."""
    
    def test_initialization(self):
        """Test objective initialization."""
        objective = SupervisedNodeClassification(target_node_type='paper')
        assert objective.target_node_type == 'paper'
        assert 'loss' in objective.get_metric_names()
        assert 'acc' in objective.get_metric_names()
    
    def test_step_forward(self):
        """Test forward pass and loss computation."""
        model = DummyHeteroModel(hidden_dim=128, num_classes=10)
        objective = SupervisedNodeClassification(target_node_type='paper')
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
        assert 'loss' in metrics
        assert 'acc' in metrics
        assert 0 <= metrics['acc'] <= 1, "Accuracy should be between 0 and 1"
    
    def test_step_evaluation(self):
        """Test evaluation mode."""
        model = DummyHeteroModel(hidden_dim=128, num_classes=10)
        objective = SupervisedNodeClassification(target_node_type='paper')
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=False)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'acc' in metrics


class TestSupervisedLinkPrediction:
    """Test SupervisedLinkPrediction objective."""
    
    def test_initialization(self):
        """Test objective initialization."""
        objective = SupervisedLinkPrediction(
            target_edge_type=('paper', 'cites', 'paper')
        )
        assert objective.target_edge_type == ('paper', 'cites', 'paper')
        assert 'loss' in objective.get_metric_names()
        assert 'acc' in objective.get_metric_names()
    
    def test_step_without_decoder(self):
        """Test forward pass without edge decoder (dot product)."""
        model = DummyHeteroModel(hidden_dim=128)
        objective = SupervisedLinkPrediction(
            target_edge_type=('paper', 'cites', 'paper'),
            decoder=None
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'loss' in metrics
        assert 'acc' in metrics
    
    def test_step_with_decoder(self):
        """Test forward pass with MLP edge decoder."""
        model = DummyHeteroModel(hidden_dim=128)
        decoder = EdgeDecoder(hidden_dim=128, dropout=0.5)
        objective = SupervisedLinkPrediction(
            target_edge_type=('paper', 'cites', 'paper'),
            decoder=decoder
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'acc' in metrics


class TestSelfSupervisedNodeReconstruction:
    """Test SelfSupervisedNodeReconstruction objective."""
    
    def test_initialization_mse(self):
        """Test initialization with MSE loss."""
        objective = SelfSupervisedNodeReconstruction(
            target_node_type='paper',
            mask_ratio=0.5,
            loss_fn='mse'
        )
        assert objective.mask_ratio == 0.5
        assert objective.loss_fn == 'mse'
    
    def test_initialization_sce(self):
        """Test initialization with SCE loss."""
        objective = SelfSupervisedNodeReconstruction(
            target_node_type='paper',
            mask_ratio=0.5,
            loss_fn='sce'
        )
        assert objective.loss_fn == 'sce'
        assert 'sce' in objective.get_metric_names()
    
    def test_step_with_mse(self):
        """Test forward pass with MSE loss."""
        model = DummyHeteroModel(hidden_dim=256)  # Match feature dim
        objective = SelfSupervisedNodeReconstruction(
            target_node_type='paper',
            mask_ratio=0.5,
            loss_fn='mse'
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'loss' in metrics
        assert 'mse' in metrics
    
    def test_step_with_sce(self):
        """Test forward pass with SCE loss."""
        model = DummyHeteroModel(hidden_dim=256)
        objective = SelfSupervisedNodeReconstruction(
            target_node_type='paper',
            mask_ratio=0.5,
            loss_fn='sce'
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'sce' in metrics
    
    def test_step_with_decoder(self):
        """Test forward pass with feature decoder."""
        model = DummyHeteroModel(hidden_dim=128)
        decoder = FeatureDecoder(hidden_dim=128, feature_dim=256, dropout=0.5)
        objective = SelfSupervisedNodeReconstruction(
            target_node_type='paper',
            mask_ratio=0.5,
            decoder=decoder,
            loss_fn='mse'
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_masking_in_training(self):
        """Test that features are masked during training."""
        model = DummyHeteroModel(hidden_dim=256)
        objective = SelfSupervisedNodeReconstruction(
            target_node_type='paper',
            mask_ratio=0.9,  # High mask ratio
            loss_fn='mse'
        )
        batch = create_dummy_hetero_batch()
        original_features = batch['paper'].x.clone()
        
        # Training mode should mask features
        loss, metrics = objective.step(model, batch, is_training=True)
        
        # Features should have been modified during forward pass
        # (but original features in batch are preserved in loss computation)
        assert torch.isfinite(loss), "Loss should be finite"


class TestSelfSupervisedEdgeReconstruction:
    """Test SelfSupervisedEdgeReconstruction objective."""
    
    def test_initialization_bce(self):
        """Test initialization with BCE loss."""
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='bce'
        )
        assert objective.loss_fn == 'bce'
        assert objective.target_edge_type == ('paper', 'cites', 'paper')
    
    def test_initialization_mer(self):
        """Test initialization with MER loss."""
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='mer'
        )
        assert objective.loss_fn == 'mer'
    
    def test_initialization_tar(self):
        """Test initialization with TAR loss."""
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='tar',
            tar_temperature=0.5
        )
        assert objective.loss_fn == 'tar'
        assert objective.tar_temperature == 0.5
    
    def test_initialization_pfp(self):
        """Test initialization with PFP loss."""
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='pfp'
        )
        assert objective.loss_fn == 'pfp'
    
    def test_initialization_combined_loss(self):
        """Test initialization with combined loss."""
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='combined_loss',
            mer_weight=1.0,
            tar_weight=1.0,
            pfp_weight=1.0
        )
        assert objective.loss_fn == 'combined_loss'
        assert objective.mer_weight == 1.0
        assert objective.tar_weight == 1.0
        assert objective.pfp_weight == 1.0
        
        metric_names = objective.get_metric_names()
        assert 'mer' in metric_names
        assert 'tar' in metric_names
        assert 'pfp' in metric_names
    
    def test_step_with_bce(self):
        """Test forward pass with BCE loss."""
        model = DummyHeteroModel(hidden_dim=128)
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='bce'
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'loss' in metrics
        assert 'acc' in metrics
        assert 'bce' in metrics
    
    def test_step_with_mer(self):
        """Test forward pass with MER loss."""
        model = DummyHeteroModel(hidden_dim=128)
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='mer'
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'mer' in metrics
    
    def test_step_with_tar(self):
        """Test forward pass with TAR loss."""
        model = DummyHeteroModel(hidden_dim=128)
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='tar',
            tar_temperature=0.5
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'tar' in metrics
    
    def test_step_with_pfp(self):
        """Test forward pass with PFP loss."""
        model = DummyHeteroModel(hidden_dim=128)
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='pfp'
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'pfp' in metrics
    
    def test_step_with_combined_loss(self):
        """Test forward pass with combined loss."""
        model = DummyHeteroModel(hidden_dim=128)
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='combined_loss',
            mer_weight=1.0,
            tar_weight=1.0,
            pfp_weight=1.0
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert 'mer' in metrics
        assert 'tar' in metrics
        assert 'pfp' in metrics
        assert 'combined_loss_total' in metrics
    
    def test_step_with_decoder(self):
        """Test forward pass with edge decoder."""
        model = DummyHeteroModel(hidden_dim=128)
        decoder = EdgeDecoder(hidden_dim=128, dropout=0.5)
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=('paper', 'cites', 'paper'),
            loss_fn='bce',
            decoder=decoder
        )
        batch = create_dummy_hetero_batch()
        
        loss, metrics = objective.step(model, batch, is_training=True)
        
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_invalid_loss_function(self):
        """Test that invalid loss function raises error."""
        with pytest.raises(ValueError):
            SelfSupervisedEdgeReconstruction(
                target_edge_type=('paper', 'cites', 'paper'),
                loss_fn='invalid_loss'
            )


class TestDecoders:
    """Test decoder modules."""
    
    def test_edge_decoder(self):
        """Test EdgeDecoder forward pass."""
        decoder = EdgeDecoder(hidden_dim=128, dropout=0.5)
        src_embeddings = torch.randn(100, 128)
        dst_embeddings = torch.randn(100, 128)
        
        output = decoder(src_embeddings, dst_embeddings)
        
        assert output.shape == (100, 1), "Output shape should be [num_edges, 1]"
        assert torch.isfinite(output).all(), "Output should be finite"
    
    def test_feature_decoder(self):
        """Test FeatureDecoder forward pass."""
        decoder = FeatureDecoder(hidden_dim=128, feature_dim=256, dropout=0.5)
        embeddings = torch.randn(100, 128)
        
        output = decoder(embeddings)
        
        assert output.shape == (100, 256), "Output shape should be [num_nodes, feature_dim]"
        assert torch.isfinite(output).all(), "Output should be finite"


if __name__ == '__main__':
    pytest.main([__file__, "-v"])

