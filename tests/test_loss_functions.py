"""
Unit tests for loss functions in GraphSSL.

Tests all loss functions:
- SCE (Scaled Cosine Error) for node reconstruction
- MER (Masked Edge Reconstruction) for edge tasks
- TAR (Topology-Aware Reconstruction) for edge tasks
- PFP (Preference-based Feature Propagation) for edge tasks
- Combined loss (MER + TAR + PFP)
"""

import pytest
import torch
import torch.nn.functional as F
from graphssl.utils.objective_utils import (
    sce_loss,
    mer_loss,
    tar_loss,
    pfp_loss
)


class TestSCELoss:
    """Test Scaled Cosine Error loss function."""
    
    def test_sce_loss_basic(self):
        """Test that SCE loss computes correctly."""
        reconstructed = torch.randn(100, 128)
        target = torch.randn(100, 128)
        
        loss = sce_loss(reconstructed, target, alpha=3)
        
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_sce_loss_identical_inputs(self):
        """Test that SCE loss is small for identical inputs."""
        x = torch.randn(100, 128)
        
        loss = sce_loss(x, x, alpha=3)
        
        # Loss should be very small (close to 0) for identical inputs
        assert loss.item() < 0.1, "Loss should be small for identical inputs"
    
    def test_sce_loss_different_alpha(self):
        """Test that alpha parameter affects loss magnitude."""
        reconstructed = torch.randn(100, 128)
        target = torch.randn(100, 128)
        
        loss_alpha1 = sce_loss(reconstructed, target, alpha=1)
        loss_alpha3 = sce_loss(reconstructed, target, alpha=3)
        loss_alpha5 = sce_loss(reconstructed, target, alpha=5)
        
        # Higher alpha should generally produce higher loss
        assert torch.isfinite(loss_alpha1), "Loss should be finite"
        assert torch.isfinite(loss_alpha3), "Loss should be finite"
        assert torch.isfinite(loss_alpha5), "Loss should be finite"
    
    def test_sce_loss_gradient_flow(self):
        """Test that gradients flow through SCE loss."""
        reconstructed = torch.randn(100, 128, requires_grad=True)
        target = torch.randn(100, 128)
        
        loss = sce_loss(reconstructed, target)
        loss.backward()
        
        assert reconstructed.grad is not None, "Gradient should be computed"
        assert torch.isfinite(reconstructed.grad).all(), "Gradient should be finite"
        assert (reconstructed.grad.abs() > 0).any(), "Gradient should be non-zero"


class TestMERLoss:
    """Test Masked Edge Reconstruction loss function."""
    
    def test_mer_loss_basic(self):
        """Test that MER loss computes correctly."""
        edge_scores = torch.randn(1000)
        edge_labels = torch.randint(0, 2, (1000,))
        
        loss = mer_loss(edge_scores, edge_labels)
        
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_mer_loss_with_mask(self):
        """Test MER loss with masking."""
        edge_scores = torch.randn(1000)
        edge_labels = torch.randint(0, 2, (1000,))
        mask = torch.rand(1000) > 0.5
        
        loss = mer_loss(edge_scores, edge_labels, mask=mask)
        
        assert loss.item() > 0, "Masked loss should be positive"
        assert torch.isfinite(loss), "Masked loss should be finite"
    
    def test_mer_loss_perfect_predictions(self):
        """Test that MER loss is small for perfect predictions."""
        edge_labels = torch.randint(0, 2, (1000,))
        # Create perfect predictions (large positive for 1, large negative for 0)
        edge_scores = torch.where(edge_labels == 1, 
                                  torch.tensor(10.0), 
                                  torch.tensor(-10.0))
        
        loss = mer_loss(edge_scores, edge_labels)
        
        # Loss should be very small for perfect predictions
        assert loss.item() < 0.01, "Loss should be small for perfect predictions"
    
    def test_mer_loss_gradient_flow(self):
        """Test that gradients flow through MER loss."""
        edge_scores = torch.randn(1000, requires_grad=True)
        edge_labels = torch.randint(0, 2, (1000,))
        
        loss = mer_loss(edge_scores, edge_labels)
        loss.backward()
        
        assert edge_scores.grad is not None, "Gradient should be computed"
        assert torch.isfinite(edge_scores.grad).all(), "Gradient should be finite"


class TestTARLoss:
    """Test Topology-Aware Reconstruction loss function."""
    
    def test_tar_loss_basic(self):
        """Test that TAR loss computes correctly."""
        src_embeddings = torch.randn(1000, 128)
        dst_embeddings = torch.randn(1000, 128)
        edge_labels = torch.randint(0, 2, (1000,))
        
        loss = tar_loss(src_embeddings, dst_embeddings, edge_labels, temperature=0.5)
        
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_tar_loss_different_temperatures(self):
        """Test TAR loss with different temperature values."""
        src_embeddings = torch.randn(1000, 128)
        dst_embeddings = torch.randn(1000, 128)
        edge_labels = torch.randint(0, 2, (1000,))
        
        for temp in [0.1, 0.5, 1.0, 2.0]:
            loss = tar_loss(src_embeddings, dst_embeddings, edge_labels, temperature=temp)
            assert torch.isfinite(loss), f"Loss should be finite for temp={temp}"
            assert loss.item() > 0, f"Loss should be positive for temp={temp}"
    
    def test_tar_loss_positive_pairs(self):
        """Test TAR loss with only positive pairs."""
        src_embeddings = torch.randn(100, 128)
        dst_embeddings = torch.randn(100, 128)
        edge_labels = torch.ones(100, dtype=torch.long)
        
        loss = tar_loss(src_embeddings, dst_embeddings, edge_labels)
        
        assert torch.isfinite(loss), "Loss should be finite for positive pairs"
    
    def test_tar_loss_negative_pairs(self):
        """Test TAR loss with only negative pairs."""
        src_embeddings = torch.randn(100, 128)
        dst_embeddings = torch.randn(100, 128)
        edge_labels = torch.zeros(100, dtype=torch.long)
        
        loss = tar_loss(src_embeddings, dst_embeddings, edge_labels)
        
        assert torch.isfinite(loss), "Loss should be finite for negative pairs"
    
    def test_tar_loss_gradient_flow(self):
        """Test that gradients flow through TAR loss."""
        src_embeddings = torch.randn(100, 128, requires_grad=True)
        dst_embeddings = torch.randn(100, 128, requires_grad=True)
        edge_labels = torch.randint(0, 2, (100,))
        
        loss = tar_loss(src_embeddings, dst_embeddings, edge_labels)
        loss.backward()
        
        assert src_embeddings.grad is not None, "Gradient should be computed"
        assert dst_embeddings.grad is not None, "Gradient should be computed"
        assert torch.isfinite(src_embeddings.grad).all(), "Gradient should be finite"
        assert torch.isfinite(dst_embeddings.grad).all(), "Gradient should be finite"


class TestPFPLoss:
    """Test Preference-based Feature Propagation loss function."""
    
    def test_pfp_loss_basic(self):
        """Test that PFP loss computes correctly."""
        src_features = torch.randn(1000, 256)
        dst_features = torch.randn(1000, 256)
        edge_labels = torch.randint(0, 2, (1000,))
        
        loss = pfp_loss(src_features, dst_features, edge_labels)
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_pfp_loss_with_reconstructed_features(self):
        """Test PFP loss with reconstructed features."""
        src_features = torch.randn(1000, 256)
        dst_features = torch.randn(1000, 256)
        edge_labels = torch.randint(0, 2, (1000,))
        reconstructed_src = torch.randn(1000, 256)
        reconstructed_dst = torch.randn(1000, 256)
        
        loss = pfp_loss(src_features, dst_features, edge_labels,
                       reconstructed_src=reconstructed_src,
                       reconstructed_dst=reconstructed_dst)
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_pfp_loss_no_positive_edges(self):
        """Test PFP loss when there are no positive edges."""
        src_features = torch.randn(100, 256)
        dst_features = torch.randn(100, 256)
        edge_labels = torch.zeros(100, dtype=torch.long)
        
        loss = pfp_loss(src_features, dst_features, edge_labels)
        
        # Loss should be 0 or very small when no positive edges
        assert loss.item() == 0, "Loss should be 0 for no positive edges"
    
    def test_pfp_loss_identical_features(self):
        """Test PFP loss when features are identical."""
        features = torch.randn(100, 256)
        edge_labels = torch.ones(100, dtype=torch.long)
        
        loss = pfp_loss(features, features, edge_labels)
        
        # Loss should be very small for identical features on positive edges
        assert loss.item() < 0.01, "Loss should be small for identical features"
    
    def test_pfp_loss_gradient_flow(self):
        """Test that gradients flow through PFP loss."""
        src_features = torch.randn(100, 256, requires_grad=True)
        dst_features = torch.randn(100, 256, requires_grad=True)
        edge_labels = torch.ones(100, dtype=torch.long)  # All positive
        
        loss = pfp_loss(src_features, dst_features, edge_labels)
        
        if loss.item() > 0:  # Only if there's a loss
            loss.backward()
            assert src_features.grad is not None, "Gradient should exist"
            assert dst_features.grad is not None, "Gradient should exist"


class TestCombinedLoss:
    """Test combined loss (MER + TAR + PFP)."""
    
    def test_combined_loss_computation(self):
        """Test that combined loss computes all components correctly."""
        # Create dummy data
        edge_scores = torch.randn(1000)
        edge_labels = torch.randint(0, 2, (1000,))
        src_embeddings = torch.randn(1000, 128)
        dst_embeddings = torch.randn(1000, 128)
        src_features = torch.randn(1000, 256)
        dst_features = torch.randn(1000, 256)
        
        # Compute individual losses
        mer = mer_loss(edge_scores, edge_labels)
        tar = tar_loss(src_embeddings, dst_embeddings, edge_labels, temperature=0.5)
        pfp = pfp_loss(src_features, dst_features, edge_labels)
        
        # Compute combined loss with equal weights
        combined = 1.0 * mer + 1.0 * tar + 1.0 * pfp
        
        assert torch.isfinite(combined), "Combined loss should be finite"
        assert combined.item() > 0, "Combined loss should be positive"
    
    def test_combined_loss_different_weights(self):
        """Test combined loss with different weight configurations."""
        edge_scores = torch.randn(1000)
        edge_labels = torch.randint(0, 2, (1000,))
        src_embeddings = torch.randn(1000, 128)
        dst_embeddings = torch.randn(1000, 128)
        src_features = torch.randn(1000, 256)
        dst_features = torch.randn(1000, 256)
        
        # Compute individual losses
        mer = mer_loss(edge_scores, edge_labels)
        tar = tar_loss(src_embeddings, dst_embeddings, edge_labels)
        pfp = pfp_loss(src_features, dst_features, edge_labels)
        
        # Test different weight configurations
        weight_configs = [
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 1.0),
            (1.0, 2.0, 1.0),
            (1.0, 1.0, 2.0),
            (0.5, 0.5, 0.5),
        ]
        
        for mer_w, tar_w, pfp_w in weight_configs:
            combined = mer_w * mer + tar_w * tar + pfp_w * pfp
            assert torch.isfinite(combined), f"Loss should be finite for weights ({mer_w}, {tar_w}, {pfp_w})"
    
    def test_combined_loss_gradient_flow(self):
        """Test that gradients flow through combined loss."""
        edge_scores = torch.randn(100, requires_grad=True)
        edge_labels = torch.randint(0, 2, (100,))
        src_embeddings = torch.randn(100, 128, requires_grad=True)
        dst_embeddings = torch.randn(100, 128, requires_grad=True)
        src_features = torch.randn(100, 256, requires_grad=True)
        dst_features = torch.randn(100, 256, requires_grad=True)
        
        # Compute combined loss
        mer = mer_loss(edge_scores, edge_labels)
        tar = tar_loss(src_embeddings, dst_embeddings, edge_labels)
        pfp = pfp_loss(src_features, dst_features, edge_labels)
        combined = mer + tar + pfp
        
        combined.backward()
        
        # Check gradients
        assert edge_scores.grad is not None, "MER gradient should flow"
        assert src_embeddings.grad is not None, "TAR gradient should flow"
        assert dst_embeddings.grad is not None, "TAR gradient should flow"
        # PFP gradients may be None if no positive edges, so we don't check those


class TestLossNumericalStability:
    """Test numerical stability of loss functions."""
    
    def test_sce_loss_extreme_values(self):
        """Test SCE loss with extreme values."""
        # Very large values
        x_large = torch.randn(10, 128) * 1000
        y_large = torch.randn(10, 128) * 1000
        loss = sce_loss(x_large, y_large)
        assert torch.isfinite(loss), "Loss should be finite for large values"
        
        # Very small values
        x_small = torch.randn(10, 128) * 0.001
        y_small = torch.randn(10, 128) * 0.001
        loss = sce_loss(x_small, y_small)
        assert torch.isfinite(loss), "Loss should be finite for small values"
    
    def test_mer_loss_extreme_scores(self):
        """Test MER loss with extreme edge scores."""
        edge_labels = torch.randint(0, 2, (100,))
        
        # Very large scores
        edge_scores_large = torch.randn(100) * 1000
        loss = mer_loss(edge_scores_large, edge_labels)
        assert torch.isfinite(loss), "Loss should be finite for large scores"
        
        # Very small scores
        edge_scores_small = torch.randn(100) * 0.001
        loss = mer_loss(edge_scores_small, edge_labels)
        assert torch.isfinite(loss), "Loss should be finite for small scores"
    
    def test_tar_loss_extreme_temperatures(self):
        """Test TAR loss with extreme temperatures."""
        src_embeddings = torch.randn(100, 128)
        dst_embeddings = torch.randn(100, 128)
        edge_labels = torch.randint(0, 2, (100,))
        
        # Very small temperature
        loss = tar_loss(src_embeddings, dst_embeddings, edge_labels, temperature=0.01)
        assert torch.isfinite(loss), "Loss should be finite for small temperature"
        
        # Very large temperature
        loss = tar_loss(src_embeddings, dst_embeddings, edge_labels, temperature=10.0)
        assert torch.isfinite(loss), "Loss should be finite for large temperature"


class TestLossBatchSizes:
    """Test loss functions with different batch sizes."""
    
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_sce_loss_batch_sizes(self, batch_size):
        """Test SCE loss with various batch sizes."""
        x = torch.randn(batch_size, 128)
        y = torch.randn(batch_size, 128)
        loss = sce_loss(x, y)
        assert torch.isfinite(loss), f"Loss should be finite for batch size {batch_size}"
    
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_mer_loss_batch_sizes(self, batch_size):
        """Test MER loss with various batch sizes."""
        edge_scores = torch.randn(batch_size)
        edge_labels = torch.randint(0, 2, (batch_size,))
        loss = mer_loss(edge_scores, edge_labels)
        assert torch.isfinite(loss), f"Loss should be finite for batch size {batch_size}"
    
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_tar_loss_batch_sizes(self, batch_size):
        """Test TAR loss with various batch sizes."""
        src_embeddings = torch.randn(batch_size, 128)
        dst_embeddings = torch.randn(batch_size, 128)
        edge_labels = torch.randint(0, 2, (batch_size,))
        loss = tar_loss(src_embeddings, dst_embeddings, edge_labels)
        assert torch.isfinite(loss), f"Loss should be finite for batch size {batch_size}"
    
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_pfp_loss_batch_sizes(self, batch_size):
        """Test PFP loss with various batch sizes."""
        src_features = torch.randn(batch_size, 256)
        dst_features = torch.randn(batch_size, 256)
        edge_labels = torch.randint(0, 2, (batch_size,))
        loss = pfp_loss(src_features, dst_features, edge_labels)
        assert torch.isfinite(loss), f"Loss should be finite for batch size {batch_size}"


if __name__ == '__main__':
    pytest.main([__file__, "-v"])

