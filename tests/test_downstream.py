"""
Test script for downstream evaluation pipeline.
Tests both node property prediction and link prediction tasks.
"""
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphssl.utils.downstream import (
    MLPClassifier,
    evaluate_node_property_prediction,
    evaluate_link_prediction,
    create_link_index_data
)
from graphssl.utils.data_utils import create_link_loaders


def _dummy_heterodata(num_nodes: int = 6, emb_dim: int = 8) -> HeteroData:
    """Create a tiny heterogeneous graph for loader regression tests."""
    data = HeteroData()
    data["paper"].x = torch.randn(num_nodes, emb_dim)
    data["paper"].y = torch.zeros(num_nodes, dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    mid = num_nodes // 2
    train_mask[:mid] = True
    val_mask[mid] = True
    if mid + 1 < num_nodes:
        test_mask[mid + 1 :] = True
    data["paper"].train_mask = train_mask
    data["paper"].val_mask = val_mask
    data["paper"].test_mask = test_mask

    # Simple bidirectional citations to keep loaders happy.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    data["paper", "cites", "paper"].edge_index = edge_index

    return data


def test_mlp_classifier():
    """Test MLP classifier creation and forward pass."""
    print("\n" + "="*80)
    print("Test 1: MLP Classifier")
    print("="*80)
    
    input_dim = 128
    hidden_dim = 64
    output_dim = 10
    batch_size = 32
    
    # Create classifier
    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        dropout=0.5
    )
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    out = classifier(x)
    
    assert out.shape == (batch_size, output_dim), f"Expected shape {(batch_size, output_dim)}, got {out.shape}"
    print(f"✓ MLP Classifier: Input {x.shape} -> Output {out.shape}")
    print("✓ Test passed!")


def test_link_prediction_data_creation():
    """Test link prediction index/label generation."""
    print("\n" + "="*80)
    print("Test 2: Link Prediction Edge Sampling")
    print("="*80)
    
    num_nodes = 100
    num_edges = 50
    
    # Create dummy edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create link prediction edge indices + labels
    sampled_edges, sampled_labels = create_link_index_data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=1
    )
    
    expected_samples = num_edges * 2  # pos + neg
    
    assert sampled_edges.shape == (2, expected_samples), \
        f"Expected shape {(2, expected_samples)}, got {sampled_edges.shape}"
    assert sampled_labels.shape == (expected_samples,), \
        f"Expected shape {(expected_samples,)}, got {sampled_labels.shape}"
    assert sampled_labels.sum().item() == num_edges, \
        f"Expected {num_edges} positive samples, got {sampled_labels.sum().item()}"
    
    print(f"✓ Created {expected_samples} edge samples ({num_edges} pos, {num_edges} neg)")
    print(f"✓ Edge index shape: {sampled_edges.shape}")
    print(f"✓ Edge labels shape: {sampled_labels.shape}")
    print("✓ Test passed!")


def test_node_property_prediction():
    """Test node property prediction evaluation (minimal)."""
    print("\n" + "="*80)
    print("Test 3: Node Property Prediction (Minimal)")
    print("="*80)
    
    # Create dummy data
    num_train = 100
    num_val = 30
    num_test = 30
    embedding_dim = 64
    num_classes = 5
    
    train_embeddings = torch.randn(num_train, embedding_dim)
    train_labels = torch.randint(0, num_classes, (num_train,))
    val_embeddings = torch.randn(num_val, embedding_dim)
    val_labels = torch.randint(0, num_classes, (num_val,))
    test_embeddings = torch.randn(num_test, embedding_dim)
    test_labels = torch.randint(0, num_classes, (num_test,))
    
    device = torch.device('cpu')
    
    # Run evaluation with minimal settings
    results = evaluate_node_property_prediction(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        num_classes=num_classes,
        device=device,
        n_runs=2,  # Minimal runs for testing
        hidden_dim=32,
        num_layers=1,
        dropout=0.5,
        batch_size=32,
        lr=0.01,
        num_epochs=5,  # Minimal epochs
        early_stopping_patience=3,
        verbose=False
    )
    
    # Check results format
    assert 'test_acc_mean' in results
    assert 'test_acc_std' in results
    assert 'test_loss_mean' in results
    assert 'test_loss_std' in results
    assert len(results['test_accuracies']) == 2
    assert len(results['test_losses']) == 2
    
    print(f"✓ Test Accuracy: {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    print(f"✓ Test Loss: {results['test_loss_mean']:.4f} ± {results['test_loss_std']:.4f}")
    print("✓ Test passed!")


def test_link_prediction():
    """Test link prediction evaluation (minimal)."""
    print("\n" + "="*80)
    print("Test 4: Link Prediction (Minimal)")
    print("="*80)
    
    # Create dummy data
    num_nodes = 100
    embedding_dim = 64
    num_train_edges = 50
    num_val_edges = 15
    num_test_edges = 15
    
    embeddings = torch.randn(num_nodes, embedding_dim)
    
    train_edge_index = torch.randint(0, num_nodes, (2, num_train_edges))
    val_edge_index = torch.randint(0, num_nodes, (2, num_val_edges))
    test_edge_index = torch.randint(0, num_nodes, (2, num_test_edges))
    
    device = torch.device('cpu')
    
    # Run evaluation with minimal settings
    results = evaluate_link_prediction(
        embeddings=embeddings,
        train_edge_index=train_edge_index,
        val_edge_index=val_edge_index,
        test_edge_index=test_edge_index,
        device=device,
        n_runs=2,  # Minimal runs
        num_neg_samples=1,
        hidden_dim=32,
        num_layers=1,
        dropout=0.5,
        batch_size=32,
        lr=0.01,
        num_epochs=5,  # Minimal epochs
        early_stopping_patience=3,
        verbose=False
    )
    
    # Check results format
    assert 'test_acc_mean' in results
    assert 'test_acc_std' in results
    assert 'test_loss_mean' in results
    assert 'test_loss_std' in results
    assert len(results['test_accuracies']) == 2
    assert len(results['test_losses']) == 2
    
    print(f"✓ Test Accuracy: {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    print(f"✓ Test Loss: {results['test_loss_mean']:.4f} ± {results['test_loss_std']:.4f}")
    print("✓ Test passed!")


def test_create_link_loaders_returns_global_loader():
    """Regression test: ensure link loaders expose the global loader handle."""
    print("\n" + "="*80)
    print("Test 5: Link Loader Regression")
    print("="*80)

    data = _dummy_heterodata()
    loaders = create_link_loaders(
        data=data,
        target_edge_type=("paper", "cites", "paper"),
        num_neighbors=[2],
        batch_size=2,
        neg_sampling_ratio=1.0,
        num_workers=0,
        split_edges=False,
        seed=0,
    )

    train_loader, val_loader, test_loader, global_loader, edge_splits = loaders
    assert global_loader is not None, "Expected global_loader in return tuple"
    assert len(edge_splits) == 3, "Edge splits tuple should contain three tensors"

    sample_batch = next(iter(global_loader))
    assert sample_batch["paper"].num_nodes > 0, "Global loader should yield data batches"

    print("✓ Link loaders expose global loader handle")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DOWNSTREAM EVALUATION TESTS")
    print("="*80)
    
    # Disable wandb for testing
    import wandb
    os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(mode='disabled', project='test')
    
    try:
        test_mlp_classifier()
        test_link_prediction_data_creation()
        test_node_property_prediction()
        test_link_prediction()
        test_create_link_loaders_returns_global_loader()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ✗")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up wandb
        wandb.finish()


if __name__ == "__main__":
    main()
