"""
Test script for downstream evaluation pipeline.
Tests both node property prediction and link prediction tasks.
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphssl.utils.downstream import (
    MLPClassifier,
    evaluate_node_property_prediction,
    evaluate_link_prediction,
    create_link_prediction_data
)


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
    """Test link prediction data creation."""
    print("\n" + "="*80)
    print("Test 2: Link Prediction Data Creation")
    print("="*80)
    
    num_nodes = 100
    embedding_dim = 64
    num_edges = 50
    
    # Create dummy embeddings and edges
    embeddings = torch.randn(num_nodes, embedding_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create link prediction data
    edge_features, edge_labels = create_link_prediction_data(
        embeddings, edge_index, num_neg_samples=1
    )
    
    expected_samples = num_edges * 2  # pos + neg
    expected_feature_dim = 2 * embedding_dim
    
    assert edge_features.shape == (expected_samples, expected_feature_dim), \
        f"Expected shape {(expected_samples, expected_feature_dim)}, got {edge_features.shape}"
    assert edge_labels.shape == (expected_samples,), \
        f"Expected shape {(expected_samples,)}, got {edge_labels.shape}"
    assert edge_labels.sum() == num_edges, \
        f"Expected {num_edges} positive samples, got {edge_labels.sum()}"
    
    print(f"✓ Created {expected_samples} edge samples ({num_edges} pos, {num_edges} neg)")
    print(f"✓ Edge features shape: {edge_features.shape}")
    print(f"✓ Edge labels shape: {edge_labels.shape}")
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
    
    train_embeddings = torch.randn(num_nodes, embedding_dim)
    val_embeddings = torch.randn(num_nodes, embedding_dim)
    test_embeddings = torch.randn(num_nodes, embedding_dim)
    
    train_edge_index = torch.randint(0, num_nodes, (2, num_train_edges))
    val_edge_index = torch.randint(0, num_nodes, (2, num_val_edges))
    test_edge_index = torch.randint(0, num_nodes, (2, num_test_edges))
    
    device = torch.device('cpu')
    
    # Run evaluation with minimal settings
    results = evaluate_link_prediction(
        train_embeddings=train_embeddings,
        train_edge_index=train_edge_index,
        val_embeddings=val_embeddings,
        val_edge_index=val_edge_index,
        test_embeddings=test_embeddings,
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


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DOWNSTREAM EVALUATION TESTS")
    print("="*80)
    
    # Disable wandb for testing
    import os
    import wandb
    os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(mode='disabled', project='test')
    
    try:
        test_mlp_classifier()
        test_link_prediction_data_creation()
        test_node_property_prediction()
        test_link_prediction()
        
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
