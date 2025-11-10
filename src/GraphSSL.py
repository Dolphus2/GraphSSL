"""
GraphSSL: Supervised Learning Pipeline for OGB_MAG Dataset
Venue prediction using heterogeneous GraphSAGE
"""
import os
import torch
import argparse
from pathlib import Path

# Import utilities
from utils.data_utils import load_ogb_mag, create_neighbor_loaders, get_dataset_info
from utils.models import create_model
from utils.training_utils import train_model, test_model, extract_embeddings


def main(args):
    """
    Main supervised learning pipeline for venue prediction on OGB_MAG.
    """
    print("="*80)
    print("GraphSSL - Supervised Learning Pipeline")
    print("Task: Venue Prediction on OGB_MAG Dataset")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # ==================== Step 1: Load Dataset ====================
    print("\n" + "="*80)
    print("Step 1: Loading Dataset")
    print("="*80)
    
    data_path = Path(args.data_root)
    data_path.mkdir(parents=True, exist_ok=True)
    
    data = load_ogb_mag(
        root_path=str(data_path),
        preprocess=args.preprocess
    )
    
    # Get dataset information
    dataset_info = get_dataset_info(data, target_node_type=args.target_node)
    print(f"\nDataset Information:")
    print(f"  Node types: {dataset_info['node_types']}")
    print(f"  Edge types: {dataset_info['edge_types']}")
    print(f"  Number of classes: {dataset_info['num_classes']}")
    print(f"  Node features: {dataset_info['num_features']}")
    
    # ==================== Step 2: Create Data Loaders ====================
    print("\n" + "="*80)
    print("Step 2: Creating Data Loaders")
    print("="*80)
    
    train_loader, val_loader, test_loader = create_neighbor_loaders(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_node_type=args.target_node
    )
    
    # ==================== Step 3: Create Model ====================
    print("\n" + "="*80)
    print("Step 3: Creating Model")
    print("="*80)
    
    model = create_model(
        data,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        target_node_type=args.target_node
    )
    model = model.to(device)
    
    # ==================== Step 4: Setup Optimizer ====================
    print("\n" + "="*80)
    print("Step 4: Setting up Optimizer")
    print("="*80)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: Adam")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    
    # ==================== Step 5: Train Model ====================
    print("\n" + "="*80)
    print("Step 5: Training Model")
    print("="*80)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        target_node_type=args.target_node,
        early_stopping_patience=args.patience,
        verbose=True
    )
    
    # ==================== Step 6: Test Model ====================
    print("\n" + "="*80)
    print("Step 6: Testing Model")
    print("="*80)
    
    test_loss, test_acc = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        target_node_type=args.target_node
    )
    
    # ==================== Step 7: Save Results ====================
    print("\n" + "="*80)
    print("Step 7: Saving Results")
    print("="*80)
    
    results_path = Path(args.results_root)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = results_path / "model_supervised.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
        'args': vars(args),
        'history': history
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training history
    history_path = results_path / "training_history.pt"
    torch.save(history, history_path)
    print(f"Training history saved to: {history_path}")
    
    # ==================== Step 8: Extract Embeddings (Optional) ====================
    if args.extract_embeddings:
        print("\n" + "="*80)
        print("Step 8: Extracting Embeddings")
        print("="*80)
        
        print("Extracting train embeddings...")
        train_embeddings, train_labels = extract_embeddings(
            model, train_loader, device, args.target_node
        )
        
        print("Extracting val embeddings...")
        val_embeddings, val_labels = extract_embeddings(
            model, val_loader, device, args.target_node
        )
        
        print("Extracting test embeddings...")
        test_embeddings, test_labels = extract_embeddings(
            model, test_loader, device, args.target_node
        )
        
        # Save embeddings
        embeddings_path = results_path / "embeddings.pt"
        torch.save({
            'train_embeddings': train_embeddings,
            'train_labels': train_labels,
            'val_embeddings': val_embeddings,
            'val_labels': val_labels,
            'test_embeddings': test_embeddings,
            'test_labels': test_labels
        }, embeddings_path)
        print(f"Embeddings saved to: {embeddings_path}")
        print(f"  Train embeddings shape: {train_embeddings.shape}")
        print(f"  Val embeddings shape: {val_embeddings.shape}")
        print(f"  Test embeddings shape: {test_embeddings.shape}")
    
    # ==================== Final Summary ====================
    print("\n" + "="*80)
    print("Pipeline Completed Successfully!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"\nAll outputs saved to: {results_path}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised Learning Pipeline for OGB_MAG Venue Prediction"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="../data",
        help="Root directory for dataset storage"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="../results",
        help="Root directory for results"
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="metapath2vec",
        choices=["metapath2vec", "transe"],
        help="Preprocessing method for node embeddings"
    )
    parser.add_argument(
        "--target_node",
        type=str,
        default="paper",
        help="Target node type for prediction"
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=128,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of GraphSAGE layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate"
    )
    
    # Data loader arguments
    parser.add_argument(
        "--num_neighbors",
        type=int,
        nargs="+",
        default=[15, 10],
        help="Number of neighbors to sample at each layer"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Additional options
    parser.add_argument(
        "--extract_embeddings",
        action="store_true",
        help="Extract and save node embeddings after training"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    main(args)
