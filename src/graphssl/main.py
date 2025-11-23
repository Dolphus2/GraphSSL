"""
GraphSSL: Supervised Learning Pipeline for OGB_MAG Dataset
Venue prediction using heterogeneous GraphSAGE

Run with: python -m graphssl.main
"""
import logging
import torch
import argparse
import wandb
import time
from pathlib import Path

from graphssl.utils.data_utils import load_ogb_mag, create_neighbor_loaders, create_link_loaders, get_dataset_info
from graphssl.utils.models import create_model
from graphssl.utils.training_utils import train_model, test_model, extract_embeddings
from graphssl.utils.objective_utils import (
    SupervisedNodeClassification,
    SupervisedLinkPrediction,
    SelfSupervisedNodeReconstruction,
    SelfSupervisedEdgeReconstruction,
    EdgeDecoder,
    FeatureDecoder
)

logger = logging.getLogger(__name__)


def run_pipeline(args):
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
    logger.debug(f"Dataset Information:")
    logger.debug(f"  Node types: {dataset_info['node_types']}")
    logger.debug(f"  Edge types: {dataset_info['edge_types']}")
    logger.debug(f"  Number of classes: {dataset_info['num_classes']}")
    logger.debug(f"  Node features: {dataset_info['num_features']}")
    
    # ==================== Step 2: Create Data Loaders ====================
    print("\n" + "="*80)
    print("Step 2: Creating Data Loaders")
    print("="*80)
    
    if args.objective_type == "supervised_link_prediction" or args.objective_type == "self_supervised_edge":
        # Create link prediction loaders
        train_loader, val_loader, test_loader = create_link_loaders(
            data,
            target_edge_type=tuple(args.target_edge_type.split(",")),
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            neg_sampling_ratio=args.neg_sampling_ratio,
            num_workers=args.num_workers,
            split_edges=True
        )
        inductive_train_loader = None  # Not used for link prediction
        transductive_train_loader = train_loader
    else:
        # Create neighbor loaders for node-level tasks
        inductive_train_loader, transductive_train_loader, val_loader, test_loader = create_neighbor_loaders(
            data,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_node_type=args.target_node
        )
    
    # ==================== Step 3: Create Training Objective ====================
    print("\n" + "="*80)
    print("Step 3: Creating Training Objective")
    print("="*80)
    
    if args.objective_type == "supervised_node_classification":
        objective = SupervisedNodeClassification(target_node_type=args.target_node)
        logger.info(f"Objective: Supervised Node Classification on '{args.target_node}'")
    
    elif args.objective_type == "supervised_link_prediction":
        target_edge_type = tuple(args.target_edge_type.split(","))
        # Create optional edge decoder if specified
        decoder = None
        if args.use_edge_decoder:
            decoder = EdgeDecoder(hidden_dim=args.hidden_channels, dropout=args.dropout)
            logger.info("Using MLP-based edge decoder")
        objective = SupervisedLinkPrediction(
            target_edge_type=target_edge_type,
            decoder=decoder
        )
        logger.info(f"Objective: Supervised Link Prediction on '{target_edge_type}'")
    
    elif args.objective_type == "self_supervised_node":
        # Create optional feature decoder
        decoder = None
        if args.use_feature_decoder:
            feature_dim = data[args.target_node].x.shape[1]
            decoder = FeatureDecoder(
                hidden_dim=args.hidden_channels,
                feature_dim=feature_dim,
                dropout=args.dropout
            )
            logger.info("Using MLP-based feature decoder")
        objective = SelfSupervisedNodeReconstruction(
            target_node_type=args.target_node,
            mask_ratio=args.mask_ratio,
            decoder=decoder,
            loss_fn=args.loss_fn
        )
        logger.info(f"Objective: Self-Supervised Node Reconstruction on '{args.target_node}'")
        logger.info(f"  Mask ratio: {args.mask_ratio}")
    
    elif args.objective_type == "self_supervised_edge":
        target_edge_type = tuple(args.target_edge_type.split(","))
        # Create optional edge decoder
        decoder = None
        if args.use_edge_decoder:
            decoder = EdgeDecoder(hidden_dim=args.hidden_channels, dropout=args.dropout)
            logger.info("Using MLP-based edge decoder")
        objective = SelfSupervisedEdgeReconstruction(
            target_edge_type=target_edge_type,
            negative_sampling_ratio=args.neg_sampling_ratio,
            decoder=decoder
        )
        logger.info(f"Objective: Self-Supervised Edge Reconstruction on '{target_edge_type}'")
        logger.info(f"  Negative sampling ratio: {args.neg_sampling_ratio}")
    
    else:
        raise ValueError(f"Unknown objective type: {args.objective_type}")
    
    # ==================== Step 4: Create Model ====================
    print("\n" + "="*80)
    print("Step 4: Creating Model")
    print("="*80)
    
    model = create_model(
        data,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batchnorm=args.use_batchnorm,
        target_node_type=args.target_node,
        aggr=args.aggr,
        aggr_rel=args.aggr_rel
    )
    model = model.to(device)
    
    # ==================== Step 5: Setup Optimizer ====================
    print("\n" + "="*80)
    print("Step 5: Setting up Optimizer")
    print("="*80)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    logger.info(f"Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # ==================== Step 6: Train Model ====================
    print("\n" + "="*80)
    print("Step 6: Training Model")
    print("="*80)
    
    # Create checkpoint directory
    results_path = Path(args.results_root)
    results_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which train loader to use
    train_loader_to_use = inductive_train_loader if inductive_train_loader is not None else transductive_train_loader
    
    history = train_model(
        model=model,
        train_loader=train_loader_to_use,
        val_loader=val_loader,
        optimizer=optimizer,
        objective=objective,
        device=device,
        num_epochs=args.epochs,
        log_interval=args.log_interval,
        early_stopping_patience=args.patience,
        checkpoint_dir=str(checkpoint_dir),
        verbose=True,
        metric_for_best=args.metric_for_best
    )
    
    # ==================== Step 7: Test Model ====================
    print("\n" + "="*80)
    print("Step 7: Testing Model")
    print("="*80)
    
    test_metrics = test_model(
        model=model,
        test_loader=test_loader,
        objective=objective,
        device=device
    )
    
    # ==================== Step 8: Save Results ====================
    print("\n" + "="*80)
    print("Step 8: Saving Results")
    print("="*80)
    
    # Results path was already created in Step 6
    
    # Save final model with complete training information
    model_path = results_path / f"model_{args.objective_type}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': test_metrics,
        'args': vars(args),
        'history': history
    }, model_path)
    logger.info(f"Final model saved to: {model_path}")
    logger.info(f"Best model checkpoint saved to: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"Last checkpoint saved to: {checkpoint_dir / 'last_checkpoint.pt'}")
    
    # Save training history
    history_path = results_path / "training_history.pt"
    torch.save(history, history_path)
    logger.info(f"Training history saved to: {history_path}")
    
    # ==================== Step 9: Extract Embeddings (Optional) ====================
    if args.extract_embeddings:
        print("\n" + "="*80)
        print("Step 9: Extracting Embeddings")
        print("="*80)
        # Consider extracting embeddings once by running the model on the full graph.
        
        logger.info("Extracting train embeddings...")
        train_embeddings, train_labels = extract_embeddings(
            model, transductive_train_loader, device, args.target_node
        ) 
        # Embeddings for inductive and transductive are not equivalent. 
        # Transductive train embeddings contain information about val and test nodes as well 
        # and so are appropriate for down stream tasks, where the graph has evolved further than test. 
        
        logger.info("Extracting val embeddings...")
        val_embeddings, val_labels = extract_embeddings(
            model, val_loader, device, args.target_node
        )
        
        logger.info("Extracting test embeddings...")
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
        logger.info(f"Embeddings saved to: {embeddings_path}")
        logger.info(f"  Train embeddings shape: {train_embeddings.shape}")
        logger.info(f"  Val embeddings shape: {val_embeddings.shape}")
        logger.info(f"  Test embeddings shape: {test_embeddings.shape}")
    
    # ==================== Final Summary ====================
    print("\n" + "="*80)
    print("Pipeline Completed Successfully!")
    print("="*80)
    print(f"\nFinal Test Results:")
    for metric, value in test_metrics.items():
        print(f"  Test {metric.capitalize()}: {value:.4f}")
    print(f"\nAll outputs saved to: {results_path}")
    print("="*80)


def cli():
    """Command-line interface for the GraphSSL pipeline."""
    parser = argparse.ArgumentParser(
        description="Supervised Learning Pipeline for OGB_MAG Venue Prediction"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for dataset storage"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
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
    
    # Objective arguments
    parser.add_argument(
        "--objective_type",
        type=str,
        default="supervised_node_classification",
        choices=[
            "supervised_node_classification",
            "supervised_link_prediction",
            "self_supervised_node",
            "self_supervised_edge"
        ],
        help="Training objective type"
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="mse",
        choices=[
            "mse",
            "sce",
            "mer",
            "tar"
        ],
        help="loss function"
    )
    parser.add_argument(
        "--target_edge_type",
        type=str,
        default="author,writes,paper",
        help="Target edge type for link prediction (comma-separated: src,relation,dst)"
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.5,
        help="Feature masking ratio for self-supervised node reconstruction"
    )
    parser.add_argument(
        "--neg_sampling_ratio",
        type=float,
        default=1.0,
        help="Negative sampling ratio for link prediction"
    )
    parser.add_argument(
        "--use_edge_decoder",
        action="store_true",
        help="Use MLP-based edge decoder for link prediction (default: dot product)"
    )
    parser.add_argument(
        "--use_feature_decoder",
        action="store_true",
        help="Use MLP-based feature decoder for node reconstruction"
    )
    parser.add_argument(
        "--metric_for_best",
        type=str,
        default="acc",
        help="Metric to use for selecting best model (e.g., 'acc', 'loss')"
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
        "--aggr",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="Aggregator inside SAGEConv"
    )
    parser.add_argument(
        "--aggr_rel",
        type=str,
        default="sum",
        choices=["sum", "mean", "max"],
        help="Aggregator for all relations"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate"
    )
    parser.add_argument(
        "--no_batchnorm",
        action="store_false",
        dest="use_batchnorm",
        default=True,
        help="Disable batch normalization"
    )
    
    # Data loader arguments
    parser.add_argument(
        "--num_neighbors",
        type=int,
        nargs="+",
        default=[30]*2,
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
        "--log_interval",
        type=int,
        default=20,
        help="Log metrics for each interval"
    )    
    parser.add_argument(
        "--extract_embeddings",
        action="store_true",
        help="Extract and save node embeddings after training"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    wandb.init(
        project="graphssl",
        name=f"graphssl_{args.objective_type}_{int(time.time())}",
        config={
            "objective_type": args.objective_type,
            "target_node": args.target_node,
            "target_edge_type": args.target_edge_type,
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "num_neighbors": args.num_neighbors,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "log_interval": args.log_interval,
            "patience": args.patience,
            "aggr": args.aggr,
            "aggr_rel": args.aggr_rel,
            "mask_ratio": args.mask_ratio,
            "neg_sampling_ratio": args.neg_sampling_ratio,
            "use_edge_decoder": args.use_edge_decoder,
            "use_feature_decoder": args.use_feature_decoder,
            "metric_for_best": args.metric_for_best,
        }
    )
    
    # Run pipeline
    run_pipeline(args)


def main():
    """Entry point for the graphssl command."""
    cli()


if __name__ == "__main__":
    cli()
