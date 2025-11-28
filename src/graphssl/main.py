"""
GraphSSL: Supervised Learning Pipeline for OGB_MAG Dataset
Venue prediction using heterogeneous GraphSAGE

Run with: python -m graphssl.main
"""
import logging
import torch
import torch_geometric.transforms as T
import wandb
from pathlib import Path

from graphssl.utils.data_utils import load_ogb_mag, create_neighbor_loaders, create_link_loaders, get_dataset_info, \
    extract_and_save_embeddings, create_edge_splits, subsample_dataset
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
from graphssl.utils.downstream import (
    run_downstream_evaluation
)
from graphssl.utils.args_utils import parse_args, setup_logging_and_wandb

logging.basicConfig(level=logging.DEBUG)
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
    
    # Subsample dataset in test mode for faster testing
    if args.test_mode:
        logger.info("Test mode: subsampling dataset for faster testing")
        data = subsample_dataset(
            data,
            target_node_type=args.target_node,
            max_nodes=5000,  # Keep only 5000 nodes for quick testing
            seed=args.seed
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
    
    # Apply test mode optimizations
    if args.test_mode:
        # Reduce neighbor sampling for faster testing
        if args.num_neighbors == [15, 10]:  # Only override if using defaults
            args.num_neighbors = [10, 5]  # Much fewer neighbors
            logger.info(f"Test mode: reducing num_neighbors to {args.num_neighbors}")
        # Reduce downstream runs
        if args.downstream_n_runs > 1:
            args.downstream_n_runs = 1
            logger.info(f"Test mode: reducing downstream_n_runs to 1")
        # Reduce downstream epochs
        if args.downstream_node_epochs > 3 or args.downstream_link_epochs > 3:
            if args.downstream_node_epochs > 3: args.downstream_node_epochs = 3
            if args.downstream_link_epochs > 3: args.downstream_link_epochs = 3
            logger.info(f"Test mode: reducing downstream_node_epochs and downstream_link_epochs to 3")
    
    # Step 2a: Create edge/node splits
    target_edge_type = tuple(args.target_edge_type.split(","))
    train_data, val_data, test_data, train_edge_index, val_edge_index, test_edge_index = create_edge_splits(
        data=data,
        target_edge_type=target_edge_type,
        seed=args.seed,
        node_inductive=args.node_inductive,
        target_node_type=args.target_node,
        dependent=args.dependent_node_edge_data_split,
        train_edge_msg_pass_prop=args.edge_msg_pass_prop[0],
        val_edge_msg_pass_prop=args.edge_msg_pass_prop[1],
        test_edge_msg_pass_prop=args.edge_msg_pass_prop[2]
    )
    edge_splits = (train_edge_index, val_edge_index, test_edge_index)
    logger.info(f"Data splits created (seed={args.seed}): train={train_edge_index.size(1)}, val={val_edge_index.size(1)}, test={test_edge_index.size(1)}")
    if args.node_inductive:
        logger.info("Using node inductive learning")
    
    # Apply ToUndirected transform to original data for global loader
    logger.debug("Applying ToUndirected transform to full dataset for global loader")
    data = T.ToUndirected(merge=True)(data)
    
    # Step 2b: Create NeighborLoaders (for node embeddings)
    train_loader, val_loader, test_loader, global_loader = create_neighbor_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        full_data=data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_node_type=args.target_node
    )
    
    # Step 2c: Create LinkNeighborLoaders for link-based objectives
    # Default to using neighbor loaders for training
    train_loader_for_training = train_loader
    val_loader_for_training = val_loader
    test_loader_for_testing = test_loader
    
    if args.objective_type == "supervised_link_prediction" or args.objective_type == "self_supervised_edge":
        link_train_loader, link_val_loader, link_test_loader = create_link_loaders(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_edge_index=train_edge_index,
            val_edge_index=val_edge_index,
            test_edge_index=test_edge_index,
            target_edge_type=target_edge_type,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            neg_sampling_ratio=args.neg_sampling_ratio,
            num_workers=args.num_workers
        )
        logger.info("LinkNeighborLoaders created for link-based objective")
        # Override with link loaders for link-based objectives
        train_loader_for_training = link_train_loader
        val_loader_for_training = link_val_loader
        test_loader_for_testing = link_test_loader
    
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
            decoder=decoder,
            loss_fn=args.loss_fn,
            mer_weight=args.mer_weight,
            tar_weight=args.tar_weight,
            pfp_weight=args.pfp_weight,
            tar_temperature=args.tar_temperature
        )
        logger.info(f"Objective: Self-Supervised Edge Reconstruction on '{target_edge_type}'")
        logger.info(f"  Loss function: {args.loss_fn}")
        logger.info(f"  Negative sampling ratio: {args.neg_sampling_ratio}")
        if args.loss_fn == "combined_loss":
            logger.info(f"  Combined loss weights - MER: {args.mer_weight}, TAR: {args.tar_weight}, PFP: {args.pfp_weight}")
            logger.info(f"  TAR temperature: {args.tar_temperature}")
    
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
    
    history = train_model(
        model=model,
        train_loader=train_loader_for_training,
        val_loader=val_loader_for_training,
        optimizer=optimizer,
        objective=objective,
        device=device,
        num_epochs=args.epochs,
        log_interval=args.log_interval,
        early_stopping_patience=args.patience,
        checkpoint_dir=str(checkpoint_dir),
        verbose=True,
        metric_for_best=args.metric_for_best,
        disable_tqdm=args.disable_tqdm
    )
    
    # ==================== Step 7: Test Model ====================
    print("\n" + "="*80)
    print("Step 7: Testing Model")
    print("="*80)
    
    test_metrics = test_model(
        model=model,
        test_loader=test_loader_for_testing,
        objective=objective,
        device=device,
        disable_tqdm=args.disable_tqdm
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
        
        embeddings_path = results_path / "embeddings.pt"
        embeddings_data = extract_and_save_embeddings(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            global_loader=global_loader,
            device=device,
            target_node_type=args.target_node,
            embeddings_path=embeddings_path,
            logger=logger,
            disable_tqdm=args.disable_tqdm
        )
    
    # ==================== Step 10: Downstream Evaluation (Optional) ====================
    if args.downstream_eval:
        print("\n" + "="*80)
        print("Step 10: Downstream Evaluation")
        print("="*80)
        
        run_downstream_evaluation(
            args=args,
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            full_data=data,
            edge_splits=edge_splits,
            device=device,
            results_path=results_path,
            num_classes=dataset_info['num_classes']
        )
    
    # ==================== Final Summary ====================
    # Clean up wandb
    wandb.finish()
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
    args = parse_args()
    setup_logging_and_wandb(args)
    run_pipeline(args)


def main():
    """Entry point for the graphssl command."""
    cli()


if __name__ == "__main__":
    cli()
