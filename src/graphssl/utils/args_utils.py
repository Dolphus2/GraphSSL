"""
Command-line argument parsing utilities for GraphSSL pipeline.
"""
import argparse
import logging
import wandb
import time

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for the GraphSSL pipeline."""
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
            "bce",
            "mer",
            "tar",
            "pfp",
            "combined_loss"
        ],
        help="Loss function for self-supervised learning. For node: mse/sce. For edge: bce/mer/tar/pfp/combined_loss"
    )
    parser.add_argument(
        "--mer_weight",
        type=float,
        default=1.0,
        help="Weight for MER loss in HGMAE combined loss"
    )
    parser.add_argument(
        "--tar_weight",
        type=float,
        default=1.0,
        help="Weight for TAR loss in HGMAE combined loss"
    )
    parser.add_argument(
        "--pfp_weight",
        type=float,
        default=1.0,
        help="Weight for PFP loss in HGMAE combined loss"
    )
    parser.add_argument(
        "--tar_temperature",
        type=float,
        default=0.5,
        help="Temperature parameter for TAR loss"
    )
    parser.add_argument(
        "--target_edge_type",
        type=str,
        default="paper, has_topic, field_of_study",
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
    parser.add_argument(
        "--node_inductive",
        action="store_true",
        default=True,
        help="Use node inductive learning for link prediction (removes val/test nodes from train data)"
    )
    parser.add_argument(
        "--no_node_inductive",
        action="store_false",
        dest="node_inductive",
        help="Disable node inductive learning for link prediction"
    )
    parser.add_argument(
        "--dependent_node_edge_data_split",
        action="store_true",
        default=True,
        help="Use dependent edge splits (control edges incident to split nodes for message passing)"
    )
    parser.add_argument(
        "--no_dependent_node_edge_data_split",
        action="store_false",
        dest="dependent_node_edge_data_split",
        help="Disable dependent edge splits"
    )
    parser.add_argument(
        "--edge_msg_pass_prop",
        type=float,
        nargs=3,
        default=[0.7, 0.7, 0.7],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Proportion of edges incident to split nodes to keep for message passing (train, val, test). Only used with --dependent_node_edge_data_split"
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
    
    # Downstream evaluation arguments
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        default=True,
        help="Run downstream evaluation tasks"
    )
    parser.add_argument(
        "--downstream_task",
        type=str,
        default="both",
        choices=["node", "link", "both"],
        help="Downstream task type: node property prediction, link prediction, or both"
    )
    parser.add_argument(
        "--downstream_n_runs",
        type=int,
        default=10,
        help="Number of independent runs for downstream evaluation (for uncertainty estimation)"
    )
    parser.add_argument(
        "--downstream_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for downstream MLP classifiers"
    )
    parser.add_argument(
        "--downstream_num_layers",
        type=int,
        default=2,
        help="Number of layers in downstream MLP classifiers"
    )
    parser.add_argument(
        "--downstream_dropout",
        type=float,
        default=0.5,
        help="Dropout rate for downstream classifiers"
    )
    parser.add_argument(
        "--downstream_batch_size",
        type=int,
        default=1024,
        help="Batch size for downstream classifier training"
    )
    parser.add_argument(
        "--downstream_lr",
        type=float,
        default=0.001,
        help="Learning rate for downstream classifiers"
    )
    parser.add_argument(
        "--downstream_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for downstream classifiers"
    )
    parser.add_argument(
        "--downstream_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs for downstream classifier training"
    )
    parser.add_argument(
        "--downstream_patience",
        type=int,
        default=10,
        help="Early stopping patience for downstream classifiers"
    )
    parser.add_argument(
        "--downstream_neg_samples",
        type=int,
        default=1,
        help="Number of negative samples per positive edge for downstream link prediction"
    )
    
    # Testing options
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode with minimal settings for quick checks (subsamples dataset, reduces neighbors, fewer downstream runs)"
    )
    parser.add_argument(
        "--skip_downstream",
        action="store_false",
        dest="downstream_eval",
        help="Skip downstream evaluation entirely for fastest testing (only train the model)"
    )
    
    return parser.parse_args()


def setup_logging_and_wandb(args):
    """Configure logging and initialize Weights & Biases."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Initialize wandb
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
            "loss_fn": args.loss_fn,
            "mer_weight": args.mer_weight,
            "tar_weight": args.tar_weight,
            "pfp_weight": args.pfp_weight,
            "tar_temperature": args.tar_temperature,
            "node_inductive": args.node_inductive,
            "downstream_eval": args.downstream_eval,
            "downstream_task": args.downstream_task,
            "downstream_n_runs": args.downstream_n_runs,
            "downstream_hidden_dim": args.downstream_hidden_dim,
            "downstream_num_layers": args.downstream_num_layers,
            "downstream_dropout": args.downstream_dropout,
            "downstream_batch_size": args.downstream_batch_size,
            "downstream_lr": args.downstream_lr,
            "downstream_weight_decay": args.downstream_weight_decay,
            "downstream_epochs": args.downstream_epochs,
            "downstream_patience": args.downstream_patience,
            "downstream_neg_samples": args.downstream_neg_samples,
        }
    )
    
    if args.test_mode:
        logger.info("Test mode enabled - using minimal settings for quick checks")
