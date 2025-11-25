import logging
import torch
import argparse
import time
import wandb
from pathlib import Path

from graphssl.utils.data_utils import load_ogb_mag, get_dataset_info
from graphssl.utils.downstream import run_downstream_evaluation

logger = logging.getLogger(__name__)

def run_pipeline(args):
    """Runs the GraphSSL downstream evaluation pipeline based on provided arguments."""
    data_path = Path(args.data_root)
    # Load dataset
    data = load_ogb_mag(
        root_path=str(data_path),
        preprocess=args.preprocess
    )
    
    # Get dataset information
    dataset_info = get_dataset_info(data, target_node_type=args.target_node)
    logger.info(f"Dataset Information:")
    logger.info(f"  Node types: {dataset_info['node_types']}")
    logger.info(f"  Edge types: {dataset_info['edge_types']}")
    logger.info(f"  Number of classes: {dataset_info['num_classes']}")
    logger.info(f"  Node features: {dataset_info['num_features']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    results_path = Path(args.results_root)

    run_downstream_evaluation(
        args=args,
        data=data,
        device=device,
        results_path=results_path,
        num_classes=dataset_info['num_classes'],
    )

def cli():
    """Command-line interface for the GraphSSL pipeline."""
    parser = argparse.ArgumentParser(
        description="Downstream Evaluation Pipeline for OGB_MAG"
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
        default="paper,cites,paper",
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
    
    # Downstream evaluation arguments
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
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

    parser.add_argument(
        "--downstream_max_edges",
        type=int,
        default=2000,
        help="Number of negative samples per positive edge for downstream link prediction"
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
        name=f"graphssl_downstream_test_{int(time.time())}",
        config={
            "target_node": args.target_node,
            "target_edge_type": args.target_edge_type,
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


    # Run pipeline
    run_pipeline(args)

    wandb.finish()


def main():
    """Entry point for the graphssl command."""
    cli()


if __name__ == "__main__":
    cli()
