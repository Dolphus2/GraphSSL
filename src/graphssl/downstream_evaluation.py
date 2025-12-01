import logging
import torch
import torch_geometric.transforms as T
import time
import wandb
from pathlib import Path

from graphssl.utils.data_utils import load_ogb_mag, get_dataset_info, create_edge_splits
from graphssl.utils.downstream import run_downstream_evaluation
from graphssl.utils.graphsage import create_model
from graphssl.utils.args_utils import parse_args

logger = logging.getLogger(__name__)

def run_pipeline(args):
    """Runs the GraphSSL downstream evaluation pipeline based on provided arguments."""
    data_path = Path(args.data_root)
    results_path = Path(args.results_root)
    model_path = Path(args.model_path) if args.model_path else results_path / "model_supervised_node_classification.pt"
    
    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train a model first or specify a valid model path using --model_path."
        )
    
    logger.info(f"Loading model from: {model_path}")
    
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
    
    # Create data splits based on seed
    logger.info(f"Creating data splits with seed={args.seed}...")
    target_edge_type = tuple(args.target_edge_type.split(","))
    train_data, val_data, test_data, train_edge_index, val_edge_index, test_edge_index = create_edge_splits(
        data=data,
        target_edge_type=target_edge_type,
        seed=args.seed,
        node_inductive=True,
        target_node_type=args.target_node,
        dependent=True,
        train_edge_msg_pass_prop=0.8,
        val_edge_msg_pass_prop=0.1,
        test_edge_msg_pass_prop=0.1
    )
    edge_splits = (train_edge_index, val_edge_index, test_edge_index)
    logger.info(f"Data edge splits created: train={train_edge_index.size(1)}, val={val_edge_index.size(1)}, test={test_edge_index.size(1)}")
    
    # Apply ToUndirected transform to original data for downstream evaluation
    logger.debug("Applying ToUndirected transform to full dataset")
    data = T.ToUndirected(merge=True)(data)
    
    # Create model
    logger.info("Initializing model...")
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
    
    # Load model weights
    logger.info(f"Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    logger.info(f"Model was trained with args: {checkpoint.get('args', 'N/A')}")
    logger.info(f"Model test metrics: {checkpoint.get('test_metrics', 'N/A')}")

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
        num_classes=dataset_info['num_classes'],
    )


def cli():
    """Command-line interface for the GraphSSL downstream evaluation pipeline."""
    args = parse_args()
    
    # Add model_path argument specific to downstream evaluation
    if not hasattr(args, 'model_path') or args.model_path is None:
        # Default to results_root/model_supervised_node_classification.pt
        args.model_path = None
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    wandb.init(
        project="graphssl-downstream",
        name=f"downstream_eval_{args.downstream_task}_{int(time.time())}",
        config={
            "seed": args.seed,
            "target_node": args.target_node,
            "target_edge_type": args.target_edge_type,
            "model_path": args.model_path,
            "downstream_task": args.downstream_task,
            "downstream_n_runs": args.downstream_n_runs,
            "downstream_hidden_dim": args.downstream_hidden_dim,
            "downstream_num_layers": args.downstream_num_layers,
            "downstream_dropout": args.downstream_dropout,
            "downstream_batch_size": args.downstream_batch_size,
            "downstream_lr": args.downstream_lr,
            "downstream_weight_decay": args.downstream_weight_decay,
            "downstream_node_epochs": args.downstream_node_epochs,
            "downstream_link_epochs": args.downstream_link_epochs,
            "downstream_patience": args.downstream_patience,
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
