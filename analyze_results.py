"""
Analyze and compare results from all 10 experiments.

Usage:
    python analyze_results.py --results_root results
"""

import torch
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_checkpoint_metrics(checkpoint_path: Path) -> Dict[str, float]:
    """Load test metrics from checkpoint file."""
    if not checkpoint_path.exists():
        return {}
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'test_metrics' in checkpoint:
        return checkpoint['test_metrics']
    return {}


def load_downstream_results(results_path: Path) -> Dict[str, float]:
    """Load downstream evaluation results."""
    if not results_path.exists():
        return {}
    
    results = torch.load(results_path, map_location='cpu')
    return {
        'mean': results.get('test_acc_mean', 0.0),
        'std': results.get('test_acc_std', 0.0),
        'runs': results.get('test_accuracies', [])
    }


def analyze_experiment(exp_name: str, exp_dir: Path) -> Dict[str, Any]:
    """Analyze results from a single experiment."""
    results = {
        'experiment': exp_name,
        'path': str(exp_dir)
    }
    
    # Check if directory exists
    if not exp_dir.exists():
        results['status'] = 'NOT RUN'
        return results
    
    results['status'] = 'COMPLETED'
    
    # Load main training results
    checkpoint_files = list(exp_dir.glob('model_*.pt'))
    if checkpoint_files:
        checkpoint_path = checkpoint_files[0]
        metrics = load_checkpoint_metrics(checkpoint_path)
        if metrics:
            for key, value in metrics.items():
                results[f'main_{key}'] = value
    
    # Load downstream node prediction results
    node_results_path = exp_dir / 'downstream_node_results.pt'
    if node_results_path.exists():
        node_results = load_downstream_results(node_results_path)
        results['downstream_node_mean'] = node_results['mean']
        results['downstream_node_std'] = node_results['std']
    
    # Load downstream link prediction results
    link_results_path = exp_dir / 'downstream_link_results.pt'
    if link_results_path.exists():
        link_results = load_downstream_results(link_results_path)
        results['downstream_link_mean'] = link_results['mean']
        results['downstream_link_std'] = link_results['std']
    
    # Check for embeddings
    embeddings_path = exp_dir / 'embeddings.pt'
    if embeddings_path.exists():
        embeddings_data = torch.load(embeddings_path, map_location='cpu')
        if 'train_embeddings' in embeddings_data:
            results['embedding_dim'] = embeddings_data['train_embeddings'].shape[1]
            results['num_train_nodes'] = embeddings_data['train_embeddings'].shape[0]
    
    return results


def create_summary_table(all_results: list) -> pd.DataFrame:
    """Create a summary table of all experiments."""
    # Define the mapping of experiments to their outputs
    experiment_mapping = {
        'exp1_supervised_node': {
            'description': 'Supervised Node Classification',
            'encoder': 'Encoder 1',
            'accuracy_1': 'main_acc',  # Accuracy 1
            'accuracy_3': 'downstream_link_mean'  # Accuracy 3 (Enc1 + Link)
        },
        'exp2_supervised_link': {
            'description': 'Supervised Link Prediction',
            'encoder': 'Encoder 2',
            'accuracy_2': 'main_acc',  # Accuracy 2
            'accuracy_4': 'downstream_node_mean'  # Accuracy 4 (Enc2 + Node)
        },
        'exp5_selfsup_node_graphmae': {
            'description': 'Self-Supervised Node (GraphMAE)',
            'encoder': 'Encoder 4',
            'accuracy_5': 'downstream_node_mean',  # Accuracy 5 (Enc4 + Node)
            'accuracy_6': 'downstream_link_mean'   # Accuracy 6 (Enc4 + Link)
        },
        'exp8_selfsup_link_combined_loss': {
            'description': 'Self-Supervised Link (HGMAE)',
            'encoder': 'Encoder 5',
            'accuracy_7': 'downstream_node_mean',  # Accuracy 7 (Enc5 + Node)
            'accuracy_8': 'downstream_link_mean'   # Accuracy 8 (Enc5 + Link)
        }
    }
    
    summary_rows = []
    
    for result in all_results:
        exp_key = Path(result['path']).name
        if exp_key not in experiment_mapping:
            continue
        
        mapping = experiment_mapping[exp_key]
        row = {
            'Experiment': mapping['description'],
            'Encoder': mapping['encoder'],
            'Status': result['status']
        }
        
        # Add accuracy metrics
        for acc_name, metric_key in mapping.items():
            if acc_name.startswith('accuracy_'):
                acc_num = acc_name.split('_')[1]
                if metric_key in result:
                    value = result[metric_key]
                    # Check if there's a std
                    std_key = metric_key.replace('_mean', '_std')
                    if std_key in result:
                        row[f'Accuracy {acc_num}'] = f"{value:.4f} ± {result[std_key]:.4f}"
                    else:
                        row[f'Accuracy {acc_num}'] = f"{value:.4f}"
                else:
                    row[f'Accuracy {acc_num}'] = 'N/A'
        
        # Add embedding info if available
        if 'embedding_dim' in result:
            row['Embedding Dim'] = result['embedding_dim']
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def print_detailed_results(all_results: list):
    """Print detailed results for each experiment."""
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for result in all_results:
        print(f"\n{result['experiment']}")
        print("-" * 80)
        print(f"Path: {result['path']}")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'NOT RUN':
            continue
        
        # Print main metrics
        main_metrics = {k: v for k, v in result.items() 
                       if k.startswith('main_') and isinstance(v, (int, float))}
        if main_metrics:
            print("\nMain Training Metrics:")
            for key, value in main_metrics.items():
                metric_name = key.replace('main_', '')
                print(f"  {metric_name}: {value:.4f}")
        
        # Print downstream metrics
        if 'downstream_node_mean' in result:
            print(f"\nDownstream Node Prediction:")
            print(f"  Accuracy: {result['downstream_node_mean']:.4f} ± {result['downstream_node_std']:.4f}")
        
        if 'downstream_link_mean' in result:
            print(f"\nDownstream Link Prediction:")
            print(f"  Accuracy: {result['downstream_link_mean']:.4f} ± {result['downstream_link_std']:.4f}")
        
        # Print embedding info
        if 'embedding_dim' in result:
            print(f"\nEmbeddings:")
            print(f"  Dimension: {result['embedding_dim']}")
            print(f"  Train nodes: {result['num_train_nodes']}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze results from all GraphSSL experiments"
    )
    parser.add_argument(
        '--results_root',
        type=str,
        default='results',
        help='Root directory containing experiment results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file to save summary table'
    )
    
    args = parser.parse_args()
    
    results_root = Path(args.results_root)
    
    if not results_root.exists():
        print(f"Error: Results directory '{results_root}' does not exist.")
        print("Please run experiments first or specify correct --results_root")
        return
    
    # Define experiments to analyze
    experiments = [
        ('Experiment 1', results_root / 'exp1_supervised_node'),
        ('Experiment 2', results_root / 'exp2_supervised_link'),
        ('Experiment 5', results_root / 'exp5_selfsup_node_graphmae'),
        ('Experiment 8', results_root / 'exp8_selfsup_link_combined_loss'),
    ]
    
    # Analyze all experiments
    all_results = []
    for exp_name, exp_dir in experiments:
        result = analyze_experiment(exp_name, exp_dir)
        all_results.append(result)
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Print detailed results
    print_detailed_results(all_results)
    
    # Print experiment mapping
    print("\n" + "="*80)
    print("EXPERIMENT TO ACCURACY MAPPING")
    print("="*80)
    print("""
    Accuracy 1: Experiment 1 (Supervised Node) - Main training accuracy
    Accuracy 2: Experiment 2 (Supervised Link) - Main training accuracy
    Accuracy 3: Experiment 1 → Downstream link prediction
    Accuracy 4: Experiment 2 → Downstream node prediction
    Accuracy 5: Experiment 5 (GraphMAE) → Downstream node prediction
    Accuracy 6: Experiment 5 (GraphMAE) → Downstream link prediction
    Accuracy 7: Experiment 8 (HGMAE) → Downstream node prediction
    Accuracy 8: Experiment 8 (HGMAE) → Downstream link prediction
    """)
    
    # Save to CSV if requested
    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f"\nSummary table saved to: {args.output}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

