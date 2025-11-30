"""
Plotting utilities for visualizing training results and downstream evaluation metrics.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(results_path: Path) -> Dict[str, Any]:
    """
    Load all available result files from a results directory.
    
    Args:
        results_path: Path to results directory
    
    Returns:
        Dictionary containing loaded results with keys:
            - 'training_history': Training history dict (if available)
            - 'downstream_node': Downstream node results (if available)
            - 'downstream_link_multiclass': Downstream multiclass link results (if available)
            - 'downstream_link': Downstream link results (if available)
            - 'model_info': Model checkpoint info (if available)
    """
    results_path = Path(results_path)
    results = {}
    
    # Load training history
    history_path = results_path / "training_history.pt"
    if history_path.exists():
        results['training_history'] = torch.load(history_path, map_location='cpu', weights_only=False)
    
    # Load downstream node results
    node_path = results_path / "downstream_node_results.pt"
    if node_path.exists():
        results['downstream_node'] = torch.load(node_path, map_location='cpu', weights_only=False)
    
    # Load downstream multiclass link results
    multiclass_path = results_path / "downstream_link_multiclass_results.pt"
    if multiclass_path.exists():
        results['downstream_link_multiclass'] = torch.load(multiclass_path, map_location='cpu', weights_only=False)
    
    # Load downstream link results
    link_path = results_path / "downstream_link_results.pt"
    if link_path.exists():
        results['downstream_link'] = torch.load(link_path, map_location='cpu', weights_only=False)
    
    # Load model checkpoint for additional info
    model_files = list(results_path.glob("model_*.pt"))
    if model_files:
        model_info = torch.load(model_files[0], map_location='cpu', weights_only=False)
        results['model_info'] = {
            'test_metrics': model_info.get('test_metrics', {}),
            'args': model_info.get('args', {})
        }
    
    return results


def plot_training_curves(
    results_path: Path,
    metrics: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation curves from training history.
    
    Args:
        results_path: Path to results directory
        metrics: List of metric names to plot (e.g., ['loss', 'acc']).
                 If None, plots all available metrics.
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    results = load_results(results_path)
    
    if 'training_history' not in results:
        raise ValueError(f"No training_history.pt found in {results_path}")
    
    history = results['training_history']
    
    # Determine which metrics to plot
    if metrics is None:
        # Extract unique metric names from keys like 'train_loss', 'val_acc'
        train_keys = [k for k in history.keys() if k.startswith('train_')]
        metrics = [k.replace('train_', '') for k in train_keys if k.replace('train_', '') != 'epoch']
        metrics = [m for m in metrics if f'val_{m}' in history]  # Only include if val exists
    
    num_metrics = len(metrics)
    if num_metrics == 0:
        raise ValueError("No valid metrics found in training history")
    
    # Create subplots
    n_cols = min(2, num_metrics)
    n_rows = (num_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    epochs = range(1, len(history[f'train_{metrics[0]}']) + 1)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history and val_key in history:
            ax.plot(epochs, history[train_key], label='Train', marker='o', markersize=3, linewidth=2)
            ax.plot(epochs, history[val_key], label='Validation', marker='s', markersize=3, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_downstream_results(
    results_path: Path,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot downstream evaluation results with error bars.
    
    Args:
        results_path: Path to results directory
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    results = load_results(results_path)
    
    # Collect available downstream results
    downstream_tasks = []
    task_names = []
    
    if 'downstream_node' in results:
        downstream_tasks.append(results['downstream_node'])
        task_names.append('Node Classification')
    
    if 'downstream_link_multiclass' in results:
        downstream_tasks.append(results['downstream_link_multiclass'])
        task_names.append('Link Prediction\n(Multiclass)')
    
    if 'downstream_link' in results:
        downstream_tasks.append(results['downstream_link'])
        task_names.append('Link Prediction\n(Binary)')
    
    if not downstream_tasks:
        raise ValueError(f"No downstream results found in {results_path}")
    
    # Create figure
    num_tasks = len(downstream_tasks)
    fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 5))
    
    if num_tasks == 1:
        axes = [axes]
    
    for idx, (task_result, task_name) in enumerate(zip(downstream_tasks, task_names)):
        ax = axes[idx]
        
        # Extract metrics
        metrics_to_plot = []
        if 'test_acc_mean' in task_result:
            metrics_to_plot.append(('Accuracy', 'test_acc_mean', 'test_acc_std'))
        if 'test_loss_mean' in task_result:
            metrics_to_plot.append(('Loss', 'test_loss_mean', 'test_loss_std'))
        if 'test_f1_mean' in task_result:
            metrics_to_plot.append(('F1 Score', 'test_f1_mean', 'test_f1_std'))
        if 'test_precision_mean' in task_result:
            metrics_to_plot.append(('Precision', 'test_precision_mean', 'test_precision_std'))
        if 'test_recall_mean' in task_result:
            metrics_to_plot.append(('Recall', 'test_recall_mean', 'test_recall_std'))
        
        x_pos = np.arange(len(metrics_to_plot))
        means = [task_result[mean_key] for _, mean_key, _ in metrics_to_plot]
        stds = [task_result[std_key] for _, _, std_key in metrics_to_plot]
        labels = [label for label, _, _ in metrics_to_plot]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_ylabel('Score')
        ax.set_title(task_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, max(means) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_downstream_distribution(
    results_path: Path,
    metric: str = 'test_accuracies',
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot distribution of downstream evaluation results across multiple runs.
    
    Args:
        results_path: Path to results directory
        metric: Name of metric to plot (e.g., 'test_accuracies', 'test_losses')
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    results = load_results(results_path)
    
    # Collect available downstream results
    data_to_plot = []
    task_names = []
    
    if 'downstream_node' in results and metric in results['downstream_node']:
        data_to_plot.append(results['downstream_node'][metric])
        task_names.append('Node\nClassification')
    
    if 'downstream_link_multiclass' in results and metric in results['downstream_link_multiclass']:
        data_to_plot.append(results['downstream_link_multiclass'][metric])
        task_names.append('Link\n(Multiclass)')
    
    if 'downstream_link' in results and metric in results['downstream_link']:
        data_to_plot.append(results['downstream_link'][metric])
        task_names.append('Link\n(Binary)')
    
    if not data_to_plot:
        raise ValueError(f"Metric '{metric}' not found in any downstream results in {results_path}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, 3 * len(data_to_plot)), 6))
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=task_names, patch_artist=True,
                    notch=True, showmeans=True)
    
    # Customize colors
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, data in enumerate(data_to_plot):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.5, color='darkblue', s=30)
    
    metric_name = metric.replace('test_', '').replace('_', ' ').title()
    ax.set_ylabel(metric_name)
    ax.set_title(f'Distribution of {metric_name} Across Multiple Runs', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_all_results(
    results_path: Path,
    save_dir: Optional[Path] = None,
    show: bool = True
) -> Dict[str, plt.Figure]:
    """
    Create all available plots for a results directory.
    
    Args:
        results_path: Path to results directory
        save_dir: Directory to save figures (if None, doesn't save)
        show: Whether to display the plots
    
    Returns:
        Dictionary of figure names to matplotlib Figure objects
    """
    results_path = Path(results_path)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Plot training curves
    try:
        fig = plot_training_curves(
            results_path,
            save_path=save_dir / 'training_curves.png' if save_dir else None,
            show=show
        )
        figures['training_curves'] = fig
    except (ValueError, KeyError) as e:
        print(f"Could not plot training curves: {e}")
    
    # Plot downstream results
    try:
        fig = plot_downstream_results(
            results_path,
            save_path=save_dir / 'downstream_results.png' if save_dir else None,
            show=show
        )
        figures['downstream_results'] = fig
    except (ValueError, KeyError) as e:
        print(f"Could not plot downstream results: {e}")
    
    # Plot downstream distributions
    for metric in ['test_accuracies', 'test_losses', 'test_f1_scores']:
        try:
            fig = plot_downstream_distribution(
                results_path,
                metric=metric,
                save_path=save_dir / f'downstream_dist_{metric}.png' if save_dir else None,
                show=show
            )
            figures[f'downstream_dist_{metric}'] = fig
        except (ValueError, KeyError) as e:
            pass  # Metric not available
    
    # Plot confusion matrix for link multiclass
    try:
        fig = plot_confusion_matrix(
            results_path,
            save_path=save_dir / 'confusion_matrix.png' if save_dir else None,
            show=show
        )
        if fig is not None:
            figures['confusion_matrix'] = fig
    except (ValueError, KeyError) as e:
        pass  # Confusion matrix not available
    
    return figures


def print_results_summary(results_path: Path) -> None:
    """
    Print a text summary of all results.
    
    Args:
        results_path: Path to results directory
    """
    results = load_results(results_path)
    
    print("=" * 80)
    print(f"Results Summary: {results_path.name}")
    print("=" * 80)
    
    # Model info
    if 'model_info' in results:
        print("\n--- Model Configuration ---")
        args = results['model_info'].get('args', {})
        if args:
            print(f"Objective: {args.get('objective_type', 'N/A')}")
            print(f"Hidden channels: {args.get('hidden_channels', 'N/A')}")
            print(f"Num layers: {args.get('num_layers', 'N/A')}")
            print(f"Epochs: {args.get('epochs', 'N/A')}")
        
        test_metrics = results['model_info'].get('test_metrics', {})
        if test_metrics:
            print("\n--- Test Metrics (Main Task) ---")
            for key, value in test_metrics.items():
                print(f"{key.capitalize()}: {value:.4f}")
    
    # Training history summary
    if 'training_history' in results:
        history = results['training_history']
        print("\n--- Training History ---")
        print(f"Total epochs: {len(history.get('train_loss', []))}")
        if 'train_loss' in history:
            print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if 'val_loss' in history:
            print(f"Best val loss: {min(history['val_loss']):.4f}")
        if 'val_acc' in history:
            print(f"Best val accuracy: {max(history['val_acc']):.4f}")
    
    # Downstream results
    if 'downstream_node' in results:
        node_res = results['downstream_node']
        print("\n--- Downstream Node Classification ---")
        print(f"Test Accuracy: {node_res['test_acc_mean']:.4f} ± {node_res['test_acc_std']:.4f}")
        print(f"Test Loss: {node_res['test_loss_mean']:.4f} ± {node_res['test_loss_std']:.4f}")
    
    if 'downstream_link_multiclass' in results:
        link_res = results['downstream_link_multiclass']
        print("\n--- Downstream Link Prediction (Multiclass) ---")
        if 'test_acc_mean' in link_res:
            print(f"Test Accuracy: {link_res['test_acc_mean']:.4f} ± {link_res['test_acc_std']:.4f}")
        if 'test_f1_mean' in link_res:
            print(f"Test F1: {link_res['test_f1_mean']:.4f} ± {link_res['test_f1_std']:.4f}")
        if 'test_precision_mean' in link_res:
            print(f"Test Precision: {link_res['test_precision_mean']:.4f} ± {link_res['test_precision_std']:.4f}")
        if 'test_recall_mean' in link_res:
            print(f"Test Recall: {link_res['test_recall_mean']:.4f} ± {link_res['test_recall_std']:.4f}")
    
    if 'downstream_link' in results:
        link_res = results['downstream_link']
        print("\n--- Downstream Link Prediction (Binary) ---")
        print(f"Test Accuracy: {link_res['test_acc_mean']:.4f} ± {link_res['test_acc_std']:.4f}")
        print(f"Test Loss: {link_res['test_loss_mean']:.4f} ± {link_res['test_loss_std']:.4f}")
    
    print("=" * 80)


def plot_confusion_matrix(
    results_path: Path,
    normalize: bool = True,
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot confusion matrix for link prediction multiclass classification.
    
    Args:
        results_path: Path to results directory
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object if confusion matrix exists, None otherwise
    """
    results = load_results(results_path)
    
    if 'downstream_link_multiclass' not in results:
        print(f"No downstream_link_multiclass results found in {results_path}")
        return None
    
    link_res = results['downstream_link_multiclass']
    
    if 'confusion_matrix' not in link_res:
        print(f"No confusion matrix found in downstream_link_multiclass results")
        return None
    
    cm = link_res['confusion_matrix']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix\n(Link Prediction Multiclass)'
    else:
        cm_display = cm
        fmt = '.0f'
        title = 'Confusion Matrix\n(Link Prediction Multiclass)'
    
    # Plot heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    classes = ['No Link', 'Link']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f'{cm_display[i, j]:.1%}\n({cm[i, j]:.0f})'
            else:
                text = f'{cm[i, j]:.0f}'
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm_display[i, j] > thresh else "black",
                   fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
