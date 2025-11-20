"""
Enhanced Evaluation Script for Transformer Classifier on OGB_MAG Venue Prediction
Load the trained model, perform predictions on the test set, and display intuitive examples.

This script includes:
- Full test set metrics (Acc, F1, Classification Report)
- Sampled examples with true/pred labels, confidence, and top-3 predictions
- Outputs a table of examples for intuitive visualization

Fixed: Handle mismatched classes in classification_report by using labels from actual data.

Run with: python -m graphssl.enhanced_evaluate_transformer
Assumes the trained model is in results/downstream_transformer/transformer_classifier.pt
and embeddings.pt is in results/.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.data_utils import load_ogb_mag
from torch_geometric.datasets import OGB_MAG

from downstream_transformer_classifier import TransformerClassifier  # Import the model class from previous script

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    test_dataset: TensorDataset,
    device: torch.device,
    num_classes: int = 349,
    num_examples: int = 10
):
    """
    Evaluate the trained model on the test set and generate intuitive examples.

    Args:
        model: Loaded Transformer model
        test_loader: Test data loader
        test_dataset: Test TensorDataset
        device: Device (CPU/GPU)
        num_classes: Number of classes
        num_examples: Number of example samples to display

    Returns:
        Dict with metrics: test_acc, macro_f1, predictions, true_labels, examples_df
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_emb, batch_y in test_loader:
            batch_emb = batch_emb.to(device)
            batch_y = batch_y.to(device)

            out = model(batch_emb)
            pred = out.argmax(dim=1)

            total_correct += (pred == batch_y).sum().item()
            total_samples += batch_y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    test_acc = total_correct / total_samples
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Get actual unique labels in test set to avoid mismatch
    unique_labels = sorted(set(all_labels))
    actual_num_classes = len(unique_labels)
    logger.info(f"Actual number of classes in test set: {actual_num_classes} (out of {num_classes} possible)")

    # Detailed report with labels parameter to match actual classes
    target_names = [f'Venue_{i}' for i in unique_labels]
    class_report_str = classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, digits=4)
    class_report_dict = classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, output_dict=True)

    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info("\nClassification Report:\n" + class_report_str)

    # Generate intuitive examples
    logger.info(f"\nGenerating {num_examples} example predictions for intuitive visualization...")
    sample_indices = np.random.choice(len(test_dataset), num_examples, replace=False)

    examples_data = []
    with torch.no_grad():
        for idx in sample_indices:
            emb, y_true = test_dataset[idx]
            emb_tensor = emb.unsqueeze(0).to(device)
            y_true = y_true.item()

            out = model(emb_tensor)
            probs = F.softmax(out, dim=1)
            y_pred = probs.argmax().item()
            confidence = probs[0, y_pred].item()

            # Top-3 predictions (clamp to valid range 0 to num_classes-1)
            top3_probs, top3_indices = torch.topk(probs, min(3, num_classes))
            top3_indices = top3_indices[0].cpu().numpy()
            top3_probs = top3_probs[0].cpu().numpy()
            top3 = [f"Venue_{i} ({p:.4f})" for i, p in zip(top3_indices, top3_probs)]

            examples_data.append({
                'Sample_Index': idx,
                'True_Venue': f"Venue_{y_true}",
                'Pred_Venue': f"Venue_{y_pred}",
                'Confidence': f"{confidence:.4f}",
                'Top_3_Predictions': '; '.join(top3),
                'Correct': 'Yes' if y_pred == y_true else 'No'
            })

    examples_df = pd.DataFrame(examples_data)
    logger.info("\nExample Predictions Table:")
    logger.info(examples_df.to_string(index=False))

    # Save predictions and examples
    results = {
        'test_acc': test_acc,
        'macro_f1': macro_f1,
        'predictions': np.array(all_preds),
        'true_labels': np.array(all_labels),
        'class_report': class_report_dict,
        'examples_df': examples_df,
        'unique_labels': unique_labels
    }

    np.save("test_predictions.npy", {'predictions': all_preds, 'true_labels': all_labels})
    examples_df.to_csv("example_predictions.csv", index=False)
    logger.info("Full predictions saved to test_predictions.npy")
    logger.info("Example table saved to example_predictions.csv")

    return results


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load trained model state
    model_path = Path(args.results_root) / "downstream_transformer" / "transformer_classifier.pt"
    logger.info(f"Loading trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Load embeddings (for test set)
    embeddings_path = Path(args.results_root) / "embeddings.pt"
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings_dict = torch.load(embeddings_path)
    test_emb = embeddings_dict['test_embeddings']
    test_labels = embeddings_dict['test_labels']

    input_dim = test_emb.shape[1]
    logger.info(f"Test embeddings shape: {test_emb.shape}")

    # Load dataset to get num_classes and args (for model recreation)
    data_path = Path(args.data_root)
    data_path.mkdir(parents=True, exist_ok=True)
    dataset = OGB_MAG(root=str(data_path), preprocess=args.preprocess)
    data = dataset[0]
    num_classes = int(data['paper'].y.max().item() + 1)
    logger.info(f"Number of classes: {num_classes}")

    # Recreate model using saved args
    saved_args = checkpoint['args']
    model = TransformerClassifier(
        input_dim=input_dim,
        hidden_dim=saved_args['hidden_dim'],
        num_classes=num_classes,
        num_layers=saved_args['num_layers'],
        nhead=saved_args['nhead'],
        dropout=saved_args['dropout'],
        dim_feedforward=saved_args['dim_feedforward']
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")

    # Create test loader and dataset
    test_dataset = TensorDataset(test_emb, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Run evaluation with examples
    logger.info("Starting enhanced evaluation on test set...")
    results = evaluate_model(model, test_loader, test_dataset, device, num_classes, args.num_examples)

    # Log summary
    logger.info(f"\n=== Evaluation Summary ===")
    logger.info(f"Test Accuracy: {results['test_acc']:.4f}")
    logger.info(f"Macro F1: {results['macro_f1']:.4f}")
    logger.info("==========================")


def cli():
    parser = argparse.ArgumentParser(description="Enhanced Evaluation of Trained Transformer Classifier with Examples")

    # Paths
    parser.add_argument("--data_root", type=str, default="data", help="Dataset root")
    parser.add_argument("--results_root", type=str, default="results", help="Results root (model and embeddings location)")
    parser.add_argument("--preprocess", type=str, default="metapath2vec", choices=["metapath2vec", "transe"],
                        help="Preprocessing method")

    # Evaluation args
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for evaluation")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of example samples to display")

    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()

    # Logging
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main(args)


if __name__ == "__main__":
    cli()