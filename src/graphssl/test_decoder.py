"""
Test the trained LinkDecoder model and evaluate performance.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Import models and data utilities
from train_decoder import LinkDecoder
from utils.data_utils import load_ogb_mag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_decoder(
    decoder: nn.Module,
    embeddings: dict,
    test_edges,
    test_labels,
    device: torch.device,
    num_examples: int = 10
):
    """
    Evaluate the decoder and show example predictions.
    
    Args:
        decoder: The trained LinkDecoder model
        embeddings: Dictionary of embeddings for each node type
        test_edges: Test edge pairs (src, dst)
        test_labels: Ground truth labels
        device: Device to run on
        num_examples: Number of example predictions to show
    """
    decoder.eval()
    
    # Prepare test data
    src_nodes, dst_nodes = test_edges
    src_emb = embeddings['author'][src_nodes]
    dst_emb = embeddings['paper'][dst_nodes]
    
    # Create DataLoader
    test_dataset = TensorDataset(src_emb, dst_emb, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Collect predictions
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for src_batch, dst_batch, label_batch in test_loader:
            src_batch = src_batch.to(device)
            dst_batch = dst_batch.to(device)
            
            # Get predictions
            logits = decoder(src_batch, dst_batch).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results:")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"{'='*50}\n")
    
    # Print classification report
    print(classification_report(all_labels, all_preds, target_names=['No Link', 'Link']))
    
    # Show example predictions
    logger.info(f"\n{'='*50}")
    logger.info(f"Example Predictions (Random {num_examples} samples):")
    logger.info(f"{'='*50}\n")
    
    # Select random examples
    num_samples = len(all_labels)
    example_indices = np.random.choice(num_samples, min(num_examples, num_samples), replace=False)
    
    examples = []
    for idx in example_indices:
        examples.append({
            'Author ID': src_nodes[idx].item(),
            'Paper ID': dst_nodes[idx].item(),
            'True Label': all_labels[idx],
            'Prediction': all_preds[idx],
            'Probability': f"{all_probs[idx]:.4f}",
            'Correct': '✓' if all_preds[idx] == all_labels[idx] else '✗'
        })
    
    df = pd.DataFrame(examples)
    print(df.to_string(index=False))
    
    # Save examples to CSV
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "test_decoder_examples.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nExample predictions saved to: {csv_path}")
    
    return auc, accuracy


def main():
    """Main function to test the decoder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load embeddings
    embeddings_path = Path("results/embeddings_link.pt")
    if not embeddings_path.exists():
        logger.error(f"Embeddings not found at {embeddings_path}")
        return
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = torch.load(embeddings_path, map_location=device)
    
    # Get embedding dimension
    emb_dim = embeddings['author'].shape[1]
    logger.info(f"Embedding dimension: {emb_dim}")
    
    # Load decoder
    decoder_path = Path("results/best_decoder.pt")
    if not decoder_path.exists():
        decoder_path = Path("results/decoder_checkpoint.pth")
        if not decoder_path.exists():
            logger.error(f"Decoder model not found!")
            return
    
    logger.info(f"Loading decoder from {decoder_path}")
    decoder = LinkDecoder(input_dim=emb_dim, hidden_dim=64)
    
    if str(decoder_path).endswith('.pt'):
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    else:
        checkpoint = torch.load(decoder_path, map_location=device)
        decoder.load_state_dict(checkpoint['model_state'])
    
    decoder = decoder.to(device)
    
    # Load graph data to get test edges
    logger.info("Loading OGB_MAG dataset for test edges...")
    data = load_ogb_mag(root_path="data/ogb_mag", preprocess="metapath2vec")
    
    # Sample test edges (for demonstration, in practice you'd have a held-out test set)
    target_edge_type = ("author", "writes", "paper")
    edge_index = data[target_edge_type].edge_index
    
    # Sample 10,000 positive edges
    num_test = 10000
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)[:num_test]
    test_pos_edges = edge_index[:, perm]
    
    # Generate negative edges
    num_authors = embeddings['author'].shape[0]
    num_papers = embeddings['paper'].shape[0]
    
    neg_src = torch.randint(0, num_authors, (num_test,))
    neg_dst = torch.randint(0, num_papers, (num_test,))
    test_neg_edges = torch.stack([neg_src, neg_dst])
    
    # Combine positive and negative edges
    test_edges = torch.cat([test_pos_edges, test_neg_edges], dim=1)
    test_labels = torch.cat([
        torch.ones(num_test),
        torch.zeros(num_test)
    ]).float()
    
    logger.info(f"Test set: {num_test} positive + {num_test} negative edges")
    
    # Evaluate
    auc, accuracy = evaluate_decoder(
        decoder=decoder,
        embeddings=embeddings,
        test_edges=test_edges,
        test_labels=test_labels,
        device=device,
        num_examples=10
    )
    
    logger.info("\nTest complete!")


if __name__ == "__main__":
    main()
