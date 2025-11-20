"""
Downstream Transformer Classifier for Node Classification using Pre-trained Embeddings
Venue Prediction on OGB_MAG using Transformer decoder on self-supervised embeddings.

This script replaces the MLP with a Transformer-based classifier for improved performance.
Run with: python -m graphssl.downstream_transformer_classifier
Assumes embeddings.pt is in the results directory from the self-supervised training.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from utils.data_utils import load_ogb_mag
from torch_geometric.datasets import OGB_MAG

logger = logging.getLogger(__name__)

class TransformerClassifier(nn.Module):
    """
    Transformer-based decoder for node classification.
    Treats embeddings as sequences (with seq_len=1 for single nodes) and uses global pooling.
    This captures more complex interactions than MLP, potentially improving accuracy.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_classes: int = 349,
            num_layers: int = 2,
            nhead: int = 8,
            dropout: float = 0.1,
            dim_feedforward: int = 512
    ):
        super().__init__()
        # Project input to model dim if needed (Transformer expects consistent d_model)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global average pooling and output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, input_dim) -> treat as seq_len=1
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, hidden_dim)

        # Transformer forward (self-attention on single "token")
        x = self.transformer(x)  # (batch, 1, hidden_dim)
        x = x.squeeze(1)  # (batch, hidden_dim)

        # Pooling (redundant for seq_len=1, but scalable if seq_len >1 later)
        x = self.pool(x.unsqueeze(-1)).squeeze(-1)

        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_classifier(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_epochs: int = 100,
        patience: int = 10
) -> dict:
    """
    Train the Transformer classifier with early stopping.

    Args:
        model: Transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        num_epochs: Max epochs
        patience: Early stopping patience

    Returns:
        Training history dict
    """
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_transformer.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(torch.load("best_transformer.pt"))
    return history


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_emb, batch_y in loader:
        batch_emb = batch_emb.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        out = model(batch_emb)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_y.size(0)
        pred = out.argmax(dim=1)
        total_correct += (pred == batch_y).sum().item()
        total_samples += batch_y.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_emb, batch_y in loader:
        batch_emb = batch_emb.to(device)
        batch_y = batch_y.to(device)

        out = model(batch_emb)
        loss = criterion(out, batch_y)

        total_loss += loss.item() * batch_y.size(0)
        pred = out.argmax(dim=1)
        total_correct += (pred == batch_y).sum().item()
        total_samples += batch_y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def test_classifier(model, test_loader, device):
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
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")

    # Save predictions
    np.save("test_predictions.npy", {"preds": all_preds, "labels": all_labels})

    return test_acc, macro_f1


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load embeddings
    embeddings_path = Path(args.results_root) / "embeddings.pt"
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings_dict = torch.load(embeddings_path)

    train_emb = embeddings_dict['train_embeddings']
    train_labels = embeddings_dict['train_labels']
    val_emb = embeddings_dict['val_embeddings']
    val_labels = embeddings_dict['val_labels']
    test_emb = embeddings_dict['test_embeddings']
    test_labels = embeddings_dict['test_labels']

    input_dim = train_emb.shape[1]
    logger.info(f"Embeddings shape - Train: {train_emb.shape}, Val: {val_emb.shape}, Test: {test_emb.shape}")

    # Load dataset to get num_classes
    data_path = Path(args.data_root)
    data_path.mkdir(parents=True, exist_ok=True)
    dataset = OGB_MAG(root=str(data_path), preprocess=args.preprocess)
    data = dataset[0]
    num_classes = int(data['paper'].y.max().item() + 1)
    logger.info(f"Number of classes: {num_classes}")

    # Create datasets and loaders
    train_dataset = TensorDataset(train_emb, train_labels)
    val_dataset = TensorDataset(val_emb, val_labels)
    test_dataset = TensorDataset(test_emb, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create Transformer model
    model = TransformerClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward
    ).to(device)

    logger.info(f"Transformer Classifier created: input_dim={input_dim}, hidden_dim={args.hidden_dim}, "
                f"num_layers={args.num_layers}, nhead={args.nhead}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Train
    logger.info("Starting Transformer classifier training...")
    history = train_classifier(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=args.epochs, patience=args.patience
    )

    # Test
    logger.info("Evaluating on test set...")
    test_acc, macro_f1 = test_classifier(model, test_loader, device)

    # Save model and history
    results_path = Path(args.results_root) / "downstream_transformer"
    results_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'macro_f1': macro_f1,
        'history': history,
        'args': vars(args)
    }, results_path / "transformer_classifier.pt")

    torch.save(history, results_path / "training_history.pt")

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Best Test Accuracy: {test_acc:.4f} | Macro F1: {macro_f1:.4f}")


def cli():
    parser = argparse.ArgumentParser(
        description="Downstream Transformer Node Classification using Pre-trained Embeddings")

    # Paths
    parser.add_argument("--data_root", type=str, default="data", help="Dataset root")
    parser.add_argument("--results_root", type=str, default="results", help="Results root (embeddings.pt location)")
    parser.add_argument("--preprocess", type=str, default="metapath2vec", choices=["metapath2vec", "transe"],
                        help="Preprocessing method")

    # Model args
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dim for Transformer")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Inner feedforward dim")

    # Training args
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()

    # Logging
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main(args)


if __name__ == "__main__":
    cli()