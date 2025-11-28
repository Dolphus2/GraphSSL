"""
Downstream-only runner using precomputed embeddings.

Loads embeddings from results/embeddings.pt (or a user-specified path),
recreates edge splits for link prediction, and evaluates node property
prediction and/or link prediction without re-training the encoder.
"""
import os
import argparse
from pathlib import Path
import torch

# Disable wandb logging for this standalone downstream script
os.environ.setdefault("WANDB_DISABLED", "true")

from graphssl.utils.data_utils import load_ogb_mag, create_edge_splits
from graphssl.utils.downstream import (
    evaluate_node_property_prediction,
    evaluate_link_prediction,
)


def parse_args():
    parser = argparse.ArgumentParser("Downstream evaluator (no training)")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="results/embeddings.pt",
        help="Path to embeddings.pt produced by --extract_embeddings",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Dataset root (used to reload graph for edge splits)",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="metapath2vec",
        choices=["metapath2vec", "transe"],
        help="Preprocess option used when embeddings were generated",
    )
    parser.add_argument(
        "--target_node",
        type=str,
        default="paper",
        help="Target node type contained in embeddings.pt",
    )
    parser.add_argument(
        "--target_edge_type",
        type=str,
        default="paper,cites,paper",
        help="Edge type for link prediction (src,relation,dst)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for create_edge_splits to match training",
    )
    parser.add_argument(
        "--downstream_task",
        type=str,
        default="both",
        choices=["node", "link", "both"],
        help="Which downstream evaluation(s) to run",
    )
    parser.add_argument(
        "--node_sample_ratio",
        type=float,
        default=1.0,
        help="Subsample ratio for nodes in embeddings (0<r<=1) for quick downstream test",
    )
    parser.add_argument(
        "--edge_sample_ratio",
        type=float,
        default=1.0,
        help="Subsample ratio for edges per split (0<r<=1) for quick downstream test",
    )
    parser.add_argument(
        "--downstream_n_runs",
        type=int,
        default=1,
        help="Number of runs for downstream evaluation",
    )
    parser.add_argument(
        "--downstream_hidden_dim",
        type=int,
        default=128,
        help="Hidden dim for downstream MLP",
    )
    parser.add_argument(
        "--downstream_num_layers",
        type=int,
        default=2,
        help="Layers for downstream MLP",
    )
    parser.add_argument(
        "--downstream_dropout",
        type=float,
        default=0.5,
        help="Dropout for downstream MLP",
    )
    parser.add_argument(
        "--downstream_batch_size",
        type=int,
        default=1024,
        help="Batch size for downstream classifiers",
    )
    parser.add_argument(
        "--downstream_lr",
        type=float,
        default=0.001,
        help="LR for downstream classifiers",
    )
    parser.add_argument(
        "--downstream_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for downstream classifiers",
    )
    parser.add_argument(
        "--downstream_epochs",
        type=int,
        default=100,
        help="Epochs for downstream classifiers",
    )
    parser.add_argument(
        "--downstream_patience",
        type=int,
        default=10,
        help="Early stopping patience for downstream classifiers",
    )
    parser.add_argument(
        "--downstream_neg_samples",
        type=int,
        default=1,
        help="Negatives per positive edge for downstream link prediction",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    emb_path = Path(args.embeddings_path)
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    print(f"Loading embeddings from {emb_path}")
    emb_data = torch.load(emb_path, map_location=device)
    train_embeddings = emb_data["train_embeddings"].to(device)
    train_labels = emb_data["train_labels"].to(device)
    val_embeddings = emb_data["val_embeddings"].to(device)
    val_labels = emb_data["val_labels"].to(device)
    test_embeddings = emb_data["test_embeddings"].to(device)
    test_labels = emb_data["test_labels"].to(device)

    def maybe_subsample_nodes(emb, labels, ratio):
        if ratio >= 1.0:
            return emb, labels
        n = emb.size(0)
        k = max(1, int(n * ratio))
        perm = torch.randperm(n, device=emb.device)[:k]
        return emb[perm], labels[perm]

    # Reload graph to build edge splits for downstream link prediction
    print("Reloading graph for edge splits...")
    data = load_ogb_mag(root_path=args.data_root, preprocess=args.preprocess)
    target_edge_type = tuple(args.target_edge_type.split(","))
    edge_index = data[target_edge_type].edge_index
    train_edge_index, val_edge_index, test_edge_index = create_edge_splits(
        edge_index=edge_index, seed=args.seed
    )

    def maybe_subsample_edges(edge_idx, ratio):
        if ratio >= 1.0:
            return edge_idx
        m = edge_idx.size(1)
        k = max(1, int(m * ratio))
        perm = torch.randperm(m, device=edge_idx.device)[:k]
        return edge_idx[:, perm]

    # Node downstream
    if args.downstream_task in {"node", "both"}:
        print("\n========== Downstream: Node Property Prediction ==========")
        num_classes = int(data[args.target_node].y.max().item() + 1)
        tr_emb, tr_lbl = maybe_subsample_nodes(train_embeddings, train_labels, args.node_sample_ratio)
        va_emb, va_lbl = maybe_subsample_nodes(val_embeddings, val_labels, args.node_sample_ratio)
        te_emb, te_lbl = maybe_subsample_nodes(test_embeddings, test_labels, args.node_sample_ratio)
        node_results = evaluate_node_property_prediction(
            train_embeddings=tr_emb,
            train_labels=tr_lbl,
            val_embeddings=va_emb,
            val_labels=va_lbl,
            test_embeddings=te_emb,
            test_labels=te_lbl,
            num_classes=num_classes,
            device=device,
            n_runs=args.downstream_n_runs,
            hidden_dim=args.downstream_hidden_dim,
            num_layers=args.downstream_num_layers,
            dropout=args.downstream_dropout,
            batch_size=args.downstream_batch_size,
            lr=args.downstream_lr,
            weight_decay=args.downstream_weight_decay,
            num_epochs=args.downstream_epochs,
            early_stopping_patience=args.downstream_patience,
            verbose=True,
        )
        for k, v in node_results.items():
            print(f"{k}: {v}")

    # Link downstream
    if args.downstream_task in {"link", "both"}:
        print("\n========== Downstream: Link Prediction ==========")
        tr_edge = maybe_subsample_edges(train_edge_index, args.edge_sample_ratio)
        va_edge = maybe_subsample_edges(val_edge_index, args.edge_sample_ratio)
        te_edge = maybe_subsample_edges(test_edge_index, args.edge_sample_ratio)
        link_results = evaluate_link_prediction(
            train_embeddings=train_embeddings,
            train_edge_index=tr_edge.to(device),
            val_embeddings=val_embeddings,
            val_edge_index=va_edge.to(device),
            test_embeddings=test_embeddings,
            test_edge_index=te_edge.to(device),
            device=device,
            n_runs=args.downstream_n_runs,
            num_neg_samples=args.downstream_neg_samples,
            hidden_dim=args.downstream_hidden_dim,
            num_layers=args.downstream_num_layers,
            dropout=args.downstream_dropout,
            batch_size=args.downstream_batch_size,
            lr=args.downstream_lr,
            weight_decay=args.downstream_weight_decay,
            num_epochs=args.downstream_epochs,
            early_stopping_patience=args.downstream_patience,
            verbose=True,
        )
        for k, v in link_results.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
