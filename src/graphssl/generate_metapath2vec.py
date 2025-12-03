import argparse
from pathlib import Path
import time
import torch
import torch_geometric.transforms as T
from torch_sparse import transpose  # still fine to use in latest PyG
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn.models import MetaPath2Vec  # explicit models import
import wandb


@torch.no_grad()
def save_embedding(model, node_type: str = "paper", out_path: str = "embedding.pt"):
    """
    Save embeddings for a given node type.

    Args:
        model: Trained MetaPath2Vec model.
        node_type: Node type to extract embeddings for (e.g. 'paper').
        out_path: Path to save the tensor.
    """
    emb = model(node_type).cpu()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, out_path)
    print(f"Saved {node_type} embeddings to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="OGBN-MAG (MetaPath2Vec)")
    parser.add_argument("--root", type=str, default="data",
                        help="Root directory for OGB_MAG.")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU id, ignored if no CUDA.")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=64)
    parser.add_argument("--context_size", type=int, default=7)
    parser.add_argument("--walks_per_node", type=int, default=5)
    parser.add_argument("--num_negative_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save embeddings every N steps.")
    parser.add_argument("--out_dir", type=str, default="data/embeddings",
                        help="Directory to save embeddings.")
    parser.add_argument("--node_type", type=str, default="paper",
                        help="Node type to save embeddings for.")
    args = parser.parse_args()

    wandb.init(
        project="graphssl",
        name=f"metapath2vec_{int(time.time())}",
        config={
            "embedding_dim": args.embedding_dim,
            "walk_length": args.walk_length,
            "context_size": args.context_size,
            "walks_per_node": args.walks_per_node,
            "num_negative_samples": args.num_negative_samples,
            "epochs": args.epochs,
            "save_every": args.save_every,
            "node_type": args.node_type,
        }
    )

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load OGB_MAG dataset
    # ------------------------------------------------------------------
    transform = T.ToUndirected(merge=True)
    dataset = OGB_MAG(root=args.root, preprocess=None, transform=transform)
    data = dataset[0]
    print(data)

    # ------------------------------------------------------------------
    # 2. Build Metapath
    # ------------------------------------------------------------------
    metapath = [
        ('paper', 'has_topic', 'field_of_study'),
        ('field_of_study', 'rev_has_topic', 'paper'),
        ('paper', 'rev_writes', 'author'),
        ('author', 'writes', 'paper'),
    ]

    # ------------------------------------------------------------------
    # 2. Build MetaPath2Vec model
    # ------------------------------------------------------------------
    model = MetaPath2Vec(
        edge_index_dict=data.edge_index_dict,
        embedding_dim=args.embedding_dim,
        metapath=metapath,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        num_negative_samples=args.num_negative_samples,
        sparse=True,
    ).to(device)

    # Random-walk DataLoader
    loader = model.loader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    optimizer = torch.optim.SparseAdam(
        list(model.parameters()), lr=args.lr
    )

    # Training loop
    model.train()
    global_step = 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0

        for i, (pos_rw, neg_rw) in enumerate(loader):
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)

            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            global_step += 1

            if (i + 1) % args.log_steps == 0:
                avg_loss = total_loss / (i + 1)
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Step: {i + 1:04d}/{len(loader)}, "
                    f"Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}"
                )
                wandb.log({
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "epoch": epoch,
                    "step": i + 1,
                    "num_steps": len(loader),
                })

            # Periodic checkpoint of embeddings
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_path = out_dir / f"{args.node_type}_emb_epoch{epoch}_step{global_step}.pt"
                save_embedding(model, node_type=args.node_type, out_path=str(save_path))

        # Save at end of each epoch as well
        epoch_save_path = out_dir / f"{args.node_type}_emb_epoch{epoch}.pt"
        save_embedding(model, node_type=args.node_type, out_path=str(epoch_save_path))

        print(f"Epoch {epoch:02d} finished, total_loss={total_loss:.4f}")

    # Final save
    final_path = out_dir / f"{args.node_type}_emb_final.pt"
    save_embedding(model, node_type=args.node_type, out_path=str(final_path))


if __name__ == "__main__":
    
    main()

