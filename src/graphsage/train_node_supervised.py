import argparse, torch, torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from models.hetero_sage import build_hetero_sage
from utils import resolve_device, set_seed, masked_accuracy
from data_mag import load_mag_hetero


def make_loaders(data, train_input_nodes, val_input_nodes, test_input_nodes,
                    num_neighbors=(15, 10), **kwargs):
    train_loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors), shuffle=True,
        input_nodes=train_input_nodes, **kwargs)
    val_loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors),
        input_nodes=val_input_nodes, **kwargs)
    test_loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors),
        input_nodes=test_input_nodes, **kwargs)

    return train_loader, val_loader, test_loader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_idx = []
    for batch in loader:
        batch = batch.to(device)
        out_dict = model(batch.x_dict, batch.edge_index_dict)
        # first K entries in 'paper' correspond to seeds in this batch
        k = batch["paper"].batch_size
        logits = out_dict["paper"][:k]
        # map local seed nodes to global node indices:
        seed_nid = batch["paper"].n_id[:k]
        all_logits.append(logits.cpu())
        all_idx.append(seed_nid.cpu())
    return torch.cat(all_idx, dim=0), torch.cat(all_logits, dim=0)

def main():    
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="../../data/ogb")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--agg", type=str, default="mean", choices=["mean","sum","max"])
    p.add_argument("--aggr_rel", type=str, default="sum", choices=["sum","mean","max"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--neighbors", type=str, default="15,10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--persist_workers", type=bool, default=True)

    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device()

    dataset, y, train_input_nodes, val_input_nodes, test_input_nodes = load_mag_hetero(root=args.root)
    data = dataset[0]
    num_classes = dataset.num_classes

    model = build_hetero_sage(
        metadata=data.metadata(), hidden=args.hidden, out_dim=num_classes,
        layers=args.layers, dropout=args.dropout, aggr=args.agg, aggr_rel=args.aggr_rel
    ).to(device)

    # loaders
    num_neighbors = tuple(int(x) for x in args.neighbors.split(","))
    kwargs = {'batch_size': args.batch_size, 'num_workers':args.num_workers, 'persistent_workers':args.persist_workers}
    train_loader, val_loader, test_loader = make_loaders(
        data, train_input_nodes, val_input_nodes, test_input_nodes, num_neighbors=num_neighbors, **kwargs
    )

    @torch.no_grad()
    def init_params():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(device)
        model(batch.x_dict, batch.edge_index_dict)

    init_params()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            out_dict = model(batch.x_dict, batch.edge_index_dict)
            k = batch["paper"].batch_size
            logits = out_dict["paper"][:k]
            y_seed = batch["paper"].y[:k]
            loss = F.cross_entropy(logits, y_seed)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total_loss += loss.item()

        # Eval
        val_idx_all, val_logits = evaluate(model, val_loader, device)
        #test_idx_all, test_logits = evaluate(model, test_loader, device)
        val_acc  = masked_accuracy(val_logits, y[val_idx_all], torch.arange(val_idx_all.numel()))
        #test_acc = masked_accuracy(test_logits, y[test_idx_all], torch.arange(test_idx_all.numel()))

        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d} | loss {total_loss:.3f} | val {val_acc:.4f} ")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        test_idx_all, test_logits = evaluate(model, test_loader, device)
        final_test = masked_accuracy(test_logits, y[test_idx_all], torch.arange(test_idx_all.numel()))
        print(f"[Best on val]  Final test acc: {final_test:.4f}")

if __name__ == "__main__":
    main()