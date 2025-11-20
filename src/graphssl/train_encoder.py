"""
Stage 1: Train Encoder using Self-Supervised Link Prediction
Output: results/embeddings.pt
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from pathlib import Path
import argparse

# 引用你现有的工具库
from utils.data_utils import load_ogb_mag
from utils.models import create_model



# ---------------------------------------------------------
#  checkpoint tools: save + load (resume training)
# ---------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, best_loss, path="results/checkpoint.pth"):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss
    }
    torch.save(checkpoint, path)
    print(f"[Checkpoint Saved] Epoch {epoch}, Best Loss {best_loss:.4f}")


def load_checkpoint(model, optimizer, path="results/checkpoint.pth"):
    if not Path(path).exists():
        print("[No checkpoint found, start fresh training]")
        return model, optimizer, 0, float("inf")

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]

    print(f"[Checkpoint Loaded] Resume from epoch {start_epoch}, best loss={best_loss:.4f}")
    return model, optimizer, start_epoch, best_loss



# ---------------------------------------------------------
# Data Loader: Link Prediction Sampling
# ---------------------------------------------------------

def create_link_loader(data, target_edge, split_edge_index, batch_size, neg_ratio=1.0, shuffle=True):
    return LinkNeighborLoader(
        data,
        num_neighbors=[10, 5],
        edge_label_index=(target_edge, split_edge_index),
        edge_label=torch.ones(split_edge_index.size(1)),
        neg_sampling_ratio=neg_ratio,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )



# ---------------------------------------------------------
#  One Training Epoch
# ---------------------------------------------------------

def train_epoch(model, loader, optimizer, device, target_edge):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training Encoder"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # 获取节点嵌入（注意：使用 model.model 才是 encoder）
        _, embeddings_dict = model.model(batch.x_dict, batch.edge_index_dict)

        src_type, _, dst_type = target_edge

        edge_label_index = batch[target_edge].edge_label_index
        edge_label = batch[target_edge].edge_label

        h_src = embeddings_dict[src_type][edge_label_index[0]]
        h_dst = embeddings_dict[dst_type][edge_label_index[1]]

        pred = (h_src * h_dst).sum(dim=-1)

        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)



# ---------------------------------------------------------
#  Extract All Embeddings from Full Graph
# ---------------------------------------------------------


@torch.no_grad()
def extract_full_embeddings(model, data, device, batch_size=2048):
    model.eval()
    from torch_geometric.loader import NeighborLoader

    embeddings_dict = {node_type: [] for node_type in data.node_types}

    for node_type in data.node_types:
        loader = NeighborLoader(
            data,
            num_neighbors=[15, 10],
            input_nodes=node_type,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True
        )

        emb_list = []
        for batch in tqdm(loader, desc=f"Extracting {node_type}"):

            batch = batch.to(device)
            _, batch_embs = model.model(batch.x_dict, batch.edge_index_dict)

            # 提取种子节点的 embedding（种子节点是 batch 中的前 batch_size 个节点）
            # 使用本地索引而非原始 node ID
            num_seeds = batch[node_type].batch_size
            emb = batch_embs[node_type][:num_seeds]
            emb_list.append(emb.cpu())

        embeddings_dict[node_type] = torch.cat(emb_list, dim=0)

    return embeddings_dict



# ---------------------------------------------------------
#  Main (with resume + best model)
# ---------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "data"
    target_edge = ("author", "writes", "paper")

    # 1. 加载数据
    data = load_ogb_mag(data_path, preprocess="metapath2vec")

    # 2. 创建数据加载器
    train_edge_index = data[target_edge].edge_index
    train_loader = create_link_loader(data, target_edge, train_edge_index, batch_size=1024)

    # 3. 创建模型 & 优化器
    model = create_model(data, hidden_channels=128, target_node_type="paper").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # 创建结果目录
    Path("results").mkdir(exist_ok=True)

    # 4. 加载 checkpoint（如果有就断点续训）
    model, optimizer, start_epoch, best_loss = load_checkpoint(
        model, optimizer, "results/checkpoint.pth"
    )

    # 5. 开始训练
    max_epochs = 20
    print("Start / Resume Encoder Pre-training...")

    for epoch in range(start_epoch, max_epochs):
        loss = train_epoch(model, train_loader, optimizer, device, target_edge)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # ---- 保存 checkpoint ----
        save_checkpoint(model, optimizer, epoch+1, best_loss, "results/checkpoint.pth")

        # ---- 保存最好模型 ----
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "results/best_encoder_link.pt")
            print(f"[Best Model Updated] Loss improved to {best_loss:.4f}")

    # 6. 训练完后提取全图 Embeddings
    print("Extracting and Saving Embeddings...")
    full_embeddings = extract_full_embeddings(model, data, device)
    torch.save(full_embeddings, "results/embeddings_link.pt")

    print("Done! Embeddings saved.")



if __name__ == "__main__":
    main()
