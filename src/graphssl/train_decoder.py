"""
Stage 2: Train Decoder using Pre-trained Embeddings
Input: results/embeddings_link.pt
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------
# Checkpoint 工具函数
# ---------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, best_loss, best_auc, path="results/decoder_checkpoint.pth"):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "best_auc": best_auc
    }
    torch.save(checkpoint, path)
    print(f"[Checkpoint Saved] Epoch {epoch}, Best Loss: {best_loss:.4f}, Best AUC: {best_auc:.4f}")


def load_checkpoint(model, optimizer, path="results/decoder_checkpoint.pth"):
    if not Path(path).exists():
        print("[No checkpoint found, starting fresh training]")
        return model, optimizer, 0, float("inf"), 0.0

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    best_auc = checkpoint.get("best_auc", 0.0)  # 兼容旧版checkpoint

    print(f"[Checkpoint Loaded] Resume from epoch {start_epoch}, best loss={best_loss:.4f}, best AUC={best_auc:.4f}")
    return model, optimizer, start_epoch, best_loss, best_auc


# 定义解码器模型 (MLP)
class LinkDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # 输入维度是 2 * embedding_dim (因为拼接了源节点和目标节点)
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)  # 输出一个 Logit 分数
        )

    def forward(self, src_emb, dst_emb):
        x = torch.cat([src_emb, dst_emb], dim=-1)
        return self.net(x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载预训练的 Embeddings
    print("Loading Embeddings...")
    emb_dict = torch.load("results/embeddings_link.pt")

    # 假设我们要预测 ('author', 'writes', 'paper')
    author_emb = emb_dict['author'].to(device)  # [Num_Authors, 128]
    paper_emb = emb_dict['paper'].to(device)  # [Num_Papers, 128]

    # 2. 准备训练数据 (正样本和负样本)
    # 注意：这里为了演示，我们需要加载真实的边索引。
    # 在实际应用中，你应该从 data 对象或磁盘加载之前划分好的 train/val 边索引
    from utils.data_utils import load_ogb_mag
    data = load_ogb_mag("data", preprocess="metapath2vec")
    edge_index = data[("author", "writes", "paper")].edge_index

    # 使用全部边作为正样本
    num_edges = edge_index.size(1)
    print(f"Total edges (positive samples): {num_edges}")
    src_idx = edge_index[0, :]  # 全部源节点
    dst_idx = edge_index[1, :]  # 全部目标节点
    labels = torch.ones(num_edges)  # 标签 1

    # 构建负样本 (随机采样，数量与正样本相同)
    print(f"Generating {num_edges} negative samples...")
    neg_src = torch.randint(0, author_emb.size(0), (num_edges,))
    neg_dst = torch.randint(0, paper_emb.size(0), (num_edges,))
    neg_labels = torch.zeros(num_edges)  # 标签 0

    # 合并数据
    train_src = torch.cat([src_idx, neg_src])
    train_dst = torch.cat([dst_idx, neg_dst])
    train_y = torch.cat([labels, neg_labels])
    
    print(f"Total training samples: {len(train_y)} (正样本: {num_edges}, 负样本: {num_edges})")

    # 创建 PyTorch DataLoader (优化性能)
    dataset = TensorDataset(train_src, train_dst, train_y)
    loader = DataLoader(
        dataset, 
        batch_size=8192,  # 增大batch size，充分利用L40S GPU
        shuffle=True,
        num_workers=32,     # 多线程加载数据
        pin_memory=True,   # 加速CPU到GPU的传输
        persistent_workers=True  # 保持worker进程
    )

    # 3. 初始化解码器
    input_dim = author_emb.size(1)  # 128
    decoder = LinkDecoder(input_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # 创建结果目录
    Path("results").mkdir(exist_ok=True)

    # 加载checkpoint（如果存在）
    decoder, optimizer, start_epoch, best_loss, best_auc = load_checkpoint(
        decoder, optimizer, "results/decoder_checkpoint.pth"
    )

    # 4. 训练解码器
    print("Start Training Decoder...")
    print(f"Using device: {device}")
    print(f"Batch size: 8192")
    print(f"Total batches per epoch: {len(loader)}")
    
    decoder.train()
    max_epochs = 100
    for epoch in range(start_epoch, max_epochs):
        total_loss = 0
        
        # 只在某些epoch计算AUC，节省时间
        compute_auc = (epoch % 10 == 0 or epoch == 999)
        if compute_auc:
            all_preds = []
            all_labels = []

        for batch_src_idx, batch_dst_idx, batch_y in loader:
            # ⚡ 关键优化：将索引也移到GPU，避免CPU-GPU传输瓶颈
            batch_src_idx = batch_src_idx.to(device)
            batch_dst_idx = batch_dst_idx.to(device)
            batch_y = batch_y.to(device)
            
            # 核心：直接查表获取向量，不需要图卷积
            batch_src_emb = author_emb[batch_src_idx]
            batch_dst_emb = paper_emb[batch_dst_idx]

            optimizer.zero_grad()
            logits = decoder(batch_src_emb, batch_dst_emb).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 只在需要时记录AUC
            if compute_auc:
                all_preds.append(torch.sigmoid(logits).detach().cpu())
                all_labels.append(batch_y.cpu())

        # 计算指标
        avg_loss = total_loss / len(loader)
        if compute_auc:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            auc = roc_auc_score(all_labels, all_preds)
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | AUC: {auc:.4f}")
            
            # 保存最好的模型（基于AUC）
            if auc > best_auc:
                best_auc = auc
                best_loss = avg_loss
                torch.save(decoder.state_dict(), "results/best_decoder.pt")
                print(f"[Best Model Updated] AUC improved to {best_auc:.4f}")
        else:
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")
            
            # 如果不计算AUC，基于loss保存
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(decoder.state_dict(), "results/best_decoder.pt")
                print(f"[Best Model Updated] Loss improved to {best_loss:.4f}")

        # 每个epoch都保存checkpoint
        save_checkpoint(decoder, optimizer, epoch + 1, best_loss, best_auc, "results/decoder_checkpoint.pth")

    # 5. 训练完成
    print("\nTraining completed!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best AUC: {best_auc:.4f}")
    print("Best model saved to: results/best_decoder.pt")


if __name__ == "__main__":
    main()