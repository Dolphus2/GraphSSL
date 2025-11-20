import torch
import torch.nn as nn


class LinkPredictorDecoder(nn.Module):
    """
    这就是你要的“解码器”。
    输入：直接读取 embeddings.pt，不需要图结构。
    """

    def __init__(self, input_dim):
        super().__init__()
        # 你可以用简单的点积，也可以用 MLP 使得预测更强
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # 输入是两个节点向量的拼接
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出是存在边的概率 logits
        )

    def forward(self, src_embedding, dst_embedding):
        # 拼接两个节点的向量
        x = torch.cat([src_embedding, dst_embedding], dim=-1)
        return self.mlp(x)


# 使用示例
def predict_links():
    # 1. 加载 embeddings.pt
    emb_dict = torch.load("results/embeddings.pt")
    paper_embs = emb_dict['val_embeddings']  # 假设我们要预测 val 集的边

    # 2. 模拟一些查询 (比如：Paper A 和 Paper B 有引用关系吗？)
    # 实际场景中，这些索引来自你的测试集
    src_idx = torch.tensor([0, 1, 2])
    dst_idx = torch.tensor([5, 6, 7])

    src_vecs = paper_embs[src_idx]
    dst_vecs = paper_embs[dst_idx]

    # 3. 解码
    decoder = LinkPredictorDecoder(input_dim=128)
    scores = decoder(src_vecs, dst_vecs)
    probs = torch.sigmoid(scores)

    print("连接概率:", probs)