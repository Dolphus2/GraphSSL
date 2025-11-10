import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero


class HomoSAGE(nn.Module):
    """Homogeneous GraphSAGE backbone (to be replicated per relation by to_hetero)."""
    def __init__(self, hidden: int, out_dim: int, layers: int = 2,
                    dropout: float = 0.5, aggr: str = "mean", bn: bool = False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        if layers == 1:
            self.convs.append(SAGEConv((-1, -1), out_dim, aggr=aggr))
        else:
            self.convs.append(SAGEConv((-1, -1), hidden, aggr=aggr))
            for _ in range(layers - 2):
                self.convs.append(SAGEConv((-1, -1), hidden, aggr=aggr))
            self.convs.append(SAGEConv((-1, -1), out_dim, aggr=aggr))
            if bn:
                for _ in range(layers - 1):
                    self.bns.append(nn.BatchNorm1d(hidden))
        self.use_bn = bn

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            if self.use_bn:
                h = self.bns[i](h)
            h = torch.relu(h)
            h = torch.dropout(h, p=self.dropout, train=self.training)
        h = self.convs[-1](h, edge_index)
        return h

def build_hetero_sage(metadata, hidden: int, out_dim: int, layers: int = 2,
                        dropout: float = 0.5, aggr: str = "mean",
                        bn: bool = False, aggr_rel: str = "sum"):
    """Wrap HomoSAGE into a heterogeneous operator via to_hetero."""
    backbone = HomoSAGE(hidden=hidden, out_dim=out_dim, layers=layers,
                        dropout=dropout, aggr=aggr, bn=bn)
    model = to_hetero(backbone, metadata=metadata, aggr=aggr_rel)
    return model

if __name__ == "__main__":
    pass