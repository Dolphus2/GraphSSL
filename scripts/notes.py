paper_data = data['paper']
if hasattr(paper_data, 'train_mask'):
    print(f"\nTrain/Val/Test splits:")
    print(f"  Train samples: {paper_data.train_mask.sum().item():,}")
    print(f"  Val samples: {paper_data.val_mask.sum().item():,}")
    print(f"  Test samples: {paper_data.test_mask.sum().item():,}")

paper_train_x = paper_data['x'][paper_data.train_mask]
paper_train_y = paper_data['y'][paper_data.train_mask]
paper_val_x = paper_data['x'][paper_data.val_mask]
paper_val_y = paper_data['y'][paper_data.val_mask]
paper_test_y = paper_data['y'][paper_data.test_mask]
paper_test_x = paper_data['x'][paper_data.test_mask]


class GraphSAGE_(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)
    
    def forward(self, data):
        x = self.conv1(data.x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return torch.log_softmax(x, dim=-1)
    

# Check which node types are missing features and add them (everything except paper nodes)
# for node_type in data_inductive.node_types:
#     if 'x' not in data_inductive[node_type]:
#         # Create dummy features (learnable embeddings will be used)
#         num_nodes = data_inductive[node_type].num_nodes
#         # Initialize with zeros or random features
#         data_inductive[node_type].x = torch.zeros((num_nodes, 1))
#         print(f"Added dummy features for '{node_type}' nodes: shape {data_inductive[node_type].x.shape}")

# Add learnable embeddings for node types without features
# embedding_dim = 128
# for node_type in data_inductive.node_types:
#     if 'x' not in data_inductive[node_type]:
#         num_nodes = data_inductive[node_type].num_nodes
#         # Use learnable embeddings instead of zeros
#         data_inductive[node_type].x = torch.nn.Embedding(num_nodes, embedding_dim).weight.data 
#         print(f"Added learnable embedding for '{node_type}' nodes: shape {data_inductive[node_type].x.shape}")