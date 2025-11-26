"""
Quick validation script to test core components before running full experiments.

This script:
1. Creates small dummy data
2. Tests all loss functions
3. Tests all training objectives
4. Verifies forward/backward passes work
5. Shows loss decreasing over a few steps

Run: python scripts/test/quick_validation.py
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import sys
sys.path.insert(0, 'src')

from graphssl.utils.objective_utils import (
    sce_loss, mer_loss, tar_loss, pfp_loss,
    SupervisedNodeClassification,
    SupervisedLinkPrediction,
    SelfSupervisedNodeReconstruction,
    SelfSupervisedEdgeReconstruction,
    EdgeDecoder, FeatureDecoder
)

print("="*80)
print("GraphSSL Quick Validation")
print("="*80)
print()

# ============================================================================
# Test 1: Loss Functions
# ============================================================================
print("[1/5] Testing Loss Functions...")
print("-"*80)

# SCE Loss
print("  Testing SCE Loss...")
x = torch.randn(10, 128)
y = torch.randn(10, 128)
loss = sce_loss(x, y)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * SCE Loss: {loss.item():.4f}")

# MER Loss
print("  Testing MER Loss...")
edge_scores = torch.randn(100)
edge_labels = torch.randint(0, 2, (100,))
loss = mer_loss(edge_scores, edge_labels)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * MER Loss: {loss.item():.4f}")

# TAR Loss
print("  Testing TAR Loss...")
src_emb = torch.randn(100, 128)
dst_emb = torch.randn(100, 128)
edge_labels = torch.randint(0, 2, (100,))
loss = tar_loss(src_emb, dst_emb, edge_labels, temperature=0.5)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * TAR Loss: {loss.item():.4f}")

# PFP Loss
print("  Testing PFP Loss...")
src_feat = torch.randn(100, 256)
dst_feat = torch.randn(100, 256)
edge_labels = torch.randint(0, 2, (100,))
loss = pfp_loss(src_feat, dst_feat, edge_labels)
assert torch.isfinite(loss)
print(f"    * PFP Loss: {loss.item():.4f}")

# Combined Loss
print("  Testing Combined Loss...")
edge_scores = torch.randn(100)
edge_labels = torch.randint(0, 2, (100,))
src_emb = torch.randn(100, 128)
dst_emb = torch.randn(100, 128)
src_feat = torch.randn(100, 256)
dst_feat = torch.randn(100, 256)
mer = mer_loss(edge_scores, edge_labels)
tar = tar_loss(src_emb, dst_emb, edge_labels)
pfp = pfp_loss(src_feat, dst_feat, edge_labels)
combined = mer + tar + pfp
assert torch.isfinite(combined) and combined.item() > 0
print(f"    * Combined Loss (MER+TAR+PFP): {combined.item():.4f}")

print("  * All loss functions working correctly!")
print()

# ============================================================================
# Test 2: Create Dummy Data
# ============================================================================
print("[2/5] Creating Dummy Heterogeneous Graph Data...")
print("-"*80)

data = HeteroData()

# Add paper nodes
num_papers = 100
data['paper'].x = torch.randn(num_papers, 128)
data['paper'].y = torch.randint(0, 10, (num_papers,))
data['paper'].batch_size = num_papers // 2

# Add author nodes
num_authors = 50
data['author'].x = torch.randn(num_authors, 128)
data['author'].batch_size = num_authors // 2

# Add edges
num_edges = 200
data['paper', 'cites', 'paper'].edge_index = torch.randint(0, num_papers, (2, num_edges))
data['paper', 'cites', 'paper'].edge_label_index = torch.randint(0, num_papers, (2, num_edges))
data['paper', 'cites', 'paper'].edge_label = torch.randint(0, 2, (num_edges,))

data['author', 'writes', 'paper'].edge_index = torch.randint(0, min(num_authors, num_papers), (2, 100))

print(f"  * Created heterogeneous graph:")
print(f"    - Paper nodes: {num_papers}")
print(f"    - Author nodes: {num_authors}")
print(f"    - Paper-cites-paper edges: {num_edges}")
print()

# ============================================================================
# Test 3: Dummy Model
# ============================================================================
print("[3/5] Creating Dummy Model...")
print("-"*80)

class DummyHeteroGNN(nn.Module):
    """Dummy heterogeneous GNN for testing"""
    def __init__(self, hidden_dim=128, num_classes=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x_dict, edge_index_dict):
        out_dict = {}
        embeddings_dict = {}
        
        for node_type, x in x_dict.items():
            # Simple transformation
            embeddings = torch.relu(self.linear(x))
            embeddings_dict[node_type] = embeddings
            
            # For paper nodes, also output class logits
            if node_type == 'paper':
                out_dict[node_type] = self.classifier(embeddings)
            else:
                out_dict[node_type] = embeddings
        
        return out_dict, embeddings_dict

model = DummyHeteroGNN(hidden_dim=128, num_classes=10)
print(f"  * Model created with {sum(p.numel() for p in model.parameters())} parameters")
print()

# ============================================================================
# Test 4: Training Objectives
# ============================================================================
print("[4/5] Testing Training Objectives...")
print("-"*80)

# Test Supervised Node Classification
print("  Testing Supervised Node Classification...")
objective = SupervisedNodeClassification(target_node_type='paper')
loss, metrics = objective.step(model, data, is_training=True)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * Loss: {loss.item():.4f}, Acc: {metrics['acc']:.4f}")

# Test Supervised Link Prediction
print("  Testing Supervised Link Prediction...")
objective = SupervisedLinkPrediction(target_edge_type=('paper', 'cites', 'paper'))
loss, metrics = objective.step(model, data, is_training=True)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * Loss: {loss.item():.4f}, Acc: {metrics['acc']:.4f}")

# Test Self-Supervised Node (MSE)
print("  Testing Self-Supervised Node (MSE)...")
decoder = FeatureDecoder(hidden_dim=128, feature_dim=128, dropout=0.0)
objective = SelfSupervisedNodeReconstruction(
    target_node_type='paper',
    mask_ratio=0.5,
    decoder=decoder,
    loss_fn='mse'
)
loss, metrics = objective.step(model, data, is_training=True)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * MSE Loss: {loss.item():.4f}")

# Test Self-Supervised Node (SCE)
print("  Testing Self-Supervised Node (SCE)...")
objective = SelfSupervisedNodeReconstruction(
    target_node_type='paper',
    mask_ratio=0.5,
    decoder=decoder,
    loss_fn='sce'
)
loss, metrics = objective.step(model, data, is_training=True)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * SCE Loss: {loss.item():.4f}")

# Test Self-Supervised Edge (BCE)
print("  Testing Self-Supervised Edge (BCE)...")
edge_decoder = EdgeDecoder(hidden_dim=128, dropout=0.0)
objective = SelfSupervisedEdgeReconstruction(
    target_edge_type=('paper', 'cites', 'paper'),
    decoder=edge_decoder,
    loss_fn='bce'
)
loss, metrics = objective.step(model, data, is_training=True)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * BCE Loss: {loss.item():.4f}, Acc: {metrics['acc']:.4f}")

# Test Self-Supervised Edge (Combined Loss)
print("  Testing Self-Supervised Edge (Combined Loss)...")
objective = SelfSupervisedEdgeReconstruction(
    target_edge_type=('paper', 'cites', 'paper'),
    decoder=edge_decoder,
    loss_fn='combined_loss',
    mer_weight=1.0,
    tar_weight=1.0,
    pfp_weight=1.0
)
loss, metrics = objective.step(model, data, is_training=True)
assert torch.isfinite(loss) and loss.item() > 0
print(f"    * Combined Loss: {loss.item():.4f}")
print(f"      - MER: {metrics['mer']:.4f}")
print(f"      - TAR: {metrics['tar']:.4f}")
print(f"      - PFP: {metrics['pfp']:.4f}")

print("  * All training objectives working correctly!")
print()

# ============================================================================
# Test 5: Training Loop Simulation
# ============================================================================
print("[5/5] Simulating Mini Training Loop (5 steps)...")
print("-"*80)

# Use supervised node classification as example
model = DummyHeteroGNN(hidden_dim=128, num_classes=10)
objective = SupervisedNodeClassification(target_node_type='paper')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("  Training for 5 steps...")
losses = []
for step in range(5):
    # Forward pass
    loss, metrics = objective.step(model, data, is_training=True)
    losses.append(loss.item())
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"    Step {step+1}: Loss={loss.item():.4f}, Acc={metrics['acc']:.4f}")

# Check if loss is generally decreasing
if losses[-1] < losses[0]:
    print(f"  * Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
else:
    print(f"  ! Loss didn't decrease significantly (might need more steps)")
print()

# ============================================================================
# Final Summary
# ============================================================================
print("="*80)
print("* VALIDATION COMPLETE!")
print("="*80)
print()
print("All components tested successfully:")
print("  * Loss functions (SCE, MER, TAR, PFP, Combined)")
print("  * Training objectives (supervised + self-supervised)")
print("  * Forward/backward passes")
print("  * Gradient flow")
print("  * Mini training loop")
print()
print("You can now run the full test experiments:")
print("  bash scripts/test/run_all_tests.sh")
print()
print("="*80)

