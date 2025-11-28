# GraphSSL Experiments - Formal Training on GPU

Complete guide for running experiments on DTU HPC with GPU acceleration.

## Overview

This guide describes 4 experiments comparing supervised and self-supervised learning methods on the OGB-MAG dataset using GraphSAGE backbone.

**Research Questions:**
1. How do different training objectives affect learned representations?
2. Are representations transferable across tasks?
3. Do self-supervised methods match supervised performance?

## Experimental Design

### The 4 Experiments

| Exp | Training Method | Main Task | Outputs |
|-----|----------------|-----------|---------|
| 1 | Supervised (node labels) | Node classification | Accuracy 1, Accuracy 3 |
| 2 | Supervised (edge labels) | Link prediction | Accuracy 2, Accuracy 4 |
| 5 | Self-supervised (GraphMAE) | Feature reconstruction | Accuracy 5, Accuracy 6 |
| 8 | Self-supervised (Combined loss) | Edge reconstruction | Accuracy 7, Accuracy 8 |

### Accuracy Measurements

- **Accuracy 1**: Exp 1 test accuracy on node classification (main task)
- **Accuracy 2**: Exp 2 test accuracy on link prediction (main task)
- **Accuracy 3**: Exp 1 downstream accuracy on link prediction (transfer test)
- **Accuracy 4**: Exp 2 downstream accuracy on node classification (transfer test)
- **Accuracy 5**: Exp 5 downstream accuracy on node classification
- **Accuracy 6**: Exp 5 downstream accuracy on link prediction
- **Accuracy 7**: Exp 8 downstream accuracy on node classification
- **Accuracy 8**: Exp 8 downstream accuracy on link prediction

### Experimental Flow

```
Experiment 1 (Supervised Node):
  Input: Graph + Node Labels
  Training: Cross-Entropy Loss
  Output: Encoder 1
  Evaluation: 
    - Main: Node Classification (Accuracy 1)
    - Downstream: Link Prediction (Accuracy 3)

Experiment 2 (Supervised Link):
  Input: Graph + Edge Labels
  Training: Binary Cross-Entropy Loss
  Output: Encoder 2
  Evaluation:
    - Main: Link Prediction (Accuracy 2)
    - Downstream: Node Classification (Accuracy 4)

Experiment 5 (GraphMAE):
  Input: Graph with Masked Features
  Training: SCE Loss (Feature Reconstruction)
  Output: Encoder 4
  Evaluation:
    - Downstream: Node Classification (Accuracy 5)
    - Downstream: Link Prediction (Accuracy 6)

Experiment 8 (Combined Loss):
  Input: Graph with Masked Edges
  Training: MER + TAR + PFP Loss
  Output: Encoder 5
  Evaluation:
    - Downstream: Node Classification (Accuracy 7)
    - Downstream: Link Prediction (Accuracy 8)
```

## Running Experiments

### Submit All Experiments

```bash
bash scripts/hpc/submit_all_experiments.sh
```

This submits all 4 experiments to run in parallel on separate GPUs. Each job will:
- Request 1x A100 GPU
- Allocate 32GB RAM
- Run for up to 24 hours
- Save results with timestamped directories

### Submit Individual Experiments

```bash
# Experiment 1: Supervised Node Classification
bsub < scripts/hpc/exp1_supervised_node.sh

# Experiment 2: Supervised Link Prediction
bsub < scripts/hpc/exp2_supervised_link.sh

# Experiment 5: GraphMAE
bsub < scripts/hpc/exp5_graphmae_node.sh

# Experiment 8: Combined Loss
bsub < scripts/hpc/exp8_combined_loss_edge.sh
```

### Monitor Jobs

```bash
# Check all jobs
bstat

# View specific job output
tail -f logs/exp1_supervised_node_JOBID.out

# View errors
tail -f logs/exp1_supervised_node_JOBID.err

# Cancel job
bkill JOBID
```

## Experiment Details

### Experiment 1: Supervised Node Classification

**Objective:** Train encoder to predict node labels (venues)

**Method:**
- Direct supervision with labeled nodes
- Multi-class classification (349 venues)
- Cross-entropy loss

**Architecture:**
```
GraphSAGE Encoder (2 layers, 128 dim)
  -> Linear Classifier (128 -> 349)
  -> Softmax
```

**Hyperparameters:**
```bash
--objective_type supervised_node_classification
--target_node paper
--hidden_channels 128
--num_layers 2
--batch_size 1024
--epochs 1000
--lr 0.001
--patience 100
```

**Outputs:**
- Encoder 1: Node embeddings from supervised learning
- Accuracy 1: Test accuracy on main task
- Accuracy 3: Downstream link prediction accuracy

**Expected Results:**
- Accuracy 1 should be high (supervised on same task)
- Accuracy 3 tests transfer ability to link prediction

---

### Experiment 2: Supervised Link Prediction

**Objective:** Train encoder to predict edge existence

**Method:**
- Direct supervision with positive/negative edge samples
- Binary classification (edge exists or not)
- Binary cross-entropy loss
- 1:1 negative sampling ratio

**Architecture:**
```
GraphSAGE Encoder (2 layers, 128 dim)
  -> Edge Embeddings (concatenate node pairs)
  -> MLP Decoder or Dot Product
  -> Sigmoid
```

**Hyperparameters:**
```bash
--objective_type supervised_link_prediction
--target_edge_type paper,cites,paper
--neg_sampling_ratio 1.0
--hidden_channels 128
--num_layers 2
--batch_size 1024
--epochs 1000
--lr 0.001
--patience 100
```

**Outputs:**
- Encoder 2: Node embeddings from supervised link learning
- Accuracy 2: Test accuracy on main task
- Accuracy 4: Downstream node classification accuracy

**Expected Results:**
- Accuracy 2 should be high (supervised on same task)
- Accuracy 4 tests transfer ability to node classification

---

### Experiment 5: Self-Supervised Node (GraphMAE)

**Objective:** Train encoder to reconstruct masked node features

**Method (GraphMAE):**
1. Randomly mask 50% of node features (set to zero)
2. Encode masked graph with GraphSAGE
3. Decode embeddings to reconstruct original features
4. Optimize with Scaled Cosine Error (SCE) loss

**SCE Loss:**
```
SCE(x, y) = mean((1 - cos_sim(x, y))^alpha)
where alpha = 3
```

Focuses on directional similarity rather than magnitude, providing better gradients for feature reconstruction.

**Architecture:**
```
Masked Node Features
  -> GraphSAGE Encoder (2 layers, 128 dim)
  -> MLP Decoder (128 -> 128 -> feature_dim)
  -> Reconstructed Features
  -> SCE Loss vs Original
```

**Hyperparameters:**
```bash
--objective_type self_supervised_node
--loss_fn sce
--mask_ratio 0.5
--use_feature_decoder
--hidden_channels 128
--num_layers 2
--batch_size 1024
--epochs 1000
--lr 0.001
--patience 100
```

**Outputs:**
- Encoder 4: Node embeddings from self-supervised learning
- Accuracy 5: Downstream node classification
- Accuracy 6: Downstream link prediction

**Expected Results:**
- No accuracy during training (unsupervised)
- Downstream tasks reveal representation quality
- Compare Acc 5 vs Acc 1 (self-supervised vs supervised)

---

### Experiment 8: Self-Supervised Edge (Combined Loss)

**Objective:** Train encoder to reconstruct edges with multi-component loss

**Method (Inspired by HGMAE):**

The combined loss consists of three components:

**1. MER (Masked Edge Reconstruction):**
- Standard binary cross-entropy for edge prediction
- Predicts which edges exist vs don't exist

**2. TAR (Topology-Aware Reconstruction):**
- Contrastive learning for topology preservation
- Maximizes embedding similarity for positive edges
- Minimizes embedding similarity for negative edges
- Temperature-scaled (tau = 0.5)

**3. PFP (Preference-based Feature Propagation):**
- Preserves feature similarity patterns
- Connected nodes should have similar features
- Complements topology-based learning

**Combined Loss:**
```
Total = mer_weight * MER + tar_weight * TAR + pfp_weight * PFP
      = 1.0 * MER + 1.0 * TAR + 1.0 * PFP
```

**Architecture:**
```
Node Features + Graph Structure
  -> GraphSAGE Encoder (2 layers, 128 dim)
  -> Node Embeddings (for TAR)
  -> MLP Edge Decoder
  -> Edge Scores (for MER)
  -> Combined Loss
```

**Hyperparameters:**
```bash
--objective_type self_supervised_edge
--loss_fn combined_loss
--mer_weight 1.0
--tar_weight 1.0
--pfp_weight 1.0
--tar_temperature 0.5
--use_edge_decoder
--neg_sampling_ratio 1.0
--hidden_channels 128
--num_layers 2
--batch_size 1024
--epochs 1000
--lr 0.001
--patience 100
```

**Tuning Loss Weights:**
- Increase mer_weight: Focus on edge prediction accuracy
- Increase tar_weight: Focus on topology preservation
- Increase pfp_weight: Focus on feature consistency
- Adjust tar_temperature: Lower for harder contrastive learning

**Outputs:**
- Encoder 5: Node embeddings from combined loss
- Accuracy 7: Downstream node classification
- Accuracy 8: Downstream link prediction

**Expected Results:**
- Three loss components logged separately during training
- Compare Acc 8 vs Acc 2 (self-supervised vs supervised)
- Compare Acc 7 vs Acc 1 (generalization to nodes)

---

## GPU Training Configuration

All experiments use:

```bash
# Resource allocation
GPU: 1x A100
Memory: 32GB
CPUs: 4
Time limit: 24 hours

# Model configuration
Hidden dimension: 128
Layers: 2
Dropout: 0.5
Batch size: 1024

# Training configuration
Epochs: 1000
Learning rate: 0.001
Weight decay: 0.0005
Early stopping patience: 100

# Downstream evaluation
Runs: 10
Hidden dim: 128
Epochs: 100
```

## Expected Timeline

- Job submission: Instant
- Queue time: 0-30 minutes (varies by cluster load)
- Training time: 12-24 hours per experiment
- Total (parallel): 12-24 hours for all 4 experiments

## Results Structure

```
results/
├── exp1_supervised_node_JOBID_TIMESTAMP/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── last_checkpoint.pt
│   ├── model_supervised_node_classification.pt  # Accuracy 1
│   ├── embeddings.pt                            # Encoder 1
│   ├── downstream_link_results.pt               # Accuracy 3
│   └── downstream_node_results.pt
├── exp2_supervised_link_JOBID_TIMESTAMP/
│   ├── model_supervised_link_prediction.pt      # Accuracy 2
│   ├── embeddings.pt                            # Encoder 2
│   ├── downstream_node_results.pt               # Accuracy 4
│   └── ...
├── exp5_graphmae_node_JOBID_TIMESTAMP/
│   ├── embeddings.pt                            # Encoder 4
│   ├── downstream_node_results.pt               # Accuracy 5
│   ├── downstream_link_results.pt               # Accuracy 6
│   └── ...
└── exp8_combined_loss_edge_JOBID_TIMESTAMP/
    ├── embeddings.pt                            # Encoder 5
    ├── downstream_node_results.pt               # Accuracy 7
    ├── downstream_link_results.pt               # Accuracy 8
    └── ...
```

## Analyzing Results

### Load Results

```python
import torch

# Load main training results
exp1 = torch.load('results/exp1_supervised_node_*/model_*.pt')
print(f"Accuracy 1: {exp1['test_metrics']['acc']:.4f}")

# Load downstream results
exp1_downstream = torch.load('results/exp1_supervised_node_*/downstream_link_results.pt')
print(f"Accuracy 3: {exp1_downstream['test_acc_mean']:.4f} ± {exp1_downstream['test_acc_std']:.4f}")

# Load embeddings
embeddings = torch.load('results/exp1_supervised_node_*/embeddings.pt')
print(f"Shape: {embeddings['train_embeddings'].shape}")
```

### Use Analysis Script

```bash
python analyze_results.py --results_root results
```

### Key Comparisons

1. **Supervised Baselines:**
   - Accuracy 1 vs Accuracy 2: Node vs link supervised

2. **Transfer Learning:**
   - Accuracy 3 vs Accuracy 2: Can node encoder do links?
   - Accuracy 4 vs Accuracy 1: Can link encoder do nodes?

3. **Self-Supervised vs Supervised:**
   - Accuracy 5 vs Accuracy 1: GraphMAE vs supervised on nodes
   - Accuracy 8 vs Accuracy 2: Combined loss vs supervised on links

4. **Self-Supervised Generalization:**
   - Accuracy 6 vs Accuracy 3: GraphMAE on links
   - Accuracy 7 vs Accuracy 4: Combined loss on nodes

## Troubleshooting

### Job Fails Immediately

Check error log:
```bash
tail logs/exp*_JOBID.err
```

Common issues:
- CUDA out of memory: Reduce batch_size or hidden_channels

### Job Runs But Poor Results

Check training curves:
```python
history = torch.load('results/.../training_history.pt')
import matplotlib.pyplot as plt
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.legend(['Train', 'Val'])
plt.show()
```

Common issues:
- Loss not decreasing: Try lower learning rate (0.0001)
- Overfitting: Increase dropout or weight_decay
- Underfitting: Increase hidden_channels or num_layers

### Out of Memory

Reduce memory usage:
```bash
--batch_size 512              # Down from 1024
--hidden_channels 64          # Down from 128
--downstream_batch_size 512   # Down from 1024
```

Or request more memory in job script:
```bash
#BSUB -R "rusage[mem=64GB]"   # Up from 32GB
```

### Jobs Stuck in Queue

Check queue status:
```bash
bstat -q gpua100
```

Try different queue:
```bash
#BSUB -q gpuv100    # Instead of gpua100
```