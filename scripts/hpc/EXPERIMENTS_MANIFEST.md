# GraphSSL HPC Experiments Manifest
# ====================================
# Generated: November 28, 2025
# 
# This document describes the main experiment configurations for the GraphSSL project.
# All experiments are configured with sensible defaults and can be submitted to DTU HPC.

## Overview

This experiment suite contains 5 main experiment types that explore different training
objectives and encoder architectures for heterogeneous graph representation learning.

**Key Features:**
- All experiments use the MAG dataset (paper nodes with heterogeneous relations)
- Each experiment includes full downstream evaluation (node classification + link prediction)
- Standardized hyperparameters for fair comparison
- Both training and downstream tasks use the same train/val/test splits
- Results include uncertainty estimates (10 runs per downstream task)

## Shared Configuration

All experiments use the following shared hyperparameters:

```
Architecture:
  - hidden_channels: 128
  - num_layers: 2
  - num_neighbors: [30, 30]
  - dropout: 0.5
  - aggr: mean (within relation)
  - aggr_rel: sum (across relations)

Training:
  - epochs: 100
  - batch_size: 1024
  - lr: 0.001
  - weight_decay: 0
  - patience: 20
  - optimizer: Adam

Data:
  - target_node: "paper"
  - edge_msg_pass_prop: [0.8, 0.8, 0.8]
  - node_inductive: True
  - dependent_node_edge_data_split: True

Downstream Evaluation:
  - downstream_n_runs: 10
  - downstream_task: both
  - downstream_hidden_dim: 128
  - downstream_num_layers: 2
  - downstream_dropout: 0.5
  - downstream_node_epochs: 100
  - downstream_link_epochs: 10
  - downstream_patience: 20
```

## Experiment Descriptions

### Experiment 1: Supervised Node Classification
**File:** `exp_supervised_node.sh`  
**Job Name:** `gssl_sup_node`

**Objective:** Supervised learning for venue (field of study) prediction

**Training Configuration:**
- objective_type: supervised_node_classification
- target_edge_type: "paper,has_topic,field_of_study"
- Loss: Cross-entropy loss on venue labels

**Encoder:** Standard heterogeneous GraphSAGE
- Node embeddings learned through direct supervision
- Messages aggregated from all edge types
- Final classification layer predicts venue

**Purpose:** Baseline for fully supervised learning. Represents the upper bound of
performance when abundant labeled data is available.

---

### Experiment 2: Supervised Link Prediction
**File:** `exp_supervised_link.sh`  
**Job Name:** `gssl_sup_link`

**Objective:** Supervised learning for citation link prediction

**Training Configuration:**
- objective_type: supervised_link_prediction
- target_edge_type: "paper,cites,paper"
- neg_sampling_ratio: 1.0
- Loss: Binary cross-entropy on positive/negative link pairs

**Encoder:** Standard heterogeneous GraphSAGE with edge decoder
- Node embeddings learned through link supervision
- Edge decoder predicts link existence from node pair embeddings
- Uses dot product or MLP decoder

**Purpose:** Baseline for supervised link prediction. Tests whether link-based
supervision produces better embeddings than node-based supervision.

---

### Experiment 3: Self-Supervised Node Reconstruction (SCE)
**File:** `exp_ssl_node_sce.sh`  
**Job Name:** `gssl_ssl_sce`

**Objective:** Masked node feature reconstruction with SCE loss

**Training Configuration:**
- objective_type: self_supervised_node
- loss_fn: sce (Symmetric Cross-Entropy)
- target_edge_type: "paper,has_topic,field_of_study"
- mask_ratio: 0.5
- Loss: SCE between original and reconstructed features

**Encoder:** GraphSAGE encoder + feature decoder
- Node features are masked (entire nodes removed)
- Encoder learns to reconstruct masked features from neighbors
- SCE loss is more robust to noise than MSE

**Purpose:** Self-supervised pre-training through denoising autoencoding. Tests whether
reconstructing features from graph structure produces useful embeddings without labels.

**Related Work:** Similar to GraphMAE and HGMAE masking strategies

---

### Experiment 4: Self-Supervised Edge Reconstruction
**File:** `exp_ssl_edge.sh`  
**Job Name:** `gssl_ssl_edge`

**Objective:** Edge reconstruction with BCE loss (link prediction pretext task)

**Training Configuration:**
- objective_type: self_supervised_edge
- loss_fn: bce
- target_edge_type: "paper,cites,paper"
- neg_sampling_ratio: 1.0
- Loss: Binary cross-entropy on masked edge reconstruction

**Encoder:** GraphSAGE encoder with edge decoder
- Edges are masked during training
- Encoder learns to predict masked edges from remaining structure
- Negative samples generated for contrastive learning

**Purpose:** Self-supervised pre-training through edge prediction. Tests whether the
task of predicting graph structure produces useful embeddings for downstream tasks.

**Related Work:** Similar to DGI, GraphCL edge dropping strategies

---

### Experiment 5: Self-Supervised Combined (MER + TAR + PFP)
**File:** `exp_ssl_tarpfp.sh`  
**Job Name:** `gssl_ssl_tarpfp`

**Objective:** Combined masked edge reconstruction, type-aware regularization, and
path feature prediction

**Training Configuration:**
- objective_type: self_supervised_tarpfp
- target_edge_type: "paper,has_topic,field_of_study"
- mer_weight: 1.0 (Masked Edge Reconstruction)
- tar_weight: 1.0 (Type-Aware Regularization)
- pfp_weight: 1.0 (Path Feature Prediction)
- mask_ratio: 0.5
- neg_sampling_ratio: 1.0
- tar_temperature: 0.5
- Loss: Weighted combination of three losses

**Encoder:** GraphSAGE encoder with multiple decoders (HGMAE-style)
- **MER:** Reconstructs masked edges (link prediction)
- **TAR:** Contrastive learning with type-aware negative sampling
- **PFP:** Predicts features of nodes along metapaths

**Purpose:** State-of-the-art self-supervised learning for heterogeneous graphs.
Combines multiple pretext tasks to learn rich representations that capture both
structure and semantics.

**Related Work:** HGMAE (Heterogeneous Graph Masked Autoencoders)

---

## Downstream Evaluation Tasks

All experiments include evaluation on two downstream tasks:

### Task 1: Node Property Prediction (Venue Classification)
- **Metric:** Accuracy, Precision, Recall, F1
- **Task:** Predict venue (field of study) from paper embeddings
- **Split:** Uses official MAG train/val/test node splits
- **Classifier:** 2-layer MLP trained on frozen embeddings

### Task 2: Binary Link Prediction
- **Metric:** Accuracy, Precision, Recall, F1, AUC
- **Task:** Predict citation links between papers
- **Negatives:** 1 negative sample per positive edge
- **Classifier:** 2-layer MLP trained on frozen edge embeddings (concatenated node pairs)

Both tasks are run 10 times with different random seeds to estimate uncertainty.

---

## Submission Instructions

### Submit Individual Experiments

```bash
# From GraphSSL root directory
bsub < scripts/hpc/exp_supervised_node.sh
bsub < scripts/hpc/exp_supervised_link.sh
bsub < scripts/hpc/exp_ssl_node_sce.sh
bsub < scripts/hpc/exp_ssl_edge.sh
bsub < scripts/hpc/exp_ssl_tarpfp.sh
```

### Submit All Experiments

```bash
# From GraphSSL root directory
for script in scripts/hpc/exp_*.sh; do
    bsub < "$script"
    sleep 2  # Avoid overwhelming the scheduler
done
```

### Check Job Status

```bash
bjobs                    # View all jobs
bjobs -l <jobid>        # Detailed job info
bpeek <jobid>           # View stdout in real-time
```

---

## Expected Runtime

Approximate runtimes on DTU HPC (gpua100 queue, single GPU):

| Experiment | Training Time | Downstream Time | Total Time |
|------------|--------------|-----------------|------------|
| Supervised Node | ~2-3 hours | ~1-2 hours | ~3-5 hours |
| Supervised Link | ~3-4 hours | ~1-2 hours | ~4-6 hours |
| SSL Node (SCE) | ~3-4 hours | ~1-2 hours | ~4-6 hours |
| SSL Edge | ~3-4 hours | ~1-2 hours | ~4-6 hours |
| SSL TAR+PFP | ~4-6 hours | ~1-2 hours | ~5-8 hours |

Total for all 5 experiments: ~20-31 hours sequential, ~5-8 hours parallel

---

## Output Structure

Each experiment creates the following output structure:

```
results/exp_<name>_<jobid>_<timestamp>/
├── model.pt                          # Trained encoder weights
├── embeddings.pt                     # Extracted node embeddings
├── training_history.json             # Training loss/metrics per epoch
├── downstream_node_classification/   # Node task results
│   ├── run_1/
│   │   ├── model.pt
│   │   ├── training_history.json
│   │   └── test_metrics.json
│   ├── run_2/
│   │   └── ...
│   └── summary_statistics.json       # Mean ± std across runs
└── downstream_link_prediction/       # Link task results
    ├── run_1/
    │   └── ...
    └── summary_statistics.json
```

---

## Comparing Results

After all experiments complete, compare downstream performance:

```bash
# Extract test accuracy for node classification
for exp in results/exp_*/downstream_node_classification/summary_statistics.json; do
    echo "$(dirname $(dirname $exp)):"
    cat $exp | grep -A 3 "test_acc"
done

# Extract test accuracy for link prediction
for exp in results/exp_*/downstream_link_prediction/summary_statistics.json; do
    echo "$(dirname $(dirname $exp)):"
    cat $exp | grep -A 3 "test_acc"
done
```

---

## Customization

To modify experiments, edit the shell scripts directly or override parameters:

### Change edge message passing:
```bash
# No message passing from val/test nodes (more challenging)
--edge_msg_pass_prop 0 0 0

# Full message passing (easier, but less realistic)
--edge_msg_pass_prop 1.0 1.0 1.0
```

### Change target edge type:
```bash
# Use different edge types for training
--target_edge_type "paper,cites,paper"
--target_edge_type "author,writes,paper"
--target_edge_type "paper,has_topic,field_of_study"
```

### Increase model capacity:
```bash
--hidden_channels 256
--num_layers 3
--num_neighbors 40 40 40
```

### Quick test mode:
```bash
--epochs 10
--downstream_n_runs 3
--downstream_node_epochs 20
--downstream_link_epochs 5
--test_mode
```

---

## Environment Setup

All experiments use `scripts/hpc/set_env.sh` for personal environment variables:

```bash
export venv_activate_path="venvs/graphssl/bin/activate"
export ROOT="GraphSSL"
```

Update this file to match your HPC setup before submitting jobs.

---

## Notes

1. **Random Seeds:** All experiments use seed=42 for reproducibility, but downstream
   evaluation uses different seeds across runs for uncertainty estimation.

2. **Early Stopping:** Training uses patience=20 epochs based on validation performance.
   The best model (by validation metric) is saved and used for downstream evaluation.

3. **GPU Memory:** All experiments fit comfortably on a single A100 GPU (16GB). If you
   encounter OOM errors, reduce batch_size or hidden_channels.

4. **WandB Logging:** All experiments log to Weights & Biases. Ensure you're logged in
   or disable WandB by setting `WANDB_MODE=offline` in the environment.

5. **Reproducibility:** PyTorch and PyG introduce some non-determinism even with fixed
   seeds. Results may vary slightly across runs, which is why we report mean ± std
   across multiple downstream runs.

---

## References

- **MAG Dataset:** Microsoft Academic Graph (via OGB)
- **GraphSAGE:** Hamilton et al., "Inductive Representation Learning on Large Graphs"
- **GraphMAE:** Hou et al., "GraphMAE: Self-Supervised Masked Graph Autoencoders"
- **HGMAE:** Tian et al., "Heterogeneous Graph Masked Autoencoders"

---

## Contact

For questions or issues, contact the GraphSSL development team or consult the main
README.md in the repository root.
