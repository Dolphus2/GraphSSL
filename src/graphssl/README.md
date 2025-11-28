# GraphSSL

Graph Self-Supervised Learning framework for heterogeneous graphs using GraphSAGE on the OGB-MAG dataset.

## Overview

GraphSSL provides a unified framework for training graph neural networks with different learning objectives:

- Supervised node classification
- Supervised link prediction  
- Self-supervised node reconstruction (GraphMAE)
- Self-supervised edge reconstruction (Combined loss: MER + TAR + PFP)

All experiments use GraphSAGE as the backbone encoder on the OGB-MAG dataset for academic paper venue prediction.

## Quick Start

### Training on GPU

Submit all experiments to HPC:

```bash
cd GraphSSL
bash scripts/hpc/submit_all_experiments.sh
```

This submits 4 experiments in parallel, each running on a separate GPU.

### Testing on CPU

Before GPU training, verify everything works:

```bash
# Quick validation (10 seconds)
python scripts/test/quick_validation.py

# Full tests (10-30 minutes)
bash scripts/test/run_all_tests.sh

# Or run in background
bash scripts/test/run_tests_background.sh
```

## Architecture

### Model

- Backbone: Heterogeneous GraphSAGE
- Hidden dimension: 128
- Layers: 2
- Aggregation: Mean
- Dropout: 0.5

### Dataset

- Name: OGB-MAG (Microsoft Academic Graph)
- Task: Venue (conference) prediction for papers
- Nodes: Papers, Authors, Institutions, Fields
- Target: 349 venue classes

## Training Objectives

### 1. Supervised Node Classification

Train encoder to predict node labels directly.

```bash
python -m graphssl.main \
    --objective_type supervised_node_classification \
    --target_node paper \
    --epochs 1000
```

### 2. Supervised Link Prediction

Train encoder to predict edge existence.

```bash
python -m graphssl.main \
    --objective_type supervised_link_prediction \
    --target_edge_type paper,cites,paper \
    --epochs 1000
```

### 3. Self-Supervised Node (GraphMAE)

Train encoder to reconstruct masked node features using SCE loss.

```bash
python -m graphssl.main \
    --objective_type self_supervised_node \
    --loss_fn sce \
    --mask_ratio 0.5 \
    --use_feature_decoder \
    --epochs 1000
```

### 4. Self-Supervised Edge (Combined Loss)

Train encoder to reconstruct edges using MER + TAR + PFP loss.

```bash
python -m graphssl.main \
    --objective_type self_supervised_edge \
    --loss_fn combined_loss \
    --mer_weight 1.0 \
    --tar_weight 1.0 \
    --pfp_weight 1.0 \
    --use_edge_decoder \
    --epochs 1000
```

## Loss Functions

### Node Reconstruction

- **MSE**: Mean Squared Error (baseline)
- **SCE**: Scaled Cosine Error (GraphMAE) - focuses on directional similarity

### Edge Reconstruction

- **BCE**: Binary Cross-Entropy (baseline)
- **MER**: Masked Edge Reconstruction - standard edge prediction
- **TAR**: Topology-Aware Reconstruction - contrastive topology preservation
- **PFP**: Preference-based Feature Propagation - feature similarity preservation
- **Combined**: MER + TAR + PFP with configurable weights

## Command-Line Arguments

### Common Parameters

```bash
--data_root data                      # Dataset directory
--results_root results/exp_name       # Output directory
--hidden_channels 128                 # Embedding dimension
--num_layers 2                        # Number of GNN layers
--dropout 0.5                         # Dropout rate
--batch_size 1024                     # Training batch size
--epochs 1000                         # Maximum epochs
--lr 0.001                            # Learning rate
--weight_decay 0.0005                 # L2 regularization
--patience 100                        # Early stopping patience
--seed 42                             # Random seed
```

### Objective-Specific

For self-supervised node:
```bash
--mask_ratio 0.5                      # Feature masking ratio
--use_feature_decoder                 # Use MLP decoder
```

For self-supervised edge:
```bash
--mer_weight 1.0                      # MER loss weight
--tar_weight 1.0                      # TAR loss weight
--pfp_weight 1.0                      # PFP loss weight
--tar_temperature 0.5                 # TAR contrastive temperature
--use_edge_decoder                    # Use MLP decoder
```

### Downstream Evaluation

```bash
--extract_embeddings                  # Save node embeddings
--downstream_eval                     # Run downstream evaluation
--downstream_task both                # node/link/both
--downstream_n_runs 10                # Number of evaluation runs
--downstream_node_epochs 100          # Downstream node classification epochs
--downstream_link_epochs 10            # Downstream link prediction epochs
```

## Output Files

Each experiment creates:

```
results/exp_name/
├── checkpoints/
│   ├── best_model.pt              # Best model by validation
│   └── last_checkpoint.pt         # Most recent checkpoint
├── model_*.pt                     # Final model with test metrics
├── embeddings.pt                  # Extracted node embeddings
├── training_history.pt            # Loss and accuracy curves
├── downstream_node_results.pt     # Downstream node evaluation
└── downstream_link_results.pt     # Downstream link evaluation
```

## Loading Results

```python
import torch

# Load test metrics
checkpoint = torch.load('results/exp_name/model_*.pt')
print(f"Test accuracy: {checkpoint['test_metrics']['acc']:.4f}")

# Load embeddings
embeddings = torch.load('results/exp_name/embeddings.pt')
train_emb = embeddings['train_embeddings']
test_emb = embeddings['test_embeddings']

# Load downstream results
downstream = torch.load('results/exp_name/downstream_node_results.pt')
print(f"Downstream accuracy: {downstream['test_acc_mean']:.4f} ± {downstream['test_acc_std']:.4f}")
```

## HPC Usage

### Submit Jobs

```bash
# All experiments in parallel
bash scripts/hpc/submit_all_experiments.sh

# Individual experiments
bsub < scripts/hpc/exp1_supervised_node.sh
bsub < scripts/hpc/exp2_supervised_link.sh
bsub < scripts/hpc/exp5_graphmae_node.sh
bsub < scripts/hpc/exp8_combined_loss_edge.sh
```

### Monitor Jobs

```bash
# Check status
bstat

# View logs
tail -f logs/exp1_supervised_node_JOBID.out

# Cancel job
bkill JOBID
```

## Project Structure

```
graphssl/
├── main.py                        # Main training script
├── downstream_evaluation.py       # Downstream evaluation
├── utils/
│   ├── data_utils.py             # Data loading
│   ├── models.py                 # Model definitions
│   ├── training_utils.py         # Training loops
│   ├── objective_utils.py        # Loss functions and objectives
│   └── downstream.py             # Downstream evaluation
└── README.md                     # This file
```

## Documentation

- **EXPERIMENTS.md**: Complete experimental guide for formal GPU training
- **scripts/test/README.md**: Testing documentation