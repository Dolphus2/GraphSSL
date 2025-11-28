# GraphSSL - Supervised Learning Pipeline

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
--downstream_link_epochs 10           # Downstream link prediction epochs
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
src/graphssl/
├── __init__.py                # Package initialization
├── main.py                    # Main pipeline script
├── test_pipeline.py           # Test script to verify setup
├── run_examples.sh            # Example run configurations
├── run_hpc.sh                 # HPC submission script
└── utils/
    ├── __init__.py
    ├── data_utils.py          # Dataset loading and data loader creation
    ├── models.py              # GraphSAGE model implementation
    └── training_utils.py      # Training and evaluation functions
```

## Installation

Install the package in editable mode from the GraphSSL root directory:

```bash
# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[dev,notebook]"
```

## Usage

**Important:** All commands should be run from the GraphSSL root directory.

### Basic Usage

Run the complete supervised learning pipeline with default parameters:

```bash
# Using the installed command
graphssl

# Or using module syntax
python -m graphssl.main
```

### Advanced Usage

Customize the pipeline with command-line arguments:

```bash
graphssl \
    --data_root data \
    --results_root results \
    --hidden_channels 256 \
    --num_layers 3 \
    --batch_size 512 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --extract_embeddings
```

### Command-Line Arguments

#### Data Arguments
- `--data_root`: Root directory for dataset storage (default: `data`)
- `--results_root`: Root directory for results (default: `results`)
- `--preprocess`: Preprocessing method for node embeddings (`metapath2vec` or `transe`, default: `metapath2vec`)
- `--target_node`: Target node type for prediction (default: `paper`)

#### Model Arguments
- `--hidden_channels`: Hidden dimension size (default: 128)
- `--num_layers`: Number of GraphSAGE layers (default: 2)
- `--dropout`: Dropout rate (default: 0.5)

#### Data Loader Arguments
- `--num_neighbors`: Number of neighbors to sample at each layer (default: 15 10)
- `--batch_size`: Batch size for training (default: 1024)
- `--num_workers`: Number of worker processes for data loading (default: 4)

#### Training Arguments
- `--epochs`: Maximum number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for L2 regularization (default: 0.0)
- `--patience`: Early stopping patience (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)

#### Additional Options
- `--extract_embeddings`: Extract and save node embeddings after training
- `--log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL, default: INFO)

## Pipeline Steps

The pipeline executes the following steps:

1. **Load Dataset**: Downloads and loads the OGB_MAG dataset with specified preprocessing
2. **Create Data Loaders**: Sets up NeighborLoader for train/val/test splits
3. **Create Model**: Initializes a heterogeneous GraphSAGE model
4. **Setup Optimizer**: Configures Adam optimizer with specified hyperparameters
5. **Train Model**: Trains the model with early stopping
6. **Test Model**: Evaluates the trained model on the test set
7. **Save Results**: Saves model checkpoint, training history, and metrics
8. **Extract Embeddings** (optional): Extracts and saves node embeddings

## Output Files

After running the pipeline, the following files will be saved in the results directory:

- `best_model.pt`: Best model checkpoint during training
- `model_supervised.pt`: Final model with training metadata
- `training_history.pt`: Training metrics (loss, accuracy) per epoch
- `embeddings.pt` (optional): Extracted node embeddings for train/val/test sets

## Example Output

```
================================================================================
GraphSSL - Supervised Learning Pipeline
Task: Venue Prediction on OGB_MAG Dataset
================================================================================

Using device: cuda
GPU: NVIDIA A100-SXM4-40GB

================================================================================
Step 1: Loading Dataset
================================================================================
Loading OGB_MAG dataset from data
Using preprocessing method: metapath2vec

Dataset loaded successfully!
Node types: ['paper', 'author', 'institution', 'field_of_study']
Edge types: [('paper', 'cites', 'paper'), ...]

Paper node statistics:
  Number of papers: 736,389
  Feature dimension: 128
  Number of venues (classes): 349
  Train samples: 629,571
  Val samples: 64,879
  Test samples: 41,939

...

Test Accuracy: 0.4532
Test Loss: 2.1234
```

## Model Architecture

The pipeline uses a heterogeneous GraphSAGE model:

- **Heterogeneous Graph Support**: Handles multiple node and edge types in OGB_MAG
- **GraphSAGE Aggregation**: Samples and aggregates neighbor features at multiple hops
- **Multi-layer Architecture**: Stacks multiple GraphSAGE layers for deeper representations
- **Dropout Regularization**: Prevents overfitting during training

## Notes

- The first run will download the OGB_MAG dataset (~2GB) to the data directory
- Training time depends on GPU availability and dataset size
- Early stopping is implemented to prevent overfitting
- The model automatically uses GPU if available, otherwise falls back to CPU

## Troubleshooting

### CUDA Out of Memory
If you encounter OOM errors, try:
- Reducing `--batch_size` (e.g., 512 or 256)
- Reducing `--hidden_channels`
- Reducing `--num_neighbors` sampling rates

### Slow Data Loading
If data loading is slow:
- Adjust `--num_workers` based on your CPU cores
- On HPC systems, you may want to set `--num_workers 0` to avoid multiprocessing issues

### Dataset Download Issues
If the dataset fails to download:
- Check your internet connection
- Manually download from [OGB website](https://ogb.stanford.edu/)
- Place the dataset in the `--data_root` directory
