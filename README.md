# GraphSSL
DTU Deep Learning Project: Self-supervised Graph Representation Learning

## Overview

This project explores self-supervised and supervised learning objectives for heterogeneous graph representation learning on the OGBN-MAG academic citation network. We evaluate eight training objectives across multiple downstream tasks to assess embedding quality and generalizability.

### Downstream Prediction Architecture
![DownstreamArchitecture](https://github.com/user-attachments/assets/3770beaa-1101-4a60-b457-c8ac87f37e71)
*Figure 1: Downstream evaluation framework showing task-specific prediction heads for node classification, binary link prediction, and multi-label field-of-study prediction.*

### Data Split Strategy
![Split2](https://github.com/user-attachments/assets/1d61821f-bc5a-438e-b563-a31fc2f827f9)
*Figure 2: Train/validation split showing train and validation nodes and edges along with message passing edges. Split are made inductively and edges are split dependent on the node split, meaning that all validation edges are incident to validation nodes, simulating a new paper node appearing with all of its links. Message passing edges (black) provide structural context with configurable retention ratios (ρ).*
## Demonstration
It is recommended to run the package using shell scripts as demonstrated in scripts/ , but there are two demonstration Jupyter notebooks scripts/graphssl_demonstration.ipynb and scripts/graphssl_demonstration_downstream.ipynb included for demonstration purposes. Note that for full runs these require a lot of RAM and can be fiddly to run. It is strongly recommended to run the project code using shell scripts instead of jupyter notebooks.

## Installation

From the GraphSSL root directory:

```bash
# Install in editable mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,notebook]"
```

## Quick Start

**Run commands from the GraphSSL root directory.**

### Training

```bash
# Self-supervised node reconstruction (SCE loss)
python -m graphssl.main \
    --objective_type self_supervised_node \
    --loss_fn sce \
    --epochs 50 \
    --patience 5 \
    --downstream_task node link

# Self-supervised TAR+PFP
python -m graphssl.main \
    --objective_type self_supervised_tarpfp \
    --lambda_tar 1.0 \
    --lambda_pfp 1.0 \
    --epochs 50

# Supervised node classification
python -m graphssl.main \
    --objective_type supervised_node \
    --epochs 50
```

### Downstream Evaluation Only

```bash
# Evaluate pre-trained model
python -m graphssl.downstream_evaluation \
    --model_path results/exp_ssl_node_sce_*/model_self_supervised_node.pt \
    --objective_type self_supervised_node \
    --downstream_task multiclass_link \
    --downstream_n_runs 5
```

### HPC Batch Submission

```bash
# Submit all training experiments
cd scripts/hpc
bsub < exp_ssl_node_sce.sh
bsub < exp_ssl_tarpfp.sh
# ... or submit all at once

# Submit all downstream evaluations
bash submit_all_downstream.sh
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions

## Project Structure

```
GraphSSL/
├── src/
│   └── graphssl/           # Main package
│       ├── __init__.py
│       ├── main.py         # Main training pipeline
│       ├── downstream_evaluation.py  # Standalone downstream evaluation
│       └── utils/          # Utility modules
│           ├── __init__.py
│           ├── args_utils.py        # Argument parsing
│           ├── data_utils.py        # Data loading and preprocessing
│           ├── downstream.py        # Downstream evaluation logic
│           ├── graphsage.py         # GraphSAGE model implementation
│           ├── models.py            # Neural network components
│           ├── objective_utils.py   # Training objective classes
│           ├── plotting_utils.py    # Visualization utilities
│           └── training_utils.py    # Training loops and utilities
├── scripts/
│   ├── hpc/                # HPC job submission scripts
│   │   ├── exp_*.sh        # Training experiment scripts (8)
│   │   ├── downstream/     # Downstream evaluation scripts (8)
│   │   └── set_env.sh      # Environment configuration
│   ├── Results.ipynb       # Results analysis notebook
│   ├── ExploratoryAnalysis.ipynb       # Initial dataset analysis
│   ├── graphssl_demonstration.ipynb    # Demonstration notebook
│   ├── graphssl_demonstration_downstream.ipynb    # More modular demonstration notebook
│   └── run_project_experiments.sh      # Demonstration experiment shell script
├── data/                   # Dataset storage (auto-created)
│   └── mag/                # OGBN-MAG dataset
├── results/                # Training outputs (auto-created)
│   ├── exp_*/              # Training run results
│   └── downstream_*/       # Downstream evaluation results
├── logs/                   # HPC job logs
├── tests/                  # Unit tests 
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Package configuration
├── QUICKSTART.md          # Quick start guide
└── README.md              # This file
```

## Features

- **8 Training Objectives**: Supervised (node classification, link prediction) and self-supervised (feature reconstruction with MSE/SCE, edge reconstruction, TAR, PFP, TAR+PFP)
- **3 Downstream Tasks**: Node classification (venue prediction), binary link prediction, multi-label link prediction (field-of-study)
- **Heterogeneous GNN**: GraphSAGE with relation-specific aggregation and type-specific transformations
- **Flexible Edge Masking**: Configurable message passing edge retention ratios (ρ) for ablation studies
- **Comprehensive Evaluation**: Multiple runs with statistical significance testing, top-K ranking metrics
- **HPC Ready**: Optimized batch scripts for SLURM/LSF job schedulers with automatic checkpointing

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide with basic examples
- [README_HPC.md](README_HPC.md) - HPC-specific instructions for DTU cluster
- [src/graphssl/README.md](src/graphssl/README.md) - Detailed API documentation

## Key Results

Structure-based self-supervised methods (TAR, PFP, TAR+PFP) significantly outperform feature reconstruction approaches on downstream tasks:

| Method | Node Classification | Binary Link Pred | Multi-label Top-20 Recall |
|--------|-------------------|------------------|--------------------------|
| Supervised Node | 0.374 ± 0.002 | 0.898 ± 0.002 | - |
| TAR+PFP | 0.243 ± 0.003 | 0.879 ± 0.006 | 0.236 ± 0.003 |
| PFP | 0.237 ± 0.005 | 0.879 ± 0.005 | 0.236 ± 0.004 |
| Feature Recon (SCE) | 0.064 ± 0.001 | 0.892 ± 0.006 | 0.224 ± 0.008 |

*Table: Test performance on OGBN-MAG with no context edges (ρ=1,1,1). Multi-label task uses 59,965 field-of-study classes.*

## License

MIT License - see LICENSE file for details.
