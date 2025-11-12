# Quick Start Guide - GraphSSL Supervised Learning

**Important:** All commands should be run from the GraphSSL root directory.

## Installation

### Option 1: Editable Install (Recommended for Development)
```bash
# Install the package in editable mode
python -m pip install -e .

# Or with optional dependencies
python -m pip install -e ".[dev,notebook]"
```

### Option 2: Install from requirements.txt
```bash
# Install dependencies only (not as a package)
python -m pip install -r requirements.txt
```

## Test Setup

```bash
python -m graphssl.test_pipeline
```

## Run Pipeline

### Using installed command
```bash
# Quick test
graphssl --epochs 10

# Full training
graphssl

# With custom parameters
graphssl \
    --hidden_channels 256 \
    --num_layers 3 \
    --batch_size 512 \
    --epochs 100 \
    --extract_embeddings
```

### Using module syntax
```bash
python -m graphssl.main --epochs 10
```

## Submit to HPC (DTU)

```bash
# Edit src/graphssl/run_hpc.sh if needed, then submit from GraphSSL root:
bsub < src/graphssl/run_hpc.sh

# Check job status:
bstat

# View logs:
tail -f logs/graphssl_<JOBID>.out
```

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hidden_channels` | 128 | Hidden dimension size |
| `--num_layers` | 2 | Number of GNN layers |
| `--batch_size` | 1024 | Training batch size |
| `--epochs` | 100 | Maximum epochs |
| `--lr` | 0.001 | Learning rate |
| `--dropout` | 0.5 | Dropout rate |
| `--patience` | 10 | Early stopping patience |
| `--extract_embeddings` | False | Save embeddings |
| `--log_level` | INFO | Logging verbosity |

## Check Results

```bash
# View saved results
ls -lh results/

# Load results in Python
python
>>> import torch
>>> results = torch.load("results/model_supervised.pt")
>>> print(f"Test Accuracy: {results['test_acc']:.4f}")
```

## Troubleshooting

**Out of Memory?**
```bash
graphssl --batch_size 256 --hidden_channels 64
```

**Want faster training?**
```bash
graphssl --batch_size 2048 --num_workers 8
```

**Verbose logging for debugging?**
```bash
graphssl --log_level DEBUG
```

**Need help?**
```bash
graphssl --help
```

## File Structure

```
GraphSSL/                     # Root directory (run commands from here)
├── src/
│   └── graphssl/            # Main package
│       ├── main.py          # Main pipeline
│       ├── test_pipeline.py # Test setup
│       ├── run_examples.sh  # Example configurations
│       ├── run_hpc.sh       # HPC submission script
│       └── utils/           # Utility modules
├── data/                    # Dataset storage (auto-created)
├── results/                 # Training outputs (auto-created)
└── requirements.txt         # Dependencies
```
├── test_pipeline.py      # Test setup
├── run_hpc.sh           # HPC submission script
├── utils/
│   ├── data_utils.py    # Data loading
│   ├── models.py        # GraphSAGE model
│   └── training_utils.py # Training functions
```

## Expected Output

```
Test Accuracy: ~0.45-0.50
Training Time: ~1-2 hours on GPU
Model Size: ~2-5M parameters
```

For detailed documentation, see `src/README.md` and `IMPLEMENTATION_SUMMARY.md`.
