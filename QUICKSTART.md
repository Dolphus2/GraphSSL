# Quick Start Guide - GraphSSL Supervised Learning

## Installation (HPC with UV)

```bash
# Install PyTorch Geometric dependencies
python -m pip install pyg-lib -f https://data.pyg.org/whl/torch-2.9.0+cu126.html
python -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

## Test Setup

```bash
cd src
python test_pipeline.py
```

## Run Pipeline

### Local Test (10 epochs)
```bash
python GraphSSL.py --epochs 10
```

### Full Training
```bash
python GraphSSL.py
```

### With Custom Parameters
```bash
python GraphSSL.py \
    --hidden_channels 256 \
    --num_layers 3 \
    --batch_size 512 \
    --epochs 100 \
    --extract_embeddings
```

## Submit to HPC (DTU)

```bash
# Edit run_hpc.sh if needed, then:
bsub < run_hpc.sh

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

## Check Results

```bash
# View saved results
ls -lh ../results/

# Load results in Python
python
>>> import torch
>>> results = torch.load("../results/model_supervised.pt")
>>> print(f"Test Accuracy: {results['test_acc']:.4f}")
```

## Troubleshooting

**Out of Memory?**
```bash
python GraphSSL.py --batch_size 256 --hidden_channels 64
```

**Want faster training?**
```bash
python GraphSSL.py --batch_size 2048 --num_workers 8
```

**Need help?**
```bash
python GraphSSL.py --help
```

## File Structure

```
src/
├── GraphSSL.py           # Main pipeline (RUN THIS)
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
