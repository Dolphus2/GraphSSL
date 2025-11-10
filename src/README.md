# GraphSSL - Supervised Learning Pipeline

This directory contains a complete supervised learning pipeline for venue prediction on the OGB_MAG dataset using heterogeneous GraphSAGE.

## Project Structure

```
src/
├── GraphSSL.py                 # Main pipeline script
├── dataset_to_inductive.py     # Utility to convert dataset to inductive setting
├── utils/
│   ├── data_utils.py          # Dataset loading and data loader creation
│   ├── models.py              # GraphSAGE model implementation
│   ├── training_utils.py      # Training and evaluation functions
│   └── loading_utils.py       # Additional loading utilities
```

## Installation

Make sure you have the required dependencies installed. If using `uv` on HPC:

```bash
# Install PyTorch Geometric dependencies
python -m pip install pyg-lib -f https://data.pyg.org/whl/torch-2.9.0+cu126.html
python -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

## Usage

### Basic Usage

Run the complete supervised learning pipeline with default parameters:

```bash
cd src
python GraphSSL.py
```

### Advanced Usage

Customize the pipeline with command-line arguments:

```bash
python GraphSSL.py \
    --data_root ../data \
    --results_root ../results \
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
- `--data_root`: Root directory for dataset storage (default: `../data`)
- `--results_root`: Root directory for results (default: `../results`)
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
Loading OGB_MAG dataset from ../data
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
