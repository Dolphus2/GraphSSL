# GraphSSL
DTU Deep Learning Project: Self-supervised Graph Representation Learning

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

```bash
# Test setup
python -m graphssl.test_pipeline

# Run pipeline
python -m graphssl.main --epochs 10

# Or use the installed command
graphssl --epochs 10
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## Project Structure

```
GraphSSL/
├── src/
│   └── graphssl/           # Main package
│       ├── __init__.py
│       ├── main.py         # Main pipeline script
│       ├── test_pipeline.py # Setup verification
│       ├── run_examples.sh  # Example configurations
│       ├── run_hpc.sh      # HPC submission script
│       └── utils/          # Utility modules
│           ├── __init__.py
│           ├── data_utils.py
│           ├── models.py
│           └── training_utils.py
├── data/                   # Dataset storage (auto-created)
├── results/               # Training outputs (auto-created)
├── requirements.txt       # Python dependencies
├── QUICKSTART.md         # Quick start guide
└── README.md             # This file
```

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [src/graphssl/README.md](src/graphssl/README.md) - Detailed documentation
- [docs/DOWNSTREAM_EVALUATION.md](docs/DOWNSTREAM_EVALUATION.md) - Downstream evaluation guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details 
