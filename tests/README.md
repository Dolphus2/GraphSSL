# GraphSSL Tests

This directory contains unit tests for the GraphSSL framework.

## Test Files

### `test_loss_functions.py`
Comprehensive unit tests for all loss functions:
- **SCE Loss** (Scaled Cosine Error): Tests for GraphMAE node reconstruction
- **MER Loss** (Masked Edge Reconstruction): Tests for edge reconstruction
- **TAR Loss** (Topology-Aware Reconstruction): Tests for contrastive topology learning
- **PFP Loss** (Preference-based Feature Propagation): Tests for feature similarity preservation
- **Combined Loss**: Tests for MER + TAR + PFP combination

**Test Coverage:**
- Basic functionality
- Gradient flow
- Numerical stability
- Edge cases (identical inputs, no positive edges, etc.)
- Different batch sizes
- Different parameter values

### `test_objectives.py`
Unit tests for training objective classes:
- **SupervisedNodeClassification**: Tests for supervised node prediction
- **SupervisedLinkPrediction**: Tests for supervised link prediction
- **SelfSupervisedNodeReconstruction**: Tests for self-supervised node tasks (GraphMAE)
- **SelfSupervisedEdgeReconstruction**: Tests for self-supervised edge tasks with all loss variants
- **Decoders**: Tests for EdgeDecoder and FeatureDecoder

**Test Coverage:**
- Initialization with various parameters
- Forward pass and loss computation
- Training vs evaluation modes
- With and without decoders
- All loss function variants (bce, mer, tar, pfp, combined_loss)

### `test_pipeline.py`
Integration tests for the full training pipeline (existing file)

### `test_downstream.py`
Tests for downstream evaluation tasks (existing file)

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
# Test loss functions
pytest tests/test_loss_functions.py -v

# Test objectives
pytest tests/test_objectives.py -v

# Test pipeline
pytest tests/test_pipeline.py -v

# Test downstream
pytest tests/test_downstream.py -v
```

### Run Specific Test Class
```bash
# Test SCE loss only
pytest tests/test_loss_functions.py::TestSCELoss -v

# Test combined loss only
pytest tests/test_loss_functions.py::TestCombinedLoss -v

# Test edge objectives only
pytest tests/test_objectives.py::TestSelfSupervisedEdgeReconstruction -v
```

### Run Specific Test Method
```bash
# Test SCE gradient flow
pytest tests/test_loss_functions.py::TestSCELoss::test_sce_loss_gradient_flow -v

# Test combined loss with different weights
pytest tests/test_objectives.py::TestSelfSupervisedEdgeReconstruction::test_step_with_combined_loss -v
```

### Run Tests with Coverage
```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage report
pytest tests/ --cov=graphssl --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Run Tests in Parallel
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (faster)
pytest tests/ -n auto
```

## Test Statistics

Current test coverage:
- **Loss Functions**: 100+ tests covering all edge cases
- **Objectives**: 40+ tests covering all configurations
- **Pipeline**: Integration tests (existing)
- **Downstream**: Evaluation tests (existing)

## Troubleshooting

### Tests are slow
```bash
# Run in parallel
pytest tests/ -n auto

# Or run only fast tests
pytest tests/ -m "not slow"
```