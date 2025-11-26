# Test Scripts

Quick validation scripts to test experiments before GPU training.

## Purpose

These scripts verify that all experiments run correctly on CPU with minimal resources before submitting expensive GPU jobs.

**Test settings:**
- CPU only (no GPU required)
- Small model (32 dim, 1 layer)
- Small batches (128)
- Few epochs (3)
- Test mode optimizations (see below)
- Fast execution (2-15 min depending on options)

## Test Mode Optimizations

When `--test_mode` is enabled, the following optimizations are automatically applied:

1. **Dataset Subsampling**: Reduces dataset to 5000 nodes (from ~736k for OGB-MAG)
2. **Reduced Neighbor Sampling**: Uses [10, 5] neighbors per layer instead of [30, 30]
3. **Fewer Downstream Runs**: Reduces from 2+ runs to 1 run for uncertainty estimation
4. **Reduced Downstream Epochs**: Limits to 3 epochs instead of default 100
5. **Limited Link Edges**: For link prediction, limits to 2000 edges max

**Additional Option**: Use `--skip_downstream` to skip downstream evaluation entirely for the fastest possible tests (2-5 minutes total).

## Quick Start

### Ultra-Fast Tests (Recommended for Quick Checks)

```bash
bash scripts/test/run_quick_tests.sh
```

Time: 2-5 minutes total
- Skips downstream evaluation entirely
- Uses all test mode optimizations
- Fastest way to verify everything works

### Run All Tests (With Downstream Evaluation)

```bash
bash scripts/test/run_all_tests.sh
```

Time: 5-15 minutes total (optimized from 10-30 minutes)
Output: Results in `results/test_exp{1,2,5,8}/`

### Run in Background

To run tests in background (safe to exit SSH):

```bash
bash scripts/test/run_tests_background.sh
```

This will:
- Run tests using nohup
- Save output to `logs/test_run_TIMESTAMP.log`
- Continue running after disconnection

Monitor progress:
```bash
tail -f logs/test_run_*.log
```

### Quick Component Validation

Before running full tests:

```bash
python scripts/test/quick_validation.py
```

Time: 10 seconds
Tests: Loss functions, objectives, forward/backward passes

## Individual Test Scripts

### Test Experiment 1: Supervised Node

```bash
bash scripts/test/test_exp1_supervised_node.sh
```

Tests:
- Data loading
- Supervised node classification
- Training for 3 epochs
- Downstream link prediction

### Test Experiment 2: Supervised Link

```bash
bash scripts/test/test_exp2_supervised_link.sh
```

Tests:
- Link prediction with negative sampling
- BCE loss
- Downstream node classification

### Test Experiment 5: GraphMAE

```bash
bash scripts/test/test_exp5_graphmae.sh
```

Tests:
- Feature masking
- SCE loss computation
- Feature reconstruction
- Downstream evaluation (both tasks)

### Test Experiment 8: Combined Loss

```bash
bash scripts/test/test_exp8_combined_loss.sh
```

Tests:
- MER loss
- TAR loss
- PFP loss
- Combined loss (MER + TAR + PFP)
- Downstream evaluation (both tasks)

## What Gets Tested

### Component Tests (quick_validation.py)

Loss Functions:
- SCE (Scaled Cosine Error)
- MER (Masked Edge Reconstruction)
- TAR (Topology-Aware Reconstruction)
- PFP (Preference-based Feature Propagation)
- Combined (MER + TAR + PFP)

Training Objectives:
- SupervisedNodeClassification
- SupervisedLinkPrediction
- SelfSupervisedNodeReconstruction (MSE & SCE)
- SelfSupervisedEdgeReconstruction (BCE & Combined)

Gradient Flow:
- Forward pass
- Backward pass
- Parameter updates
- Loss decreasing over steps

### Full Experiment Tests

Each test script validates:

1. Data Loading: Dataset loads without errors
2. Model Creation: Model initializes correctly
3. Training Loop: Runs for 3 epochs without crashes
4. Loss Computation: All loss variants compute correctly
5. Gradient Flow: Parameters update properly
6. Metrics Tracking: Loss/accuracy logged correctly
7. Embedding Extraction: Embeddings saved correctly
8. Downstream Evaluation: Both tasks evaluate successfully
9. Checkpoint Saving: All files saved properly

## Expected Output

### During Test Run

```
========================================================================
Testing Experiment 1: Supervised Node Classification
========================================================================

Epoch 1/3:
  Train Loss: 2.3456, Train Acc: 0.1234
  Val Loss: 2.2345, Val Acc: 0.1456

Epoch 2/3:
  Train Loss: 2.1234, Train Acc: 0.2345
  Val Loss: 2.0123, Val Acc: 0.2567

Epoch 3/3:
  Train Loss: 1.9876, Train Acc: 0.3456
  Val Loss: 1.8765, Val Acc: 0.3678

Test: Loss: 1.8543, Acc: 0.3712

Downstream Node: 0.3524 ± 0.0123
Downstream Link: 0.6234 ± 0.0234

Test 1 completed!
========================================================================
```

### Success Indicators

- Loss decreases over epochs
- Accuracy increases over epochs
- No errors or warnings
- All result files created:
  - model_*.pt
  - embeddings.pt
  - training_history.pt
  - downstream_node_results.pt
  - downstream_link_results.pt

### Warning Signs

- Loss is NaN: Check learning rate or data
- Loss doesn't decrease: May need more epochs (normal for 3 epochs)
- Accuracy stays at ~10%: Might be random guessing (normal for 3 epochs)
- CUDA errors: Make sure you're on CPU (not GPU)

## Test vs Production Settings

| Parameter | Test | Production (GPU) |
|-----------|------|------------------|
| Device | CPU | GPU (A100) |
| Hidden dim | 32 | 128 |
| Layers | 1 | 2 |
| Batch size | 128 | 1024 |
| Epochs | 3 | 1000 |
| Dropout | 0.3 | 0.5 |
| Learning rate | 0.01 | 0.001 |
| Downstream runs | 2 | 10 |
| Downstream epochs | 3 | 100 |
| Test mode | Yes | No |
| Time | 5-10 min | 12-24 hours |

Note: Test settings are designed for speed, not accuracy. Low accuracies in tests are expected.

## Troubleshooting

### Script Won't Run

Error: Permission denied

Solution:
```bash
chmod +x scripts/test/*.sh
```

### Out of Memory (Even on CPU)

Solution: Reduce batch size in test scripts:
```bash
# Edit test script and change:
--batch_size 64        # Down from 128
--downstream_batch_size 64
```

### Tests Pass But Different Results

This is normal because:
- Random initialization differs
- CPU vs GPU floating point differences
- Test mode uses subset of data
- Few epochs don't converge

As long as no errors occur, tests are successful.

## After Testing

### If All Tests Pass

Ready for GPU training:

```bash
bash scripts/hpc/submit_all_experiments.sh
```

### If Any Test Fails

1. Check error message in terminal output
2. Check error log (if test creates one)
3. Run quick validation to isolate issue:
   ```bash
   python scripts/test/quick_validation.py
   ```
4. Fix the issue before submitting to GPU
5. Re-run tests to verify fix

## Common Test Failures

### Loss is NaN

Cause: Learning rate too high or numerical instability

Fix: Already handled in test scripts (lr=0.01, smaller model)

If still happens: Check data loading

### Accuracy Stays at Baseline

Cause: Too few epochs (3 is very little)

Fix: This is expected in tests. GPU training uses 1000 epochs.

### Downstream Eval Fails

Cause: Embeddings not extracted or wrong shape

Fix: Check `--extract_embeddings` flag is set

## Clean Up Test Results

After testing, remove test results:

```bash
rm -rf results/test_exp*
```

This frees up space before running actual experiments.

## Background Execution

To run tests in background (persists after SSH disconnect):

### Option 1: Using Helper Script

```bash
bash scripts/test/run_tests_background.sh
```

### Option 2: Manual nohup

```bash
nohup bash scripts/test/run_all_tests.sh > logs/test_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Monitoring Background Tests

```bash
# View log in real-time
tail -f logs/test_run_*.log

# Check if still running
ps aux | grep run_all_tests

# Kill if needed
pkill -f run_all_tests
```

## Summary

Before GPU training, always run:

```bash
# Quick component check (10 seconds)
python scripts/test/quick_validation.py

# Full experiment tests (10-30 minutes)
bash scripts/test/run_all_tests.sh
# or for background:
bash scripts/test/run_tests_background.sh
```

After tests pass, submit to GPU:

```bash
bash scripts/hpc/submit_all_experiments.sh
```

This workflow catches errors early and saves expensive GPU time.

## Files

- `quick_validation.py` - Component tests (10 sec)
- `test_exp1_supervised_node.sh` - Test Experiment 1
- `test_exp2_supervised_link.sh` - Test Experiment 2
- `test_exp5_graphmae.sh` - Test Experiment 5
- `test_exp8_combined_loss.sh` - Test Experiment 8
- `run_all_tests.sh` - Run all tests sequentially
- `run_tests_background.sh` - Run tests in background with logging
- `list_tests.sh` - Show available test options
- `README.md` - This file
