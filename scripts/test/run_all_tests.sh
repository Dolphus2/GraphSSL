#!/bin/bash
#
# Run all experiment tests sequentially
# This verifies that all experiments can run without errors
# Uses CPU and minimal settings for fast testing
#
# Usage:
#   bash scripts/test/run_all_tests.sh              # Run in foreground
#   nohup bash scripts/test/run_all_tests.sh &      # Run in background
#   bash scripts/test/run_all_tests.sh > test.log 2>&1 &  # Background with custom log
#

# Determine if running in background
if [ -t 1 ]; then
    BACKGROUND=false
else
    BACKGROUND=true
fi

echo "========================================================================"
echo "GraphSSL - Testing All Experiments"
echo "========================================================================"
echo ""
echo "This will run all 4 experiments with minimal settings to verify"
echo "that everything works correctly before submitting to GPU."
echo ""
echo "Settings:"
echo "  - CPU only (no GPU required)"
echo "  - Small model (32 dim, 1 layer)"
echo "  - Small batches (128)"
echo "  - Few epochs (3)"
echo "  - Test mode (subsamples dataset to 5000 nodes, reduces neighbors to [10,5])"
echo "  - Reduced downstream runs (1 run instead of 2)"
echo "  - Reduced downstream epochs (3 instead of default)"
echo ""
echo "Estimated time: 5-15 minutes total (optimized from 10-30 minutes)"
echo ""
echo "For even faster tests (skip downstream evaluation):"
echo "  bash scripts/test/run_quick_tests.sh  (2-5 minutes)"
echo ""
echo "Date: $(date)"
echo "Hostname: $(hostname)"
if [ "$BACKGROUND" = true ]; then
    echo "Mode: Background"
else
    echo "Mode: Foreground"
fi
echo "========================================================================"
echo ""

cd /dtu/blackhole/1a/222842/GraphSSL
source .venv/bin/activate

# Create test results directory
mkdir -p results

# Track start time
START_TIME=$(date +%s)

# Test 1: Supervised Node Classification
echo ""
echo "[1/4] Testing Experiment 1: Supervised Node Classification"
echo "------------------------------------------------------------------------"
bash scripts/test/test_exp1_supervised_node.sh
if [ $? -eq 0 ]; then
    echo "* Test 1 PASSED"
else
    echo "✗ Test 1 FAILED"
    exit 1
fi

# Test 2: Supervised Link Prediction
echo ""
echo "[2/4] Testing Experiment 2: Supervised Link Prediction"
echo "------------------------------------------------------------------------"
bash scripts/test/test_exp2_supervised_link.sh
if [ $? -eq 0 ]; then
    echo "* Test 2 PASSED"
else
    echo "✗ Test 2 FAILED"
    exit 1
fi

# Test 3: GraphMAE
echo ""
echo "[3/4] Testing Experiment 5: GraphMAE (Self-Supervised Node)"
echo "------------------------------------------------------------------------"
bash scripts/test/test_exp5_graphmae.sh
if [ $? -eq 0 ]; then
    echo "* Test 3 PASSED"
else
    echo "✗ Test 3 FAILED"
    exit 1
fi

# Test 4: Combined Loss
echo ""
echo "[4/4] Testing Experiment 8: Combined Loss (Self-Supervised Edge)"
echo "------------------------------------------------------------------------"
bash scripts/test/test_exp8_combined_loss.sh
if [ $? -eq 0 ]; then
    echo "* Test 4 PASSED"
else
    echo "✗ Test 4 FAILED"
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================================================"
echo "ALL TESTS PASSED"
echo "========================================================================"
echo ""
echo "Completion time: $(date)"
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in:"
echo "  results/test_exp1/  - Supervised Node Classification"
echo "  results/test_exp2/  - Supervised Link Prediction"
echo "  results/test_exp5/  - GraphMAE"
echo "  results/test_exp8/  - Combined Loss"
echo ""
echo "What was tested:"
echo "  * Data loading and preprocessing"
echo "  * Model initialization"
echo "  * Training loop (3 epochs each)"
echo "  * Loss computation (all variants)"
echo "  * Gradient flow and optimization"
echo "  * Embedding extraction"
echo "  * Downstream evaluation (node + link)"
echo "  * Checkpoint saving"
echo "  * Result file generation"
echo ""
echo "You can now safely submit to GPU:"
echo "  bash scripts/hpc/submit_all_experiments.sh"
echo ""
echo "========================================================================"

