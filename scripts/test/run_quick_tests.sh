#!/bin/bash
#
# Ultra-fast test script - skips downstream evaluation
# Use this for the fastest possible check that everything works
# Runs all 4 experiments with minimal settings and no downstream eval
#
# Usage:
#   bash scripts/test/run_quick_tests.sh              # Run in foreground
#   nohup bash scripts/test/run_quick_tests.sh &      # Run in background

echo "========================================================================"
echo "GraphSSL - Quick Tests (No Downstream Evaluation)"
echo "========================================================================"
echo ""
echo "This will run all 4 experiments with minimal settings and SKIP"
echo "downstream evaluation for fastest possible testing."
echo ""
echo "Settings:"
echo "  - CPU only (no GPU required)"
echo "  - Small model (32 dim, 1 layer)"
echo "  - Small batches (128)"
echo "  - Few epochs (3)"
echo "  - Test mode (subsamples dataset, reduces neighbors)"
echo "  - NO downstream evaluation (fastest)"
echo ""
echo "Estimated time: 2-5 minutes total"
echo ""
echo "Date: $(date)"
echo "Hostname: $(hostname)"
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
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_test_exp1 \
    --objective_type supervised_node_classification \
    --target_node paper \
    --hidden_channels 32 \
    --num_layers 1 \
    --dropout 0.3 \
    --batch_size 128 \
    --epochs 3 \
    --lr 0.01 \
    --weight_decay 0.0 \
    --patience 10 \
    --num_workers 2 \
    --log_interval 5 \
    --extract_embeddings \
    --test_mode \
    --skip_downstream \
    --seed 42

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
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_test_exp2 \
    --objective_type supervised_link_prediction \
    --target_edge_type paper,cites,paper \
    --hidden_channels 32 \
    --num_layers 1 \
    --dropout 0.3 \
    --batch_size 128 \
    --epochs 3 \
    --lr 0.01 \
    --weight_decay 0.0 \
    --patience 10 \
    --neg_sampling_ratio 1.0 \
    --num_workers 2 \
    --log_interval 5 \
    --extract_embeddings \
    --test_mode \
    --skip_downstream \
    --seed 42

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
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_test_exp5 \
    --objective_type self_supervised_node \
    --target_node paper \
    --hidden_channels 32 \
    --num_layers 1 \
    --dropout 0.3 \
    --batch_size 128 \
    --epochs 3 \
    --lr 0.01 \
    --weight_decay 0.0 \
    --patience 10 \
    --num_workers 2 \
    --log_interval 5 \
    --mask_ratio 0.5 \
    --use_feature_decoder \
    --loss_fn sce \
    --extract_embeddings \
    --test_mode \
    --skip_downstream \
    --seed 42

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
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_test_exp8 \
    --objective_type self_supervised_edge \
    --target_edge_type paper,cites,paper \
    --hidden_channels 32 \
    --num_layers 1 \
    --dropout 0.3 \
    --batch_size 128 \
    --epochs 3 \
    --lr 0.01 \
    --weight_decay 0.0 \
    --patience 10 \
    --neg_sampling_ratio 1.0 \
    --num_workers 2 \
    --log_interval 5 \
    --use_edge_decoder \
    --loss_fn combined_loss \
    --mer_weight 1.0 \
    --tar_weight 1.0 \
    --pfp_weight 1.0 \
    --extract_embeddings \
    --test_mode \
    --skip_downstream \
    --seed 42

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
echo "ALL QUICK TESTS PASSED"
echo "========================================================================"
echo ""
echo "Completion time: $(date)"
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in:"
echo "  results/quick_test_exp1/  - Supervised Node Classification"
echo "  results/quick_test_exp2/  - Supervised Link Prediction"
echo "  results/quick_test_exp5/  - GraphMAE"
echo "  results/quick_test_exp8/  - Combined Loss"
echo ""
echo "What was tested:"
echo "  * Data loading and preprocessing"
echo "  * Dataset subsampling (test mode)"
echo "  * Model initialization"
echo "  * Training loop (3 epochs each)"
echo "  * Loss computation (all variants)"
echo "  * Gradient flow and optimization"
echo "  * Embedding extraction"
echo ""
echo "NOTE: Downstream evaluation was skipped for speed."
echo "For full tests with downstream evaluation, run:"
echo "  bash scripts/test/run_all_tests.sh"
echo ""
echo "========================================================================"

