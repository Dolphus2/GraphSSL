#!/bin/bash
#
# Quick CPU tests to verify GraphSSL pipeline works correctly
# These tests use minimal settings for fast validation on CPU
# Run from GraphSSL root directory: bash scripts/test/quick_test.sh
#

echo "=========================================="
echo "GraphSSL Quick CPU Tests"
echo "=========================================="
echo "Running 5 quick tests with minimal settings"
echo "Expected runtime: ~5-10 minutes on CPU"
echo ""

# Create necessary directories
mkdir -p data
mkdir -p results/quick_test_1

# Track overall success
ALL_TESTS_PASSED=true

# Test 1: Supervised Node Classification
echo ""
echo "Test 1/5: Supervised Node Classification"
echo "----------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_test_1 \
    --objective_type supervised_node_classification \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 10 10 \
    --batch_size 1024 \
    --epochs 2 \
    --lr 0.001 \
    --patience 5 \
    --num_workers 0 \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 1 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 1 \
    --downstream_node_epochs 20 \
    --downstream_link_epochs 1 \
    --downstream_batch_size 1024 \
    --edge_msg_pass_prop 0.5 0.5 0.5 \
    --seed 42 \
    --test_mode