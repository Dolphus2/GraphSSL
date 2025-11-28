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
mkdir -p results/quick_tests

# Track overall success
ALL_TESTS_PASSED=true

# Test 1: Supervised Node Classification
echo ""
echo "Test 1/5: Supervised Node Classification"
echo "----------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_tests/test1_sup_node \
    --objective_type supervised_node_classification \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --hidden_channels 64 \
    --num_layers 2 \
    --num_neighbors 10 10 \
    --batch_size 256 \
    --epochs 2 \
    --lr 0.001 \
    --patience 5 \
    --num_workers 0 \
    --downstream_eval \
    --downstream_task node \
    --downstream_n_runs 1 \
    --downstream_hidden_dim 64 \
    --downstream_num_layers 1 \
    --downstream_node_epochs 2 \
    --downstream_batch_size 256 \
    --edge_msg_pass_prop 0.5 0.5 0.5 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✓ Test 1 passed"
else
    echo "✗ Test 1 failed"
    ALL_TESTS_PASSED=false
fi

# Test 2: Supervised Link Prediction
echo ""
echo "Test 2/5: Supervised Link Prediction"
echo "-------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_tests/test2_sup_link \
    --objective_type supervised_link_prediction \
    --target_node "paper" \
    --target_edge_type "paper,cites,paper" \
    --hidden_channels 64 \
    --num_layers 2 \
    --num_neighbors 10 10 \
    --batch_size 256 \
    --epochs 2 \
    --lr 0.001 \
    --neg_sampling_ratio 1.0 \
    --patience 5 \
    --num_workers 0 \
    --downstream_eval \
    --downstream_task link \
    --downstream_n_runs 1 \
    --downstream_hidden_dim 64 \
    --downstream_num_layers 1 \
    --downstream_link_epochs 2 \
    --downstream_batch_size 256 \
    --edge_msg_pass_prop 0.5 0.5 0.5 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✓ Test 2 passed"
else
    echo "✗ Test 2 failed"
    ALL_TESTS_PASSED=false
fi

# Test 3: Self-Supervised Node (SCE)
echo ""
echo "Test 3/5: Self-Supervised Node Reconstruction (SCE)"
echo "----------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_tests/test3_ssl_sce \
    --objective_type self_supervised_node \
    --loss_fn sce \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --mask_ratio 0.5 \
    --hidden_channels 64 \
    --num_layers 2 \
    --num_neighbors 10 10 \
    --batch_size 256 \
    --epochs 2 \
    --lr 0.001 \
    --patience 5 \
    --num_workers 0 \
    --downstream_eval \
    --downstream_task node \
    --downstream_n_runs 1 \
    --downstream_hidden_dim 64 \
    --downstream_num_layers 1 \
    --downstream_node_epochs 2 \
    --downstream_batch_size 256 \
    --edge_msg_pass_prop 0.5 0.5 0.5 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✓ Test 3 passed"
else
    echo "✗ Test 3 failed"
    ALL_TESTS_PASSED=false
fi

# Test 4: Self-Supervised Edge Reconstruction
echo ""
echo "Test 4/5: Self-Supervised Edge Reconstruction"
echo "----------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_tests/test4_ssl_edge \
    --objective_type self_supervised_edge \
    --loss_fn bce \
    --target_node "paper" \
    --target_edge_type "paper,cites,paper" \
    --neg_sampling_ratio 1.0 \
    --hidden_channels 64 \
    --num_layers 2 \
    --num_neighbors 10 10 \
    --batch_size 256 \
    --epochs 2 \
    --lr 0.001 \
    --patience 5 \
    --num_workers 0 \
    --downstream_eval \
    --downstream_task link \
    --downstream_n_runs 1 \
    --downstream_hidden_dim 64 \
    --downstream_num_layers 1 \
    --downstream_link_epochs 2 \
    --downstream_batch_size 256 \
    --edge_msg_pass_prop 0.5 0.5 0.5 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✓ Test 4 passed"
else
    echo "✗ Test 4 failed"
    ALL_TESTS_PASSED=false
fi

# Test 5: Self-Supervised Combined (TAR+PFP)
echo ""
echo "Test 5/5: Self-Supervised Combined (MER+TAR+PFP)"
echo "-------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_tests/test5_ssl_tarpfp \
    --objective_type self_supervised_tarpfp \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --mer_weight 1.0 \
    --tar_weight 1.0 \
    --pfp_weight 1.0 \
    --mask_ratio 0.5 \
    --neg_sampling_ratio 1.0 \
    --tar_temperature 0.5 \
    --hidden_channels 64 \
    --num_layers 2 \
    --num_neighbors 10 10 \
    --batch_size 256 \
    --epochs 2 \
    --lr 0.001 \
    --patience 5 \
    --num_workers 0 \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 1 \
    --downstream_hidden_dim 64 \
    --downstream_num_layers 1 \
    --downstream_node_epochs 2 \
    --downstream_link_epochs 2 \
    --downstream_batch_size 256 \
    --edge_msg_pass_prop 0.5 0.5 0.5 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✓ Test 5 passed"
else
    echo "✗ Test 5 failed"
    ALL_TESTS_PASSED=false
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
if [ "$ALL_TESTS_PASSED" = true ]; then
    echo "✓ All 5 tests passed!"
    echo "GraphSSL pipeline is working correctly."
    exit 0
else
    echo "✗ Some tests failed. Check output above for details."
    exit 1
fi
