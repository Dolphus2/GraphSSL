#!/bin/bash
#
# Example script to run the supervised learning pipeline on HPC
# This script demonstrates different configurations for training
# Run from GraphSSL root directory: bash src/graphssl/run_examples.sh
#
pwd

echo "GraphSSL Supervised Learning Pipeline - Example Runs"
echo "======================================================"

# Create directories if they don't exist
mkdir -p data
mkdir -p results

# Example 1: Quick test run with small model
echo ""
echo "Example 1: Quick test run (small model, few epochs)"
echo "----------------------------------------------------"

python -m graphssl.main \
    --data_root data \
    --results_root results/quick_test \
    --objective_type self_supervised_node \
    --loss_fn sce \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 10240 \
    --epochs 1 \
    --lr 0.001 \
    --patience 5 \
    --extract_embeddings

# # Example 2: Standard training configuration
# echo ""
# echo "Example 2: Standard training configuration"
# echo "-------------------------------------------"
# python -m graphssl.main \
#     --data_root data \
#     --results_root results/standard \
#     --hidden_channels 128 \
#     --num_layers 2 \
#     --batch_size 1024 \
#     --epochs 100 \
#     --lr 0.001 \
#     --dropout 0.5 \
#     --patience 10 \
#     --extract_embeddings

# # Example 3: Large model with more capacity
# echo ""
# echo "Example 3: Large model configuration"
# echo "-------------------------------------"
# python -m graphssl.main \
#     --data_root data \
#     --results_root results/large_model \
#     --hidden_channels 256 \
#     --num_layers 3 \
#     --batch_size 512 \
#     --epochs 100 \
#     --lr 0.001 \
#     --dropout 0.5 \
#     --weight_decay 1e-5 \
#     --patience 15 \
#     --extract_embeddings

# # Example 4: Smaller batch size for memory-constrained environments
# echo ""
# echo "Example 4: Memory-efficient configuration"
# echo "------------------------------------------"
# python -m graphssl.main \
#     --data_root data \
#     --results_root results/memory_efficient \
#     --hidden_channels 128 \
#     --num_layers 2 \
#     --batch_size 256 \
#     --num_neighbors 10 5 \
#     --epochs 100 \
#     --lr 0.001 \
#     --dropout 0.5 \
#     --patience 10

# echo ""
# echo "======================================================"
# echo "All example runs completed!"
# echo "Check the results directory for outputs."
