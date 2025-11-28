#!/bin/bash
#
# Example script to run different training objectives
# This script demonstrates 5 different training configurations
# Run from GraphSSL root directory: bash scripts/run_examples.sh
#

echo "========================================="
echo "GraphSSL Training Examples - 5 Scenarios"
echo "========================================="

# Create directories if they don't exist
mkdir -p data
mkdir -p results

# Example 1: Supervised Node Classification (field_of_study edges)
echo ""
echo "Example 1: Supervised Node Classification (paper->field_of_study)"
echo "-------------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ex1_supervised_node_fos \
    --objective_type supervised_node_classification \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 10 \
    --lr 0.001 \
    --patience 5 \
    --skip_downstream

echo ""
echo "========================================="

# Example 2: Supervised Link Prediction (citation edges)
echo ""
echo "Example 2: Supervised Link Prediction (paper->paper citations)"
echo "---------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ex2_supervised_link_cites \
    --objective_type supervised_link_prediction \
    --target_node "paper" \
    --target_edge_type "paper,cites,paper" \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 10 \
    --lr 0.001 \
    --neg_sampling_ratio 1.0 \
    --patience 5 \
    --skip_downstream

echo ""
echo "========================================="

# Example 3: Self-Supervised Node Reconstruction with SCE loss
echo ""
echo "Example 3: Self-Supervised Node (SCE loss, field_of_study edges)"
echo "------------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ex3_self_supervised_sce \
    --objective_type self_supervised_node \
    --loss_fn sce \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --mask_ratio 0.5 \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 10 \
    --lr 0.001 \
    --patience 5 \
    --skip_downstream

echo ""
echo "========================================="

# Example 4: Self-Supervised Link Prediction (citation edges)
echo ""
echo "Example 4: Self-Supervised Edge Reconstruction (paper citations)"
echo "------------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ex4_self_supervised_edge \
    --objective_type self_supervised_edge \
    --target_node "paper" \
    --target_edge_type "paper,cites,paper" \
    --neg_sampling_ratio 1.0 \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 10 \
    --lr 0.001 \
    --patience 5 \
    --skip_downstream

echo ""
echo "========================================="

# Example 5: Self-Supervised TAR + PFP (combined losses, field_of_study edges)
echo ""
echo "Example 5: Self-Supervised TAR + PFP (paper->field_of_study)"
echo "--------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ex5_tarpfp \
    --objective_type self_supervised_tarpfp \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --lambda_tar 1.0 \
    --lambda_pfp 1.0 \
    --mask_ratio 0.5 \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 10 \
    --lr 0.001 \
    --patience 5 \
    --skip_downstream

echo ""
echo "========================================="
echo "All 5 training examples completed!"
echo "Results saved in results/ex*/ directories"
echo "========================================="
