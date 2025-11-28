#!/bin/bash
#
# Example script for running downstream evaluation
# This script demonstrates 5 different training + downstream configurations
# Run from GraphSSL root directory: bash scripts/run_downstream_examples.sh
#

echo "========================================="
echo "GraphSSL Downstream Examples - 5 Scenarios"
echo "========================================="

# Create directories if they don't exist
mkdir -p data
mkdir -p results

# Example 1: Supervised Node Classification + Downstream (field_of_study edges)
echo ""
echo "Example 1: Supervised Node Classification + Downstream (paper->field_of_study)"
echo "-------------------------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ds_ex1_supervised_node_fos \
    --objective_type supervised_node_classification \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 10 \
    --lr 0.001 \
    --patience 5 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_batch_size 1024 \
    --downstream_node_epochs 20 \
    --downstream_link_epochs 10 \
    --downstream_patience 5

echo ""
echo "========================================="

# Example 2: Supervised Link Prediction + Downstream (citation edges)
echo ""
echo "Example 2: Supervised Link Prediction + Downstream (paper->paper citations)"
echo "---------------------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ds_ex2_supervised_link_cites \
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
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_batch_size 1024 \
    --downstream_node_epochs 20 \
    --downstream_link_epochs 10 \
    --downstream_patience 5

echo ""
echo "========================================="

# Example 3: Self-Supervised Node (SCE) + Downstream (field_of_study edges)
echo ""
echo "Example 3: Self-Supervised Node (SCE) + Downstream (paper->field_of_study)"
echo "---------------------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ds_ex3_self_supervised_sce \
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
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_batch_size 1024 \
    --downstream_node_epochs 20 \
    --downstream_link_epochs 10 \
    --downstream_patience 5

echo ""
echo "========================================="

# Example 4: Self-Supervised Edge + Downstream (citation edges)
echo ""
echo "Example 4: Self-Supervised Edge + Downstream (paper citations)"
echo "---------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ds_ex4_self_supervised_edge \
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
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_batch_size 1024 \
    --downstream_node_epochs 20 \
    --downstream_link_epochs 10 \
    --downstream_patience 5

echo ""
echo "========================================="

# Example 5: Self-Supervised TAR + PFP + Downstream (field_of_study edges)
echo ""
echo "Example 5: Self-Supervised TAR + PFP + Downstream (paper->field_of_study)"
echo "--------------------------------------------------------------------------"
python -m graphssl.main \
    --data_root data \
    --results_root results/ds_ex5_tarpfp \
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
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_batch_size 1024 \
    --downstream_node_epochs 20 \
    --downstream_link_epochs 10 \
    --downstream_patience 5

echo ""
echo "========================================="
echo "All 5 downstream examples completed!"
echo "Results saved in results/ds_ex*/ directories"
echo "=========================================="
