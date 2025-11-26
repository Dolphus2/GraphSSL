#!/bin/bash
# Example script for running downstream evaluation
# This demonstrates different ways to use the downstream evaluation pipeline

echo "=========================================="
echo "GraphSSL Downstream Evaluation Examples"
echo "=========================================="

# Example 1: Train with supervised learning + downstream evaluation
echo ""
echo "Example 1: Supervised Node Classification with Downstream Evaluation"
python -m graphssl.downstream_evaluation \
    --data_root data \
    --results_root results/quick_test \
    --objective_type supervised_node_classification \
    --loss_fn sce \
    --target_node "paper" \
    --target_edge_type "paper,cites,paper" \
    --downstream_lr 0.001 \
    --downstream_patience 5 \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 1 \
    --downstream_batch_size 512 \
    --downstream_epochs 5

exit 1


python -m graphssl.main \
    --data_root data \
    --results_root results/supervised_downstream \
    --objective_type supervised_node_classification \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 50 \
    --lr 0.001 \
    --patience 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_epochs 50

echo ""
echo "=========================================="

# Example 2: Self-supervised training + node property prediction only
echo ""
echo "Example 2: Self-Supervised Node Reconstruction + Node Downstream Task"
python -m graphssl.main \
    --data_root data \
    --results_root results/self_supervised_downstream \
    --objective_type self_supervised_node \
    --hidden_channels 256 \
    --num_layers 3 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --mask_ratio 0.5 \
    --use_feature_decoder \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task node \
    --downstream_n_runs 10

echo ""
echo "=========================================="

# Example 3: Link prediction training + link downstream evaluation
echo ""
echo "Example 3: Supervised Link Prediction + Link Downstream Task"
python -m graphssl.main \
    --data_root data \
    --results_root results/link_pred_downstream \
    --objective_type supervised_link_prediction \
    --target_edge_type "author,writes,paper" \
    --hidden_channels 128 \
    --num_layers 2 \
    --epochs 50 \
    --neg_sampling_ratio 1.0 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task link \
    --downstream_n_runs 10 \
    --downstream_neg_samples 2

echo ""
echo "=========================================="

# Example 4: Quick test with minimal runs
echo ""
echo "Example 4: Quick Downstream Test (Minimal Configuration)"
python -m graphssl.main \
    --data_root data \
    --results_root results/quick_downstream_test \
    --objective_type supervised_node_classification \
    --hidden_channels 64 \
    --num_layers 2 \
    --epochs 5 \
    --batch_size 512 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 3 \
    --downstream_epochs 10 \
    --downstream_batch_size 256

echo ""
echo "=========================================="
echo "All examples completed!"
echo "=========================================="
