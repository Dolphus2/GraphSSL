#!/bin/bash
# Example script for running downstream evaluation
# This demonstrates different ways to use the downstream evaluation pipeline

echo "=========================================="
echo "GraphSSL Downstream Evaluation Examples"
echo "=========================================="

# Example 1: Train with supervised learning + downstream evaluation
echo ""
echo "Example 1: Supervised Node Classification with Downstream Evaluation"
python -m graphssl.downstream_evaluation  \
    --data_root data \
    --results_root results/quick_test5 \
    --objective_type supervised_node_classification \
    --loss_fn sce \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --epochs 1 \
    --downstream_lr 0.001 \
    --downstream_patience 5 \
    --downstream_task multiclass_link \
    --downstream_n_runs 1 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 1 \
    --downstream_batch_size 512 \
    --downstream_node_epochs 1 \
    --downstream_link_epochs 1 \
    --edge_msg_pass_prop 0.8 0.8 0.8 
exit 1