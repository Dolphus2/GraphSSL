#!/bin/bash
# Test script for TAR+PFP self-supervised learning
# Tests automatic MetaPath2Vec embedding generation and multi-task learning

echo "=========================================="
echo "GraphSSL SCE Test"
echo "=========================================="

# Test TAR+PFP with automatic positional embedding generation
echo ""
echo "Testing TAR+PFP Self-Supervised Learning with Downstream Evaluation"
python -m graphssl.main  \
    --data_root data \
    --results_root results/quick_test_sce \
    --objective_type self_supervised_node \
    --loss_fn sce \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --lambda_tar 0.7 \
    --lambda_pfp 0.3 \
    --metapath2vec_embeddings_path "pos_embedding.pt" \
    --epochs 1 \
    --hidden_channels 128 \
    --downstream_lr 0.001 \
    --downstream_patience 5 \
    --downstream_task both \
    --downstream_n_runs 2 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 5 \
    --edge_msg_pass_prop 0.8 0.8 0.8 \
    --test_mode

echo ""
echo "=========================================="
echo "Test completed. Check results/quick_test_tarpfp for outputs."
echo "=========================================="