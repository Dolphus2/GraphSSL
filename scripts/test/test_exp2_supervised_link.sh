#!/bin/bash
#
# Quick test for Experiment 2: Supervised Link Prediction
# Runs on CPU with minimal settings to verify everything works
#

echo "========================================================================"
echo "Testing Experiment 2: Supervised Link Prediction"
echo "========================================================================"

cd /dtu/blackhole/1a/222842/GraphSSL
source .venv/bin/activate

python -m graphssl.main \
    --data_root data \
    --results_root results/test_exp2 \
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
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 2 \
    --downstream_hidden_dim 32 \
    --downstream_node_epochs 3 \
    --downstream_link_epochs 3 \
    --downstream_batch_size 128 \
    --test_mode \
    --seed 42

echo ""
echo "* Experiment 2 test completed!"
echo "Check: results/test_exp2/"
echo "========================================================================"

