#!/bin/bash
#
# Quick test for Experiment 8: Combined Loss (Self-Supervised Edge)
# Runs on CPU with minimal settings to verify everything works
#

echo "========================================================================"
echo "Testing Experiment 8: Combined Loss (Self-Supervised Edge)"
echo "========================================================================"

cd /dtu/blackhole/1a/222842/GraphSSL
source .venv/bin/activate

python -m graphssl.main \
    --data_root data \
    --results_root results/test_exp8 \
    --objective_type self_supervised_edge \
    --target_edge_type paper,cites,paper \
    --loss_fn combined_loss \
    --mer_weight 1.0 \
    --tar_weight 1.0 \
    --pfp_weight 1.0 \
    --tar_temperature 0.5 \
    --use_edge_decoder \
    --neg_sampling_ratio 1.0 \
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
echo "* Experiment 8 test completed!"
echo "Check: results/test_exp8/"
echo "========================================================================"

