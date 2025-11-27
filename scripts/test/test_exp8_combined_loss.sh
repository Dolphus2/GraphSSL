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
    --metapath2vec_embeddings_path paper_emb_epoch3_step13000.pt \
    --results_root results/test_exp8 \
    --objective_type self_supervised_tarpfp \
    --target_edge_type paper,cites,paper \
    --use_edge_decoder \
    --hidden_channels 32 \
    --num_layers 2 \
    --dropout 0.3 \
    --batch_size 128 \
    --epochs 50 \
    --lr 0.01 \
    --weight_decay 0.0 \
    --patience 10 \
    --lambda_tar 1.0 \
    --lambda_pfp 2.0 \
    --num_workers 2 \
    --log_interval 5 \
    --metric_for_best loss \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 2 \
    --downstream_hidden_dim 32 \
    --downstream_epochs 3 \
    --downstream_batch_size 128 \
    --test_mode \
    --test_max_nodes 20000 \
    --seed 42

echo ""
echo "* Experiment 8 test completed!"
echo "Check: results/test_exp8/"
echo "========================================================================"

