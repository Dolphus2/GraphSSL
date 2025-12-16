#!/bin/bash
#
# Run all GraphSSL project experiments locally (no HPC specific commands)
echo "=============================================="
echo "GraphSSL - Complete Experiment Suite"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p results

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# ============================================
# Experiment 1: Supervised Node Classification
# ============================================
echo ""
echo "=============================================="
echo "Experiment 1: Supervised Node Classification"
echo "Training: Supervised learning for venue prediction"
echo "=============================================="
echo "Start time: $(date)"
echo ""

python -m graphssl.main \
    --data_root data \
    --results_root "results/exp_supervised_node_$(date +%Y%m%d_%H%M%S)" \
    --objective_type supervised_node_classification \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 5 \
    --num_workers 0 \
    --weight_decay 0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --multiclass_batch_size 256 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 1 \
    --downstream_patience 4 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42

echo ""
echo "Experiment 1 completed at: $(date)"

# ============================================
# Experiment 2: Supervised Link Prediction
# ============================================
echo ""
echo "=============================================="
echo "Experiment 2: Supervised Link Prediction"
echo "Training: Supervised learning for link prediction"
echo "=============================================="
echo "Start time: $(date)"
echo ""

python -m graphssl.main \
    --data_root data \
    --results_root "results/exp_supervised_link_$(date +%Y%m%d_%H%M%S)" \
    --objective_type supervised_link_prediction \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 5 \
    --num_workers 0 \
    --weight_decay 0 \
    --neg_sampling_ratio 1.0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --multiclass_batch_size 256 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 1 \
    --downstream_patience 4 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42

echo ""
echo "Experiment 2 completed at: $(date)"

# ============================================
# Experiment 3: Self-Supervised Node (SCE)
# ============================================
echo ""
echo "=============================================="
echo "Experiment 3: Self-Supervised Node Reconstruction (SCE)"
echo "Training: Masked node feature reconstruction with SCE loss"
echo "=============================================="
echo "Start time: $(date)"
echo ""

python -m graphssl.main \
    --data_root data \
    --results_root "results/exp_ssl_node_sce_$(date +%Y%m%d_%H%M%S)" \
    --objective_type self_supervised_node \
    --loss_fn sce \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --mask_ratio 0.5 \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 5 \
    --num_workers 0 \
    --weight_decay 0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --multiclass_batch_size 64 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 1 \
    --downstream_patience 4 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42

echo ""
echo "Experiment 3 completed at: $(date)"

# ============================================
# Experiment 4: Self-Supervised Node (MSE)
# ============================================
echo ""
echo "=============================================="
echo "Experiment 4: Self-Supervised Node Reconstruction (MSE)"
echo "Training: Masked node feature reconstruction with MSE loss"
echo "=============================================="
echo "Start time: $(date)"
echo ""

python -m graphssl.main \
    --data_root data \
    --results_root "results/exp_ssl_node_mse_$(date +%Y%m%d_%H%M%S)" \
    --objective_type self_supervised_node \
    --loss_fn mse \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --mask_ratio 0.5 \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 5 \
    --num_workers 0 \
    --weight_decay 0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --multiclass_batch_size 256 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 1 \
    --downstream_patience 4 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42

echo ""
echo "Experiment 4 completed at: $(date)"

# ============================================
# Experiment 5: Self-Supervised Edge
# ============================================
echo ""
echo "=============================================="
echo "Experiment 5: Self-Supervised Edge Reconstruction"
echo "Training: Edge reconstruction with BCE loss"
echo "=============================================="
echo "Start time: $(date)"
echo ""

python -m graphssl.main \
    --data_root data \
    --results_root "results/exp_ssl_edge_$(date +%Y%m%d_%H%M%S)" \
    --objective_type self_supervised_edge \
    --loss_fn bce \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --neg_sampling_ratio 1.0 \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 5 \
    --num_workers 0 \
    --weight_decay 0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --multiclass_batch_size 256 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 1 \
    --downstream_patience 4 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42

echo ""
echo "Experiment 5 completed at: $(date)"

# ============================================
# Experiment 6: Self-Supervised TAR
# ============================================
echo ""
echo "=============================================="
echo "Experiment 6: Self-Supervised TAR (Type-Aware Regularization)"
echo "Training: Type-aware regularization loss only"
echo "=============================================="
echo "Start time: $(date)"
echo ""

python -m graphssl.main \
    --data_root data \
    --results_root "results/exp_ssl_tar_$(date +%Y%m%d_%H%M%S)" \
    --objective_type self_supervised_tarpfp \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --lambda_tar 1.0 \
    --lambda_pfp 0.0 \
    --mask_ratio 0.5 \
    --neg_sampling_ratio 1.0 \
    --tar_temperature 0.5 \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 5 \
    --num_workers 0 \
    --weight_decay 0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --multiclass_batch_size 256 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 1 \
    --downstream_patience 4 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42

echo ""
echo "Experiment 6 completed at: $(date)"

# ============================================
# Experiment 7: Self-Supervised PFP
# ============================================
echo ""
echo "=============================================="
echo "Experiment 7: Self-Supervised PFP (Path Feature Prediction)"
echo "Training: Path feature prediction loss only"
echo "=============================================="
echo "Start time: $(date)"
echo ""

python -m graphssl.main \
    --data_root data \
    --metapath2vec_embeddings_path pos_embedding.pt \
    --results_root "results/exp_ssl_pfp_$(date +%Y%m%d_%H%M%S)" \
    --objective_type self_supervised_tarpfp \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --lambda_tar 0.0 \
    --lambda_pfp 1.0 \
    --mask_ratio 0.5 \
    --neg_sampling_ratio 1.0 \
    --tar_temperature 0.5 \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 5 \
    --num_workers 0 \
    --weight_decay 0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --multiclass_batch_size 256 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 1 \
    --downstream_patience 4 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42

echo ""
echo "Experiment 7 completed at: $(date)"

# ============================================
# Experiment 8: Self-Supervised TAR+PFP
# ============================================
echo ""
echo "=============================================="
echo "Experiment 8: Self-Supervised Combined (TAR + PFP)"
echo "Training: Combined type-aware regularization and path feature prediction"
echo "=============================================="
echo "Start time: $(date)"
echo ""

python -m graphssl.main \
    --data_root data \
    --metapath2vec_embeddings_path pos_embedding.pt \
    --results_root "results/exp_ssl_tarpfp_$(date +%Y%m%d_%H%M%S)" \
    --objective_type self_supervised_tarpfp \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --lambda_tar 1.0 \
    --lambda_pfp 1.0 \
    --mask_ratio 0.5 \
    --neg_sampling_ratio 1.0 \
    --tar_temperature 0.5 \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 5 \
    --num_workers 0 \
    --weight_decay 0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --multiclass_batch_size 256 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 1 \
    --downstream_patience 4 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42

echo ""
echo "Experiment 8 completed at: $(date)"

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "All Experiments Completed"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - results/exp_supervised_node_*"
echo "  - results/exp_supervised_link_*"
echo "  - results/exp_ssl_node_sce_*"
echo "  - results/exp_ssl_node_mse_*"
echo "  - results/exp_ssl_edge_*"
echo "  - results/exp_ssl_tar_*"
echo "  - results/exp_ssl_pfp_*"
echo "  - results/exp_ssl_tarpfp_*"
echo ""
