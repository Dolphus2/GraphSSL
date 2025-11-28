#!/bin/bash
#BSUB -J exp8_combined
#BSUB -o logs/exp8_combined_loss_%J.out
#BSUB -e logs/exp8_combined_loss_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 24:00
##BSUB -u s252683@dtu.dk
#BSUB -B
#BSUB -N
#
# Experiment 8: Self-Supervised Edge Reconstruction (Combined Loss)
#
# Description:
#   Train a GraphSAGE encoder using self-supervised edge reconstruction with
#   a combined loss function inspired by HGMAE. This approach learns embeddings
#   by predicting edges while preserving both topology and feature information.
#
# Method:
#   - Combined loss = MER + TAR + PFP
#   - MER (Masked Edge Reconstruction): Standard edge prediction loss
#   - TAR (Topology-Aware Reconstruction): Contrastive loss for topology
#   - PFP (Preference-based Feature Propagation): Feature similarity preservation
#   - Use MLP decoder for edge prediction
#
# Loss Components:
#   1. MER: Binary cross-entropy for edge existence
#   2. TAR: Contrastive learning - maximize similarity for positive edges,
#           minimize for negative edges (temperature-scaled)
#   3. PFP: Preserve feature similarity between connected nodes
#
# Hyperparameters:
#   - Loss weights (all 1.0 for balanced contribution)
#   - TAR temperature (0.5 for moderate contrastive hardness)
#
# Outputs:
#   - Encoder 5 (node embeddings learned via combined self-supervised loss)
#   - Accuracy 7 (downstream node classification using Encoder 5)
#   - Accuracy 8 (downstream link prediction using Encoder 5)
#
# Submit: bsub < scripts/hpc/exp8_combined_loss_edge.sh
#

echo "========================================================================"
echo "Experiment 8: Self-Supervised Edge Reconstruction (Combined Loss)"
echo "========================================================================"
echo "Job ID: $LSB_JOBID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load modules
module purge
module load python3/3.12.9
module load cuda/12.0
module load cudnn/v8.3.2.44-prod-cuda-11.X

# Activate virtual environment
source /dtu/blackhole/1a/222842/GraphSSL/.venv/bin/activate

# Navigate to project root
cd /dtu/blackhole/1a/222842/GraphSSL

# Create necessary directories
mkdir -p logs
mkdir -p results

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Run experiment
python -m graphssl.main \
    --data_root data \
    --results_root results/exp8_combined_loss_edge_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type self_supervised_edge \
    --target_edge_type paper,cites,paper \
    --loss_fn combined_loss \
    --mer_weight 1.0 \
    --tar_weight 1.0 \
    --pfp_weight 1.0 \
    --tar_temperature 0.5 \
    --use_edge_decoder \
    --neg_sampling_ratio 1.0 \
    --hidden_channels 128 \
    --num_layers 2 \
    --dropout 0.5 \
    --batch_size 1024 \
    --epochs 1000 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --patience 100 \
    --num_workers 4 \
    --log_interval 20 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 10 \
    --downstream_hidden_dim 128 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 10 \
    --seed 42

echo ""
echo "Experiment 8 completed at: $(date)"
echo "========================================================================"

