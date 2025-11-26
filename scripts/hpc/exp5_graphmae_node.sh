#!/bin/bash
#BSUB -J exp5_graphmae
#BSUB -o logs/exp5_graphmae_node_%J.out
#BSUB -e logs/exp5_graphmae_node_%J.err
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
# Experiment 5: Self-Supervised Node Reconstruction (GraphMAE)
#
# Description:
#   Train a GraphSAGE encoder using self-supervised node feature reconstruction.
#   Uses GraphMAE's approach: mask node features and reconstruct them using
#   graph structure. The scaled cosine error (SCE) loss encourages the model
#   to learn feature directions rather than magnitudes.
#
# Method:
#   - Mask 50% of node features
#   - Use MLP decoder to reconstruct features from embeddings
#   - Optimize with Scaled Cosine Error (SCE) loss
#
# Outputs:
#   - Encoder 4 (node embeddings learned via self-supervision)
#   - Accuracy 5 (downstream node classification using Encoder 4)
#   - Accuracy 6 (downstream link prediction using Encoder 4)
#
# Submit: bsub < scripts/hpc/exp5_graphmae_node.sh
#

echo "========================================================================"
echo "Experiment 5: Self-Supervised Node Reconstruction (GraphMAE)"
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
    --results_root results/exp5_graphmae_node_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type self_supervised_node \
    --target_node paper \
    --loss_fn sce \
    --mask_ratio 0.5 \
    --use_feature_decoder \
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
    --downstream_epochs 100 \
    --seed 42

echo ""
echo "Experiment 5 completed at: $(date)"
echo "========================================================================"

