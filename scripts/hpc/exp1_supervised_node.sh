#!/bin/bash
#BSUB -J exp1_sup_node
#BSUB -o logs/exp1_supervised_node_%J.out
#BSUB -e logs/exp1_supervised_node_%J.err
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
# Experiment 1: Supervised Node Classification
# 
# Description:
#   Train a GraphSAGE encoder using supervised node classification.
#   This learns node embeddings by directly predicting node labels.
#
# Outputs:
#   - Encoder 1 (node embeddings)
#   - Accuracy 1 (test accuracy on node classification)
#   - Accuracy 3 (downstream link prediction using Encoder 1)
#
# Submit: bsub < scripts/hpc/exp1_supervised_node.sh
#

echo "========================================================================"
echo "Experiment 1: Supervised Node Classification"
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
    --results_root results/exp1_supervised_node_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type supervised_node_classification \
    --target_node paper \
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
echo "Experiment 1 completed at: $(date)"
echo "========================================================================"

