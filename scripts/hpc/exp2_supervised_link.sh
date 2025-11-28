#!/bin/bash
#BSUB -J exp2_sup_link
#BSUB -o logs/exp2_supervised_link_%J.out
#BSUB -e logs/exp2_supervised_link_%J.err
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
# Experiment 2: Supervised Link Prediction
#
# Description:
#   Train a GraphSAGE encoder using supervised link prediction.
#   This learns node embeddings by predicting which edges exist.
#
# Outputs:
#   - Encoder 2 (node embeddings)
#   - Accuracy 2 (test accuracy on link prediction)
#   - Accuracy 4 (downstream node classification using Encoder 2)
#
# Submit: bsub < scripts/hpc/exp2_supervised_link.sh
#

echo "========================================================================"
echo "Experiment 2: Supervised Link Prediction"
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
    --results_root results/exp2_supervised_link_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type supervised_link_prediction \
    --target_edge_type paper,cites,paper \
    --hidden_channels 128 \
    --num_layers 2 \
    --dropout 0.5 \
    --batch_size 1024 \
    --epochs 1000 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --patience 100 \
    --neg_sampling_ratio 1.0 \
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
echo "Experiment 2 completed at: $(date)"
echo "========================================================================"

