#!/bin/bash
#BSUB -J graphssl_supervised
#BSUB -o logs/graphssl_%J.out
#BSUB -e logs/graphssl_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 24:00
#BSUB -N
#
# LSF script for running GraphSSL supervised learning on DTU HPC
# Adjust the parameters above according to your needs
#

echo "Starting GraphSSL Supervised Learning Pipeline"
echo "=============================================="
echo "Job ID: $LSB_JOBID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load modules (adjust according to your HPC setup)
# module load python3/3.10.12
# module load cuda/12.6

# Activate virtual environment if using one
# source /path/to/your/venv/bin/activate

# Create necessary directories
mkdir -p logs
mkdir -p ../data
mkdir -p ../results

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# Navigate to source directory
cd /zhome/5c/0/167753/DTU/E2025/GraphSSL/src

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Run the supervised learning pipeline
python main.py \
    --data_root ../data \
    --results_root ../results/hpc_run_$(date +%Y%m%d_%H%M%S) \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --num_neighbors 15 10 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 10 \
    --num_workers 4 \
    --extract_embeddings \
    --seed 42

echo ""
echo "Job completed at: $(date)"
echo "=============================================="
