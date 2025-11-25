#!/bin/bash
#BSUB -J gssl_self_e
#BSUB -o logs/graphssl_%J.out
#BSUB -e logs/graphssl_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00 
##BSUB -u s252683@dtu.dk
#BSUB -B 
#BSUB -N 
#
# LSF script for running GraphSSL supervised learning on DTU HPC
# Adjust the parameters above according to your needs
# Submit from GraphSSL root directory: bsub < scripts/hpc/hpc_run.sh
#

echo "Starting GraphSSL Supervised Learning Pipeline"
echo "=============================================="
echo "Job ID: $LSB_JOBID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load modules (adjust according to your HPC setup)
module purge
module load python3/3.12.9
module load cuda/12.0
module load cudnn/v8.3.2.44-prod-cuda-11.X

# Activate virtual environment if using one
source /dtu/blackhole/1a/222842/GraphSSL/.venv/bin/activate

# Navigate to GraphSSL root directory
cd /dtu/blackhole/1a/222842/GraphSSL


# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p results

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Run the supervised learning pipeline
python -m graphssl.main \
    --data_root data \
    --results_root results/hpc_run5_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type supervised_node_classification \
    --loss_fn sce \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 1000 \
    --lr 0.001 \
    --dropout 0.5 \
    --patience 100 \
    --num_workers 4 \
    --weight_decay 0.0005 \
    --log_interval 20 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 50 \
    --downstream_hidden_dim 128 \
    --downstream_epochs 100 \
    --seed 42

echo ""
echo "Job completed at: $(date)"
echo "=============================================="
