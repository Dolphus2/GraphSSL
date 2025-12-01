#!/bin/bash
#BSUB -J gssl_default
#BSUB -o logs/graphssl_%J.out
#BSUB -e logs/graphssl_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00 
#BSUB -B 
#BSUB -N 
#
# LSF script for running GraphSSL on DTU HPC with sensible defaults
# Default configuration: Supervised node classification with full downstream evaluation
# Submit from GraphSSL root directory: bsub < scripts/hpc/hpc_run.sh
#

echo "Starting GraphSSL Pipeline (Default Configuration)"
echo "=============================================="
echo "Job ID: $LSB_JOBID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load environment variables
source scripts/hpc/set_env.sh

# Load modules (adjust according to your HPC setup)
module purge
module load python3/3.12.9
module load cuda/12.0
module load cudnn/v8.3.2.44-prod-cuda-11.X

# Activate virtual environment
source ${venv_activate_path}

# Navigate to GraphSSL root directory
cd ${ROOT}

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

# Run the pipeline with sensible defaults
# Default: Supervised node classification with paper->field_of_study edges
python -m graphssl.main \
    --data_root data \
    --results_root results/hpc_default_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
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
    --patience 20 \
    --num_workers 4 \
    --weight_decay 0 \
    --log_interval 10 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 10 \
    --downstream_hidden_dim 128 \
    --downstream_num_layers 2 \
    --downstream_dropout 0.5 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 10 \
    --downstream_patience 20 \
    --edge_msg_pass_prop 0.8 0.8 0.8 \
    --seed 42

echo ""
echo "Job completed at: $(date)"
echo "=============================================="
