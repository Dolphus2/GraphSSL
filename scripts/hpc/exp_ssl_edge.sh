#!/bin/bash
#BSUB -J gssl_ssl_edge
#BSUB -o logs/exp_ssl_edge_%J.out
#BSUB -e logs/exp_ssl_edge_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00 
#BSUB -B 
#BSUB -N 
#
# Experiment 4: Self-Supervised Edge Reconstruction
# Training: Edge reconstruction with BCE loss (link prediction pretext task)
# Encoder: GraphSAGE encoder with edge decoder
# Submit: bsub < scripts/hpc/exp_ssl_edge.sh
#

echo "Starting Experiment: Self-Supervised Edge Reconstruction"
echo "=============================================="
echo "Job ID: $LSB_JOBID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load environment variables
source scripts/hpc/set_env.sh

# Load modules
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

# Run experiment
python -m graphssl.main \
    --data_root data \
    --results_root results/exp_ssl_edge_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
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
    --multiclass_batch_size 256 \
    --downstream_node_epochs 100 \
    --downstream_link_epochs 2 \
    --downstream_patience 20 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0 0 0 \
    --seed 42 \
    --disable_tqdm

echo ""
echo "Job completed at: $(date)"
echo "=============================================="
