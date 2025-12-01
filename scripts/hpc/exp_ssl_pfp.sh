#!/bin/bash
#BSUB -J gssl_ssl_pfp
#BSUB -o logs/exp_ssl_pfp_%J.out
#BSUB -e logs/exp_ssl_pfp_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00 
#BSUB -B 
#BSUB -N 
#
# Experiment: Self-Supervised PFP (Path Feature Prediction)
# Training: Path feature prediction loss only (PFP component)
# Encoder: GraphSAGE encoder with position decoder
# Submit: bsub < scripts/hpc/exp_ssl_pfp.sh
#

echo "Starting Experiment: Self-Supervised PFP (Path Feature Prediction)"
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
    --results_root results/exp_ssl_pfp_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type self_supervised_tarpfp \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --mer_weight 0.0 \
    --tar_weight 1.0 \
    --pfp_weight 1.0 \
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
    --downstream_link_epochs 5 \
    --downstream_patience 20 \
    --downstream_lr 0.0001 \
    --edge_msg_pass_prop 0.8 0.8 0.8 \
    --seed 42 \
    --disable_tqdm

echo ""
echo "Job completed at: $(date)"
echo "=============================================="
