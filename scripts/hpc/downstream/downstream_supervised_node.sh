#!/bin/bash
#BSUB -J gssl_down_supnode
#BSUB -o logs/downstream_supervised_node_%J.out
#BSUB -e logs/downstream_supervised_node_%J.err
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 24:00 
#BSUB -B 
#BSUB -N 
#
# Downstream Evaluation: Supervised Node Classification
# Runs only multiclass link prediction downstream task
# Submit: bsub < scripts/hpc/downstream/downstream_supervised_node.sh
#

echo "Starting Downstream Evaluation: Supervised Node Classification"
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

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Run downstream evaluation
python -m graphssl.downstream_evaluation \
    --data_root data \
    --results_root results/downstream_supervised_node_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --model_path results/exp_supervised_node_27276446_20251204_195325/model_supervised_node_classification.pt \
    --objective_type supervised_node_classification \
    --target_node "paper" \
    --target_edge_type "paper,has_topic,field_of_study" \
    --use_feature_decoder \
    --use_edge_decoder \
    --hidden_channels 128 \
    --num_layers 2 \
    --num_neighbors 30 30 \
    --batch_size 1024 \
    --num_workers 4 \
    --downstream_task multiclass_link \
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
    --seed 42 \
    --disable_tqdm

echo ""
echo "Job completed at: $(date)"
echo "=============================================="
