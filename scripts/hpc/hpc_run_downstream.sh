#!/bin/bash
#BSUB -J gssl_downstream
#BSUB -o logs/graphssl_downstream_%J.out
#BSUB -e logs/graphssl_downstream_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00 
#BSUB -B 
#BSUB -N 
#
# LSF script for running GraphSSL downstream evaluation on DTU HPC
# Adjust the parameters above according to your needs
# Submit from GraphSSL root directory: bsub < scripts/hpc/hpc_run_downstream.sh
#

echo "Starting GraphSSL Downstream Evaluation Pipeline"
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
source /dtu/blackhole/09/167753/venvs/graphssl/bin/activate

# Navigate to GraphSSL root directory
cd /dtu/blackhole/09/167753/GraphSSL

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

echo "=========================================="
echo "GraphSSL Downstream Evaluation Examples"
echo "=========================================="

# Example 1: Train with supervised learning + downstream evaluation
echo ""
echo "Example 1: Supervised Node Classification with Downstream Evaluation"
python -m graphssl.main \
    --data_root data \
    --results_root results/hpc_supervised_downstream_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type supervised_node_classification \
    --hidden_channels 128 \
    --num_layers 2 \
    --batch_size 1024 \
    --epochs 50 \
    --lr 0.001 \
    --patience 10 \
    --num_workers 4 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 5 \
    --downstream_hidden_dim 128 \
    --downstream_epochs 50

echo ""
echo "=========================================="

# Example 2: Self-supervised training + node property prediction only
echo ""
echo "Example 2: Self-Supervised Node Reconstruction + Node Downstream Task"
python -m graphssl.main \
    --data_root data \
    --results_root results/hpc_self_supervised_downstream_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type self_supervised_node \
    --hidden_channels 256 \
    --num_layers 3 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --mask_ratio 0.5 \
    --use_feature_decoder \
    --num_workers 4 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task node \
    --downstream_n_runs 10

echo ""
echo "=========================================="

# Example 3: Link prediction training + link downstream evaluation
echo ""
echo "Example 3: Supervised Link Prediction + Link Downstream Task"
python -m graphssl.main \
    --data_root data \
    --results_root results/hpc_link_pred_downstream_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type supervised_link_prediction \
    --target_edge_type "author,writes,paper" \
    --hidden_channels 128 \
    --num_layers 2 \
    --epochs 50 \
    --neg_sampling_ratio 1.0 \
    --num_workers 4 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task link \
    --downstream_n_runs 10 \
    --downstream_neg_samples 2

echo ""
echo "=========================================="

# Example 4: Link prediction training + link downstream evaluation
echo ""
echo "Example 3: Self Supervised Link Reconstruction + Both Downstream Task"
python -m graphssl.main \
    --data_root data \
    --results_root results/hpc_link_pred_downstream_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type self_supervised_edge \
    --target_edge_type "author,writes,paper" \
    --hidden_channels 128 \
    --num_layers 2 \
    --epochs 50 \
    --neg_sampling_ratio 1.0 \
    --num_workers 4 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 10 \
    --downstream_neg_samples 2

echo ""
echo "=========================================="

# Example 5: Quick test with minimal runs
echo ""
echo "Example 5: Quick Downstream Test (Minimal Configuration)"
python -m graphssl.main \
    --data_root data \
    --results_root results/hpc_quick_downstream_test_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S) \
    --objective_type supervised_node_classification \
    --hidden_channels 64 \
    --num_layers 2 \
    --epochs 5 \
    --batch_size 512 \
    --num_workers 4 \
    --extract_embeddings \
    --downstream_eval \
    --downstream_task both \
    --downstream_n_runs 3 \
    --downstream_epochs 10 \
    --downstream_batch_size 256

echo ""
echo "Job completed at: $(date)"
echo "=========================================="
echo "All examples completed!"
echo "=========================================="

