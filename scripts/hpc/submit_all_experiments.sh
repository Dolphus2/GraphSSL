#!/bin/bash
#
# Submit all 8 experiments to HPC
# Usage: bash scripts/hpc/submit_all_experiments.sh [GPU_QUEUE] [USE_MSG_PASSING]
# Example: bash scripts/hpc/submit_all_experiments.sh gpul40s 1
#
# Arguments:
#   GPU_QUEUE       - GPU queue to use (default: gpul40s)
#   USE_MSG_PASSING - 0 for no message passing (0 0 0), 1 for message passing (0.8 0.8 0.8) (default: 0)
#
# Recommended queues:
#   gpul40s  - 16 free slots, 0 queue, L40S (48GB, Ada architecture) [BEST]
#   gpua40   - 8 slots, 0 queue, A40 (48GB, Ampere)
#   gpua10   - Variable availability, A10 (24GB, Ampere)
#

# Set GPU queue (default: gpul40s for best performance and availability)
GPU_QUEUE="${1:-gpul40s}"

# Set edge message passing proportion (default: 0 = no message passing)
USE_MSG_PASSING="${2:-0}"

if [ "$USE_MSG_PASSING" -eq 1 ]; then
    EDGE_MSG_PASS="0.8 0.8 0.8"
    MSG_LABEL="with message passing (0.8 0.8 0.8)"
else
    EDGE_MSG_PASS="0 0 0"
    MSG_LABEL="without message passing (0 0 0)"
fi

echo "Submitting all experiments to HPC..."
echo "GPU Queue: $GPU_QUEUE"
echo "Message Passing: $MSG_LABEL"
echo "===================================="
echo ""

# Temporary directory for modified scripts
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to submit experiment with modified GPU queue
submit_experiment() {
    local script_name="$1"
    local exp_name="$2"
    local script_path="scripts/hpc/$script_name"
    local temp_script="$TEMP_DIR/$script_name"
    
    # Copy and modify the queue and edge_msg_pass_prop in the script
    sed -e "s/#BSUB -q [a-z0-9]*/#BSUB -q $GPU_QUEUE/g" \
        -e "s/--edge_msg_pass_prop [0-9.]* [0-9.]* [0-9.]*$/--edge_msg_pass_prop $EDGE_MSG_PASS \\\\/g" \
        "$script_path" > "$temp_script"
    
    echo "$exp_name"
    bsub < "$temp_script"
}

# Submit each experiment
submit_experiment "exp_supervised_node.sh" "1/8 Submitting: Supervised Node Classification"
submit_experiment "exp_supervised_link.sh" "2/8 Submitting: Supervised Link Prediction"
submit_experiment "exp_ssl_edge.sh" "3/8 Submitting: Self-Supervised Edge Reconstruction"
submit_experiment "exp_ssl_node_sce.sh" "4/8 Submitting: Self-Supervised Node SCE"
submit_experiment "exp_ssl_node_mse.sh" "5/8 Submitting: Self-Supervised Node MSE"
submit_experiment "exp_ssl_tar.sh" "6/8 Submitting: Self-Supervised TAR"
submit_experiment "exp_ssl_tarpfp.sh" "7/8 Submitting: Self-Supervised TAR+PFP"
submit_experiment "exp_ssl_pfp.sh" "8/8 Submitting: Self-Supervised PFP"

echo ""
echo "===================================="
echo "All experiments submitted to $GPU_QUEUE!"
echo ""
echo "Check status with: bjobs"
echo "Check queue with: bqueues | grep $GPU_QUEUE"
