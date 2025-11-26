#!/bin/bash
#
# Submit All Experiments to HPC
#
# This script submits all 4 main experiments as separate jobs.
# Each experiment will run on its own GPU in parallel.
#
# Usage:
#   bash scripts/hpc/submit_all_experiments.sh
#
# Or submit individually:
#   bsub < scripts/hpc/exp1_supervised_node.sh
#   bsub < scripts/hpc/exp2_supervised_link.sh
#   bsub < scripts/hpc/exp5_graphmae_node.sh
#   bsub < scripts/hpc/exp8_combined_loss_edge.sh
#

echo "========================================================================="
echo "Submitting All GraphSSL Experiments to HPC"
echo "========================================================================="
echo ""

# Change to project root
cd /dtu/blackhole/1a/222842/GraphSSL

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Submitting experiments..."
echo ""

# Submit Experiment 1
echo "[1/4] Submitting Experiment 1: Supervised Node Classification"
JOB1=$(bsub < scripts/hpc/exp1_supervised_node.sh 2>&1 | grep -oP 'Job <\K[0-9]+')
echo "      Job ID: $JOB1"
echo ""

# Submit Experiment 2
echo "[2/4] Submitting Experiment 2: Supervised Link Prediction"
JOB2=$(bsub < scripts/hpc/exp2_supervised_link.sh 2>&1 | grep -oP 'Job <\K[0-9]+')
echo "      Job ID: $JOB2"
echo ""

# Submit Experiment 5
echo "[3/4] Submitting Experiment 5: GraphMAE (Self-Supervised Node)"
JOB5=$(bsub < scripts/hpc/exp5_graphmae_node.sh 2>&1 | grep -oP 'Job <\K[0-9]+')
echo "      Job ID: $JOB5"
echo ""

# Submit Experiment 8
echo "[4/4] Submitting Experiment 8: Combined Loss (Self-Supervised Edge)"
JOB8=$(bsub < scripts/hpc/exp8_combined_loss_edge.sh 2>&1 | grep -oP 'Job <\K[0-9]+')
echo "      Job ID: $JOB8"
echo ""

echo "========================================================================="
echo "All experiments submitted!"
echo "========================================================================="
echo ""
echo "Job Summary:"
echo "  Experiment 1 (Supervised Node):       Job $JOB1"
echo "  Experiment 2 (Supervised Link):       Job $JOB2"
echo "  Experiment 5 (GraphMAE):              Job $JOB5"
echo "  Experiment 8 (Combined Loss):         Job $JOB8"
echo ""
echo "Check job status:"
echo "  bstat"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/exp1_supervised_node_${JOB1}.out"
echo "  tail -f logs/exp2_supervised_link_${JOB2}.out"
echo "  tail -f logs/exp5_graphmae_node_${JOB5}.out"
echo "  tail -f logs/exp8_combined_loss_${JOB8}.out"
echo ""
echo "Kill jobs if needed:"
echo "  bkill $JOB1 $JOB2 $JOB5 $JOB8"
echo ""
echo "========================================================================="

