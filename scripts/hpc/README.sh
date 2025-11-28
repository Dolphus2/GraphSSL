#!/bin/bash
#
# GraphSSL HPC Experiments - Quick Reference
# ==========================================
#
# This directory contains 5 main experiment configurations for GraphSSL.
# All experiments use sensible defaults and include full downstream evaluation.
#

echo "GraphSSL HPC Experiments"
echo "========================"
echo ""
echo "Available Experiments:"
echo "  1. exp_supervised_node.sh    - Supervised Node Classification"
echo "  2. exp_supervised_link.sh    - Supervised Link Prediction"
echo "  3. exp_ssl_node_sce.sh       - Self-Supervised Node (SCE loss)"
echo "  4. exp_ssl_edge.sh           - Self-Supervised Edge Reconstruction"
echo "  5. exp_ssl_tarpfp.sh         - Self-Supervised Combined (MER+TAR+PFP)"
echo ""
echo "Additional Scripts:"
echo "  - hpc_run.sh                 - Default supervised baseline"
echo "  - hpc_run_downstream.sh      - Self-supervised with downstream eval"
echo "  - set_env.sh                 - Environment configuration (EDIT THIS!)"
echo ""
echo "Documentation:"
echo "  - EXPERIMENTS_MANIFEST.md    - Full experiment descriptions"
echo ""
echo "Quick Start:"
echo "  1. Edit set_env.sh with your HPC paths"
echo "  2. Submit individual experiment:"
echo "     bsub < scripts/hpc/exp_supervised_node.sh"
echo "  3. Submit all experiments:"
echo "     for script in scripts/hpc/exp_*.sh; do bsub < \$script; sleep 2; done"
echo ""
echo "Check Status:"
echo "  bjobs                # View all jobs"
echo "  bjobs -l <jobid>     # Detailed job info"
echo "  bpeek <jobid>        # View stdout"
echo ""
echo "For more details, see EXPERIMENTS_MANIFEST.md"
echo ""
