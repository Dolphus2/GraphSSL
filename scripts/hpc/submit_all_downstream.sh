#!/bin/bash
#
# Submit all downstream evaluation experiments
# Usage: bash scripts/hpc/submit_all_downstream.sh
#

echo "Submitting all downstream evaluation experiments..."
echo "=============================================="
echo ""

# Submit each downstream experiment
bsub < scripts/hpc/downstream/downstream_ssl_node_sce.sh
echo "Submitted: downstream_ssl_node_sce"

bsub < scripts/hpc/downstream/downstream_ssl_node_mse.sh
echo "Submitted: downstream_ssl_node_mse"

bsub < scripts/hpc/downstream/downstream_ssl_edge.sh
echo "Submitted: downstream_ssl_edge"

bsub < scripts/hpc/downstream/downstream_ssl_tar.sh
echo "Submitted: downstream_ssl_tar"

bsub < scripts/hpc/downstream/downstream_ssl_pfp.sh
echo "Submitted: downstream_ssl_pfp"

bsub < scripts/hpc/downstream/downstream_ssl_tarpfp.sh
echo "Submitted: downstream_ssl_tarpfp"

bsub < scripts/hpc/downstream/downstream_supervised_node.sh
echo "Submitted: downstream_supervised_node"

bsub < scripts/hpc/downstream/downstream_supervised_link.sh
echo "Submitted: downstream_supervised_link"

echo ""
echo "=============================================="
echo "All 8 downstream experiments submitted!"
echo "Check status with: bstat"
