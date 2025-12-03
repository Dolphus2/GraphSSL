#!/bin/bash
mkdir -p logs/generate_metapath2vec

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/generate_metapath2vec/generate_metapath2vec_${TIMESTAMP}.log"

echo "========================================================================"
echo "Starting Generate Metapath2Vec"
echo "========================================================================"
echo ""
echo "Log file: $LOGFILE"
echo "Monitor: tail -f $LOGFILE"
echo "Check if running: ps aux | grep generate_metapath2vec"
echo ""
echo "Starting generate metapath2vec..."

# Run generate metapath2vec in background with nohup
nohup python src/graphssl/generate_metapath2vec.py \
  --root data \
  --embedding_dim 128 \
  --epochs 5 \
  --log_steps 10 \
  --save_every 1000 \
  --out_dir data/embeddings \
  --node_type paper > "$LOGFILE" 2>&1 &

# Get process ID
PID=$!

echo "Process ID: $PID"
echo ""
echo "Generate Metapath2Vec is now running in background."
echo "You can safely close this terminal."
echo ""
echo "To monitor progress:"
echo "  tail -f $LOGFILE"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To kill if needed:"
echo "  kill $PID"
echo ""
echo "========================================================================"
