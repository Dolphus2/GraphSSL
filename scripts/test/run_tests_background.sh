#!/bin/bash
#
# Run all tests in background with logging
# Safe to exit SSH session after starting
#
# This script will:
# - Run all tests in background using nohup
# - Save output to logs/test_run_TIMESTAMP.log
# - Continue running even if you disconnect
#

# Create logs directory
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/test_run_${TIMESTAMP}.log"

echo "========================================================================"
echo "Starting Tests in Background"
echo "========================================================================"
echo ""
echo "Log file: $LOGFILE"
echo "Monitor: tail -f $LOGFILE"
echo "Check if running: ps aux | grep run_all_tests"
echo ""
echo "Starting tests..."

# Run tests in background with nohup
nohup bash scripts/test/run_all_tests.sh > "$LOGFILE" 2>&1 &

# Get process ID
PID=$!

echo "Process ID: $PID"
echo ""
echo "Tests are now running in background."
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

