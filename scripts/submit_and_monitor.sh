#!/bin/bash
#
# Submit a SLURM job and automatically monitor its output
# Usage: ./submit_and_monitor.sh <script_path>
# Example: ./submit_and_monitor.sh lstm/test_lstm.sh
#

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_path>"
    echo ""
    echo "Examples:"
    echo "  $0 lstm/test_lstm.sh"
    echo "  $0 distilbert/test_distilbert.sh"
    echo "  $0 comparison/compare_all.sh"
    exit 1
fi

SCRIPT_PATH="$1"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Extract job name from the script
JOB_NAME=$(grep "^#SBATCH -J" "$SCRIPT_PATH" | head -1 | awk '{print $3}')

if [ -z "$JOB_NAME" ]; then
    echo "Error: Could not determine job name from script"
    exit 1
fi

# Determine log directory (always use absolute path to project root)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"

# Submit job and capture job ID
echo "Submitting job: $SCRIPT_PATH"
echo "----------------------------------------"
JOB_OUTPUT=$(sbatch --parsable "$SCRIPT_PATH" 2>&1)

if [ $? -ne 0 ]; then
    echo "Error submitting job:"
    echo "$JOB_OUTPUT"
    exit 1
fi

JOB_ID="$JOB_OUTPUT"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"

echo "✓ Job submitted: $JOB_ID"
echo "  Job name: $JOB_NAME"
echo "  Log file: $LOG_FILE"
echo ""
echo "Waiting for log file..."

# Wait for log file to be created (max 60 seconds)
COUNTER=0
while [ ! -f "$LOG_FILE" ] && [ $COUNTER -lt 60 ]; do
    sleep 1
    COUNTER=$((COUNTER + 1))
    
    # Show progress every 10 seconds
    if [ $((COUNTER % 10)) -eq 0 ]; then
        JOB_STATE=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
        if [ -n "$JOB_STATE" ]; then
            echo "  Waiting... (Job state: $JOB_STATE)"
        else
            echo "  Waiting... (Job may have completed)"
            break
        fi
    fi
done

if [ ! -f "$LOG_FILE" ]; then
    echo ""
    echo "Warning: Log file not created after 60 seconds"
    JOB_STATE=$(sacct -j $JOB_ID --format=State --noheader | head -1 | xargs)
    echo "Job state: $JOB_STATE"
    echo ""
    echo "The job may have completed very quickly or failed immediately."
    echo "Check with: cat $LOG_FILE"
    exit 0
fi

echo ""
echo "✓ Log file created, monitoring output..."
echo "=========================================="
echo "(Press Ctrl+C to stop monitoring)"
echo ""

# Monitor the log file
tail -f "$LOG_FILE"
