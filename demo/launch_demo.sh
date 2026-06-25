#!/bin/bash
# Usage: sbatch launch_demo.sh
# Or: bash launch_demo.sh (for interactive)
#
# Launches evaluation of V0 and V4.9 large on the 3 demo benchmarks.
# Results saved as checkpoint_v0_4400.json / checkpoint_v49_3200.json in demo/eval/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env for env vars
set -a
source "$PROJECT_ROOT/.env"
set +a

echo "==========================================="
echo "Launching Demo: V4.9 Large vs V0 (3 benches)"
echo "==========================================="
echo "Benchmarks:"
echo "  1. matmul_256_768_128                       (matmul)"
echo "  2. conv_2d_nchw_fchw_128_240_7_7_256_1_1_4_4 (conv)"
echo "  3. pooling_nchw_max_256_512_120_120_1_60_60  (pooling)"
echo ""
echo "Baseline times: $(cat $SCRIPT_DIR/bench_eval_base.json | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(f"  {k}: {v/1e6:.2f}ms") for k,v in d.items()]')"
echo ""

# Use EVAL_LABEL to avoid overwriting existing checkpoint files
export EVAL_LABEL="v0"  # or set per-job

# Submit V0 eval (checkpoint 4400)
echo ">>> Submitting V0 eval (checkpoint 4400)..."
V0_JOB=$(sbatch --parsable \
  --job-name="demo-v0" \
  --cpus-per-task=28 --mem=64G \
  --output="$PROJECT_ROOT/logs/demo_v0_%j.out" \
  "$PROJECT_ROOT/scripts/eval/eval.sh" \
  "$SCRIPT_DIR/eval_v0.json" \
  --checkpoint 4400)
echo "    V0 job ID: $V0_JOB"

# Submit V4.9 large eval (checkpoint 3200)
echo ">>> Submitting V4.9 large eval (checkpoint 3200)..."
V49_JOB=$(sbatch --parsable \
  --job-name="demo-v49" \
  --cpus-per-task=28 --mem=64G \
  --output="$PROJECT_ROOT/logs/demo_v49_%j.out" \
  "$PROJECT_ROOT/scripts/eval/eval.sh" \
  "$SCRIPT_DIR/eval_v49.json" \
  --checkpoint 3200)
echo "    V4.9 job ID: $V49_JOB"

echo ""
echo "After both jobs complete, results will be at:"
echo "  V0:    $PROJECT_ROOT/demo/eval/checkpoint_v0_4400.json"
echo "  V4.9:  $PROJECT_ROOT/demo/eval/checkpoint_v49_3200.json"
echo ""
echo "To compare results, run:"
echo "  python3 $SCRIPT_DIR/compare_demo.py"
