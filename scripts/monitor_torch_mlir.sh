#!/usr/bin/env bash
# Monitor script: runs import + compile checks in a loop for a given number of iterations
set -euo pipefail

INTERVAL=${1:-10}
ITER=${2:-3}

echo "Monitoring torch-mlir tests: interval=${INTERVAL}s iterations=${ITER}"

for i in $(seq 1 $ITER); do
    echo "---- Run: $i $(date) ----"
    ./scripts/test_torch_mlir.sh || echo "One or more checks failed (see above)."
    if [ $i -lt $ITER ]; then
        echo "Sleeping for ${INTERVAL}s..."
        sleep ${INTERVAL}
    fi
done

echo "Monitor finished"
