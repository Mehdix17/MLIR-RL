#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
# Probe: reproduce the worker-side deserialization in the exported worker env.
cd /scratch/mb10856/MLIR-RL
set -a; source .env; set +a
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/slurm/default/bin:$PATH"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular"
export CONFIG_FILE_PATH=$PWD/config/v5/v5_distributed_smoke.json
export AUTOSCHEDULER_IMPL=rl_autoschedular_v5
echo "=== env ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | head -c 200; echo
python - <<'EOF'
print('--- importing torch ---')
import torch
print('torch', torch.__version__)
print('--- protocol round trip ---')
from distributed.protocol import serialize, deserialize
from rl_autoschedular_v5.model import HiearchyModel
from rl_autoschedular_v5.distributed import rollout_group
m = HiearchyModel()
payload = (rollout_group, 0, [1,2,3,4], m.state_dict(), 1, '/tmp/x.json')
header, frames = serialize(payload)
try:
    deserialize(header, frames)
    print('ROUND TRIP OK')
except Exception as e:
    print('ROUND TRIP FAILED:', type(e).__name__, e)
EOF
