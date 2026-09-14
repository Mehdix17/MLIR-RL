#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
# Probe: does a compute-partition job see sbatch under train.sh's env?
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/slurm/default/bin:$PATH"
echo "which sbatch: $(which sbatch 2>&1)"
echo "ls /opt/slurm/default/bin/sbatch: $(ls -la /opt/slurm/default/bin/sbatch 2>&1)"
ls /opt/slurm/*/bin/sbatch 2>/dev/null
find / -maxdepth 4 -name sbatch -type f 2>/dev/null | head -3
source ~/envs/mlir/bin/activate
python - <<'EOF'
import asyncio, os
print("py PATH has slurm dir:", "/opt/slurm/default/bin" in os.environ["PATH"])
async def main():
    p = await asyncio.create_subprocess_exec("sbatch", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await p.communicate()
    print("asyncio sbatch ->", (out or err).decode().strip())
asyncio.run(main())
EOF
