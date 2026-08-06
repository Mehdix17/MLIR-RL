# HPC Hardware Reference

Hardware specifics for the NYUAD Jubail HPC cluster, C2 QOS GPU nodes, Dalma partition, and the Kindi standalone machine. Verified against live Slurm state (`scontrol`, `sinfo`, `sacctmgr`) on 2026-08-03 and cross-referenced with the [Guide to Using C2 Machines](Guide%20to%20Using%20C2%20Machines.md).

---

## 1. Bergamo Nodes — `compute` partition (CPU only)

| Item | Specification |
|------|---------------|
| **Number of nodes** | 57 (`bn001-058`) |
| **CPUs per node** | **256** (2 × 128-core sockets) |
| **CPU Model** | AMD EPYC 9754 (Genoa, Zen 4) |
| **RAM per node** | **1 TB** (1,010,688 MB) |
| **GPU** | **None** (`Gres=(null)`) |
| **Partitions** | `compute`, `bigmem`, `admin`, `preempt`, `xxl` |
| **Features** | `bergamo,compute,fat,bigmem` |
| **Use case** | High-memory / high-core CPU jobs. Currently used by `train.sh` (`--constraint=bergamo`). |

This is the largest single-node memory + core option on the Jubail HPC. All MLIR-RL training jobs currently run here.

```bash
sbatch --partition=compute --constraint=bergamo --cpus-per-task=12 --mem=32G scripts/train/train.sh
```

---

## 2. Jubail Standard Nodes — `compute` partition (CPU only)

| Item | Specification |
|------|---------------|
| **Number of nodes** | ~233 (`cn018-024,034-246,248-250`, etc.) |
| **CPUs per node** | **128** (2 × 64-core sockets) |
| **CPU Model** | AMD EPYC 7742 (Rome, Zen 2) @ 2.25 GHz |
| **RAM per node** | **480 GB** usable (512 GB theoretical) |
| **GPU** | **None** (`Gres=(null)`) |
| **Partitions** | `compute`, `admin`, `preempt`, `xxl` |
| **Features** | `jubail,compute,512g` |
| **Use case** | Balanced high-core CPU jobs. Same `compute` partition as Bergamo, just smaller nodes. |

Note: `cn023` is a Jubail standard node (128 cores, no GPU) despite being mentioned as a GPU node in some older docs. Guide identifies `cn009` as the C2 GPU node — confirmed below.

---

## 3. C2 QOS GPU Nodes — `nvidia` partition (CPU + GPU)

Accessed via `--qos=c2 -p nvidia --gres=gpu:a100:1`. These nodes have **both CPU and GPU on the same machine**.

| Node(s) | GPUs | CPU Cores | RAM | Notes |
|---------|------|-----------|-----|-------|
| `cn001-017,262` | **3× A100 (80GB)** | 128 | 480 GB | Primary C2 GPU nodes |
| `cn252-255` | **1× A100 (80GB)** | 128 | 480 GB | Single-GPU A100 nodes |
| `cn256-259` | **2× A100 (80GB)** | 128 | 480 GB | Dual-GPU A100 nodes |
| `cn263-268` | **4× A100 (80GB)** | 128 | 480 GB | Quad-GPU A100 nodes |
| `cn270` | **7× H100** | 64 | — | H100 node |
| `cn272-273` | **2× H100** | — | — | H100 nodes |
| `cn271,276` | **8× H200** | 64 | ~2 TB | H200 nodes (newest) |

### C2 QOS Limits

- **Team GPU cap**: ~5 concurrent GPUs (`GrpTRES` includes `gres` limit). Shared across all C2 users.
- **CPU cap**: 384 concurrent CPUs across all C2 jobs.
- **Your access**: User `mb10856` has `c2` QOS (confirmed via `sacctmgr`).
- **Partition**: `nvidia` (also in `condo` partition).
- **Max wall time**: Check `sacctmgr show qos c2` or `show-my-limits`.

### How to Use

```bash
# Interactive access to a C2 GPU node
srun --pty -n1 -q c2 -p nvidia --gres=gpu:a100:1 bash

# Specific node (e.g. cn009 with 3 A100s)
srun --pty -n1 -q c2 -p nvidia --gres=gpu:a100:1 -w cn009 bash

# Training job on GPU
sbatch --qos=c2 --partition=nvidia --gres=gpu:a100:1 --cpus-per-task=64 --mem=128G \
  scripts/train/train.sh config/ops_and_blocks/train/paper_transformer_small.json
```

**Use case**: GPU acceleration for the PyTorch model (forward pass, PPO gradients). MLIR compilation still runs on the node's CPU cores. The code already supports CUDA — `device = torch.device("cuda") if torch.cuda.is_available()`.

---

## 4. Dalma Partition — V100 GPU Nodes (CPU + GPU)

| Node(s) | GPUs | CPU Cores | RAM | CPU Model |
|---------|------|-----------|-----|-----------|
| `dn001-002` | **8× V100** | 40 | 1 TB | Intel Xeon E5-2680 v4 (Broadwell) |
| `dn003-005,007-008,011-014` | **2× V100** | 40 | 365 GB | Intel Xeon E5-2680 v4 |
| `dn019-032,037-144,177-180` | **None** | 28 | 105 GB | Intel Xeon E5-2680 v4 |

V100 GPUs have 16-32GB memory each (older generation). Less powerful than A100 but available via `dalma` partition.

```bash
sbatch --partition=dalma --gres=gpu:v100:1 --cpus-per-task=20 --mem=64G \
  scripts/train/train.sh config/ops_and_blocks/train/paper_transformer_small.json
```

---

## 5. Kindi Machine (Standalone, Not on Slurm)

| Item | Specification |
|------|---------------|
| **Access** | SSH: `ssh -p 4410 <netid>@kindi.abudhabi.nyu.edu` (requires NYUAD VPN) |
| **CPU** | AMD EPYC 7742 64-core @ 2.25 GHz (128 cores total) |
| **RAM** | 1 TB (16 × 64GB DDR4) |
| **GPUs** | **8× NVIDIA A100 (80GB each)** |
| **Storage** | 2× 3.84TB SSD (RAID 1, mounted at `/` and `/home`), 4× NVMe (14TB at `/data`) |
| **CUDA** | 11.4 preinstalled |
| **Slurm** | **No** — no job scheduling, no `sbatch`. Run training directly with `python`. |
| **File system** | Separate from Jubail HPC `/scratch`. Has its own disks. |

**Use case**: Heavy GPU workloads when C2 QOS GPUs are busy. Must install your own conda environment (no HPC preinstalled software). Use `screen` for session persistence.

---

## Quick Comparison

| Cluster / Type | Nodes | Cores/Node | RAM/Node | GPU | Access |
|----------------|-------|------------|----------|-----|--------|
| **Bergamo** (`compute`) | 57 | **256** | **1 TB** | ❌ None | Default `compute` |
| **Jubail standard** (`compute`) | 233 | **128** | 480 GB | ❌ None | Default `compute` |
| **C2 GPU** (`nvidia`) | ~25 | 128 | 480 GB | ✅ A100 / H100 / H200 | `--qos=c2` |
| **Dalma GPU** (`dalma`) | 12 | 40 | 365 GB-1TB | ✅ V100 (2 or 8) | `-p dalma` |
| **Dalma CPU** (`dalma`) | ~140 | 28 | 105 GB | ❌ None | `-p dalma` |
| **Kindi** (standalone) | 1 | 128 | 1 TB | ✅ 8× A100 (80GB) | SSH only |

---

## Tips for Job Submission

| Need | Use |
|------|-----|
| Massive CPU parallelism (current training) | **Bergamo** — `--constraint=bergamo --cpus-per-task=128` |
| GPU for model + CPU for MLIR compilation | **C2 QOS** — `--qos=c2 -p nvidia --gres=gpu:a100:1` |
| GPU when C2 is busy | **Dalma** — `-p dalma --gres=gpu:v100:1` (V100, older) |
| Dedicated 8× A100, no Slurm overhead | **Kindi** — SSH and run directly |
| Multiple training seeds in parallel | **Bergamo** — `sbatch --array=0-3 --cpus-per-task=32` (4 × 32 = 128 cores) |

**Note**: MLIR compilation is always CPU. GPU nodes help the PyTorch model but not the MLIR execution bottleneck. The biggest training speedup comes from increasing `--cpus-per-task` on Bergamo (currently 12, max 256).