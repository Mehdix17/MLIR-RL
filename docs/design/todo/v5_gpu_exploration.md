# V5: GPU Exploration — Analysis Archive

**Status**: Archive / analysis only — **no GPU in V5**
**Moved from**: `v5_training_acceleration.md` on 2026-08-06 (user request — keep the main design doc free of GPU exploration and limits)
**Related**: [v5_training_acceleration.md](../done/v5_training_acceleration.md) — the operational decisions live there; the reasoning lives here.

## Why this file exists

The V5 design doc previously carried GPU exploration content (partition analysis,
C2 QOS limits, GPUOccupier discussion). Per user request, all GPU exploration and
limits content was moved here so the main design doc stays focused on the CPU-only
platform. The decisions — V5 runs without GPU, and **GPUOccupier logic is removed
entirely** from the V5 package — are made in the main doc; this file records the
analysis behind them.

## Partitions with GPUs (live `sinfo` + `sacctmgr`, verified 2026-08-06)

| Partition | Nodes | CPU cores | RAM | GPU | Max wall |
|-----------|-------|-----------|-----|-----|----------|
| `nvidia` (C2 QOS) | 48 | **128 on A100 nodes**, **64 on H100/H200 nodes** | 365-491G | A100/H100/H200 | **4-00:00:00** |
| `dalma` | — | 40 | 105G-1T | V100 (legacy) | — |

C2 QOS live limits (`sacctmgr show qos c2`):
- `GrpTRES: cpu=384, gres+` → **~384 concurrent CPUs, ~5 concurrent GPUs** across the whole team.
- No explicit MaxWall at QOS level (blank); the `nvidia` partition TIMELIMIT is **4-00:00:00** — that's the max wall for any job there.

## A100 vs H100/H200 — analysis

| | A100 nodes (cn001-268) | H100/H200 nodes (cn270-276) |
|---|---|---|
| CPU cores | **128** | **64** |
| GPU | 1-4× A100 80GB | 2-8× H100/H200 (faster) |

H100's raw GPU advantage (FP8, more FLOPs) is **irrelevant** for V5: the model is
tiny (`d_model` 64/256) and would use the GPU ~3% of the time. But H100/H200 nodes
have **half the CPU cores**, and CPU cores drive MLIR execution (the ~95% bottleneck).
So even if GPU were revisited, a bare `--gres=gpu:1` could land on a 64-core H100
node and *slow training down*.

## Why V5 runs WITHOUT GPU (decision, 2026-08-06)

- GPU accelerates only the **~3% sampling/PPO fraction** → **~1.05-1.1x** at best.
- CPU parallelism 12 → 64/128 accelerates the **~95% MLIR-exec fraction** →
  **5-10x** — and it's free on the `compute` partition's 128-core nodes
  (404 nodes available, no shared caps).
- C2 `nvidia` is a **scarce shared resource** (team cap ~5 GPUs / 384 CPUs).
  Spending quota for ~1.05x while compute nodes sit idle is a bad trade.
- V5.1 (full-model eval) is CPU-bound anyway; V5.2 (more actions) makes
  trajectories longer — both need CPU headroom, not GPU.
- **Revisit GPU only if** a future version scales the model (full-model PPO,
  GNN encoder, larger d_model) — at which point the A100-pin rule below applies.

## GPUOccupier — what it was, and why V5 removes it

### What it is
A singleton that spawns a background process running a dummy `torch.matmul(64×64)`
loop on the GPU **whenever the model isn't using it** (`gpu_needed()` events gate
it). Purpose: keep the GPU "hot" (avoid downclocking between sampling bursts) and
guard against idle-GPU policies. Lives in `utils/gpu_occupier.py`.

### Wiring audit (paper_transformer, verified 2026-08-06)
- `utils/gpu_occupier.py` — the class itself.
- `ppo.py:21,69,318` — import + `gpu_needed()` wraps around model sampling and PPO update.
- `train.py:36,51,89,128,132,165` — import, `go.start(device)` (CUDA-only), wraps, `go.stop()`.
- `evaluate.py:36,58,88,117` — import, instantiate, wraps.
- Unified `scripts/train/train.py` (what Slurm actually runs): **never calls `start()`** → today it is a silent no-op in the real pipeline.

### Decision — REMOVED from V5
V5 deletes `utils/gpu_occupier.py` and every `gpu_needed()` wrap in the package
(`ppo.py`, `train.py`, `evaluate.py`). V5 is CPU-only; no GPU code paths remain.
The occupier is moot on a CPU-only pipeline (`device` is CPU, `start()` raises
`ValueError` — it requires CUDA), so removal is pure dead-code cleanup.

## If GPU is ever revisited

- Pin `--gres=gpu:a100:1`, **never** bare `--gres=gpu:1` (see A100 vs H100 above).
- Wire `GPUOccupier().start(device)` into `scripts/train/train.py` (the unified
  entry currently never starts it) — or reintroduce per-package wiring.
- Re-evaluate C2 QOS caps (team quota) before committing.
