# V5.2: Expanded Transformation Action Space — Design

**Status**: Design Phase
**Version**: **V5.2** of the new MLIR-RL generation (V5 → V5.1 → V5.2)
**Target**: `rl_autoschedular_v5` package (extends V5's package in place)
**Base**: `rl_autoschedular_paper_transformer` action space (6 actions, Transformer encoder — no HW features, no shaped reward). The transforms referenced below live in `rl_autoschedular_v4_9/transforms.py` (padding, unrolling, packing already implemented there) and are inherited/ported into the V5 package.
**Depends on**: V5 (`v5_training_acceleration.md`) — more actions → longer trajectories → more MLIR executions per iteration, so V5's acceleration is a prerequisite
**Precedes/parallel**: V5.1 (full-model eval, `v5_1_full_model_eval.md`) — see the sequencing note in §Overview

## Overview

This document proposes new actions for the MLIR-RL agent's action space. The current action space (V4.9 / V5 base) contains 6 actions: NoTransformation, Tiling, TiledParallelization, TiledFusion, Interchange, and Vectorization. Several MLIR Transform Dialect operations are already implemented in `transforms.py` but have no corresponding action class. Others require new transform code.

**Goal:** Expand the action space to expose finer-grained MLIR transformations, enabling the agent to discover optimization schedules that the current action space cannot express.

> **Sequencing note (open question for the architect)**: the user's stated order is
> V5 → V5.1 (full-model eval) → V5.2 (action space). For *paper sequencing*,
> expanding the action space before full-model eval would make V5.1's headline
> full-model numbers reflect the new agent. Keeping the stated order means V5.1
> results are for the 6-action agent and may need re-running after V5.2. Both are
> defensible — resolve before Phase 3.

---

## Current Action Space (V4.9)

| Symbol | Action | File | Parameters | Terminal | Description |
|:------:|--------|------|-----------|:--------:|-------------|
| `NT` | NoTransformation | `no_transformation.py` | none | yes | Skip this operation |
| `T` | Tiling | `tiling.py` | tile sizes per loop (power-of-2) | no | Tile loops using `transform.structured.tile_using_for` |
| `TP` | TiledParallelization | `tiled_parallelization.py` | tile sizes (parallel non-reduction) | no | Parallelize + tile using `transform.structured.tile_using_forall` |
| `TPF` | TiledFusion | `tiled_fusion.py` | tile + fuse producer | no | Tile consumer + fuse producer into forall loop |
| `I` | Interchange | `interchange.py` | loop permutation (pointers) | no | Permute loop dimensions via `transform.structured.interchange` |
| `V` | Vectorization | `vectorization.py` | none | yes | Vectorize + bufferize + lower to LLVM |

---

## Proposed New Actions

### Phase 1 — Already Implemented in `transforms.py`

These transforms have working implementations in `rl_autoschedular_v4_9/transforms.py` but no action class.

#### 1. Padding (`P`)

- **Transform function:** `transform_pad()` at `transforms.py:116`
- **MLIR op:** `transform.structured.pad pad_to_multiple_of`
- **Purpose:** Pad tensor dimensions to multiples of powers of 2 for aligned SIMD memory access
- **Parameters:** `pad_multiples` per dimension (config: `num_pad_multiples: 3` = {2, 4, 8})
- **Action mask:** Pad is allowed when the operation has static dimensions and the dimension is not already a multiple of the chosen pad value
- **Order:** Should come before Vectorization (`V`) — padding enables successful vectorization
- **Config field:** `num_pad_multiples` (already exists, default 3)
- **Estimated implementation:** ~40 lines for action class

**Why it matters:** Currently Vectorization (`V`) sometimes fails because tensor dimensions aren't multiples of the SIMD vector width. Padding as a separate action gives the agent explicit control over alignment, rather than relying on Vectorization's internal preprocessing.

**Example transform code:**
```mlir
transform.structured.pad %op pad_to_multiple_of [4, 8] :
  (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
```

#### 2. Unrolling (`U`)

- **Transform function:** `transform_unroll()` at `transforms.py:172`
- **MLIR op:** `transform.loop.unroll`
- **Purpose:** Tile loops then unroll them with a configurable factor, exposing instruction-level parallelism
- **Parameters:** tile sizes + unroll factor (config: `num_unroll_factors: 3` = {2, 4, 8})
- **Action mask:** Unroll is allowed when the operation has loops with known upper bounds and the unroll factor divides the innermost loop bound
- **Order:** Should come after Tiling (`T`), before Vectorization (`V`)
- **Config field:** `num_unroll_factors` (already exists, default 3)
- **Estimated implementation:** ~50 lines

**Why it matters:** Unrolling reduces loop overhead, enables constant propagation, and exposes instruction-level parallelism for SIMD units. The manuscript explicitly identifies "Loop Unrolling" as a key optimization (`manuscript/ressources/chapter1/loop_unrolling.png`).

**Example transform code:**
```mlir
%tiled, %loops:2 = transform.structured.tile_using_for %op tile_sizes [4, 0]
transform.loop.unroll %loops:0 { factor = 4 } : !transform.any_op
```

#### 3. Packing (`PK`)

- **Transform function:** `transform_pack()` at `transforms.py:144`
- **MLIR op:** `transform.structured.pack`
- **Purpose:** Reorganize data into blocked/tiled memory layouts, improving cache locality for tiled access patterns
- **Parameters:** packed sizes per dimension
- **Action mask:** Pack is allowed when the operation has static dimensions and the packed sizes divide the dimensions evenly
- **Order:** Should come before Tiling (`T`)
- **Config field:** New field `num_pack_sizes` or reuse `num_tile_sizes`
- **Estimated implementation:** ~40 lines

**Why it matters:** Packing restructures the memory layout of tensors to match the tiling pattern, which can dramatically improve cache hit rates for operations with non-trivial access patterns (e.g., strided convolutions).

**Example transform code:**
```mlir
transform.structured.pack %op packed_sizes = [32, 64] :
  (!transform.any_op) -> !transform.any_op
```

---

### Phase 2 — Need New Transform Code

#### 4. Canonicalize + CSE (`OPT`)

- **Transform function:** New — needs implementation
- **MLIR ops:** `transform.apply_patterns.canonicalization` + `transform.apply_cse`
- **Purpose:** Apply canonicalization patterns and common subexpression elimination to clean up the IR after transformations
- **Parameters:** none (always applies to current operation's enclosing function)
- **Action mask:** Always allowed when not terminal
- **Order:** Can be applied after any transformation
- **Estimated implementation:** ~20 lines for transform function + ~30 for action class

**Why it matters:** After each transformation, redundant operations, dead code, and equivalent subexpressions accumulate in the IR. A dedicated cleanup step can reduce IR size by 20-40%, improving downstream transformation success rates and reducing compilation time.

**Proposed transform code:**
```python
def transform_canonicalize_and_cse(code: str):
    transform_code = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
            %func = transform.structured.match ops{["func.func"]} in %arg0
                : (!transform.any_op) -> !transform.any_op
            transform.apply_patterns to %func {
                transform.apply_patterns.canonicalization
            } : !transform.any_op
            transform.apply_cse to %func : !transform.any_op
            transform.yield
        }
    }"""
    return __run_transform_code(code, transform_code)
```

#### 5. LICM (`LICM`)

- **Transform function:** New — needs implementation
- **MLIR op:** `transform.apply_licm`
- **Purpose:** Loop Invariant Code Motion — hoist computations that don't change across loop iterations out of the loop body
- **Parameters:** none
- **Action mask:** Always allowed when the operation is inside a loop
- **Order:** Most effective after Tiling or Fusion (when new loop-invariant code appears)
- **Estimated implementation:** ~15 lines for transform function + ~30 for action class

**Why it matters:** After tiling and fusion, new loop-invariant computations often appear (e.g., address calculations, shape computations). LICM reduces redundant computation inside loops, decreasing dynamic instruction count.

**Proposed transform code:**
```python
def transform_licm(code: str):
    transform_code = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
            %all_loops = transform.structured.match interface{LoopLikeInterface} in %arg0
                : (!transform.any_op) -> !transform.any_op
            transform.apply_licm to %all_loops : !transform.any_op
            transform.yield
        }
    }"""
    return __run_transform_code(code, transform_code)
```

---

### Phase 3 — Need Careful Integration

#### 6. Distributed Tiling (`DT`)

- **Transform function:** New — needs implementation
- **MLIR op:** `transform.structured.tile_using_forall` with `num_threads`
- **Purpose:** Tile loops and distribute tiles across a specified number of threads for parallel execution
- **Parameters:** tile sizes + number of threads
- **Action mask:** Allowed when the operation has parallel dimensions and thread count > 1
- **Order:** Alternative to `TP` — use when explicit thread control is desired
- **Estimated implementation:** ~40 lines

**Why it matters:** More fine-grained control over parallelization than `TiledParallelization`. The agent can choose thread counts based on hardware features (from Novelty 1: hardware-aware observation), enabling better load balancing on heterogeneous systems.

**Considerations:** Overlaps with `TiledParallelization` — may be better as a parameter variant of `TP` rather than a separate action.

#### 7. Multi-Op Fusion (`MF`)

- **Transform function:** New — needs implementation
- **MLIR op:** `transform.structured.fuse_into_containing_op` (multiple producers)
- **Purpose:** Fuse multiple producer operations into a single consumer's loop, reducing memory traffic
- **Parameters:** which producers to fuse (indices into the producer list)
- **Action mask:** Allowed when the operation has multiple fusible producers
- **Order:** Should come before Tiling
- **Estimated implementation:** ~60 lines

**Why it matters:** Currently `TiledFusion` only fuses one producer at a time. Multi-op fusion can dramatically reduce memory traffic when multiple small producers feed into one consumer (common in attention mechanisms, batch normalization).

**Considerations:** Complex parameter space — may need hierarchical action selection or a separate policy head. Could also be implemented as a sequence of `TPF` actions.

#### 8. Loop Fusion across Operations (`LOF`)

- **Transform function:** New — needs implementation
- **MLIR op:** `transform.structured.fuse` (different from `fuse_into_containing_op`)
- **Purpose:** Fuse two separate loop nests operating on the same data into one loop nest
- **Parameters:** pair of operations to fuse
- **Action mask:** Allowed when two operations share loop bounds and access overlapping data
- **Order:** Early in the schedule (before individual op transformations)
- **Estimated implementation:** ~50 lines

**Why it matters:** Cross-operation loop fusion reduces memory bandwidth requirements by keeping intermediate data in registers/cache. This is especially impactful for element-wise operations (add, relu, batch_norm) that follow compute-heavy operations (matmul, conv2d).

---

## Recommended Action Order

The `config.order` field controls which actions are available at each step. A recommended order for the expanded action space:

```json
{
  "order": [
    ["P", "PK", "!"],
    ["T", "I"],
    ["!", "I", "NT"],
    ["V", "U", "NT"]
  ]
}
```

| Step | Allowed Actions | Rationale |
|:----:|----------------|-----------|
| 0 | Padding, Packing, or skip | Pre-processing: align data before transformations |
| 1 | Tiling or Interchange | Core transformation: choose tile sizes or loop order |
| 2 | Anything except Interchange (or skip) | Secondary transformation: parallelize, fuse, etc. |
| 3 | Vectorization, Unrolling, or skip | Terminal: lower to vector instructions or finalize |

---

## Implementation Checklist

### Phase 1 (Padding, Unrolling, Packing)

- [ ] Create `actions/padding.py` — new action class
- [ ] Create `actions/unrolling.py` — new action class
- [ ] Create `actions/packing.py` — new action class
- [ ] Add `num_pad_multiples`, `num_unroll_factors` to Config (already exist)
- [ ] Add `num_pack_sizes` to Config (new field)
- [ ] Update `actions/__init__.py` — add to `supported_actions`
- [ ] Update `observation.py` — add mask/history sizes for new actions
- [ ] Update `config.order` — include new actions in scheduling sequence
- [ ] Test with ablation smoke test (`ablation.py` pattern)
- [ ] Run short training (500 iterations) to verify stability

### Phase 2 (Canonicalize, LICM)

- [ ] Add `transform_canonicalize_and_cse()` to `transforms.py`
- [ ] Add `transform_licm()` to `transforms.py`
- [ ] Create `actions/canonicalize.py` — new action class
- [ ] Create `actions/licm.py` — new action class
- [ ] Update `actions/__init__.py`, `observation.py`, config

### Phase 3 (Distributed Tiling, Multi-Op Fusion, Loop Fusion)

- [ ] Design parameter space for multi-op actions
- [ ] Implement transform functions
- [ ] Create action classes
- [ ] Consider hierarchical action selection for multi-op actions
- [ ] May require changes to `BenchmarkFeatures` / `OperationFeatures`

---

## Parameter Space Summary

| Action | Parameters | Config Field | Range |
|--------|-----------|-------------|-------|
| `P` (Pad) | pad_multiples per dim | `num_pad_multiples` | {2, 4, 8} per dim |
| `U` (Unroll) | tile_sizes + factor | `num_unroll_factors` | {2, 4, 8} |
| `PK` (Pack) | packed_sizes per dim | `num_pack_sizes` (new) | {16, 32, 64} per dim |
| `OPT` (Canonicalize) | none | — | — |
| `LICM` | none | — | — |
| `DT` (Distributed Tile) | tile_sizes + num_threads | (new) | {1, 2, 4, 8, 16} threads |
| `MF` (Multi-Op Fusion) | producer indices | — | set of fusible producers |
| `LOF` (Loop Fusion) | op pair | — | pair of fusible operations |

---

## References

- [MLIR Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- [transform.structured.pad](https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredpad)
- [transform.structured.pack](https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredpack)
- [transform.loop.unroll](https://mlir.llvm.org/docs/Dialects/Transform/#transformloopunroll)
- [NOVELTIES.md](NOVELTIES.md) — Novelty 5: Expanded Transformation Action Space
- [v0_original_baseline.md](v0_original_baseline.md) — Current action space definition
- [v0_model_detailed.md](v0_model_detailed.md) — Action masking and hierarchical index
