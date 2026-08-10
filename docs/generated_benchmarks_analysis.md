# Generated Benchmarks Analysis

**Date:** 2026-03-06  
**Context:** Attempted to use `data/nn/generated/code_files/` as training data for the RL scheduler.  
**Outcome:** All 22 files returned `-1` (execution failed). Training on this dataset is not possible.

---

## What the RL scheduler expects

The scheduler is a **loop-nest optimizer**. It takes a file containing a **single operation**
(one conv, one matmul, one pooling, etc.) with a fixed `@main` entry point and static tensor
shapes, applies tiling/vectorization transforms, then measures speedup.

A working benchmark looks like:

```mlir
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<256x128x14x14xf64>, %arg1: tensor<32x128x3x3xf64>,
                  %arg2: tensor<256x32x12x12xf64>) -> (tensor<256x32x12x12xf64>, i64)
    attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    %1  = linalg.conv_2d_nchw_fchw { ... }     // ← single loop nest to optimize
    %t1 = call @nanoTime() : () -> i64
    return %1, (t1 - t0) : tensor<...>, i64    // ← returns timing
  }
}
```

Key properties:
- Entry point named exactly `@main`
- Returns execution time alongside result
- **Single** linalg operation (conv / matmul / pooling / relu / add)
- Fully **static** tensor shapes (no `?` dimensions)
- All weight/input data **embedded** (no `dense_resource<__elided__>`)

---

## What the generated files actually are

The generated files are **full neural network graphs** — e.g. `mobilenet_v3_small_linalg.mlir`
is a 1315-line MLIR file containing one function with hundreds of operations
(conv + relu + pool + matmul + ...).

Example:
```mlir
func.func @main_graph(%arg0: tensor<?x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst   = arith.constant dense_resource<__elided__> : tensor<576x5x5xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<576x5x5xf32>
    ...  // hundreds more ops
}
```

---

## Blockers (why all files return -1)

### Blocker 1 — Weights are gone (`dense_resource<__elided__>`) ❌ Unrecoverable

`data_utils/strip_mlir.py` replaced all weight tensors with `dense_resource<__elided__>` to
save disk space. This is intentional but means **the file cannot be executed** — the JIT needs
actual numbers to fill input arrays.

- `transformers2mlir.py` backs up the unstripped version to `data/nn/non_stripped_models/`
- However, even unstripped files hit the other blockers below

**Attempted fix:** Cannot be fixed without regenerating files with `--no-strip-weights`.

---

### Blocker 2 — Wrong function name (`@main_graph` vs `@main`) ⚠️ Easy but moot

The executor's `__create_params()` in `rl_autoschedular/execution.py` does:
```python
next(op for op in module.body.operations if op.name.value == 'main')
# → StopIteration: no function named 'main' found
```
The ONNX exporter names the function `@main_graph` by default.

**Fix:** A simple regex rename — but irrelevant until blockers 1, 3, and 4 are resolved.

---

### Blocker 3 — Dynamic shapes (`tensor<?x3x224x224xf32>`) ❌

The `?` batch dimension is unknown at compile time. The JIT cannot allocate tensors of unknown
size. Working benchmarks hard-code all dimensions:
```
tensor<256x128x14x14xf64>   ← static, works
tensor<?x3x224x224xf32>     ← dynamic, fails
```

**Fix:** Run `mlir-opt --specialize-tensor-shapes` or add explicit `--batch-size` specialization
during generation. Requires changes to `transformers2mlir.py` / `vision2mlir.py`.

---

### Blocker 4 — Wrong granularity (structural mismatch) ❌

Even if all the above were fixed, a 1315-line full-model file is the **wrong input** for this
scheduler. The RL agent's action space and observation space are designed around a
**single `linalg.conv_2d` or `linalg.matmul`** loop nest — one set of tile sizes, one
vectorization decision. It has no concept of optimizing 200+ ops simultaneously.

---

## Error messages

**`*_torch.mlir` files** — Torch dialect not registered:
```
MLIRError: Unable to parse module assembly:
error: "-":2:39: `!"torch"<"vtensor<[1,3,224,224],f32>">` type created with
unregistered dialect.
```
These files contain raw Torch IR and need to be lowered to linalg first
(via `torch-mlir-opt` lowering passes).

**`*_linalg.mlir` files** — Missing `@main` + dynamic shapes + elided weights:
```
StopIteration  (no @main function found)
```
```
Expected<T> must be checked before access or destruction.
Failed to materialize symbols: { (main, { __kmpc_barrier, ... }) }
```

---

## Current working dataset

`data/nn/code_files/` — **68 valid benchmarks** (out of 75):

| Type | Count | In train split | In eval split |
|---|---|---|---|
| `conv` | 18 | 14 | 4 |
| `matmul` | 15 | 12 | 3 |
| `relu` | 14 | 12 | 2 |
| `pooling` | 10 | 8 | 2 |
| `add` | 10 | 8 | 2 |
| `model-*` | 7 | 0 (all fail) | 0 |

Training config: `config/train1.json` with `benchmarks_folder_path: "data/nn/code_files"`.

---

## Paths forward

| Goal | What's needed |
|---|---|
| **Train now (recommended)** | Use `data/nn/code_files/` — 68 valid benchmarks, works today |
| **Generate more single-op kernels** | Modify `transformers2mlir.py` / `vision2mlir.py` to extract individual layers (conv, matmul) as separate benchmark files instead of exporting the full model graph |
| **Train on full models** | Requires a completely different RL formulation — the current scheduler is not designed for multi-op graphs |

---

## Files involved

| File | Role |
|---|---|
| `data_utils/transformers2mlir.py` | Exports HuggingFace models → MLIR (full graph, strips weights) |
| `data_utils/vision2mlir.py` | Exports vision models (torchvision) → MLIR (same issue) |
| `data_utils/strip_mlir.py` | Replaces weight constants with `dense_resource<__elided__>` |
| `data/nn/generated/code_files/` | 22 full-model MLIR files — **not usable for training** |
| `data/nn/non_stripped_models/` | Backup of unstripped versions (still hit blockers 3 & 4) |
| `data/nn/code_files/` | 75 single-op benchmark files — **68 usable for training** |
| `rl_autoschedular/execution.py` | JIT executor — expects `@main`, static shapes, embedded data |
