# Plan: Fix All 4 Benchmark Generation Blockers

**Date:** 2026-03-06  
**Goal:** Make `data/nn/generated/code_files/` produce single-operation `.mlir` files that the
RL scheduler can execute and train on.  
**Reference:** `docs/generated_benchmarks_analysis.md`

---

## Concrete Evidence (resnet18_linalg.mlir)

Running on the actual generated file shows all 4 problems simultaneously:

```
func.func @main_graph(%arg0: tensor<?x3x224x224xf32>) ...   ← Blockers 2 & 3
  %cst = arith.constant dense_resource<__elided__> : tensor<512xf32>  ← Blocker 1
  ...99 linalg operations...                                ← Blocker 4
    tensor<?x64x112x112xf32>  ← dynamic shapes everywhere  ← Blocker 3
```

Op breakdown in resnet18: 20 conv_2d, 28 linalg.generic (relu/bn/add), 1 matmul, 1 pooling.
Each of these is a valid training benchmark — we just need to extract them individually.

---

## Root Cause

All 4 blockers have the same root cause: **the generation pipeline exports the whole model
graph** (one file = one full network). The fix is not to patch the broken files — it is to
change what the pipeline produces:

```
Current:  model → full_graph.mlir (1 file, 100+ ops, dynamic shapes, elided weights)
Target:   model → op_0.mlir, op_1.mlir, ... op_N.mlir (N files, 1 op each, static, executable)
```

---

## The 4 Blockers and Their Fixes

---

### Blocker 1 — `dense_resource<__elided__>` weights

**Root cause:** `strip_mlir.py` replaces `dense<"0x...">` weight tensors with
`dense_resource<__elided__>` to save disk space. This happens *after* ONNX export.

**Why it's "unrecoverable" in the full-graph files:** The full model has weights embedded as
`arith.constant` nodes. When stripped, those constants are gone and cannot be reconstructed
without re-running the export.

**Fix:** This blocker **disappears automatically** when Blocker 4 is fixed. When individual
operations are extracted, their inputs are function *arguments* (not embedded constants):

```mlir
func.func @main(%arg0: tensor<1x3x230x230xf32>,  ← input (provided by executor)
                %arg1: tensor<64x3x7x7xf32>,     ← kernel (provided by executor)
                %arg2: tensor<1x64x112x112xf32>)  ← output (provided by executor)
```

There are no `arith.constant dense<...>` nodes in single-op files — weights become tensor
arguments that the JIT allocates and fills with zeros/random values. The RL agent only measures
*how fast* a loop nest runs, not whether the output values are correct.

**Action required:** Ensure `--no-strip-weights` is passed (or stripping is skipped) during
extraction. In practice the extracted files will have no large constants to strip anyway.

---

### Blocker 2 — Wrong function name (`@main_graph` vs `@main`)

**Root cause:** The ONNX exporter names the entry function `@main_graph` by default.
The RL executor requires exactly `@main`:

```python
# rl_autoschedular/execution.py
next(op for op in module.body.operations if op.name.value == 'main')
# → StopIteration when function is named @main_graph
```

**Fix A (generation side — preferred):** Pass `func_name='main'` to the export API in
`vision2mlir.py` and `transformers2mlir.py`:

```python
# vision2mlir.py — convert_direct_route()
mlir_module = export_and_import(
    model, dummy_input,
    output_type=OutputType.LINALG_ON_TENSORS,
    func_name='main',          # ← add this
)
```

For the ONNX route, the function name comes from `torch-mlir-opt` lowering. Pass
`--tf-saved-model-exported-names=main` or a post-processing rename.

**Fix B (post-processing fallback):** Rename after export with a one-line sed:

```bash
sed -i 's/@main_graph/@main/g' data/nn/generated/code_files/*_linalg.mlir
```

**For the new per-op extraction script (Blocker 4 fix), the wrapper always writes `@main`
directly — so Fix B is implicit.**

---

### Blocker 3 — Dynamic shapes (`tensor<?x3x224x224xf32>`)

**Root cause:** The ONNX exporter uses symbolic batch dimensions (`?`) because `torch.onnx.export`
is called without a fixed batch size in the concrete shapes. The downstream `torch-mlir-opt`
lowering passes preserve this dynamic dimension.

**Evidence:**
```mlir
tensor<?x3x224x224xf32>      ← batch dim is dynamic
tensor<?x64x112x112xf32>     ← propagates through all ops
```

The JIT cannot allocate tensors of unknown size.

**Fix A (at ONNX export — cleanest):** Use a concrete batch size in `torch.onnx.export`:

```python
# vision2mlir.py — convert_onnx_route()
# Before:
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, f"{base}.onnx", ...)

# After: add dynamic_axes=None to freeze all dimensions
torch.onnx.export(
    model, dummy_input, f"{base}.onnx",
    opset_version=opset,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,          # ← freeze all shapes as static
)
```

**Fix B (at torch-mlir lowering — belt-and-suspenders):** Pass shape specialization flag:

```bash
torch-mlir-opt \
  --specialize-tensor-shapes="batch_size=1" \
  --torch-backend-to-linalg-on-tensors-backend-pipeline \
  input.mlir -o output_linalg.mlir
```

**Fix C (per-op extraction — implicit fix):** When a single op is extracted with concrete
input/output SSA values, the extractor resolves the `?` to its concrete value at that point
in the graph (e.g., `?` = 1 because `dummy_input` had batch=1). The extraction script
propagates the concrete value.

**For the new extraction pipeline, Fix A is applied first, then Fix C handles any remaining `?`.**

---

### Blocker 4 — Wrong granularity (full model graph)

**Root cause:** The RL scheduler is designed for **one loop nest per file**. The generated
files are entire neural networks (resnet18 = 99 linalg operations in one function).

**This is the primary blocker — fixing it automatically resolves Blockers 1 and 3.**

**Fix:** Write a new script `data_utils/extract_ops.py` that:
1. Reads a `*_linalg.mlir` file
2. Uses the `AstDumper` binary (already in the project) to parse all operations
3. For each `linalg.*` operation (conv, matmul, pooling, generic, add):
   - Resolves the concrete shapes from the SSA type annotations
   - Generates a standalone `.mlir` file with that single op + a timed `@main` wrapper
4. Names the output file `{model}_{op_type}_{shape_signature}.mlir`

**Skeleton of `extract_ops.py`:**

```python
"""
extract_ops.py
--------------
Extract individual linalg operations from a full-model MLIR file, producing
one benchmark file per operation ready for RL training.

Usage:
    python data_utils/extract_ops.py \
        --input data/nn/generated/code_files/resnet18_linalg.mlir \
        --output-dir data/nn/extracted/resnet18/

Output files: resnet18_conv2d_0.mlir, resnet18_matmul_0.mlir, ...
"""

import re
import argparse
import os

TARGET_OPS = [
    r'linalg\.conv_2d_nchw_fchw',
    r'linalg\.conv_2d_nhwc_hwcf',
    r'linalg\.matmul',
    r'linalg\.pooling_nchw_max',
    r'linalg\.pooling_nhwc_max',
    r'linalg\.generic',
    r'linalg\.add',
]

def extract_tensor_shapes(op_line: str) -> list[str]:
    """Extract all tensor<...> shapes from an op line."""
    return re.findall(r'tensor<[^>]+>', op_line)

def specialize_dynamic_dim(shape: str, batch_size: int = 1) -> str:
    """Replace ? with a concrete batch size."""
    return shape.replace('?', str(batch_size))

def build_benchmark_mlir(op_line: str, op_name: str, affine_maps: str) -> str:
    """Wrap a single linalg op in a timed @main function."""
    shapes = [specialize_dynamic_dim(s) for s in extract_tensor_shapes(op_line)]
    # Last shape = output (outs(...))
    # All shapes = function arguments
    args = [f'%arg{i}: {s}' for i, s in enumerate(shapes)]
    ret_type = shapes[-1]

    # Patch the op line: replace SSA names with %arg0, %arg1, ...
    patched_op = patch_ssa_names(op_line, len(shapes))

    return f"""{affine_maps}
module {{
  func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}
  func.func @main({', '.join(args)}) -> ({ret_type}, i64)
    attributes {{llvm.emit_c_interface}} {{
    %t0 = call @nanoTime() : () -> i64
    %result = {patched_op}
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return %result, %delta : {ret_type}, i64
  }}
}}"""

def patch_ssa_names(op_line: str, num_args: int) -> str:
    """Replace all SSA value references with %arg0..%argN."""
    # Find all %name references in ins(...) and outs(...)
    ssa_refs = re.findall(r'%\w+', op_line)
    unique_refs = list(dict.fromkeys(ssa_refs))
    result = op_line
    for i, ref in enumerate(unique_refs):
        if i >= num_args:
            break
        result = result.replace(ref, f'%arg{i}')
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model-name', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model_name or os.path.basename(args.input).replace('_linalg.mlir', '')

    with open(args.input) as f:
        content = f.read()

    # Extract affine map declarations at the top of the module
    affine_maps = '\n'.join(
        line for line in content.splitlines()
        if line.startswith('#map')
    )

    op_counts = {}  # {op_type: count}
    written = 0

    for line in content.splitlines():
        for pattern in TARGET_OPS:
            if re.search(pattern, line):
                op_type = re.search(r'linalg\.(\w+)', line).group(1)
                idx = op_counts.get(op_type, 0)
                op_counts[op_type] = idx + 1

                shapes = extract_tensor_shapes(line)
                if not shapes:
                    continue
                # Skip if any shape is still dynamic after specialization
                if '?' in ''.join(specialize_dynamic_dim(s, args.batch_size) for s in shapes):
                    print(f"  Skipping {op_type}_{idx}: unresolvable dynamic shape")
                    continue

                try:
                    mlir = build_benchmark_mlir(line, op_type, affine_maps)
                    out_path = os.path.join(
                        args.output_dir, f'{model_name}_{op_type}_{idx}.mlir'
                    )
                    with open(out_path, 'w') as f:
                        f.write(mlir)
                    written += 1
                except Exception as e:
                    print(f"  Skipping {op_type}_{idx}: {e}")
                break  # matched, don't test other patterns

    print(f"Extracted {written} benchmark files to {args.output_dir}/")
    print(f"Op breakdown: {dict(sorted(op_counts.items()))}")
```

---

## Implementation Plan

### Phase 1 — Fix `vision2mlir.py` and `transformers2mlir.py` (Blockers 2 & 3)

**Files:** `data_utils/vision2mlir.py`, `data_utils/transformers2mlir.py`

**Changes:**

1. **`convert_onnx_route()`** — add `dynamic_axes=None` to `torch.onnx.export()`:
   ```python
   torch.onnx.export(
       model, dummy_input, f"{base}.onnx",
       opset_version=opset,
       input_names=["input"], output_names=["output"],
       dynamic_axes=None,     # ← freeze all dims as static
   )
   ```

2. **`convert_direct_route()`** — add `func_name='main'`:
   ```python
   mlir_module = export_and_import(
       model, dummy_input,
       output_type=OutputType.LINALG_ON_TENSORS,
       func_name='main',      # ← correct entry name
   )
   ```

3. **Both files** — add `--no-strip-weights` flag (already have `--strip-weights`):
   Change default from `default=True` to `default=False`:
   ```python
   parser.add_argument('--strip-weights', action='store_true', default=False, ...)
   ```

**Expected outcome:** Re-running `vision2mlir.py --backend direct --model resnet18` produces
a file with `@main` and all static tensor shapes. Intermediate large files are not stripped.

---

### Phase 2 — Write `data_utils/extract_ops.py` (Blocker 4 → resolves Blockers 1 & 3)

**New file:** `data_utils/extract_ops.py`

Implement the skeleton above with these refinements:

1. **Shape propagation:** When a `?` dimension cannot be resolved from the type annotation,
   substitute the `--batch-size` argument value (default=1). Log skipped ops.

2. **Deduplicate by shape:** Two `conv_2d` ops with identical shapes produce identical
   benchmarks. Keep only unique `(op_type, shapes_tuple)` combinations per model.

3. **`linalg.generic` filtering:** Generic ops include both compute-heavy fused kernels
   (relu, batch-norm) and trivial broadcast/fill ops. Filter by minimum loop depth
   (skip ops with fewer than 2 parallel loops) to avoid trivial benchmarks.

4. **Affine map preservation:** Include only the `#map_N` declarations that are referenced
   by the extracted op (avoid cluttering files with irrelevant maps).

5. **Batch-process all models:**
   ```bash
   for f in data/nn/generated/code_files/*_linalg.mlir; do
       model=$(basename $f _linalg.mlir)
       python data_utils/extract_ops.py \
           --input $f \
           --output-dir data/nn/extracted/$model \
           --batch-size 1 \
           --model-name $model
   done
   ```

---

### Phase 3 — Validate extracted files

Before training on all extracted files, validate a sample:

```bash
# 1. Verify one file parses and executes
source .env
python -c "
import sys; sys.path.insert(0, '.')
from rl_autoschedular.evaluation import evaluate_code_with_cmd_and_timeout
with open('data/nn/extracted/resnet18/resnet18_conv_2d_nchw_fchw_0.mlir') as f:
    code = f.read()
t, ok = evaluate_code_with_cmd_and_timeout(code, 'tmp/test_extract.mlir', timeout=60)
print(f'OK={ok}  time={t} ns')
"

# 2. Check shapes are all static
grep -r "tensor<?" data/nn/extracted/ | wc -l  # should be 0

# 3. Check function name
grep -r "func.func @main_graph" data/nn/extracted/ | wc -l  # should be 0
grep -r "func.func @main(" data/nn/extracted/ | wc -l       # should equal file count
```

---

### Phase 4 — Generate baseline execution times

Once validation passes:

```bash
# Flatten all extracted files into one directory
mkdir -p data/nn/from_models
find data/nn/extracted/ -name "*.mlir" -exec cp {} data/nn/from_models/ \;

# Count
ls data/nn/from_models/ | wc -l   # expected: 500–2000 depending on models

# Measure baseline execution times
python get_base.py \
  --benchmarks-dir data/nn/from_models \
  --output data/nn/from_models_exec_times.json \
  --backend cmd \
  --timeout 60
```

---

### Phase 5 — Create training config

Create `config/nn-models.json`:

```json
{
    "max_num_stores_loads": 7,
    "max_num_loops": 7,
    "max_num_load_store_dim": 7,
    "num_tile_sizes": 7,
    "vect_size_limit": 512,
    "order": [["I"], ["!", "I", "NT"], ["!", "I"], ["V", "NT"]],
    "interchange_mode": "enumerate",
    "exploration": ["entropy"],
    "init_epsilon": 0.5,
    "normalize_bounds": "log",
    "normalize_adv": "standard",
    "reuse_experience": "none",
    "benchmarks_folder_path": "data/nn/from_models",
    "json_file": "data/nn/from_models_train.json",
    "eval_json_file": "data/nn/from_models_eval.json",
    "bench_count": 64,
    "replay_count": 10,
    "nb_iterations": 10000,
    "ppo_epochs": 4,
    "ppo_batch_size": 32,
    "value_epochs": 0,
    "value_batch_size": 32,
    "value_coef": 0.5,
    "value_clip": false,
    "entropy_coef": 0.01,
    "lr": 0.001,
    "truncate": 5,
    "tags": [],
    "debug": false,
    "main_exec_data_file": "",
    "results_dir": "results"
}
```

Then train:
```bash
python train.py --config config/nn-models.json
```

---

## Work Summary

| Phase | Task | File(s) to change | Effort |
|---|---|---|---|
| 1a | Add `dynamic_axes=None` to ONNX export | `vision2mlir.py`, `transformers2mlir.py` | 10 min |
| 1b | Add `func_name='main'` to direct route | `vision2mlir.py`, `transformers2mlir.py` | 10 min |
| 1c | Default `--strip-weights` to `False` | `vision2mlir.py`, `transformers2mlir.py` | 5 min |
| 2 | Write `extract_ops.py` | New file | 2–3 hours |
| 3 | Validate extracted files | — (run commands) | 30 min |
| 4 | Generate baseline times | — (run `get_base.py`) | 1–2 hours (runtime) |
| 5 | Create `config/nn-models.json` | New config file | 15 min |

**Total estimated implementation time: ~4 hours** (excluding model regeneration and training runtime)

---

## Dependency Graph

```
Phase 1: fix vision2mlir.py + transformers2mlir.py
    ↓
    Re-run: python data_utils/vision2mlir.py --model resnet18 --backend direct
    (produces clean resnet18_linalg.mlir with @main and static shapes)
    ↓
Phase 2: extract_ops.py
    ↓
    Produces: data/nn/from_models/*.mlir  (one file per linalg op)
    ↓
Phase 3: validate
    ↓
Phase 4: get_base.py → from_models_exec_times.json
    ↓
Phase 5: train.py --config config/nn-models.json
```

---

## Notes

- **Blocker 4 is the critical path.** Phases 2–5 cannot proceed without it. If time is
  limited, skip Phases 1 and 2 and use `data/nn/other/` directly — it already has 1,515
  extracted single-op files that are ready for training today.

- **The `data/nn/other/` folder is the reference implementation** of what the extraction
  script should produce. Verify output format against those files.

- **Re-run `vision2mlir.py` after Phase 1 changes** before Phase 2 — the current
  `*_linalg.mlir` files still have dynamic shapes and `@main_graph`. The extraction script
  can handle `@main_graph` as a fallback, but static shapes are required.
