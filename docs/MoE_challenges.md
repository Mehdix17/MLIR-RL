# MoE Model → MLIR Benchmark: Challenges & Solutions

A reference guide for converting Mixture-of-Experts (MoE) model weights into MLIR benchmarks
for training an RL optimization agent.

---

## Pipeline Overview

```
MoE Model Weights (PyTorch / HuggingFace)
        ↓
Export (ONNX  /  torch.export()  /  JAX)
        ↓
MLIR Conversion (onnx-mlir / torch-mlir / IREE)
        ↓
Graph Slicing → Individual Benchmark Units
        ↓
Validation (mlir-opt)
        ↓
RL Agent Training Dataset
```

---

## Challenge 1 — Dynamic Shapes in MoE Routing

### Problem

MoE models use **dynamic token dispatch**: the number of tokens routed to each expert varies
per inference step. This makes the computational graph shape-dynamic, which most MLIR lowering
tools handle poorly or reject entirely.

Symptoms:
- `onnx-mlir` fails with `unknown rank` or `dynamic dim` errors
- `torch-mlir` emits `?` dimensions that block lowering to `linalg`
- Shape inference passes silently produce incorrect tile sizes

### Solutions

**Option A — Static shape specialization (recommended for benchmarking)**

Fix shapes at export time using a representative token budget:

```python
# Fix: export with concrete shapes — no dynamic dims
torch.onnx.export(
    expert_block,
    torch.randn(64, 4096),        # fix batch=64, hidden=4096
    "expert.onnx",
    dynamic_axes=None,            # disable dynamic axes entirely
    opset_version=17
)
```

Then generate multiple exports with different fixed shapes (e.g., 16, 32, 64, 128 tokens)
to cover shape diversity in your benchmark dataset.

**Option B — Symbolic shape annotation**

Use `torch.export()` with explicit constraints:

```python
from torch.export import export, Dim

batch = Dim("batch", min=1, max=128)
exported = export(
    expert_block,
    (torch.randn(64, 4096),),
    dynamic_shapes={"x": {0: batch}}
)
```

Then lower with `torch-mlir` using `--shape-dtype-refinement`.

**Option C — IREE shape inference**

IREE's importer handles dynamic shapes better than raw `onnx-mlir`:

```bash
iree-import-onnx expert.onnx \
  --opset-version=17 \
  --mlir-print-debuginfo \
  -o expert.mlir
```

---

## Challenge 2 — Full Model Too Large to Export

### Problem

Full MoE models (Mixtral 8x7B = ~90GB, DeepSeek-V3 = ~650GB) cannot be exported as a
single ONNX graph — OOM during tracing, or files too large to process.

### Solutions

**Export layer by layer**

Trace and export individual components rather than the full model:

```python
from transformers import MixtralModel
import torch

model = MixtralModel.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    device_map="cpu",
    torch_dtype=torch.float32
)

# Export one transformer layer at a time
for layer_idx in range(len(model.layers)):
    layer = model.layers[layer_idx]
    dummy = torch.randn(1, 32, 4096)
    torch.onnx.export(
        layer,
        dummy,
        f"layer_{layer_idx}.onnx",
        opset_version=17
    )
```

**Export individual experts (finest granularity)**

Single expert FFN blocks are small (~50MB each) and map cleanly to `linalg` ops:

```python
# Single expert: gate_proj, up_proj, down_proj
expert = model.layers[0].block_sparse_moe.experts[0]
torch.onnx.export(expert, torch.randn(1, 4096), "expert_0_0.onnx")
```

**Subgraph taxonomy for benchmark diversity**

| Subgraph | Size | Key Ops | RL Optimization Target |
|---|---|---|---|
| Single expert FFN | Small | `matmul → silu → matmul` | Op fusion |
| Gating network | Small | `matmul → softmax → topk` | Control flow |
| Attention block | Medium | `batch_matmul → softmax → batch_matmul` | Tiling |
| Full MoE layer | Large | All of the above | End-to-end |

---

## Challenge 3 — Sparse Ops Don't Lower to `linalg`

### Problem

MoE dispatch uses **sparse gather/scatter** patterns (routing tokens to experts) that have no
direct `linalg` dialect equivalent. Tools like `onnx-mlir` may emit them as `tensor.gather`
or leave them in `tosa` dialect without lowering further.

```mlir
# This may get stuck — no linalg lowering path
%routed = tensor.gather %tokens[%indices] ...
```

### Solutions

**Option A — Densify sparse ops manually**

Replace sparse dispatch with an equivalent dense einsum / matmul formulation.
The routing matrix `R ∈ {0,1}^{tokens × experts}` makes dispatch a masked matmul:

```mlir
// Dense equivalent of sparse MoE dispatch
%dispatch = linalg.matmul
    ins(%routing_mask, %expert_weights : tensor<T×E×f32>, tensor<E×D×f32>)
    outs(%result : tensor<T×D×f32>) -> tensor<T×D×f32>
```

This loses sparsity fidelity but produces valid `linalg` benchmarks.

**Option B — Use custom lowering passes**

Write an MLIR pass that pattern-matches `tensor.gather + matmul` and rewrites to `linalg`:

```bash
mlir-opt \
  --pass-pipeline="builtin.module(
      func.func(
          convert-tensor-to-linalg,
          linalg-generalize-named-ops
      )
  )" \
  expert_with_gather.mlir -o expert_linalg.mlir
```

**Option C — Exclude sparse subgraphs from dataset**

For a first iteration, simply filter out any subgraph containing `tensor.gather`,
`tensor.scatter`, or `sparse_tensor` ops and focus benchmarks on the dense FFN experts.
These already cover the most optimization-rich patterns.

---

## Challenge 4 — ONNX → MLIR Type Mismatches

### Problem

ONNX uses its own type system (`FLOAT`, `INT64`, `BOOL`) which doesn't always map cleanly
to MLIR types (`f32`, `i64`, `i1`). Common failures:

- `INT64` indices used in matmul → rejected by `linalg.matmul` (expects float)
- `FLOAT16` weights → `onnx-mlir` may silently upcast or error
- `BOOL` gating masks → no direct `linalg` equivalent

### Solutions

**Cast to f32 before export**

```python
model = model.to(torch.float32)   # avoid fp16 type issues
```

**Use onnx shape inference + type cleanup**

```python
import onnx
from onnx import shape_inference

model_proto = onnx.load("expert.onnx")
inferred = shape_inference.infer_shapes(model_proto)
onnx.save(inferred, "expert_inferred.onnx")
```

**Explicitly cast INT64 indices in the graph**

```bash
# Use onnx-simplifier to clean up type casts
pip install onnxsim
python -m onnxsim expert.onnx expert_simplified.onnx
```

---

## Challenge 5 — Incomplete Lowering to `linalg` Dialect

### Problem

The export pipeline may produce MLIR that mixes dialects (`onnx`, `tosa`, `stablehlo`,
`torch`) without fully lowering to `linalg`. The RL agent expects pure `linalg` dialect.

```bash
# You may get output like this — still in onnx dialect
%0 = "onnx.MatMul"(%A, %B) : (tensor<4096×4096×f32>, ...) -> tensor<...>
```

### Solutions

**Full lowering pipeline with `mlir-opt`**

Chain the appropriate conversion passes for your source dialect:

```bash
# From onnx dialect → linalg
mlir-opt \
  --convert-onnx-to-krnl \
  --convert-krnl-to-affine \
  --convert-affine-to-standard \
  --convert-linalg-to-loops \
  expert.mlir -o expert_linalg.mlir

# From torch dialect → linalg (torch-mlir path)
mlir-opt \
  --torch-backend-to-linalg-on-tensors-backend-pipeline \
  expert_torch.mlir -o expert_linalg.mlir
```

**Use IREE's complete lowering pipeline**

IREE provides a battle-tested end-to-end lowering to `linalg`:

```bash
iree-compile \
  --iree-input-type=onnx \
  --iree-mlir-to-vm-bytecode-module \
  --mlir-print-ir-after=iree-convert-to-linalg \
  expert.onnx -o /dev/null 2> expert_linalg.mlir
```

**Validate dialect purity**

After lowering, verify no foreign ops remain:

```bash
mlir-opt --verify-each expert_linalg.mlir | \
  grep -v "linalg\|arith\|tensor\|func\|memref\|affine\|scf" | \
  head -20   # should be empty
```

---

## Challenge 6 — Benchmark Validity & Semantic Correctness

### Problem

Even syntactically valid MLIR may be semantically wrong — incorrect tensor shapes, type
mismatches between ins/outs, or structurally unsound ops that pass `--verify-each` but
produce wrong results when compiled.

### Solutions

**Use `mlir-opt` as primary filter**

```bash
#!/bin/bash
# validate.sh — run on every generated benchmark
FILE=$1
mlir-opt \
  --verify-each \
  --mlir-print-debuginfo \
  "$FILE" > /dev/null 2>&1

if [ $? -eq 0 ]; then
  echo "VALID: $FILE"
else
  echo "INVALID: $FILE — discarded"
  rm "$FILE"
fi
```

**Round-trip test**

Lower to loops and back — if round-trip succeeds, the benchmark is semantically sound:

```bash
mlir-opt \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --verify-each \
  benchmark.mlir > /dev/null
```

**Numerical verification (gold standard)**

Run the original PyTorch op and the lowered MLIR op on the same input and compare outputs:

```python
import numpy as np

# PyTorch reference
ref_output = expert_block(test_input).detach().numpy()

# MLIR compiled output (via mlir-cpu-runner or IREE)
mlir_output = run_mlir_benchmark("expert_linalg.mlir", test_input)

assert np.allclose(ref_output, mlir_output, atol=1e-4), "Numerical mismatch!"
```

---

## Challenge 7 — Benchmark Diversity for RL Training

### Problem

Naively exporting the same expert block repeatedly produces a dataset with low variance —
the RL agent overfits to a narrow set of op patterns and fails to generalize.

### Solutions

**Shape diversity**

Export the same op with varied tensor dimensions:

```python
shapes = [
    (16, 4096), (32, 4096), (64, 4096), (128, 4096),   # token counts
    (64, 2048), (64, 8192),                              # hidden dims
]
for tokens, hidden in shapes:
    export_expert(tokens, hidden, f"expert_{tokens}x{hidden}.onnx")
```

**Layer diversity**

Sample from different layers — early, middle, and late transformer layers have different
weight distributions:

```python
layer_indices = [0, 4, 8, 16, 24, 31]   # Mixtral has 32 layers
for i in layer_indices:
    export_layer(model.layers[i], layer_idx=i)
```

**Expert diversity**

Each MoE layer has multiple experts (8 in Mixtral). Export all — their weights differ,
giving the RL agent exposure to varied matmul value distributions:

```python
for expert_idx in range(8):
    expert = model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
    export_expert(expert, f"layer{layer_idx}_expert{expert_idx}.onnx")
```

**Diversity metrics to track**

| Metric | Tool | Target |
|---|---|---|
| Op type distribution | `grep linalg. *.mlir \| sort \| uniq -c` | Balanced across op types |
| Shape distribution | Custom script | Cover 1D / 2D / 3D tensors |
| Loop depth | `grep scf.for *.mlir \| wc -l` | Mix of 1–4 loop nests |
| Flop count | `mlir-opt --linalg-analytical-model` | Log-uniform distribution |

---

## Recommended Toolchain Summary

| Task | Tool | Notes |
|---|---|---|
| PyTorch → ONNX | `torch.onnx.export` | Use opset 17, disable dynamic axes |
| PyTorch → MLIR (direct) | `torch-mlir` | Cleaner than ONNX path |
| ONNX → MLIR | `onnx-mlir` or `iree-import-onnx` | IREE more robust |
| MLIR lowering | `mlir-opt` with pass pipeline | Chain dialect conversions |
| Validation | `mlir-opt --verify-each` | Run on every benchmark |
| Numerical check | `mlir-cpu-runner` or IREE runtime | Gold standard verification |
| Shape simplification | `onnxsim` | Before MLIR conversion |

---

## Quick-Start Checklist

- [ ] Export model layer-by-layer, not as a full graph
- [ ] Fix all shapes at export time (no dynamic axes for benchmarking)
- [ ] Cast weights to `float32` before export
- [ ] Run `onnxsim` on ONNX files before MLIR conversion
- [ ] Chain full dialect lowering pipeline to pure `linalg`
- [ ] Validate every benchmark with `mlir-opt --verify-each`
- [ ] Cover all 8 experts per MoE layer
- [ ] Sample from multiple layer depths
- [ ] Track shape and op-type diversity metrics
- [ ] Numerical verify at least a sample of benchmarks against PyTorch reference

---

*Generated for RL-based MLIR compiler optimization research.*