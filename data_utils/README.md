# data_utils — Reference

## File overview

| File | Purpose |
|---|---|
| `mlir_generators.py` | Core library of MLIR op generators (matmul, conv, bert, vgg, …) |
| `build_benchmark.py` | Build JSON benchmark datasets from the generators |
| `vision2mlir.py` | Export torchvision models → linalg MLIR |
| `transformers2mlir.py` | Export HuggingFace transformer models → linalg MLIR |
| `wrap_mlir.py` | Wrap an existing `.mlir` model file with a timed `@main` function |
| `strip_mlir.py` | Strip large weight constants from MLIR files (in-place or to new file) |
| `orchestrate.py` | Single CLI entry point that dispatches all of the above |
| `tf2mlir.py` | Legacy: TensorFlow → TOSA → linalg MLIR |

---

## Environment setup

```bash
# The mlir conda env requires a newer libstdc++ on this cluster:
export GCC14_LIB=/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-14.2.0-wfwb3ds4a5thcsh5w5o23k6wq7ob5ok3/lib64
export LD_LIBRARY_PATH="$GCC14_LIB:$LD_LIBRARY_PATH"
export PATH="/home/mb10856/envs/mlir/bin:$PATH"
# Project must be on PYTHONPATH for relative imports
export PYTHONPATH="/scratch/mb10856/MLIR-RL:$PYTHONPATH"
```

---

## Generating benchmark datasets (`build_benchmark.py`)

Produces JSON files with timed MLIR operation samples (matmul, conv, etc.).

```bash
# Auto-detects bindings vs cmd backend
python data_utils/build_benchmark.py --output data/benchmarks/out.json

# Force a specific backend
python data_utils/build_benchmark.py --backend bindings --output data/benchmarks/out.json
python data_utils/build_benchmark.py --backend cmd      --output data/benchmarks/out.json
```

---

## Converting vision models (`vision2mlir.py`)

**Pipeline (ONNX route, default):**
`PyTorch → .onnx → _inferred.onnx → _torch.mlir → _linalg.mlir`

**Supported models:** `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenet_v2`,
`mobilenet_v3_small`, `densenet121`, `vit_b_16`, `convnext_tiny/small/base/large`, `vgg11`

```bash
# Basic usage (ONNX route, strip weights, delete ONNX intermediates on success)
python data_utils/vision2mlir.py --model resnet18

# Choose output directory
python data_utils/vision2mlir.py --model vit_b_16 --output-dir data/nn/generated/code_files

# Use direct torch_mlir.fx route (bypasses ONNX — needed for convnext_tiny, resnet50)
python data_utils/vision2mlir.py --model convnext_tiny --backend direct

# Keep .onnx/.onnx.data/_inferred.onnx after success (deleted by default)
python data_utils/vision2mlir.py --model resnet18 --keep-onnx

# Disable weight stripping
python data_utils/vision2mlir.py --model resnet18 --no-strip-weights
```

**Notes:**
- On success, ONNX intermediates (`.onnx`, `.onnx.data`, `_inferred.onnx`) are **automatically deleted** unless `--keep-onnx` is passed. On failure they are kept so the pipeline can resume from step 3/4.
- The unstripped `_linalg.mlir` is automatically copied to `data/nn/non_stripped_models/` before stripping.
- `convnext_tiny` and `resnet50` require `--backend direct` due to ONNX export limitations (`onnx.Loop` / bias shape bug).

---

## Converting transformer models (`transformers2mlir.py`)

**Pipeline (ONNX route, default):**
`PyTorch → .onnx → _inferred.onnx → _torch.mlir → _linalg.mlir`

**Supported models:** `bert`, `distilbert`, `roberta`, `albert`, `gpt2`, `t5`, `bart`, `lstm`

```bash
# Basic usage
python data_utils/transformers2mlir.py --model bert

# Choose output directory
python data_utils/transformers2mlir.py --model distilbert --output-dir data/nn/generated/code_files

# Use direct torch_mlir route (bypasses ONNX)
python data_utils/transformers2mlir.py --model bert --backend direct

# Keep ONNX intermediates after success
python data_utils/transformers2mlir.py --model bert --keep-onnx
```

**Notes:**
- Same cleanup and backup behaviour as `vision2mlir.py` above.
- Requires `transformers` (`pip install transformers`).

---

## Stripping weights from MLIR files (`strip_mlir.py`)

Removes embedded weight constants from large MLIR files. Handles both
`dense<"0xHEX...">` (transformer models) and `dense_resource<torch_tensor_*>`
+ binary `{-# ... #-}` sections (vision models).

```bash
# Strip in-place (overwrites original)
python data_utils/strip_mlir.py path/to/model_linalg.mlir --replace

# Strip to a new file
python data_utils/strip_mlir.py path/to/model_linalg.mlir --output path/to/model_stripped.mlir

# Verbose mode (shows reduction %)
python data_utils/strip_mlir.py path/to/model_linalg.mlir --replace -v
```

---

## Wrapping models for benchmarking (`wrap_mlir.py`)

Adds a timed `@main` entry-point to a linalg MLIR file so it can be used by the RL evaluation loop.

```bash
python data_utils/wrap_mlir.py \
    --input  data/nn/generated/code_files/resnet18_linalg.mlir \
    --model-name resnet18 \
    --output data/nn/code_files/model_resnet18.mlir
```

---

## Using the orchestrator (`orchestrate.py`)

Single entry point that dispatches all subcommands.

```bash
python -m data_utils.orchestrate vision       --model resnet18
python -m data_utils.orchestrate transformer  --model bert
python -m data_utils.orchestrate build-benchmark --output data/benchmarks/out.json
python -m data_utils.orchestrate strip        path/to/model.mlir --replace
python -m data_utils.orchestrate wrap         --input model.mlir --model-name foo --output wrapped.mlir
```

---

## Output directory layout

```
data/nn/
├── generated/code_files/    # _torch.mlir + _linalg.mlir (stripped) per model
├── non_stripped_models/     # automatic backup of each _linalg.mlir before stripping
└── code_files/              # final wrapped .mlir files ready for RL training
```