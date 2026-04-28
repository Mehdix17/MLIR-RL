# Why Some Benchmarks Fail

38,739 `.mlir` files, ~6% succeed. Here is why the other 94% fail.

---

## 1. Input is NOT a Tensor/MemRef — the function signature is wrong

Most files have a function like:

```
func.func @main(%arg0: f32) -> i64 { ... }
```

The `main` function must accept **tensors** or **memrefs** as input (e.g. `tensor<1024xf32>`),
not plain scalars like `f32`, `i64`, or `i32`.

**These files cannot be compiled with the current MLIR pipeline.** They likely came from
an older generator or a different export path.

---

## 2. The MLIR has syntax errors or references missing symbols

Example:

```
'func.call' op 'softmax' does not reference a valid function
```

The MLIR file calls a function (like `softmax`) that is **defined nowhere** in the module.
This is a broken reference — probably the function was meant to be inlined or the
export was incomplete.

---

## 3. No `main` function

Every benchmark must have a `func.func @main(...)` entry point. Some files have only
helper functions (`@matmul`, `@conv2d`, etc.) but no `@main`. These are fragments,
not standalone benchmarks.

---

## 4. Execution times out (takes >120 seconds)

A few files compile and run but take **minutes** instead of seconds. This happens when:
- The input tensors are very large (e.g. batch size 2048)
- The MLIR pass pipeline struggles with certain loop structures
- The compiled code does heavy computation

These are genuine benchmarks but too slow for RL training. The 120s timeout skips them.

---

## 5. PyTorch conversion fails (only in `get_pytorch_times.py`)

When measuring PyTorch baselines, some MLIR files fail because:
- The linalg operation has no PyTorch equivalent (`pooling_nhwc_min`, `fill`, `transpose`)
- Not enough arguments to form a PyTorch call

This is a **different tool** (`get_pytorch_times.py`), not related to `get_base.py`.

---

## Summary

| Cause | Frequency | Can we fix it? |
|---|---|---|
| Input is a scalar, not a tensor | ~60% | ❌ Bad source data |
| Missing function references | ~20% | ❌ Bad source data |
| No `main` function | ~10% | ❌ Bad source data |
| Execution timeout (>120s) | ~4% | ✅ Timeout handles it |
| PyTorch conversion fails | varies | ❌ PyTorch-incompatible ops |

**The data in `data/all/` is a mix of valid benchmarks and broken/partial files.**
The ~6% that succeed are the only ones usable for RL training.
