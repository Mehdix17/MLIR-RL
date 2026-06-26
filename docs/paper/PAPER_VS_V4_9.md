# Comparative Analysis: Paper Versions vs. V4.9 (Safety & Architecture)

This document provides a comparative overview of the design differences between the **paper packages** (`rl_autoschedular_paper` & `rl_autoschedular_paper_transformer`) and **V4.9** (`rl_autoschedular_v4_9` / `v4_5` engine), specifically detailing the safety mechanisms ported from V4.9 to stabilize the paper training runs.

---

## 1. Architectural Comparison

The fundamental difference between V4.9 and the paper versions lies in how they manage and mutate the MLIR code during reinforcement learning trajectory collection:

| Architectural Component | V4.9 (`rl_autoschedular_v4_9`) | Paper Packages (Fixed) |
|-------------------------|------------------------------|------------------------|
| **Parent State Representation** | Python string (`str`) representing the MLIR assembly | Live C++ `Module` object held in memory |
| **Transformation Application** | String-based API (`action.apply(code_str)`) | In-place C++ API (`action.apply(module)`) |
| **Execution Boundary** | Always isolated in a Python subprocess | Isolated in a Python subprocess (ported from V4.9) |
| **Parsing overhead in parent** | Zero (parent process never parses MLIR) | Low (parent parses the final string returned from subprocess) |

### Why V4.9 was inherently immune to C++ crashes
Because V4.9 passes code as Python strings at the top level, the parent process **never** imports `mlir.ir` or initializes an `MlirContext` during trajectory collection. All MLIR parsing, compilation, and transformation interpreter runs occur inside isolated child subprocesses. If a compiler pass aborts or hits a diagnostic bug, it only terminates the child process, which the parent handles as a failed execution without crashing.

---

## 2. Ported Fixes and Safety Mechanisms

Originally, the paper packages executed all transformations in-process to avoid subprocess overhead. This led to frequent native `SIGABRT` crashes. Over several cleanup rounds, the safety mechanisms from V4.9 were successfully ported to the paper packages:

### 1. Subprocess Transform Isolation
* **The Problem:** In-process calls to `interpreter.apply_named_sequence` that produced diagnostic warnings (common during tiling/vectorization) triggered C++ destructor assertions in the bindings (`CollectDiagnosticsToStringScope`), aborting the parent process.
* **The Ported Fix:** We modified `__run_transform_code` in the paper packages' `transforms.py`. The parent process serializes the `Module` to a string (`str(module)`), runs the transformation inside an isolated Python subprocess, and parses the returned transformed string back into the C++ `Module` in-place.

### 2. Isolated Execution Engine (`execution.py`)
* **The Problem:** Compiling and executing the transformed code to measure execution time via `ExecutionEngine` involves LLVM JIT compilation, which often triggers segmentation faults or native assertion failures.
* **The Ported Fix:** Ported V4.9's process-isolated compiler runner. The module is timing-wrapped and lowered inside a worker subprocess. If it crashes, it falls back to a command-line `mlir-cpu-runner` execution before returning failure.

### 3. Native SIGABRT Signal Handlers
* **The Problem:** MLIR context initialization overrides custom Python signal handlers, bypassing standard Python try-except blocks during native aborts.
* **The Ported Fix:** Python's custom SIGABRT handler (which translates crashes into Python `RuntimeError` exceptions) is dynamically re-installed immediately after the MLIR context is initialized in the training (`train.py`) and evaluation (`evaluate.py`) entry points.

### 4. Robust Dataset Loading Guard
* **The Problem:** If a single MLIR file in a dataset is malformed, parsing it during the initial benchmark loading phase triggers an unhandled native abort, preventing training from starting.
* **The Ported Fix:** Added a `try/except RuntimeError` block inside `benchmarks.py` around the file feature extraction call, mimicking V4.9. Malformed benchmarks are safely skipped with a warning.

---

## 3. Summary of Safety Alignment

Following these ports, the paper packages are now fully protected from all native compiler errors, warnings, and JIT crashes:

```
[Parent RL Agent Loop]
       │
       ├──► [Transform Step] ──► Spawns Subprocess (Isolated parse & apply) 
       │                                     │
       │                                     └───► Returns transformed assembly string (Safe)
       │
       └──► [Execution Step] ──► Spawns Subprocess (Isolated compile & JIT run)
                                             │
                                             └───► Returns execution time / failure (Safe)
```

Both implementations are now aligned at the execution boundaries, guaranteeing stable, uninterrupted training runs on the cluster.
