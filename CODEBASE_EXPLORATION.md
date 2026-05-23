# MLIR-RL Codebase Exploration Summary

**Date:** May 23, 2026  
**Scope:** Architecture overview, PyTorch baseline measurements, model loading patterns, and key design insights

---

## 1. Customization Files Found

### 1.1 Files Discovered

| File | Location | Type | Status |
|------|----------|------|--------|
| `AGENTS.md` | `/scratch/mb10856/MLIR-RL/AGENTS.md` | Agent/environment setup guide | ✅ Present |
| `README.md` | `/scratch/mb10856/MLIR-RL/README.md` | Main project overview | ✅ Present |
| `CLAUDE.md` | N/A | AI assistant config | ❌ Not found |
| `.cursorrules` | N/A | Cursor IDE rules | ❌ Not found |
| `copilot-instructions.md` | N/A | Copilot config | ❌ Not found |
| `docs/` READMEs | Multiple docs/ files | Architecture/workflow guides | ✅ Present |

### 1.2 Key Documentation Files in `docs/`

| File | Purpose |
|------|---------|
| `TRAINING_GUIDE.md` | Step-by-step training workflow (configs, setup, pipeline) |
| `RL_AGENT_TUTORIAL.md` | Foundational RL theory + PPO + MLIR integration |
| `FULL_MODEL.md` | End-to-end model optimization details (9 successes, 10 failures, PyTorch baselines) |
| `PIPELINE.md` | Data flow from raw MLIR → training → evaluation |
| `VERSIONS.md` | Changelog for v0–v4.5+ implementations |
| `NOVELTIES.md` | Architectural contributions per version |
| `HPC_setup.md` | Cluster-specific setup (Slurm, LLVM build) |

---

## 2. PyTorch Baseline Architecture

### 2.1 File Locations

| Script | Purpose | Lines |
|--------|---------|-------|
| [scripts/measure_full_model_baselines.py](scripts/measure_full_model_baselines.py) | Measures PyTorch eager + JIT for 22 models | ~380 |
| [scripts/get_pytorch_times.py](scripts/get_pytorch_times.py) | Per-benchmark PyTorch timing (eager, compile, jit) | ~400+ |
| [scripts/get_pytorch_baselines.py](scripts/get_pytorch_baselines.py) | Alternative PyTorch baseline tool | |
| [results/full_model/baselines/full_baselines.csv](results/full_model/baselines/full_baselines.csv) | Output: model timings (MLIR, PyTorch eager, JIT) | |

### 2.2 Model Loaders (22 Models)

**Location:** [scripts/measure_full_model_baselines.py#L30-L205](scripts/measure_full_model_baselines.py)

Models organized by category:

```python
MODEL_LOADERS = {
    # Vision (9)
    "vgg11":              lambda: (_vision_model("vgg11"),              _vision_input()),
    "resnet18":           lambda: (_vision_model("resnet18"),           _vision_input()),
    "resnet50":           lambda: (_vision_model("resnet50"),           _vision_input()),
    "efficientnet_b0":    lambda: (_vision_model("efficientnet_b0"),    _vision_input()),
    "mobilenet_v3_small": lambda: (_vision_model("mobilenet_v3_small"), _vision_input()),
    "resnext50":          lambda: (_vision_model("resnext50"),          _vision_input()),
    "convnext_tiny":      lambda: (_vision_model("convnext_tiny"),      _vision_input()),
    "densenet121":        lambda: (_vision_model("densenet121"),        _vision_input()),
    "vit_b_16":           lambda: (_vision_model("vit_b_16"),           _vision_input()),
    
    # Transformers (7)
    "t5":       _load_t5,
    "gpt2":     _load_gpt2,
    "bert":     _load_bert,
    "distilbert": _load_distilbert,
    "roberta":  _load_roberta,
    "albert":   _load_albert,
    "bart":     _load_bart,
    
    # LSTM (1)
    "lstm":     _load_lstm,
    
    # GNN (2)
    "gcn":      _load_gcn,
    "gat":      _load_gat,
}
```

#### 2.2.1 Vision Model Wrapper Example (ViT)

**Why:** `torch.jit.trace()` requires plain tensor outputs, not wrapper objects.

```python
def _vision_model(name):
    # ... model loading ...
    if name == "vit_b_16":
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1).eval()
        class ViTWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out = self.m(x)
                # Return plain tensor for JIT compatibility
                if hasattr(out, 'logits'):
                    return out.logits
                return out
        return ViTWrapper(model)
```

#### 2.2.2 Transformer Model Wrapper Example (BERT)

```python
def _load_bert():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval()
    enc = tokenizer("Hello from MLIR", return_tensors="pt",
                     padding="max_length", max_length=16, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    class BertWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ids, mask):
            # Extract tensor from HuggingFace output dict
            return self.m(input_ids=ids, attention_mask=mask).last_hidden_state
    
    return BertWrapper(model), (input_ids, attention_mask)
```

### 2.3 PyTorch Timing Pipeline

**Location:** [scripts/measure_full_model_baselines.py#L255-L280](scripts/measure_full_model_baselines.py)

```python
def _time_jit(model, inputs, n_warmup=WARMUP, n_measure=MEASURE):
    """Trace + time. Falls back to script if trace fails. Returns median ns or None."""
    try:
        # PRIMARY: torch.jit.trace
        traced = torch.jit.trace(model, inputs)
        traced = traced.eval()
        for _ in range(n_warmup + 10):
            _run_model(traced, inputs)
        return _time_model(traced, inputs, n_warmup, n_measure)
    except Exception:
        pass
    
    # FALLBACK: torch.jit.script (for Python-control-flow models)
    try:
        scripted = torch.jit.script(model)
        scripted = scripted.eval()
        for _ in range(n_warmup + 10):
            _run_model(scripted, inputs)
        return _time_model(scripted, inputs, n_warmup, n_measure)
    except Exception as e:
        print(f"    JIT script also failed: {e}")
        return None
```

### 2.4 Why JIT Fails: Root Causes

**Document:** [docs/FULL_MODEL.md#PyTorch-Baselines](docs/FULL_MODEL.md)

| Model | Issue | Type | Failure Mode |
|-------|-------|------|--------------|
| **gpt2** | `torch.jit.trace()` doesn't support cached attention states | Control Flow | ❌ JIT FAILED |
| **albert** | Dictionary outputs + conditional logic in forward pass | Output Type | ❌ JIT FAILED |
| **bert** | Same as albert | Output Type | ❌ JIT FAILED |
| **bart** | Encoder-decoder conditional merging logic | Control Flow | ❌ JIT FAILED |
| **distilbert** | Similar to bert | Output Type | ❌ JIT FAILED |
| **roberta** | AST dumper tool failure (dialect mismatch) | External Tool | ⚠️ Not JIT, upstream |
| **convnext_tiny** | MLIR bufferization failure (not PyTorch JIT) | Compilation | ⚠️ Structural MLIR |
| **deberta** | Complex attention mechanisms + type mismatches | Control Flow | ❌ JIT FAILED |
| **densenet121** | Unlikely to fail; possible edge case | Rare | ? |
| **vit_b_16** | Base model fails → wrapped successfully | Wrapper Fix | ✅ Workaround |

**Success Rate:** 18/22 models traced successfully (81.8%)

### 2.5 Measurement Strategy

**Warmup + Median Approach:**
```python
WARMUP = 10  # skip caches
MEASURE = 20  # repeated runs for stability

def _time_model(model, inputs, n_warmup=WARMUP, n_measure=MEASURE):
    """Return median execution time in ns."""
    for _ in range(n_warmup):
        _run_model(model, inputs)
    
    times = []
    for _ in range(n_measure):
        start = time.perf_counter_ns()
        _run_model(model, inputs)
        end = time.perf_counter_ns()
        times.append(end - start)
    
    return int(np.median(times))
```

---

## 3. Model Loading Patterns

### 3.1 Three Main Sources

#### 3.1.1 TorchVision Models (Vision)

**Framework:** `torchvision.models`  
**Format:** Direct model instantiation + `eval()` mode  
**Example:**

```python
import torchvision.models as tv_models

model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1).eval()
input_tensor = torch.randn(1, 3, 224, 224)  # ImageNet standard
```

#### 3.1.2 HuggingFace Transformers

**Framework:** `transformers` (AutoTokenizer, AutoModel)  
**Format:** Tokenizer + pretrained encoder/decoder  
**Example:**

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()

# Encode text input
encoded = tokenizer(text, return_tensors="pt", padding="max_length", max_length=16, truncation=True)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

# Returns dict with .last_hidden_state
output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
```

#### 3.1.3 Custom/Lightweight Models (GNN, LSTM)

**Format:** Manually defined `torch.nn.Module`  
**Example (GNN):**

```python
# From: data_utils/gnn2mlir.py
from data_utils.gnn2mlir import GCN

model = GCN().eval()
inputs = (torch.randn(128, 64), torch.randn(128, 128))  # nodes, adjacency
```

### 3.2 Data Flow: Model → MLIR

**Pipeline:** [docs/PIPELINE.md](docs/PIPELINE.md)

```
PyTorch Model
    ↓
[ONNX Export] ← torch.onnx.export()
    ↓
[Shape Inference] ← onnxruntime.symbolic_shape_infer()
    ↓
[torch-mlir Import] ← torch_mlir.tools.import_onnx()
    ↓
[Linalg Lowering] ← mlir-opt -convert-torch-to-linalg
    ↓
MLIR Linalg-on-Tensors Code
    ↓
[AST Dumper] ← C++ tool (AST_DUMPER_BIN_PATH)
    ↓
Structured Text:
  - #START_OPERATION blocks (with {tag = "operation_NNN"})
  - #START_NESTED_LOOPS (loop bounds, iterator types)
  - #START_LOAD_DATA / #START_STORE_DATA
  - #BEGIN_GRAPH (producer→consumer edges)
    ↓
[RL Agent Feature Extraction] ← state.py
    ↓
Observation Tensor (for policy)
```

### 3.3 Model Loading Entry Points

**Primary:** [scripts/measure_full_model_baselines.py](scripts/measure_full_model_baselines.py)  
**Alternative:** [data_utils/transformers2mlir.py](data_utils/transformers2mlir.py)

---

## 4. Feature Extraction & Observation Construction

### 4.1 Architecture: State → Observation → Policy

**Location:** [rl_autoschedular/rl_autoschedular_v4_5/](rl_autoschedular/rl_autoschedular_v4_5/)

```
MLIR Code + AST Dumper Output
    ↓
[extract_bench_features_from_code] ← state.py:extract_bench_features_from_code()
    ↓
BenchmarkFeatures:
  - bench_name (str)
  - code (str) – raw MLIR
  - operation_tags (list[str])
  - operations (dict[str, OperationFeatures])
  - root_exec_time (int, nanoseconds)
    ↓
[OperationState] ← For each operation:
  - operation_features: OperationFeatures
  - producer_features: Optional[OperationFeatures]
  - transformation_history: list[list[Action]]
  - terminal (bool)
    ↓
[Observation.from_state()] ← observation.py
    ↓
torch.Tensor (observation): [
  OpFeatures (op_type_onehot, loop_bounds, ...),
  ProducerOpFeatures (same as above),
  ActionHistory (multi-hot of prev actions),
  NumLoops (scalar),
  ActionMask (bitmask of legal actions)
]
    ↓
[Policy/Value Networks] ← model.py
    ↓
Action Probabilities + Value Estimate
```

### 4.2 Key Dataclasses

**Location:** [rl_autoschedular/rl_autoschedular_v4_5/state.py](rl_autoschedular/rl_autoschedular_v4_5/state.py)

```python
@dataclass
class OperationFeatures:
    operation_name: str
    operation_type: OperationType  # enum: Generic, Matmul, Conv, Pooling, Add
    op_count: dict[str, int]  # {mul, add, ...}
    load_data: list[list[str]]  # access patterns per dimension
    store_data: list[list[str]]
    nested_loops: list[NestedLoopFeatures]  # bounds, steps, iterator types
    producers: list[tuple[str, int]]  # (producer_tag, operand_idx)
    consumers: list[tuple[str, int]]
    vectorizable: bool
    pre_actions: list[Action]

@dataclass
class BenchmarkFeatures:
    bench_name: str
    code: str  # full MLIR
    operation_tags: list[str]
    operations: dict[str, OperationFeatures]  # tag → features
    root_exec_time: int  # baseline nanoseconds

@dataclass
class OperationState:
    bench_name: str
    operation_tag: str  # identifies op in MLIR
    operation_features: OperationFeatures
    producer_features: Optional[OperationFeatures]
    transformation_history: list[list[Action]]  # sequences of transforms
    terminal: bool
```

### 4.3 Hardware-Aware Observation (V1 Innovation)

**Location:** [rl_autoschedular/rl_autoschedular_v4_5/observation.py](rl_autoschedular/rl_autoschedular_v4_5/observation.py)

Automatic hardware detection on each machine:

```python
def _build_hardware_vector() -> torch.Tensor:
    """Detect: cache sizes, cores, SIMD width, clock speed."""
    l1_kb = _read_cache_level_kb(1)      # from /sys/devices/system/cpu/cpu0/cache/
    l2_kb = _read_cache_level_kb(2)
    l3_kb = _read_cache_level_kb(3)
    logical_cores = os.cpu_count()
    physical_cores = _detect_physical_cores(cpuinfo_text, logical_cores)
    simd_width = _detect_simd_width(cpuinfo_text)  # AVX512/AVX2/SSE2/NEON
    clock_mhz = _detect_clock_mhz(cpuinfo_text)    # from /proc/cpuinfo
    
    # Normalize to 7-element vector (hardness normalized)
    return torch.tensor([
        l1_kb / 256.0,          # L1 (KB)
        l2_kb / 4096.0,         # L2 (KB)
        l3_kb / 65536.0,        # L3 (KB)
        physical_cores / 256.0,
        logical_cores / 512.0,
        simd_width / 1024.0,
        clock_mhz / 6000.0,
    ], dtype=torch.float32)

HARDWARE_VECTOR = _build_hardware_vector()
```

This vector is **concatenated** to the LSTM output in the forward pass, allowing the policy/value networks to adapt scheduling decisions to the target hardware.

---

## 5. Block-Based vs Full-Model Optimization

### 5.1 Two Execution Paths

#### 5.1.1 Full-Model (9 successes)

**Pipeline:** [scripts/optimize_full_model.py](scripts/optimize_full_model.py)

**Process:**
1. Preprocess: Inject {tag} attributes via AST dumper
2. Add @nanoTime() timing wrapper
3. Measure baseline (no transforms)
4. For each operation: Run RL agent (greedy) → get optimal schedule
5. Apply all schedules in-place to full model
6. Measure optimized time

**Models succeeding (bufferization-compatible):**
- t5, lstm, vgg11, resnet18, resnet50, resnext50, gcn, efficientnet_b0, mobilenet_v3_small

**Speedup:** Average 2.67x (t5: 10.07x!)

#### 5.1.2 Block-Based (10 failures + fallback)

**Pipeline:** [scripts/optimize_model_via_blocks.py](scripts/optimize_model_via_blocks.py)

**Process:**
1. Extract operations into blocks (window=5, stride=3)
2. Filter to compute-heavy ops (matmul, conv)
3. Run RL agent per block
4. Sum speedups across blocks

**Models using blocks (bufferization-incompatible):**
- albert, bert, distilbert, bart, deberta, convnext_tiny, densenet121, gat, gpt2, vit_b_16

**Speedup:** Modest 1.08–1.08x (conservative, but stable)

### 5.2 Why Full-Model Fails for Some Models

**Root Causes:**

| Model | Issue | Failure Point |
|-------|-------|---------------|
| **albert** | Missing SSA values (encoder weights not in block context) | `one-shot-bufferize` MLIR pass |
| **bert** | Dictionary outputs + control flow | Torch→Linalg lowering |
| **bart** | Encoder-decoder interaction | Bufferization |
| **gpt2** | KV cache conditionals | JIT trace, MLIR |
| **convnext_tiny** | Vectorization cascades crash LLVM | Bufferization |
| **vit_b_16** | Attention + position embedding indexing | Bufferization |

**Structural issue:** Models must be "self-contained" — all weights, attention masks, and intermediate states must be accessible in the final bufferized MLIR. Models with sparse patterns (encoder → sparse mixer → decoder) break this assumption.

### 5.3 Results Summary

**Full Model (9 successes, checkpoint 715):**

| Model | Baseline | Optimized | Speedup |
|-------|----------|-----------|---------|
| t5 | 818M ns | 81M ns | **10.07x** |
| lstm | 222M ns | 49M ns | 4.53x |
| vgg11 | 11.6B ns | 5.5B ns | 2.13x |
| resnet18 | 2.3B ns | 1.3B ns | 1.83x |

**Block-Based (10 models, checkpoint 1999):**

| Model | Speedup | Heavy Blocks |
|-------|---------|--------------|
| albert | 1.079x | 652 |
| bert | 1.073x | 504 |
| distilbert | 1.079x | 252 |
| vit_b_16 | 1.082x | 402 |

---

## 6. Key Architecture Insights

### 6.1 The Observation Pipeline

**Critical Path:**

```
MLIR Code
    ↓
[C++ AST Dumper] ← subprocess, env var AST_DUMPER_BIN_PATH
    ↓
Structured AST (tags, loops, access patterns, graph)
    ↓
[Python state.py]
    ↓
OperationFeatures, BenchmarkFeatures
    ↓
[Python observation.py]
    ↓
Tensor Observation (operation, producer, actions, mask)
    ↓
[RL Policy Network]
    ↓
Action Probabilities
```

**Why AST Dumper is Critical:**  
The RL agent does **not** parse MLIR directly. The C++ AST dumper extracts:
- Operation metadata (type, name, tag)
- Loop structure (bounds, steps, iterator types)
- Memory access patterns (per dimension)
- Data flow graph (producer→consumer edges)

Without it, feature extraction fails → observation is empty → policy cannot run.

### 6.2 Three-Tier Execution Fallback

**Location:** [rl_autoschedular/rl_autoschedular_v4_5/execution.py](rl_autoschedular/rl_autoschedular_v4_5/execution.py)

```python
def execute_code(code: str, bench_name: str, seq: list[Action], 
                 root_exec_time: Optional[int] = None) -> tuple[int, bool, bool, Optional[str]]:
    """Execute transformed MLIR with three fallback tiers."""
    
    # Tier 1: MLIR Bindings (in-process, fastest)
    try:
        bufferized_code = transform_bufferize_and_lower_v(code)
        real_exec_time, success, error_msg = self.__execute_bufferized_code(bufferized_code, timeout_s)
        if success:
            return real_exec_time, True, True, None
    except Exception as e:
        error_msg = str(e)
    
    # Tier 2: mlir-opt + mlir-cpu-runner (subprocess, robust)
    try:
        real_exec_time, success = self.__execute_code_with_cmd(code, timeout_s)
        if success:
            return real_exec_time, True, True, None
    except Exception:
        pass
    
    # Tier 3: Failure
    return -1, False, True, error_msg
```

### 6.3 Versioning & Implementation Switching

**Location:** [utils/implementation.py](utils/implementation.py)

Each version is a **standalone copy** of the baseline:
- `rl_autoschedular_v0` (baseline) ← original
- `rl_autoschedular_v1` (hardware-aware) ← add hw vector
- `rl_autoschedular_v2` (shaped reward) ← add dense intermediate rewards
- `rl_autoschedular_v3` (transformer) ← replace LSTM with Transformer encoder
- `rl_autoschedular_v4` (extended actions) ← add Pad, Pack, Unroll
- `rl_autoschedular_v4_5` (integrated + hardened) ← v1 + v2 + v3 + resilience
- `rl_autoschedular_v45_no_hw` (ablation) ← v4.5 without hardware awareness
- `rl_autoschedular_v45_no_shaped_reward` (ablation) ← v4.5 without reward shaping
- `rl_autoschedular_v45_no_transformer` (ablation) ← v4.5 with LSTM baseline

**Why Full Copies?**  
Imports are `from rl_autoschedular_v4_5.state import ...`. Mixing versions breaks the namespace. Each version must be self-contained.

### 6.4 Configuration Hierarchy

**Location:** [utils/config.py](utils/config.py) / [config/*.json](config/)

```
CONFIG_FILE_PATH (env var)
    ↓
config/*.json (reads JSON)
    ↓
Config() singleton (first import)
    ↓ (all code accesses same Config instance)
Global state throughout training/eval
```

**Critical:** `CONFIG_FILE_PATH` must be set **before** importing `utils.config.Config`. Otherwise, the Config singleton reads an undefined path.

```bash
# Correct order:
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export CONFIG_FILE_PATH=config/v4_5.json
python scripts/train.py  # Now Config() will read the right file
```

---

## 7. Summary: Data Flow End-to-End

```
┌─────────────────────────────────────────────────────────────┐
│                  Training/Evaluation Pipeline               │
└─────────────────────────────────────────────────────────────┘

1. MLIR Benchmark Preparation
   ├─ vision2mlir.py / transformers2mlir.py (model→ONNX→MLIR)
   ├─ wrap_mlir.py (add @main, @nanoTime)
   └─ extract_blocks.py (window ops into contextual blocks)

2. Baseline Measurement
   ├─ get_base.py (MLIR execution, unoptimized)
   ├─ get_pytorch_times.py (eager + JIT timing)
   └─ split_json.py (stratified train/eval split)

3. Feature Extraction (per benchmark)
   ├─ AST dumper (C++ subprocess)
   ├─ state.py :: extract_bench_features_from_code()
   └─ BenchmarkFeatures + OperationFeatures

4. RL Training
   ├─ observation.py :: Observation.from_state()
   ├─ model.py :: HiearchyModel (LSTM/Transformer encoder + policy/value)
   ├─ ppo.py :: PPO update loop
   └─ execution.py :: [transform → bufferize → execute] × N

5. RL Evaluation
   ├─ eval.py (greedy: pick argmax action per op)
   ├─ optimize_full_model.py (full-model end-to-end)
   ├─ optimize_model_via_blocks.py (block-based fallback)
   └─ results → logs/eval/eval_exec_times.json

6. Dashboard
   └─ dashboard.py (Streamlit: compare speedups across versions)
```

---

## 8. Testing & Validation Entry Points

### 8.1 Sanity Checks

```bash
# MLIR bindings available?
python -c "from mlir.ir import Context; print('OK')"

# Config loading works?
export CONFIG_FILE_PATH=config/v4_5.json
python -c "from utils.config import Config; c = Config(); print(c.implementation)"

# Hardware detection works?
python -c "from rl_autoschedular_v4_5.observation import HARDWARE_VECTOR; print(HARDWARE_VECTOR)"

# Single benchmark optimization?
export CONFIG_FILE_PATH=config/v3_sanity.json
python scripts/train.py  # on tiny dataset

# PyTorch JIT tracing?
python scripts/measure_full_model_baselines.py  # see which models succeed/fail
```

### 8.2 Key Debug Scripts

| Script | Purpose |
|--------|---------|
| [scripts/test_torch_mlir_compile.py](scripts/test_torch_mlir_compile.py) | Verify torch-mlir import + ONNX round-trip |
| [scripts/preprocess_model.py](scripts/preprocess_model.py) | Tag operations via AST dumper |
| [scripts/add_timing_wrapper.py](scripts/add_timing_wrapper.py) | Wrap @main with @nanoTime() |
| [data_utils/extract_blocks.py](data_utils/extract_blocks.py) | Window consumer→producer paths |

---

## 9. Open Questions & Future Work

### 9.1 PyTorch JIT Improvements

**Current:** 18/22 models succeed (81.8%)  
**Failures:** gpt2, albert, bert, bart, distilbert, deberta (6 models)

**Options:**
1. **Wrapper abstractions** — Pre-process models to remove control flow
2. **torch.compile()** — Use torch.compile (TorchDynamo) instead of trace
3. **ONNX export + torch-mlir** — Skip PyTorch JIT entirely, go straight to MLIR

### 9.2 Full-Model Generalization

**Current:** 9/19 models bufferize end-to-end; speedups up to 10.07x  
**Challenge:** Missing SSA values in block extraction for 10 models

**Options:**
1. **SSA reconstruction** — Re-inject producer weights into block context
2. **Larger window sizes** — Increase block size to capture more context
3. **Sparse graph extraction** — Extract actual computation subgraphs (not fixed windows)

---

## 10. File Structure Reference

```
rl_autoschedular/
├── rl_autoschedular_v0/  (baseline — LSTM, 6 actions, sparse reward)
├── rl_autoschedular_v1/  (+ hardware detection)
├── rl_autoschedular_v2/  (+ shaped reward)
├── rl_autoschedular_v2_5/  (+ hardened shaped reward)
├── rl_autoschedular_v3/  (+ Transformer encoder)
├── rl_autoschedular_v4/  (+ extended actions)
├── rl_autoschedular_v4_5/  (+ integrated + resilience) ← PRIMARY
├── rl_autoschedular_v45_no_hw/  (ablation)
├── rl_autoschedular_v45_no_shaped_reward/  (ablation)
└── rl_autoschedular_v45_no_transformer/  (ablation)

scripts/
├── train.py  (PPO training loop)
├── eval.py   (greedy evaluation)
├── get_base.py  (MLIR baseline timing)
├── get_pytorch_times.py  (PyTorch timing)
├── measure_full_model_baselines.py  (full-model PyTorch + MLIR)
├── optimize_full_model.py  (end-to-end RL optimization)
├── optimize_model_via_blocks.py  (block-based fallback)
└── split_json.py  (train/eval split)

data_utils/
├── transformers2mlir.py  (HuggingFace → MLIR)
├── vision2mlir.py  (TorchVision → MLIR)
├── extract_blocks.py  (window extraction)
├── gnn2mlir.py  (GCN/GAT generators)
└── mlir_generators.py  (synthetic benchmarks)

docs/
├── TRAINING_GUIDE.md  (step-by-step)
├── RL_AGENT_TUTORIAL.md  (theory + code walkthrough)
├── FULL_MODEL.md  (end-to-end results)
├── PIPELINE.md  (data flow)
└── VERSIONS.md  (changelog)

utils/
├── config.py  (singleton config)
├── implementation.py  (version switching)
├── bindings_process.py  (disabled: fork safety)
└── log.py  (colored output)
```

---

**End of Exploration Summary**
