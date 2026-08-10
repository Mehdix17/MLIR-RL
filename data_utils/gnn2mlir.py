#!/usr/bin/env python3
"""
gnn2mlir.py
-----------
Convert Graph Neural Network models to linalg-on-tensors MLIR.

GNNs operate on sparse graphs, but torch-mlir only handles dense tensor
computations. The approach here is to represent each model as a fixed-size
dense graph (N nodes, F features) so that:
  - neighborhood aggregation → dense matrix multiply  (A @ X  or A @ X @ W)
  - node feature transform   → nn.Linear

This preserves all the meaningful operator shapes (matmul, generic activations)
that the RL scheduler will see in practice, while staying fully static.

Supported models: gcn, graphsage, gat, gin

Conversion goes through the ONNX route by default (same pipeline as
vision2mlir / transformers2mlir), with the direct torch_mlir route as fallback.

Usage:
    python data_utils/gnn2mlir.py --model gcn
    python data_utils/gnn2mlir.py --model gat --backend direct
    python data_utils/gnn2mlir.py --model all   # run every supported model
"""

import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import re

from data_utils.model_catalog import GNN_MODELS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "nn", "raw")

SUPPORTED_MODELS = GNN_MODELS

# Fixed graph: 128 nodes, 64 input features, 3 layers of 128 hidden, 32 output
N_NODES    = 128
IN_FEATS   = 64
HIDDEN     = 128
OUT_FEATS  = 32
N_HEADS    = 4   # GAT attention heads


# ---------------------------------------------------------------------------
# Model definitions (dense, static, torch-mlir–compatible)
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """Graph Convolutional Network (Kipf & Welling, 2017).
    Aggregation: H' = ReLU(A_hat @ H @ W)  where A_hat = D^{-1/2} A D^{-1/2}
    Represented as two matmuls + activation — identical to what a sparse GCN
    produces after densification.
    """
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(IN_FEATS, HIDDEN, bias=False)
        self.W2 = nn.Linear(HIDDEN,   HIDDEN, bias=False)
        self.W3 = nn.Linear(HIDDEN,   OUT_FEATS, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   (N, F)   adj: (N, N) normalised adjacency
        h = F.relu(adj @ self.W1(x))         # (N, HIDDEN)
        h = F.relu(adj @ self.W2(h))         # (N, HIDDEN)
        return adj @ self.W3(h)              # (N, OUT_FEATS)


class GraphSAGE(nn.Module):
    """GraphSAGE (Hamilton et al., 2017) — mean aggregation.
    For each layer: h_v = ReLU(W @ concat(h_v, mean(h_N(v))))
    With fixed dense adj, mean aggregation = row-wise mean of adj @ X.
    """
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(IN_FEATS * 2, HIDDEN)
        self.W2 = nn.Linear(HIDDEN * 2,   HIDDEN)
        self.W3 = nn.Linear(HIDDEN * 2,   OUT_FEATS)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Layer 1
        neigh1 = adj @ x                                    # (N, F) mean agg
        h1 = F.relu(self.W1(torch.cat([x, neigh1], dim=-1)))
        # Layer 2
        neigh2 = adj @ h1
        h2 = F.relu(self.W2(torch.cat([h1, neigh2], dim=-1)))
        # Layer 3
        neigh3 = adj @ h2
        return self.W3(torch.cat([h2, neigh3], dim=-1))


class GAT(nn.Module):
    """Graph Attention Network (Velickovic et al., 2018) — multi-head.
    Attention scores: e_ij = LeakyReLU(a^T [W h_i || W h_j])
    For a fixed dense graph this is a batched outer product + softmax + matmul.
    """
    def __init__(self):
        super().__init__()
        self.head_dim = HIDDEN // N_HEADS
        self.W   = nn.Linear(IN_FEATS, HIDDEN, bias=False)
        self.a   = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.W2  = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.a2  = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.out = nn.Linear(HIDDEN, OUT_FEATS)

    def _attn_layer(self, h: torch.Tensor, W: nn.Linear, a: nn.Linear) -> torch.Tensor:
        # h: (N, HIDDEN)
        Wh = W(h).view(N_NODES, N_HEADS, self.head_dim)   # (N, H, d)
        # Broadcast attention: (N, H, d) x (N, H, d) → (N, N, H, 1)
        src = Wh.unsqueeze(1).expand(N_NODES, N_NODES, N_HEADS, self.head_dim)
        dst = Wh.unsqueeze(0).expand(N_NODES, N_NODES, N_HEADS, self.head_dim)
        e = F.leaky_relu(a(torch.cat([src, dst], dim=-1)).squeeze(-1))  # (N,N,H)
        alpha = F.softmax(e, dim=1)                                      # (N,N,H)
        # Aggregate: sum over neighbours weighted by attention
        out = (alpha.unsqueeze(-1) * Wh.unsqueeze(0)).sum(dim=1)        # (N,H,d)
        return out.reshape(N_NODES, HIDDEN)

    def forward(self, x: torch.Tensor, _adj: torch.Tensor) -> torch.Tensor:
        h = F.elu(self._attn_layer(self.W(x), self.W2, self.a2))
        return self.out(h)


class GIN(nn.Module):
    """Graph Isomorphism Network (Xu et al., 2019).
    Aggregation: h_v = MLP((1+eps) h_v + sum_{u in N(v)} h_u)
    sum aggregation = adj @ X (dense); eps is a learnable scalar.
    """
    def __init__(self):
        super().__init__()
        self.eps1 = nn.Parameter(torch.zeros(1))
        self.eps2 = nn.Parameter(torch.zeros(1))
        # MLP per layer (2-layer each)
        self.mlp1 = nn.Sequential(nn.Linear(IN_FEATS, HIDDEN), nn.ReLU(),
                                   nn.Linear(HIDDEN, HIDDEN))
        self.mlp2 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN),   nn.ReLU(),
                                   nn.Linear(HIDDEN, HIDDEN))
        self.mlp3 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN),   nn.ReLU(),
                                   nn.Linear(HIDDEN, OUT_FEATS))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.mlp1((1 + self.eps1) * x   + adj @ x)
        h = self.mlp2((1 + self.eps2) * h   + adj @ h)
        return self.mlp3(h)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _load_model_and_inputs(model_name: str):
    """Return (model, example_inputs, input_names, output_names)."""
    name = model_name.lower()
    x   = torch.randn(N_NODES, IN_FEATS)
    # Row-normalised adjacency (symmetric, values in [0,1]) — static, fully dense
    raw = torch.rand(N_NODES, N_NODES)
    adj = (raw + raw.t()) / 2
    adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-6)

    models_map = {
        "gcn":       (GCN,       (x, adj), ["x", "adj"], ["out"]),
        "graphsage": (GraphSAGE, (x, adj), ["x", "adj"], ["out"]),
        "gat":       (GAT,       (x, adj), ["x", "adj"], ["out"]),
        "gin":       (GIN,       (x, adj), ["x", "adj"], ["out"]),
    }
    if name not in models_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {SUPPORTED_MODELS}")
    cls, inputs, in_names, out_names = models_map[name]
    return cls().eval(), inputs, in_names, out_names


# ---------------------------------------------------------------------------
# Conversion routes (reuse same logic as vision2mlir / transformers2mlir)
# ---------------------------------------------------------------------------

def _specialize_tensor_types(content: str, static_batch_size: int) -> tuple[str, int]:
    """Specialize dynamic tensor dims by replacing '?' with static_batch_size."""
    replaced = 0

    def _repl_tensor(m: re.Match) -> str:
        nonlocal replaced
        token = m.group(0)
        count = token.count("?")
        if count:
            replaced += count
            token = token.replace("?", str(static_batch_size))
        return token

    content = re.sub(r"tensor<[^>]+>", _repl_tensor, content)
    content = re.sub(r"!torch\.vtensor<\[[^\]]+\],[^>]+>", _repl_tensor, content)
    return content, replaced


def _postprocess_linalg_file(linalg_path: str, static_batch_size: int) -> None:
    """Normalize entry name and specialize dynamic shapes in a linalg MLIR file."""
    with open(linalg_path, "r") as fh:
        content = fh.read()

    changed = False
    if "@main_graph" in content:
        content = content.replace("@main_graph", "@main")
        changed = True
        print("  Renamed @main_graph -> @main")

    if static_batch_size > 0 and "?" in content:
        content, replaced = _specialize_tensor_types(content, static_batch_size)
        if replaced:
            changed = True
            print(f"  Specialized dynamic dims with static batch={static_batch_size} ({replaced} replacement(s))")

    if changed:
        with open(linalg_path, "w") as fh:
            fh.write(content)

def _onnx_route(model, model_name: str, example_inputs, output_dir: str,
                strip_weights: bool, keep_onnx: bool = False,
                static_batch_size: int = 1) -> str:
    import onnx
    try:
        import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
        _has_shape_infer = True
    except ImportError:
        _has_shape_infer = False

    base = os.path.join(output_dir, model_name)
    in_names = [f"input_{i}" for i in range(len(example_inputs))]

    print(f"  Exporting {model_name} to ONNX...")
    torch.onnx.export(
        model, example_inputs, f"{base}.onnx",
        opset_version=18,
        input_names=in_names,
        output_names=["output"],
        dynamic_axes=None,
    )

    print("  Running shape inference...")
    if _has_shape_infer:
        try:
            symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
                f"{base}.onnx", f"{base}_inferred.onnx",
                auto_merge=True, int_max=100000, guess_output_rank=True,
            )
        except Exception:
            import shutil
            shutil.copy(f"{base}.onnx", f"{base}_inferred.onnx")
    else:
        m = onnx.load(f"{base}.onnx")
        m2 = onnx.shape_inference.infer_shapes(m)
        onnx.save(m2, f"{base}_inferred.onnx")

    # Import to torch MLIR
    torch_mlir_path = os.path.join(base + "_torch.mlir")
    import_onnx_exe = os.path.join(
        os.path.dirname(sys.executable), "torch-mlir-import-onnx"
    )
    if not os.path.isfile(import_onnx_exe):
        raise RuntimeError("torch-mlir-import-onnx not found; use --backend direct")

    subprocess.run(
        [import_onnx_exe, f"{base}_inferred.onnx", "-o", torch_mlir_path],
        check=True,
    )

    # Lower to linalg
    torch_mlir_opt = os.path.join(os.path.dirname(sys.executable), "torch-mlir-opt")
    linalg_path = os.path.join(output_dir, f"{model_name}_linalg.mlir")
    passes = (
        "torch-lower-to-backend-contract,"
        "torch-backend-to-linalg-on-tensors-backend-pipeline"
    )
    subprocess.run(
        [torch_mlir_opt, torch_mlir_path, f"--pass-pipeline=builtin.module({passes})",
         "-o", linalg_path],
        check=True,
    )

    # Post-process: normalize entrypoint and specialize dynamic dims.
    _postprocess_linalg_file(linalg_path, static_batch_size)

    # Cleanup intermediates
    if not keep_onnx:
        for ext in [".onnx", ".onnx.data", "_inferred.onnx"]:
            p = base + ext
            if os.path.exists(p):
                os.remove(p)
                print(f"  Removed intermediate: {os.path.basename(p)}")

    return linalg_path


def _direct_route(model, model_name: str, example_inputs, output_dir: str,
                  strip_weights: bool, static_batch_size: int = 1) -> str:
    from torch_mlir.compiler_utils import OutputType
    print(f"  Using direct torch_mlir route for {model_name}...")
    mlir_module = None

    try:
        import torch_mlir
        if hasattr(torch_mlir, "compile") and callable(torch_mlir.compile):
            try:
                mlir_module = torch_mlir.compile(
                    model, example_inputs,
                    output_type=OutputType.LINALG_ON_TENSORS,
                    use_tracing=True, func_name="main",
                )
            except TypeError:
                mlir_module = torch_mlir.compile(
                    model, example_inputs,
                    output_type=OutputType.LINALG_ON_TENSORS,
                    func_name="main",
                )
    except Exception as e:
        print(f"  torch_mlir.compile failed: {e}")

    if mlir_module is None:
        from torch_mlir.fx import export_and_import
        if isinstance(example_inputs, tuple):
            mlir_module = export_and_import(
                model, *example_inputs,
                output_type=OutputType.LINALG_ON_TENSORS, func_name="main",
            )
        else:
            mlir_module = export_and_import(
                model, example_inputs,
                output_type=OutputType.LINALG_ON_TENSORS, func_name="main",
            )

    linalg_path = os.path.join(output_dir, f"{model_name}_linalg.mlir")
    with open(linalg_path, "w") as f:
        f.write(str(mlir_module))
    _postprocess_linalg_file(linalg_path, static_batch_size)
    return linalg_path


def _maybe_strip(output_path: str, strip_weights: bool, verbose: bool = False):
    if not strip_weights:
        return
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb > 1:
        print(f"  File is {size_mb:.1f} MB — stripping weights...")
        from data_utils.strip_mlir import strip_weights as do_strip
        reduction = do_strip(output_path, output_path, verbose=verbose)
        print(f"  Stripped: {reduction:.1f}% reduction.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(model_name: str, output_dir: str, backend: str = "onnx",
            strip_weights: bool = False, keep_onnx: bool = False,
            verbose: bool = False,
            static_batch_size: int = 1) -> str:
    """Convert one GNN model. Returns path to produced *_linalg.mlir."""
    import subprocess  # noqa — ensure available in this scope
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {model_name}...")
    model, example_inputs, in_names, out_names = _load_model_and_inputs(model_name)
    output_path = None

    if backend == "onnx":
        try:
            output_path = _onnx_route(model, model_name, example_inputs,
                                       output_dir, strip_weights, keep_onnx,
                                       static_batch_size=static_batch_size)
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            print("  Falling back to direct route...")
            output_path = _direct_route(model, model_name, example_inputs,
                                         output_dir, strip_weights,
                                         static_batch_size=static_batch_size)
    else:
        output_path = _direct_route(model, model_name, example_inputs,
                                     output_dir, strip_weights,
                                     static_batch_size=static_batch_size)

    _maybe_strip(output_path, strip_weights, verbose)
    print(f"\nDone: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Graph Neural Network models to linalg MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="gcn",
        choices=SUPPORTED_MODELS + ["all"],
        help="GNN model to convert. Use 'all' to run every supported model.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/nn/raw).",
    )
    parser.add_argument(
        "--backend", choices=["onnx", "direct"], default="onnx",
        help="Conversion backend.",
    )
    parser.add_argument(
        "--strip-weights", action="store_true", default=False,
    )
    parser.add_argument(
        "--keep-onnx", action="store_true", default=False,
        help="Keep ONNX intermediate files after success.",
    )
    parser.add_argument(
           "--static-batch-size", type=int, default=0,
           help="Static value used to replace dynamic '?' dimensions in final linalg MLIR. "
               "Set 0 to preserve dynamic shapes. Default: 0",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    targets = SUPPORTED_MODELS if args.model == "all" else [args.model]
    for m in targets:
        convert(m, args.output_dir, args.backend,
                args.strip_weights, args.keep_onnx, args.verbose,
                static_batch_size=args.static_batch_size)


if __name__ == "__main__":
    main()
