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

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

from data_utils.model_catalog import GNN_MODELS
from data_utils._convert_common import (
    PROJECT_ROOT,
    onnx_route,
    direct_route,
    strip_only,
)

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
        h = F.relu(adj @ self.W1(x))
        h = F.relu(adj @ self.W2(h))
        return adj @ self.W3(h)


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
        neigh1 = adj @ x
        h1 = F.relu(self.W1(torch.cat([x, neigh1], dim=-1)))
        neigh2 = adj @ h1
        h2 = F.relu(self.W2(torch.cat([h1, neigh2], dim=-1)))
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
        Wh = W(h).view(N_NODES, N_HEADS, self.head_dim)
        src = Wh.unsqueeze(1).expand(N_NODES, N_NODES, N_HEADS, self.head_dim)
        dst = Wh.unsqueeze(0).expand(N_NODES, N_NODES, N_HEADS, self.head_dim)
        e = F.leaky_relu(a(torch.cat([src, dst], dim=-1)).squeeze(-1))
        alpha = F.softmax(e, dim=1)
        out = (alpha.unsqueeze(-1) * Wh.unsqueeze(0)).sum(dim=1)
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
# Main
# ---------------------------------------------------------------------------

def convert(model_name: str, output_dir: str, backend: str = "onnx",
            strip_weights: bool = False, keep_onnx: bool = False,
            verbose: bool = False,
            static_batch_size: int = 1) -> str:
    """Convert one GNN model. Returns path to produced *_linalg.mlir."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {model_name}...")
    model, example_inputs, in_names, out_names = _load_model_and_inputs(model_name)
    output_path = None

    if backend == "onnx":
        try:
            output_path = onnx_route(
                model, model_name, example_inputs, output_dir,
                input_names=in_names, output_names=out_names,
                opset=18, keep_onnx=keep_onnx,
                static_batch_size=static_batch_size,
                shape_infer="fallback_copy",
                lowering_method="pipeline",
            )
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            print("  Falling back to direct route...")
            output_path = direct_route(
                model, model_name, example_inputs, output_dir,
                static_batch_size=static_batch_size,
            )
    else:
        output_path = direct_route(
            model, model_name, example_inputs, output_dir,
            static_batch_size=static_batch_size,
        )

    strip_only(output_path, strip_weights, verbose)
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
