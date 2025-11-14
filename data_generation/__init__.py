"""
Data generation utilities for MLIR-RL training

This module provides:
- Random MLIR code generation for training
- Neural network → MLIR conversion
- Benchmark suite creation
"""

from .random_mlir_gen import RandomMLIRGenerator
from .nn_to_mlir import NeuralNetworkToMLIR

__all__ = [
    'RandomMLIRGenerator',
    'NeuralNetworkToMLIR',
]
