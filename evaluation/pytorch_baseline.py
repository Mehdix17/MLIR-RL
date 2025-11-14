"""
PyTorch baseline for comparison

Measures PyTorch's native performance on benchmarks
to compare against MLIR-optimized versions.
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import numpy as np


class PyTorchBaseline:
    """Measure PyTorch baseline performance"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize PyTorch baseline evaluator
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
    
    def benchmark_matmul(
        self,
        M: int,
        N: int,
        K: int,
        num_runs: int = 100
    ) -> Dict:
        """
        Benchmark matrix multiplication
        
        Args:
            M, N, K: Matrix dimensions (M×K @ K×N = M×N)
            num_runs: Number of runs for timing
            
        Returns:
            Timing results
        """
        # Create random matrices
        A = torch.randn(M, K, device=self.device)
        B = torch.randn(K, N, device=self.device)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(A, B)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            result = torch.matmul(A, B)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        return {
            'operation': 'matmul',
            'shape': f"{M}x{K} @ {K}x{N}",
            'mean_time': float(np.mean(times)),
            'median_time': float(np.median(times)),
            'std_time': float(np.std(times))
        }
    
    def benchmark_conv2d(
        self,
        batch: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_size: int,
        num_runs: int = 100
    ) -> Dict:
        """
        Benchmark 2D convolution
        
        Args:
            batch, in_channels, out_channels: Tensor dimensions
            height, width: Spatial dimensions
            kernel_size: Convolution kernel size
            num_runs: Number of runs
            
        Returns:
            Timing results
        """
        # Create input and conv layer
        input_tensor = torch.randn(batch, in_channels, height, width, device=self.device)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = conv(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                result = conv(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        return {
            'operation': 'conv2d',
            'shape': f"({batch}, {in_channels}, {height}, {width}) -> ({batch}, {out_channels}, {height}, {width})",
            'kernel': kernel_size,
            'mean_time': float(np.mean(times)),
            'median_time': float(np.median(times)),
            'std_time': float(np.std(times))
        }
    
    def benchmark_pooling(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        pool_size: int,
        num_runs: int = 100
    ) -> Dict:
        """
        Benchmark max pooling
        
        Args:
            batch, channels, height, width: Tensor dimensions
            pool_size: Pooling window size
            num_runs: Number of runs
            
        Returns:
            Timing results
        """
        input_tensor = torch.randn(batch, channels, height, width, device=self.device)
        pool = nn.MaxPool2d(pool_size, stride=pool_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = pool(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                result = pool(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        return {
            'operation': 'maxpool',
            'shape': f"({batch}, {channels}, {height}, {width}) -> ({batch}, {channels}, {height//pool_size}, {width//pool_size})",
            'pool_size': pool_size,
            'mean_time': float(np.mean(times)),
            'median_time': float(np.median(times)),
            'std_time': float(np.std(times))
        }
    
    def benchmark_neural_network(
        self,
        model: nn.Module,
        input_shape: Tuple,
        num_runs: int = 50
    ) -> Dict:
        """
        Benchmark complete neural network
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_runs: Number of runs
            
        Returns:
            Timing results
        """
        model = model.to(self.device)
        model.eval()
        
        input_tensor = torch.randn(*input_shape, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                output = model(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        return {
            'operation': 'neural_network',
            'model': model.__class__.__name__,
            'input_shape': input_shape,
            'mean_time': float(np.mean(times)),
            'median_time': float(np.median(times)),
            'std_time': float(np.std(times))
        }
    
    def run_benchmark_suite(self, output_file: Optional[Path] = None) -> Dict:
        """
        Run comprehensive benchmark suite
        
        Args:
            output_file: Optional path to save results
            
        Returns:
            All benchmark results
        """
        print("Running PyTorch Benchmark Suite...")
        print("="*60)
        
        results = {}
        
        # Matrix multiplication benchmarks
        print("\n1. Matrix Multiplication Benchmarks")
        matmul_configs = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024)
        ]
        
        results['matmul'] = []
        for M, N, K in matmul_configs:
            print(f"   - {M}×{K} @ {K}×{N}...")
            result = self.benchmark_matmul(M, N, K)
            results['matmul'].append(result)
            print(f"     Time: {result['median_time']*1000:.2f}ms")
        
        # Convolution benchmarks
        print("\n2. Convolution Benchmarks")
        conv_configs = [
            (1, 3, 64, 224, 224, 3),
            (8, 64, 128, 56, 56, 3),
            (16, 128, 256, 28, 28, 3)
        ]
        
        results['conv2d'] = []
        for batch, in_ch, out_ch, h, w, k in conv_configs:
            print(f"   - Conv2d: ({batch},{in_ch},{h},{w}) → ({batch},{out_ch},{h},{w})...")
            result = self.benchmark_conv2d(batch, in_ch, out_ch, h, w, k)
            results['conv2d'].append(result)
            print(f"     Time: {result['median_time']*1000:.2f}ms")
        
        # Pooling benchmarks
        print("\n3. Pooling Benchmarks")
        pool_configs = [
            (1, 64, 112, 112, 2),
            (8, 128, 56, 56, 2),
            (16, 256, 28, 28, 2)
        ]
        
        results['pooling'] = []
        for batch, ch, h, w, pool_size in pool_configs:
            print(f"   - MaxPool: ({batch},{ch},{h},{w}) with {pool_size}×{pool_size}...")
            result = self.benchmark_pooling(batch, ch, h, w, pool_size)
            results['pooling'].append(result)
            print(f"     Time: {result['median_time']*1000:.2f}ms")
        
        print("\n" + "="*60)
        print("Benchmark suite complete!")
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to {output_file}")
        
        return results


def main():
    """Example usage"""
    # CPU baseline
    print("CPU Baseline:")
    cpu_baseline = PyTorchBaseline(device="cpu")
    cpu_results = cpu_baseline.run_benchmark_suite(
        output_file=Path("results/pytorch_baseline_cpu.json")
    )
    
    # GPU baseline (if available)
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU Baseline:")
        gpu_baseline = PyTorchBaseline(device="cuda")
        gpu_results = gpu_baseline.run_benchmark_suite(
            output_file=Path("results/pytorch_baseline_gpu.json")
        )


if __name__ == "__main__":
    main()
