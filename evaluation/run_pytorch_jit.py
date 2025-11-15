"""
Execute PyTorch JIT-compiled models
"""

import time
import torch
import json
import sys
from pathlib import Path
from typing import Dict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PyTorchJITExecutor:
    """Execute JIT-compiled PyTorch models"""
    
    def __init__(self, models_dir: str, output_dir: str):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def execute_jit_model(self, jit_file: Path, num_runs: int = 100) -> Dict:
        """Execute JIT-compiled model"""
        benchmark_name = jit_file.stem.replace('_jit', '')
        print(f"\n⚡ PyTorch JIT: {benchmark_name}")
        
        # Load JIT model
        jit_model = torch.jit.load(jit_file)
        jit_model.eval()
        
        # Get example input based on benchmark name
        example_input = self._get_example_input(benchmark_name)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = jit_model(example_input)
        
        # Benchmark
        exec_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = jit_model(example_input)
                end = time.perf_counter()
                exec_times.append((end - start) * 1000)  # Convert to ms
        
        exec_times = np.array(exec_times)
        
        result = {
            'method': 'PyTorch-JIT',
            'benchmark': benchmark_name,
            'num_runs': num_runs,
            'mean_time_ms': float(exec_times.mean()),
            'min_time_ms': float(exec_times.min()),
            'max_time_ms': float(exec_times.max()),
            'std_time_ms': float(exec_times.std()),
            'median_time_ms': float(np.median(exec_times)),
        }
        
        print(f"  Mean time: {result['mean_time_ms']:.3f} ms")
        return result
    
    def _get_example_input(self, benchmark_name: str):
        """Get example input based on benchmark name"""
        if 'matmul_small' in benchmark_name:
            return torch.randn(256, 256)
        elif 'matmul_medium' in benchmark_name:
            return torch.randn(512, 512)
        elif 'matmul_large' in benchmark_name:
            return torch.randn(1024, 1024)
        elif 'conv2d_small' in benchmark_name:
            return torch.randn(8, 64, 28, 28)
        elif 'conv2d_medium' in benchmark_name:
            return torch.randn(16, 128, 56, 56)
        elif 'conv2d_large' in benchmark_name:
            return torch.randn(32, 256, 112, 112)
        elif 'resnet_block' in benchmark_name:
            return torch.randn(32, 64, 56, 56)
        elif 'linear_small' in benchmark_name:
            return torch.randn(64, 512)
        elif 'linear_large' in benchmark_name:
            return torch.randn(256, 2048)
        else:
            return torch.randn(32, 64, 56, 56)
    
    def run_all_benchmarks(self) -> Dict:
        """Run all JIT benchmarks"""
        print("\n" + "="*60)
        print("Running PyTorch JIT Benchmarks")
        print("="*60)
        
        jit_files = sorted(self.models_dir.glob("*_jit.pt"))
        
        if not jit_files:
            print(f"No JIT models found in {self.models_dir}")
            return {}
        
        for jit_file in jit_files:
            try:
                result = self.execute_jit_model(jit_file)
                benchmark_name = jit_file.stem.replace('_jit', '')
                self.results[benchmark_name] = result
            except Exception as e:
                print(f"  ✗ Error processing {jit_file.name}: {e}")
                benchmark_name = jit_file.stem.replace('_jit', '')
                self.results[benchmark_name] = {'error': str(e)}
        
        # Save results
        results_file = self.output_dir / "pytorch_jit_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ PyTorch JIT results saved to {results_file}")
        return self.results


def main():
    """Main execution function"""
    executor = PyTorchJITExecutor(
        models_dir="benchmarks/pytorch",
        output_dir="evaluation/results"
    )
    
    results = executor.run_all_benchmarks()
    
    print("\n" + "="*60)
    print("PyTorch JIT Benchmarks Complete")
    print("="*60)


if __name__ == "__main__":
    main()
