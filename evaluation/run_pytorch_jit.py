"""
Execute PyTorch JIT-compiled models
"""

import time
import torch
import json
import sys
import os
from pathlib import Path
from typing import Dict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load config from environment variable or default
config_file = os.environ.get('CONFIG_FILE', 'config/config.json')
if os.path.exists(config_file):
    print(f"Using config: {config_file}")
else:
    print(f"Config not found: {config_file}, using defaults")


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
    # Check for command-line argument
    if len(sys.argv) >= 2:
        # Argument: output_dir
        output_dir = Path(sys.argv[1])
        run_dir = output_dir.parent
        print(f"\n✓ Using specified output directory: {output_dir}")
        print(f"✓ Model: {run_dir.parent.name}/{run_dir.name}\n")
    else:
        # Auto-detect latest run directory from any model type
        results_dir = Path("results")
        run_dir = None
        
        if results_dir.exists():
            all_run_dirs = []
            for model_type_dir in results_dir.iterdir():
                if model_type_dir.is_dir() and model_type_dir.name != 'comparison_rl_vs_pytorch':
                    run_dirs = [d for d in model_type_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('run_')]
                    all_run_dirs.extend(run_dirs)
            
            if all_run_dirs:
                # Sort by modification time and get latest
                all_run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                run_dir = all_run_dirs[0]
        
        if run_dir is None:
            print("Error: No run directory found in results/")
            print("Train a model first")
            sys.exit(1)
        
        output_dir = run_dir / "benchmarks"
        print(f"\n✓ Using run directory: {run_dir.parent.name}/{run_dir.name}")
        print(f"✓ Results will be saved to: {output_dir}\n")
    
    executor = PyTorchJITExecutor(
        models_dir="data/benchmarks/pytorch",
        output_dir=str(output_dir)
    )
    
    results = executor.run_all_benchmarks()
    
    # Rename output file to standard name
    old_file = output_dir / "pytorch_jit_results.json"
    new_file = output_dir / "pytorch_jit_output.json"
    if old_file.exists():
        old_file.rename(new_file)
        print(f"\n✓ PyTorch JIT results saved to {new_file}")
    
    print("\n" + "="*60)
    print("PyTorch JIT Benchmarks Complete")
    print("="*60)


if __name__ == "__main__":
    main()
