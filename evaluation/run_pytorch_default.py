"""
Execute PyTorch models with default compilation (no optimization)
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

# Import benchmark classes
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "benchmarks"))
from benchmark_suite import (
    MatMulBenchmark, Conv2DBenchmark, SimpleResNetBlock, LinearBenchmark
)


class PyTorchDefaultExecutor:
    """Execute PyTorch models with default settings (no JIT)"""
    
    def __init__(self, models_dir: str, output_dir: str):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
        # Disable any optimizations
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    
    def execute_model(self, model_file: Path, num_runs: int = 100) -> Dict:
        """Execute PyTorch model with default compilation"""
        benchmark_name = model_file.stem.replace('_pytorch', '')
        print(f"\n🔥 PyTorch Default: {benchmark_name}")
        
        # Load model
        checkpoint = torch.load(model_file)
        model_info = checkpoint['model_info']
        
        # Reconstruct model based on class name
        model = self._reconstruct_model(checkpoint['model_class'], model_info)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # Get example input
        example_input = model.get_example_input()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input)
        
        # Benchmark
        exec_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(example_input)
                end = time.perf_counter()
                exec_times.append((end - start) * 1000)  # Convert to ms
        
        exec_times = np.array(exec_times)
        
        result = {
            'method': 'PyTorch-Default',
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
    
    def _reconstruct_model(self, model_class: str, model_info: Dict):
        """Reconstruct model from saved info"""
        if model_class == 'MatMulBenchmark':
            return MatMulBenchmark(
                model_info['M'],
                model_info['N'],
                model_info['K']
            )
        elif model_class == 'Conv2DBenchmark':
            return Conv2DBenchmark(
                model_info['batch'],
                model_info['in_ch'],
                model_info['out_ch'],
                model_info['size']
            )
        elif model_class == 'SimpleResNetBlock':
            return SimpleResNetBlock(model_info['channels'])
        elif model_class == 'LinearBenchmark':
            return LinearBenchmark(
                model_info['batch'],
                model_info['in_features'],
                model_info['out_features']
            )
        else:
            raise ValueError(f"Unknown model class: {model_class}")
    
    def run_all_benchmarks(self) -> Dict:
        """Run all PyTorch benchmarks"""
        print("\n" + "="*60)
        print("Running PyTorch Default Benchmarks")
        print("="*60)
        
        model_files = sorted(self.models_dir.glob("*_pytorch.pt"))
        
        if not model_files:
            print(f"No PyTorch models found in {self.models_dir}")
            return {}
        
        for model_file in model_files:
            try:
                result = self.execute_model(model_file)
                benchmark_name = model_file.stem.replace('_pytorch', '')
                self.results[benchmark_name] = result
            except Exception as e:
                print(f"  ✗ Error processing {model_file.name}: {e}")
                benchmark_name = model_file.stem.replace('_pytorch', '')
                self.results[benchmark_name] = {'error': str(e)}
        
        # Save results
        results_file = self.output_dir / "pytorch_default_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ PyTorch Default results saved to {results_file}")
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
    
    executor = PyTorchDefaultExecutor(
        models_dir="data/benchmarks/pytorch",
        output_dir=str(output_dir)
    )
    
    results = executor.run_all_benchmarks()
    
    # Rename output file to standard name
    old_file = output_dir / "pytorch_default_results.json"
    new_file = output_dir / "pytorch_output.json"
    if old_file.exists():
        old_file.rename(new_file)
        print(f"\n✓ PyTorch results saved to {new_file}")
    
    print("\n" + "="*60)
    print("PyTorch Default Benchmarks Complete")
    print("="*60)


if __name__ == "__main__":
    main()
