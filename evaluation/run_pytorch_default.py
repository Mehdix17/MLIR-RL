"""
Execute PyTorch models with default compilation (no optimization)
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

# Import benchmark classes
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
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
    executor = PyTorchDefaultExecutor(
        models_dir="benchmarks/pytorch",
        output_dir="evaluation/results"
    )
    
    results = executor.run_all_benchmarks()
    
    print("\n" + "="*60)
    print("PyTorch Default Benchmarks Complete")
    print("="*60)


if __name__ == "__main__":
    main()
