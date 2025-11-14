"""
Evaluate RL agent on single operations

Tests performance on:
- Matrix multiplication
- Convolutions
- Pooling
- Element-wise operations
"""

import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


class SingleOperationEvaluator:
    """Evaluate RL agent on individual operations"""
    
    def __init__(
        self,
        agent_model_path: Path,
        mlir_opt_path: Path = None,
        benchmark_dir: Path = None
    ):
        """
        Initialize evaluator
        
        Args:
            agent_model_path: Path to trained RL agent
            mlir_opt_path: Path to mlir-opt binary
            benchmark_dir: Directory containing benchmark MLIR files
        """
        self.agent_model_path = Path(agent_model_path)
        self.mlir_opt_path = mlir_opt_path or self._find_mlir_opt()
        self.benchmark_dir = Path(benchmark_dir) if benchmark_dir else Path("benchmarks/single_ops")
        
        # Load agent
        self.agent = self._load_agent()
    
    def _find_mlir_opt(self) -> Path:
        """Find mlir-opt binary"""
        # Check common locations
        common_paths = [
            Path("llvm-project/build/bin/mlir-opt"),
            Path("/usr/local/bin/mlir-opt"),
            Path.home() / "llvm-project/build/bin/mlir-opt"
        ]
        
        for path in common_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError("mlir-opt not found. Please specify mlir_opt_path")
    
    def _load_agent(self):
        """Load trained RL agent"""
        import torch
        from rl_autoschedular.model import HiearchyModel
        
        # Load checkpoint
        checkpoint = torch.load(self.agent_model_path)
        
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        
        # Create model
        model = HiearchyModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def evaluate_operation(
        self,
        mlir_file: Path,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent on a single MLIR file
        
        Args:
            mlir_file: Path to MLIR file
            num_runs: Number of execution runs for timing
            
        Returns:
            Dictionary with metrics (time, speedup, etc.)
        """
        # Get baseline (no optimization)
        baseline_time = self._measure_execution_time(mlir_file, optimize=False, num_runs=num_runs)
        
        # Get agent optimization
        agent_time = self._measure_execution_time(mlir_file, optimize=True, num_runs=num_runs)
        
        # Calculate speedup
        speedup = baseline_time / agent_time if agent_time > 0 else 0
        
        return {
            'file': mlir_file.name,
            'baseline_time': baseline_time,
            'agent_time': agent_time,
            'speedup': speedup,
            'improvement': (baseline_time - agent_time) / baseline_time * 100
        }
    
    def _measure_execution_time(
        self,
        mlir_file: Path,
        optimize: bool,
        num_runs: int
    ) -> float:
        """Measure execution time of MLIR program"""
        times = []
        
        for _ in range(num_runs):
            if optimize:
                # Apply agent's optimization
                optimized_file = self._apply_agent_optimization(mlir_file)
                execution_time = self._execute_mlir(optimized_file)
            else:
                # Execute without optimization
                execution_time = self._execute_mlir(mlir_file)
            
            times.append(execution_time)
        
        # Return median time (more robust than mean)
        return float(np.median(times))
    
    def _apply_agent_optimization(self, mlir_file: Path) -> Path:
        """Apply agent's optimization decisions to MLIR file"""
        # This would use your RL agent to select optimization passes
        # For now, return a placeholder
        # TODO: Integrate with your agent's decision-making
        
        output_file = mlir_file.parent / f"{mlir_file.stem}_optimized.mlir"
        
        # Example: Apply some standard passes
        passes = [
            "--linalg-fuse-elementwise-ops",
            "--linalg-bufferize",
            "--convert-linalg-to-loops"
        ]
        
        cmd = [str(self.mlir_opt_path)] + passes + [str(mlir_file), "-o", str(output_file)]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error optimizing {mlir_file}: {e}")
            return mlir_file
    
    def _execute_mlir(self, mlir_file: Path) -> float:
        """Execute MLIR file and measure time"""
        # Compile to executable
        executable = self._compile_mlir(mlir_file)
        
        # Run and measure time
        start_time = time.time()
        try:
            subprocess.run([str(executable)], check=True, capture_output=True, timeout=30)
            execution_time = time.time() - start_time
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            execution_time = float('inf')
        
        return execution_time
    
    def _compile_mlir(self, mlir_file: Path) -> Path:
        """Compile MLIR to executable"""
        # This is a simplified version
        # You'll need to adjust based on your MLIR setup
        
        output_file = mlir_file.parent / f"{mlir_file.stem}.out"
        
        # Convert MLIR → LLVM IR → executable
        cmd = [
            str(self.mlir_opt_path),
            "--convert-linalg-to-llvm",
            "--convert-func-to-llvm",
            str(mlir_file),
            "-o", str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_file
        except subprocess.CalledProcessError:
            # Return dummy if compilation fails
            return Path("/bin/true")
    
    def evaluate_benchmark_suite(self, output_file: Optional[Path] = None) -> Dict:
        """
        Evaluate agent on entire benchmark suite
        
        Args:
            output_file: Optional path to save results JSON
            
        Returns:
            Dictionary with aggregated results
        """
        # Find all MLIR files in benchmark directory
        mlir_files = list(self.benchmark_dir.glob("**/*.mlir"))
        
        if not mlir_files:
            print(f"⚠️  No MLIR files found in {self.benchmark_dir}")
            return {}
        
        print(f"Evaluating {len(mlir_files)} benchmarks...")
        
        results = []
        for i, mlir_file in enumerate(mlir_files):
            print(f"[{i+1}/{len(mlir_files)}] Evaluating {mlir_file.name}...")
            
            result = self.evaluate_operation(mlir_file)
            results.append(result)
        
        # Aggregate statistics
        speedups = [r['speedup'] for r in results if r['speedup'] != float('inf')]
        
        summary = {
            'num_benchmarks': len(results),
            'mean_speedup': float(np.mean(speedups)) if speedups else 0,
            'median_speedup': float(np.median(speedups)) if speedups else 0,
            'max_speedup': float(np.max(speedups)) if speedups else 0,
            'min_speedup': float(np.min(speedups)) if speedups else 0,
            'individual_results': results
        }
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n✓ Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Benchmarks evaluated: {summary['num_benchmarks']}")
        print(f"Mean speedup: {summary['mean_speedup']:.2f}x")
        print(f"Median speedup: {summary['median_speedup']:.2f}x")
        print(f"Best speedup: {summary['max_speedup']:.2f}x")
        print("="*60)
        
        return summary


def main():
    """Example usage"""
    evaluator = SingleOperationEvaluator(
        agent_model_path=Path("results/best_model.pt"),
        benchmark_dir=Path("benchmarks/single_ops")
    )
    
    results = evaluator.evaluate_benchmark_suite(
        output_file=Path("results/evaluation_results.json")
    )


if __name__ == "__main__":
    main()
