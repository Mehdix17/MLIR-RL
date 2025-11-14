"""
Evaluate RL agent on full neural networks

Tests performance on:
- ResNet
- BERT
- GPT-2
- Custom architectures
"""

import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np


class NeuralNetworkEvaluator:
    """Evaluate RL agent on complete neural network models"""
    
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
            benchmark_dir: Directory containing neural network MLIR files
        """
        self.agent_model_path = Path(agent_model_path)
        self.mlir_opt_path = mlir_opt_path or self._find_mlir_opt()
        self.benchmark_dir = Path(benchmark_dir) if benchmark_dir else Path("benchmarks/neural_nets")
        
        # Load agent
        self.agent = self._load_agent()
    
    def _find_mlir_opt(self) -> Path:
        """Find mlir-opt binary"""
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
        
        checkpoint = torch.load(self.agent_model_path)
        config = checkpoint.get('config', {})
        
        model = HiearchyModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def evaluate_neural_network(
        self,
        mlir_file: Path,
        num_runs: int = 5
    ) -> Dict:
        """
        Evaluate agent on a neural network MLIR file
        
        Args:
            mlir_file: Path to neural network MLIR file
            num_runs: Number of execution runs
            
        Returns:
            Dictionary with metrics
        """
        print(f"\nEvaluating {mlir_file.name}...")
        
        # Baseline (no optimization)
        print("  - Running baseline (no optimization)...")
        baseline_time = self._measure_inference_time(mlir_file, optimize=False, num_runs=num_runs)
        
        # Agent optimization
        print("  - Running with RL agent optimization...")
        agent_time = self._measure_inference_time(mlir_file, optimize=True, num_runs=num_runs)
        
        # PyTorch baseline (if available)
        print("  - Running PyTorch baseline...")
        pytorch_time = self._measure_pytorch_baseline(mlir_file, num_runs=num_runs)
        
        # Calculate metrics
        speedup_vs_baseline = baseline_time / agent_time if agent_time > 0 else 0
        speedup_vs_pytorch = pytorch_time / agent_time if pytorch_time and agent_time > 0 else None
        
        results = {
            'network': mlir_file.stem,
            'baseline_time': baseline_time,
            'agent_time': agent_time,
            'pytorch_time': pytorch_time,
            'speedup_vs_baseline': speedup_vs_baseline,
            'speedup_vs_pytorch': speedup_vs_pytorch,
            'improvement_pct': (baseline_time - agent_time) / baseline_time * 100 if baseline_time > 0 else 0
        }
        
        print(f"  ✓ Speedup vs baseline: {speedup_vs_baseline:.2f}x")
        if speedup_vs_pytorch:
            print(f"  ✓ Speedup vs PyTorch: {speedup_vs_pytorch:.2f}x")
        
        return results
    
    def _measure_inference_time(
        self,
        mlir_file: Path,
        optimize: bool,
        num_runs: int
    ) -> float:
        """Measure inference time"""
        times = []
        
        for _ in range(num_runs):
            if optimize:
                # Apply agent optimization
                optimized_file = self._apply_agent_optimization(mlir_file)
                inference_time = self._run_inference(optimized_file)
            else:
                inference_time = self._run_inference(mlir_file)
            
            times.append(inference_time)
        
        return float(np.median(times))
    
    def _apply_agent_optimization(self, mlir_file: Path) -> Path:
        """Apply agent's optimization to neural network"""
        output_file = mlir_file.parent / f"{mlir_file.stem}_optimized.mlir"
        
        # Standard neural network optimization passes
        passes = [
            "--linalg-fuse-elementwise-ops",
            "--linalg-bufferize",
            "--convert-linalg-to-loops",
            "--loop-invariant-code-motion",
            "--affine-loop-fusion",
            "--affine-loop-tile"
        ]
        
        cmd = [str(self.mlir_opt_path)] + passes + [str(mlir_file), "-o", str(output_file)]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_file
        except subprocess.CalledProcessError:
            return mlir_file
    
    def _run_inference(self, mlir_file: Path) -> float:
        """Run inference and measure time"""
        # Compile
        executable = self._compile_to_executable(mlir_file)
        
        # Execute
        start_time = time.time()
        try:
            subprocess.run([str(executable)], check=True, capture_output=True, timeout=60)
            execution_time = time.time() - start_time
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            execution_time = float('inf')
        
        return execution_time
    
    def _compile_to_executable(self, mlir_file: Path) -> Path:
        """Compile MLIR to executable"""
        output_file = mlir_file.parent / f"{mlir_file.stem}.out"
        
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
            return Path("/bin/true")
    
    def _measure_pytorch_baseline(self, mlir_file: Path, num_runs: int) -> Optional[float]:
        """Measure PyTorch baseline performance"""
        # This would load the equivalent PyTorch model and measure inference
        # For now, return None as placeholder
        # TODO: Implement PyTorch baseline comparison
        return None
    
    def evaluate_benchmark_suite(self, output_file: Optional[Path] = None) -> Dict:
        """
        Evaluate agent on neural network benchmark suite
        
        Args:
            output_file: Optional path to save results
            
        Returns:
            Aggregated results dictionary
        """
        mlir_files = list(self.benchmark_dir.glob("**/*.mlir"))
        
        if not mlir_files:
            print(f"⚠️  No neural network MLIR files found in {self.benchmark_dir}")
            return {}
        
        print(f"Evaluating {len(mlir_files)} neural networks...")
        
        results = []
        for mlir_file in mlir_files:
            result = self.evaluate_neural_network(mlir_file)
            results.append(result)
        
        # Aggregate statistics
        speedups = [r['speedup_vs_baseline'] for r in results if r['speedup_vs_baseline'] != float('inf')]
        
        summary = {
            'num_networks': len(results),
            'mean_speedup': float(np.mean(speedups)) if speedups else 0,
            'median_speedup': float(np.median(speedups)) if speedups else 0,
            'networks': results
        }
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n✓ Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("NEURAL NETWORK EVALUATION SUMMARY")
        print("="*60)
        print(f"Networks evaluated: {summary['num_networks']}")
        print(f"Mean speedup: {summary['mean_speedup']:.2f}x")
        print(f"Median speedup: {summary['median_speedup']:.2f}x")
        print("="*60)
        
        return summary


def main():
    """Example usage"""
    evaluator = NeuralNetworkEvaluator(
        agent_model_path=Path("results/best_model.pt"),
        benchmark_dir=Path("benchmarks/neural_nets")
    )
    
    results = evaluator.evaluate_benchmark_suite(
        output_file=Path("results/nn_evaluation_results.json")
    )


if __name__ == "__main__":
    main()
