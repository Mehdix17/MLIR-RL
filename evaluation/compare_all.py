"""
Compare RL-optimized MLIR vs PyTorch Default vs PyTorch JIT
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np


def load_results(results_dir: Path) -> dict:
    """Load all result files"""
    results = {}
    
    # Load RL results
    rl_file = results_dir / "rl_optimized_results.json"
    if rl_file.exists():
        with open(rl_file) as f:
            results['RL-Optimized'] = json.load(f)
    
    # Load PyTorch Default results
    pytorch_file = results_dir / "pytorch_default_results.json"
    if pytorch_file.exists():
        with open(pytorch_file) as f:
            results['PyTorch-Default'] = json.load(f)
    
    # Load PyTorch JIT results
    jit_file = results_dir / "pytorch_jit_results.json"
    if jit_file.exists():
        with open(jit_file) as f:
            results['PyTorch-JIT'] = json.load(f)
    
    return results


def create_comparison_data(results: dict) -> pd.DataFrame:
    """Create comparison DataFrame"""
    rows = []
    
    # Get all benchmarks
    all_benchmarks = set()
    for method_results in results.values():
        all_benchmarks.update(method_results.keys())
    
    for benchmark in sorted(all_benchmarks):
        row = {'Benchmark': benchmark}
        
        for method, method_results in results.items():
            if benchmark in method_results:
                data = method_results[benchmark]
                if 'error' not in data:
                    row[f'{method}_mean'] = data['mean_time_ms']
                    row[f'{method}_std'] = data['std_time_ms']
                else:
                    row[f'{method}_mean'] = None
                    row[f'{method}_std'] = None
            else:
                row[f'{method}_mean'] = None
                row[f'{method}_std'] = None
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots"""
    methods = ['RL-Optimized', 'PyTorch-Default', 'PyTorch-JIT']
    colors = {'RL-Optimized': '#2ecc71', 'PyTorch-Default': '#e74c3c', 'PyTorch-JIT': '#3498db'}
    
    # Check which methods have data
    available_methods = []
    for method in methods:
        mean_col = f'{method}_mean'
        if mean_col in df.columns and df[mean_col].notna().any():
            available_methods.append(method)
    
    if not available_methods:
        print("⚠️  No data available for plotting")
        return
    
    benchmarks = df['Benchmark'].tolist()
    x = np.arange(len(benchmarks))
    width = 0.8 / len(available_methods)  # Adjust width based on number of methods
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, method in enumerate(available_methods):
        mean_col = f'{method}_mean'
        std_col = f'{method}_std'
        
        means = df[mean_col].fillna(0).values
        stds = df[std_col].fillna(0).values
        
        offset = (i - len(available_methods)/2 + 0.5) * width
        
        ax.bar(
            x + offset,
            means,
            width,
            label=method,
            color=colors[method],
            yerr=stds,
            capsize=3,
            alpha=0.8
        )
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('3-Way Comparison: RL-Optimized vs PyTorch Default vs PyTorch JIT',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_bar_plot.png", dpi=300, bbox_inches='tight')
    print(f"✓ Bar plot saved to {output_dir / 'comparison_bar_plot.png'}")
    plt.close()
    
    # Create speedup plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate speedups relative to PyTorch Default
    if 'PyTorch-Default_mean' not in df.columns:
        print("⚠️  No PyTorch Default data for speedup calculation")
        plt.close()
        return
    
    pytorch_default_means = df['PyTorch-Default_mean'].fillna(1).values
    
    # Check which methods to compare
    speedup_methods = [m for m in ['RL-Optimized', 'PyTorch-JIT'] 
                       if f'{m}_mean' in df.columns and df[f'{m}_mean'].notna().any()]
    
    if not speedup_methods:
        print("⚠️  No methods available for speedup comparison")
        plt.close()
        return
    
    width_speedup = 0.8 / len(speedup_methods)
    
    for i, method in enumerate(speedup_methods):
        mean_col = f'{method}_mean'
        means = df[mean_col].fillna(1).values
        speedups = pytorch_default_means / means
        
        offset = (i - len(speedup_methods)/2 + 0.5) * width_speedup
        
        ax.bar(
            x + offset,
            speedups,
            width_speedup,
            label=f'{method} vs Default',
            color=colors[method],
            alpha=0.8
        )
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (PyTorch Default)')
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Comparison (Higher is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Speedup plot saved to {output_dir / 'speedup_comparison.png'}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate summary table with speedups"""
    # Create display DataFrame
    display_rows = []
    
    for _, row in df.iterrows():
        display_row = {'Benchmark': row['Benchmark']}
        
        # Add times with std dev
        for method in ['RL-Optimized', 'PyTorch-Default', 'PyTorch-JIT']:
            mean = row[f'{method}_mean']
            std = row[f'{method}_std']
            if pd.notna(mean):
                display_row[f'{method} (ms)'] = f"{mean:.3f} ± {std:.3f}"
            else:
                display_row[f'{method} (ms)'] = 'N/A'
        
        # Calculate speedups
        pytorch_default = row['PyTorch-Default_mean']
        
        if pd.notna(pytorch_default) and pytorch_default > 0:
            # RL vs PyTorch Default
            rl_mean = row['RL-Optimized_mean']
            if pd.notna(rl_mean) and rl_mean > 0:
                speedup = pytorch_default / rl_mean
                display_row['RL Speedup'] = f"{speedup:.2f}×"
            else:
                display_row['RL Speedup'] = 'N/A'
            
            # JIT vs PyTorch Default
            jit_mean = row['PyTorch-JIT_mean']
            if pd.notna(jit_mean) and jit_mean > 0:
                speedup = pytorch_default / jit_mean
                display_row['JIT Speedup'] = f"{speedup:.2f}×"
            else:
                display_row['JIT Speedup'] = 'N/A'
        else:
            display_row['RL Speedup'] = 'N/A'
            display_row['JIT Speedup'] = 'N/A'
        
        display_rows.append(display_row)
    
    display_df = pd.DataFrame(display_rows)
    
    # Save to CSV
    csv_file = output_dir / "comparison_summary.csv"
    display_df.to_csv(csv_file, index=False)
    print(f"✓ Summary table saved to {csv_file}")
    
    # Print to console
    print("\n" + "="*80)
    print("COMPARISON SUMMARY TABLE")
    print("="*80)
    print(display_df.to_string(index=False))
    print("="*80)
    
    # Calculate and print statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    for method in ['RL-Optimized', 'PyTorch-JIT']:
        valid_speedups = []
        for _, row in df.iterrows():
            pytorch_default = row['PyTorch-Default_mean']
            method_mean = row[f'{method}_mean']
            if pd.notna(pytorch_default) and pd.notna(method_mean) and \
               pytorch_default > 0 and method_mean > 0:
                speedup = pytorch_default / method_mean
                valid_speedups.append(speedup)
        
        if valid_speedups:
            print(f"\n{method} vs PyTorch Default:")
            print(f"  Average Speedup: {np.mean(valid_speedups):.2f}×")
            print(f"  Median Speedup:  {np.median(valid_speedups):.2f}×")
            print(f"  Min Speedup:     {np.min(valid_speedups):.2f}×")
            print(f"  Max Speedup:     {np.max(valid_speedups):.2f}×")
    
    print("="*80)


def run_comparison(results_dir: Path, output_dir: Path):
    """Run complete comparison"""
    print("="*80)
    print("3-WAY COMPARISON: RL-Optimized vs PyTorch Default vs PyTorch JIT")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("\nLoading results...")
    results = load_results(results_dir)
    
    if not results:
        print("Error: No results found. Run benchmark executors first.")
        sys.exit(1)
    
    print(f"  ✓ Loaded results for {len(results)} methods")
    
    # Create comparison data
    print("\nCreating comparison data...")
    df = create_comparison_data(results)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(df, output_dir)
    
    # Generate summary table
    print("\nGenerating summary table...")
    generate_summary_table(df, output_dir)
    
    # Save combined results
    combined_file = output_dir / "comparison_results.json"
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Combined results saved to {combined_file}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print(f"Results directory: {output_dir}")
    print("="*80)


def main():
    """Main execution function"""
    results_dir = Path("evaluation/results")
    output_dir = Path("results/comparison_rl_vs_pytorch")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run the benchmark executors first:")
        print("  1. python benchmarks/benchmark_suite.py")
        print("  2. python evaluation/run_rl_optimized.py")
        print("  3. python evaluation/run_pytorch_default.py")
        print("  4. python evaluation/run_pytorch_jit.py")
        sys.exit(1)
    
    run_comparison(results_dir, output_dir)


if __name__ == "__main__":
    main()
