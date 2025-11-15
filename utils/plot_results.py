#!/usr/bin/env python3
"""
Enhanced plotting script for MLIR-RL results
Generates comparison plots by operation type (add, matmul, conv2d, pooling, relu)
Compares baseline vs RL model performance
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_run_data(run_dir):
    """Load all metrics from a run directory"""
    logs_dir = Path(run_dir) / 'logs'
    
    data = {
        'train': {},
        'eval': {},
        'config': None,
        'tags': []
    }
    
    # Load tags
    tags_file = Path(run_dir) / 'tags'
    if tags_file.exists():
        with open(tags_file) as f:
            data['tags'] = f.read().splitlines()
    
    # Load evaluation metrics
    eval_dir = logs_dir / 'eval'
    if eval_dir.exists():
        # Per-benchmark speedups
        speedup_dir = eval_dir / 'speedup'
        if speedup_dir.exists():
            data['eval']['per_benchmark_speedup'] = {}
            for f in speedup_dir.glob('*'):
                bench_name = f.name
                with open(f) as file:
                    speedups = [float(line.strip()) for line in file if line.strip()]
                    if speedups:
                        data['eval']['per_benchmark_speedup'][bench_name] = speedups[-1]
        
        # Per-benchmark execution times
        exec_time_dir = eval_dir / 'exec_time'
        if exec_time_dir.exists():
            data['eval']['per_benchmark_exec_time'] = {}
            for f in exec_time_dir.glob('*'):
                bench_name = f.name
                with open(f) as file:
                    times = [float(line.strip()) for line in file if line.strip()]
                    if times:
                        data['eval']['per_benchmark_exec_time'][bench_name] = times[-1]
        
        # Average speedup
        avg_speedup_file = eval_dir / 'average_speedup'
        if avg_speedup_file.exists():
            with open(avg_speedup_file) as f:
                data['eval']['average_speedup'] = float(f.read().strip())
        
        # Load other eval metrics
        for metric_file in ['reward', 'cumulative_reward', 'final_speedup', 'entropy']:
            path = eval_dir / metric_file
            if path.exists():
                with open(path) as f:
                    values = [float(line.strip()) for line in f if line.strip()]
                    data['eval'][metric_file] = values
    
    # Load training metrics
    train_dir = logs_dir / 'train'
    if train_dir.exists():
        for metric_file in ['reward', 'entropy', 'final_speedup']:
            path = train_dir / metric_file
            if path.exists():
                with open(path) as f:
                    values = [float(line.strip()) for line in f if line.strip()]
                    data['train'][metric_file] = values
    
    # Load PPO training metrics
    ppo_dir = logs_dir / 'train_ppo'
    if ppo_dir.exists():
        data['train_ppo'] = {}
        for metric_file in ppo_dir.glob('*'):
            with open(metric_file) as f:
                values = [float(line.strip()) for line in f if line.strip()]
                data['train_ppo'][metric_file.name] = values
    
    return data


def get_operation_type(bench_name):
    """Extract operation type from benchmark name"""
    parts = bench_name.split('_')
    if len(parts) >= 2 and parts[0] == 'conv' and parts[1] == '2d':
        return 'conv2d'
    return parts[0]


def group_by_operation_type(per_benchmark_data, baseline_exec_times=None):
    """Group benchmarks by operation type and compute statistics"""
    groups = defaultdict(lambda: {'benchmarks': [], 'speedups': [], 'exec_times': [], 'baseline_times': []})
    
    for bench_name, speedup in per_benchmark_data.items():
        op_type = get_operation_type(bench_name)
        groups[op_type]['benchmarks'].append(bench_name)
        groups[op_type]['speedups'].append(speedup)
        
        if baseline_exec_times and bench_name in baseline_exec_times:
            groups[op_type]['baseline_times'].append(baseline_exec_times[bench_name])
    
    # Compute statistics
    for op_type in groups:
        groups[op_type]['mean_speedup'] = np.mean(groups[op_type]['speedups'])
        groups[op_type]['std_speedup'] = np.std(groups[op_type]['speedups'])
        groups[op_type]['geometric_mean_speedup'] = np.exp(np.mean(np.log(groups[op_type]['speedups'])))
    
    return dict(groups)


def plot_speedup_by_operation_type(groups, output_path, title="Speedup by Operation Type"):
    """Create bar chart of mean speedup by operation type"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    op_types = sorted(groups.keys())
    means = [groups[op]['mean_speedup'] for op in op_types]
    stds = [groups[op]['std_speedup'] for op in op_types]
    
    x = np.arange(len(op_types))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Operation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(op_types, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Baseline')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_per_benchmark_speedup(groups, output_path, title="Speedup per Benchmark"):
    """Create horizontal bar chart showing speedup for each benchmark, grouped by type"""
    fig, ax = plt.subplots(figsize=(14, max(8, len(sum([g['benchmarks'] for g in groups.values()], [])) * 0.3)))
    
    y_pos = 0
    y_ticks = []
    y_labels = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    
    for (op_type, data), color in zip(sorted(groups.items()), colors):
        benchmarks = data['benchmarks']
        speedups = data['speedups']
        
        # Sort by speedup
        sorted_indices = np.argsort(speedups)
        benchmarks = [benchmarks[i] for i in sorted_indices]
        speedups = [speedups[i] for i in sorted_indices]
        
        for bench, speedup in zip(benchmarks, speedups):
            bars = ax.barh(y_pos, speedup, color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            ax.text(speedup + 0.1, y_pos, f'{speedup:.2f}x', va='center', fontsize=8)
            y_ticks.append(y_pos)
            y_labels.append(bench)
            y_pos += 1
        
        y_pos += 0.5  # Gap between operation types
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Speedup', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline (1x)')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_training_metrics(data, output_dir):
    """Plot training metrics (value loss, policy loss, entropy, cumulative reward)"""
    train_ppo = data.get('train_ppo', {})
    train = data.get('train', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Value loss
    if 'value_loss' in train_ppo:
        axes[0, 0].plot(train_ppo['value_loss'], color='orange', linewidth=1.5)
        axes[0, 0].set_title('Value Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(alpha=0.3)
    
    # Policy loss
    if 'policy_loss' in train_ppo:
        axes[0, 1].plot(train_ppo['policy_loss'], color='magenta', linewidth=1.5)
        axes[0, 1].set_title('Policy Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(alpha=0.3)
    
    # Entropy
    if 'entropy' in train:
        axes[1, 0].plot(train['entropy'], color='green', linewidth=1.5)
        axes[1, 0].set_title('Entropy', fontweight='bold')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(alpha=0.3)
    
    # Cumulative reward (from eval)
    if 'cumulative_reward' in data.get('eval', {}):
        axes[1, 1].plot(data['eval']['cumulative_reward'], color='purple', linewidth=1.5)
        axes[1, 1].set_title('Cumulative Reward (Eval)', fontweight='bold')
        axes[1, 1].set_xlabel('Evaluation')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_geometric_mean_comparison(groups, output_path, title="Geometric Mean Speedup by Operation Type"):
    """Plot geometric mean speedup comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    op_types = sorted(groups.keys())
    geom_means = [groups[op]['geometric_mean_speedup'] for op in op_types]
    
    x = np.arange(len(op_types))
    bars = ax.bar(x, geom_means, alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=2)
    
    for bar, gmean in zip(bars, geom_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{gmean:.2f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Operation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Geometric Mean Speedup', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(op_types, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <run_directory>")
        print("Example: python plot_results.py results/run_9")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        sys.exit(1)
    
    print(f"Loading data from {run_dir}...")
    data = load_run_data(run_dir)
    
    # Create plots directory
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    print(f"Saving plots to {plots_dir}/")
    
    # Load baseline execution times (from the JSON files)
    with open('config/config.json') as f:
        config = json.load(f)
    
    baseline_times = {}
    if 'json_file' in config:
        with open(config['json_file']) as f:
            baseline_times.update(json.load(f))
    if 'eval_json_file' in config:
        with open(config['eval_json_file']) as f:
            baseline_times.update(json.load(f))
    
    # Group benchmarks by operation type
    if 'per_benchmark_speedup' in data['eval']:
        print("\nGenerating plots...")
        groups = group_by_operation_type(data['eval']['per_benchmark_speedup'], baseline_times)
        
        # Print statistics
        print("\n" + "="*70)
        print("SPEEDUP STATISTICS BY OPERATION TYPE")
        print("="*70)
        for op_type in sorted(groups.keys()):
            g = groups[op_type]
            print(f"\n{op_type.upper()}:")
            print(f"  Benchmarks: {len(g['benchmarks'])}")
            print(f"  Mean speedup: {g['mean_speedup']:.3f}x ± {g['std_speedup']:.3f}")
            print(f"  Geometric mean speedup: {g['geometric_mean_speedup']:.3f}x")
            print(f"  Min speedup: {min(g['speedups']):.3f}x")
            print(f"  Max speedup: {max(g['speedups']):.3f}x")
        
        # Generate plots
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        plot_speedup_by_operation_type(groups, plots_dir / 'speedup_by_op_type.png')
        plot_geometric_mean_comparison(groups, plots_dir / 'geometric_mean_speedup.png')
        plot_per_benchmark_speedup(groups, plots_dir / 'per_benchmark_speedup.png')
    
    # Training metrics
    if data['train'] or data.get('train_ppo'):
        plot_training_metrics(data, plots_dir)
    
    print("\n" + "="*70)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nView plots in: {plots_dir}/")


if __name__ == '__main__':
    main()
