#!/usr/bin/env python3
"""
Summarize generated MLIR files: count models, file sizes, operations
"""
import os
import re
import sys
from pathlib import Path

def count_operations(mlir_file):
    """Count different operation types in MLIR file"""
    ops = {
        'linalg.generic': 0,
        'linalg.matmul': 0,
        'linalg.batch_matmul': 0,
        'linalg.conv': 0,
        'arith': 0,
        'tensor': 0,
        'func.func': 0,
    }
    
    try:
        with open(mlir_file, 'r') as f:
            content = f.read()
            
        ops['linalg.generic'] = len(re.findall(r'linalg\.generic', content))
        ops['linalg.matmul'] = len(re.findall(r'linalg\.matmul', content))
        ops['linalg.batch_matmul'] = len(re.findall(r'linalg\.batch_matmul', content))
        ops['linalg.conv'] = len(re.findall(r'linalg\.conv', content))
        ops['arith'] = len(re.findall(r'arith\.', content))
        ops['tensor'] = len(re.findall(r'tensor\.', content))
        ops['func.func'] = len(re.findall(r'func\.func', content))
        
    except Exception as e:
        print(f"Error reading {mlir_file}: {e}")
        
    return ops

def get_file_size_str(size_bytes):
    """Convert bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def main():
    results_dir = Path("data/generated/code files")
    
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)
    
    # Find all *_linalg.mlir files
    mlir_files = sorted(results_dir.glob("*_linalg.mlir"))
    
    if not mlir_files:
        print(f"No MLIR files found in {results_dir}")
        sys.exit(0)
    
    print("=" * 100)
    print("MLIR Generation Summary")
    print("=" * 100)
    print()
    
    total_size = 0
    
    # Table header
    print(f"{'Model':<20} {'Size':<10} {'Linalg':<8} {'Matmul':<8} {'Conv':<8} {'Arith':<8} {'Tensor':<8} {'Funcs':<8}")
    print("-" * 100)
    
    for mlir_file in mlir_files:
        model_name = mlir_file.stem.replace('_linalg', '')
        file_size = mlir_file.stat().st_size
        total_size += file_size
        size_str = get_file_size_str(file_size)
        
        ops = count_operations(mlir_file)
        linalg_total = ops['linalg.generic'] + ops['linalg.matmul'] + ops['linalg.batch_matmul'] + ops['linalg.conv']
        
        print(f"{model_name:<20} {size_str:<10} {linalg_total:<8} "
              f"{ops['linalg.matmul'] + ops['linalg.batch_matmul']:<8} "
              f"{ops['linalg.conv']:<8} {ops['arith']:<8} {ops['tensor']:<8} {ops['func.func']:<8}")
    
    print("-" * 100)
    print(f"{'Total':<20} {get_file_size_str(total_size):<10}")
    print(f"Models: {len(mlir_files)}")
    print()
    
    # Check for any error logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        error_logs = list(logs_dir.glob("gen_*.err"))
        non_empty_errors = [log for log in error_logs if log.stat().st_size > 0]
        
        if non_empty_errors:
            print(f"\n⚠️  Warning: {len(non_empty_errors)} jobs had errors. Check logs/gen_*.err")
        else:
            print("\n✅ All jobs completed successfully (no error logs)")

if __name__ == "__main__":
    main()
