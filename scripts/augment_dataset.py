"""
Augment existing dataset with additional random MLIR programs

This script:
1. Analyzes existing data/all/code_files/ structure
2. Generates additional MLIR files matching the same format
3. Creates execution_times JSON matching existing structure
"""

import random
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple


class DatasetAugmenter:
    """Augment existing MLIR dataset"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
    
    def analyze_existing_data(self, data_dir: Path) -> dict:
        """Analyze existing dataset to match its format"""
        print(f"📊 Analyzing existing data in {data_dir}...")
        
        code_files_dir = data_dir / "code_files"
        if not code_files_dir.exists():
            print(f"  ⚠️  {code_files_dir} not found")
            return {}
        
        # Count operation types
        op_types = {}
        dimensions_used = set()
        
        for mlir_file in code_files_dir.glob("*.mlir"):
            # Parse filename: add_112_112_120_150.mlir
            parts = mlir_file.stem.split('_')
            if len(parts) >= 1:
                op_type = parts[0]
                op_types[op_type] = op_types.get(op_type, 0) + 1
                
                # Extract dimensions
                if len(parts) > 1:
                    try:
                        dims = [int(p) for p in parts[1:] if p.isdigit()]
                        dimensions_used.update(dims)
                    except ValueError:
                        pass
        
        analysis = {
            'total_files': sum(op_types.values()),
            'operation_types': op_types,
            'dimensions': sorted(list(dimensions_used))
        }
        
        print(f"  ✓ Found {analysis['total_files']} files")
        print(f"  ✓ Operation types: {dict(op_types)}")
        print(f"  ✓ Dimensions used: {analysis['dimensions'][:10]}..." if len(analysis['dimensions']) > 10 else f"  ✓ Dimensions used: {analysis['dimensions']}")
        
        return analysis
    
    def generate_add_operation(self, dims_pool: List[int]) -> Tuple[str, List[int]]:
        """Generate ADD operation matching existing format"""
        # Select 4 random dimensions from the pool
        dims = [random.choice(dims_pool) for _ in range(4)]
        
        mlir_code = f"""func.func @add(%arg0: tensor<{dims[0]}x{dims[1]}xf32>, %arg1: tensor<{dims[2]}x{dims[3]}xf32>) -> tensor<{dims[0]}x{dims[1]}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.add ins(%arg0, %arg1 : tensor<{dims[0]}x{dims[1]}xf32>, tensor<{dims[2]}x{dims[3]}xf32>) outs(%arg0 : tensor<{dims[0]}x{dims[1]}xf32>) -> tensor<{dims[0]}x{dims[1]}xf32>
  return %0 : tensor<{dims[0]}x{dims[1]}xf32>
}}
"""
        return mlir_code, dims
    
    def generate_matmul_operation(self, dims_pool: List[int]) -> Tuple[str, List[int]]:
        """Generate MATMUL operation"""
        M = random.choice(dims_pool)
        N = random.choice(dims_pool)
        K = random.choice(dims_pool)
        dims = [M, K, N]
        
        mlir_code = f"""func.func @matmul(%arg0: tensor<{M}x{K}xf32>, %arg1: tensor<{K}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<{M}x{N}xf32>
  %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%1 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  return %2 : tensor<{M}x{N}xf32>
}}
"""
        return mlir_code, dims
    
    def generate_conv2d_operation(self, dims_pool: List[int]) -> Tuple[str, List[int]]:
        """Generate CONV2D operation"""
        batch = random.choice([1, 8, 16, 32])
        in_ch = random.choice(dims_pool[:len(dims_pool)//2])
        out_ch = random.choice(dims_pool[:len(dims_pool)//2])
        height = random.choice(dims_pool[len(dims_pool)//2:])
        width = height  # Keep square for simplicity
        kernel = random.choice([3, 5, 7])
        dims = [batch, in_ch, out_ch, height, kernel]
        
        mlir_code = f"""func.func @conv2d(%arg0: tensor<{batch}x{in_ch}x{height}x{width}xf32>, %arg1: tensor<{out_ch}x{in_ch}x{kernel}x{kernel}xf32>) -> tensor<{batch}x{out_ch}x{height}x{width}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<{batch}x{out_ch}x{height}x{width}xf32>
  %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<{batch}x{out_ch}x{height}x{width}xf32>) -> tensor<{batch}x{out_ch}x{height}x{width}xf32>
  %2 = linalg.conv_2d_nchw_fchw {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}} ins(%arg0, %arg1 : tensor<{batch}x{in_ch}x{height}x{width}xf32>, tensor<{out_ch}x{in_ch}x{kernel}x{kernel}xf32>) outs(%1 : tensor<{batch}x{out_ch}x{height}x{width}xf32>) -> tensor<{batch}x{out_ch}x{height}x{width}xf32>
  return %2 : tensor<{batch}x{out_ch}x{height}x{width}xf32>
}}
"""
        return mlir_code, dims
    
    def augment_dataset(
        self,
        source_dir: Path,
        output_dir: Path,
        num_samples: int = 500,
        operation_types: List[str] = None
    ) -> dict:
        """
        Augment dataset with additional samples
        
        Args:
            source_dir: Source directory to analyze (e.g., data/all)
            output_dir: Output directory for new files (e.g., data/generated/code_files)
            num_samples: Number of new samples to generate
            operation_types: Which operations to generate
        """
        print(f"\n🎯 Augmenting dataset...")
        print(f"  Source: {source_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Samples: {num_samples}\n")
        
        # Analyze existing data
        analysis = self.analyze_existing_data(source_dir)
        
        if not analysis:
            print("  ⚠️  Using default configuration")
            dims_pool = [7, 14, 15, 28, 56, 112, 120, 130, 150, 224, 228, 240]
            operation_types = operation_types or ['add', 'matmul', 'conv2d']
        else:
            dims_pool = analysis['dimensions'] or [7, 14, 15, 28, 56, 112, 120, 130, 150, 224, 228, 240]
            if not operation_types:
                # Use same operation types as existing data
                operation_types = list(analysis['operation_types'].keys())
                if not operation_types:
                    operation_types = ['add']
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        generated_files = []
        execution_times = {}
        
        op_generators = {
            'add': self.generate_add_operation,
            'matmul': self.generate_matmul_operation,
            'conv2d': self.generate_conv2d_operation
        }
        
        print(f"📝 Generating {num_samples} new MLIR files...")
        
        for i in range(num_samples):
            # Select operation type
            op_type = random.choice(operation_types)
            
            # Generate code
            if op_type in op_generators:
                mlir_code, dims = op_generators[op_type](dims_pool)
            else:
                # Default to add if unknown type
                mlir_code, dims = self.generate_add_operation(dims_pool)
                op_type = 'add'
            
            # Create filename matching existing format
            dim_str = '_'.join(map(str, dims))
            filename = f"{op_type}_{dim_str}.mlir"
            filepath = output_dir / filename
            
            # Write file
            with open(filepath, 'w') as f:
                f.write(mlir_code)
            
            generated_files.append(filepath)
            
            # Generate mock execution times (will be replaced by actual measurements)
            execution_times[filename] = random.uniform(0.1, 2.0)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} files...")
        
        print(f"\n✅ Generated {len(generated_files)} new MLIR files")
        
        # Save execution times JSON
        times_file = output_dir.parent / "execution_times_generated.json"
        with open(times_file, 'w') as f:
            json.dump(execution_times, f, indent=2)
        
        print(f"✅ Saved execution times to {times_file}")
        
        return {
            'generated_files': len(generated_files),
            'output_dir': str(output_dir),
            'execution_times_file': str(times_file)
        }


def main():
    """Main augmentation workflow"""
    print("="*60)
    print("Dataset Augmentation Tool")
    print("="*60)
    
    augmenter = DatasetAugmenter(seed=42)
    
    # Augment training data
    result = augmenter.augment_dataset(
        source_dir=Path("data/all"),
        output_dir=Path("data/generated/code_files"),
        num_samples=500,
        operation_types=['add', 'matmul', 'conv2d']
    )
    
    print("\n" + "="*60)
    print("✅ Augmentation Complete!")
    print("="*60)
    print(f"\nGenerated: {result['generated_files']} files")
    print(f"Location: {result['output_dir']}")
    print(f"Times: {result['execution_times_file']}")
    
    print("\n📝 Next steps:")
    print("  1. Review generated files in data/generated/code_files/")
    print("  2. Update config.json to use augmented data:")
    print('     "use_augmentation": true')
    print('     "augmentation_ratio": 0.3')
    print("  3. Train with augmented dataset:")
    print("     CONFIG_FILE_PATH=config/config_augmented.json python bin/train.py")


if __name__ == "__main__":
    main()
