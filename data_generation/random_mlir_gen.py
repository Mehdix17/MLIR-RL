"""
Generate random MLIR code for training the RL agent

Supports:
- Matrix operations (matmul, transpose)
- Convolutions (conv2d, conv3d)
- Pooling (max, avg)
- Element-wise ops (add, mul, relu)
"""

import random
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


class RandomMLIRGenerator:
    """Generate random MLIR code for various operations"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Operation templates
        self.operations = {
            'matmul': self._generate_matmul,
            'conv2d': self._generate_conv2d,
            'pooling': self._generate_pooling,
            'elementwise': self._generate_elementwise,
        }
    
    def generate_dataset(
        self,
        num_samples: int,
        output_dir: Path,
        operation_types: List[str] = None
    ) -> List[Path]:
        """
        Generate a dataset of random MLIR programs
        
        Args:
            num_samples: Number of programs to generate
            output_dir: Where to save the MLIR files
            operation_types: Which operations to include (None = all)
            
        Returns:
            List of paths to generated files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if operation_types is None:
            operation_types = list(self.operations.keys())
        
        generated_files = []
        
        for i in range(num_samples):
            # Randomly select operation type
            op_type = random.choice(operation_types)
            
            # Generate MLIR code
            mlir_code = self.operations[op_type]()
            
            # Save to file
            filename = output_dir / f"{op_type}_{i:04d}.mlir"
            with open(filename, 'w') as f:
                f.write(mlir_code)
            
            generated_files.append(filename)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} files...")
        
        print(f"✓ Generated {num_samples} MLIR files in {output_dir}")
        return generated_files
    
    def _generate_matmul(self) -> str:
        """Generate random matrix multiplication"""
        # Random dimensions
        M = random.choice([64, 128, 256, 512, 1024])
        N = random.choice([64, 128, 256, 512, 1024])
        K = random.choice([64, 128, 256, 512, 1024])
        
        mlir_template = f"""
func.func @matmul(%A: tensor<{M}x{K}xf32>, %B: tensor<{K}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<{M}x{N}xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  
  %result = linalg.matmul
    ins(%A, %B : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>)
    outs(%fill : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  
  return %result : tensor<{M}x{N}xf32>
}}
"""
        return mlir_template.strip()
    
    def _generate_conv2d(self) -> str:
        """Generate random 2D convolution"""
        # Random dimensions
        batch = random.choice([1, 8, 16, 32])
        channels_in = random.choice([3, 16, 32, 64, 128])
        channels_out = random.choice([16, 32, 64, 128, 256])
        height = random.choice([28, 32, 64, 128, 224])
        width = random.choice([28, 32, 64, 128, 224])
        kernel = random.choice([3, 5, 7])
        
        mlir_template = f"""
func.func @conv2d(
  %input: tensor<{batch}x{channels_in}x{height}x{width}xf32>,
  %filter: tensor<{channels_out}x{channels_in}x{kernel}x{kernel}xf32>
) -> tensor<{batch}x{channels_out}x{height}x{width}xf32> {{
  
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<{batch}x{channels_out}x{height}x{width}xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<{batch}x{channels_out}x{height}x{width}xf32>) -> tensor<{batch}x{channels_out}x{height}x{width}xf32>
  
  %result = linalg.conv_2d_nchw_fchw
    {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}
    ins(%input, %filter : tensor<{batch}x{channels_in}x{height}x{width}xf32>, tensor<{channels_out}x{channels_in}x{kernel}x{kernel}xf32>)
    outs(%fill : tensor<{batch}x{channels_out}x{height}x{width}xf32>) -> tensor<{batch}x{channels_out}x{height}x{width}xf32>
  
  return %result : tensor<{batch}x{channels_out}x{height}x{width}xf32>
}}
"""
        return mlir_template.strip()
    
    def _generate_pooling(self) -> str:
        """Generate random pooling operation"""
        # Random dimensions
        batch = random.choice([1, 8, 16, 32])
        channels = random.choice([16, 32, 64, 128])
        height = random.choice([28, 32, 64, 128])
        width = random.choice([28, 32, 64, 128])
        pool_size = random.choice([2, 3, 4])
        
        out_height = height // pool_size
        out_width = width // pool_size
        
        mlir_template = f"""
func.func @maxpool(
  %input: tensor<{batch}x{channels}x{height}x{width}xf32>
) -> tensor<{batch}x{channels}x{out_height}x{out_width}xf32> {{
  
  %init = tensor.empty() : tensor<{batch}x{channels}x{out_height}x{out_width}xf32>
  %kernel = tensor.empty() : tensor<{pool_size}x{pool_size}xf32>
  
  %result = linalg.pooling_nchw_max
    {{dilations = dense<1> : tensor<2xi64>, strides = dense<{pool_size}> : tensor<2xi64>}}
    ins(%input, %kernel : tensor<{batch}x{channels}x{height}x{width}xf32>, tensor<{pool_size}x{pool_size}xf32>)
    outs(%init : tensor<{batch}x{channels}x{out_height}x{out_width}xf32>) -> tensor<{batch}x{channels}x{out_height}x{out_width}xf32>
  
  return %result : tensor<{batch}x{channels}x{out_height}x{out_width}xf32>
}}
"""
        return mlir_template.strip()
    
    def _generate_elementwise(self) -> str:
        """Generate random element-wise operation"""
        # Random dimensions
        dims = [random.choice([64, 128, 256, 512, 1024]) for _ in range(random.randint(2, 4))]
        shape = 'x'.join(map(str, dims))
        
        op = random.choice(['add', 'mul', 'max'])
        
        mlir_template = f"""
func.func @elementwise_{op}(
  %lhs: tensor<{shape}xf32>,
  %rhs: tensor<{shape}xf32>
) -> tensor<{shape}xf32> {{
  
  %result = linalg.{op}
    ins(%lhs, %rhs : tensor<{shape}xf32>, tensor<{shape}xf32>)
    outs(%lhs : tensor<{shape}xf32>) -> tensor<{shape}xf32>
  
  return %result : tensor<{shape}xf32>
}}
"""
        return mlir_template.strip()


def main():
    """Example usage"""
    print("="*60)
    print("Random MLIR Generator")
    print("="*60)
    print("\nNote: For augmenting existing data/all/ with matching format,")
    print("      use: python augment_dataset.py")
    print("\nThis script generates random operations for testing/benchmarking.\n")
    
    generator = RandomMLIRGenerator(seed=42)
    
    # Generate training dataset
    train_files = generator.generate_dataset(
        num_samples=100,
        output_dir=Path("data/generated/train"),
        operation_types=['matmul', 'conv2d', 'pooling']
    )
    
    # Generate test dataset
    test_files = generator.generate_dataset(
        num_samples=20,
        output_dir=Path("data/generated/test"),
        operation_types=['matmul', 'conv2d', 'pooling']
    )
    
    print(f"\n✓ Generated {len(train_files)} training files")
    print(f"✓ Generated {len(test_files)} test files")
    print(f"\n📁 Files saved to:")
    print(f"   - data/generated/train/")
    print(f"   - data/generated/test/")


if __name__ == "__main__":
    main()
