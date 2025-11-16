"""
Benchmark suite for comparing RL-optimized MLIR vs PyTorch

This creates standardized benchmarks that can be:
1. Optimized by RL agent (MLIR)
2. Run with PyTorch default
3. Run with PyTorch JIT
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Dict, List


class BenchmarkModel(nn.Module):
    """Base class for benchmark models"""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def get_example_input(self):
        """Return example input tensor for this model"""
        raise NotImplementedError


class MatMulBenchmark(BenchmarkModel):
    """Matrix multiplication benchmark"""
    def __init__(self, M=1024, N=1024, K=1024):
        super().__init__(f"matmul_{M}x{N}x{K}")
        self.M, self.N, self.K = M, N, K
        self.weight = nn.Parameter(torch.randn(K, N))
    
    def forward(self, x):
        return torch.matmul(x, self.weight)
    
    def get_example_input(self):
        return torch.randn(self.M, self.K)


class Conv2DBenchmark(BenchmarkModel):
    """2D Convolution benchmark"""
    def __init__(self, batch=32, in_ch=128, out_ch=256, size=56):
        super().__init__(f"conv2d_{batch}x{in_ch}x{size}x{size}")
        self.batch = batch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.size = size
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)
    
    def get_example_input(self):
        return torch.randn(self.batch, self.in_ch, self.size, self.size)


class SimpleResNetBlock(BenchmarkModel):
    """Simple ResNet block for benchmarking"""
    def __init__(self, channels=64):
        super().__init__(f"resnet_block_{channels}")
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
    
    def get_example_input(self):
        return torch.randn(32, self.channels, 56, 56)


class LinearBenchmark(BenchmarkModel):
    """Linear layer benchmark"""
    def __init__(self, batch=256, in_features=1024, out_features=1024):
        super().__init__(f"linear_{batch}x{in_features}x{out_features}")
        self.batch = batch
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
    
    def get_example_input(self):
        return torch.randn(self.batch, self.in_features)


class BenchmarkSuite:
    """Collection of benchmarks for comparison"""
    
    def __init__(self):
        self.benchmarks = {
            'matmul_small': MatMulBenchmark(256, 256, 256),
            'matmul_medium': MatMulBenchmark(512, 512, 512),
            'matmul_large': MatMulBenchmark(1024, 1024, 1024),
            'conv2d_small': Conv2DBenchmark(8, 64, 128, 28),
            'conv2d_medium': Conv2DBenchmark(16, 128, 256, 56),
            'conv2d_large': Conv2DBenchmark(32, 256, 512, 112),
            'resnet_block': SimpleResNetBlock(64),
            'linear_small': LinearBenchmark(64, 512, 512),
            'linear_large': LinearBenchmark(256, 2048, 2048),
        }
    
    def save_pytorch_models(self, output_dir: Path):
        """Save PyTorch models for baseline comparison"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_info = {}
        
        for name, model in self.benchmarks.items():
            print(f"Processing {name}...")
            
            # Save model architecture info
            model_info[name] = {
                'class': model.__class__.__name__,
                'name': model.name,
            }
            
            # Add specific parameters based on model type
            if isinstance(model, MatMulBenchmark):
                model_info[name].update({
                    'M': model.M, 'N': model.N, 'K': model.K
                })
            elif isinstance(model, Conv2DBenchmark):
                model_info[name].update({
                    'batch': model.batch,
                    'in_ch': model.in_ch,
                    'out_ch': model.out_ch,
                    'size': model.size
                })
            elif isinstance(model, SimpleResNetBlock):
                model_info[name].update({
                    'channels': model.channels
                })
            elif isinstance(model, LinearBenchmark):
                model_info[name].update({
                    'batch': model.batch,
                    'in_features': model.in_features,
                    'out_features': model.out_features
                })
            
            # Save standard PyTorch model
            torch_file = output_dir / f"{name}_pytorch.pt"
            torch.save({
                'model_state': model.state_dict(),
                'model_class': model.__class__.__name__,
                'model_info': model_info[name]
            }, torch_file)
            
            # Save JIT compiled version
            jit_file = output_dir / f"{name}_jit.pt"
            model.eval()
            example_input = model.get_example_input()
            try:
                jit_model = torch.jit.trace(model, example_input)
                torch.jit.save(jit_model, jit_file)
                print(f"  ✓ Saved PyTorch models for {name}")
            except Exception as e:
                print(f"  ⚠️  Could not JIT compile {name}: {e}")
        
        # Save model info
        info_file = output_dir / "model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return model_info
    
    def export_to_mlir_format(self, output_dir: Path):
        """
        Export benchmark models to MLIR-compatible format
        Note: This creates placeholder MLIR files. 
        Actual MLIR generation would require torch-mlir or similar tools.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mlir_files = []
        
        for name, model in self.benchmarks.items():
            mlir_file = output_dir / f"{name}.mlir"
            
            # Create a simple MLIR template
            # In production, you'd use torch-mlir to generate proper MLIR
            mlir_content = self._generate_mlir_template(model)
            
            with open(mlir_file, 'w') as f:
                f.write(mlir_content)
            
            mlir_files.append(str(mlir_file))
            print(f"  ✓ Exported {name} → {mlir_file.name}")
        
        return mlir_files
    
    def _generate_mlir_template(self, model: BenchmarkModel) -> str:
        """Generate MLIR template for a model"""
        # This is a simplified template
        # In production, use torch-mlir or similar tools
        return f"""// MLIR representation for {model.name}
// This is a placeholder - actual MLIR would be generated by torch-mlir

module @{model.name} {{
  func.func @forward(%arg0: tensor<*xf32>) -> tensor<*xf32> {{
    // Model operations would be here
    return %arg0 : tensor<*xf32>
  }}
}}
"""


def create_benchmark_suite():
    """Create and export benchmark suite"""
    print("="*60)
    print("Creating Benchmark Suite")
    print("="*60)
    
    suite = BenchmarkSuite()
    
    # Save PyTorch models (both standard and JIT)
    print("\n1. Saving PyTorch models...")
    model_info = suite.save_pytorch_models("data/benchmarks/pytorch")
    
    # Export to MLIR format
    print("\n2. Exporting to MLIR format...")
    mlir_files = suite.export_to_mlir_format("data/benchmarks/mlir")
    
    # Create metadata
    metadata = {
        "benchmarks": list(suite.benchmarks.keys()),
        "num_benchmarks": len(suite.benchmarks),
        "mlir_files": mlir_files,
        "pytorch_dir": "data/benchmarks/pytorch",
        "mlir_dir": "data/benchmarks/mlir",
        "description": "Benchmark suite for comparing RL vs PyTorch optimizations",
        "model_info": model_info
    }
    
    metadata_file = Path("data/benchmarks/metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created {len(suite.benchmarks)} benchmarks")
    print(f"✓ PyTorch models: data/benchmarks/pytorch/")
    print(f"✓ MLIR files: data/benchmarks/mlir/")
    print(f"✓ Metadata: {metadata_file}")
    print("\n" + "="*60)


if __name__ == "__main__":
    create_benchmark_suite()
