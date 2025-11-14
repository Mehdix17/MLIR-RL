# Data Generation for MLIR-RL

This module generates training data and benchmarks for the MLIR-RL agent.

## Overview

The data generation system provides tools to create:
1. **Random MLIR programs** for training the RL agent
2. **Neural network conversions** from PyTorch to MLIR
3. **Benchmark suites** for evaluation

## Components

### 1. Random MLIR Generator (`random_mlir_gen.py`)

Generates synthetic MLIR programs for training.

**Supported Operations:**
- Matrix multiplication (matmul)
- 2D Convolution (conv2d)
- Max/Average Pooling
- Element-wise operations (add, mul, max)

**Usage:**
```python
from data_generation import RandomMLIRGenerator

generator = RandomMLIRGenerator(seed=42)

# Generate training dataset
files = generator.generate_dataset(
    num_samples=1000,
    output_dir="data/train",
    operation_types=['matmul', 'conv2d', 'pooling']
)
```

**Command Line:**
```bash
python data_generation/random_mlir_gen.py
```

This generates:
- 1000 training files in `data/generated/train/`
- 200 test files in `data/generated/test/`

### 2. Neural Network to MLIR Converter (`nn_to_mlir.py`)

Converts PyTorch models to MLIR representation.

**Supported Models:**
- ResNet family (18, 34, 50, 101, 152)
- BERT (base, large)
- GPT-2 (small, medium, large)
- Custom PyTorch models

**Usage:**
```python
from data_generation import NeuralNetworkToMLIR
from torchvision.models import resnet18

converter = NeuralNetworkToMLIR(output_dir="data/neural_nets")

model = resnet18()
mlir_file = converter.convert_model(
    model=model,
    input_shape=(1, 3, 224, 224),
    model_name="resnet18",
    use_torch_mlir=False  # Set True if torch-mlir installed
)
```

**Command Line:**
```bash
python data_generation/nn_to_mlir.py
```

## Installation

### Basic Requirements
```bash
pip install torch torchvision numpy
```

### Optional: torch-mlir (for better conversion)
```bash
pip install torch-mlir
```

Without torch-mlir, the converter uses manual layer extraction (works for common layers).

## Quick Start

### Generate Training Data

```bash
# Generate 1000 random MLIR programs
python -c "
from data_generation import RandomMLIRGenerator
from pathlib import Path

gen = RandomMLIRGenerator()
gen.generate_dataset(1000, Path('data/train'))
"
```

### Convert Neural Network

```bash
# Convert ResNet-18 to MLIR
python data_generation/nn_to_mlir.py
```

### Custom Generation Script

```python
from data_generation import RandomMLIRGenerator, NeuralNetworkToMLIR
from pathlib import Path

# 1. Generate random operations
generator = RandomMLIRGenerator(seed=42)
generator.generate_dataset(
    num_samples=500,
    output_dir=Path("data/custom_train"),
    operation_types=['matmul', 'conv2d']
)

# 2. Convert neural networks
converter = NeuralNetworkToMLIR(output_dir=Path("data/neural_nets"))

# Example: Convert a custom model
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64*111*111, 10)
    
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleNet()
converter.convert_model(
    model=model,
    input_shape=(1, 3, 224, 224),
    model_name="simple_net"
)
```

## Operation Templates

### Matrix Multiplication
```mlir
func.func @matmul(%A: tensor<MxKxf32>, %B: tensor<KxNxf32>) -> tensor<MxNxf32> {
  %result = linalg.matmul
    ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>)
    outs(%init : tensor<MxNxf32>) -> tensor<MxNxf32>
  return %result : tensor<MxNxf32>
}
```

### 2D Convolution
```mlir
func.func @conv2d(
  %input: tensor<NxCxHxWxf32>,
  %filter: tensor<KxCxRxSxf32>
) -> tensor<NxKxHxWxf32> {
  %result = linalg.conv_2d_nchw_fchw
    ins(%input, %filter : tensor<NxCxHxWxf32>, tensor<KxCxRxSxf32>)
    outs(%init : tensor<NxKxHxWxf32>) -> tensor<NxKxHxWxf32>
  return %result : tensor<NxKxHxWxf32>
}
```

### Pooling
```mlir
func.func @maxpool(%input: tensor<NxCxHxWxf32>) -> tensor<NxCxH'xW'xf32> {
  %result = linalg.pooling_nchw_max
    ins(%input, %kernel : tensor<NxCxHxWxf32>, tensor<PxPxf32>)
    outs(%init : tensor<NxCxH'xW'xf32>) -> tensor<NxCxH'xW'xf32>
  return %result : tensor<NxCxH'xW'xf32>
}
```

## Generated Data Structure

```
data/
├── generated/
│   ├── train/
│   │   ├── matmul_0000.mlir
│   │   ├── conv2d_0001.mlir
│   │   └── ...
│   ├── test/
│   │   └── ...
│   └── neural_nets/
│       ├── resnet18.mlir
│       ├── bert_base.mlir
│       └── ...
```

## Configuration

### Random Generation Parameters

```python
generator = RandomMLIRGenerator(seed=42)

# Customize dimensions
generator.generate_dataset(
    num_samples=1000,
    output_dir="data/train",
    operation_types=['matmul', 'conv2d', 'pooling', 'elementwise']
)
```

**Dimension Ranges:**
- Matrix sizes: 64, 128, 256, 512, 1024
- Batch sizes: 1, 8, 16, 32
- Channels: 3, 16, 32, 64, 128, 256
- Spatial dims: 28, 32, 64, 128, 224
- Kernels: 3, 5, 7

### Neural Network Conversion

```python
converter = NeuralNetworkToMLIR(output_dir="data/neural_nets")

# With torch-mlir (better accuracy)
converter.convert_model(
    model=model,
    input_shape=(1, 3, 224, 224),
    model_name="resnet18",
    use_torch_mlir=True
)

# Manual mode (faster, common layers only)
converter.convert_model(
    model=model,
    input_shape=(1, 3, 224, 224),
    model_name="resnet18",
    use_torch_mlir=False
)
```

## Troubleshooting

### Issue: torch-mlir not found
**Solution:** Install torch-mlir or use manual conversion:
```python
converter.convert_model(..., use_torch_mlir=False)
```

### Issue: Unsupported layer type
**Solution:** Extend `_convert_manual` in `nn_to_mlir.py` to handle custom layers:
```python
def _convert_manual(self, model, input_shape, model_name):
    for name, module in model.named_modules():
        if isinstance(module, CustomLayer):
            mlir_ops.append(self._custom_to_mlir(module, name))
```

### Issue: Out of memory during generation
**Solution:** Generate in smaller batches:
```python
for i in range(10):
    generator.generate_dataset(
        num_samples=100,
        output_dir=f"data/train/batch_{i}"
    )
```

## Next Steps

1. **Generate data**: Use `random_mlir_gen.py` to create training dataset
2. **Convert models**: Use `nn_to_mlir.py` for neural network benchmarks
3. **Train agent**: Use generated data with `bin/train.py`
4. **Evaluate**: Use evaluation module to measure performance

## Related Documentation

- [Evaluation Guide](../evaluation/README.md)
- [Benchmarks](../benchmarks/README.md)
- [Training Guide](../docs/guides/SLURM_GUIDE.md)
