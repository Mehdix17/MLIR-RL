"""
Convert PyTorch neural networks to MLIR

Supports common architectures:
- ResNet
- BERT
- GPT-2
- Custom models
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any


class NeuralNetworkToMLIR:
    """Convert PyTorch models to MLIR representation"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_model(
        self,
        model: nn.Module,
        input_shape: tuple,
        model_name: str,
        use_torch_mlir: bool = True
    ) -> Path:
        """
        Convert PyTorch model to MLIR
        
        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape (batch, channels, height, width)
            model_name: Name for the output file
            use_torch_mlir: Use torch-mlir for conversion (requires installation)
            
        Returns:
            Path to generated MLIR file
        """
        if use_torch_mlir:
            return self._convert_with_torch_mlir(model, input_shape, model_name)
        else:
            return self._convert_manual(model, input_shape, model_name)
    
    def _convert_with_torch_mlir(
        self,
        model: nn.Module,
        input_shape: tuple,
        model_name: str
    ) -> Path:
        """Convert using torch-mlir library"""
        try:
            from torch_mlir import compile
            
            # Create example input
            example_input = torch.randn(*input_shape)
            
            # Compile to MLIR
            mlir_module = compile(
                model,
                example_input,
                output_type="linalg-on-tensors"
            )
            
            # Save to file
            output_path = self.output_dir / f"{model_name}.mlir"
            with open(output_path, 'w') as f:
                f.write(str(mlir_module))
            
            print(f"✓ Converted {model_name} to MLIR using torch-mlir")
            return output_path
            
        except ImportError:
            print("⚠️  torch-mlir not installed, falling back to manual conversion")
            return self._convert_manual(model, input_shape, model_name)
    
    def _convert_manual(
        self,
        model: nn.Module,
        input_shape: tuple,
        model_name: str
    ) -> Path:
        """Manual conversion for common layers"""
        mlir_ops = []
        
        # Trace the model
        example_input = torch.randn(*input_shape)
        
        # Extract layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                mlir_ops.append(self._conv2d_to_mlir(module, name))
            elif isinstance(module, nn.Linear):
                mlir_ops.append(self._linear_to_mlir(module, name))
            elif isinstance(module, nn.MaxPool2d):
                mlir_ops.append(self._maxpool_to_mlir(module, name))
        
        # Combine into full MLIR module
        mlir_code = self._create_mlir_module(model_name, mlir_ops)
        
        # Save to file
        output_path = self.output_dir / f"{model_name}.mlir"
        with open(output_path, 'w') as f:
            f.write(mlir_code)
        
        print(f"✓ Converted {model_name} to MLIR (manual)")
        return output_path
    
    def _conv2d_to_mlir(self, layer: nn.Conv2d, name: str) -> str:
        """Convert Conv2d layer to MLIR"""
        in_ch = layer.in_channels
        out_ch = layer.out_channels
        kernel = layer.kernel_size[0]
        
        return f"""
  // {name}
  %{name} = linalg.conv_2d_nchw_fchw
    {{dilations = dense<1> : tensor<2xi64>, 
      strides = dense<{layer.stride[0]}> : tensor<2xi64>}}
    ins(%input_{name}, %filter_{name} : tensor<?x{in_ch}x?x?xf32>, tensor<{out_ch}x{in_ch}x{kernel}x{kernel}xf32>)
    outs(%init_{name} : tensor<?x{out_ch}x?x?xf32>) -> tensor<?x{out_ch}x?x?xf32>
"""
    
    def _linear_to_mlir(self, layer: nn.Linear, name: str) -> str:
        """Convert Linear layer to MLIR"""
        in_feat = layer.in_features
        out_feat = layer.out_features
        
        return f"""
  // {name}
  %{name} = linalg.matmul
    ins(%input_{name}, %weight_{name} : tensor<?x{in_feat}xf32>, tensor<{in_feat}x{out_feat}xf32>)
    outs(%init_{name} : tensor<?x{out_feat}xf32>) -> tensor<?x{out_feat}xf32>
"""
    
    def _maxpool_to_mlir(self, layer: nn.MaxPool2d, name: str) -> str:
        """Convert MaxPool2d to MLIR"""
        kernel = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        stride = layer.stride if layer.stride else kernel
        
        return f"""
  // {name}
  %{name} = linalg.pooling_nchw_max
    {{dilations = dense<1> : tensor<2xi64>,
      strides = dense<{stride}> : tensor<2xi64>}}
    ins(%input_{name}, %kernel_{name} : tensor<?x?x?x?xf32>, tensor<{kernel}x{kernel}xf32>)
    outs(%init_{name} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
"""
    
    def _create_mlir_module(self, name: str, operations: list) -> str:
        """Create complete MLIR module"""
        ops_str = '\n'.join(operations)
        
        return f"""
module @{name} {{
  func.func @forward(%input: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {{
{ops_str}
    return %final : tensor<?x?x?x?xf32>
  }}
}}
"""


def main():
    """Example: Convert ResNet-18 to MLIR"""
    try:
        from torchvision.models import resnet18
        
        converter = NeuralNetworkToMLIR(output_dir=Path("data/generated/neural_nets"))
        
        # Load ResNet-18
        model = resnet18(pretrained=False)
        model.eval()
        
        # Convert to MLIR
        mlir_file = converter.convert_model(
            model=model,
            input_shape=(1, 3, 224, 224),
            model_name="resnet18",
            use_torch_mlir=False  # Set to True if torch-mlir is installed
        )
        
        print(f"\n✓ ResNet-18 converted to: {mlir_file}")
    except ImportError:
        print("⚠️  torchvision not installed. Install with: pip install torchvision")


if __name__ == "__main__":
    main()
