"""
Test data generation and evaluation modules

Quick verification that all components are working correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_generation import RandomMLIRGenerator, NeuralNetworkToMLIR
        print("  ✓ data_generation imports successful")
    except ImportError as e:
        print(f"  ✗ data_generation import failed: {e}")
        return False
    
    try:
        from evaluation import SingleOperationEvaluator, NeuralNetworkEvaluator, PyTorchBaseline
        print("  ✓ evaluation imports successful")
    except ImportError as e:
        print(f"  ✗ evaluation import failed: {e}")
        return False
    
    return True


def test_random_generator():
    """Test random MLIR generator"""
    print("\nTesting RandomMLIRGenerator...")
    
    try:
        from data_generation import RandomMLIRGenerator
        
        generator = RandomMLIRGenerator(seed=42)
        
        # Generate a few test files
        test_dir = Path("data/test_generation")
        files = generator.generate_dataset(
            num_samples=5,
            output_dir=test_dir,
            operation_types=['matmul', 'conv2d']
        )
        
        if len(files) == 5:
            print(f"  ✓ Generated 5 test files in {test_dir}")
            
            # Check first file content
            with open(files[0]) as f:
                content = f.read()
                if 'func.func' in content and 'linalg.' in content:
                    print("  ✓ Generated MLIR looks valid")
                else:
                    print("  ✗ Generated MLIR doesn't look valid")
                    return False
        else:
            print(f"  ✗ Expected 5 files, got {len(files)}")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Random generator test failed: {e}")
        return False


def test_nn_converter():
    """Test neural network converter"""
    print("\nTesting NeuralNetworkToMLIR...")
    
    try:
        from data_generation import NeuralNetworkToMLIR
        import torch
        import torch.nn as nn
        
        # Create simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.pool = nn.MaxPool2d(2)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        converter = NeuralNetworkToMLIR(output_dir=Path("data/test_nn_conversion"))
        
        mlir_file = converter.convert_model(
            model=model,
            input_shape=(1, 3, 32, 32),
            model_name="simple_test",
            use_torch_mlir=False
        )
        
        if mlir_file.exists():
            print(f"  ✓ Converted model to {mlir_file}")
            
            with open(mlir_file) as f:
                content = f.read()
                if 'module' in content and 'func.func' in content:
                    print("  ✓ Converted MLIR looks valid")
                else:
                    print("  ✗ Converted MLIR doesn't look valid")
                    return False
        else:
            print(f"  ✗ Output file not created: {mlir_file}")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ NN converter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_baseline():
    """Test PyTorch baseline"""
    print("\nTesting PyTorchBaseline...")
    
    try:
        from evaluation import PyTorchBaseline
        
        baseline = PyTorchBaseline(device="cpu")
        
        # Test simple matmul benchmark
        result = baseline.benchmark_matmul(64, 64, 64, num_runs=5)
        
        if result['mean_time'] > 0:
            print(f"  ✓ Matmul benchmark: {result['mean_time']*1000:.2f}ms")
        else:
            print("  ✗ Invalid benchmark time")
            return False
        
        # Test conv2d benchmark
        result = baseline.benchmark_conv2d(1, 3, 16, 32, 32, 3, num_runs=5)
        
        if result['mean_time'] > 0:
            print(f"  ✓ Conv2d benchmark: {result['mean_time']*1000:.2f}ms")
        else:
            print("  ✗ Invalid benchmark time")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ PyTorch baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that all directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        "data_generation",
        "evaluation",
        "benchmarks",
        "benchmarks/single_ops",
        "benchmarks/neural_nets"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ missing")
            all_exist = False
    
    return all_exist


def test_readme_files():
    """Test that README files exist"""
    print("\nChecking README files...")
    
    readme_files = [
        "data_generation/README.md",
        "evaluation/README.md",
        "benchmarks/README.md",
        "docs/guides/DATA_GENERATION_INTEGRATION.md"
    ]
    
    all_exist = True
    for readme in readme_files:
        path = Path(readme)
        if path.exists():
            print(f"  ✓ {readme}")
        else:
            print(f"  ✗ {readme} missing")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests"""
    print("="*60)
    print("MLIR-RL Data Generation & Evaluation Test Suite")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['directory_structure'] = test_directory_structure()
    results['readme_files'] = test_readme_files()
    results['random_generator'] = test_random_generator()
    results['nn_converter'] = test_nn_converter()
    results['pytorch_baseline'] = test_pytorch_baseline()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print("="*60)
    print(f"Result: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Integration is complete.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
