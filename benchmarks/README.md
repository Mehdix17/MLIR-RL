# MLIR-RL Benchmarks

This directory contains benchmark suites for evaluating the RL agent's optimization performance.

## Structure

```
benchmarks/
├── single_ops/     # Individual operations (matmul, conv2d, pooling)
└── neural_nets/    # Complete neural networks (ResNet, BERT, etc.)
```

## Single Operations

The `single_ops/` directory contains MLIR representations of individual operations:

- **Matrix multiplication** (various sizes)
- **Convolutions** (2D, 3D)
- **Pooling** (max, average)
- **Element-wise operations** (add, mul, relu)

### Generating Single Operation Benchmarks

```bash
python data_generation/random_mlir_gen.py
```

This will populate `benchmarks/single_ops/` with randomly generated operations.

## Neural Networks

The `neural_nets/` directory contains MLIR representations of complete neural networks:

- **ResNet** (18, 34, 50)
- **BERT** (base, large)
- **GPT-2** (small, medium)
- Custom architectures

### Converting PyTorch Models to MLIR

```bash
python data_generation/nn_to_mlir.py
```

This will convert PyTorch models to MLIR and save them in `benchmarks/neural_nets/`.

## Benchmark Naming Convention

### Single Operations
- Format: `{operation}_{dimension}_{id}.mlir`
- Examples:
  - `matmul_512x512_0001.mlir`
  - `conv2d_224x224_0042.mlir`
  - `pooling_2x2_0015.mlir`

### Neural Networks
- Format: `{model_name}_{variant}.mlir`
- Examples:
  - `resnet18.mlir`
  - `bert_base.mlir`
  - `gpt2_small.mlir`

## Using Benchmarks

### Evaluate Single Operations

```python
from evaluation import SingleOperationEvaluator

evaluator = SingleOperationEvaluator(
    agent_model_path="results/best_model.pt",
    benchmark_dir="benchmarks/single_ops"
)

results = evaluator.evaluate_benchmark_suite()
```

### Evaluate Neural Networks

```python
from evaluation import NeuralNetworkEvaluator

evaluator = NeuralNetworkEvaluator(
    agent_model_path="results/best_model.pt",
    benchmark_dir="benchmarks/neural_nets"
)

results = evaluator.evaluate_benchmark_suite()
```

## Adding Custom Benchmarks

### Manual MLIR Files

Place your `.mlir` files directly in the appropriate directory:
- Single ops → `single_ops/`
- Neural nets → `neural_nets/`

### Programmatic Generation

Use the data generation utilities:

```python
from data_generation import RandomMLIRGenerator

generator = RandomMLIRGenerator()
generator.generate_dataset(
    num_samples=100,
    output_dir="benchmarks/single_ops/custom",
    operation_types=['matmul', 'conv2d']
)
```

## Benchmark Metadata

Each benchmark should include metadata for tracking:
- Operation type
- Input dimensions
- Expected output dimensions
- Baseline performance (if available)

This can be stored in companion `.json` files:

```json
{
  "operation": "matmul",
  "dimensions": "512x512 @ 512x512",
  "baseline_time_ms": 15.4,
  "pytorch_time_ms": 12.1
}
```
