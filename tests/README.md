# Tests

This directory contains test suites for the MLIR-RL project.

## Test Files

### Integration Tests
- **`test_integration.py`** - Tests data generation and evaluation modules
  ```bash
  python tests/test_integration.py
  ```

### DistilBERT Tests
- **`test_distilbert.py`** - Basic DistilBERT model tests
- **`test_distilbert_data.py`** - DistilBERT data preprocessing tests (comprehensive)
- **`test_distilbert_simple.py`** - Simple DistilBERT tests

### Architecture Tests
- **`test_refactoring.py`** - Tests for modular architecture refactoring

## Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/

# Or run individually
python tests/test_integration.py
python tests/test_distilbert_data.py
```

### Run Specific Test
```bash
python tests/test_integration.py
```

## Test Coverage

- ✅ Data generation (RandomMLIRGenerator, NeuralNetworkToMLIR)
- ✅ Evaluation (SingleOperationEvaluator, PyTorchBaseline)
- ✅ DistilBERT tokenization and data preprocessing
- ✅ Model architecture (LSTM, DistilBERT)
- ✅ Modular structure imports

## Adding New Tests

Create test files following the pattern:
```python
"""
Test description
"""

def test_feature():
    # Test implementation
    assert True

if __name__ == "__main__":
    test_feature()
```
