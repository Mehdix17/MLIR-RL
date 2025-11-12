# DistilBERT Model Support

This document explains how to use the DistilBERT neural network architecture in the MLIR-RL project.

## Overview

The project now supports multiple neural network architectures for the policy and value networks:
- **LSTM** (original): Bidirectional LSTM for sequential processing
- **DistilBERT** (new): Transformer-based architecture with self-attention

## Architecture Details

### DistilBERT Embedding

The DistilBERT embedding layer processes operation features as follows:

1. **Feature Projection**: Projects operation features from their original dimension to DistilBERT's hidden size (768)
2. **Sequence Construction**: Creates a sequence: `[CLS] consumer_operation producer_operation [SEP]`
3. **Self-Attention**: Uses DistilBERT's 6-layer transformer with 12 attention heads
4. **Output**: Uses the [CLS] token representation concatenated with action history

**Configuration:**
- Hidden size: 768
- Number of layers: 6
- Number of attention heads: 12
- Dropout: 0.1
- Max sequence length: 4 tokens (CLS + consumer + producer + SEP)

### Model Parameters

Approximate parameter counts:
- **LSTM model**: ~2-3M parameters
- **DistilBERT model**: ~50-60M parameters (due to transformer layers)

**Note**: DistilBERT requires significantly more memory and compute time but may learn better representations.

## Installation

Install the transformers library:

```bash
pip install transformers>=4.30.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Configuration

Set `model_type` in your config file:

```json
{
  "model_type": "distilbert",
  ...
}
```

Available options: `"lstm"`, `"distilbert"`

### 2. Training

Train with DistilBERT using the provided test config:

```bash
CONFIG_FILE_PATH=config/config_distilbert.json python bin/train.py
```

Or create your own config and specify it:

```bash
CONFIG_FILE_PATH=config/my_config.json python bin/train.py
```

### 3. Evaluation

Evaluation works the same way:

```bash
CONFIG_FILE_PATH=config/config_distilbert.json python bin/evaluate.py
```

## Testing

A test script is provided to verify the implementation:

```bash
python test_distilbert.py
```

This will:
- Initialize the DistilBERT embedding layer
- Create the hierarchical model
- Test forward pass with dummy data
- Verify gradient flow

## Performance Considerations

### Memory

DistilBERT uses significantly more memory than LSTM:
- **LSTM**: ~500MB-1GB
- **DistilBERT**: ~2-4GB

You may need to:
- Reduce `ppo_batch_size` and `value_batch_size`
- Use a GPU with sufficient memory
- Enable gradient checkpointing (future feature)

### Training Time

DistilBERT is slower than LSTM:
- **LSTM**: ~1x baseline
- **DistilBERT**: ~3-5x slower

This is due to:
- More parameters
- Self-attention complexity O(n²)
- More layers (6 transformer blocks vs 1 LSTM layer)

### Learning Rate

Transformer models typically require different learning rates:
- **LSTM**: 0.001 (default)
- **DistilBERT**: 0.0001 or lower (recommended)

The test config uses `lr: 0.0001`.

## Hyperparameter Recommendations

For DistilBERT training, consider:

```json
{
  "model_type": "distilbert",
  "lr": 0.0001,
  "ppo_batch_size": 16,
  "value_batch_size": 16,
  "ppo_epochs": 4,
  "entropy_coef": 0.01,
  ...
}
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Reduce batch sizes: `ppo_batch_size: 8` or `4`
2. Reduce `bench_count` to process fewer benchmarks at once
3. Use CPU instead of GPU (slower but more memory)

### Slow Training

If training is too slow:
1. Reduce number of transformer layers (modify `n_layers` in `DistilBertEmbedding.__init__`)
2. Reduce hidden size (modify `dim` parameter)
3. Use fewer attention heads
4. Consider using LSTM for faster iterations

### NaN Loss

If you see NaN losses:
1. Lower the learning rate: `lr: 0.00005`
2. Enable gradient clipping in training script
3. Check for exploding gradients in DistilBERT layers

## Future Extensions

The architecture is designed to be extensible. To add more models:

1. Create a new embedding class (e.g., `BertEmbedding`, `ConvNextEmbedding`)
2. Add the model type to `Config.model_type` in `utils/config.py`
3. Update `get_embedding_layer()` in `rl_autoschedular/model.py`
4. Test with `test_distilbert.py` as a template

## Comparison: LSTM vs DistilBERT

| Feature | LSTM | DistilBERT |
|---------|------|------------|
| Parameters | ~2-3M | ~50-60M |
| Memory | 0.5-1 GB | 2-4 GB |
| Speed | 1x | 0.2-0.3x |
| Sequence Modeling | Sequential | Parallel (self-attention) |
| Long Dependencies | Limited | Better |
| Recommended Use | Fast iterations | Better representations |

## Questions?

If you encounter issues or have questions about the DistilBERT implementation, check:
1. The test script output: `python test_distilbert.py`
2. Model file: `rl_autoschedular/model.py`
3. Config file: `config/config_distilbert.json`
