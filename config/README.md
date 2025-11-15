# Configuration Files

This directory contains standardized configuration files for MLIR-RL training.

## 📋 Standardized Structure

All config files follow this consistent structure:

```json
{
    "_description": "Brief description of this config",
    "model_type": "lstm|distilbert|gpt2|...",
    "model_config": { /* model-specific params */ },
    "observation_space": { /* observation dims */ },
    "action_space": { /* action definitions */ },
    "exploration": { /* exploration strategy */ },
    "architecture": { /* architecture settings */ },
    "reward": { /* reward function */ },
    "training": { /* training hyperparams */ },
    "ppo": { /* PPO algorithm params */ },
    "value_function": { /* value function params */ },
    "data_paths": { /* dataset paths */ },
    "augmentation": { /* optional augmentation */ },
    "logging": { /* experiment tracking */ }
}
```

## 📁 Available Configurations

### `config.json` - LSTM Baseline
- **Model**: LSTM (original architecture)
- **Dataset**: `data/all/` (9,441 programs)
- **Use case**: Baseline experiments
- **Training time**: ~5 iterations

### `config_distilbert.json` - DistilBERT Transformer
- **Model**: DistilBERT (6-layer transformer)
- **Dataset**: `data/all/` (9,441 programs)
- **Use case**: Transformer-based approach
- **Note**: Lower learning rate (0.0001), smaller batch size (16)

### `config_augmented.json` - Augmented Dataset
- **Model**: LSTM
- **Dataset**: `data/all/` + `data/generated/` (augmented)
- **Use case**: Training with augmented data
- **Features**: Includes `augmentation` section
- **Training time**: ~1000 iterations

### `test.json` - Quick Testing
- **Model**: LSTM
- **Dataset**: `data/test/` (17 programs only)
- **Use case**: Fast iteration testing
- **Training time**: 3 iterations (quick validation)

### `config_template.json` - New Model Template
- **Purpose**: Template for creating new configurations
- **Usage**: Copy and modify for new models (GPT-2, BERT, etc.)

## 🎯 Configuration Sections

### 1. **Model Type & Config**
```json
{
    "model_type": "lstm",           // Model architecture identifier
    "model_config": {               // Model-specific hyperparameters
        "hidden_size": 768,         // For transformers only
        "num_attention_heads": 12,  // Empty {} for LSTM
        ...
    }
}
```

**Supported model types**:
- `lstm` - LSTM baseline (empty config)
- `distilbert` - DistilBERT transformer
- `gpt2` - GPT-2 (coming soon)
- `bert` - Full BERT (coming soon)
- `convnext` - ConvNext (coming soon)

### 2. **Observation Space**
```json
{
    "observation_space": {
        "max_num_stores_loads": 7,      // Max memory operations
        "max_num_loops": 12,            // Max loop nesting depth
        "max_num_load_store_dim": 12,   // Max dimensions
        "num_tile_sizes": 7,            // Number of tiling options
        "vect_size_limit": 512          // Vectorization limit
    }
}
```

### 3. **Action Space**
```json
{
    "action_space": {
        "order": [["I"], ["!", "I", "NT"], ["!", "I"], ["V", "NT"]],
        "interchange_mode": "pointers"  // Action representation
    }
}
```

### 4. **Exploration**
```json
{
    "exploration": {
        "strategy": ["entropy"],    // Exploration method
        "init_epsilon": 0.5         // Initial epsilon for ε-greedy
    }
}
```

### 5. **Architecture**
```json
{
    "architecture": {
        "new_architecture": false,      // Use new architecture?
        "activation": "relu",           // Activation function
        "normalize_bounds": "max",      // Bound normalization
        "normalize_adv": "standard"     // Advantage normalization
    }
}
```

### 6. **Reward**
```json
{
    "reward": {
        "sparse_reward": true,      // Sparse vs dense rewards
        "split_ops": true           // Split operations in reward
    }
}
```

### 7. **Training**
```json
{
    "training": {
        "bench_count": 64,          // Number of benchmarks per iteration
        "replay_count": 10,         // Number of replays
        "nb_iterations": 5,         // Total training iterations
        "reuse_experience": "none", // Experience replay strategy
        "truncate": 5               // Trajectory truncation
    }
}
```

### 8. **PPO Algorithm**
```json
{
    "ppo": {
        "ppo_epochs": 4,            // PPO update epochs
        "ppo_batch_size": 32,       // Batch size for PPO
        "lr": 0.001,                // Learning rate
        "gamma": 0.99,              // Discount factor
        "clip_epsilon": 0.2,        // PPO clipping parameter
        "entropy_coef": 0.01        // Entropy bonus coefficient
    }
}
```

### 9. **Value Function**
```json
{
    "value_function": {
        "value_epochs": 0,          // Value function training epochs
        "value_batch_size": 32,     // Batch size for value updates
        "value_coef": 0.5,          // Value loss coefficient
        "value_clip": false         // Clip value function updates?
    }
}
```

### 10. **Data Paths**
```json
{
    "data_paths": {
        "benchmarks_folder_path": "data/all/code_files",
        "json_file": "data/all/execution_times_train.json",
        "eval_json_file": "data/all/execution_times_eval.json"
    }
}
```

### 11. **Augmentation** (Optional)
```json
{
    "augmentation": {
        "use_augmentation": true,
        "augmentation_ratio": 0.3,
        "augmentation_folder_path": "data/generated/code_files",
        "augmentation_json_file": "data/generated/execution_times_generated.json"
    }
}
```

Only include this section if using augmented data.

### 12. **Logging**
```json
{
    "logging": {
        "tags": ["lstm", "baseline", "all"],
        "debug": false,
        "results_dir": "results",
        "main_exec_data_file": ""
    }
}
```

## 🚀 Creating New Configurations

### Method 1: Copy Template
```bash
# Copy template
cp config/config_template.json config/config_gpt2.json

# Edit new file
# 1. Change "_description"
# 2. Set "model_type" to "gpt2"
# 3. Update "model_config" with GPT-2 hyperparameters
# 4. Adjust "ppo" learning rate if needed
# 5. Update "logging" tags
```

### Method 2: Copy Similar Config
```bash
# For transformer models, copy DistilBERT config
cp config/config_distilbert.json config/config_bert.json

# For LSTM variants, copy base config
cp config/config.json config/config_gru.json
```

### Method 3: Quick Generation
```python
import json

config = {
    "_description": "GPT-2 transformer configuration",
    "model_type": "gpt2",
    "model_config": {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "max_seq_length": 128,
        "vocab_size": 10000,
        "dropout": 0.1
    },
    # ... copy other sections from template
}

with open("config/config_gpt2.json", "w") as f:
    json.dump(config, f, indent=4)
```

## 📊 Configuration Best Practices

### 1. **Model-Specific Settings**

**For LSTM**:
- `model_config: {}`
- `lr: 0.001`
- `ppo_batch_size: 32`

**For Transformers** (DistilBERT, GPT-2, BERT):
- Include full `model_config` with attention parameters
- Lower learning rate: `lr: 0.0001` or `0.00005`
- Smaller batch size: `ppo_batch_size: 16` or `8`
- Higher dropout: `dropout: 0.1` to prevent overfitting

### 2. **Learning Rate Guidelines**

| Model Type | Recommended LR | Batch Size |
|------------|---------------|------------|
| LSTM       | 0.001         | 32         |
| DistilBERT | 0.0001        | 16         |
| GPT-2      | 0.0001        | 16         |
| BERT       | 0.00005       | 8          |
| ConvNext   | 0.0001        | 16         |

### 3. **Training Duration**

| Config Type | Iterations | Time Estimate |
|-------------|-----------|---------------|
| test.json   | 3         | ~15 minutes   |
| config.json | 5         | ~30 minutes   |
| config_augmented.json | 1000 | ~10 hours |

### 4. **Data Path Consistency**

Always use relative paths from project root:
```json
"benchmarks_folder_path": "data/all/code_files",    // ✅ Good
"benchmarks_folder_path": "/scratch/.../data/...",  // ❌ Bad (absolute)
```

### 5. **Naming Conventions**

- `config.json` - Default/baseline
- `config_<model>.json` - Model-specific (e.g., `config_gpt2.json`)
- `config_<feature>.json` - Feature-specific (e.g., `config_augmented.json`)
- `test.json` - Quick testing

## 🔧 Usage Examples

### Basic Training
```bash
# LSTM baseline
CONFIG_FILE_PATH=config/config.json python bin/train.py

# DistilBERT
CONFIG_FILE_PATH=config/config_distilbert.json python bin/train.py

# Augmented data
CONFIG_FILE_PATH=config/config_augmented.json python bin/train.py
```

### Quick Testing
```bash
# Fast iteration test (3 iterations only)
CONFIG_FILE_PATH=config/test.json python bin/train.py
```

### Evaluation
```bash
# Evaluate trained model
python bin/evaluate.py --config config/config_distilbert.json
```

## 🐛 Troubleshooting

### Issue: Model config not recognized
**Solution**: Ensure `model_type` matches registered models in `rl_autoschedular/models/embeddings/factory.py`

### Issue: Data paths not found
**Solution**: Verify paths exist and use relative paths from project root

### Issue: Out of memory
**Solution**: Reduce `ppo_batch_size` and `value_batch_size`

### Issue: Training too slow
**Solution**: Use `test.json` for quick validation first

## 📚 Related Documentation

- **Model Registry**: `rl_autoschedular/models/embeddings/factory.py`
- **Training Script**: `bin/train.py`
- **Project Roadmap**: `docs/ROADMAP.md`
- **Data Organization**: `docs/guides/DATA_ORGANIZATION_COMPLETE.md`

---

*Last updated: November 15, 2025*
