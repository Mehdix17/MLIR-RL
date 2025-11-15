# LSTM Training Scripts

Scripts for training LSTM baseline models.

## 📋 Available Scripts

### `test_lstm.sh` ⚡ Quick Test (15 min)
Test LSTM setup with minimal data
```bash
sbatch scripts/lstm/test_lstm.sh
```
- **Config**: `config/test.json`
- **Data**: 17 files from `data/test`
- **Iterations**: 3
- **Purpose**: Verify LSTM setup works

### `train_lstm_baseline.sh` 🎯 Baseline (1 hour)
Train LSTM with standard dataset
```bash
sbatch scripts/lstm/train_lstm_baseline.sh
```
- **Config**: `config/config.json`
- **Data**: 9,441 files from `data/all`
- **Iterations**: 5
- **LR**: 0.001, Batch: 32
- **Purpose**: Baseline comparison

### `train_lstm_augmented.sh` 📊 Extended (12 hours)
Train LSTM with augmented dataset
```bash
sbatch scripts/lstm/train_lstm_augmented.sh
```
- **Config**: `config/config_augmented.json`
- **Data**: 9,941 files (`data/all` + `data/generated`)
- **Iterations**: 1000
- **LR**: 0.0001, Batch: 32
- **Purpose**: Best LSTM performance

### `eval_lstm.sh` 📈 Evaluation
Evaluate trained LSTM models
```bash
# Evaluate latest run
sbatch scripts/lstm/eval_lstm.sh

# Evaluate specific run
export EVAL_DIR=results/lstm/run_0
sbatch scripts/lstm/eval_lstm.sh
```
- **Config**: `config/config.json`
- **Auto-discovery**: Finds latest run if EVAL_DIR not set
- **Purpose**: Measure model performance on test set

## 🚀 Quick Start

1. **Test first** (recommended):
   ```bash
   sbatch scripts/lstm/test_lstm.sh
   ```

2. **Monitor**:
   ```bash
   tail -f logs/test-lstm_*.out
   ```

3. **Full training** (after test passes):
   ```bash
   sbatch scripts/lstm/train_lstm_baseline.sh
   ```

4. **Evaluate model**:
   ```bash
   sbatch scripts/lstm/eval_lstm.sh
   ```

## 📊 Expected Results

- Test should complete in ~15 minutes
- Baseline training in ~1 hour
- Augmented training in ~12 hours
- Evaluation in ~10-15 minutes

Check results in `results/lstm/` directory.
