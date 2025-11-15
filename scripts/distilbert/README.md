# DistilBERT Training Scripts

Scripts for training DistilBERT transformer models.

## 📋 Available Scripts

### `test_distilbert.sh` ⚡ Quick Test (20 min)
Test DistilBERT setup with minimal data
```bash
sbatch scripts/distilbert/test_distilbert.sh
```
- **Config**: `config/test_distilbert.json`
- **Data**: 17 files from `data/test`
- **Iterations**: 3
- **Model**: 6-layer DistilBERT (768 hidden, 12 heads)
- **Purpose**: Verify transformer setup works

### `train_distilbert.sh` 🤖 Full Training (3 hours)
Train DistilBERT with full dataset
```bash
sbatch scripts/distilbert/train_distilbert.sh
```
- **Config**: `config/config_distilbert.json`
- **Data**: 9,441 files from `data/all`
- **Iterations**: 5
- **LR**: 0.0001, Batch: 16
- **Purpose**: Transformer-based optimization

### `eval_distilbert.sh` 📈 Evaluation
Evaluate trained DistilBERT models
```bash
# Evaluate latest run
sbatch scripts/distilbert/eval_distilbert.sh

# Evaluate specific run
export EVAL_DIR=results/distilbert/run_0
sbatch scripts/distilbert/eval_distilbert.sh
```
- **Config**: `config/config_distilbert.json`
- **Auto-discovery**: Finds latest run if EVAL_DIR not set
- **Purpose**: Measure model performance on test set

## 🚀 Quick Start

1. **Test first** (recommended):
   ```bash
   sbatch scripts/distilbert/test_distilbert.sh
   ```

2. **Monitor**:
   ```bash
   tail -f logs/test-distilbert_*.out
   ```

3. **Full training** (after test passes):
   ```bash
   sbatch scripts/distilbert/train_distilbert.sh
   ```

4. **Evaluate model**:
   ```bash
   sbatch scripts/distilbert/eval_distilbert.sh
   ```

## ⚠️ Important Notes

- **Slower than LSTM**: Expect 3-5x longer training time
- **Memory intensive**: Uses ~4GB vs ~1GB for LSTM
- **Smaller batches**: Batch size 16 vs 32 for LSTM
- **Higher accuracy**: Usually outperforms LSTM on complex tasks

## 📊 Expected Results

- Test should complete in ~20 minutes
- Full training in ~3 hours
- Evaluation in ~15-20 minutes
- Better performance on complex MLIR programs

Check results in `results/distilbert/` directory.
