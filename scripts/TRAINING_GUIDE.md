# Scripts Directory

Training and utility scripts for the MLIR-RL project.

---

## 🚀 Training Scripts

### **Quick Tests** (Start here!)

#### `test_lstm.sh` ⚡ Fast (~5-15 min)
Test LSTM setup with minimal data
```bash
sbatch scripts/test_lstm.sh          # Submit to SLURM
# OR
bash scripts/test_lstm.sh            # Run locally
```
- **Config**: `config/test.json`
- **Data**: 17 files (`data/test`)
- **Iterations**: 3
- **Purpose**: Verify setup works

#### `test_distilbert.sh` ⚡ Fast (~10-20 min)
Test DistilBERT setup with minimal data
```bash
sbatch scripts/test_distilbert.sh
```
- **Config**: `config/test_distilbert.json`
- **Data**: 17 files (`data/test`)
- **Iterations**: 3
- **Purpose**: Verify transformer works

---

### **Full Training Scripts**

#### `train_lstm_baseline.sh` 🎯 Baseline
Train LSTM with standard dataset
```bash
sbatch scripts/train_lstm_baseline.sh
```
- **Config**: `config/config.json`
- **Data**: 9,441 files (`data/all`)
- **Time**: ~30-60 minutes
- **Iterations**: 5
- **LR**: 0.001, Batch: 32

#### `train_lstm_augmented.sh` 📊 Extended
Train LSTM with augmented data
```bash
sbatch scripts/train_lstm_augmented.sh
```
- **Config**: `config/config_augmented.json`
- **Data**: 9,941 files (`data/all` + `data/generated`)
- **Time**: ~10-15 hours
- **Iterations**: 1000
- **LR**: 0.0001, Batch: 32

#### `train_distilbert.sh` 🤖 Transformer
Train DistilBERT transformer model
```bash
sbatch scripts/train_distilbert.sh
```
- **Config**: `config/config_distilbert.json`
- **Data**: 9,441 files (`data/all`)
- **Time**: ~2-4 hours (slower than LSTM)
- **Iterations**: 5
- **LR**: 0.0001, Batch: 16

---

## 📊 Data Management Scripts

### `augment_dataset.py`
Generate synthetic MLIR programs for training
```bash
python scripts/augment_dataset.py --num-samples 1000
```

### `organize_data.py`
Organize data folder structure
```bash
python scripts/organize_data.py
```

### `data_quickref.sh`
Quick reference for data statistics
```bash
bash scripts/data_quickref.sh
```

---

## 🔄 Legacy Scripts

### `train.sh`
Original generic training script (use specific scripts above instead)
```bash
export CONFIG_FILE_PATH=config/config.json
sbatch scripts/train.sh
```

### `eval.sh`
Evaluation script for trained models
```bash
export EVAL_DIR=results/run_0
sbatch scripts/eval.sh
```

### `neptune-sync.sh`
Sync experiments to Neptune.ai
```bash
bash scripts/neptune-sync.sh
```

---

## 📋 Recommended Workflow

### **Step 1: Quick Validation** (~15 min)
```bash
# Test LSTM first (faster)
sbatch scripts/test_lstm.sh

# Check results
tail -f logs/test-lstm_*.out
```

### **Step 2: Test DistilBERT** (~20 min)
```bash
# If LSTM test passed, try DistilBERT
sbatch scripts/test_distilbert.sh

# Monitor progress
tail -f logs/test-distilbert_*.out
```

### **Step 3: Full Training** (hours)
```bash
# Choose one based on your goals:

# Baseline comparison
sbatch scripts/train_lstm_baseline.sh

# Best performance (recommended)
sbatch scripts/train_lstm_augmented.sh

# Transformer approach
sbatch scripts/train_distilbert.sh
```

### **Step 4: Compare Results**
```bash
# Check all results
ls -lh results/

# View logs
ls -lh logs/

# Compare performance metrics
python analysis/plot_results.py
```

---

## 🎯 Which Script Should I Use?

| Goal | Script | Time | Data |
|------|--------|------|------|
| **Just testing setup** | `test_lstm.sh` | 15 min | 17 files |
| **Test transformer** | `test_distilbert.sh` | 20 min | 17 files |
| **Quick baseline** | `train_lstm_baseline.sh` | 1 hour | 9,441 files |
| **Best LSTM results** | `train_lstm_augmented.sh` | 12 hours | 9,941 files |
| **Try transformer** | `train_distilbert.sh` | 3 hours | 9,441 files |

---

## ⚙️ Environment Variables

All scripts automatically set:
- `CONFIG_FILE_PATH` - Configuration file path
- `DASK_NODES` - Number of Dask workers
- `OMP_NUM_THREADS` - OpenMP threads
- `AST_DUMPER_BIN_PATH` - AST dumper binary

You can override by setting before running:
```bash
export CONFIG_FILE_PATH=config/custom.json
sbatch scripts/train_lstm_baseline.sh
```

---

## 📝 Script Output

All scripts create:
- **Logs**: `logs/[job-name]_[job-id].out`
- **Errors**: `logs/[job-name]_[job-id].err`
- **Results**: `results/run_[N]/`
- **Models**: Saved checkpoints in results directory

---

## 🐛 Troubleshooting

### Script fails immediately
```bash
# Check permissions
ls -l scripts/*.sh

# Make executable if needed
chmod +x scripts/*.sh
```

### Out of memory
- Use test scripts first (`test_lstm.sh`)
- Reduce batch size in config file
- Request more memory in SLURM header

### Job doesn't start
```bash
# Check SLURM queue
squeue -u $USER

# Check job status
scontrol show job [JOB_ID]
```

### Training is slow
- DistilBERT is 3-5x slower than LSTM (expected)
- Check DASK_NODES setting
- Monitor with `htop` or `nvidia-smi`

---

For more information, see:
- **Configs**: `config/README.md`
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`
- **Roadmap**: `docs/ROADMAP.md`
