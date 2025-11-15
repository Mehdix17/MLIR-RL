# 🚀 Training Quick Start

## ✅ **All Scripts Ready!**

### **Available Training Scripts**

```
scripts/
├── lstm/
│   ├── test_lstm.sh              ⚡ Quick test (15 min)
│   ├── train_lstm_baseline.sh    🎯 Baseline (1 hour)
│   └── train_lstm_augmented.sh   📊 Extended (12 hours)
├── distilbert/
│   ├── test_distilbert.sh        ⚡ Quick test (20 min)
│   └── train_distilbert.sh       🤖 Transformer (3 hours)
├── utils/
│   ├── augment_dataset.py
│   ├── organize_data.py
│   └── data_quickref.sh
└── run_training.sh               🎮 Interactive launcher
```

---

## 🎮 **Easy Way: Use the Launcher**

```bash
# Interactive menu
bash scripts/run_training.sh

# Or directly specify
bash scripts/run_training.sh 1              # Test LSTM
bash scripts/run_training.sh test-lstm      # Test LSTM
bash scripts/run_training.sh distilbert     # Train DistilBERT
```

---

## ⚡ **Quick Start: Test First!**

### **1. Test LSTM** (recommended first step)
```bash
sbatch scripts/lstm/test_lstm.sh

# Check status
squeue -u $USER

# Monitor logs
tail -f logs/test-lstm_*.out
```

**What it does:**
- Uses 17 files from `data/test`
- Runs 3 iterations
- Takes ~15 minutes
- Verifies your setup works

### **2. Test DistilBERT** (if LSTM passed)
```bash
sbatch scripts/distilbert/test_distilbert.sh

# Monitor
tail -f logs/test-distilbert_*.out
```

**What it does:**
- Uses 17 files from `data/test`
- Runs 3 iterations
- Takes ~20 minutes
- Verifies transformer works

---

## 🎯 **Full Training: Choose Your Model**

### **Option A: LSTM Baseline** (fastest)
```bash
sbatch scripts/lstm/train_lstm_baseline.sh
```
- ⏱️ **Time**: ~1 hour
- 📁 **Data**: 9,441 files
- 🎯 **Use**: Baseline comparison

### **Option B: LSTM Augmented** (best results)
```bash
sbatch scripts/lstm/train_lstm_augmented.sh
```
- ⏱️ **Time**: ~12 hours
- 📁 **Data**: 9,941 files (includes augmentation)
- 🎯 **Use**: Best LSTM performance

### **Option C: DistilBERT** (transformer)
```bash
sbatch scripts/distilbert/train_distilbert.sh
```
- ⏱️ **Time**: ~3 hours (slower but powerful)
- 📁 **Data**: 9,441 files
- 🎯 **Use**: Transformer approach

---

## 📊 **Data Available**

| Dataset | Location | Files | Description |
|---------|----------|-------|-------------|
| Test | `data/test` | 17 | Quick validation |
| Training | `data/all` | 9,441 | Main dataset |
| Augmented | `data/generated` | 500 | Extra diversity |
| **Total** | - | **9,958** | All available |

---

## 📋 **Corresponding Configs**

Each script uses a specific config file:

| Script | Config File |
|--------|-------------|
| `lstm/test_lstm.sh` | `config/test.json` |
| `distilbert/test_distilbert.sh` | `config/test_distilbert.json` |
| `lstm/train_lstm_baseline.sh` | `config/config.json` |
| `lstm/train_lstm_augmented.sh` | `config/config_augmented.json` |
| `distilbert/train_distilbert.sh` | `config/config_distilbert.json` |

---

## 📈 **Monitor Your Training**

### **Check Job Status**
```bash
squeue -u $USER
```

### **View Logs**
```bash
# Live monitoring
tail -f logs/train-lstm-baseline_*.out

# View all logs
ls -lh logs/

# Check errors
cat logs/train-lstm-baseline_*.err
```

### **Check Results**
```bash
# List LSTM results
ls -lh results/lstm/

# List DistilBERT results
ls -lh results/distilbert/

# View saved models
ls -lh results/lstm/run_*/models/
ls -lh results/distilbert/run_*/models/
```

---

## 🎯 **Recommended Order**

1. ✅ **Test LSTM** → Verify setup (15 min)
2. ✅ **Test DistilBERT** → Verify transformer (20 min)
3. 🎯 **Train baseline** → Get baseline metrics (1 hour)
4. 📈 **Evaluate models** → Measure performance (10-20 min)
5. 📊 **Compare models** → Analyze results
6. 🚀 **Full training** → Best model + augmentation

---

## 📈 **Evaluation**

After training, evaluate your models:

### **Evaluate LSTM**
```bash
# Evaluate latest run
sbatch scripts/lstm/eval_lstm.sh

# Evaluate specific run
export EVAL_DIR=results/lstm/run_0
sbatch scripts/lstm/eval_lstm.sh
```

### **Evaluate DistilBERT**
```bash
# Evaluate latest run
sbatch scripts/distilbert/eval_distilbert.sh

# Evaluate specific run
export EVAL_DIR=results/distilbert/run_0
sbatch scripts/distilbert/eval_distilbert.sh
```

### **View Evaluation Results**
```bash
# Check evaluation logs
cat results/lstm/run_0/logs/eval/average_speedup
cat results/distilbert/run_0/logs/eval/average_speedup
```

---

## 🛠️ **Troubleshooting**

### **Script won't run**
```bash
chmod +x scripts/*.sh
```

### **Job pending forever**
```bash
scontrol show job [JOB_ID]
```

### **Out of memory**
- Start with test scripts
- Reduce batch size in config
- Request more memory in script

### **Training fails**
```bash
# Check error logs
cat logs/[job-name]_*.err

# Validate config
python tests/test_config_loading.py
```

---

## 📚 **Documentation**

- **Scripts Guide**: `scripts/TRAINING_GUIDE.md`
- **Results Guide**: `results/README.md`
- **Config Guide**: `config/README.md`
- **Project Roadmap**: `docs/ROADMAP.md`
- **Config Update**: `docs/CONFIG_UPDATE_SUMMARY.md`

---

## 🎉 **You're Ready!**

Start with:
```bash
bash scripts/run_training.sh
```

Then choose option **1** to test LSTM!
