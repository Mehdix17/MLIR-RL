# Scripts Directory

Training and utility scripts for the MLIR-RL project, organized by model type.

---

## 📁 Directory Structure

```
scripts/
├── lstm/                    # LSTM training scripts
│   ├── test_lstm.sh
│   ├── train_lstm_baseline.sh
│   ├── train_lstm_augmented.sh
│   ├── eval_lstm.sh
│   └── README.md
├── distilbert/              # DistilBERT training scripts
│   ├── test_distilbert.sh
│   ├── train_distilbert.sh
│   ├── eval_distilbert.sh
│   └── README.md
├── comparison/              # Comparison framework
│   ├── compare_all.sh
│   ├── test_comparison.sh
│   └── README.md
├── utils/                   # Utility scripts
│   ├── augment_dataset.py
│   ├── organize_data.py
│   ├── data_quickref.sh
│   └── README.md
├── run_training.sh          # Interactive launcher
├── train.sh                 # Legacy generic trainer
├── eval.sh                  # Model evaluation
└── neptune-sync.sh          # Neptune sync
```

---

## 🚀 Quick Start

### **Easy Way: Interactive Launcher**

```bash
bash scripts/run_training.sh
```

Choose from menu:
1. Test LSTM (15 min)
2. Test DistilBERT (20 min)
3. Train LSTM baseline (1 hour)
4. Train LSTM augmented (12 hours)
5. Train DistilBERT (3 hours)

---

## 📋 Model-Specific Scripts

### **LSTM Scripts** → `lstm/`

| Script | Time | Data | Purpose |
|--------|------|------|---------|
| test_lstm.sh | 15 min | 17 files | Quick test |
| train_lstm_baseline.sh | 1 hour | 9,441 files | Baseline |
| train_lstm_augmented.sh | 12 hours | 9,941 files | Best results |

```bash
# Quick test
sbatch scripts/lstm/test_lstm.sh

# Full training
sbatch scripts/lstm/train_lstm_baseline.sh
```

See `lstm/README.md` for details.

---

### **DistilBERT Scripts** → `distilbert/`

| Script | Time | Data | Purpose |
|--------|------|------|---------|
| test_distilbert.sh | 20 min | 17 files | Quick test |
| train_distilbert.sh | 3 hours | 9,441 files | Full training |

```bash
# Quick test
sbatch scripts/distilbert/test_distilbert.sh

# Full training
sbatch scripts/distilbert/train_distilbert.sh
```

See `distilbert/README.md` for details.

---

### **Utility Scripts** → `utils/`

| Script | Purpose |
|--------|---------|
| augment_dataset.py | Generate synthetic MLIR data |
| organize_data.py | Organize data folder structure |
| data_quickref.sh | Data statistics and commands |

```bash
# Generate more data
python scripts/utils/augment_dataset.py --num-samples 1000

# Check data stats
bash scripts/utils/data_quickref.sh
```

See `utils/README.md` for details.
