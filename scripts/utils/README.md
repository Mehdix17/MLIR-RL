# Utility Scripts

Data management and utility scripts for MLIR-RL.

## 📋 Available Scripts

### `augment_dataset.py` 📊 Data Generation
Generate synthetic MLIR programs for training
```bash
python scripts/utils/augment_dataset.py --num-samples 1000
```

**What it does:**
- Analyzes existing data format
- Generates new MLIR files (add, matmul, conv2d)
- Creates execution_times JSON
- Saves to `data/generated/code_files/`

**Options:**
```bash
python scripts/utils/augment_dataset.py \
  --num-samples 2000 \
  --operations matmul,conv2d,pooling
```

### `organize_data.py` 🗂️ Data Organization
Organize data folder structure
```bash
python scripts/utils/organize_data.py
```

**What it does:**
- Creates subdirectories (generated, neural_nets, benchmarks)
- Moves test files
- Creates .gitignore files
- Generates README files

### `data_quickref.sh` 📚 Quick Reference
Display data statistics and common commands
```bash
bash scripts/utils/data_quickref.sh
```

**What it shows:**
- Current data statistics
- File counts per directory
- Common training commands
- Data generation examples

## 🚀 Common Workflows

### Generate More Training Data
```bash
cd /scratch/mb10856/MLIR-RL
python scripts/utils/augment_dataset.py --num-samples 1000
```

### Check Data Statistics
```bash
bash scripts/utils/data_quickref.sh
```

### Organize Data Folder
```bash
python scripts/utils/organize_data.py
```

## 📂 Data Structure

```
data/
├── all/              # 9,441 original files
├── test/             # 17 test files
├── generated/        # Augmented data
├── neural_nets/      # Converted neural networks
└── benchmarks/       # Benchmark results
```
