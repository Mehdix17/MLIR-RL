# Scripts

This directory contains utility scripts for the MLIR-RL project.

## Data Management Scripts

### `organize_data.py`
Organizes the data folder structure, creating subdirectories for generated data, neural networks, and benchmarks.

**Usage:**
```bash
python scripts/organize_data.py
```

**What it does:**
- Creates data/generated/, data/neural_nets/, data/benchmarks/
- Moves test files to appropriate locations
- Creates .gitignore files
- Generates README files

---

### `augment_dataset.py`
Generates additional MLIR files matching the format of existing data in `data/all/`.

**Usage:**
```bash
python scripts/augment_dataset.py
```

**What it does:**
- Analyzes existing data/all/code_files/ format
- Generates 500 new MLIR files (add, matmul, conv2d)
- Creates execution_times_generated.json
- Saves to data/generated/code_files/

---

### `data_quickref.sh`
Quick reference guide showing all data-related commands and current statistics.

**Usage:**
```bash
bash scripts/data_quickref.sh
```

**What it shows:**
- Current data statistics
- Common commands for data generation
- Training commands
- Evaluation commands

---

## Training Scripts

### `train.sh`
SLURM job submission script for training on clusters.

**Usage:**
```bash
sbatch scripts/train.sh
```

### `eval.sh`
SLURM job submission script for evaluation.

**Usage:**
```bash
sbatch scripts/eval.sh
```

### `neptune-sync.sh`
Syncs results to Neptune.ai for experiment tracking.

**Usage:**
```bash
bash scripts/neptune-sync.sh
```

---

## Quick Reference

```bash
# Organize data structure
python scripts/organize_data.py

# Generate augmentation data
python scripts/augment_dataset.py

# View data commands
bash scripts/data_quickref.sh

# Submit training job
sbatch scripts/train.sh

# Submit evaluation job
sbatch scripts/eval.sh
```
