# MLIR-RL Project Structure

This document describes the organized structure of the MLIR-RL project.

## 📁 Directory Structure

```
MLIR-RL/
├── bin/                    # Main executable scripts
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation script
│
├── rl_autoschedular/      # Core RL implementation
│   ├── actions/           # Action space definitions
│   ├── benchmarks.py      # Benchmark management
│   ├── env.py             # RL environment
│   ├── execution.py       # Code execution and measurement
│   ├── model.py           # Neural network models
│   ├── observation.py     # State observation
│   ├── ppo.py             # PPO algorithm implementation
│   ├── state.py           # State representation
│   ├── trajectory.py      # Trajectory data structures
│   └── transforms.py      # MLIR transformations
│
├── utils/                 # Utility modules
│   ├── config.py          # Configuration management
│   ├── dask_manager.py    # Dask cluster management
│   ├── file_logger.py     # File logging utilities
│   ├── log.py             # Logging helpers
│   └── singleton.py       # Singleton pattern
│
├── utils/              # Analysis and plotting
│   ├── plot_results.py    # Generate comparison plots
│   └── filelog_clean.py   # Log file cleaning
│
├── utils/           # Experiment utilities
│   ├── neptune_sync.py              # Neptune continuous sync
│   ├── sync_neptune_with_plots.py   # Neptune sync with plots
│   ├── test_neptune.py              # Neptune connection test
│   ├── gen.py                       # Benchmark generation
│   ├── get_base.py                  # Baseline extraction
│   └── fill_db.py                   # Database filling
│
├── notebooks/             # Jupyter notebooks
│   ├── demo.ipynb         # Demo notebook
│   └── demo.py            # Demo script
│
├── docs/                  # Documentation
│   ├── README.md          # Main README (symlink)
│   ├── PLOTTING_README.md # Plotting documentation
│   ├── MLIR_Python_Setup_Steps.md  # Setup guide
│   ├── quick_reference.sh # Quick reference commands
│   └── PROJECT_STRUCTURE.md        # This file
│
├── scripts/               # SLURM job scripts
│   ├── train.sh           # Training job
│   ├── eval.sh            # Evaluation job
│   └── neptune-sync.sh    # Neptune sync job
│
├── config/                # Configuration files
│   ├── config.json        # Main configuration
│   └── example.json       # Example configuration
│
├── data/                  # Data directory
│   ├── all/               # Full dataset
│   ├── test/              # Test dataset
│   ├── debug/             # Debug data
│   ├── features/          # Feature data
│   ├── multi/             # Multi-benchmark data
│   ├── nn/                # Neural network data
│   ├── polybench/         # Polybench benchmarks
│   └── ...
│
├── results/               # Training results
│   ├── run_0/             # Run directories
│   ├── run_1/
│   └── ...
│
├── logs/                  # Log files
│   ├── neptune/           # Neptune sync logs
│   └── *.debug            # Debug logs
│
├── tools/                 # External tools
│   ├── ast_dumper/        # AST dumper tool
│   ├── pre_vec/           # Pre-vectorization tool
│   └── vectorizer/        # Vectorizer tool
│
├── llvm-project/          # LLVM/MLIR source and build
│   ├── build/             # Build directory
│   ├── mlir/              # MLIR source
│   └── ...
│
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
└── .gitignore            # Git ignore rules
```

## 🚀 Quick Start

### Training
```bash
bash scripts/train.sh
```

### Evaluation
```bash
bash scripts/eval.sh
```

### Generate Plots
```bash
python utils/plot_results.py results/run_X
```

### Sync to Neptune
```bash
# Continuous sync during training
bash scripts/neptune-sync.sh

# One-time sync with plots
python utils/sync_neptune_with_plots.py results/run_X
```

## 📊 Workflow

1. **Configure**: Edit `config/config.json`
2. **Train**: Run `bash scripts/train.sh`
3. **Monitor**: (Optional) Run `bash scripts/neptune-sync.sh` in another terminal
4. **Analyze**: Generate plots with `python utils/plot_results.py results/run_X`
5. **Share**: Sync to Neptune with `python utils/sync_neptune_with_plots.py results/run_X`

## 📚 Documentation

- **Main README**: `README.md`
- **Setup Guide**: `docs/MLIR_Python_Setup_Steps.md`
- **Plotting Guide**: `docs/PLOTTING_README.md`
- **Quick Reference**: `bash docs/quick_reference.sh`

## 🔧 File Purposes

### Main Executables (`bin/`)
- **train.py**: Main training loop with PPO
- **evaluate.py**: Evaluate trained models

### Analysis (`utils/`)
- **plot_results.py**: Generate comparison plots by operation type
- **filelog_clean.py**: Clean up log files

### Experiments (`utils/`)
- **neptune_sync.py**: Continuous Neptune synchronization
- **sync_neptune_with_plots.py**: One-time sync with plot generation
- **test_neptune.py**: Test Neptune connection
- **gen.py**: Generate synthetic benchmarks
- **get_base.py**: Extract baseline performance
- **fill_db.py**: Populate execution database

### Notebooks (`notebooks/`)
- **demo.ipynb**: Interactive demonstration
- **demo.py**: Python demo script

## 🎯 Key Directories

- **`rl_autoschedular/`**: Core RL implementation (don't move)
- **`utils/`**: Utility modules (don't move)
- **`bin/`**: Main entry points
- **`utils/`**: Post-training analysis
- **`utils/`**: Research utilities
- **`docs/`**: All documentation in one place
- **`notebooks/`**: Interactive exploration

## 📝 Notes

- All Python scripts are now organized by purpose
- Documentation is centralized in `docs/`
- Main executables are in `bin/` for clarity
- Script paths have been updated in SLURM job files
- Run `bash docs/quick_reference.sh` for command reference
