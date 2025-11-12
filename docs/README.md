# MLIR-RL Documentation

Welcome to the MLIR-RL documentation! This directory contains comprehensive documentation for the project, organized by topic.

## 📁 Documentation Structure

```
docs/
├── README.md (this file)
├── architecture/          # System architecture & design
├── models/               # Neural network models
├── guides/               # User guides & workflows
└── setup/                # Installation & configuration
```

## 📚 Quick Navigation

### 🏗️ Architecture

**Location**: [`architecture/`](architecture/)

System design, project structure, and refactoring documentation.

- **[PROJECT_STRUCTURE.md](architecture/PROJECT_STRUCTURE.md)** - Overview of codebase organization
- **[ORGANIZATION_SUMMARY.md](architecture/ORGANIZATION_SUMMARY.md)** - High-level project summary
- **[REFACTORING_PHASE1_COMPLETE.md](architecture/REFACTORING_PHASE1_COMPLETE.md)** - Modular architecture refactoring
- **[MODEL_ARCHITECTURE_COMPARISON.py](architecture/MODEL_ARCHITECTURE_COMPARISON.py)** - LSTM vs DistilBERT comparison
- **[refactoring_architecture_diagrams.py](architecture/refactoring_architecture_diagrams.py)** - Visual architecture diagrams

**Start here if you want to**:
- Understand the project structure
- Learn about the modular architecture
- See how models are organized

---

### 🧠 Models

**Location**: [`models/`](models/)

Documentation for neural network models (embeddings, architectures).

- **[DISTILBERT_MODEL.md](models/DISTILBERT_MODEL.md)** - DistilBERT user guide
- **[DISTILBERT_IMPLEMENTATION.md](models/DISTILBERT_IMPLEMENTATION.md)** - Implementation details
- **[DISTILBERT_DATA_INTEGRATION.md](models/DISTILBERT_DATA_INTEGRATION.md)** - Data preprocessing & tokenization
- **[distilbert_quickref.sh](models/distilbert_quickref.sh)** - Quick reference commands

**Available Models**:
- **LSTM** - Original sequential model (fast, 2-3M params)
- **DistilBERT** - Transformer-based (better representations, 66M params)

**Start here if you want to**:
- Use a specific model (LSTM, DistilBERT)
- Add a new model architecture
- Understand model performance trade-offs

---

### 📖 Guides

**Location**: [`guides/`](guides/)

User guides, workflows, and best practices.

- **[SLURM_GUIDE.md](guides/SLURM_GUIDE.md)** - Running on SLURM clusters
- **[PLOTTING_README.md](guides/PLOTTING_README.md)** - Visualization & plotting results
- **[NEPTUNE_AUTO_SYNC.md](guides/NEPTUNE_AUTO_SYNC.md)** - Experiment tracking with Neptune
- **[GITHUB_CHECKLIST.md](guides/GITHUB_CHECKLIST.md)** - Pre-push checklist

**Start here if you want to**:
- Run training on a cluster
- Track experiments
- Visualize results
- Prepare code for GitHub

---

### ⚙️ Setup

**Location**: [`setup/`](setup/)

Installation, configuration, and environment setup.

- **[MLIR_Python_Setup_Steps.md](setup/MLIR_Python_Setup_Steps.md)** - MLIR Python bindings setup
- **[quick_reference.sh](setup/quick_reference.sh)** - Quick setup commands

**Start here if you want to**:
- Set up the development environment
- Install dependencies
- Configure MLIR bindings

---

## 🚀 Getting Started

### New Users

1. **Setup**: Read [`setup/MLIR_Python_Setup_Steps.md`](setup/MLIR_Python_Setup_Steps.md)
2. **Structure**: Read [`architecture/PROJECT_STRUCTURE.md`](architecture/PROJECT_STRUCTURE.md)
3. **Training**: Read [`guides/SLURM_GUIDE.md`](guides/SLURM_GUIDE.md)

### Adding a New Model

1. Read [`architecture/REFACTORING_PHASE1_COMPLETE.md`](architecture/REFACTORING_PHASE1_COMPLETE.md)
2. Review [`models/DISTILBERT_IMPLEMENTATION.md`](models/DISTILBERT_IMPLEMENTATION.md) as an example
3. Follow the modular architecture pattern

### Running Experiments

1. Configure: Edit `config/config.json`
2. Train: `CONFIG_FILE_PATH=config/config.json python bin/train.py`
3. Track: See [`guides/NEPTUNE_AUTO_SYNC.md`](guides/NEPTUNE_AUTO_SYNC.md)
4. Visualize: See [`guides/PLOTTING_README.md`](guides/PLOTTING_README.md)

---

## 📊 Model Comparison

| Model | Parameters | Speed | Memory | Use Case |
|-------|-----------|-------|---------|----------|
| **LSTM** | 2-3M | 1x | 0.5-1 GB | Fast iterations, baseline |
| **DistilBERT** | 66M | 0.2-0.3x | 2-4 GB | Better performance, production |

See [`architecture/MODEL_ARCHITECTURE_COMPARISON.py`](architecture/MODEL_ARCHITECTURE_COMPARISON.py) for detailed comparison.

---

## 🔧 Configuration

Models are selected via config file:

```json
{
  "model_type": "lstm",      // Fast, default
  "model_type": "distilbert" // Better, slower
}
```

See individual model docs for hyperparameter recommendations.

---

## 🐛 Troubleshooting

### Common Issues

**Out of Memory**: Reduce batch sizes
```json
{
  "ppo_batch_size": 16,  // Reduce from 32
  "value_batch_size": 16
}
```

**Slow Training**: Use LSTM for fast iterations
```json
{
  "model_type": "lstm"
}
```

**Environment Issues**: Check [`setup/MLIR_Python_Setup_Steps.md`](setup/MLIR_Python_Setup_Steps.md)

---

## 📝 Contributing

When adding documentation:

1. **Architecture docs** → `architecture/`
2. **Model docs** → `models/`
3. **User guides** → `guides/`
4. **Setup docs** → `setup/`

Update this README.md with links to new documents.

---

## 📞 Support

- **Issues**: Check troubleshooting sections in relevant docs
- **Questions**: See architecture and guide documentation
- **New Features**: Review refactoring docs for best practices

---

## 📜 Document Index

### Architecture
- [PROJECT_STRUCTURE.md](architecture/PROJECT_STRUCTURE.md)
- [ORGANIZATION_SUMMARY.md](architecture/ORGANIZATION_SUMMARY.md)
- [REFACTORING_PHASE1_COMPLETE.md](architecture/REFACTORING_PHASE1_COMPLETE.md)
- [MODEL_ARCHITECTURE_COMPARISON.py](architecture/MODEL_ARCHITECTURE_COMPARISON.py)
- [refactoring_architecture_diagrams.py](architecture/refactoring_architecture_diagrams.py)

### Models
- [DISTILBERT_MODEL.md](models/DISTILBERT_MODEL.md)
- [DISTILBERT_IMPLEMENTATION.md](models/DISTILBERT_IMPLEMENTATION.md)
- [DISTILBERT_DATA_INTEGRATION.md](models/DISTILBERT_DATA_INTEGRATION.md)
- [distilbert_quickref.sh](models/distilbert_quickref.sh)

### Guides
- [SLURM_GUIDE.md](guides/SLURM_GUIDE.md)
- [PLOTTING_README.md](guides/PLOTTING_README.md)
- [NEPTUNE_AUTO_SYNC.md](guides/NEPTUNE_AUTO_SYNC.md)
- [GITHUB_CHECKLIST.md](guides/GITHUB_CHECKLIST.md)

### Setup
- [MLIR_Python_Setup_Steps.md](setup/MLIR_Python_Setup_Steps.md)
- [quick_reference.sh](setup/quick_reference.sh)

---

**Last Updated**: November 13, 2025  
**Documentation Structure**: v2.0 (Organized)
