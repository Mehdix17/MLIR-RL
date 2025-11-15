# MLIR-RL Project Roadmap

**Last Updated**: November 15, 2025

---

## 📍 Current Status

### **What's Working**
- ✅ **Two model architectures**: LSTM (original) and DistilBERT (newly integrated)
- ✅ **Training data**: 9,441 MLIR programs in `data/all/`
- ✅ **Training pipeline**: `bin/train.py` trains the RL agent using PPO
- ✅ **Clean code structure**: Modular, extensible, well-documented
- ✅ **Data generation**: Automatic MLIR program generation
- ✅ **Neural network conversion**: PyTorch models → MLIR conversion
- ✅ **Evaluation system**: Comprehensive benchmarking and testing
- ✅ **Modular architecture**: Easy to add new transformer models

### **Project Components**
```
MLIR-RL/
├── bin/                      # Training and evaluation scripts
│   ├── train.py             # Main training pipeline
│   └── evaluate.py          # Model evaluation
├── config/                   # Configuration files
│   ├── config.json          # LSTM baseline config
│   ├── config_distilbert.json  # DistilBERT config
│   └── config_augmented.json   # Augmented data config
├── rl_autoschedular/        # Core RL implementation
│   ├── model.py             # Actor-critic models
│   ├── ppo.py               # PPO algorithm
│   ├── env.py               # MLIR optimization environment
│   └── models/              # Model architectures
├── data_generation/         # Data generation tools
│   ├── random_mlir_gen.py   # Random MLIR generator
│   └── nn_to_mlir.py        # Neural network converter
├── evaluation/              # Evaluation and benchmarking
│   ├── single_op_eval.py    # Single operation evaluator
│   ├── nn_eval.py           # Neural network evaluator
│   └── pytorch_baseline.py  # PyTorch baseline comparison
└── data/                    # Training data
    ├── all/                 # 9,441 MLIR programs
    ├── test/                # Test set
    ├── generated/           # Augmented data
    └── neural_nets/         # Converted neural networks
```

---

## 🎯 Development Phases

### **Phase 1: Validation & Baseline** (1-2 days)
**Goal**: Verify everything works and establish baseline performance

#### Tasks
1. **Train LSTM baseline**
   ```bash
   CONFIG_FILE_PATH=config/config.json python bin/train.py
   ```

2. **Train DistilBERT model**
   ```bash
   CONFIG_FILE_PATH=config/config_distilbert.json python bin/train.py
   ```

3. **Evaluate and compare**
   ```bash
   python bin/evaluate.py --model-type lstm
   python bin/evaluate.py --model-type distilbert
   ```

#### Expected Outcomes
- Training runs without errors
- Baseline performance metrics established
- DistilBERT vs LSTM comparison data
- Understanding of which model architecture works best

#### Success Metrics
- [ ] LSTM trains successfully
- [ ] DistilBERT trains successfully
- [ ] Performance comparison completed
- [ ] Results saved in `results/` directory

---

### **Phase 2: Expand Model Support** (3-5 days)
**Goal**: Add support for multiple transformer architectures

#### Target Models
- ✅ LSTM (complete)
- ✅ DistilBERT (complete)
- ⏳ BERT (full version)
- ⏳ GPT-2
- ⏳ ConvNext
- ⏳ Llama (or other large models)

#### Implementation Pattern (per model)
1. **Create embedding module**
   ```python
   # rl_autoschedular/models/embeddings/gpt2_embedding.py
   class GPT2Embedding(nn.Module):
       def __init__(self, config):
           # Initialize GPT-2 based embedding
           pass
   ```

2. **Register in factory**
   ```python
   # rl_autoschedular/models/embeddings/factory.py
   EMBEDDING_REGISTRY = {
       'lstm': LSTMEmbedding,
       'distilbert': DistilBERTEmbedding,
       'gpt2': GPT2Embedding,  # Add new model
   }
   ```

3. **Create config file**
   ```json
   // config/config_gpt2.json
   {
     "model_type": "gpt2",
     "model_config": {
       "hidden_size": 768,
       "num_hidden_layers": 12,
       ...
     }
   }
   ```

4. **Test and benchmark**
   ```bash
   CONFIG_FILE_PATH=config/config_gpt2.json python bin/train.py
   ```

#### Success Metrics
- [ ] GPT-2 embedding implemented
- [ ] BERT embedding implemented
- [ ] ConvNext embedding implemented
- [ ] All models train successfully
- [ ] Comparative analysis completed

---

### **Phase 3: Data Augmentation** (2-3 days)
**Goal**: Scale up training data for better generalization

#### Tasks
1. **Generate diverse MLIR programs**
   ```bash
   # Generate 1000 additional samples
   python scripts/augment_dataset.py --num-samples 1000
   
   # Generate specific operation types
   python scripts/augment_dataset.py \
     --num-samples 2000 \
     --operations matmul,conv2d,pooling,add,transpose
   ```

2. **Train with augmented data**
   ```bash
   CONFIG_FILE_PATH=config/config_augmented.json python bin/train.py
   ```

3. **Validate improvements**
   - Compare performance on original vs augmented data
   - Test generalization to unseen programs
   - Measure optimization success rate

#### Expected Outcomes
- More diverse training dataset (10,000+ programs)
- Better generalization to unseen MLIR code
- 10-20% improvement in optimization success rate

#### Success Metrics
- [ ] 2000+ augmented samples generated
- [ ] Training converges on augmented data
- [ ] Improved performance on test set
- [ ] Better handling of edge cases

---

### **Phase 4: Real Neural Network Benchmarks** (3-5 days)
**Goal**: Test RL agent on actual production neural networks

#### Target Networks
- ResNet-18, ResNet-50
- MobileNetV2
- EfficientNet
- DistilBERT, BERT-base
- GPT-2 (small)

#### Workflow
1. **Convert PyTorch models to MLIR**
   ```bash
   python data_generation/nn_to_mlir.py --model resnet18
   python data_generation/nn_to_mlir.py --model mobilenetv2
   python data_generation/nn_to_mlir.py --model distilbert
   ```

2. **Optimize with RL agent**
   ```bash
   python bin/train.py \
     --benchmark data/neural_nets/resnet18.mlir \
     --model-type distilbert
   ```

3. **Compare with baselines**
   ```bash
   # PyTorch baseline
   python evaluation/pytorch_baseline.py --model resnet18
   
   # Neural network evaluator
   python evaluation/nn_eval.py \
     --model resnet18 \
     --rl-optimized data/neural_nets/resnet18_optimized.mlir
   ```

#### Expected Outcomes
- RL agent optimizes real networks 10-30% better than default compiler
- Competitive with or better than TVM AutoScheduler
- Demonstrates practical applicability

#### Success Metrics
- [ ] 5+ neural networks converted to MLIR
- [ ] RL agent optimizes all networks
- [ ] 15%+ average speedup vs baseline
- [ ] Competitive with state-of-the-art autotuners

---

### **Phase 5: Research & Publication** (Ongoing)

#### Comparison Targets
Compare MLIR-RL approach with:
- **TVM AutoScheduler**: Apache TVM's automatic scheduler
- **Ansor**: Learning-based tensor program optimizer
- **FlexTensor**: Flexible tensor program optimization
- **Google AutoTune**: ML-based compiler optimization
- **MLIR's built-in optimizations**: Default MLIR passes

#### Evaluation Metrics
- **Execution time**: Speedup vs baseline
- **Memory usage**: Peak memory consumption
- **Compilation time**: Time to optimize
- **Generalization**: Performance on unseen programs
- **Scalability**: Performance on large models

#### Publication Strategy
1. **Technical Report**
   - Architecture overview
   - Experimental results
   - Comparative analysis

2. **Conference Paper**
   - Target: MLSys, ICML, NeurIPS, or PLDI
   - Novel contributions: RL for MLIR optimization
   - Strong experimental validation

3. **Open Source Release**
   - Clean, documented codebase
   - Reproducible experiments
   - Community engagement

#### Success Metrics
- [ ] Comprehensive benchmark suite completed
- [ ] 20%+ average improvement over baselines
- [ ] Technical report written
- [ ] Paper submitted to top-tier conference
- [ ] Open source release published

---

## 🚀 Quick Start Guide

### **Immediate Next Steps**

#### Option A: Quick Win (Recommended)
```bash
# 1. Train LSTM baseline
CONFIG_FILE_PATH=config/config.json python bin/train.py

# 2. Train DistilBERT
CONFIG_FILE_PATH=config/config_distilbert.json python bin/train.py

# 3. Compare results (look at results/* directory)
ls -lh results/
```

**Why this first**: Validates setup, gives baseline, shows if DistilBERT improves performance

#### Option B: Add More Models
Focus on implementing next transformer architecture (GPT-2, BERT, ConvNext)

#### Option C: Scale Up Data
```bash
# Generate diverse training data
python scripts/augment_dataset.py \
  --num-samples 2000 \
  --operations matmul,conv2d,pooling,add,transpose
```

**Why this**: More data → better generalization → higher success rate

---

## 📊 Success Criteria

### **Project Success Indicators**
1. ✅ RL agent optimizes MLIR code **20%+ faster** than baseline compiler
2. ✅ Works on **real neural networks** (ResNet, BERT, GPT-2, etc.)
3. ✅ **Multiple model architectures** perform competitively
4. ✅ **Published results** showing novel approach
5. ✅ **Open source** with community adoption

### **Technical Milestones**
- [ ] **Baseline established**: LSTM and DistilBERT trained
- [ ] **Multi-model support**: 3+ transformer architectures
- [ ] **Data scaling**: 10,000+ training programs
- [ ] **Real-world validation**: 5+ production networks optimized
- [ ] **Benchmark suite**: Comprehensive comparison with SOTA
- [ ] **Publication ready**: Technical report + paper draft

---

## 🔧 Development Commands

### **Training**
```bash
# LSTM baseline
CONFIG_FILE_PATH=config/config.json python bin/train.py

# DistilBERT
CONFIG_FILE_PATH=config/config_distilbert.json python bin/train.py

# With augmented data
CONFIG_FILE_PATH=config/config_augmented.json python bin/train.py
```

### **Evaluation**
```bash
# Evaluate specific model
python bin/evaluate.py --model-type lstm
python bin/evaluate.py --model-type distilbert

# Benchmark on neural network
python evaluation/nn_eval.py --model resnet18
```

### **Data Generation**
```bash
# Generate augmented data
python scripts/augment_dataset.py --num-samples 1000

# Convert neural network to MLIR
python data_generation/nn_to_mlir.py --model resnet18

# View data statistics
bash scripts/data_quickref.sh
```

### **Testing**
```bash
# Run all tests
python -m pytest tests/

# Run integration tests
python tests/test_integration.py
```

---

## 📚 Documentation

### **Key Documents**
- **README.md**: Project overview and setup
- **docs/PROJECT_STRUCTURE.md**: Code organization
- **docs/guides/DATA_ORGANIZATION_COMPLETE.md**: Data structure
- **docs/guides/INTEGRATION_COMPLETE.md**: Integration details
- **scripts/README.md**: Script usage
- **tests/README.md**: Testing guide

### **Quick References**
- **Data commands**: `bash scripts/data_quickref.sh`
- **Config examples**: `config/` directory
- **Model registry**: `rl_autoschedular/models/embeddings/factory.py`

---

## 🎯 Timeline Estimate

| Phase | Duration | Priority |
|-------|----------|----------|
| Phase 1: Validation | 1-2 days | 🔴 Critical |
| Phase 2: Model Expansion | 3-5 days | 🟡 High |
| Phase 3: Data Augmentation | 2-3 days | 🟡 High |
| Phase 4: Real Benchmarks | 3-5 days | 🟢 Medium |
| Phase 5: Publication | Ongoing | 🟢 Medium |

**Total Estimated Time**: 2-3 weeks for core development + ongoing research

---

## 📞 Next Actions

**Choose your path:**
1. **Validate first** → Run baseline experiments (Phase 1)
2. **Add models** → Implement GPT-2/BERT/ConvNext (Phase 2)
3. **Generate data** → Create diverse datasets (Phase 3)
4. **Benchmark** → Test on real neural networks (Phase 4)

**Decision factors:**
- **Timeline**: Tight deadline? → Start with Phase 1
- **Research focus**: Novel architectures? → Focus on Phase 2
- **Performance**: Need better results? → Start with Phase 3
- **Practical impact**: Production systems? → Jump to Phase 4

---

## 🔄 Maintenance & Updates

This roadmap will be updated as:
- New model architectures are added
- Experimental results become available
- Research direction evolves
- Community feedback is incorporated

**Update schedule**: After completing each phase

---

*For questions or suggestions, refer to project documentation or raise an issue on GitHub.*
