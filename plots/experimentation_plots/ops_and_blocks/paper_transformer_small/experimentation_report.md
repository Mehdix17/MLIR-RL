# Experimentation Report

**Generated**: 2026-08-14 04:59  
**Dataset**: `ops_and_blocks`  
**Agents**: `paper_transformer_small`  
**Experiment directory**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_small`

---

## 1. Best Checkpoint Summary

| Agent | Best Checkpoint | Peak Geo-Mean Speedup |
|:------|:--------------:|----------------------:|
| `paper_transformer_small` | 18600 | **3.8192×** |

### Per-Agent Detailed Stats (at best checkpoint)

| Agent | Best CP | Valid | Failed | Geo-Mean | Arith-Mean | Best Speedup | Worst Speedup |
|:------|:-------:|------:|-------:|---------:|-----------:|-------------:|---------------:|
| `paper_tf_small` | 18600 | 1497 | 132 | 3.8192× | 180.33× | 42482.97× | 0.0479× |

---

## 2. Model Family Performance (Best Checkpoint)

Geo-mean speedup per model family across all agents.

| Model Family | `paper_transformer_small` |
|:-------------|:------:|
| **Albert** | 9.8228× |
| **Bart** | 8.5208× |
| **Bert** | 9.8455× |
| **Convnext Tiny** | 1.0892× |
| **Distilbert** | 9.3314× |
| **Efficientnet B0** | 1.7306× |
| **Gat** | 0.6913× |
| **Gin** | 1.9624× |
| **Gpt2** | 1.1864× |
| **Llama3 2 1B** | 1948.6683× |
| **Mobilenet V3 Small** | 1.6506× |
| **Resnet50** | 4.0011× |
| **Resnext50** | 3.9463× |
| **T5** | 12.3795× |
| **Vgg16** | 1.6848× |
| **Vit B 16** | 6.6721× |
| **Whisper Base** | 6.1013× |
| **Yolov8M** | 2.3940× |

**Top-3 families by geo-mean speedup (averaged across agents):**

1. `llama3_2_1b` — avg geo-mean **1948.6683×**
2. `t5` — avg geo-mean **12.3795×**
3. `bert` — avg geo-mean **9.8455×**

---

## 3. Operation Type Performance (Best Checkpoint)

Only synthetic operation-type benchmarks (`add`, `conv_2d`, `matmul`, `pooling`, `relu`).

| Operation | `paper_transformer_small` |
|:----------|:------:|
| **Add** | 0.2773× |
| **Conv 2D** | 2.3468× |
| **Matmul** | 2.9576× |
| **Pooling** | 0.4275× |
| **Relu** | 0.1732× |

---

## 4. Top Individual Benchmark Performances

Best individual benchmark speedups from each agent's best checkpoint.

### `paper_tf_small` (checkpoint 18600)

**Top-5 model benchmarks:**

| Rank | Benchmark | Family | Speedup |
|:----:|:----------|:-------|--------:|
| 1 | `llama3_2_1b_block_1079` | llama3_2_1b | 42482.97× |
| 2 | `llama3_2_1b_block_679` | llama3_2_1b | 41518.82× |
| 3 | `llama3_2_1b_block_430` | llama3_2_1b | 41486.81× |
| 4 | `llama3_2_1b_block_546` | llama3_2_1b | 41088.95× |
| 5 | `llama3_2_1b_block_635` | llama3_2_1b | 40796.94× |

**Top-5 operation-type benchmarks:**

| Rank | Benchmark | Op Type | Speedup |
|:----:|:----------|:--------|--------:|
| 1 | `conv_2d_nchw_fchw_256_512_15_15_48_1_1_8_8` | conv_2d | 4.84× |
| 2 | `matmul_512_512_128` | matmul | 4.55× |
| 3 | `conv_2d_nchw_fchw_128_96_28_28_288_1_1_14_14` | conv_2d | 4.43× |
| 4 | `matmul_512_512_256` | matmul | 4.29× |
| 5 | `matmul_256_256_256` | matmul | 4.25× |

---

## 5. Generated Plots

- **Best Checkpoint Benchmark Family Results**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_small/pngs/best_checkpoint_benchmark_family_results.png`
- **Best Checkpoint Benchmark Family Results No Llama3**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_small/pngs/best_checkpoint_benchmark_family_results_no_llama3.png`
- **Best Checkpoint Operation Type Results**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_small/pngs/best_checkpoint_operation_type_results.png`
- **Checkpoint Evolution**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_small/pngs/checkpoint_evolution.png`
