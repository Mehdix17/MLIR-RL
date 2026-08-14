# Experimentation Report

**Generated**: 2026-08-14 04:59  
**Dataset**: `ops_and_blocks`  
**Agents**: `paper_original`  
**Experiment directory**: `plots/experimentation_plots/ops_and_blocks/paper_original`

---

## 1. Best Checkpoint Summary

| Agent | Best Checkpoint | Peak Geo-Mean Speedup |
|:------|:--------------:|----------------------:|
| `paper_original` | 12200 | **3.2174×** |

### Per-Agent Detailed Stats (at best checkpoint)

| Agent | Best CP | Valid | Failed | Geo-Mean | Arith-Mean | Best Speedup | Worst Speedup |
|:------|:-------:|------:|-------:|---------:|-----------:|-------------:|---------------:|
| `paper_original` | 12200 | 1499 | 130 | 3.2174× | 103.59× | 28249.28× | 0.0448× |

---

## 2. Model Family Performance (Best Checkpoint)

Geo-mean speedup per model family across all agents.

| Model Family | `paper_original` |
|:-------------|:------:|
| **Albert** | 8.7573× |
| **Bart** | 7.5479× |
| **Bert** | 8.2127× |
| **Convnext Tiny** | 1.1098× |
| **Distilbert** | 8.0458× |
| **Efficientnet B0** | 1.5266× |
| **Gat** | 0.3799× |
| **Gin** | 1.6055× |
| **Gpt2** | 1.2415× |
| **Llama3 2 1B** | 1048.6873× |
| **Mobilenet V3 Small** | 1.4332× |
| **Resnet50** | 3.6576× |
| **Resnext50** | 3.7365× |
| **T5** | 11.0637× |
| **Vgg16** | 1.0467× |
| **Vit B 16** | 5.6514× |
| **Whisper Base** | 5.8571× |
| **Yolov8M** | 2.8509× |

**Top-3 families by geo-mean speedup (averaged across agents):**

1. `llama3_2_1b` — avg geo-mean **1048.6873×**
2. `t5` — avg geo-mean **11.0637×**
3. `albert` — avg geo-mean **8.7573×**

---

## 3. Operation Type Performance (Best Checkpoint)

Only synthetic operation-type benchmarks (`add`, `conv_2d`, `matmul`, `pooling`, `relu`).

| Operation | `paper_original` |
|:----------|:------:|
| **Add** | 0.2521× |
| **Conv 2D** | 1.5617× |
| **Matmul** | 0.7123× |
| **Pooling** | 0.2380× |
| **Relu** | 0.2118× |

---

## 4. Top Individual Benchmark Performances

Best individual benchmark speedups from each agent's best checkpoint.

### `paper_original` (checkpoint 12200)

**Top-5 model benchmarks:**

| Rank | Benchmark | Family | Speedup |
|:----:|:----------|:-------|--------:|
| 1 | `llama3_2_1b_block_635` | llama3_2_1b | 28249.28× |
| 2 | `llama3_2_1b_block_546` | llama3_2_1b | 27577.11× |
| 3 | `llama3_2_1b_block_430` | llama3_2_1b | 25754.41× |
| 4 | `llama3_2_1b_block_1080` | llama3_2_1b | 24661.08× |
| 5 | `llama3_2_1b_block_679` | llama3_2_1b | 14821.79× |

**Top-5 operation-type benchmarks:**

| Rank | Benchmark | Op Type | Speedup |
|:----:|:----------|:--------|--------:|
| 1 | `conv_2d_nchw_fchw_128_128_7_7_192_1_1_7_7` | conv_2d | 3.22× |
| 2 | `matmul_512_512_128` | matmul | 2.92× |
| 3 | `conv_2d_nchw_fchw_128_256_15_15_64_1_1_15_15` | conv_2d | 2.69× |
| 4 | `conv_2d_nchw_fchw_128_240_7_7_192_1_1_7_7` | conv_2d | 2.58× |
| 5 | `conv_2d_nchw_fchw_256_48_56_56_96_1_1_28_28` | conv_2d | 2.57× |

---

## 5. Generated Plots

- **Best Checkpoint Benchmark Family Results**: `plots/experimentation_plots/ops_and_blocks/paper_original/pngs/best_checkpoint_benchmark_family_results.png`
- **Best Checkpoint Benchmark Family Results No Llama3**: `plots/experimentation_plots/ops_and_blocks/paper_original/pngs/best_checkpoint_benchmark_family_results_no_llama3.png`
- **Best Checkpoint Operation Type Results**: `plots/experimentation_plots/ops_and_blocks/paper_original/pngs/best_checkpoint_operation_type_results.png`
- **Checkpoint Evolution**: `plots/experimentation_plots/ops_and_blocks/paper_original/pngs/checkpoint_evolution.png`
