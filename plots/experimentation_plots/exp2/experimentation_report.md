# Experimentation Report

**Generated**: 2026-07-12 14:14  
**Dataset**: `ops_and_blocks`  
**Agents**: `paper_original`, `paper_transformer_small`, `paper_transformer_large`  
**Experiment directory**: `plots/experimentation_plots/exp2`

---

## 1. Best Checkpoint Summary

| Agent | Best Checkpoint | Peak Geo-Mean Speedup |
|:------|:--------------:|----------------------:|
| `paper_tf_small` | 7900 | **3.2503×** |
| `paper_tf_large` | 1200 | **2.7651×** |
| `paper_original` | 6100 | **2.6843×** |

### Per-Agent Detailed Stats (at best checkpoint)

| Agent | Best CP | Valid | Failed | Geo-Mean | Arith-Mean | Best Speedup | Worst Speedup |
|:------|:-------:|------:|-------:|---------:|-----------:|-------------:|---------------:|
| `paper_original` | 6100 | 1552 | 135 | 2.6843× | 109.37× | 28647.78× | 0.0105× |
| `paper_tf_small` | 7900 | 1555 | 132 | 3.2503× | 123.35× | 28521.73× | 0.0489× |
| `paper_tf_large` | 1200 | 1559 | 128 | 2.7651× | 121.59× | 28668.59× | 0.0162× |

---

## 2. Model Family Performance (Best Checkpoint)

Geo-mean speedup per model family across all agents.

| Model Family | `paper_original` | `paper_tf_large` | `paper_tf_small` |
|:-------------|:------:|:------:|:------:|
| **Albert** | 6.9392× | 8.0256× | 9.0101× |
| **Bart** | 5.6200× | 6.3429× | 7.2564× |
| **Bert** | 7.0499× | 7.5646× | 8.4713× |
| **Convnext Tiny** | 0.7669× | 0.9267× | 0.8790× |
| **Distilbert** | 5.9247× | 6.8034× | 7.6958× |
| **Efficientnet B0** | 1.2376× | 1.1959× | 1.4017× |
| **Gat** | 1.7000× | 1.3082× | 0.9073× |
| **Gin** | 7.5066× | 1.8772× | 1.4106× |
| **Gpt2** | 1.1856× | 1.3192× | 1.3730× |
| **Llama3 2 1B** | 196.4559× | 218.9669× | 238.2355× |
| **Mobilenet V3 Small** | 1.3742× | 1.2883× | 1.3581× |
| **Resnet50** | 1.1042× | 1.4573× | 1.8450× |
| **Resnext50** | 1.2372× | 1.4083× | 1.8686× |
| **T5** | 6.9947× | 8.8480× | 10.0422× |
| **Vgg16** | 1.6235× | 0.6324× | 0.8828× |
| **Vit B 16** | 3.0809× | 3.7499× | 5.5300× |
| **Whisper Base** | 4.6777× | 4.5106× | 5.1276× |
| **Yolov8M** | 1.1979× | 0.4114× | 1.4271× |

**Top-3 families by geo-mean speedup (averaged across agents):**

1. `llama3_2_1b` — avg geo-mean **217.8861×**
2. `t5` — avg geo-mean **8.6283×**
3. `albert` — avg geo-mean **7.9916×**

---

## 3. Model Family Performance — Excluding LLaMA

| Model Family | `paper_original` | `paper_tf_large` | `paper_tf_small` |
|:-------------|:------:|:------:|:------:|
| **Albert** | 6.9392× | 8.0256× | 9.0101× |
| **Bart** | 5.6200× | 6.3429× | 7.2564× |
| **Bert** | 7.0499× | 7.5646× | 8.4713× |
| **Convnext Tiny** | 0.7669× | 0.9267× | 0.8790× |
| **Distilbert** | 5.9247× | 6.8034× | 7.6958× |
| **Efficientnet B0** | 1.2376× | 1.1959× | 1.4017× |
| **Gat** | 1.7000× | 1.3082× | 0.9073× |
| **Gin** | 7.5066× | 1.8772× | 1.4106× |
| **Gpt2** | 1.1856× | 1.3192× | 1.3730× |
| **Mobilenet V3 Small** | 1.3742× | 1.2883× | 1.3581× |
| **Resnet50** | 1.1042× | 1.4573× | 1.8450× |
| **Resnext50** | 1.2372× | 1.4083× | 1.8686× |
| **T5** | 6.9947× | 8.8480× | 10.0422× |
| **Vgg16** | 1.6235× | 0.6324× | 0.8828× |
| **Vit B 16** | 3.0809× | 3.7499× | 5.5300× |
| **Whisper Base** | 4.6777× | 4.5106× | 5.1276× |
| **Yolov8M** | 1.1979× | 0.4114× | 1.4271× |

---

## 4. Operation Type Performance (Best Checkpoint)

Only synthetic operation-type benchmarks (`add`, `conv_2d`, `matmul`, `pooling`, `relu`).

| Operation | `paper_original` | `paper_tf_large` | `paper_tf_small` |
|:----------|:------:|:------:|:------:|
| **Add** | 0.4592× | 0.3284× | 0.4431× |
| **Conv 2D** | 0.8846× | 3.8479× | 2.2470× |
| **Matmul** | 7.2048× | 1.6261× | 2.2111× |
| **Pooling** | 0.4046× | 0.1505× | 0.4484× |
| **Relu** | 0.3754× | 0.3212× | 0.3110× |

---

## 5. Top Individual Benchmark Performances

Best individual benchmark speedups from each agent's best checkpoint.

### `paper_original` (checkpoint 6100)

**Top-5 model benchmarks:**

| Rank | Benchmark | Family | Speedup |
|:----:|:----------|:-------|--------:|
| 1 | `llama3_2_1b_block_1079` | llama3_2_1b | 28647.78× |
| 2 | `llama3_2_1b_block_546` | llama3_2_1b | 28636.75× |
| 3 | `llama3_2_1b_block_430` | llama3_2_1b | 27565.30× |
| 4 | `llama3_2_1b_block_635` | llama3_2_1b | 24040.02× |
| 5 | `llama3_2_1b_block_1080` | llama3_2_1b | 23808.86× |

**Top-5 operation-type benchmarks:**

| Rank | Benchmark | Op Type | Speedup |
|:----:|:----------|:--------|--------:|
| 1 | `matmul_1536_512_128` | matmul | 11.02× |
| 2 | `matmul_512_512_256` | matmul | 10.82× |
| 3 | `matmul_512_512_128` | matmul | 10.77× |
| 4 | `matmul_256_512_768` | matmul | 10.68× |
| 5 | `matmul_1536_768_256` | matmul | 10.63× |

### `paper_tf_small` (checkpoint 7900)

**Top-5 model benchmarks:**

| Rank | Benchmark | Family | Speedup |
|:----:|:----------|:-------|--------:|
| 1 | `llama3_2_1b_block_1079` | llama3_2_1b | 28521.73× |
| 2 | `llama3_2_1b_block_635` | llama3_2_1b | 27916.80× |
| 3 | `llama3_2_1b_block_430` | llama3_2_1b | 27669.49× |
| 4 | `llama3_2_1b_block_679` | llama3_2_1b | 25349.51× |
| 5 | `llama3_2_1b_block_546` | llama3_2_1b | 25345.63× |

**Top-5 operation-type benchmarks:**

| Rank | Benchmark | Op Type | Speedup |
|:----:|:----------|:--------|--------:|
| 1 | `conv_2d_nchw_fchw_128_240_7_7_192_1_1_7_7` | conv_2d | 5.05× |
| 2 | `conv_2d_nchw_fchw_256_256_14_14_48_1_1_7_7` | conv_2d | 5.02× |
| 3 | `conv_2d_nchw_fchw_128_256_15_15_64_1_1_15_15` | conv_2d | 4.79× |
| 4 | `conv_2d_nchw_fchw_256_288_15_15_32_1_1_15_15` | conv_2d | 4.57× |
| 5 | `conv_2d_nchw_fchw_128_128_14_14_192_1_1_7_7` | conv_2d | 4.48× |

### `paper_tf_large` (checkpoint 1200)

**Top-5 model benchmarks:**

| Rank | Benchmark | Family | Speedup |
|:----:|:----------|:-------|--------:|
| 1 | `llama3_2_1b_block_635` | llama3_2_1b | 28668.59× |
| 2 | `llama3_2_1b_block_1079` | llama3_2_1b | 28177.08× |
| 3 | `llama3_2_1b_block_679` | llama3_2_1b | 27937.57× |
| 4 | `llama3_2_1b_block_430` | llama3_2_1b | 27010.40× |
| 5 | `llama3_2_1b_block_546` | llama3_2_1b | 24646.43× |

**Top-5 operation-type benchmarks:**

| Rank | Benchmark | Op Type | Speedup |
|:----:|:----------|:--------|--------:|
| 1 | `conv_2d_nchw_fchw_256_512_15_15_48_1_1_8_8` | conv_2d | 9.10× |
| 2 | `conv_2d_nchw_fchw_128_288_7_7_384_1_1_4_4` | conv_2d | 9.02× |
| 3 | `conv_2d_nchw_fchw_128_96_7_7_384_1_1_4_4` | conv_2d | 6.79× |
| 4 | `conv_2d_nchw_fchw_128_240_7_7_512_1_1_7_7` | conv_2d | 5.86× |
| 5 | `conv_2d_nchw_fchw_256_128_7_7_48_1_1_4_4` | conv_2d | 5.85× |

---

## 6. Generated Plots

- **Best Checkpoint Results**: `plots/experimentation_plots/exp2/pngs/best_checkpoint_results.png`
- **Best Checkpoint Results No Llama3**: `plots/experimentation_plots/exp2/pngs/best_checkpoint_results_no_llama3.png`
- **Checkpoint Evolution**: `plots/experimentation_plots/exp2/pngs/checkpoint_evolution.png`
- **Operation Type Results**: `plots/experimentation_plots/exp2/pngs/operation_type_results.png`
