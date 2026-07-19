# Experimentation Report

**Generated**: 2026-07-19 18:19  
**Dataset**: `ops_and_blocks`  
**Agents**: `paper_original`, `paper_transformer_small`, `paper_transformer_large`  
**Experiment directory**: `plots/experimentation_plots/exp1`

---

## 1. Best Checkpoint Summary

| Agent | Best Checkpoint | Peak Geo-Mean Speedup |
|:------|:--------------:|----------------------:|
| `paper_tf_small` | 14600 | **3.5749×** |
| `paper_original` | 12200 | **3.0084×** |
| `paper_tf_large` | 1200 | **2.7651×** |

### Per-Agent Detailed Stats (at best checkpoint)

| Agent | Best CP | Valid | Failed | Geo-Mean | Arith-Mean | Best Speedup | Worst Speedup |
|:------|:-------:|------:|-------:|---------:|-----------:|-------------:|---------------:|
| `paper_original` | 12200 | 1557 | 130 | 3.0084× | 99.76× | 28249.28× | 0.0448× |
| `paper_tf_small` | 14600 | 1557 | 130 | 3.5749× | 165.73× | 41157.10× | 0.1026× |
| `paper_tf_large` | 1200 | 1559 | 128 | 2.7651× | 121.59× | 28668.59× | 0.0162× |

---

## 2. Model Family Performance (Best Checkpoint)

Geo-mean speedup per model family across all agents.

| Model Family | `paper_original` | `paper_tf_large` | `paper_tf_small` |
|:-------------|:------:|:------:|:------:|
| **Albert** | 8.3475× | 8.0256× | 8.0716× |
| **Bart** | 6.9545× | 6.3429× | 7.1946× |
| **Bert** | 7.7254× | 7.5646× | 8.2781× |
| **Convnext Tiny** | 1.0279× | 0.9267× | 1.1013× |
| **Distilbert** | 7.1146× | 6.8034× | 7.4481× |
| **Efficientnet B0** | 1.5021× | 1.1959× | 1.4253× |
| **Gat** | 0.2492× | 1.3082× | 1.4463× |
| **Gin** | 1.2052× | 1.8772× | 2.0663× |
| **Gpt2** | 1.2010× | 1.3192× | 1.4144× |
| **Llama3 2 1B** | 169.3781× | 218.9669× | 312.1997× |
| **Mobilenet V3 Small** | 1.3714× | 1.2883× | 1.4730× |
| **Resnet50** | 3.1682× | 1.4573× | 3.1267× |
| **Resnext50** | 3.1736× | 1.4083× | 3.2961× |
| **T5** | 10.0462× | 8.8480× | 9.6571× |
| **Vgg16** | 0.9101× | 0.6324× | 1.7423× |
| **Vit B 16** | 5.2436× | 3.7499× | 5.6169× |
| **Whisper Base** | 5.2540× | 4.5106× | 5.0410× |
| **Yolov8M** | 1.8300× | 0.4114× | 1.9710× |

**Top-3 families by geo-mean speedup (averaged across agents):**

1. `llama3_2_1b` — avg geo-mean **233.5149×**
2. `t5` — avg geo-mean **9.5171×**
3. `albert` — avg geo-mean **8.1482×**

---

## 3. Operation Type Performance (Best Checkpoint)

Only synthetic operation-type benchmarks (`add`, `conv_2d`, `matmul`, `pooling`, `relu`).

| Operation | `paper_original` | `paper_tf_large` | `paper_tf_small` |
|:----------|:------:|:------:|:------:|
| **Add** | 0.2521× | 0.3284× | 0.5334× |
| **Conv 2D** | 1.5617× | 3.8479× | 3.2910× |
| **Matmul** | 0.7123× | 1.6261× | 2.9429× |
| **Pooling** | 0.2380× | 0.1505× | 0.4896× |
| **Relu** | 0.2118× | 0.3212× | 0.5152× |

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

### `paper_tf_small` (checkpoint 14600)

**Top-5 model benchmarks:**

| Rank | Benchmark | Family | Speedup |
|:----:|:----------|:-------|--------:|
| 1 | `llama3_2_1b_block_635` | llama3_2_1b | 41157.10× |
| 2 | `llama3_2_1b_block_679` | llama3_2_1b | 40727.86× |
| 3 | `llama3_2_1b_block_430` | llama3_2_1b | 40110.23× |
| 4 | `llama3_2_1b_block_546` | llama3_2_1b | 39447.62× |
| 5 | `llama3_2_1b_block_1079` | llama3_2_1b | 38184.38× |

**Top-5 operation-type benchmarks:**

| Rank | Benchmark | Op Type | Speedup |
|:----:|:----------|:--------|--------:|
| 1 | `conv_2d_nchw_fchw_128_96_28_28_288_1_1_14_14` | conv_2d | 5.77× |
| 2 | `conv_2d_nchw_fchw_128_256_15_15_64_1_1_15_15` | conv_2d | 5.35× |
| 3 | `conv_2d_nchw_fchw_256_512_15_15_48_1_1_8_8` | conv_2d | 5.12× |
| 4 | `conv_2d_nchw_fchw_256_128_14_14_64_1_1_14_14` | conv_2d | 4.77× |
| 5 | `conv_2d_nchw_fchw_256_48_56_56_96_1_1_28_28` | conv_2d | 4.74× |

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

## 5. Generated Plots

- **Best Checkpoint Results**: `plots/experimentation_plots/exp1/pngs/best_checkpoint_results.png`
- **Best Checkpoint Results No Llama3**: `plots/experimentation_plots/exp1/pngs/best_checkpoint_results_no_llama3.png`
- **Checkpoint Evolution**: `plots/experimentation_plots/exp1/pngs/checkpoint_evolution.png`
- **Operation Type Results**: `plots/experimentation_plots/exp1/pngs/operation_type_results.png`
