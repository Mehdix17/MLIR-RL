# Experimentation Report

**Generated**: 2026-08-14 04:59  
**Dataset**: `ops_and_blocks`  
**Agents**: `paper_transformer_large`  
**Experiment directory**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_large`

---

## 1. Best Checkpoint Summary

| Agent | Best Checkpoint | Peak Geo-Mean Speedup |
|:------|:--------------:|----------------------:|
| `paper_transformer_large` | 1200 | **2.9578×** |

### Per-Agent Detailed Stats (at best checkpoint)

| Agent | Best CP | Valid | Failed | Geo-Mean | Arith-Mean | Best Speedup | Worst Speedup |
|:------|:-------:|------:|-------:|---------:|-----------:|-------------:|---------------:|
| `paper_tf_large` | 1200 | 1501 | 128 | 2.9578× | 126.27× | 28668.59× | 0.0162× |

---

## 2. Model Family Performance (Best Checkpoint)

Geo-mean speedup per model family across all agents.

| Model Family | `paper_transformer_large` |
|:-------------|:------:|
| **Albert** | 8.4482× |
| **Bart** | 7.1600× |
| **Bert** | 7.9346× |
| **Convnext Tiny** | 1.0305× |
| **Distilbert** | 7.6426× |
| **Efficientnet B0** | 1.2120× |
| **Gat** | 1.5756× |
| **Gin** | 2.2835× |
| **Gpt2** | 1.3612× |
| **Llama3 2 1B** | 1437.9270× |
| **Mobilenet V3 Small** | 1.3490× |
| **Resnet50** | 1.6593× |
| **Resnext50** | 1.6207× |
| **T5** | 9.4596× |
| **Vgg16** | 0.7154× |
| **Vit B 16** | 4.0961× |
| **Whisper Base** | 4.9497× |
| **Yolov8M** | 0.4183× |

**Top-3 families by geo-mean speedup (averaged across agents):**

1. `llama3_2_1b` — avg geo-mean **1437.9270×**
2. `t5` — avg geo-mean **9.4596×**
3. `albert` — avg geo-mean **8.4482×**

---

## 3. Operation Type Performance (Best Checkpoint)

Only synthetic operation-type benchmarks (`add`, `conv_2d`, `matmul`, `pooling`, `relu`).

| Operation | `paper_transformer_large` |
|:----------|:------:|
| **Add** | 0.3284× |
| **Conv 2D** | 3.8479× |
| **Matmul** | 1.6261× |
| **Pooling** | 0.1505× |
| **Relu** | 0.3212× |

---

## 4. Top Individual Benchmark Performances

Best individual benchmark speedups from each agent's best checkpoint.

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

- **Best Checkpoint Benchmark Family Results**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_large/pngs/best_checkpoint_benchmark_family_results.png`
- **Best Checkpoint Benchmark Family Results No Llama3**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_large/pngs/best_checkpoint_benchmark_family_results_no_llama3.png`
- **Best Checkpoint Operation Type Results**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_large/pngs/best_checkpoint_operation_type_results.png`
- **Checkpoint Evolution**: `plots/experimentation_plots/ops_and_blocks/paper_transformer_large/pngs/checkpoint_evolution.png`
