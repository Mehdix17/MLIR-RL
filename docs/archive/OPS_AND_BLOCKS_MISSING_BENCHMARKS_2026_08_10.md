# Missing Benchmarks Removed from ops_and_blocks Baselines

- **Date**: 2026-08-10
- **Why**: 147 benchmark names present in `base_train.json` / `base_eval.json`
  have no `.mlir` file in `data/ops_and_blocks` (they were lost in the 2026-08-06
  `data/` wipe and were not part of the restored backup).
- **What was done**: the names below were removed from the two split JSONs so
  `Benchmarks` loading does not crash (a missing file raises an uncaught
  `Exception` from the AST dumper: "Could not open input file").
- **Backups**: `results/ops_and_blocks_results/baselines/mlir/base_train.json.bak`
  and `base_eval.json.bak` (originals with all names intact).
- **To restore**: copy the names back into the JSONs once the `.mlir` files exist
  again (e.g. regenerated from `data/raw_models/`), then delete the `.bak` files.

After removal: train split = 6464 benches, eval split = 1629 benches.

## Train split — missing (89)

| `albert_add_9` | `albert_mul_19` | `albert_reduce_sum_24` | `albert_reduce_sum_4`|
| `bart_reduce_sum_4` | `bert_reduce_sum_4` | `convnext_tiny_add_14` | `convnext_tiny_add_29`|
| `convnext_tiny_add_50` | `convnext_tiny_mul_49` | `convnext_tiny_reduce_sum_0` | `convnext_tiny_reduce_sum_30`|
| `convnext_tiny_reduce_sum_45` | `convnext_tiny_reduce_sum_61` | `distilbert_reduce_sum_4` | `distilbert_sub_20`|
| `efficientnet_b0_add_48` | `efficientnet_b0_mul_1` | `efficientnet_b0_mul_36` | `efficientnet_b0_mul_43`|
| `efficientnet_b0_mul_45` | `efficientnet_b0_mul_71` | `efficientnet_b0_mul_9` | `efficientnet_b0_reduce_sum_12`|
| `efficientnet_b0_reduce_sum_20` | `efficientnet_b0_reduce_sum_29` | `efficientnet_b0_reduce_sum_33` | `efficientnet_b0_reduce_sum_42`|
| `efficientnet_b0_reduce_sum_55` | `efficientnet_b0_reduce_sum_64` | `efficientnet_b0_reduce_sum_68` | `efficientnet_b0_reduce_sum_77`|
| `gat_reduce_sum_9` | `gpt2_add_13` | `gpt2_mul_15` | `gpt2_reduce_sum_4`|
| `llama3_2_1b_block_1392` | `llama3_2_1b_block_179` | `llama3_2_1b_block_180` | `llama3_2_1b_block_232`|
| `llama3_2_1b_block_288` | `llama3_2_1b_block_368` | `llama3_2_1b_reduce_sum_27` | `mobilenet_v3_small_add_108`|
| `mobilenet_v3_small_add_91` | `mobilenet_v3_small_mul_22` | `mobilenet_v3_small_mul_27` | `mobilenet_v3_small_mul_4`|
| `mobilenet_v3_small_mul_81` | `mobilenet_v3_small_mul_9` | `mobilenet_v3_small_mul_94` | `mobilenet_v3_small_reduce_sum_100`|
| `mobilenet_v3_small_reduce_sum_28` | `mobilenet_v3_small_reduce_sum_41` | `mobilenet_v3_small_reduce_sum_6` | `mobilenet_v3_small_reduce_sum_68`|
| `mobilenet_v3_small_reduce_sum_87` | `mobilenet_v3_small_relu_14` | `mobilenet_v3_small_relu_51` | `mobilenet_v3_small_relu_64`|
| `resnet50_add_28` | `resnet50_add_41` | `resnet50_add_49` | `resnet50_mul_14`|
| `resnet50_mul_40` | `resnet50_relu_25` | `resnet50_relu_38` | `resnet50_relu_42`|
| `resnet50_sub_0` | `resnext50_add_2` | `resnext50_add_33` | `resnext50_add_6`|
| `resnext50_mul_27` | `resnext50_relu_25` | `resnext50_relu_30` | `t5_add_4`|
| `t5_reduce_sum_25` | `vit_b_16_reduce_sum_1` | `whisper_base_mul_18` | `whisper_base_mul_22`|
| `whisper_base_mul_36` | `whisper_base_reduce_sum_11` | `yolov8m_add_14` | `yolov8m_add_15`|
| `yolov8m_add_19` | `yolov8m_block_14` | `yolov8m_block_36` | `yolov8m_block_58`|
| `yolov8m_block_69`   |

## Eval split — missing (58)

| `albert_add_17` | `albert_mul_11` | `albert_reduce_sum_27` | `albert_sub_22`|
| `bart_add_24` | `bart_mul_17` | `bart_reduce_sum_21` | `bart_sub_19`|
| `bert_add_2` | `bert_reduce_sum_22` | `bert_sub_7` | `convnext_tiny_add_25`|
| `convnext_tiny_mul_43` | `convnext_tiny_reduce_sum_15` | `convnext_tiny_sub_3` | `distilbert_add_3`|
| `distilbert_mul_18` | `distilbert_reduce_sum_22` | `distilbert_sub_7` | `efficientnet_b0_add_24`|
| `efficientnet_b0_mul_38` | `efficientnet_b0_reduce_sum_2` | `efficientnet_b0_reduce_sum_46` | `gat_reduce_sum_12`|
| `gin_add_5` | `gpt2_add_9` | `gpt2_mul_11` | `gpt2_reduce_sum_19`|
| `llama3_2_1b_block_1391` | `llama3_2_1b_block_511` | `llama3_2_1b_mul_9` | `llama3_2_1b_reduce_sum_4`|
| `llama3_2_1b_sub_25` | `mobilenet_v3_small_add_36` | `mobilenet_v3_small_mul_58` | `mobilenet_v3_small_mul_71`|
| `mobilenet_v3_small_reduce_sum_55` | `mobilenet_v3_small_relu_15` | `resnet50_add_11` | `resnet50_mul_35`|
| `resnet50_relu_16` | `resnet50_sub_8` | `resnext50_add_15` | `resnext50_mul_5`|
| `resnext50_relu_16` | `resnext50_sub_4` | `t5_add_27` | `t5_reduce_sum_2`|
| `vgg16_relu_4` | `vit_b_16_mul_25` | `vit_b_16_reduce_sum_16` | `vit_b_16_sub_14`|
| `whisper_base_add_2` | `whisper_base_mul_8` | `whisper_base_reduce_sum_27` | `whisper_base_sub_14`|
| `yolov8m_add_2` | `yolov8m_mul_13`  |
