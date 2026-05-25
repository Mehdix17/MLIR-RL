# MLIR-RL Dataset: Model Selection Rationale

## Removed Models

| Model | Category | Reason for Removal |
|---|---|---|
| ResNet-18 | CNN | Redundant with ResNet-50. Same architectural pattern (residual conv stacks), just shallower. Keeping both provides no additional op diversity for the RL agent. |
| DeBERTa | Encoder Transformer | The encoder-only family (BERT, ALBERT, RoBERTa, DistilBERT, DeBERTa, ELECTRA) is heavily over-represented. All produce near-identical MLIR compute graphs dominated by self-attention and feed-forward blocks. Removing the entire group avoids redundancy. |
| RoBERTa | Encoder Transformer | Same rationale as DeBERTa. RoBERTa is BERT with different pretraining, not a different compute graph. No marginal value for schedule optimization benchmarking. |
| ELECTRA | Encoder Transformer | Same rationale. ELECTRA's discriminator/generator design is a training-time distinction that disappears in inference IR. The resulting MLIR is essentially BERT-equivalent. |
| GRU | RNN | Sequential recurrent ops are structurally incompatible with the parallelizable workloads that dominate modern deployment. Adds noise to the benchmark without representing a relevant optimization target. |
| BiLSTM | RNN | Same rationale as GRU. Bidirectional LSTM introduces gate-heavy sequential compute that is not representative of current production inference workloads. |
| GCN | Graph Neural Network | Functionally subsumed by GIN. GCN is a simplified special case of GIN's aggregation scheme. Keeping both provides no additional op diversity. |
| GraphSAGE | Graph Neural Network | Its sampling-based mean/max aggregation collapses to patterns already covered by GIN. The data-loading irregularity it introduces is harder to attribute to schedule optimization specifically. |

---

## Kept Models

### Encoder Transformers

| Model | Reason for Keeping |
|---|---|
| BERT | Retained as the single representative of the encoder-only family. Provides the standard self-attention + FFN op pattern (matmul-heavy) as a clean baseline. |
| ALBERT | Parameter-sharing design produces a distinct IR footprint compared to BERT despite similar architecture. Worth keeping as a lightweight attention representative. |
| DistilBERT | Distilled architecture with fewer layers. Useful for testing agent generalization across model depth variations within the same op family. |

### Decoder / Seq2Seq

| Model | Reason for Keeping |
|---|---|
| GPT-2 | Autoregressive decoder with causal attention mask. Different compute pattern from encoder-only models, important for covering the decoder side of the Transformer family. |
| BART | Encoder-decoder architecture. Covers the seq2seq compute pattern and serves as a bridge between pure encoder and pure decoder workloads. |
| T5 | Encoder-decoder with a distinct tokenization and attention scheme. Widely deployed in production, high benchmark value. |

### Vision

| Model | Reason for Keeping |
|---|---|
| ViT-B/16 | The canonical vision Transformer. Attention-heavy, matmul-dominant workload. Fundamentally different from CNN compute graphs, essential for Transformer-side vision benchmarking. |
| ResNet-50 | The canonical CNN backbone. Conv2D-dominant with residual connections. Non-negotiable baseline for any vision benchmark. |
| ResNeXt-50 | Grouped convolutions introduce a meaningfully different op pattern from vanilla ResNet. Worth keeping as a CNN variant that stress-tests group conv scheduling. |
| MobileNetV3-Small | Depthwise separable convolutions and squeeze-and-excitation blocks. Critical for edge/efficiency optimization scenarios, very different op distribution from full convolutions. |
| EfficientNet-B0 | Compound scaling across width, depth, and resolution. Diverse op shapes across the network make it a strong test of agent generalization. |
| ConvNeXt-Tiny | Modernized CNN with large 7x7 kernels, LayerNorm, and GELU. Bridges the gap between CNNs and Transformers in terms of op patterns. |
| VGG-11 | Simple, uniform conv stacks with no residuals. Useful as a low-complexity sanity baseline and for isolating pure conv scheduling behavior. |

### Graph Neural Networks

| Model | Reason for Keeping |
|---|---|
| GAT | Attention-weighted neighbor aggregation. Introduces softmax + weighted sum on irregular sparse graphs, the most distinct compute pattern in the GNN group and the closest to modern attention mechanisms. |
| GIN | Theoretically most expressive GNN (Weisfeiler-Lehman equivalent). Clean sum aggregation followed by MLP layers, providing a high-frequency matmul target on irregular data. Strong research benchmark coverage. |

### Object Detection

| Model | Reason for Adding |
|---|---|
| YOLOv8m | The most important missing model in the dataset. Conv2D-dominant with 200-350 conv ops (1x1 and 3x3 kernels), an SPPF pooling block, and feature pyramid neck concats. Entirely different op distribution from all existing models. High real-world deployment relevance and a strong latency-constrained benchmark target for the RL agent. |

### Modern LLMs

| Model | Reason for Adding |
|---|---|
| Llama 3.2 1B | Covers modern LLM op patterns absent from GPT-2: Grouped Query Attention (GQA), RoPE positional encoding, and RMSNorm. GQA specifically produces a distinct batch_matmul shape that differs from standard MHA, making it a high-value addition for schedule diversity. Compact enough (1B) to remain tractable as a benchmark target. |

---

## Future Works

Models evaluated but deferred to future dataset iterations due to architectural redundancy with existing entries or specialized complexity requiring dedicated benchmarking infrastructure.

| Model | Category | Rationale for Deferral |
|---|---|---|
| Whisper (encoder) | Audio Transformer | Op types are close to ViT-B/16. Main differentiator is input tensor shape (time x frequency). Low priority given current coverage, but worth adding for cross-modality generalization studies. |
| Qwen 0.8B | LLM | Architecturally very close to Llama (GQA, RMSNorm, rotary embeddings). Adding both provides size-point variation but minimal op diversity beyond what Llama already covers. |
| DeepSeek 1.5B (dense) | LLM | Dense variant has near-identical op structure to Llama. The interesting version is DeepSeek MoE (V2/V3) which introduces expert routing ops, deferred alongside Gemma MoE. |
| Gemma MoE 2B | Mixture-of-Experts | Most architecturally distinct of the group. Top-k routing and conditional expert dispatch introduce data-dependent scheduling problems not present anywhere in the current dataset. High future value but requires dedicated MoE benchmarking support. |

---

## Final Dataset Summary

| Category | Models | Count |
|---|---|---|
| Encoder Transformers | BERT, ALBERT, DistilBERT | 3 |
| Decoder / Seq2Seq | GPT-2, BART, T5 | 3 |
| Vision Transformers | ViT-B/16 | 1 |
| CNN Backbones | ResNet-50, ResNeXt-50, MobileNetV3-Small, EfficientNet-B0, ConvNeXt-Tiny, VGG-11 | 6 |
| Object Detection | YOLOv8m | 1 |
| Modern LLMs | Llama 3.2 1B | 1 |
| Graph Neural Networks | GAT, GIN | 2 |
| **Total** | | **17** |
