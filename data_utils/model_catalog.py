"""
model_catalog.py
----------------
Central model catalog for data_utils.

Edit this file to add/remove supported models, HuggingFace repo IDs,
or transformer input policy defaults.
"""

VISION_MODELS = [
    "resnet18", "resnet50", "resnext50",
    "efficientnet_b0",
    "mobilenet_v2", "mobilenet_v3_small",
    "densenet121",
    "vit_b_16",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "vgg11",
]

TRANSFORMER_MODELS = [
    "bert", "distilbert", "roberta", "albert", "deberta", "electra",
    "gpt2", "t5", "bart",
    "switch_base_8_moe",
    "lstm", "lstm_seq2seq", "gru", "bilstm",
]

GNN_MODELS = ["gcn", "graphsage", "gat", "gin"]

# Models that come from HuggingFace hubs (non-synthetic transformer variants).
TRANSFORMER_HF_REPOS = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "albert": "albert-base-v2",
    "gpt2": "gpt2-medium",
    "t5": "t5-small",
    "bart": "facebook/bart-base",
    "deberta": "microsoft/deberta-v3-base",
    "electra": "google/electra-small-discriminator",
    "switch_base_8_moe": "google/switch-base-8",
}


# Synthetic-first input policy used by transformers2mlir.py.
#
# Schema:
#   default_mode: "synthetic" or "tokenizer"
#   defaults: scalar defaults used to resolve symbolic dims in synthetic_tensors
#   synthetic_tensors: ordered tensor specs consumed by the model forward
#     - name: tensor name
#     - dtype: one of int64, int32, float32, float64
#     - shape: list of ints or symbolic dim names (resolved from defaults/runtime)
#     - init: ones | zeros | randn | randint
#     - randint_low / randint_high (optional): randint bounds
#   tokenizer_text: optional fallback text when --input-mode tokenizer is used
#   output_names: names passed to ONNX export
TRANSFORMER_INPUT_SPECS = {
    "bert": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 30522},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "distilbert": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 30522},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "roberta": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 50265},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "albert": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 30000},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "deberta": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 128000},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "electra": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 30522},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "gpt2": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 50257, "use_cache": False},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "Hello from MLIR-RL!",
        "output_names": ["last_hidden_state"],
    },
    "t5": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 32128},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
        ],
        "tokenizer_text": "Studies show that owning a dog is good for you",
        "output_names": ["last_hidden_state"],
    },
    "bart": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 50265, "use_cache": False},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "The cat sat on the mat.",
        "output_names": ["last_hidden_state"],
    },
    "switch_base_8_moe": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "vocab_size": 32128},
        "synthetic_tensors": [
            {"name": "input_ids", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "randint", "randint_low": 0, "randint_high": "vocab_size"},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch_size", "seq_len"], "init": "ones"},
        ],
        "tokenizer_text": "MoE routing test sentence.",
        "output_names": ["last_hidden_state"],
    },
    "lstm": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "input_size": 512, "hidden_size": 768},
        "synthetic_tensors": [
            {"name": "input", "dtype": "float32", "shape": ["batch_size", "seq_len", "input_size"], "init": "randn"},
            {"name": "h0", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
            {"name": "h1", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
        ],
        "output_names": ["output"],
    },
    "lstm_seq2seq": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "input_size": 512, "hidden_size": 768},
        "synthetic_tensors": [
            {"name": "src", "dtype": "float32", "shape": ["batch_size", "seq_len", "input_size"], "init": "randn"},
            {"name": "tgt", "dtype": "float32", "shape": ["batch_size", "seq_len", "hidden_size"], "init": "randn"},
            {"name": "h", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
        ],
        "output_names": ["output"],
    },
    "gru": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "input_size": 512, "hidden_size": 768},
        "synthetic_tensors": [
            {"name": "input", "dtype": "float32", "shape": ["batch_size", "seq_len", "input_size"], "init": "randn"},
            {"name": "h0", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
        ],
        "output_names": ["output"],
    },
    "bilstm": {
        "default_mode": "synthetic",
        "defaults": {"batch_size": 1, "seq_len": 16, "input_size": 512, "hidden_size": 384},
        "synthetic_tensors": [
            {"name": "input", "dtype": "float32", "shape": ["batch_size", "seq_len", "input_size"], "init": "randn"},
            {"name": "hf", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
            {"name": "hb", "dtype": "float32", "shape": ["batch_size", "hidden_size"], "init": "randn"},
        ],
        "output_names": ["output"],
    },
}
