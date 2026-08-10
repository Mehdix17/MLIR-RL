#!/bin/bash
# regen_all.sh
# Re-generates all MLIR models with multi-size / multi-seq-len variants,
# then re-extracts all single-op benchmarks with multi-batch and the
# reduction-loop filter.
#
# Usage:
#   nohup bash scripts/regen_all.sh > logs/regen_all.log 2>&1 &
#
# Estimated runtime: 3-6 hours depending on HPC load.

set -uo pipefail

export GCC14_LIB=/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-14.2.0-wfwb3ds4a5thcsh5w5o23k6wq7ob5ok3/lib64
export LD_LIBRARY_PATH="$GCC14_LIB:$LD_LIBRARY_PATH"
export PYTHONPATH="/scratch/tb3654/MLIR-RL:$PYTHONPATH"

cd /scratch/tb3654/MLIR-RL

mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Run a command, log pass/fail, but do NOT abort the whole script on failure.
run() {
    local label="$1"; shift
    log "START  $label"
    if "$@" 2>&1; then
        log "OK     $label"
    else
        log "FAILED $label (exit $?) — continuing"
    fi
}

# ---------------------------------------------------------------------------
# Phase 1: Vision models  (4 image sizes)
# ---------------------------------------------------------------------------
log "======== Phase 1: Vision models ========"

VISION_MULTI="resnet18 resnet50 resnext50 efficientnet_b0 mobilenet_v2
              mobilenet_v3_small densenet121 convnext_tiny vgg11"

for model in $VISION_MULTI; do
    run "vision/$model @224,192,160,128" \
        python data_utils/vision2mlir.py \
            --model "$model" \
            --img-sizes 224 192 160 128 \
            --backend direct

    # Remove old unsuffixed file — superseded by _sz{N} variants
    rm -f "data/nn/raw/${model}_linalg.mlir" \
          "data/nn/raw/${model}_torch.mlir"
done

# ViT requires exactly 224 — keep original filename (no size suffix)
run "vision/vit_b_16 @224" \
    python data_utils/vision2mlir.py \
        --model vit_b_16 \
        --img-sizes 224 \
        --backend direct

# ---------------------------------------------------------------------------
# Phase 2: Transformer models  (4 sequence lengths)
# ---------------------------------------------------------------------------
log "======== Phase 2: Transformer models ========"

TRANSFORMER_MULTI="bert distilbert roberta albert electra
                   gpt2 t5 bart
                   lstm lstm_seq2seq gru bilstm"

for model in $TRANSFORMER_MULTI; do
    run "transformer/$model @sl=16,32,64,128" \
        python data_utils/transformers2mlir.py \
            --model "$model" \
            --seq-lens 16 32 64 128 \
            --backend direct

    rm -f "data/nn/raw/${model}_linalg.mlir" \
          "data/nn/raw/${model}_torch.mlir"
done

# DeBERTa is large/slow — limit to 16 + 32
run "transformer/deberta @sl=16,32" \
    python data_utils/transformers2mlir.py \
        --model deberta \
        --seq-lens 16 32 \
        --backend direct

rm -f "data/nn/raw/deberta_linalg.mlir" \
      "data/nn/raw/deberta_torch.mlir"

# ---------------------------------------------------------------------------
# Phase 3: GNN models  (fixed graph size — regenerate as-is)
# ---------------------------------------------------------------------------
log "======== Phase 3: GNN models ========"

for model in gcn graphsage gat gin; do
    run "gnn/$model" \
        python data_utils/gnn2mlir.py --model "$model"
done

# ---------------------------------------------------------------------------
# Phase 4: Inventory
# ---------------------------------------------------------------------------
log "======== Phase 4: raw/ inventory ========"
log "Total raw linalg files: $(ls data/nn/raw/*_linalg.mlir 2>/dev/null | wc -l)"
ls data/nn/raw/*_linalg.mlir 2>/dev/null | sort | while read f; do
    log "  $(basename $f)"
done

# ---------------------------------------------------------------------------
# Phase 5: Extract all benchmarks
# ---------------------------------------------------------------------------
log "======== Phase 5: Extraction ========"

# Clear old code_files/ — will be fully rebuilt from raw/
rm -rf data/nn/code_files/
mkdir -p data/nn/code_files/

TOTAL_FILES=0
for f in $(ls data/nn/raw/*_linalg.mlir 2>/dev/null | sort); do
    model=$(basename "$f" _linalg.mlir)
    run "extract/$model bs=1,4,16" \
        python data_utils/extract_ops.py \
            --input "$f" \
            --output-dir "data/nn/code_files/$model" \
            --batch-sizes 1 4 16 \
            --model-name "$model"
    n=$(ls "data/nn/code_files/$model/"*.mlir 2>/dev/null | wc -l)
    TOTAL_FILES=$((TOTAL_FILES + n))
    log "  $model: $n benchmarks"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "======== Done ========"
log "Raw MLIR files  : $(ls data/nn/raw/*_linalg.mlir 2>/dev/null | wc -l)"
log "Model subdirs   : $(ls -d data/nn/code_files/*/ 2>/dev/null | wc -l)"
log "Total benchmarks: $(find data/nn/code_files/ -name '*.mlir' 2>/dev/null | wc -l)"
log ""
log "Per-model counts:"
for d in $(ls -d data/nn/code_files/*/ 2>/dev/null | sort); do
    n=$(ls "$d"*.mlir 2>/dev/null | wc -l)
    log "  $(basename $d): $n"
done
