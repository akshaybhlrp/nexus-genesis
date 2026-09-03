#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NEXUS AUTONOMOUS HOURLY SELF-OPTIMIZATION & EVOLUTION DAEMON
# Runs continuously every hour:
# 1. Hybrid MoE Training on ingested SmolLM2 weights
# 2. SVD Memory Consolidation & Dead Expert Pruning
# 3. Held-out Retention & Perplexity Evaluation
# 4. Text Generation Sampling & Progress Logging
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATASET="${DATASET:-data/full_50m_stream.bin}"
MODEL="${MODEL:-data/models/smollm2-135m-rectified}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LAYERS="${LAYERS:-4}"
EXPERTS="${EXPERTS:-4}"
REPORT_FILE="${REPORT_FILE:-data/hourly_report.log}"
INTERVAL="${INTERVAL:-3600}"

if [ ! -f "$DATASET" ]; then
    if [ -f "data/fineweb.bin" ]; then
        DATASET="data/fineweb.bin"
    fi
fi

mkdir -p data/checkpoints data/warehouse

# PID Lockfile Guard: Prevent duplicate daemon processes from competing for GPU
PID_FILE="data/.nexus_hourly.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️ Another Nexus daemon instance is already active (PID $OLD_PID). Exiting."
        exit 0
    fi
fi
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT INT TERM

# Dynamic Hardware & Backend Auto-Detection
if [ -n "${NEXUS_BACKEND:-}" ]; then
    BACKEND_FEATURE="$NEXUS_BACKEND"
    HW_DESC="Custom specified: $NEXUS_BACKEND"
elif command -v nvidia-smi &>/dev/null; then
    BACKEND_FEATURE="cuda"
    HW_DESC="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1) (CUDA Native)"
elif command -v rocm-smi &>/dev/null || [ -d "/opt/rocm" ]; then
    BACKEND_FEATURE="rocm"
    HW_DESC="AMD Radeon / Instinct (ROCm Native)"
elif [ "$(uname -s)" = "Darwin" ]; then
    BACKEND_FEATURE="metal"
    HW_DESC="$(uname -m) Apple Silicon (Metal Native)"
elif command -v vulkaninfo &>/dev/null; then
    BACKEND_FEATURE="vulkan"
    HW_DESC="Universal Vulkan GPU"
else
    BACKEND_FEATURE="wgpu"
    HW_DESC="$(uname -s -m) Universal WGPU / CPU"
fi

echo "==================================================================" | tee -a "$REPORT_FILE"
echo "  🚀 NEXUS HOURLY CONTINUOUS TRAINING DAEMON STARTED: $(date)" | tee -a "$REPORT_FILE"
echo "  Hardware Acceleration: $HW_DESC [Backend: $BACKEND_FEATURE]" | tee -a "$REPORT_FILE"
echo "  Model: $MODEL | Dataset: $DATASET | Steps/Hour: $STEPS | Layers: $LAYERS | Batch: $BATCH_SIZE" | tee -a "$REPORT_FILE"
echo "==================================================================" | tee -a "$REPORT_FILE"

cycle=1
while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "" | tee -a "$REPORT_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$REPORT_FILE"
    echo "  🔄 HOURLY CYCLE #$cycle - $TIMESTAMP" | tee -a "$REPORT_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$REPORT_FILE"

    echo ">> [Phase 1/4] 🧠 Hybrid Training & Backprop ($BACKEND_FEATURE)..." | tee -a "$REPORT_FILE"
    cargo run --release -p nexus-core --bin nexus-train-hybrid \
        --no-default-features --features "$BACKEND_FEATURE" -- \
        "$STEPS" \
        "$DATASET" \
        --model "$MODEL" \
        --checkpoint-dir data/checkpoints \
        --batch-size "$BATCH_SIZE" \
        --experts "$EXPERTS" \
        --layers "$LAYERS" 2>&1 | tee -a "$REPORT_FILE"

    echo "" | tee -a "$REPORT_FILE"
    echo ">> [Phase 2/4] 🌙 Sleeping: SVD Memory Consolidation & Pruning ($BACKEND_FEATURE)..." | tee -a "$REPORT_FILE"
    cargo run --release -p nexus-core --bin nexus-consolidate \
        --no-default-features --features "$BACKEND_FEATURE" -- \
        0.95 0.02 --checkpoint-dir data/checkpoints --model "$MODEL" --layers "$LAYERS" 2>&1 | tee -a "$REPORT_FILE"

    echo "" | tee -a "$REPORT_FILE"
    echo ">> [Phase 3/4] 🪞 Mirror: Held-Out Perplexity & Retention Eval ($BACKEND_FEATURE)..." | tee -a "$REPORT_FILE"
    cargo run --release -p nexus-eval --bin nexus-eval \
        --no-default-features --features "$BACKEND_FEATURE" -- \
        "$DATASET" \
        --model "$MODEL" \
        --layers "$LAYERS" \
        --n-seqs 20 2>&1 | tee -a "$REPORT_FILE"

    echo "" | tee -a "$REPORT_FILE"
    echo ">> [Phase 4/4] 🗣️ Voice: Sampling Text Progression ($BACKEND_FEATURE)..." | tee -a "$REPORT_FILE"
    cargo run --release -p nexus-core --bin nexus-generate \
        --no-default-features --features "$BACKEND_FEATURE" -- \
        "The fundamental law of intelligence states that" \
        --model "$MODEL" \
        --layers "$LAYERS" \
        --tokens 30 \
        --temperature 0.7 2>&1 | tee -a "$REPORT_FILE"

    echo "" | tee -a "$REPORT_FILE"
    echo "✅ Cycle #$cycle completed at $(date). Sleeping for 1 hour..." | tee -a "$REPORT_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$REPORT_FILE"

    cycle=$((cycle + 1))

    # Log rotation: Keep hourly report bounded (max 5000 lines) to prevent disk bloat
    if [ -f "$REPORT_FILE" ] && [ "$(wc -l < "$REPORT_FILE")" -gt 5000 ]; then
        tail -n 2500 "$REPORT_FILE" > "${REPORT_FILE}.tmp" && mv "${REPORT_FILE}.tmp" "$REPORT_FILE"
    fi

    sleep "$INTERVAL"
done
