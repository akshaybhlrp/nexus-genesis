#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# NEXUS AUTONOMOUS DIGITAL ORGANISM RUNNER
# Automates continuous cycles: Training -> Teacher Validation -> Nightly Sleep -> Eval -> Generation
# ==============================================================================

CYCLES=${1:-1}
STEPS_PER_CYCLE=${2:-30}
DATASET=${3:-"data/fineweb.bin"}
MODEL_DIR=${4:-"data/models/smollm2-135m"}

echo "════════════════════════════════════════════════════════════════"
echo "  🧬 NEXUS AUTONOMOUS ORGANISM LIFECYCLE CONTROLLER"
echo "  Cycles: $CYCLES | Steps/Cycle: $STEPS_PER_CYCLE"
echo "  Dataset: $DATASET | Model: $MODEL_DIR"
echo "════════════════════════════════════════════════════════════════"

# Ensure dataset exists
if [ ! -f "$DATASET" ]; then
    echo "⚠️  Dataset '$DATASET' not found. Packing default sample..."
    if [ -f "data/sample_corpus.parquet" ] && [ -f "data/tokenizer.json" ]; then
        cargo run --release -p nexus-core --bin nexus-pack-data -- \
            --tokenizer data/tokenizer.json \
            --out "$DATASET" \
            --seq-len 128 \
            --max-tokens 500000 \
            data/sample_corpus.parquet
    else
        echo "❌ Missing data prerequisites in data/. Exiting."
        exit 1
    fi
fi

for cycle in $(seq 1 "$CYCLES"); do
    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│ 🔄 ORGANISM CYCLE $cycle / $CYCLES"
    echo "└──────────────────────────────────────────────────────────────┘"

    echo ""
    echo ">> [Phase A] 🧠 Waking state: Hybrid MoE Training + Teacher Conscience..."
    cargo run --release -p nexus-core --bin nexus-train-hybrid -- \
        "$STEPS_PER_CYCLE" \
        "$DATASET" \
        --with-teacher \
        --tiered \
        --batch-size 2

    echo ""
    echo ">> [Phase B] 🌙 Sleeping state: Nightly Expert Consolidation & Pruning..."
    cargo run --release -p nexus-core --bin nexus-consolidate -- 0.95 0.02

    echo ""
    echo ">> [Phase C] 🪞 Mirror state: Offline Retention & Perplexity Evaluation..."
    cargo run --release -p nexus-eval --bin nexus-eval -- \
        "$DATASET" \
        --n-seqs 20

    echo ""
    echo ">> [Phase D] 🗣️ Voice state: Sampling Post-Evolution Output..."
    cargo run --release -p nexus-core --bin nexus-generate -- \
        "The digital organism has evolved to" \
        --tokens 25 \
        --temperature 0.7

    echo ""
    echo "✓ Cycle $cycle complete. Memory consolidated into L3 SSD warehouse."
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✨ ALL $CYCLES ORGANISM CYCLES EXECUTED SUCCESSFULLY"
echo "════════════════════════════════════════════════════════════════"
