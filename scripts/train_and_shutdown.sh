#!/usr/bin/env bash
# ==============================================================================
# NEXUS TRAINING RUN WITH AUTO-SHUTDOWN
# Runs specified number of cycles, persists all weights to SSD, then shuts down PC.
# Usage:
#   ./scripts/train_and_shutdown.sh [CYCLES] [STEPS_PER_CYCLE]
# Example:
#   ./scripts/train_and_shutdown.sh 10 50
# ==============================================================================

set -euo pipefail

CYCLES=${1:-10}
STEPS_PER_CYCLE=${2:-50}
DATASET="data/full_50m_stream.bin"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  🧬 NEXUS TRAINING RUN WITH AUTO-SHUTDOWN"
echo "  Target Cycles: $CYCLES ($STEPS_PER_CYCLE steps each)"
echo "  Dataset: $DATASET"
echo "  Action upon finish: SAVE WEIGHTS & SHUTDOWN SYSTEM"
echo "════════════════════════════════════════════════════════════════"

# Temporarily pause background daemon to allocate 100% compute to this run
systemctl --user stop nexus-organism.service 2>/dev/null || true

# Execute requested training cycles
./scripts/run_organism.sh "$CYCLES" "$STEPS_PER_CYCLE" "$DATASET"

echo ""
echo "[✓] All $CYCLES training & evolution cycles finished."
echo "[✓] Evolved weights safely persisted in L3 SSD Warehouse."
echo "[!] Syncing disk buffers and shutting down in 10 seconds..."
sync
sleep 10

# Shutdown command (systemd or shutdown)
systemctl poweroff || sudo shutdown -h now
