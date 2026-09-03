#!/usr/bin/env bash
# ==============================================================================
# NEXUS DYNAMIC BACKGROUND ORGANISM DAEMON
# Runs continuous self-evolution in the background with dynamic resource yielding:
# - Lowest CPU/IO Priority (nice 19 / ionice idle) -> yields immediately to foreground apps
# - VRAM Governor: Pauses if GPU free memory is below 1.5GB
# - Auto-resumes continuous learning cycles
# ==============================================================================

set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$PROJECT_DIR/data/full_50m_stream.bin"
LOG_FILE="$PROJECT_DIR/data/nexus_daemon.log"
MIN_FREE_VRAM_MB=1400

cd "$PROJECT_DIR" || exit 1
mkdir -p "$PROJECT_DIR/data"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Nexus Background Organism Daemon..." >> "$LOG_FILE"

while true; do
    # 1. Dynamic VRAM Contention Check
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}' || echo "4096")
    
    if [ "$FREE_VRAM" -lt "$MIN_FREE_VRAM_MB" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU in high foreground use (Free: ${FREE_VRAM}MB < ${MIN_FREE_VRAM_MB}MB). Yielding GPU for 30s..." >> "$LOG_FILE"
        sleep 30
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Organism Evolution Cycle (Free VRAM: ${FREE_VRAM}MB)..." >> "$LOG_FILE"

    # 2. Run 1 cycle of 30 steps with lowest CPU/IO priority
    nice -n 19 ionice -c 3 ./scripts/run_organism.sh 1 30 "$DATASET" >> "$LOG_FILE" 2>&1 || {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Organism cycle returned with error or interruption. Backing off 15s..." >> "$LOG_FILE"
        sleep 15
    }

    # Short rest between cycles to let CPU/GPU flush thermals
    sleep 5
done
