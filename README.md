<div align="center">

# 🧬 Nexus Genesis

**An Autonomous, Continuously Self-Evolving Neural Organism Built in Pure Rust**

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/engine-Burn%200.21-purple.svg)](https://burn.dev/)
[![Hardware](https://img.shields.io/badge/backends-CUDA%20%7C%20ROCm%20%7C%20Vulkan%20%7C%20Metal%20%7C%20CPU-brightgreen.svg)](#hardware-acceleration)

</div>

---

## 🌟 Overview

**Nexus** is a production-ready, autonomous artificial intelligence engine designed to operate as a continuous digital organism. Unlike traditional static deep learning pipelines that require full parameter retention in expensive GPU VRAM, Nexus implements a biological life cycle powered by **Sparse Top-2 Mixture-of-Experts (MoE)**, **Halley-Muon Stiefel manifold optimization**, and **Tiered Memory Warehousing**.

Nexus can ingest foundation models (such as SmolLM2 or Llama architectures), upcycle dense layers into dynamic expert banks, and evolve autonomously in the background—yielding to foreground system tasks and operating efficiently on hardware ranging from 4GB laptop GPUs to multi-GPU enterprise servers.

---

## 🔬 Core Innovations

### 1. The 4-Phase Biological Life Cycle
Nexus structures learning into an autonomous continuous rhythm:
- **Phase 1: Awakening (Hybrid Training)** — Sparse Top-2 MoE routing combined with backward propagation, adaptive mutation ($\mu$), and entropy-governed teacher validation.
- **Phase 2: Sleep (SVD Memory Consolidation)** — Offline sleep cycle computing pairwise cosine similarity across expert matrices, consolidating redundant pathways via Singular Value Decomposition (SVD), pruning dormant experts to SSD, and spawning exploratory mutants.
- **Phase 3: Mirror (Self-Reflection & Evaluation)** — Zero-shot evaluation across held-out tokens measuring retention rate and perplexity to guard against catastrophic forgetting.
- **Phase 4: Voice (Generative Sampling)** — Native autoregressive sampling generating qualitative evidence of conceptual synthesis.

### 2. Halley-Muon 5th-Order Manifold Optimization
Standard AdamW suffers from gradient explosion and dimensional collapse in sparse architectures. Nexus incorporates a 5th-order Halley-Schulz matrix iteration that projects gradient steps onto the **Stiefel Manifold** ($W^T W = I$), ensuring isometric representation capacity and scale-invariant learning dynamics.

### 3. Tiered Expert Warehouse
Physical VRAM is no longer a bottleneck. Nexus decouples active routing compute from long-term parameter storage:
- **L1 Cache (VRAM/RAM)**: Hot working experts currently engaged by router gating.
- **L2 Cache (Memory-Mapped)**: Zero-copy virtual memory access for warm experts.
- **L3 Cold Warehouse (NVMe SSD)**: Persistent, serialized expert banks loaded on-demand.

### 4. Zero-Amnesia Atomic Checkpointing
Checkpoints are committed using OS-level atomic temporary writes (`.tmp.<pid>` $\rightarrow$ `sync_all()` $\rightarrow$ atomic rename). State continuity tracks global step counts, loss trajectory, and circular dataset cursors without stalling or memory exhaustion.

---

## 🚀 Hardware Acceleration

Nexus compiles natively across major modern compute backends via Burn and CubeCL:

| Backend | Platform | Target Devices | Build Command |
| :--- | :--- | :--- | :--- |
| **CUDA** *(Default)* | Linux / Windows | NVIDIA GPUs (GTX, RTX, Tesla, A100, H100) | `cargo run --release --features cuda` |
| **ROCm** | Linux | AMD Radeon / Instinct GPUs (RDNA, CDNA) | `cargo run --release --no-default-features --features rocm` |
| **Metal** | macOS / iOS | Apple Silicon (M1, M2, M3, M4) | `cargo run --release --no-default-features --features metal` |
| **Vulkan** | Linux / Windows / Android | AMD, Intel ARC, NVIDIA, Adreno | `cargo run --release --no-default-features --features vulkan` |
| **WGPU** | Cross-Platform | Universal WebGPU / WGPU abstraction | `cargo run --release --no-default-features --features wgpu` |
| **CPU** | Any OS | Portable fallback (NdArray / AVX2 / NEON) | `cargo run --release --no-default-features` |

---

## 📦 Project Architecture

```
nexus/
├── crates/
│   ├── nexus-core/         # Core MoE Llama, Halley-Muon, Checkpointing, and Binaries
│   │   ├── src/bin/nexus-train-hybrid.rs   # Phase 1: Hybrid MoE Training
│   │   ├── src/bin/nexus-consolidate.rs    # Phase 2: SVD Sleep Memory Consolidation
│   │   ├── src/bin/nexus-generate.rs       # Phase 4: Native Autoregressive Generation
│   │   ├── src/bin/nexus-serve.rs          # Low-latency OpenAI-compatible API Server
│   │   ├── src/checkpoint.rs               # Atomic crash-resilient checkpoint engine
│   │   └── src/halley_muon.rs              # 5th-order Stiefel manifold retraction
│   ├── nexus-memory/       # Tiered L1/L2/L3 Expert Warehouse & Poison-Safe Locks
│   ├── nexus-eval/         # Mirror: Offline Perplexity & Retention Benchmark
│   ├── nexus-teacher/      # External Teacher Validation & LRU Cache
│   ├── nexus-emns/         # Emergent Macro-Resistance Mutation Kernels
│   └── nexus-weight-gen/   # Invariant-Preserving Weight Initializers
├── scripts/
│   ├── nexus_hourly_train.sh  # Autonomous 24/7 background self-evolution daemon
│   ├── rectify_smollm.py      # Ingests & rectifies Hugging Face weights to Nexus layout
│   └── train_and_shutdown.sh  # Scheduled execution with automated host poweroff
└── ui/
    ├── index.html             # Glassmorphic real-time telemetry dashboard
    └── server.py              # WebSocket telemetry bridge & API gateway
```

---

## ⚡ Getting Started

### 1. Prerequisites
- **Rust Toolchain**: 1.80 or newer (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Python 3.10+** (for dataset tokenization and UI server)
- Compute drivers (CUDA toolkit, ROCm, Vulkan SDK, or Xcode Command Line Tools for macOS)

### 2. Ingest Foundation Model (SmolLM2)
Nexus can build from scratch or ingest and rectify existing open weights:

```bash
# Ingest and rectify SmolLM2-135M into Nexus binary tensor layout
python3 scripts/rectify_smollm.py
```

### 3. Run Autonomous Hourly Daemon
Launch the continuous self-evolution daemon. The script automatically probes host hardware, assigns optimal backends, acquires a PID lock, and begins the 4-phase biological cycle:

```bash
bash scripts/nexus_hourly_train.sh
```

### 4. Interactive Text Generation
Sample tokens natively from the trained checkpoint:

```bash
cargo run --release -p nexus-core --bin nexus-generate -- \
    "The fundamental law of intelligence states that" \
    --model data/models/smollm2-135m-rectified \
    --tokens 50 \
    --temperature 0.7
```

### 5. Launch Real-time Web Dashboard
Monitor live loss trajectories, expert activity heatmaps, and routing entropy:

```bash
python3 ui/server.py
# Open http://localhost:8080 in your browser
```

---

## ⚙️ Configuration Reference

All binaries accept standard CLI flags and environment variables:

| Argument | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--layers <N>` | `NEXUS_LAYERS` | `4` (or `0` for all) | Number of transformer layers to load (fits 4GB to 80GB VRAM) |
| `--batch-size <N>` | `NEXUS_BATCH_SIZE` | `1` | Micro-batch size for training/eval |
| `--experts <N>` | `EXPERTS` | `4` | Number of experts per MoE block |
| `--lr <FLOAT>` | `NEXUS_LR` | `0.0003` | Base learning rate for AdamW / Muon |
| `--checkpoint-dir <P>`| `CHECKPOINT_DIR`| `data/checkpoints` | Atomic checkpoint storage directory |
| `--model <P>` | `MODEL` | SmolLM2 rectified | Foundation model directory |
| `--dataset <P>` | `DATASET` | `data/full_50m_stream.bin` | Packed binary dataset stream |
| `--backend <NAME>` | `NEXUS_BACKEND` | Auto-detected | Force compute backend (`cuda`, `rocm`, `vulkan`, `metal`) |

---

## 📄 License

Nexus Genesis is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.
