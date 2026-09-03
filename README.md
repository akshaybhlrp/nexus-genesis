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
└── ui/
    └── index.html          # Glassmorphic real-time telemetry dashboard
```

---

## ⚡ Getting Started & Usage Guide

### 1. Prerequisites & Compilation
Ensure Rust 1.80+ is installed (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`).

Build all optimized release binaries for your specific compute hardware:

```bash
# NVIDIA GPUs (CUDA)
cargo build --release --features cuda

# AMD Radeon / Instinct GPUs (ROCm)
cargo build --release --no-default-features --features rocm

# Apple Silicon Macs (Metal)
cargo build --release --no-default-features --features metal

# Cross-platform Vulkan (Linux, Windows, Intel ARC, AMD)
cargo build --release --no-default-features --features vulkan

# CPU Fallback (NdArray)
cargo build --release --no-default-features
```

### 2. Dataset Preparation
Nexus streams pre-tokenized binary sequences (`[u32 vocab][u32 seq_len][u32 n_seqs]...`).

- **Zero-Setup Quickstart**: If no dataset is specified, Nexus automatically streams infinite synthetic data for immediate exploration.
- **Custom Tokenized Stream**: Convert any Parquet dataset into a binary stream using `nexus-pack-data`:
  ```bash
  cargo run --release -p nexus-core --bin nexus-pack-data -- \
      --tokenizer data/tokenizer.json \
      --out data/my_stream.bin \
      --seq-len 128 \
      --max-tokens 50000000 \
      path/to/dataset/*.parquet
  ```

### 3. Foundation Weight Ingestion (Optional)
Nexus can train from scratch or ingest and upcycle open Hugging Face weights into dynamic MoE expert banks:

```bash
cargo run --release -p nexus-core --bin nexus-import-hf -- \
    data/models/smollm2-135m \
    --upcycle \
    --warehouse
```

### 4. Running the Biological Life Cycle

Execute each biological phase directly using the compiled native Rust binaries:

```bash
# Phase 1: Awakening (Hybrid MoE Training with Halley-Muon)
./target/release/nexus-train-hybrid 50 data/my_stream.bin \
    --model data/models/smollm2-135m \
    --checkpoint-dir data/checkpoints \
    --batch-size 1 \
    --layers 4 \
    --experts 4

# Phase 2: Sleep (SVD Memory Consolidation & Expert Pruning)
./target/release/nexus-consolidate 0.95 0.02 \
    --checkpoint-dir data/checkpoints \
    --model data/models/smollm2-135m \
    --layers 4

# Phase 3: Mirror (Held-Out Perplexity & Catastrophic Forgetting Eval)
./target/release/nexus-eval data/my_stream.bin \
    --model data/models/smollm2-135m \
    --layers 4 \
    --n-seqs 20

# Phase 4: Voice (Native Text Generation)
./target/release/nexus-generate "The fundamental law of intelligence states that" \
    --model data/models/smollm2-135m \
    --layers 4 \
    --tokens 50 \
    --temperature 0.7
```

### 5. Interactive Chat & Serving
Interact with the organism via interactive terminal chat or the low-latency resident inference server:

```bash
# Interactive CLI Conversation
./target/release/nexus-chat --model data/models/smollm2-135m

# Resident GPU Inference Server (serves stdin/stdout JSON queries)
./target/release/nexus-serve --model data/models/smollm2-135m
```

---

## 🛠️ Production Deployment (Systemd Timer)

Run Nexus autonomously using native Linux systemd timers without any bash scripts:

Create `/etc/systemd/system/nexus.service`:
```ini
[Unit]
Description=Nexus Autonomous Self-Evolving Cycle
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/path/to/nexus
ExecStart=/path/to/nexus/target/release/nexus-train-hybrid 50 data/full_50m_stream.bin --layers 4 --batch-size 1
ExecStartPost=/path/to/nexus/target/release/nexus-consolidate 0.95 0.02 --layers 4
ExecStartPost=/path/to/nexus/target/release/nexus-eval data/full_50m_stream.bin --layers 4 --n-seqs 20
ExecStartPost=/path/to/nexus/target/release/nexus-generate "The fundamental law of intelligence states that" --layers 4 --tokens 30
```

Create `/etc/systemd/system/nexus.timer`:
```ini
[Unit]
Description=Run Nexus self-evolution cycle hourly

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start the timer:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nexus.timer
sudo systemctl list-timers | grep nexus
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
| `--model <P>` | `MODEL` | SmolLM2 | Foundation model directory |
| `--dataset <P>` | `DATASET` | `data/full_50m_stream.bin` | Packed binary dataset stream |
| `--backend <NAME>` | `NEXUS_BACKEND` | Auto-detected | Force compute backend (`cuda`, `rocm`, `vulkan`, `metal`) |

---

## 📄 License

Nexus Genesis is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.
