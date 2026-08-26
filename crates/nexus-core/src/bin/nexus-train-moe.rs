//! Phase 3 "The Heart": hybrid MoE training with EMNS mutation.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-train-moe -- [steps] [dataset.bin]

use nexus_core::hybrid::{train_hybrid, HybridConfig};
use nexus_core::model::LlamaConfig;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_core::stream::{packed_stream, synthetic_stream};
use std::sync::Arc;

type B = burn::backend::Autodiff<burn::backend::Wgpu>;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut args = std::env::args().skip(1);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let dataset_path = args.next();

    let (vocab_size, seq_len) = match &dataset_path {
        Some(p) => {
            let ds = nexus_core::data::PackedDataset::open(std::path::Path::new(p))
                .expect("open packed dataset");
            let sl = ds.seq_len;
            (50_257u32.max(ds.seq(0).iter().copied().max().unwrap_or(0) + 1) as usize, sl)
        }
        None => (1024, 128),
    };

    // 1. Build dense model.
    let cfg = LlamaConfig::new(vocab_size, 256, 8, 4)
        .with_max_seq_len(seq_len)
        .with_d_ff(512);
    tracing::info!(steps, params = cfg.num_params(), "building dense model");
    let device = Default::default();
    let dense = cfg.init::<B>(&device);

    // 2. Upcycle to MoE (4 experts per block, top-2 routing).
    let router_cfg = RouterConfig::new(4);
    let moe = upcycle_dense(&dense, &router_cfg);
    let n_experts = moe.blocks.first().map(|b| b.experts.len()).unwrap_or(4);
    let n_blocks = moe.blocks.len();
    tracing::info!(n_blocks, n_experts, "upcycled to MoE");

    // 3. Run hybrid training.
    let hybrid_cfg = HybridConfig::default();
    tracing::info!(?hybrid_cfg, "starting hybrid training");

    let (_model, metrics) = if let Some(p) = &dataset_path {
        let ds = Arc::new(
            nexus_core::data::PackedDataset::open(std::path::Path::new(p)).unwrap(),
        );
        train_hybrid(moe, packed_stream(ds), steps, 16, hybrid_cfg)
    } else {
        train_hybrid(
            moe,
            synthetic_stream(steps * 16, vocab_size as u32),
            steps,
            16,
            hybrid_cfg,
        )
    };

    // 4. Summary.
    if let (Some(first), Some(last)) = (metrics.first(), metrics.last()) {
        println!(
            "first_loss={:.4} last_loss={:.4} drop={:.1}%",
            first.loss,
            last.loss,
            (1.0 - last.loss / first.loss) * 100.0,
        );
        println!(
            "first_entropy={:.4} last_entropy={:.4}",
            first.mean_entropy, last.mean_entropy,
        );
        println!(
            "first_mu={:.6} last_mu={:.6}",
            first.mu, last.mu,
        );
    }
}
