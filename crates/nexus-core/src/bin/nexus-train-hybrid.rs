//! Phase 5: Hybrid MoE training with external Teacher validation and Tiered memory support.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-train-hybrid -- [steps] [dataset.bin] [--with-teacher] [--tiered]

use nexus_core::hybrid::{train_hybrid, HybridConfig};
use nexus_core::model::LlamaConfig;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_core::stream::{packed_stream, synthetic_stream};
use nexus_core::tiered::offload_model_to_warehouse;
use nexus_memory::{ExpertWarehouse, WarehouseConfig};
use nexus_teacher::{TeacherConfig, TeacherValidator};
use std::path::Path;
use std::sync::Arc;

type B = burn::backend::Autodiff<burn::backend::Wgpu>;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut steps: usize = 100;
    let mut dataset_path: Option<String> = None;
    let mut with_teacher = false;
    let mut tiered = false;

    let mut batch_size: usize = 4;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--with-teacher" {
            with_teacher = true;
        } else if arg == "--tiered" {
            tiered = true;
        } else if arg == "--batch-size" && i + 1 < args.len() {
            i += 1;
            if let Ok(b) = args[i].parse::<usize>() {
                batch_size = b;
            }
        } else if let Ok(s) = arg.parse::<usize>() {
            steps = s;
        } else if arg.ends_with(".bin") {
            dataset_path = Some(arg.clone());
        }
        i += 1;
    }

    let (vocab_size, seq_len) = match &dataset_path {
        Some(p) => {
            let ds = nexus_core::data::PackedDataset::open(Path::new(p))
                .expect("open packed dataset");
            let sl = ds.seq_len;
            (
                50_257u32.max(ds.seq(0).iter().copied().max().unwrap_or(0) + 1) as usize,
                sl,
            )
        }
        None => (1024, 128),
    };

    // 1. Build dense model.
    let cfg = LlamaConfig::new(vocab_size, 256, 8, 4)
        .with_max_seq_len(seq_len)
        .with_d_ff(512);
    tracing::info!(steps, batch_size, params = cfg.num_params(), "building dense model");
    let device = Default::default();
    let dense = cfg.init::<B>(&device);

    // 2. Upcycle to MoE (4 experts per block, top-2 routing).
    let router_cfg = RouterConfig::new(4);
    let moe = upcycle_dense(&dense, &router_cfg);
    let n_experts = moe.blocks.first().map(|b| b.experts.len()).unwrap_or(4);
    let n_blocks = moe.blocks.len();
    tracing::info!(n_blocks, n_experts, "upcycled to MoE");

    // 3. Configure Teacher if requested.
    let teacher = if with_teacher {
        let teacher_cfg = TeacherConfig::from_env();
        tracing::info!(
            api_url = %teacher_cfg.api_url,
            model = %teacher_cfg.model,
            mock = teacher_cfg.mock_mode,
            "connected to external Teacher"
        );
        Some(Arc::new(TeacherValidator::new(teacher_cfg)))
    } else {
        None
    };

    let mut hybrid_cfg = HybridConfig::default();
    hybrid_cfg.teacher = teacher;
    tracing::info!(?hybrid_cfg, "starting hybrid training");

    // 4. Run hybrid training loop.
    let (trained_model, metrics) = if let Some(p) = &dataset_path {
        let ds = Arc::new(
            nexus_core::data::PackedDataset::open(Path::new(p)).unwrap(),
        );
        train_hybrid(moe, packed_stream(ds), steps, batch_size, hybrid_cfg)
    } else {
        train_hybrid(
            moe,
            synthetic_stream(steps * batch_size, vocab_size as u32),
            steps,
            batch_size,
            hybrid_cfg,
        )
    };

    // 5. Optionally offload model to tiered warehouse.
    if tiered {
        let wh_cfg = WarehouseConfig::default();
        if let Ok(warehouse) = ExpertWarehouse::<B>::new(wh_cfg) {
            match offload_model_to_warehouse(&trained_model, &warehouse) {
                Ok(count) => tracing::info!(count, "persisted experts to tiered L1/L2/L3 warehouse"),
                Err(e) => tracing::error!(error = %e, "failed to persist experts to warehouse"),
            }
        }
    }

    // 6. Summary.
    if let (Some(first), Some(last)) = (metrics.first(), metrics.last()) {
        let teacher_queries: usize = metrics.iter().filter(|m| m.teacher_queried).count();
        println!("\n=== Nexus Hybrid Training Summary ===");
        println!("Steps Completed: {}", metrics.len());
        println!(
            "Loss: {:.4} -> {:.4} ({:+.1}%)",
            first.loss,
            last.loss,
            (last.loss - first.loss) / first.loss * 100.0
        );
        println!(
            "Entropy: {:.4} -> {:.4}",
            first.mean_entropy, last.mean_entropy
        );
        println!("Mutation Rate (mu): {:.6} -> {:.6}", first.mu, last.mu);
        println!("Teacher Validations: {teacher_queries}");
    }
}
