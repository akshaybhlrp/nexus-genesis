//! Phase 5: Hybrid MoE training with external Teacher validation and Tiered memory support.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-train-hybrid -- [steps] [dataset.bin] [--with-teacher] [--tiered]

use nexus_core::checkpoint::{CheckpointManager, CheckpointMeta};
use nexus_core::hybrid::{train_hybrid, HybridConfig};
use nexus_core::model::LlamaConfig;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_core::stream::{packed_stream_from, synthetic_stream};
use nexus_core::tiered::{load_model_from_warehouse, offload_model_to_warehouse};
use nexus_memory::{ExpertWarehouse, WarehouseConfig};
use nexus_teacher::{TeacherConfig, TeacherValidator};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::io::Write;

#[cfg(feature = "cuda")]
type B = burn::backend::Autodiff<burn::backend::Cuda>;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
type B = burn::backend::Autodiff<burn::backend::Rocm>;
#[cfg(all(feature = "vulkan", not(feature = "cuda"), not(feature = "rocm")))]
type B = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(all(feature = "metal", not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan")))]
type B = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal")))]
type B = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal"), not(feature = "wgpu")))]
type B = burn::backend::Autodiff<burn::backend::NdArray>;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut steps: usize = 100;
    let mut dataset_path: Option<String> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut checkpoint_dir = PathBuf::from("data/checkpoints");
    let mut with_teacher = false;
    let mut batch_size: usize = std::env::var("NEXUS_BATCH_SIZE")
        .ok()
        .and_then(|b| b.parse().ok())
        .unwrap_or(1);
    let mut n_experts: usize = 4;
    let mut max_layers: Option<usize> = std::env::var("NEXUS_LAYERS")
        .ok()
        .and_then(|l| l.parse().ok())
        .or(Some(4));
    let mut lr: f64 = 0.0003;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--with-teacher" {
            with_teacher = true;
        } else if arg == "--batch-size" && i + 1 < args.len() {
            i += 1;
            if let Ok(b) = args[i].parse::<usize>() {
                batch_size = b;
            }
        } else if arg == "--experts" && i + 1 < args.len() {
            i += 1;
            if let Ok(e) = args[i].parse::<usize>() {
                n_experts = e;
            }
        } else if arg == "--layers" && i + 1 < args.len() {
            i += 1;
            if let Ok(l) = args[i].parse::<usize>() {
                max_layers = if l == 0 { None } else { Some(l) };
            }
        } else if arg == "--lr" && i + 1 < args.len() {
            i += 1;
            if let Ok(val) = args[i].parse::<f64>() {
                lr = val;
            }
        } else if arg == "--model" && i + 1 < args.len() {
            i += 1;
            model_path = Some(PathBuf::from(&args[i]));
        } else if arg == "--checkpoint-dir" && i + 1 < args.len() {
            i += 1;
            checkpoint_dir = PathBuf::from(&args[i]);
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

    #[cfg(feature = "cuda")]
    let device = burn::backend::cuda::CudaDevice::default();
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    let device = burn::backend::rocm::RocmDevice::default();
    #[cfg(all(feature = "vulkan", not(feature = "cuda"), not(feature = "rocm")))]
    let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);
    #[cfg(all(feature = "metal", not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan")))]
    let device = burn::backend::wgpu::WgpuDevice::IntegratedGpu(0);
    #[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal")))]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal"), not(feature = "wgpu")))]
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    println!("⚡ Target Hardware Device: {:?}", device);

    let checkpoint_mgr = CheckpointManager::new(&checkpoint_dir).expect("init checkpoint manager");
    let mut start_step = 0usize;
    let mut dataset_cursor = 0usize;

    // 1. Build or load foundation model
    let (dense, is_resumed) = if checkpoint_mgr.has_checkpoint() {
        let meta = checkpoint_mgr.load_meta().expect("load checkpoint meta");
        start_step = meta.step;
        dataset_cursor = meta.dataset_cursor;
        println!("🔄 Resuming from checkpoint: step={}, dataset_cursor={}, previous_loss={:.4}",
            start_step, dataset_cursor, meta.loss);

        let dense = if let Some(ref mpath) = model_path {
            println!("🧠 Ingesting foundation architecture from: {}", mpath.display());
            nexus_core::import::import_hf_to_llama_scaled::<B>(mpath, &device, max_layers)
                .expect("Failed to load foundation model")
        } else {
            let cfg = LlamaConfig::new(vocab_size, 384, 12, 8)
                .with_max_seq_len(seq_len)
                .with_d_ff(1024);
            cfg.init_math_governed::<B>(42, &device)
        };
        (dense, true)
    } else if let Some(ref mpath) = model_path {
        println!("🧠 Loading foundation model weights from: {}", mpath.display());
        let dense = nexus_core::import::import_hf_to_llama_scaled::<B>(mpath, &device, max_layers)
            .expect("Failed to load foundation model");
        (dense, false)
    } else {
        let cfg = LlamaConfig::new(vocab_size, 384, 12, 8)
            .with_max_seq_len(seq_len)
            .with_d_ff(1024);
        tracing::info!(steps, batch_size, params = cfg.num_params(), "building dense model");
        (cfg.init_math_governed::<B>(42, &device), false)
    };

    // 2. Upcycle to MoE (Top-2 routing)
    let router_cfg = RouterConfig::new(n_experts);
    let mut moe = upcycle_dense(&dense, &router_cfg);

    if is_resumed {
        match checkpoint_mgr.load_model(moe.clone(), &device) {
            Ok(resumed_moe) => {
                moe = resumed_moe;
                println!("✓ Successfully loaded full model parameters from checkpoint!");
            }
            Err(e) => {
                eprintln!("⚠️ Warning: could not load model weights from checkpoint ({e}), using base model");
            }
        }
    }

    // Auto-resume existing weights from L3 SSD Warehouse if present
    let warehouse = ExpertWarehouse::<B>::new(WarehouseConfig::default()).ok();
    if let Some(wh) = &warehouse {
        if let Ok(count) = load_model_from_warehouse(&mut moe, wh, &device) {
            if count > 0 {
                tracing::info!(count, "resumed evolved expert weights from L3 SSD warehouse");
            }
        }
    }

    let actual_experts = moe.blocks.first().map(|b| b.experts.len()).unwrap_or(n_experts);
    let n_blocks = moe.blocks.len();
    tracing::info!(n_blocks, actual_experts, "upcycled to MoE");

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
    hybrid_cfg.lr = lr;
    hybrid_cfg.teacher = teacher;
    tracing::info!(?hybrid_cfg, "starting hybrid training");

    // 4. Run hybrid training loop with circular cursor stream.
    let (trained_model, metrics) = if let Some(p) = &dataset_path {
        let ds = Arc::new(
            nexus_core::data::PackedDataset::open(Path::new(p)).unwrap(),
        );
        let n_seqs = ds.len();
        let stream = packed_stream_from(ds, dataset_cursor);
        dataset_cursor = (dataset_cursor + steps * batch_size) % n_seqs;
        train_hybrid(moe, stream, steps, batch_size, hybrid_cfg)
    } else {
        train_hybrid(
            moe,
            synthetic_stream(steps * batch_size, vocab_size as u32),
            steps,
            batch_size,
            hybrid_cfg,
        )
    };

    // 5. Always persist model to tiered warehouse for continuous learning.
    if let Some(wh) = &warehouse {
        match offload_model_to_warehouse(&trained_model, wh) {
            Ok(count) => tracing::info!(count, "persisted evolved experts to tiered L1/L2/L3 warehouse"),
            Err(e) => tracing::error!(error = %e, "failed to persist experts to warehouse"),
        }
    }

    // 6. Summary, logging, and atomic checkpoint persistence.
    if let (Some(first), Some(last)) = (metrics.first(), metrics.last()) {
        let teacher_queries: usize = metrics.iter().filter(|m| m.teacher_queried).count();
        let drop_pct = (last.loss - first.loss) / first.loss * 100.0;
        println!("\n=== Nexus Hybrid Training Summary ===");
        println!("Steps Completed: {} (Total Epoch Steps: {})", metrics.len(), start_step + metrics.len());
        println!(
            "Loss: {:.4} -> {:.4} ({:+.1}%)",
            first.loss,
            last.loss,
            drop_pct
        );
        println!(
            "Entropy: {:.4} -> {:.4}",
            first.mean_entropy, last.mean_entropy
        );
        println!("Mutation Rate (mu): {:.6} -> {:.6}", first.mu, last.mu);
        println!("Teacher Validations: {teacher_queries}");

        let time_str = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let log_line = format!(
            "{{\"epoch_ts\": {}, \"steps\": {}, \"initial_loss\": {:.4}, \"final_loss\": {:.4}, \"drop_pct\": {:.2}, \"entropy\": {:.4}, \"mu\": {:.6}}}\n",
            time_str, metrics.len(), first.loss, last.loss, drop_pct, last.mean_entropy, last.mu
        );
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("data/training_progress.jsonl") {
            let _ = f.write_all(log_line.as_bytes());
        }

        // Persist full atomic model & metadata checkpoint
        let meta = CheckpointMeta {
            step: start_step + metrics.len(),
            dataset_cursor,
            loss: last.loss,
            mean_entropy: last.mean_entropy,
            timestamp_secs: time_str,
        };
        match checkpoint_mgr.save(&trained_model, &meta) {
            Ok(_) => println!("✓ Saved full model & state checkpoint to {}", checkpoint_dir.display()),
            Err(e) => eprintln!("⚠️ Failed to save checkpoint: {e}"),
        }
    }
}
