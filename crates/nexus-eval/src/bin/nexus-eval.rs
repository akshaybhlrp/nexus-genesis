//! Offline evaluation and retention benchmark binary.
//!
//! Usage:
//!   cargo run --release -p nexus-eval --bin nexus-eval -- [dataset.bin] [--n-seqs 100]

use nexus_core::data::PackedDataset;
use nexus_core::model::LlamaConfig;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_eval::{compute_retention_rate, evaluate, evaluate_moe};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "cuda")]
type Backend = burn::backend::Cuda;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
type Backend = burn::backend::Rocm;
#[cfg(all(feature = "vulkan", not(feature = "cuda"), not(feature = "rocm")))]
type Backend = burn::backend::Wgpu;
#[cfg(all(feature = "metal", not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan")))]
type Backend = burn::backend::Wgpu;
#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal")))]
type Backend = burn::backend::Wgpu;
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal"), not(feature = "wgpu")))]
type Backend = burn::backend::NdArray;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let dataset_path = args
        .iter()
        .find(|a| a.ends_with(".bin"))
        .cloned()
        .unwrap_or_else(|| "data/fineweb.bin".to_string());

    let mut n_seqs = 20usize;
    let mut model_dir: Option<PathBuf> = None;
    let mut max_layers: Option<usize> = std::env::var("NEXUS_LAYERS")
        .ok()
        .and_then(|l| l.parse().ok())
        .or(Some(4));

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--n-seqs" && i + 1 < args.len() {
            i += 1;
            if let Ok(n) = args[i].parse() {
                n_seqs = n;
            }
        } else if args[i] == "--model" && i + 1 < args.len() {
            i += 1;
            model_dir = Some(PathBuf::from(&args[i]));
        } else if args[i] == "--layers" && i + 1 < args.len() {
            i += 1;
            if let Ok(l) = args[i].parse::<usize>() {
                max_layers = if l == 0 { None } else { Some(l) };
            }
        }
        i += 1;
    }

    println!("=== Nexus Mirror Offline Evaluation ===");
    println!("Hardware: NVIDIA T500 (CUDA Native)");
    println!("Dataset: {dataset_path}");
    println!("Eval Sequences: {n_seqs}");

    let ds_path = Path::new(&dataset_path);
    if !ds_path.exists() {
        eprintln!("Dataset file '{}' not found.", ds_path.display());
        std::process::exit(1);
    }

    let dataset = Arc::new(PackedDataset::open(ds_path)?);
    println!("Total sequences in dataset: {}", dataset.len());

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
    println!("Active Compute Device: {:?}", device);

    // 1. Build and evaluate Dense Baseline
    let (dense, moe) = if let Some(ref dir) = model_dir {
        println!("Loading pretrained foundation model from: {}", dir.display());
        let dense = nexus_core::import::import_hf_to_llama_scaled::<Backend>(dir, &device, max_layers)?;
        let router_cfg = RouterConfig::new(4);
        let mut moe = upcycle_dense(&dense, &router_cfg);

        // Auto-resume checkpoint if available
        let checkpoint_mgr = nexus_core::checkpoint::CheckpointManager::new("data/checkpoints").ok();
        if let Some(mgr) = checkpoint_mgr {
            if mgr.has_checkpoint() {
                if let Ok(loaded) = mgr.load_model(moe.clone(), &device) {
                    moe = loaded;
                    println!("✓ Resumed active trained weights from checkpoint for evaluation");
                }
            }
        }
        (dense, moe)
    } else {
        let vocab_size = 50_257usize;
        let seq_len = dataset.seq_len;
        let cfg = LlamaConfig::new(vocab_size, 384, 12, 8)
            .with_max_seq_len(seq_len)
            .with_d_ff(1024);
        let dense = cfg.init::<Backend>(&device);
        let router_cfg = RouterConfig::new(4);
        let moe = upcycle_dense(&dense, &router_cfg);
        (dense, moe)
    };

    let skip = dataset.len().saturating_sub(n_seqs);
    println!("\n[1. Evaluating Dense Model on Held-out Data (skip={skip})...]");
    let dense_report = evaluate(&dense, &dataset, skip, n_seqs, 2)
        .expect("dense evaluation failed");
    println!("  ✓ Dense Result: {dense_report}");

    // 2. Evaluate MoE Model
    println!("\n[2. Evaluating MoE Brain Structure...]");
    let moe_report = evaluate_moe(&moe, &dataset, skip, n_seqs, 2)
        .expect("moe evaluation failed");
    println!("  ✓ MoE Result: {moe_report}");

    // 3. Retention and specialisation
    let retention = compute_retention_rate(dense_report.mean_loss, moe_report.mean_loss);
    println!("\n[3. Life & Memory Metrics]");
    println!("  - Retention Rate: {:.1}%", retention * 100.0);
    println!("  - Mean Router Entropy: {:.4}", moe_report.mean_entropy);
    println!("  - Routing Mass per Block:");
    for (blk, mass) in moe_report.expert_masses.iter().enumerate() {
        let mass_str: Vec<String> = mass.iter().map(|m| format!("{:.3}", m)).collect();
        println!("    Block {blk}: [{}]", mass_str.join(", "));
    }

    println!("\n✓ Offline evaluation benchmark complete.");
    Ok(())
}
