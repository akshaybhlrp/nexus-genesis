//! Offline evaluation and retention benchmark binary.
//!
//! Usage:
//!   cargo run --release -p nexus-eval --bin nexus-eval -- [dataset.bin] [--n-seqs 100]

use nexus_core::data::PackedDataset;
use nexus_core::model::LlamaConfig;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_eval::{compute_retention_rate, evaluate, evaluate_moe};
use std::path::Path;
use std::sync::Arc;

type Backend = burn::backend::Wgpu;

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
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--n-seqs" && i + 1 < args.len() {
            i += 1;
            if let Ok(n) = args[i].parse() {
                n_seqs = n;
            }
        }
        i += 1;
    }

    println!("=== Nexus Mirror Offline Evaluation ===");
    println!("Dataset: {dataset_path}");
    println!("Eval Sequences: {n_seqs}");

    let ds_path = Path::new(&dataset_path);
    if !ds_path.exists() {
        eprintln!("Dataset file '{}' not found.", ds_path.display());
        std::process::exit(1);
    }

    let dataset = Arc::new(PackedDataset::open(ds_path)?);
    println!("Total sequences in dataset: {}", dataset.len());

    let vocab_size = 50_257usize;
    let seq_len = dataset.seq_len;

    // 1. Build and evaluate Dense Baseline
    let cfg = LlamaConfig::new(vocab_size, 256, 8, 4)
        .with_max_seq_len(seq_len)
        .with_d_ff(512);
    let device = Default::default();
    let dense = cfg.init::<Backend>(&device);

    let skip = dataset.len().saturating_sub(n_seqs);
    println!("\n[1. Evaluating Dense Model on Held-out Data (skip={skip})...]");
    let dense_report = evaluate(&dense, &dataset, skip, n_seqs, 2)
        .expect("dense evaluation failed");
    println!("  ✓ Dense Result: {dense_report}");

    // 2. Upcycle and evaluate MoE Model
    println!("\n[2. Evaluating MoE Brain Structure...]");
    let router_cfg = RouterConfig::new(4);
    let moe = upcycle_dense(&dense, &router_cfg);
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
