//! Phase 6: Nightly expert consolidation, merging, and pruning runner.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-consolidate -- [sim_threshold] [activity_threshold]

use nexus_core::consolidation::{consolidate_model, ConsolidationConfig};
use nexus_core::model::LlamaConfig;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_memory::{ExpertWarehouse, WarehouseConfig};

type B = burn::backend::Wgpu;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let sim_threshold: f32 = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.95);
    let activity_threshold: f32 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02);

    let config = ConsolidationConfig {
        similarity_threshold: sim_threshold,
        activity_threshold,
        spawn_mutation_rate: 0.05,
    };

    println!("=== Nexus Nightly Memory Consolidation ===");
    println!("Similarity Threshold: {:.2}", config.similarity_threshold);
    println!("Activity Threshold: {:.2}", config.activity_threshold);

    // Initialize an MoE architecture to consolidate
    let device = Default::default();
    let model_cfg = LlamaConfig::new(1024, 256, 8, 4)
        .with_max_seq_len(128)
        .with_d_ff(512);
    let dense = model_cfg.init::<B>(&device);
    let router_cfg = RouterConfig::new(4);
    let mut moe = upcycle_dense(&dense, &router_cfg);

    let wh_cfg = WarehouseConfig::default();
    let warehouse = ExpertWarehouse::<B>::new(wh_cfg).ok();

    // Simulated historical routing mass across 4 blocks x 4 experts
    let sample_activity = vec![
        vec![0.45, 0.40, 0.14, 0.01],
        vec![0.70, 0.00, 0.20, 0.10],
        vec![0.33, 0.33, 0.33, 0.01],
        vec![0.25, 0.25, 0.25, 0.25],
    ];

    let report = consolidate_model(
        &mut moe,
        &sample_activity,
        warehouse.as_ref(),
        &config,
        &device,
    );

    println!("\n--- Consolidation Results ---");
    println!("Mean Pairwise Similarity: {:.4}", report.mean_similarity);
    println!("Merged Redundant Pairs: {}", report.merged.len());
    for &(blk, e1, e2, sim) in &report.merged {
        println!("  - Block {blk}: merged expert {e2} into {e1} (similarity={sim:.4})");
    }

    println!("Pruned Dormant Experts: {}", report.pruned.len());
    for &(blk, exp, mass) in &report.pruned {
        println!("  - Block {blk}: pruned expert {exp} (activity={mass:.4}) to SSD");
    }

    println!("Spawned Mutated Experts: {}", report.spawned.len());
    for &(blk, parent, new_idx) in &report.spawned {
        println!("  - Block {blk}: spawned expert {new_idx} from parent {parent}");
    }

    println!("\nConsolidation pass complete. Ready for next training epoch.");
}
