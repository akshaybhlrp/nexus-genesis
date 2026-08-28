use nexus_core::consolidation::{consolidate_model, ConsolidationConfig};

use nexus_core::model::LlamaConfig;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_memory::{ExpertWarehouse, WarehouseConfig};
use tempfile::tempdir;

type TestBackend = burn::backend::Wgpu;

#[test]
fn test_expert_similarity_and_merging() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let device = Default::default();
    let model_cfg = LlamaConfig::new(64, 32, 2, 2)
        .with_max_seq_len(32)
        .with_d_ff(64);
    let dense = model_cfg.init::<TestBackend>(&device);
    let router_cfg = RouterConfig::new(4);
    let mut moe = upcycle_dense(&dense, &router_cfg);

    let wh_cfg = WarehouseConfig {
        l1_capacity: 4,
        l2_capacity: 8,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = ExpertWarehouse::<TestBackend>::new(wh_cfg)?;

    let config = ConsolidationConfig {
        similarity_threshold: 0.85,
        activity_threshold: 0.05,
        spawn_mutation_rate: 0.05,
    };

    // Simulated usage: block 0 has dead expert 1 and active expert 0
    let activity = vec![
        vec![0.85, 0.01, 0.08, 0.06],
        vec![0.50, 0.30, 0.10, 0.10],
    ];

    let report = consolidate_model(
        &mut moe,
        &activity,
        Some(&warehouse),
        &config,
        &device,
    );

    assert!(report.mean_similarity > 0.0);
    assert!(!report.pruned.is_empty(), "Dormant expert should be pruned");
    assert!(!report.spawned.is_empty(), "New expert should be spawned from parent");

    // Verify pruned expert exists in L3 warehouse
    assert!(warehouse.contains_expert(nexus_core::tiered::global_expert_id(0, 1)));

    Ok(())
}
