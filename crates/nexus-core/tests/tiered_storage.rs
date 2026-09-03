use nexus_core::model::LlamaConfig;

use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_core::tiered::{load_model_from_warehouse, offload_model_to_warehouse, TieredExpertManager};
use nexus_memory::{ExpertWarehouse, WarehouseConfig};
use std::sync::Arc;
use tempfile::tempdir;

type TestBackend = burn::backend::NdArray;

#[test]
fn test_tiered_model_persistence_and_restore() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let device = Default::default();
    let model_cfg = LlamaConfig::new(64, 32, 2, 2)
        .with_max_seq_len(32)
        .with_d_ff(64);
    let dense = model_cfg.init::<TestBackend>(&device);
    let router_cfg = RouterConfig::new(4);
    let mut moe = upcycle_dense(&dense, &router_cfg);

    let wh_cfg = WarehouseConfig {
        l1_capacity: 2,
        l2_capacity: 4,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = Arc::new(ExpertWarehouse::<TestBackend>::new(wh_cfg)?);

    // Offload all 2 blocks * 4 experts = 8 experts to warehouse
    let offloaded = offload_model_to_warehouse(&moe, &warehouse)?;
    assert_eq!(offloaded, 8);

    // Restore back from warehouse
    let restored = load_model_from_warehouse(&mut moe, &warehouse, &device)?;
    assert_eq!(restored, 8);

    // Test TieredExpertManager
    let manager = TieredExpertManager::new(warehouse, device);
    let (inner, outer, down) = manager.get_expert_weights(0, 2)?;
    assert_eq!(inner.dims(), [32, 64]);
    assert_eq!(outer.dims(), [32, 64]);
    assert_eq!(down.dims(), [64, 32]);

    Ok(())
}
