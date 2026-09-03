use nexus_memory::{AsyncPrefetcher, ExpertWarehouse, WarehouseConfig};
use burn::tensor::Tensor;

use std::sync::Arc;
use tempfile::tempdir;

type TestBackend = burn::backend::NdArray;

#[tokio::test]
async fn test_warehouse_tiered_caching_and_eviction() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let config = WarehouseConfig {
        l1_capacity: 2,
        l2_capacity: 3,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = Arc::new(ExpertWarehouse::<TestBackend>::new(config)?);
    let device = Default::default();

    // Create 4 dummy experts
    for id in 0..4 {
        let t: Tensor<TestBackend, 2> = Tensor::from_data([[id as f32, 1.0], [0.0, 1.0]], &device);
        warehouse.persist_expert(id, &t, &t, &t)?;
    }

    let (l1, l2) = warehouse.cache_stats();
    assert!(l1 <= 2, "L1 cache should not exceed capacity 2, got {l1}");
    assert!(l2 <= 3, "L2 cache should not exceed capacity 3, got {l2}");

    // Check that all 4 experts are known (in L1/L2 or persisted on L3 SSD)
    for id in 0..4 {
        assert!(warehouse.contains_expert(id), "Expert {id} should exist in warehouse");
    }

    // Retrieve expert 0 (which may have been evicted to L3)
    let (inner, outer, down) = warehouse.get_expert(0, &device)?;
    assert_eq!(inner.dims(), [2, 2]);
    assert_eq!(outer.dims(), [2, 2]);
    assert_eq!(down.dims(), [2, 2]);

    // Test async prefetcher
    let mut prefetcher = AsyncPrefetcher::new(Arc::clone(&warehouse), device, 4);
    prefetcher.request_many(&[1, 2, 3]).await?;

    let mut received = 0;
    for _ in 0..3 {
        if let Some(res) = prefetcher.recv().await {
            assert!(res.expert_id <= 3);
            received += 1;
        }
    }
    assert_eq!(received, 3);

    Ok(())
}
