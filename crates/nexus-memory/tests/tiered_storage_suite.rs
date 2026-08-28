//! Tiered storage and async prefetching integration tests.
//!
//! Validates:
//! - Pillar 3 (The Memory): L1 VRAM -> L2 CPU RAM -> L3 SSD hierarchy.
//! - Critical Patch 2: Async double buffer prefetcher with Tokio channels.
//! - LRU eviction policies at L1 and L2 cache boundaries.
//! - On-disk zstd compression + memmap2 recovery.
//! - Cache coherence and non-blocking asynchronous expert staging.

use burn::tensor::Tensor;
use nexus_memory::{
    AsyncPrefetcher, ExpertWarehouse, SerializedExpert, SerializedTensor, WarehouseConfig,
};
use std::sync::Arc;
use tempfile::tempdir;

type TestBackend = burn::backend::Wgpu;

fn device() -> burn::backend::wgpu::WgpuDevice {
    Default::default()
}

fn make_tensor(val: f32, dims: [usize; 2]) -> Tensor<TestBackend, 2> {
    let num = dims[0] * dims[1];
    let data = vec![val; num];
    Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(data, dims),
        &device(),
    )
}

// =========================================================================
// 1. SerializedTensor and SerializedExpert Roundtrips
// =========================================================================

#[test]
fn test_serialized_tensor_fidelity_various_shapes() {
    let dev = device();

    for shape in [[1, 1], [4, 8], [64, 32], [128, 1]] {
        let original: Tensor<TestBackend, 2> = make_tensor(3.14159, shape);
        let serialized = SerializedTensor::from_tensor(&original);

        assert_eq!(serialized.shape, shape.to_vec());
        assert_eq!(serialized.data.len(), shape[0] * shape[1]);

        let reconstructed: Tensor<TestBackend, 2> = serialized.to_tensor(&dev);
        assert_eq!(original.to_data(), reconstructed.to_data());
    }
}

#[test]
fn test_serialized_expert_construction_and_decomposition() {
    let dev = device();
    let inner = make_tensor(1.0, [32, 64]);
    let outer = make_tensor(2.0, [32, 64]);
    let down = make_tensor(3.0, [64, 32]);

    let expert = SerializedExpert::from_expert_weights(42, &inner, &outer, &down);

    assert_eq!(expert.id, 42);
    assert_eq!(expert.inner_weight.shape, vec![32, 64]);
    assert_eq!(expert.outer_weight.shape, vec![32, 64]);
    assert_eq!(expert.down_weight.shape, vec![64, 32]);

    let restored_inner: Tensor<TestBackend, 2> = expert.inner_weight.to_tensor(&dev);
    let restored_outer: Tensor<TestBackend, 2> = expert.outer_weight.to_tensor(&dev);
    let restored_down: Tensor<TestBackend, 2> = expert.down_weight.to_tensor(&dev);

    assert_eq!(inner.to_data(), restored_inner.to_data());
    assert_eq!(outer.to_data(), restored_outer.to_data());
    assert_eq!(down.to_data(), restored_down.to_data());
}

// =========================================================================
// 2. Pillar 3: Tiered Caching (L1 -> L2 -> L3) & LRU Eviction
// =========================================================================

#[test]
fn test_l1_cache_strict_capacity_and_eviction() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let config = WarehouseConfig {
        l1_capacity: 2, // Only 2 experts fit in VRAM
        l2_capacity: 10,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = ExpertWarehouse::<TestBackend>::new(config)?;
    let dev = device();

    // Persist 3 experts into warehouse
    for id in 0..3 {
        let t = make_tensor(id as f32, [4, 4]);
        warehouse.persist_expert(id, &t, &t, &t)?;
    }

    let (l1_len, l2_len) = warehouse.cache_stats();
    assert_eq!(l1_len, 2, "L1 cache must be strictly capped at capacity 2");
    assert_eq!(l2_len, 3, "L2 cache must contain all 3 experts");

    // Retrieve expert 0 (evicted from L1, but still in L2) -> promotes back to L1
    let (inner, _, _) = warehouse.get_expert(0, &dev)?;
    assert_eq!(inner.dims(), [4, 4]);

    let (l1_after, _) = warehouse.cache_stats();
    assert_eq!(l1_after, 2, "L1 cache remains capped at 2 after re-promoting");

    Ok(())
}

#[test]
fn test_l2_cache_eviction_spills_to_l3_ssd() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let config = WarehouseConfig {
        l1_capacity: 1,
        l2_capacity: 2, // Only 2 experts fit in CPU RAM
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = ExpertWarehouse::<TestBackend>::new(config)?;
    let dev = device();

    // Insert 4 experts sequentially
    for id in 0..4 {
        let t = make_tensor(id as f32, [4, 4]);
        warehouse.persist_expert(id, &t, &t, &t)?;
    }

    let (l1_len, l2_len) = warehouse.cache_stats();
    assert_eq!(l1_len, 1, "L1 holds 1 expert");
    assert_eq!(l2_len, 2, "L2 holds 2 experts");

    // All 4 experts must exist either in cache or on SSD
    for id in 0..4 {
        assert!(
            warehouse.contains_expert(id),
            "Expert {id} must be resident or persisted on L3 SSD"
        );
    }

    // Expert 0 was evicted from L2 to L3 SSD -> fetch from L3
    let (inner_0, _, _) = warehouse.get_expert(0, &dev)?;
    let data_0: Vec<f32> = inner_0.into_data().iter().collect();
    assert_eq!(data_0[0], 0.0, "Expert 0 data retrieved from L3 must match");

    Ok(())
}

#[test]
fn test_l3_zstd_compression_and_decompression_roundtrip() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let config = WarehouseConfig {
        l1_capacity: 2,
        l2_capacity: 4,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = ExpertWarehouse::<TestBackend>::new(config)?;

    let tensor_data = SerializedTensor {
        shape: vec![16, 16],
        data: (0..256).map(|i| (i as f32) * 0.01).collect(),
    };
    let expert = SerializedExpert {
        id: 999,
        inner_weight: tensor_data.clone(),
        outer_weight: tensor_data.clone(),
        down_weight: tensor_data,
    };

    // Write directly to L3
    warehouse.persist_to_l3(&expert)?;

    // Verify compressed file exists on disk
    let file_path = tmp.path().join("expert_999.bin");
    assert!(file_path.exists(), "Expert file must exist in L3 SSD dir");

    let file_size = std::fs::metadata(&file_path)?.len();
    assert!(file_size > 0, "Compressed file must have non-zero size");

    // Load back and verify payload
    let loaded = warehouse.load_from_l3(999)?;
    assert_eq!(loaded.id, 999);
    assert_eq!(loaded.inner_weight.shape, vec![16, 16]);
    assert_eq!(loaded.inner_weight.data, expert.inner_weight.data);

    Ok(())
}

#[test]
fn test_evict_all_to_l3_persists_entire_l2_cache() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let config = WarehouseConfig {
        l1_capacity: 2,
        l2_capacity: 8,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = ExpertWarehouse::<TestBackend>::new(config)?;

    for id in 10..15 {
        let tensor_data = SerializedTensor {
            shape: vec![2, 2],
            data: vec![id as f32; 4],
        };
        let expert = SerializedExpert {
            id,
            inner_weight: tensor_data.clone(),
            outer_weight: tensor_data.clone(),
            down_weight: tensor_data,
        };
        warehouse.put_l2(expert)?;
    }

    let evicted_count = warehouse.evict_all_to_l3()?;
    assert_eq!(evicted_count, 5);

    // Verify each file is written to SSD
    for id in 10..15 {
        let path = tmp.path().join(format!("expert_{id}.bin"));
        assert!(path.exists(), "expert_{id}.bin must be on disk");
    }

    Ok(())
}

// =========================================================================
// 3. Critical Patch 2: Async Double-Buffered Prefetcher
// =========================================================================

#[tokio::test]
async fn test_async_prefetcher_single_and_batch_requests() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let config = WarehouseConfig {
        l1_capacity: 4,
        l2_capacity: 8,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = Arc::new(ExpertWarehouse::<TestBackend>::new(config)?);
    let dev = device();

    // Populate 6 experts
    for id in 0..6 {
        let t = make_tensor(id as f32 + 10.0, [4, 4]);
        warehouse.persist_expert(id, &t, &t, &t)?;
    }

    let mut prefetcher = AsyncPrefetcher::new(Arc::clone(&warehouse), dev, 8);

    // Test single request
    prefetcher.request(0).await?;
    let res0 = prefetcher.recv().await.expect("Must receive expert 0");
    assert_eq!(res0.expert_id, 0);
    assert_eq!(res0.weights.0.dims(), [4, 4]);

    // Test batch requests
    prefetcher.request_many(&[1, 2, 3, 4, 5]).await?;

    let mut received_ids = Vec::new();
    for _ in 0..5 {
        if let Some(res) = prefetcher.recv().await {
            received_ids.push(res.expert_id);
        }
    }

    assert_eq!(received_ids, vec![1, 2, 3, 4, 5]);

    Ok(())
}

#[tokio::test]
async fn test_async_prefetcher_try_recv_non_blocking() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let config = WarehouseConfig {
        l1_capacity: 2,
        l2_capacity: 4,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = Arc::new(ExpertWarehouse::<TestBackend>::new(config)?);
    let dev = device();

    let t = make_tensor(1.0, [2, 2]);
    warehouse.persist_expert(100, &t, &t, &t)?;

    let mut prefetcher = AsyncPrefetcher::new(Arc::clone(&warehouse), dev, 4);

    // Initial try_recv on empty queue should return None
    assert!(prefetcher.try_recv().is_none());

    prefetcher.request(100).await?;

    // Wait slightly for async background worker to process
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let res = prefetcher.try_recv();
    assert!(res.is_some(), "try_recv should retrieve prefetched expert");
    assert_eq!(res.unwrap().expert_id, 100);

    Ok(())
}
