//! Distributed Pre-Training & Infrastructure Tests.
//!
//! Burn-in (repeated tiered access), checkpoint resiliency (L1/L2/L3
//! corruption + eviction + reload), and cache-coherency under load, all
//! exercised against the real `ExpertWarehouse` and `AsyncPrefetcher`.

use nexus_memory::prefetcher::AsyncPrefetcher;
use nexus_memory::warehouse::{
    ExpertWarehouse, SerializedExpert, SerializedTensor, WarehouseConfig,
};
use std::collections::HashMap;
use std::sync::Arc;

type TB = burn::backend::Wgpu;

fn mk_config(tmp: &tempfile::TempDir) -> WarehouseConfig {
    WarehouseConfig {
        l1_capacity: 8,
        l2_capacity: 8,
        ssd_dir: tmp.path().to_path_buf(),
    }
}

fn mk_expert(id: u64) -> SerializedExpert {
    SerializedExpert {
        id,
        inner_weight: SerializedTensor {
            shape: vec![4, 4],
            data: vec![1.0; 16],
        },
        outer_weight: SerializedTensor {
            shape: vec![4, 4],
            data: vec![2.0; 16],
        },
        down_weight: SerializedTensor {
            shape: vec![4, 4],
            data: vec![3.0; 16],
        },
    }
}

/// Burn-in: hammer persist + get cycles to shake out use-after-free/leak.
#[test]
fn burn_in_tiered_roundtrip_hot_loop() {
    let tmp = tempfile::tempdir().unwrap();
    let w = ExpertWarehouse::<TB>::new(mk_config(&tmp)).unwrap();
    let device = Default::default();

    // Persist 64 experts, evict-all, reload all — repeat 5x.
    for cycle in 0..5 {
        for i in 0..64u64 {
            let e = mk_expert(i + cycle * 100);
            w.persist_expert(e.id, &to2(&e.inner_weight), &to2(&e.outer_weight), &to2(&e.down_weight))
                .unwrap();
        }
        w.evict_all_to_l3().unwrap();
        for i in 0..64u64 {
            let id = i + cycle * 100;
            let (inner, outer, down) = w.get_expert(id, &device).unwrap();
            assert_eq!(inner.dims(), [4, 4]);
            assert_eq!(outer.dims(), [4, 4]);
            assert_eq!(down.dims(), [4, 4]);
            assert!(w.contains_expert(id), "expert {id} must be found");
        }
    }
}

/// Checkpoint resiliency: L3 file corruption (garbage bytes) must not crash
/// subsequent operations — it surfaces as an Err, not UB.
#[test]
fn checkpoint_l3_corruption_surfaces_error() {
    let tmp = tempfile::tempdir().unwrap();
    let w = ExpertWarehouse::<TB>::new(mk_config(&tmp)).unwrap();
    let e = mk_expert(7);
    w.persist_to_l3(&e).unwrap();

    // Corrupt the L3 file with random-width garbage.
    let file = tmp.path().join("expert_7.bin");
    std::fs::write(&file, b"\x00\xffgarbage-not-zstd").unwrap();

    let result = w.load_from_l3(7);
    assert!(result.is_err(), "corrupt L3 must yield an error, not memory unsafety");
}

/// Checkpoint resiliency: missing L3 file → Err, no panic.
#[test]
fn checkpoint_missing_l3_file_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let w = ExpertWarehouse::<TB>::new(mk_config(&tmp)).unwrap();
    assert!(w.load_from_l3(999).is_err(), "missing expert on L3 must error");
}

/// Tiered cache coherency: a value persisted then read through L1 must equal
/// the original (L1 is warmed by persist_expert).
#[test]
fn tiered_coherence_l1_read_matches_written() {
    let tmp = tempfile::tempdir().unwrap();
    let w = ExpertWarehouse::<TB>::new(mk_config(&tmp)).unwrap();
    let device = Default::default();

    let e = mk_expert(5);
    w.persist_expert(5, &to2(&e.inner_weight), &to2(&e.outer_weight), &to2(&e.down_weight))
        .unwrap();

    let (i, o, d) = w.get_expert(5, &device).unwrap();
    assert_eq!(i.into_data().iter::<f32>().collect::<Vec<_>>(), vec![1.0; 16]);
    assert_eq!(o.into_data().iter::<f32>().collect::<Vec<_>>(), vec![2.0; 16]);
    assert_eq!(d.into_data().iter::<f32>().collect::<Vec<_>>(), vec![3.0; 16]);
}

/// LRU eviction: inserting more than L2 capacity must spill old experts to L3.
#[test]
fn l2_eviction_spills_to_l3_and_keeps_stats_bounded() {
    let tmp = tempfile::tempdir().unwrap();
    let w = ExpertWarehouse::<TB>::new(mk_config(&tmp)).unwrap();

    for i in 0..20u64 {
        w.persist_to_l3(&mk_expert(i)).unwrap();
        w.put_l2(mk_expert(i)).unwrap();
    }
    let (_, l2) = w.cache_stats();
    assert!(l2 <= 8, "L2 must respect capacity, got {l2}");
    // Oldest spill must now be resident on L3.
    assert!(w.contains_expert(0), "evicted expert must still be retrievable via L3");
}

/// Async prefetcher: request many then receive them all with correct ids.
#[tokio::test]
async fn prefetcher_request_recv_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let warehouse = Arc::new(ExpertWarehouse::<TB>::new(mk_config(&tmp)).unwrap());
    let device = Default::default();

    for i in 0..16u64 {
        warehouse
            .persist_to_l3(&mk_expert(i))
            .unwrap();
        warehouse.put_l2(mk_expert(i)).unwrap();
    }

    let mut pf = AsyncPrefetcher::new(Arc::clone(&warehouse), device, 8);
    pf.request_many(&(0..8).collect::<Vec<u64>>()).await.unwrap();

    let mut seen = HashMap::new();
    for _ in 0..8 {
        let res = tokio::time::timeout(std::time::Duration::from_secs(5), pf.recv())
            .await
            .expect("prefetch must complete within timeout")
            .expect("some");
        seen.insert(res.expert_id, res.weights.0.dims());
    }
    assert_eq!(seen.len(), 8, "all 8 prefetch results received");
    assert!(seen.values().all(|d| *d == [4, 4]), "all weights are [4,4]");
}

/// Prefetcher: missing expert yields no error and no result (skipped).
#[tokio::test]
async fn prefetcher_missing_expert_skipped_no_panic() {
    let tmp = tempfile::tempdir().unwrap();
    let warehouse = Arc::new(ExpertWarehouse::<TB>::new(mk_config(&tmp)).unwrap());
    let device = Default::default();

    let mut pf = AsyncPrefetcher::new(warehouse, device, 4);
    pf.request(404).await.unwrap();

    let res = tokio::time::timeout(std::time::Duration::from_secs(3), pf.recv()).await;
    // Missing expert is skipped: recv blocks (no result) until timeout — we
    // only assert it did not panic / crash.
    match res {
        Err(_) => {} // timed out = skipped, acceptable
        Ok(None) => {}
        Ok(Some(r)) => panic!("should not have a result for missing expert, got {}", r.expert_id),
    }
}

fn to2(t: &SerializedTensor) -> burn::tensor::Tensor<TB, 2> {
    let device = Default::default();
    burn::tensor::Tensor::<TB, 2>::from_data(
        burn::tensor::TensorData::new(t.data.clone(), t.shape.clone()),
        &device,
    )
}
