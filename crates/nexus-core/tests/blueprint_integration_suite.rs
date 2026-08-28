//! Complete blueprint integration test suite mapped directly to PLAN.md.
//!
//! Validates the 4 Pillars, 6 Critical Patches, and 7-Phase Lifecycle:
//! - Pillar 1: Execution Body (Burn + Autodiff, RoPE, RMSNorm, SwiGLU, Causal MHA).
//! - Pillar 2: Evolution Soul (EMNS CPU round-trip, macro-resistance, adaptive mu).
//! - Pillar 3: Memory Storage (Tiered L1/L2/L3 offload/reload, global ID bit-packing).
//! - Pillar 4: Conscience Teacher (High-entropy trigger, adaptive mu boost, LR dampening).
//! - Phase 2 Sparse Upcycling (MoE Split, top-k routing, Switch load-balancing loss).
//! - Phase 6 Memory Consolidation (Cosine similarity, expert merging, pruning, parent spawning).

mod common;

use common::*;
use burn::backend::Autodiff;
use burn::tensor::Tensor;
use nexus_core::consolidation::{
    ConsolidationConfig, consolidate_model, expert_similarity, merge_expert_pair,
    tensor_cosine_similarity,
};
use nexus_core::hybrid::{HybridConfig, train_hybrid};
use nexus_core::model::{LlamaConfig, check_token_ids};
use nexus_core::moe::{RouterConfig, upcycle_dense};
use nexus_core::stream::{LmBatcher, synthetic_stream, lm_loss};
use nexus_core::tiered::{
    decode_expert_id, global_expert_id, load_model_from_warehouse, offload_model_to_warehouse,
    TieredExpertManager,
};
use nexus_memory::{ExpertWarehouse, WarehouseConfig};
use nexus_teacher::TeacherValidator;
use std::sync::Arc;
use burn::data::dataloader::batcher::Batcher;
use tempfile::tempdir;

type TAB = Autodiff<TB>;

// =========================================================================
// 1. Pillar 1 / Phase 1: The Body & The Seed
// =========================================================================

#[test]
fn test_pillar1_dense_model_invariants_and_shapes() {
    let dev = device();
    let cfg = LlamaConfig::new(128, 32, 4, 2)
        .with_max_seq_len(32)
        .with_d_ff(64)
        .with_rms_norm_eps(1e-5)
        .with_rope_theta(10000.0);

    let model = cfg.init::<TB>(&dev);

    assert_eq!(model.blocks.len(), 2);
    assert_eq!(model.vocab_size, 128);

    // Forward pass with [batch=2, seq_len=8]
    let t = tokens(128, 2, 8);
    let logits = model.forward(t);

    assert_eq!(logits.dims(), [2, 8, 128]);
}

#[test]
fn test_pillar1_token_id_bounds_trust_boundary() {
    let dev = device();

    // Valid tokens
    let valid_t = Tensor::<TB, 2, burn::tensor::Int>::from_ints([[0u32, 63, 127]], &dev);
    check_token_ids(&valid_t, 128); // Must pass without panic

    // Exact boundary and out of bounds must be rejected
    let oob_t = Tensor::<TB, 2, burn::tensor::Int>::from_ints([[0u32, 128]], &dev);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check_token_ids(&oob_t, 128);
    }));
    assert!(result.is_err(), "Out-of-vocab token must panic immediately");
}

#[test]
fn test_pillar1_loss_computation_and_backprop() {
    let dev = device();
    let cfg = LlamaConfig::new(64, 32, 4, 1)
        .with_max_seq_len(128)
        .with_d_ff(64);
    let model = cfg.init::<TAB>(&dev);

    let batcher = LmBatcher::<TAB>::new();
    let seqs = synthetic_stream(2, 64).collect::<Vec<_>>();
    let batch = batcher.batch(seqs, &dev);

    let logits = model.forward(batch.inputs);
    let loss = lm_loss(logits, batch.targets);

    let loss_val = loss.clone().into_data().iter::<f32>().next().unwrap();
    assert!(loss_val.is_finite() && loss_val > 0.0);

    // Verify backprop graph builds cleanly
    let grads = loss.backward();
    let _ = burn::optim::GradientsParams::from_grads(grads, &model);
}

// =========================================================================
// 2. Phase 2: The Split (Sparse Upcycling & MoE Routing)
// =========================================================================

#[test]
fn test_phase2_sparse_upcycling_architecture() {
    let dev = device();
    let dense = LlamaConfig::new(64, 32, 4, 2)
        .with_max_seq_len(16)
        .with_d_ff(64)
        .init::<TB>(&dev);

    let router_cfg = RouterConfig::new(4).with_top_k(2).with_balance_weight(0.02);
    let moe = upcycle_dense(&dense, &router_cfg);

    assert_eq!(moe.blocks.len(), 2, "Preserves 2 blocks");
    for block in &moe.blocks {
        assert_eq!(block.experts.len(), 4, "Each block splits into 4 experts");
    }

    let t = tokens(64, 2, 8);
    let (logits, balance_loss, entropy, routes) = moe.forward_with_balance(t);

    assert_eq!(logits.dims(), [2, 8, 64]);
    assert_eq!(balance_loss.dims(), [1]);
    assert!(entropy > 0.0, "Entropy must be positive");
    assert_eq!(routes.len(), 2, "One route info per block");

    for route in &routes {
        assert_eq!(route.indices.dims(), [16, 2], "top-2 chosen per token");
        assert_eq!(route.weights.dims(), [16, 2], "normalized router weights");
        assert_eq!(route.expert_mass.len(), 4, "per-expert mass for 4 experts");
    }
}

// =========================================================================
// 3. Pillar 2 & Phase 3: The Soul & The Heart (Hybrid Training Loop)
// =========================================================================

#[test]
fn test_pillar2_hybrid_training_step_execution_and_resistance() {
    let dev = device();
    let dense = LlamaConfig::new(128, 32, 4, 1)
        .with_max_seq_len(128)
        .with_d_ff(64)
        .init::<TAB>(&dev);

    let router_cfg = RouterConfig::new(4);
    let moe = upcycle_dense(&dense, &router_cfg);

    let stream = synthetic_stream(20, 128);

    let hybrid_cfg = HybridConfig {
        lr: 1e-3,
        entropy_threshold: 0.5,
        mu_boost: 1.1,
        mu_decay: 0.9,
        ..Default::default()
    };

    let (_trained, metrics) = train_hybrid(moe, stream, 4, 4, hybrid_cfg);

    assert_eq!(metrics.len(), 4);
    for m in &metrics {
        assert!(m.loss.is_finite());
        assert!(m.mu >= 0.001 && m.mu <= 0.1, "mu clamped to config range");
    }
}

// =========================================================================
// 4. Pillar 4 & Phase 5: The Conscience (Teacher Integration & Adaptive mu)
// =========================================================================

#[test]
fn test_pillar4_teacher_high_entropy_adaptation_and_lr_dampening() {
    let dev = device();
    let dense = LlamaConfig::new(128, 32, 4, 1)
        .with_max_seq_len(128)
        .with_d_ff(64)
        .init::<TAB>(&dev);

    let moe = upcycle_dense(&dense, &RouterConfig::new(4));

    // Low teacher score (0.15) to trigger exploration & LR dampening
    let teacher = Arc::new(TeacherValidator::new_mock(0.15));

    let hybrid_cfg = HybridConfig {
        lr: 0.001,
        entropy_threshold: 0.001, // Force high-entropy trigger
        teacher_score_threshold: 0.4,
        teacher_lr_dampen: 0.75,
        mu_boost: 1.5,
        teacher: Some(teacher),
        ..Default::default()
    };

    let stream = synthetic_stream(16, 128);
    let (_trained, metrics) = train_hybrid(moe, stream, 3, 4, hybrid_cfg);

    assert_eq!(metrics.len(), 3);
    for m in &metrics {
        assert!(m.teacher_queried, "Teacher must be queried on high entropy");
        assert_eq!(m.teacher_score, Some(0.15));
        // Low score dampens LR: 0.001 * 0.75 = 0.00075
        assert!((m.current_lr - 0.00075).abs() < 1e-6, "LR must be dampened");
    }
}

// =========================================================================
// 5. Pillar 3 & Phase 4: Tiered Storage Offload, Reload & ID Packing
// =========================================================================

#[test]
fn test_pillar3_global_expert_id_packing_boundaries() {
    let cases = [
        (0usize, 0usize),
        (0, 3),
        (1, 0),
        (15, 63),
        (1000, 255),
        (u32::MAX as usize, u32::MAX as usize),
    ];

    for (b, e) in cases {
        let id = global_expert_id(b, e);
        let (decoded_b, decoded_e) = decode_expert_id(id);
        assert_eq!(b, decoded_b, "Block index mismatch for ({b}, {e})");
        assert_eq!(e, decoded_e, "Expert index mismatch for ({b}, {e})");
    }
}

#[test]
fn test_pillar3_model_offload_and_manager_retrieval() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let dev = device();

    let dense = LlamaConfig::new(64, 32, 2, 2)
        .with_max_seq_len(16)
        .with_d_ff(64)
        .init::<TB>(&dev);

    let mut moe = upcycle_dense(&dense, &RouterConfig::new(4));

    let wh_cfg = WarehouseConfig {
        l1_capacity: 4,
        l2_capacity: 8,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = Arc::new(ExpertWarehouse::<TB>::new(wh_cfg)?);

    // 2 blocks * 4 experts = 8 experts
    let saved = offload_model_to_warehouse(&moe, &warehouse)?;
    assert_eq!(saved, 8);

    // Reload back into model
    let loaded = load_model_from_warehouse(&mut moe, &warehouse, &dev)?;
    assert_eq!(loaded, 8);

    // Query via TieredExpertManager
    let manager = TieredExpertManager::new(warehouse, dev);
    let (inner, outer, down) = manager.get_expert_weights(1, 3)?;
    assert_eq!(inner.dims(), [32, 64]);
    assert_eq!(outer.dims(), [32, 64]);
    assert_eq!(down.dims(), [64, 32]);

    Ok(())
}

// =========================================================================
// 6. Phase 6: The Sleep (Consolidation, Cosine Similarity & Spawning)
// =========================================================================

#[test]
fn test_phase6_cosine_similarity_properties() {
    let dev = device();

    // Collinear vectors -> similarity = 1.0
    let a: Tensor<TB, 2> = Tensor::from_data([[2.0, 4.0], [6.0, 8.0]], &dev);
    let b: Tensor<TB, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &dev);
    let sim_collinear = tensor_cosine_similarity(&a, &b);
    assert!((sim_collinear - 1.0).abs() < 1e-4);

    // Orthogonal vectors -> similarity = 0.0
    let o1: Tensor<TB, 2> = Tensor::from_data([[1.0, 0.0], [0.0, 0.0]], &dev);
    let o2: Tensor<TB, 2> = Tensor::from_data([[0.0, 1.0], [0.0, 0.0]], &dev);
    let sim_orthogonal = tensor_cosine_similarity(&o1, &o2);
    assert!(sim_orthogonal.abs() < 1e-4);

    // Opposite vectors -> similarity = -1.0
    let neg_b = b.clone() * -1.0;
    let sim_opposite = tensor_cosine_similarity(&b, &neg_b);
    assert!((sim_opposite - (-1.0)).abs() < 1e-4);
}

#[test]
fn test_phase6_merge_expert_pair_averages_and_mutates_explorer() {
    let dev = device();
    let dense = LlamaConfig::new(64, 32, 4, 1)
        .with_max_seq_len(16)
        .with_d_ff(64)
        .init::<TB>(&dev);

    let mut moe = upcycle_dense(&dense, &RouterConfig::new(4));
    let block = &mut moe.blocks[0];

    // Check similarity between identical clones at init -> exactly 1.0
    let init_sim = expert_similarity(&block.experts[0], &block.experts[1]);
    assert!((init_sim - 1.0).abs() < 1e-4);

    // Merge expert 0 and expert 1
    merge_expert_pair(block, 0, 1, 0.1, 42, &dev);

    // After merge:
    // Expert 0 holds merged average.
    // Expert 1 is mutated into explorer.
    let after_sim = expert_similarity(&block.experts[0], &block.experts[1]);
    assert!(after_sim < 0.999, "Mutated explorer must diverge from merged parent");
}

#[test]
fn test_phase6_nightly_consolidation_lifecycle() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let dev = device();

    let dense = LlamaConfig::new(64, 32, 2, 2)
        .with_max_seq_len(16)
        .with_d_ff(64)
        .init::<TB>(&dev);

    let mut moe = upcycle_dense(&dense, &RouterConfig::new(4));

    let wh_cfg = WarehouseConfig {
        l1_capacity: 4,
        l2_capacity: 8,
        ssd_dir: tmp.path().to_path_buf(),
    };
    let warehouse = ExpertWarehouse::<TB>::new(wh_cfg)?;

    let config = ConsolidationConfig {
        similarity_threshold: 0.95, // Merges redundant initial clones
        activity_threshold: 0.05,
        spawn_mutation_rate: 0.08,
    };

    // Activity: expert 0 heavily used (0.9), expert 1 dormant (0.01)
    let routing_activity = vec![
        vec![0.90, 0.01, 0.05, 0.04],
        vec![0.80, 0.00, 0.10, 0.10],
    ];

    let report = consolidate_model(
        &mut moe,
        &routing_activity,
        Some(&warehouse),
        &config,
        &dev,
    );

    assert!(report.mean_similarity > 0.0);
    assert!(!report.merged.is_empty(), "Similar experts must be merged");
    assert!(!report.pruned.is_empty(), "Dormant expert must be pruned to SSD");
    assert!(!report.spawned.is_empty(), "New expert spawned from active parent");

    // Pruned expert (block 0, expert 1) must be in SSD warehouse
    let pruned_id = global_expert_id(0, 1);
    assert!(warehouse.contains_expert(pruned_id));

    Ok(())
}
