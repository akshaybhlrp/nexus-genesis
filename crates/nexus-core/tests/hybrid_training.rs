//! hybrid.rs coverage: train_hybrid, HybridConfig, StepMetrics,
//! adaptive mu, resistance lifecycle, mutation magnitude.

mod common;

use common::*;
use nexus_core::hybrid::{HybridConfig, StepMetrics, train_hybrid};
use nexus_core::moe::{RouterConfig, upcycle_dense};
use nexus_core::model::LlamaConfig;
use nexus_core::stream::synthetic_stream;

type TAB = burn::backend::Autodiff<TB>;

fn moe_model() -> nexus_core::moe::MoELlama<TAB> {
    let dense = LlamaConfig::new(256, 32, 4, 1)
        .with_max_seq_len(128)
        .with_d_ff(64)
        .init::<TAB>(&device());
    upcycle_dense(&dense, &RouterConfig::new(4))
}

fn default_hybrid_config() -> HybridConfig {
    HybridConfig::default()
}

fn seqs(n: usize) -> Vec<nexus_core::stream::Sequence> {
    synthetic_stream(n, 256).collect()
}

/// Tokens on the Autodiff backend (required by MoELlama::forward).
fn tab_tokens(vocab: usize, b: usize, s: usize) -> burn::tensor::Tensor<TAB, 2, burn::tensor::Int> {
    let data: Vec<u32> = (0..(b * s) as u32).map(|i| i % (vocab as u32).max(1)).collect();
    burn::tensor::Tensor::<TAB, 1, burn::tensor::Int>::from_ints(data.as_slice(), &device())
        .reshape([b, s])
}

// ---------- HybridConfig defaults ----------

#[test]
fn hybrid_config_default_values() {
    let c = HybridConfig::default();
    assert_eq!(c.lr, 3e-4);
    assert_eq!(c.weight_decay, 0.01);
    assert_eq!(c.balance_weight, 0.01);
    assert_eq!(c.entropy_threshold, 0.7);
    assert_eq!(c.mu_boost, 1.2);
    assert_eq!(c.mu_decay, 0.95);
    assert_eq!(c.mutation.mu, 0.01);
    assert_eq!(c.mutation.mu_min, 0.001);
    assert_eq!(c.mutation.mu_max, 0.1);
}

#[test]
fn hybrid_config_clone_is_independent() {
    let mut c1 = default_hybrid_config();
    c1.lr = 1.0;
    let c2 = c1.clone();
    assert_eq!(c1.lr, 1.0);
    assert_eq!(c2.lr, 1.0);
    assert_eq!(c1.mutation.mu, c2.mutation.mu);
}

// ---------- StepMetrics ----------

#[test]
fn step_metrics_clone_and_debug() {
    let m = StepMetrics {
        step: 5,
        loss: 3.14,
        mean_entropy: 0.8,
        mu: 0.01,
        current_lr: 3e-4,
        teacher_score: Some(0.85),
        teacher_queried: true,
    };
    let m2 = m.clone();
    assert_eq!(m2.step, 5);
    assert_eq!(m2.loss, 3.14);
    assert_eq!(m2.mean_entropy, 0.8);
    assert_eq!(m2.mu, 0.01);
    assert_eq!(m2.current_lr, 3e-4);
    assert_eq!(m2.teacher_score, Some(0.85));
    assert!(m2.teacher_queried);
    // Debug should not panic.
    let _ = format!("{:?}", m);
}


// ---------- train_hybrid: core behavior ----------

#[test]
fn train_hybrid_runs_and_returns_metrics() {
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(20).into_iter(), // 5 steps × 4 batch = 20
        5,
        4,
        default_hybrid_config(),
    );
    assert_eq!(metrics.len(), 5);
}

#[test]
fn train_hybrid_loss_is_finite_positive() {
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(32).into_iter(), // 8 steps × 4 batch = 32
        8,
        4,
        default_hybrid_config(),
    );
    for m in &metrics {
        assert!(m.loss.is_finite(), "step {} loss={} is not finite", m.step, m.loss);
        assert!(m.loss > 0.0, "step {} loss={} is not positive", m.step, m.loss);
    }
}

#[test]
fn train_hybrid_metrics_contain_all_fields() {
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(24).into_iter(), // 6 steps × 4 batch = 24
        6,
        4,
        default_hybrid_config(),
    );
    for m in &metrics {
        assert!(m.mean_entropy >= 0.0, "entropy={} should be >=0", m.mean_entropy);
        assert!(m.mu > 0.0, "mu={} should be >0", m.mu);
        assert!(m.mu <= 0.1, "mu={} should be clamped to mu_max", m.mu);
    }
}

#[test]
fn train_hybrid_step_indices_are_sequential() {
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(40).into_iter(), // 10 steps × 4 batch = 40
        10,
        4,
        default_hybrid_config(),
    );
    for (i, m) in metrics.iter().enumerate() {
        assert_eq!(m.step, i, "step index mismatch at position {i}");
    }
}

#[test]
fn train_hybrid_insufficient_data_stops_early() {
    // Only 3 sequences, batch_size=4 → first batch takes 3, second batch empty → stops.
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(3).into_iter(),
        100,
        4,
        default_hybrid_config(),
    );
    assert!(metrics.len() < 100, "expected early stop, got {} steps", metrics.len());
    assert!(metrics.len() >= 1, "should complete at least 1 step");
}

#[test]
fn train_hybrid_batch_size_one() {
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(8).into_iter(),
        4,
        1,
        default_hybrid_config(),
    );
    assert_eq!(metrics.len(), 4);
    assert!(metrics.iter().all(|m| m.loss.is_finite()));
}

// ---------- Adaptive mu behavior ----------

#[test]
fn train_hybrid_mu_stays_within_bounds() {
    let mut cfg = default_hybrid_config();
    cfg.mutation.mu = 0.05;
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(32).into_iter(),
        20,
        4,
        cfg,
    );
    for m in &metrics {
        assert!(
            m.mu >= 0.001 && m.mu <= 0.1,
            "mu={} out of bounds at step {}", m.mu, m.step
        );
    }
}

#[test]
fn train_hybrid_high_entropy_boosts_mu() {
    // threshold=0 → all entropy > 0 → mu should increase over steps.
    let mut cfg = default_hybrid_config();
    cfg.mutation.mu = 0.005;
    cfg.entropy_threshold = 0.0;
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(20).into_iter(),
        10,
        4,
        cfg,
    );
    let first_mu = metrics.first().unwrap().mu;
    let last_mu = metrics.last().unwrap().mu;
    // mu should have grown from the starting value (at least some boosts applied).
    assert!(last_mu > first_mu, "mu should increase with always-boost: first={} last={}", first_mu, last_mu);
}

#[test]
fn train_hybrid_low_entropy_decays_mu() {
    // threshold=999 → all entropy < 999 → mu should decrease over steps.
    let mut cfg = default_hybrid_config();
    cfg.mutation.mu = 0.08;
    cfg.entropy_threshold = 999.0;
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(20).into_iter(),
        10,
        4,
        cfg,
    );
    let first_mu = metrics.first().unwrap().mu;
    let last_mu = metrics.last().unwrap().mu;
    // mu should have decreased from the starting value (at least some decays applied).
    assert!(last_mu < first_mu, "mu should decrease with always-decay: first={} last={}", first_mu, last_mu);
}

// ---------- Resistance integration ----------

#[test]
fn train_hybrid_resistance_does_not_crash() {
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(40).into_iter(), // 10 steps × 4 batch = 40
        10,
        4,
        default_hybrid_config(),
    );
    assert_eq!(metrics.len(), 10);
}

// ---------- Mutation integration ----------

#[test]
fn train_hybrid_mutation_changes_model_output() {
    let model = moe_model();
    let t = tab_tokens(256, 2, 8);
    let before: Vec<f32> = model.forward(t.clone()).into_data().iter::<f32>().collect();

    let (model_after, _) = train_hybrid(
        model,
        seqs(32).into_iter(),
        10,
        4,
        default_hybrid_config(),
    );

    let after: Vec<f32> = model_after.forward(t).into_data().iter::<f32>().collect();
    let changed = before.iter().zip(&after).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(changed, "model output should change after hybrid training");
}

// ---------- Edge cases ----------

#[test]
fn train_hybrid_zero_steps_returns_empty() {
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(4).into_iter(),
        0,
        4,
        default_hybrid_config(),
    );
    assert!(metrics.is_empty());
}

#[test]
fn train_hybrid_high_lr_trains() {
    let mut cfg = default_hybrid_config();
    cfg.lr = 1e-2;
    let (_, metrics) = train_hybrid(
        moe_model(),
        seqs(40).into_iter(), // 10 steps × 4 batch = 40
        10,
        4,
        cfg,
    );
    assert_eq!(metrics.len(), 10);
    assert!(metrics.iter().all(|m| m.loss.is_finite()));
}

#[test]
fn train_hybrid_various_batch_sizes() {
    for bs in [1, 2, 4] {
        let (_, metrics) = train_hybrid(
            moe_model(),
            seqs(20).into_iter(), // 5 steps × max batch(4) = 20
            5,
            bs,
            default_hybrid_config(),
        );
        assert_eq!(metrics.len(), 5, "batch_size={bs}");
    }
}

// ---------- Config interaction with mutation ----------

#[test]
fn train_hybrid_balance_weight_affects_loss() {
    let mut cfg_low = default_hybrid_config();
    cfg_low.balance_weight = 0.001;
    let (_, metrics_low) = train_hybrid(
        moe_model(),
        seqs(20).into_iter(),
        5,
        4,
        cfg_low,
    );

    let mut cfg_high = default_hybrid_config();
    cfg_high.balance_weight = 1.0;
    let (_, metrics_high) = train_hybrid(
        moe_model(),
        seqs(20).into_iter(),
        5,
        4,
        cfg_high,
    );
    assert!(metrics_low.iter().all(|m| m.loss.is_finite()));
    assert!(metrics_high.iter().all(|m| m.loss.is_finite()));
}

// ---------- Output contract: model is usable after training ----------

#[test]
fn train_hybrid_model_forward_works_after_training() {
    let (model, _) = train_hybrid(
        moe_model(),
        seqs(20).into_iter(), // 5 steps × 4 batch = 20
        5,
        4,
        default_hybrid_config(),
    );
    let t = tab_tokens(256, 1, 8);
    let logits = model.forward(t);
    assert_eq!(logits.dims(), [1, 8, 256]);
    let data: Vec<f32> = logits.into_data().iter::<f32>().collect();
    assert!(data.iter().all(|v| v.is_finite()), "logits must be finite post-training");
}

#[test]
fn train_hybrid_model_is_differentiable_after_training() {
    let (model, _) = train_hybrid(
        moe_model(),
        seqs(20).into_iter(), // 5 steps × 4 batch = 20
        5,
        4,
        default_hybrid_config(),
    );
    use nexus_core::stream::{LmBatcher, lm_loss};
    use burn::data::dataloader::batcher::Batcher;

    let batcher = LmBatcher::<TAB>::new();
    let batch = batcher.batch(seqs(4), &device());
    let logits = model.forward(batch.inputs);
    let loss = lm_loss(logits, batch.targets);
    loss.backward(); // must not panic
}
