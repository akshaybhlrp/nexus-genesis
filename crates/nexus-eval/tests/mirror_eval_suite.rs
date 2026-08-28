//! Phase 1.5 "The Mirror" evaluation harness & metrics integration tests.
//!
//! Validates:
//! - Perplexity calculation: exp(mean cross-entropy loss).
//! - MoE Evaluation: routing entropy (-Σ p log p) and per-expert utilization mass.
//! - Retention Benchmark: no-forgetting retention metric formula.
//! - Success Metrics (from PLAN.md):
//!   * Expert entropy > 0.7 target.
//!   * Retention rate > 95% threshold.

use burn::tensor::Tensor;
use nexus_core::data::PackedDataset;
use nexus_core::model::LlamaConfig;
use nexus_core::moe::{RouterConfig, upcycle_dense};
use nexus_eval::{
    EvalReport, MoEEvalReport, compute_retention_rate, evaluate, evaluate_moe,
};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

type TestBackend = burn::backend::Wgpu;

fn device() -> burn::backend::wgpu::WgpuDevice {
    Default::default()
}

fn create_temp_dataset(n_seqs: usize, seq_len: usize, vocab_size: u32) -> (TempDir, Arc<PackedDataset>) {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("tokens.bin");

    let mut f = File::create(&path).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap(); // vocab sentinel
    f.write_all(&(seq_len as u32).to_le_bytes()).unwrap();
    f.write_all(&(n_seqs as u32).to_le_bytes()).unwrap();

    for i in 0..(n_seqs * seq_len) as u32 {
        let tok = ((i as u64 * 2654435761) % vocab_size.max(1) as u64) as u32;
        f.write_all(&tok.to_le_bytes()).unwrap();
    }
    f.sync_all().unwrap();

    let ds = Arc::new(PackedDataset::open(&path).unwrap());
    (dir, ds)
}

// =========================================================================
// 1. Retention Rate Metric Validation (PLAN.md Success Metric: > 95%)
// =========================================================================

#[test]
fn test_retention_rate_exact_preservation() {
    // Zero loss increase -> 100% retention
    let ret = compute_retention_rate(2.5, 2.5);
    assert_eq!(ret, 1.0);
}

#[test]
fn test_retention_rate_improvement_no_penalty() {
    // Model improved on old task (lower loss) -> 100% retention
    let ret = compute_retention_rate(3.0, 2.4);
    assert_eq!(ret, 1.0);
}

#[test]
fn test_retention_rate_slight_degradation() {
    // 5% increase in loss: baseline 2.0 -> new 2.1
    // degradation = (2.1 - 2.0)/2.0 = 0.05 -> retention = 0.95 (meets 95% target!)
    let ret = compute_retention_rate(2.0, 2.1);
    assert!((ret - 0.95).abs() < 1e-5);
}

#[test]
fn test_retention_rate_catastrophic_forgetting_clamps_to_zero() {
    // Loss doubled or more: baseline 1.5 -> new 4.0
    // degradation = (4.0 - 1.5)/1.5 = 1.666 -> retention = 1.0 - 1.666 = -0.666 -> clamped to 0.0
    let ret = compute_retention_rate(1.5, 4.0);
    assert_eq!(ret, 0.0);
}

#[test]
fn test_retention_rate_zero_baseline_safety() {
    // Baseline loss <= 0.0 edge case
    let ret = compute_retention_rate(0.0, 1.2);
    assert_eq!(ret, 1.0);
}

// =========================================================================
// 2. Report Formatting and Mathematical Invariants
// =========================================================================

#[test]
fn test_eval_report_formatting_and_math() {
    let report = EvalReport {
        mean_loss: 2.302585, // ln(10)
        perplexity: 10.0,
        n_seqs: 16,
    };

    let formatted = format!("{report}");
    assert!(formatted.contains("loss=2.3026"));
    assert!(formatted.contains("ppl=10.00"));
    assert!(formatted.contains("16 seqs"));
}

#[test]
fn test_moe_eval_report_formatting() {
    let moe_report = MoEEvalReport {
        mean_loss: 1.85,
        perplexity: (1.85f32).exp(),
        mean_entropy: 0.892,
        expert_masses: vec![vec![0.25, 0.25, 0.25, 0.25]],
        n_seqs: 32,
    };

    let formatted = format!("{moe_report}");
    assert!(formatted.contains("loss=1.8500"));
    assert!(formatted.contains("entropy=0.892"));
    assert!(formatted.contains("32 seqs"));
}

// =========================================================================
// 3. evaluate_moe: MoE Model Evaluation Harness
// =========================================================================

#[test]
fn test_evaluate_moe_computes_entropy_and_expert_masses() {
    let (_dir, ds) = create_temp_dataset(16, 32, 64);
    let dev = device();

    let dense = LlamaConfig::new(64, 32, 4, 1)
        .with_max_seq_len(32)
        .with_d_ff(64)
        .init::<TestBackend>(&dev);

    let router_cfg = RouterConfig::new(4);
    let moe = upcycle_dense(&dense, &router_cfg);

    let report = evaluate_moe(&moe, &ds, 0, 8, 4).expect("evaluate_moe should return report");

    assert_eq!(report.n_seqs, 8);
    assert!(report.mean_loss.is_finite() && report.mean_loss > 0.0);
    assert!((report.perplexity - report.mean_loss.exp()).abs() < 1e-4);

    // Initial uniform router gate with 4 experts has maximum entropy = ln(4) ≈ 1.386
    assert!(
        report.mean_entropy > 0.7,
        "Entropy ({}) must exceed 0.7 target",
        report.mean_entropy
    );

    // Expert mass distribution per block
    assert_eq!(report.expert_masses.len(), 1, "1 MoE block");
    assert_eq!(report.expert_masses[0].len(), 4, "4 experts");

    let sum_mass: f32 = report.expert_masses[0].iter().sum();
    assert!(
        (sum_mass - 1.0).abs() < 0.1,
        "Total routed mass per token should sum close to 1.0, got {sum_mass}"
    );
}

#[test]
fn test_evaluate_moe_out_of_bounds_returns_none() {
    let (_dir, ds) = create_temp_dataset(4, 16, 64);
    let dev = device();

    let dense = LlamaConfig::new(64, 32, 2, 1)
        .with_max_seq_len(16)
        .with_d_ff(32)
        .init::<TestBackend>(&dev);

    let moe = upcycle_dense(&dense, &RouterConfig::new(2));

    // Skip at exact dataset len
    assert!(evaluate_moe(&moe, &ds, 4, 2, 2).is_none());

    // Skip beyond dataset len
    assert!(evaluate_moe(&moe, &ds, 10, 2, 2).is_none());

    // Zero seqs requested
    assert!(evaluate_moe(&moe, &ds, 0, 0, 2).is_none());
}
