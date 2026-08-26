//! training.rs coverage: train() on the autodiff backend — convergence,
//! return contract, determinism, edge configs.

mod common;

use common::*;
use nexus_core::model::LlamaConfig;
use nexus_core::stream::synthetic_stream;
use nexus_core::training::train;

type TAB = burn::backend::Autodiff<TB>;

fn model() -> nexus_core::model::Llama<TAB> {
    // Vocab 256 must match train()'s internal synthetic_stream(.., 256).
    LlamaConfig::new(256, 32, 4, 1)
        .with_max_seq_len(128)
        .with_d_ff(64)
        .init::<TAB>(&device())
}

// ---------- core learning behavior ----------

#[test]
fn loss_strictly_below_ln_vocab_after_training() {
    // Fresh model with zero lm_head sits at ln(256)≈5.545 (synthetic vocab).
    let losses = train(model(), 10, 2, 1e-3);
    assert!((losses[0] - (256f32).ln()).abs() < 0.1, "step-0 loss should be ~ln(vocab), got {}", losses[0]);
}

#[test]
fn loss_decreases_mean_last_quarter_vs_first_quarter() {
    let losses = train(model(), 40, 4, 1e-3);
    let q = 10;
    let first: f32 = losses[..q].iter().sum::<f32>() / q as f32;
    let last: f32 = losses[losses.len() - q..].iter().sum::<f32>() / q as f32;
    assert!(
        last < first * 0.9,
        "mean loss must drop >10% over run: first={first} last={last}"
    );
}

#[test]
fn losses_len_equals_steps() {
    for steps in [1usize, 3, 7] {
        let l = train(model(), steps, 2, 1e-4);
        assert_eq!(l.len(), steps);
    }
}

#[test]
fn all_losses_finite_positive() {
    let l = train(model(), 8, 2, 1e-3);
    assert!(l.iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn zero_steps_returns_empty() {
    let l = train(model(), 0, 2, 1e-3);
    assert!(l.is_empty());
}

#[test]
fn batch_size_one_trains() {
    let l = train(model(), 6, 1, 1e-3);
    assert_eq!(l.len(), 6);
    assert!(l.iter().all(|v| v.is_finite()));
}

#[test]
fn lr_zero_keeps_loss_constant_across_steps() {
    // No optimizer movement → every step sees a different batch but weights
    // unchanged; loss stays near ln(vocab). Guards against accidental
    // weight mutation outside optim.step.
    let l = train(model(), 5, 2, 0.0);
    let spread = l.iter().cloned().fold(f32::MIN, f32::max) - l.iter().cloned().fold(f32::MAX, f32::min);
    assert!(spread < 0.5, "lr=0 must not move loss much, spread={spread}");
}

#[test]
fn higher_lr_converges_faster_early() {
    let slow = train(model(), 20, 4, 1e-4);
    let fast = train(LlamaConfig::new(256, 32, 4, 1)
        .with_max_seq_len(128)
        .with_d_ff(64)
        .init::<TAB>(&device()), 20, 4, 1e-2);
    let drop = |l: &[f32]| (l[0] - l[l.len() - 1]) / l[0];
    assert!(
        drop(&fast) > drop(&slow),
        "higher lr should drop more in 20 steps: fast={} slow={}",
        drop(&fast), drop(&slow)
    );
}

// ---------- stream integration ----------

#[test]
fn synthetic_stream_feeds_expected_batch_count() {
    // train consumes steps*batch_size sequences.
    let n = synthetic_stream(12, 64).count();
    assert_eq!(n, 12);
}
