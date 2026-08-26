//! Config + model construction/forward coverage. GPU (Wgpu) required.

mod common;

use common::*;
use nexus_core::model::{Llama, LlamaConfig};

// ---------- LlamaConfig: defaults ----------

#[test]
fn config_defaults() {
    let c = LlamaConfig::new(1000, 64, 4, 2);
    assert_eq!(c.max_seq_len, 1024);
    assert_eq!(c.d_ff, 1376);
    assert!((c.rms_norm_eps - 1e-5).abs() < f64::EPSILON);
    assert!((c.rope_theta - 10000.0).abs() < 1e-4);
}

#[test]
fn config_overrides_stick() {
    let c = LlamaConfig::new(100, 32, 2, 1)
        .with_max_seq_len(8)
        .with_d_ff(16)
        .with_rms_norm_eps(1e-3)
        .with_rope_theta(5000.0);
    assert_eq!(c.max_seq_len, 8);
    assert_eq!(c.d_ff, 16);
    assert!((c.rms_norm_eps - 1e-3).abs() < f64::EPSILON);
    assert!((c.rope_theta - 5000.0).abs() < 1e-3);
}

// ---------- num_params: hand-computed against the formula ----------

fn expected_params(vocab: usize, d: usize, _h: usize, l: usize, dff: usize) -> usize {
    let head = d * d;
    let ffn = 3 * d * dff + dff * d;
    vocab * d + l * (head + ffn + 2 * d)
}

#[test]
fn num_params_matches_formula() {
    for &(v, d, h, l, dff) in &[
        (256usize, 64usize, 4usize, 2usize, 128usize),
        (50_257, 768, 12, 12, 2048),
        (32, 16, 2, 1, 8),
    ] {
        let c = LlamaConfig::new(v, d, h, l).with_d_ff(dff);
        assert_eq!(c.num_params(), expected_params(v, d, h, l, dff), "{v},{d},{h},{l},{dff}");
    }
}

#[test]
fn num_params_scales_monotonically_per_axis() {
    // All axes pinned explicitly so defaults never mask direction.
    let base = LlamaConfig::new(100, 64, 4, 1).with_d_ff(128).num_params();
    assert!(LlamaConfig::new(200, 64, 4, 1).with_d_ff(128).num_params() > base); // vocab
    assert!(LlamaConfig::new(100, 128, 4, 1).with_d_ff(128).num_params() > base); // d_model
    assert!(LlamaConfig::new(100, 64, 4, 1).with_d_ff(256).num_params() > base); // d_ff
    assert!(LlamaConfig::new(100, 64, 4, 2).with_d_ff(128).num_params() > base); // layers
}

#[test]
fn num_params_zero_layers_is_embedding_only() {
    // n_layers=0 → pure embedding table.
    let c = LlamaConfig::new(1000, 64, 4, 0);
    assert_eq!(c.num_params(), 1000 * 64);
}

// ---------- init: panics ----------

#[test]
#[should_panic(expected = "d_model must be divisible by n_heads")]
fn init_panics_on_indivisible_heads() {
    // 30 % 4 != 0 — must hit the assert_eq! guard.
    LlamaConfig::new(64, 30, 4, 1)
        .with_max_seq_len(16)
        .with_d_ff(32)
        .init::<TB>(&device());
}

#[test]
fn init_accepts_divisible_heads() {
    // Control for the panic test: 32 % 4 == 0 must construct cleanly.
    let m = LlamaConfig::new(64, 32, 4, 1)
        .with_max_seq_len(16)
        .with_d_ff(32)
        .init::<TB>(&device());
    assert_eq!(m.forward(tokens(64, 1, 4)).dims(), [1, 4, 64]);
}

// ---------- init + forward shapes ----------

#[test]
fn init_builds_all_blocks() {
    let m = LlamaConfig::new(64, 32, 4, 3)
        .with_max_seq_len(16)
        .with_d_ff(64)
        .init::<TB>(&device());
    // Public surface only — verify via forward on varying batch.
    let logits = m.forward(tokens(64, 2, 8));
    assert_eq!(logits.dims(), [2, 8, 64]);
}

#[test]
fn forward_single_token_sequence() {
    let m = tiny_cfg().init::<TB>(&device());
    let logits = m.forward(tokens(64, 1, 1));
    assert_eq!(logits.dims(), [1, 1, 64]);
}

#[test]
fn forward_full_context_window() {
    let m = LlamaConfig::new(64, 32, 4, 1)
        .with_max_seq_len(16)
        .with_d_ff(64)
        .init::<TB>(&device());
    let logits = m.forward(tokens(64, 1, 16));
    assert_eq!(logits.dims(), [1, 16, 64]);
}

#[test]
#[should_panic]
fn forward_beyond_max_seq_len_panics_or_errors() {
    // RoPE built for max_seq_len=16; seq=17 must fail loudly, not silently wrap.
    let m = LlamaConfig::new(64, 32, 4, 1)
        .with_max_seq_len(16)
        .with_d_ff(64)
        .init::<TB>(&device());
    let _ = m.forward(tokens(64, 1, 17));
}

#[test]
#[should_panic(expected = "out of range")]
fn forward_rejects_out_of_vocab_token() {
    // Guard added to Llama::forward: OOB ids previously read garbage GPU
    // memory silently (burn's Embedding does not bounds-check). Now must
    // panic loudly naming the offending id and vocab bound.
    let m = tiny_cfg().init::<TB>(&device());
    let t = burn::tensor::Tensor::<TB, 1, burn::tensor::Int>::from_ints(
        [63u32, 999], // last valid + far OOB
        &device(),
    )
    .reshape([1, 2]);
    let _ = m.forward(t);
}

#[test]
#[should_panic(expected = "out of range")]
fn moe_forward_rejects_out_of_vocab_token() {
    // Same guard must hold through the MoE wrapper (guard carried via
    // upcycle_dense → MoELlama.vocab_size).
    use nexus_core::moe::{RouterConfig, upcycle_dense};
    let d = tiny_cfg().init::<TB>(&device());
    let moe = upcycle_dense(&d, &RouterConfig::new(4));
    let t = burn::tensor::Tensor::<TB, 1, burn::tensor::Int>::from_ints(
        [500u32],
        &device(),
    )
    .reshape([1, 1]);
    let _ = moe.forward(t);
}

#[test]
fn forward_accepts_max_valid_token() {
    // Boundary control: id == vocab-1 is legal and must pass the guard.
    let m = tiny_cfg().init::<TB>(&device());
    let t = burn::tensor::Tensor::<TB, 1, burn::tensor::Int>::from_ints([63u32], &device())
        .reshape([1, 1]);
    assert_eq!(m.forward(t).dims(), [1, 1, 64]);
}

#[test]
fn forward_batch_size_one_vs_many_consistent_shapes() {
    let m = tiny_cfg().init::<TB>(&device());
    for b in [1usize, 2, 4] {
        assert_eq!(m.forward(tokens(64, b, 8)).dims(), [b, 8, 64]);
    }
}

// ---------- determinism / statelessness of inference ----------

#[test]
fn forward_deterministic_across_calls() {
    let m = tiny_cfg().init::<TB>(&device());
    let t = tokens(64, 2, 8);
    let a = m.forward(t.clone()).into_data();
    let b = m.forward(t).into_data();
    let av = a.iter::<f32>().collect::<Vec<_>>();
    let bv = b.iter::<f32>().collect::<Vec<_>>();
    assert_eq!(av, bv, "same weights + same input must give same logits");
}

// ---------- forward_logits alias ----------

#[test]
fn forward_logits_matches_forward() {
    let m = tiny_cfg().init::<TB>(&device());
    let t = tokens(64, 2, 8);
    let a = m.forward_logits(t.clone()).into_data();
    let b = m.forward(t).into_data();
    let av: Vec<f32> = a.iter().collect();
    let bv: Vec<f32> = b.iter().collect();
    assert_eq!(av, bv);
}

// ---------- lm_head zero-init contract (Phase 1 quirk worth pinning) ----------

#[test]
fn fresh_lm_head_outputs_all_zero_logits() {
    // lm_head uses Initializer::Zeros — a fresh model must emit uniform zeros,
    // which pins the loss at ln(vocab) before any training step.
    let m = tiny_cfg().init::<TB>(&device());
    let data = m.forward(tokens(64, 1, 4)).into_data();
    let all_zero = data.iter::<f32>().all(|x| x == 0.0);
    assert!(all_zero, "fresh lm_head must be zero-initialized");
}

// ---------- module clone semantics ----------

#[test]
fn cloned_model_shares_weights_forward_identical() {
    let m = tiny_cfg().init::<TB>(&device());
    let t = tokens(64, 1, 8);
    let m2 = m.clone();
    let a: Vec<f32> = m.forward(t.clone()).into_data().iter().collect();
    let b: Vec<f32> = m2.forward(t).into_data().iter().collect();
    assert_eq!(a, b);
}

#[test]
fn devices_reports_at_least_one() {
    use burn::prelude::Module;
    let m: Llama<TB> = tiny_cfg().init(&device());
    assert!(!m.devices().is_empty());
}
