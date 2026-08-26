//! nexus-eval coverage: EvalReport display, evaluate() over real packed
//! datasets — correctness of aggregation, skip/bounds handling, edge cases.

use nexus_eval::{EvalReport, evaluate};
use std::fs::File;
use std::io::Write;

// Local copy of fixture helpers (test crates can't share a `common` module
// across integration binaries cleanly without a shared dev-lib).
mod fixtures {
    use nexus_core::data::PackedDataset;
    use std::fs::File;
    use std::sync::Arc;
    use std::io::Write;
    use std::path::PathBuf;

    pub type TB = burn::backend::Wgpu;

    pub struct TempBin(pub tempfile::TempDir);
    impl TempBin {
        pub fn new() -> Self {
            Self(tempfile::tempdir().unwrap())
        }
        pub fn path(&self) -> PathBuf {
            self.0.path().join("tokens.bin")
        }
        pub fn write_valid(&self, n: usize, seq_len: usize, vocab: u32) -> PathBuf {
            let mut f = File::create(self.path()).unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
            f.write_all(&(seq_len as u32).to_le_bytes()).unwrap();
            f.write_all(&(n as u32).to_le_bytes()).unwrap();
            for i in 0..(n * seq_len) as u32 {
                let v = ((i as u64 * 2654435761) % vocab.max(1) as u64) as u32;
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            f.sync_all().unwrap();
            self.path()
        }
    }
    impl Default for TempBin {
        fn default() -> Self { Self::new() }
    }

    pub fn device() -> burn::backend::wgpu::WgpuDevice {
        Default::default()
    }

    pub fn tokens(vocab: usize, b: usize, s: usize) -> burn::tensor::Tensor<TB, 2, burn::tensor::Int> {
        let data: Vec<u32> =
            (0..(b * s) as u32).map(|i| i % (vocab as u32).max(1)).collect();
        burn::tensor::Tensor::<TB, 1, burn::tensor::Int>::from_ints(data.as_slice(), &device())
            .reshape([b, s])
    }

    pub fn tiny_model() -> nexus_core::model::Llama<TB> {
        nexus_core::model::LlamaConfig::new(64, 32, 4, 1)
            .with_max_seq_len(16)
            .with_d_ff(64)
            .init::<TB>(&device())
    }

    pub fn open_arc(p: &std::path::Path) -> Arc<PackedDataset> {
        Arc::new(PackedDataset::open(p).unwrap())
    }
}

use fixtures::*;

// ---------- EvalReport ----------

#[test]
fn report_display_format() {
    let r = EvalReport { mean_loss: 3.0, perplexity: 20.0855, n_seqs: 7 };
    let s = format!("{r}");
    assert!(s.contains("loss=3.0000"), "{s}");
    assert!(s.contains("ppl=20.09"), "{s}");
    assert!(s.contains("7 seqs"), "{s}");
}

#[test]
fn perplexity_field_is_exp_of_loss_by_construction() {
    // evaluate() must set ppl = loss.exp(); verify on a real run below and
    // here pin the invariant the Display assumes.
    let r = EvalReport { mean_loss: 0.0, perplexity: 1.0, n_seqs: 1 };
    assert!((r.perplexity - r.mean_loss.exp()).abs() < f32::EPSILON);
}

// ---------- evaluate(): happy paths ----------

#[test]
fn evaluate_returns_report_with_requested_counts() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(10, 16, 64));
    let m = tiny_model();
    let r = evaluate(&m, &ds, 0, 8, 4).expect("eval over in-bounds range");
    assert_eq!(r.n_seqs, 8);
    assert_eq!(r.perplexity, r.mean_loss.exp());
    assert!(r.mean_loss.is_finite());
}

#[test]
fn evaluate_skips_training_prefix() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(12, 16, 64));
    let m = tiny_model();
    let r = evaluate(&m, &ds, 4, 8, 4).expect("skip within bounds");
    assert_eq!(r.n_seqs, 8);
}

#[test]
fn evaluate_clamps_to_dataset_end() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(6, 16, 64));
    let m = tiny_model();
    // Ask for 100 seqs from a 6-seq dataset → all 6.
    let r = evaluate(&m, &ds, 0, 100, 4).expect("clamped eval");
    assert_eq!(r.n_seqs, 6);
}

#[test]
fn evaluate_partial_final_batch_counted() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(7, 16, 64));
    let m = tiny_model();
    // batch_size=4 over 5 seqs → batches of 4 then 1; total must be 5.
    let r = evaluate(&m, &ds, 2, 5, 4).expect("ragged tail");
    assert_eq!(r.n_seqs, 5);
}

// ---------- evaluate(): edge / boundary cases ----------

#[test]
fn evaluate_skip_at_exact_end_returns_none() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(5, 16, 64));
    let m = tiny_model();
    assert!(evaluate(&m, &ds, 5, 3, 2).is_none(), "skip==len ⇒ empty window");
}

#[test]
fn evaluate_skip_beyond_end_returns_none() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(5, 16, 64));
    let m = tiny_model();
    assert!(evaluate(&m, &ds, 50, 3, 2).is_none());
}

#[test]
fn evaluate_empty_dataset_returns_none() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(0, 8, 64));
    let m = tiny_model();
    assert!(evaluate(&m, &ds, 0, 4, 2).is_none());
}

#[test]
fn evaluate_single_seq_single_batch_minimal() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(1, 16, 64));
    let m = tiny_model();
    let r = evaluate(&m, &ds, 0, 1, 1).expect("single-seq eval");
    assert_eq!(r.n_seqs, 1);
    assert!(r.mean_loss.is_finite());
}

#[test]
fn evaluate_zero_n_seqs_returns_none() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(5, 16, 64));
    let m = tiny_model();
    // end == skip → loop never runs → count==0 → None.
    assert!(evaluate(&m, &ds, 0, 0, 2).is_none());
}

#[test]
fn evaluate_batch_size_larger_than_window_ok() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(4, 16, 64));
    let m = tiny_model();
    let r = evaluate(&m, &ds, 0, 2, 128).expect("oversized batch clamped");
    assert_eq!(r.n_seqs, 2);
}

// ---------- determinism + consistency ----------

#[test]
fn evaluate_deterministic_across_calls() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(8, 16, 64));
    let m = tiny_model();
    let a = evaluate(&m, &ds, 0, 8, 4).unwrap();
    let b = evaluate(&m, &ds, 0, 8, 4).unwrap();
    assert_eq!(a.mean_loss, b.mean_loss, "same weights+data ⇒ identical loss");
}

#[test]
fn evaluate_disjoint_windows_independent() {
    // Loss over [0..4) computed alone equals loss over [0..4) when followed by
    // another call — no hidden state carried across calls.
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(8, 16, 64));
    let m = tiny_model();
    let first_half = evaluate(&m, &ds, 0, 4, 4).unwrap().mean_loss;
    let _ignored = evaluate(&m, &ds, 4, 4, 4).unwrap();
    let again = evaluate(&m, &ds, 0, 4, 4).unwrap().mean_loss;
    assert_eq!(first_half, again);
}

// ---------- integration: train→eval pipeline shape ----------

#[test]
fn trained_model_beats_fresh_on_pseudostructured_data() {
    // Synthetic stream is arithmetic-structured; packed file built FROM that
    // stream should be learnable. Train briefly, eval both fresh + trained:
    // trained mean_loss < ln(vocab), and strictly lower than fresh.
    use burn::data::dataloader::batcher::Batcher;
    use nexus_core::stream::{LmBatcher, Sequence};

    type TAB = burn::backend::Autodiff<TB>;

    let t = TempBin::new();
    // Build dataset from the synthetic generator (vocab 64 to match model).
    {
        let mut f = File::create(t.path()).unwrap();
        f.write_all(&64u32.to_le_bytes()).unwrap();
        f.write_all(&128u32.to_le_bytes()).unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap();
        for s in nexus_core::stream::synthetic_stream(16, 64).take(16) {
            for tok in &s.tokens {
                f.write_all(&tok.to_le_bytes()).unwrap();
            }
        }
        f.sync_all().unwrap();
    }
    let ds = open_arc(&t.path());

    let cfg = nexus_core::model::LlamaConfig::new(64, 32, 4, 1)
        .with_max_seq_len(128)
        .with_d_ff(64);
    let fresh = cfg.init::<TB>(&device());

    let device_default = device();
    let mut trained = cfg.init::<TAB>(&device_default);
    {
        let batcher = LmBatcher::<TAB>::new();
        use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
        use burn::prelude::Module;
        let mut optim = AdamWConfig::new()
            .with_weight_decay(0.01)
            .init::<TAB, nexus_core::model::Llama<TAB>>();
        for chunk in nexus_core::stream::synthetic_stream(160, 64).collect::<Vec<_>>().chunks(8) {
            let seqs: Vec<Sequence> = chunk.to_vec();
            let batch = batcher.batch(seqs, &Default::default());
            let logits = trained.forward(batch.inputs);
            let loss = nexus_core::stream::lm_loss(logits, batch.targets);
            let grads = GradientsParams::from_grads(loss.backward(), &trained);
            trained = optim.step(3e-3, trained, grads);
        }
    }

    let l_fresh = evaluate(&fresh, &ds, 0, 8, 4).unwrap().mean_loss;
    let l_trained = evaluate(&trained, &ds, 0, 8, 4).unwrap().mean_loss;
    assert!(
        (l_fresh - (64f32).ln()).abs() < 0.2,
        "fresh zero-head model ≈ ln(vocab), got {l_fresh}"
    );
    assert!(
        l_trained < l_fresh * 0.9,
        "trained ({l_trained}) must beat fresh ({l_fresh})"
    );
}
