//! Load / performance / robustness suite. Timing assertions are generous
//! ceilings (CI machines vary); they catch gross regressions, not jitter.

mod common;

use common::*;
use burn::data::dataloader::batcher::Batcher;
use nexus_core::data::{HEADER_LEN, PackedDataset};
use nexus_core::model::LlamaConfig;
use std::time::Instant;

// ---------- throughput: forward pass ----------

#[test]
fn forward_throughput_small_batch() {
    let m = LlamaConfig::new(1024, 128, 8, 2)
        .with_max_seq_len(256)
        .with_d_ff(256)
        .init::<TB>(&device());
    let t = tokens(1024, 8, 256);

    // Warmup (kernel compile / shader cache).
    let _ = m.forward(t.clone());

    let start = Instant::now();
    let iters = 10;
    for _ in 0..iters {
        let logits = m.forward(t.clone());
        // Force readback so timing includes real compute, not queue submission.
        let _n = logits.into_data().iter::<f32>().count();
    }
    let per_call = start.elapsed().as_secs_f64() / iters as f64;
    assert!(
        per_call < 5.0,
        "forward on [8,256]×2-layer took {per_call:.3}s/call — regression?"
    );
}

#[test]
fn training_step_under_ceiling() {
    type TAB = burn::backend::Autodiff<TB>;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
    use burn::prelude::Module;
    use nexus_core::stream::{LmBatcher, lm_loss, synthetic_stream};

    let device = Default::default();
    let cfg = LlamaConfig::new(256, 64, 4, 2)
        .with_max_seq_len(128)
        .with_d_ff(128);
    let mut model = cfg.init::<TAB>(&device);
    let mut optim = AdamWConfig::new().with_weight_decay(0.01).init::<TAB, _>();
    let batcher = LmBatcher::<TAB>::new();
    let seqs: Vec<_> = synthetic_stream(16, 256).collect();

    let start = Instant::now();
    for chunk in seqs.chunks(4) {
        let batch = batcher.batch(chunk.to_vec(), &device);
        let logits = model.forward(batch.inputs);
        let loss = lm_loss(logits, batch.targets);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-3, model, grads);
    }
    let per_step = start.elapsed().as_secs_f64() / seqs.chunks(4).count() as f64;
    assert!(per_step < 5.0, "train step took {per_step:.3}s — regression?");
}

// ---------- load: large batches + long sequences ----------

#[test]
fn load_large_batch_forward() {
    let m = tiny_cfg().init::<TB>(&device());
    let out = m.forward(tokens(64, 256, 8));
    assert_eq!(out.dims(), [256, 8, 64]);
}

#[test]
fn load_many_layers_deep() {
    let m = LlamaConfig::new(64, 32, 4, 12)
        .with_max_seq_len(16)
        .with_d_ff(32)
        .init::<TB>(&device());
    assert_eq!(m.forward(tokens(64, 1, 8)).dims(), [1, 8, 64]);
}

#[test]
fn load_large_vocab_head() {
    // Real tokenizer vocab at small d_model — stresses the [*, *, V] logits.
    let m = LlamaConfig::new(50_257, 32, 4, 1)
        .with_max_seq_len(16)
        .with_d_ff(64)
        .init::<TB>(&device());
    assert_eq!(m.forward(tokens(50_257.max(1), 1, 8)).dims(), [1, 8, 50_257]);
}

#[test]
fn load_repeated_forwards_no_state_leak() {
    // 200 sequential forwards; results must remain identical — catches
    // accidental in-place mutation of shared weights.
    let m = tiny_cfg().init::<TB>(&device());
    let t = tokens(64, 1, 8);
    let reference: Vec<f32> = m.forward(t.clone()).into_data().iter().collect();
    for i in 0..200 {
        let now: Vec<f32> = m.forward(t.clone()).into_data().iter().collect();
        assert_eq!(now, reference, "iteration {i} diverged");
    }
}

// ---------- dataset mmap under load ----------

#[test]
fn dataset_thousand_seqs_random_access_ok() {
    let t = TempBin::new();
    let p = t.write_valid(1000, 64, 512);
    let ds = PackedDataset::open(&p).unwrap();
    let start = Instant::now();
    // Jump around the file — exercises mmap page faults.
    for i in (0..1000).step_by(37) {
        assert_eq!(ds.seq(i).len(), 64);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 30,
        "1000 random seq reads took {elapsed:?} — mmap path broken?"
    );
}

#[test]
fn dataset_large_file_header_boundary() {
    // ~1M tokens: big enough to exceed trivial page caches on CI runners,
    // small enough to write in milliseconds.
    let t = TempBin::new();
    let n = 4096usize;
    let sl = 256usize;
    let p = t.write_valid(n, sl, 1024);
    let ds = PackedDataset::open(&p).unwrap();
    assert_eq!(ds.len(), n);
    assert_eq!(ds.seq(n - 1).len(), sl);
}

// ---------- adversarial inputs ----------

#[test]
fn adversarial_max_u32_counts_in_header() {
    // Header claiming u32::MAX seqs with a matching-length file is impossible
    // to build here; instead verify open() rejects the claim fast.
    let t = TempBin::new();
    let mut b = Vec::with_capacity(HEADER_LEN);
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&(u32::MAX - 1).to_le_bytes());
    let p = t.write_raw(&b);
    let start = Instant::now();
    assert!(PackedDataset::open(&p).is_err());
    assert!(
        start.elapsed().as_millis() < 1000,
        "corrupt-header rejection must be O(1), not proportional to claim"
    );
}

#[test]
fn stress_synthetic_stream_long_run() {
    use nexus_core::stream::synthetic_stream;
    // 20k seqs × 128 tokens — generator must hold up (no drift, no panic).
    let total: usize = synthetic_stream(20_000, 256)
        .map(|s| s.tokens.len())
        .sum();
    assert_eq!(total, 20_000 * 128);
}

#[test]
fn stress_batcher_hetero_but_uniform_rows() {
    use nexus_core::stream::{LmBatcher, Sequence};
    use burn::data::dataloader::batcher::Batcher;
    let batcher = LmBatcher::<TB>::new();
    // Max realistic batch in one call.
    let seqs: Vec<Sequence> = (0..1024)
        .map(|i| Sequence { tokens: vec![(i % 61) as u32; 128] })
        .collect();
    let b = batcher.batch(seqs, &device());
    assert_eq!(b.inputs.dims(), [1024, 127]);
}
