//! Phase 1 "The Seed": dense LLaMA training on the synthetic stream.
//!
//! Usage: cargo run --release -p nexus-core --bin nexus-train-dense -- [steps]

use burn::data::dataloader::batcher::Batcher;
use burn::optim::Optimizer;
use burn::prelude::Module;
use nexus_core::model::LlamaConfig;
use nexus_core::stream::{LmBatcher, packed_stream};
use std::sync::Arc;

type B = burn::backend::Autodiff<burn::backend::Wgpu>;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut args = std::env::args().skip(1);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let dataset_path = args.next(); // optional: path to packed tokens.bin

    // Tiny-but-honest config (~2M params); scale toward 0.1B after validation.
    let (vocab_size, seq_len) = match &dataset_path {
        Some(p) => {
            let ds = nexus_core::data::PackedDataset::open(std::path::Path::new(p))
                .expect("open packed dataset");
            let sl = ds.seq_len;
            (50_257u32.max(ds.seq(0).iter().copied().max().unwrap_or(0) + 1) as usize, sl)
        }
        None => (1024, 128), // synthetic fallback
    };

    let cfg = LlamaConfig::new(vocab_size, 256, 8, 4).with_max_seq_len(seq_len).with_d_ff(512);
    tracing::info!(steps, params = cfg.num_params(), ?dataset_path, "starting dense training");

    let device = Default::default();
    let model = cfg.init::<B>(&device);

    // ponytail: train() still consumes a synthetic stream internally; real-data
    // batches wired once train() takes an iterator instead of generating data.
    let losses = if let Some(p) = &dataset_path {
        let ds = Arc::new(nexus_core::data::PackedDataset::open(std::path::Path::new(p)).unwrap());
        train_streamed(model, packed_stream(ds), steps, 16, 3e-4, seq_len)
    } else {
        train_synthetic(model, steps)
    };

    let first = losses.first().copied().unwrap_or(f32::NAN);
    let last = losses.last().copied().unwrap_or(f32::NAN);
    println!("first={first:.4} last={last:.4} drop={:.1}%", (1.0 - last / first) * 100.0);
}

fn train_synthetic(model: nexus_core::model::Llama<B>, steps: usize) -> Vec<f32> {
    use nexus_core::training::train;
    train(model, steps, 16, 3e-4)
}

/// Thin adapter until training.rs is generalized to take any Sequence iterator.
fn train_streamed(
    model: nexus_core::model::Llama<B>,
    mut seqs: impl Iterator<Item = nexus_core::stream::Sequence>,
    steps: usize,
    batch_size: usize,
    lr: f64,
    _seq_len: usize,
) -> Vec<f32> {
    let device = model.devices().first().cloned().unwrap_or_default();
    let mut optim = burn::optim::AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, nexus_core::model::Llama<B>>();
    let batcher = LmBatcher::<B>::new();
    let mut model = model;
    let mut losses = Vec::with_capacity(steps);

    for step in 0..steps {
        let items: Vec<_> = (&mut seqs).take(batch_size).collect();
        if items.is_empty() {
            break;
        }
        let batch = batcher.batch(items, &device);
        let logits = model.forward(batch.inputs.clone());
        let loss = nexus_core::stream::lm_loss(logits, batch.targets);
        let value = loss
            .clone()
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap_or(f32::INFINITY);
        losses.push(value);
        let grads_params =
            burn::optim::GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads_params);
        if step % 10 == 0 || step == steps - 1 {
            tracing::info!(step, loss = value, "train");
        }
    }
    losses
}
