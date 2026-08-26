//! Phase 1 training loop: plain backprop on the dense model.
//!
//! Hand-rolled step loop instead of burn-train's Learner — full visibility
//! into where the EMNS mutation (Phase 3) and teacher hooks will slot in.

use crate::stream::{LmBatcher, Sequence, lm_loss, synthetic_stream};
use burn::data::dataloader::batcher::Batcher;
use crate::model::Llama;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Module;
use burn::tensor::backend::AutodiffBackend;

/// Run `steps` training steps. Returns per-step losses (CPU f32).
pub fn train<B>(model: Llama<B>, steps: usize, batch_size: usize, lr: f64) -> Vec<f32>
where
    B: AutodiffBackend,
{
    let device = model.devices().first().cloned().unwrap_or_default();
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, Llama<B>>();
    let batcher = LmBatcher::<B>::new();
    // Deterministic stream → distinct batches per step.
    let batches: Vec<Vec<Sequence>> = {
        let all: Vec<Sequence> = synthetic_stream(steps * batch_size, 256).collect();
        all.chunks(batch_size).map(<[Sequence]>::to_vec).collect()
    };

    let mut model = model;
    let mut losses = Vec::with_capacity(steps);

    for (step, seqs) in batches.into_iter().enumerate() {
        let batch = batcher.batch(seqs, &device);
        let targets = batch.targets;

        let logits = model.forward(batch.inputs);
        let loss = lm_loss(logits, targets);
        let value = loss
            .clone()
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap_or(f32::INFINITY);
        losses.push(value);

        // ponytail: fixed LR — no scheduler until real FineWeb run needs it.
        let grads_params = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads_params);

        if step % 10 == 0 || step == steps - 1 {
            tracing::info!(step, loss = value, "train");
        }
    }
    losses
}

#[cfg(test)]
mod tests {
    use super::*;
    type TB = burn::backend::Wgpu;
    type TAB = burn::backend::Autodiff<TB>;

    #[test]
    fn loss_decreases_on_synthetic_data() {
        let device = Default::default();
        let cfg = crate::model::LlamaConfig::new(256, 64, 4, 2)
            .with_max_seq_len(128)
            .with_d_ff(128);
        let model = cfg.init::<TAB>(&device);
        let losses = train(model, 30, 4, 1e-3);
        let first = losses.first().copied().unwrap();
        let last = losses.last().copied().unwrap();
        tracing::info!(first, last, "loss delta");
        assert!(
            last < first * 0.95,
            "expected loss to drop, first={first} last={last}"
        );
    }
}
