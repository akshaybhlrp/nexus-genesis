//! Phase 1.5 "The Mirror": offline evaluation harness.
//!
//! Measures the plan's "life" metrics on a held-out packed dataset:
//! - perplexity (exp of mean CE loss)
//! - loss on held-out vs train tail (retention proxy until multi-domain data)

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::Module;
use burn::tensor::backend::Backend;
use nexus_core::model::Llama;
use nexus_core::stream::{LmBatcher, Sequence, lm_loss};

pub struct EvalReport {
    pub mean_loss: f32,
    pub perplexity: f32,
    pub n_seqs: usize,
}

impl std::fmt::Display for EvalReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "loss={:.4} ppl={:.2} ({} seqs)",
            self.mean_loss, self.perplexity, self.n_seqs
        )
    }
}

/// Evaluate on `n_seqs` sequences from `dataset`, skipping `skip` first ones
/// so eval never overlaps the training prefix of the same file.
pub fn evaluate<B: Backend>(
    model: &Llama<B>,
    dataset: &std::sync::Arc<nexus_core::data::PackedDataset>,
    skip: usize,
    n_seqs: usize,
    batch_size: usize,
) -> Option<EvalReport> {
    if skip >= dataset.len() {
        return None;
    }
    let end = (skip + n_seqs).min(dataset.len());
    let device = model.devices().first().cloned().unwrap_or_default();
    let batcher = LmBatcher::<B>::new();

    let mut total = 0f64;
    let mut count = 0usize;
    let mut start = skip;
    while start < end {
        let take = batch_size.min(end - start);
        let items: Vec<Sequence> = (start..start + take)
            .map(|i| Sequence { tokens: dataset.seq(i) })
            .collect();
        let batch = batcher.batch(items, &device);
        let logits = model.forward(batch.inputs);
        let loss = lm_loss(logits, batch.targets);
        let v = loss.into_data().iter::<f32>().next()?;
        total += v as f64 * take as f64; // seq-weighted; each seq same length
        count += take;
        start += take;
    }
    if count == 0 {
        return None;
    }
    let mean_loss = (total / count as f64) as f32;
    Some(EvalReport {
        mean_loss,
        perplexity: mean_loss.exp(),
        n_seqs: count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perplexity_is_exp_of_loss() {
        // Pure math sanity — no GPU needed.
        let r = EvalReport { mean_loss: 3.0, perplexity: 20.085, n_seqs: 4 };
        assert!((r.perplexity - r.mean_loss.exp()).abs() < 0.01);
    }
}
