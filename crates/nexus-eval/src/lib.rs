//! Phase 1.5 "The Mirror": offline evaluation harness.
//!
//! Measures the plan's "life" metrics on a held-out packed dataset:
//! - perplexity (exp of mean CE loss)
//! - loss on held-out vs train tail (retention proxy until multi-domain data)

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::Module;
use burn::tensor::backend::Backend;
use nexus_core::model::Llama;
use nexus_core::moe::MoELlama;
use nexus_core::stream::{LmBatcher, Sequence, lm_loss};

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct MoEEvalReport {
    pub mean_loss: f32,
    pub perplexity: f32,
    pub mean_entropy: f32,
    pub expert_masses: Vec<Vec<f32>>,
    pub n_seqs: usize,
}

impl std::fmt::Display for MoEEvalReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "loss={:.4} ppl={:.2} entropy={:.3} ({} seqs)",
            self.mean_loss, self.perplexity, self.mean_entropy, self.n_seqs
        )
    }
}

/// Compute retention score between an old baseline and new evaluation.
/// Retention Rate: 1.0 - max(0, (new_loss - baseline_loss) / baseline_loss).
pub fn compute_retention_rate(baseline_loss: f32, new_loss: f32) -> f32 {
    if baseline_loss <= 0.0 {
        return 1.0;
    }
    let degradation = (new_loss - baseline_loss) / baseline_loss;
    (1.0 - degradation.max(0.0)).clamp(0.0, 1.0)
}

/// Evaluate `MoELlama` model on `n_seqs` sequences from `dataset`.
pub fn evaluate_moe<B: Backend>(
    model: &MoELlama<B>,
    dataset: &std::sync::Arc<nexus_core::data::PackedDataset>,
    skip: usize,
    n_seqs: usize,
    batch_size: usize,
) -> Option<MoEEvalReport> {
    if skip >= dataset.len() {
        return None;
    }
    let end = (skip + n_seqs).min(dataset.len());
    let device = model.devices().first().cloned().unwrap_or_default();
    let batcher = LmBatcher::<B>::new();

    let mut total_loss = 0f64;
    let mut total_entropy = 0f64;
    let mut count = 0usize;
    let mut accumulated_masses: Option<Vec<Vec<f64>>> = None;

    let mut start = skip;
    while start < end {
        let take = batch_size.min(end - start);
        let items: Vec<Sequence> = (start..start + take)
            .map(|i| Sequence { tokens: dataset.seq(i) })
            .collect();
        let batch = batcher.batch(items, &device);
        let (logits, _, entropy, routes) = model.forward_with_balance(batch.inputs);
        let loss = lm_loss(logits, batch.targets);
        let v = loss.into_data().iter::<f32>().next()?;

        total_loss += v as f64 * take as f64;
        total_entropy += entropy as f64 * take as f64;

        if accumulated_masses.is_none() {
            accumulated_masses = Some(
                routes
                    .iter()
                    .map(|r| r.expert_mass.iter().map(|&m| m as f64 * take as f64).collect())
                    .collect(),
            );
        } else if let Some(ref mut acc) = accumulated_masses {
            for (blk_idx, r) in routes.iter().enumerate() {
                if let Some(blk_acc) = acc.get_mut(blk_idx) {
                    for (exp_idx, &m) in r.expert_mass.iter().enumerate() {
                        if let Some(slot) = blk_acc.get_mut(exp_idx) {
                            *slot += m as f64 * take as f64;
                        }
                    }
                }
            }
        }

        count += take;
        start += take;
    }

    if count == 0 {
        return None;
    }

    let mean_loss = (total_loss / count as f64) as f32;
    let mean_entropy = (total_entropy / count as f64) as f32;
    let expert_masses = accumulated_masses
        .map(|acc| {
            acc.into_iter()
                .map(|blk| blk.into_iter().map(|m| (m / count as f64) as f32).collect())
                .collect()
        })
        .unwrap_or_default();

    Some(MoEEvalReport {
        mean_loss,
        perplexity: mean_loss.exp(),
        mean_entropy,
        expert_masses,
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
