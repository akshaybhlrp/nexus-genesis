//! Phase 3 hybrid training loop: backprop + EMNS mutation on the MoE model.
//!
//! Each step:
//! 1. Forward MoELlama → logits + balance loss
//! 2. CE loss + balance loss → backward → AdamW step
//! 3. Read router entropy from RouteInfo
//! 4. High entropy → boost mu (explore); low → decay mu (commit)
//! 5. Mutate expert weights via Mutator CPU round-trip
//! 6. Update per-expert resistance based on routing mass

use crate::moe::MoELlama;
use crate::stream::{LmBatcher, Sequence, lm_loss};
use burn::data::dataloader::batcher::Batcher;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Module;
use burn::tensor::backend::AutodiffBackend;
use nexus_emns::mutator::{MutationConfig, Mutator};

/// Hyperparameters for hybrid training.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub lr: f64,
    pub weight_decay: f32,
    /// Balance loss weight (Switch-Transformer convention).
    pub balance_weight: f32,
    /// Router entropy threshold for boosting mutation rate.
    pub entropy_threshold: f32,
    /// mu multiplier when entropy > threshold.
    pub mu_boost: f32,
    /// mu multiplier when entropy <= threshold.
    pub mu_decay: f32,
    pub mutation: MutationConfig,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            weight_decay: 0.01,
            balance_weight: 0.01,
            entropy_threshold: 0.7,
            mu_boost: 1.2,
            mu_decay: 0.95,
            mutation: MutationConfig::default(),
        }
    }
}

/// Per-step metrics for logging.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f32,
    pub mean_entropy: f32,
    pub mu: f32,
}

/// Run `steps` of hybrid backprop+mutation training on the MoE model.
///
/// Returns per-step metrics for the caller to log or plot.
pub fn train_hybrid<B>(
    model: MoELlama<B>,
    seqs: impl Iterator<Item = Sequence>,
    steps: usize,
    batch_size: usize,
    config: HybridConfig,
) -> (MoELlama<B>, Vec<StepMetrics>)
where
    B: AutodiffBackend,
{
    let device = model.devices().first().cloned().unwrap_or_default();
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init::<B, MoELlama<B>>();
    let batcher = LmBatcher::<B>::new();

    let n_blocks = model.blocks.len();
    let n_experts = model.blocks.first().map(|b| b.experts.len()).unwrap_or(4);
    let mut mutator = Mutator::new(config.mutation.clone(), n_blocks, n_experts);

    let mut model = model;
    let mut metrics = Vec::with_capacity(steps);
    let mut seq_iter = seqs;

    for step in 0..steps {
        let items: Vec<_> = (&mut seq_iter).take(batch_size).collect();
        if items.is_empty() {
            tracing::warn!(step, "ran out of data");
            break;
        }

        let batch = batcher.batch(items, &device);
        let targets = batch.targets;

        // 1. Forward with balance loss and routing info.
        let (logits, balance_loss, mean_entropy, routes) =
            model.forward_with_balance(batch.inputs);
        let ce_loss = lm_loss(logits, targets);
        let total_loss = ce_loss.clone()
            + balance_loss * burn::tensor::Tensor::from_floats(
                [config.balance_weight],
                &device,
            );

        let loss_val = ce_loss
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap_or(f32::INFINITY);

        // 2. Backward + optimizer step.
        let grads_params = GradientsParams::from_grads(total_loss.backward(), &model);
        model = optim.step(config.lr, model, grads_params);

        // 3. Update per-expert resistance from route mass in forward pass.
        for (blk_idx, route) in routes.iter().enumerate() {
            if let Some(res) = mutator.resistances.get_mut(blk_idx) {
                res.update(&route.expert_mass, config.mutation.resistance_decay);
            }
        }

        // 4. Adaptive mu based on router entropy.
        if mean_entropy > config.entropy_threshold {
            mutator.config.mu *= config.mu_boost;
        } else {
            mutator.config.mu *= config.mu_decay;
        }
        mutator.config.clamp_mu();

        // 5. Mutate experts via CPU round-trip.
        mutate_moe_experts(&mut model, &mut mutator);
        mutator.advance();

        let m = StepMetrics {
            step,
            loss: loss_val,
            mean_entropy,
            mu: mutator.config.mu,
        };

        if step % 10 == 0 || step == steps - 1 {
            tracing::info!(
                step = m.step,
                loss = m.loss,
                entropy = m.mean_entropy,
                mu = m.mu,
                "hybrid train"
            );
        }
        metrics.push(m);
    }

    (model, metrics)
}

/// Apply EMNS mutation to every expert in every MoE block.
fn mutate_moe_experts<B: AutodiffBackend>(
    model: &mut MoELlama<B>,
    mutator: &mut Mutator,
) {
    for (blk_idx, block) in model.blocks.iter_mut().enumerate() {
        let resistance = &mutator.resistances[blk_idx];
        for (exp_idx, expert) in block.experts.iter_mut().enumerate() {
            let r = resistance.values.get(exp_idx).copied().unwrap_or(0.5);
            let seed_offset = exp_idx + blk_idx * 1000;

            // Mutate the down projection weight.
            expert.down.weight = mutator.mutate_weight::<B>(
                expert.down.weight.clone(),
                seed_offset,
                r,
            );

            // SwiGlu has internal linear layers; access via the Module fields.
            // Burn's SwiGlu exposes `linear_inner` and `linear_outer` fields.
            expert.gate_up.linear_inner.weight = mutator.mutate_weight::<B>(
                expert.gate_up.linear_inner.weight.clone(),
                seed_offset + 100,
                r,
            );
            expert.gate_up.linear_outer.weight = mutator.mutate_weight::<B>(
                expert.gate_up.linear_outer.weight.clone(),
                seed_offset + 200,
                r,
            );
        }
    }
}

