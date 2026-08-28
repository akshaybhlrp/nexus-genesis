//! Phase 6 "The Sleep": Nightly memory consolidation, pruning, merging, and expert spawning.
//!
//! Maintains long-term capacity and prevents catastrophic forgetting by:
//! 1. Merging redundant/highly-correlated experts (cosine similarity >= threshold).
//! 2. Pruning/offloading dormant experts to L3 SSD cold storage.
//! 3. Spawning mutated experts from top performers to colonize new domains.

use crate::moe::{Expert, MoEBlock, MoELlama};
use burn::module::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use nexus_memory::ExpertWarehouse;


/// Hyperparameters for nightly consolidation pass.
#[derive(Clone, Debug)]
pub struct ConsolidationConfig {
    /// Cosine similarity threshold above which two experts are merged.
    pub similarity_threshold: f32,
    /// Minimum routing mass fraction required to keep an expert active in hot memory.
    pub activity_threshold: f32,
    /// Mutation rate applied when spawning a new expert from a parent.
    pub spawn_mutation_rate: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.95,
            activity_threshold: 0.02,
            spawn_mutation_rate: 0.05,
        }
    }
}

/// Detailed outcome of a consolidation pass.
#[derive(Clone, Debug, Default)]
pub struct ConsolidationReport {
    /// Pairs of experts merged: (block_idx, kept_idx, merged_idx, similarity).
    pub merged: Vec<(usize, usize, usize, f32)>,
    /// Experts pruned to cold storage: (block_idx, expert_idx, activity).
    pub pruned: Vec<(usize, usize, f32)>,
    /// Experts spawned from parents: (block_idx, parent_idx, new_idx).
    pub spawned: Vec<(usize, usize, usize)>,
    /// Mean pairwise similarity across all expert pairs.
    pub mean_similarity: f32,
}

/// Compute cosine similarity between two 2D weight tensors.
pub fn tensor_cosine_similarity<B: Backend>(t1: &Tensor<B, 2>, t2: &Tensor<B, 2>) -> f32 {
    let d1: Vec<f32> = t1.to_data().iter::<f32>().collect();
    let d2: Vec<f32> = t2.to_data().iter::<f32>().collect();

    if d1.len() != d2.len() || d1.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm1 = 0.0f64;
    let mut norm2 = 0.0f64;

    for (v1, v2) in d1.iter().zip(d2.iter()) {
        let a = *v1 as f64;
        let b = *v2 as f64;
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }

    let denom = (norm1.sqrt() * norm2.sqrt()).max(1e-12);
    (dot / denom) as f32
}

/// Measure total similarity between two experts across all projection matrices.
pub fn expert_similarity<B: Backend>(e1: &Expert<B>, e2: &Expert<B>) -> f32 {
    let s_inner = tensor_cosine_similarity(
        &e1.gate_up.linear_inner.weight.val(),
        &e2.gate_up.linear_inner.weight.val(),
    );
    let s_outer = tensor_cosine_similarity(
        &e1.gate_up.linear_outer.weight.val(),
        &e2.gate_up.linear_outer.weight.val(),
    );
    let s_down = tensor_cosine_similarity(
        &e1.down.weight.val(),
        &e2.down.weight.val(),
    );

    (s_inner + s_outer + s_down) / 3.0
}

/// Average two weight tensors element-wise.
fn average_tensors<B: Backend>(
    t1: &Tensor<B, 2>,
    t2: &Tensor<B, 2>,
    device: &B::Device,
) -> Tensor<B, 2> {
    let d1: Vec<f32> = t1.to_data().iter::<f32>().collect();
    let d2: Vec<f32> = t2.to_data().iter::<f32>().collect();
    let shape = t1.to_data().shape.to_vec();

    let merged_data: Vec<f32> = d1
        .iter()
        .zip(d2.iter())
        .map(|(a, b)| (a + b) * 0.5)
        .collect();

    Tensor::from_data(TensorData::new(merged_data, shape), device)
}

/// Mutate tensor with gaussian noise for spawning new experts.
fn mutate_tensor_for_spawn<B: Backend>(
    t: &Tensor<B, 2>,
    mutation_rate: f32,
    seed: u32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let raw: Vec<f32> = t.to_data().iter::<f32>().collect();
    let shape = t.to_data().shape.to_vec();

    let mutated: Vec<f32> = raw
        .iter()
        .enumerate()
        .map(|(idx, &w)| {
            let mut s = (idx as u32) ^ seed.wrapping_mul(2654435761);
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u1 = ((s as f32) / 4294967295.0).max(1e-7);
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * 0.5).cos();
            w + noise * mutation_rate
        })
        .collect();

    Tensor::from_data(TensorData::new(mutated, shape), device)
}

/// Merge two similar experts into one, mutating the second slot into a distinct explorer.
pub fn merge_expert_pair<B: Backend>(
    block: &mut MoEBlock<B>,
    e1_idx: usize,
    e2_idx: usize,
    spawn_rate: f32,
    seed: u32,
    device: &B::Device,
) {
    let inner_avg = average_tensors(
        &block.experts[e1_idx].gate_up.linear_inner.weight.val(),
        &block.experts[e2_idx].gate_up.linear_inner.weight.val(),
        device,
    );
    let outer_avg = average_tensors(
        &block.experts[e1_idx].gate_up.linear_outer.weight.val(),
        &block.experts[e2_idx].gate_up.linear_outer.weight.val(),
        device,
    );
    let down_avg = average_tensors(
        &block.experts[e1_idx].down.weight.val(),
        &block.experts[e2_idx].down.weight.val(),
        device,
    );

    // Update e1 with merged weights
    block.experts[e1_idx].gate_up.linear_inner.weight = Param::from_tensor(inner_avg.clone());
    block.experts[e1_idx].gate_up.linear_outer.weight = Param::from_tensor(outer_avg.clone());
    block.experts[e1_idx].down.weight = Param::from_tensor(down_avg.clone());

    // Spawn mutated explorer in e2 slot
    let e2_inner = mutate_tensor_for_spawn(&inner_avg, spawn_rate, seed, device);
    let e2_outer = mutate_tensor_for_spawn(&outer_avg, spawn_rate, seed + 100, device);
    let e2_down = mutate_tensor_for_spawn(&down_avg, spawn_rate, seed + 200, device);

    block.experts[e2_idx].gate_up.linear_inner.weight = Param::from_tensor(e2_inner);
    block.experts[e2_idx].gate_up.linear_outer.weight = Param::from_tensor(e2_outer);
    block.experts[e2_idx].down.weight = Param::from_tensor(e2_down);
}

/// Run full consolidation pass across all blocks in `MoELlama`.
pub fn consolidate_model<B: Backend>(
    model: &mut MoELlama<B>,
    recent_expert_mass: &[Vec<f32>],
    warehouse: Option<&ExpertWarehouse<B>>,
    config: &ConsolidationConfig,
    device: &B::Device,
) -> ConsolidationReport {
    let mut report = ConsolidationReport::default();
    let mut total_sim = 0.0f64;
    let mut sim_pairs_count = 0usize;

    for (blk_idx, block) in model.blocks.iter_mut().enumerate() {
        let n_exp = block.experts.len();
        let blk_mass = recent_expert_mass.get(blk_idx);

        // 1. Check pairwise similarities and merge redundant experts
        for i in 0..n_exp {
            for j in (i + 1)..n_exp {
                let sim = expert_similarity(&block.experts[i], &block.experts[j]);
                total_sim += sim as f64;
                sim_pairs_count += 1;

                if sim >= config.similarity_threshold {
                    let seed = (blk_idx * 1000 + i * 100 + j) as u32;
                    merge_expert_pair(block, i, j, config.spawn_mutation_rate, seed, device);
                    report.merged.push((blk_idx, i, j, sim));
                }
            }
        }

        // 2. Check for dormant experts to prune/offload
        if let Some(masses) = blk_mass {
            for (exp_idx, &mass) in masses.iter().enumerate() {
                if mass < config.activity_threshold && exp_idx < block.experts.len() {
                    // Offload to SSD warehouse if available
                    if let Some(wh) = warehouse {
                        let id = crate::tiered::global_expert_id(blk_idx, exp_idx);
                        let inner = block.experts[exp_idx].gate_up.linear_inner.weight.val();
                        let outer = block.experts[exp_idx].gate_up.linear_outer.weight.val();
                        let down = block.experts[exp_idx].down.weight.val();
                        let _ = wh.persist_expert(id, &inner, &outer, &down);
                    }

                    report.pruned.push((blk_idx, exp_idx, mass));

                    // Re-spawn dormant expert from highest active expert in same block
                    let best_parent_idx = masses
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    if best_parent_idx != exp_idx {
                        let seed = (blk_idx * 5000 + exp_idx * 77) as u32;
                        let p_inner = block.experts[best_parent_idx].gate_up.linear_inner.weight.val();
                        let p_outer = block.experts[best_parent_idx].gate_up.linear_outer.weight.val();
                        let p_down = block.experts[best_parent_idx].down.weight.val();

                        let m_inner = mutate_tensor_for_spawn(&p_inner, config.spawn_mutation_rate, seed, device);
                        let m_outer = mutate_tensor_for_spawn(&p_outer, config.spawn_mutation_rate, seed + 10, device);
                        let m_down = mutate_tensor_for_spawn(&p_down, config.spawn_mutation_rate, seed + 20, device);

                        block.experts[exp_idx].gate_up.linear_inner.weight = Param::from_tensor(m_inner);
                        block.experts[exp_idx].gate_up.linear_outer.weight = Param::from_tensor(m_outer);
                        block.experts[exp_idx].down.weight = Param::from_tensor(m_down);

                        report.spawned.push((blk_idx, best_parent_idx, exp_idx));
                    }
                }
            }
        }
    }

    report.mean_similarity = if sim_pairs_count > 0 {
        (total_sim / sim_pairs_count as f64) as f32
    } else {
        0.0
    };

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LlamaConfig;
    use crate::moe::{RouterConfig, upcycle_dense};

    type TestBackend = burn::backend::Wgpu;

    #[test]
    fn test_tensor_cosine_similarity() {
        let device = Default::default();
        let t1: Tensor<TestBackend, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let t2: Tensor<TestBackend, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let sim = tensor_cosine_similarity(&t1, &t2);
        assert!((sim - 1.0).abs() < 1e-4);

        let t3: Tensor<TestBackend, 2> = Tensor::from_data([[-1.0, -2.0], [-3.0, -4.0]], &device);
        let sim_neg = tensor_cosine_similarity(&t1, &t3);
        assert!((sim_neg - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_consolidation_pipeline() {
        let device = Default::default();
        let cfg = LlamaConfig::new(64, 32, 2, 2)
            .with_max_seq_len(32)
            .with_d_ff(64);
        let dense = cfg.init::<TestBackend>(&device);
        let router_cfg = RouterConfig::new(4);
        let mut moe = upcycle_dense(&dense, &router_cfg);

        let config = ConsolidationConfig {
            similarity_threshold: 0.90,
            activity_threshold: 0.05,
            spawn_mutation_rate: 0.05,
        };

        // Simulated expert activity: expert 0 high, expert 1 dead
        let activity = vec![
            vec![0.8, 0.01, 0.1, 0.09],
            vec![0.7, 0.00, 0.2, 0.10],
        ];

        let report = consolidate_model(&mut moe, &activity, None, &config, &device);
        assert!(report.mean_similarity > 0.0);
        assert!(!report.pruned.is_empty());
        assert!(!report.spawned.is_empty());
    }
}
