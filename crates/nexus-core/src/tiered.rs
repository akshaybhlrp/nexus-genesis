//! Phase 4: Tiered storage integration for MoELlama.
//!
//! Provides bidirectional synchronization between in-memory `MoELlama` models
//! and `nexus_memory::ExpertWarehouse` (L1 GPU VRAM -> L2 CPU RAM -> L3 SSD).

use crate::moe::MoELlama;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use nexus_memory::{AsyncPrefetcher, ExpertWarehouse};
use std::sync::Arc;

/// Compute global expert ID from block index and expert index.
#[inline]
pub fn global_expert_id(block_idx: usize, expert_idx: usize) -> u64 {
    ((block_idx as u64) << 32) | (expert_idx as u64)
}

/// Decode global expert ID into (block_idx, expert_idx).
#[inline]
pub fn decode_expert_id(id: u64) -> (usize, usize) {
    ((id >> 32) as usize, (id & 0xFFFFFFFF) as usize)
}

/// Persist all experts from a `MoELlama` model into an `ExpertWarehouse`.
/// Returns total number of experts persisted.
pub fn offload_model_to_warehouse<B: Backend>(
    model: &MoELlama<B>,
    warehouse: &ExpertWarehouse<B>,
) -> anyhow::Result<usize> {
    let mut total = 0;
    for (blk_idx, block) in model.blocks.iter().enumerate() {
        for (exp_idx, expert) in block.experts.iter().enumerate() {
            let id = global_expert_id(blk_idx, exp_idx);
            let inner = expert.gate_up.linear_inner.weight.val();
            let outer = expert.gate_up.linear_outer.weight.val();
            let down = expert.down.weight.val();

            warehouse.persist_expert(id, &inner, &outer, &down)?;
            total += 1;
        }
    }
    Ok(total)
}

/// Reload expert weights from `ExpertWarehouse` into an existing `MoELlama` model.
pub fn load_model_from_warehouse<B: Backend>(
    model: &mut MoELlama<B>,
    warehouse: &ExpertWarehouse<B>,
    device: &B::Device,
) -> anyhow::Result<usize> {
    let mut total = 0;
    for (blk_idx, block) in model.blocks.iter_mut().enumerate() {
        for (exp_idx, expert) in block.experts.iter_mut().enumerate() {
            let id = global_expert_id(blk_idx, exp_idx);
            if let Ok((inner, outer, down)) = warehouse.get_expert(id, device) {
                // Verify shape compatibility before applying
                if inner.shape() == expert.gate_up.linear_inner.weight.val().shape()
                    && outer.shape() == expert.gate_up.linear_outer.weight.val().shape()
                    && down.shape() == expert.down.weight.val().shape()
                {
                    expert.gate_up.linear_inner.weight = burn::module::Param::from_tensor(inner);
                    expert.gate_up.linear_outer.weight = burn::module::Param::from_tensor(outer);
                    expert.down.weight = burn::module::Param::from_tensor(down);
                    total += 1;
                }
            }
        }
    }
    Ok(total)
}

/// Tiered expert manager that coordinates dynamic lookahead prefetching
/// and tiered swapping for an MoE layer.
pub struct TieredExpertManager<B: Backend> {
    warehouse: Arc<ExpertWarehouse<B>>,
    device: B::Device,
}

impl<B: Backend + 'static> TieredExpertManager<B> {
    pub fn new(warehouse: Arc<ExpertWarehouse<B>>, device: B::Device) -> Self {
        Self { warehouse, device }
    }

    /// Create an asynchronous prefetcher for double-buffered weight transfers.
    pub fn create_prefetcher(&self, buffer_size: usize) -> AsyncPrefetcher<B> {
        AsyncPrefetcher::new(Arc::clone(&self.warehouse), self.device.clone(), buffer_size)
    }

    /// Retrieve expert weights on demand, searching L1 -> L2 -> L3.
    pub fn get_expert_weights(
        &self,
        block_idx: usize,
        expert_idx: usize,
    ) -> anyhow::Result<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)> {
        let id = global_expert_id(block_idx, expert_idx);
        self.warehouse.get_expert(id, &self.device)
    }

    /// Access warehouse reference.
    pub fn warehouse(&self) -> &Arc<ExpertWarehouse<B>> {
        &self.warehouse
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LlamaConfig;
    use crate::moe::{RouterConfig, upcycle_dense};
    use tempfile::tempdir;

    type TestBackend = burn::backend::Wgpu;

    #[test]
    fn test_global_expert_id_roundtrip() {
        for b in [0, 1, 4, 16] {
            for e in [0, 3, 7, 63] {
                let id = global_expert_id(b, e);
                let (dec_b, dec_e) = decode_expert_id(id);
                assert_eq!(b, dec_b);
                assert_eq!(e, dec_e);
            }
        }
    }

    #[test]
    fn test_offload_and_reload_model_to_warehouse() -> anyhow::Result<()> {
        let device = Default::default();
        let tmp = tempdir()?;
        let cfg = LlamaConfig::new(64, 32, 2, 2)
            .with_max_seq_len(32)
            .with_d_ff(64);
        let dense = cfg.init::<TestBackend>(&device);
        let router_cfg = RouterConfig::new(4);
        let mut moe = upcycle_dense(&dense, &router_cfg);

        let wh_cfg = nexus_memory::WarehouseConfig {
            l1_capacity: 4,
            l2_capacity: 16,
            ssd_dir: tmp.path().to_path_buf(),
        };
        let warehouse = ExpertWarehouse::<TestBackend>::new(wh_cfg)?;

        // Offload all 2 blocks * 4 experts = 8 experts
        let saved = offload_model_to_warehouse(&moe, &warehouse)?;
        assert_eq!(saved, 8);

        // Reload back
        let loaded = load_model_from_warehouse(&mut moe, &warehouse, &device)?;
        assert_eq!(loaded, 8);

        Ok(())
    }
}
