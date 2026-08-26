//! Tiered storage warehouse for MoE experts: L1 (VRAM) -> L2 (RAM) -> L3 (SSD).
//!
//! Handles eviction, decompression, memory mapping, and loading.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, create_dir_all};
use std::path::PathBuf;
use std::sync::RwLock;

/// Serialized parameter tensor container (shape + raw float data).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializedTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl SerializedTensor {
    pub fn from_tensor<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Self {
        let tensor_data = tensor.to_data();
        Self {
            shape: tensor_data.shape.to_vec(),
            data: tensor_data.iter::<f32>().collect(),
        }
    }

    pub fn to_tensor<B: Backend, const D: usize>(&self, device: &B::Device) -> Tensor<B, D> {
        let data = TensorData::new(self.data.clone(), self.shape.clone());
        Tensor::<B, D>::from_data(data, device)
    }
}

/// Representation of a single expert's neural weights.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializedExpert {
    pub id: u64,
    pub inner_weight: SerializedTensor,
    pub outer_weight: SerializedTensor,
    pub down_weight: SerializedTensor,
}

/// Tiered storage configuration.
#[derive(Clone, Debug)]
pub struct WarehouseConfig {
    /// Max number of active experts held in L1 (VRAM).
    pub l1_capacity: usize,
    /// Max number of experts held in L2 (RAM).
    pub l2_capacity: usize,
    /// Root path for L3 SSD persistent files.
    pub ssd_dir: PathBuf,
}

impl Default for WarehouseConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 8,
            l2_capacity: 64,
            ssd_dir: PathBuf::from("data/warehouse"),
        }
    }
}

/// Expert Warehouse managing L1, L2, and L3 tiers.
pub struct ExpertWarehouse<B: Backend> {
    config: WarehouseConfig,
    /// L1 Cache: In-VRAM active weights. (Expert ID -> SerializedExpert representation on device).
    l1_cache: RwLock<HashMap<u64, (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>>,
    /// L1 usage queue for LRU eviction.
    l1_order: RwLock<VecDeque<u64>>,
    /// L2 Cache: In-RAM serialized data.
    l2_cache: RwLock<HashMap<u64, SerializedExpert>>,
    /// L2 usage queue for LRU eviction.
    l2_order: RwLock<VecDeque<u64>>,
}

impl<B: Backend> ExpertWarehouse<B> {
    pub fn new(config: WarehouseConfig) -> std::io::Result<Self> {
        if !config.ssd_dir.exists() {
            create_dir_all(&config.ssd_dir)?;
        }
        Ok(Self {
            config,
            l1_cache: RwLock::new(HashMap::new()),
            l1_order: RwLock::new(VecDeque::new()),
            l2_cache: RwLock::new(HashMap::new()),
            l2_order: RwLock::new(VecDeque::new()),
        })
    }

    /// Save an expert directly to L3 (SSD) with zstd compression.
    pub fn persist_to_l3(&self, expert: &SerializedExpert) -> anyhow::Result<()> {
        let file_path = self.config.ssd_dir.join(format!("expert_{}.bin", expert.id));
        let encoded = bincode::serialize(expert)?;
        let compressed = zstd::encode_all(&encoded[..], 3)?;
        std::fs::write(&file_path, compressed)?;
        Ok(())
    }

    /// Load an expert from L3 (SSD) using memory mapping and decompression.
    pub fn load_from_l3(&self, id: u64) -> anyhow::Result<SerializedExpert> {
        let file_path = self.config.ssd_dir.join(format!("expert_{}.bin", id));
        let file = File::open(&file_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let decompressed = zstd::decode_all(&mmap[..])?;
        let expert: SerializedExpert = bincode::deserialize(&decompressed)?;
        Ok(expert)
    }

    /// Insert a serialized expert into L2 cache, spilling to L3 if full.
    pub fn put_l2(&self, expert: SerializedExpert) -> anyhow::Result<()> {
        let id = expert.id;
        let mut l2 = self.l2_cache.write().unwrap();
        let mut order = self.l2_order.write().unwrap();

        if l2.len() >= self.config.l2_capacity && !l2.contains_key(&id) {
            if let Some(evicted_id) = order.pop_front() {
                if let Some(evicted_expert) = l2.remove(&evicted_id) {
                    self.persist_to_l3(&evicted_expert)?;
                }
            }
        }

        l2.insert(id, expert);
        order.push_back(id);
        Ok(())
    }

    /// Retrieve an expert's tensors on device, traversing L1 -> L2 -> L3.
    pub fn get_expert(
        &self,
        id: u64,
        device: &B::Device,
    ) -> anyhow::Result<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)> {
        // 1. Check L1 Cache
        {
            let l1 = self.l1_cache.read().unwrap();
            if let Some(tensors) = l1.get(&id) {
                return Ok((tensors.0.clone(), tensors.1.clone(), tensors.2.clone()));
            }
        }

        // 2. Check L2 Cache
        let expert = {
            let l2 = self.l2_cache.read().unwrap();
            l2.get(&id).cloned()
        };

        // 3. Fallback to L3 (SSD) if not in L2
        let expert = match expert {
            Some(e) => e,
            None => {
                let e = self.load_from_l3(id)?;
                self.put_l2(e.clone())?;
                e
            }
        };

        // Convert serialized data to device tensors
        let inner: Tensor<B, 2> = expert.inner_weight.to_tensor(device);
        let outer: Tensor<B, 2> = expert.outer_weight.to_tensor(device);
        let down: Tensor<B, 2> = expert.down_weight.to_tensor(device);

        // Put into L1 Cache
        {
            let mut l1 = self.l1_cache.write().unwrap();
            let mut order = self.l1_order.write().unwrap();

            if l1.len() >= self.config.l1_capacity && !l1.contains_key(&id) {
                if let Some(evicted_id) = order.pop_front() {
                    l1.remove(&evicted_id);
                }
            }

            l1.insert(id, (inner.clone(), outer.clone(), down.clone()));
            order.push_back(id);
        }

        Ok((inner, outer, down))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::Wgpu;

    #[test]
    fn test_serialized_tensor_roundtrip() {
        let device = Default::default();
        let original: Tensor<TestBackend, 2> =
            Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let serialized = SerializedTensor::from_tensor(&original);
        let reconstructed: Tensor<TestBackend, 2> = serialized.to_tensor(&device);

        assert_eq!(original.to_data(), reconstructed.to_data());
    }

    #[test]
    fn test_tiered_warehouse_roundtrip() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let config = WarehouseConfig {
            l1_capacity: 2,
            l2_capacity: 4,
            ssd_dir: temp_dir.path().to_path_buf(),
        };
        let warehouse = ExpertWarehouse::<TestBackend>::new(config)?;

        let tensor_data = SerializedTensor {
            shape: vec![2, 2],
            data: vec![0.5, 0.2, -0.1, 0.9],
        };
        let expert = SerializedExpert {
            id: 42,
            inner_weight: tensor_data.clone(),
            outer_weight: tensor_data.clone(),
            down_weight: tensor_data,
        };

        // Put in L2
        warehouse.put_l2(expert.clone())?;

        // Retrieve onto device via get_expert (promotes to L1)
        let device = Default::default();
        let (inner, outer, down) = warehouse.get_expert(42, &device)?;

        assert_eq!(inner.dims(), [2, 2]);
        assert_eq!(outer.dims(), [2, 2]);
        assert_eq!(down.dims(), [2, 2]);

        // Evict to L3 by writing directly and reloading
        warehouse.persist_to_l3(&expert)?;
        let from_l3 = warehouse.load_from_l3(42)?;
        assert_eq!(from_l3.id, 42);

        Ok(())
    }
}

