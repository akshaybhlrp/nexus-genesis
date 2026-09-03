//! HuggingFace open weights ingestion engine.
//!
//! Loads open foundation models (e.g. `SmolLM2-135M`, `Qwen2.5`, `Llama-3.2`)
//! directly from `.safetensors` files without Python or Torch runtime dependencies.
//!
//! Maps standard HuggingFace LLaMA tensors into Nexus [`Llama`], [`MoELlama`],
//! and [`ExpertWarehouse`] structures.

use crate::model::{Llama, LlamaConfig};
use anyhow::{Context, Result, anyhow, bail};
use burn::module::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use memmap2::Mmap;
use nexus_memory::{ExpertWarehouse, SerializedExpert, SerializedTensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Metadata for a single tensor stored inside a `.safetensors` container.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Zero-dependency, memory-mapped `.safetensors` reader.
pub struct SafetensorsReader {
    _mmap: Mmap,
    base_offset: usize,
    tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsReader {
    /// Open and memory-map a `.safetensors` file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("Failed to open safetensors file: {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            bail!("Safetensors file too small (less than 8 bytes)");
        }

        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if mmap.len() < 8 + header_len {
            bail!("Safetensors header length ({header_len}) exceeds file size");
        }

        let header_str = std::str::from_utf8(&mmap[8..8 + header_len])
            .context("Safetensors header is not valid UTF-8")?;

        let json_val: serde_json::Value = serde_json::from_str(header_str)
            .context("Failed to parse safetensors JSON header")?;

        let obj = json_val.as_object().ok_or_else(|| anyhow!("Header JSON is not an object"))?;
        let mut tensors = HashMap::new();

        for (k, v) in obj {
            if k == "__metadata__" {
                continue;
            }
            if let Some(t_obj) = v.as_object() {
                let dtype = t_obj.get("dtype").and_then(|d| d.as_str()).unwrap_or("F32").to_string();
                let shape = t_obj
                    .get("shape")
                    .and_then(|s| s.as_array())
                    .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
                    .unwrap_or_default();
                let offsets = t_obj
                    .get("data_offsets")
                    .and_then(|o| o.as_array())
                    .and_then(|arr| {
                        if arr.len() == 2 {
                            Some((arr[0].as_u64()? as usize, arr[1].as_u64()? as usize))
                        } else {
                            None
                        }
                    })
                    .unwrap_or((0, 0));

                tensors.insert(k.clone(), TensorMeta {
                    dtype,
                    shape,
                    data_offsets: offsets,
                });
            }
        }

        Ok(Self {
            _mmap: mmap,
            base_offset: 8 + header_len,
            tensors,
        })
    }

    /// Retrieve tensor names present in this container.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Read raw tensor data and convert into `Vec<f32>`.
    pub fn get_tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{name}' not found in safetensors"))?;

        let start = self.base_offset + meta.data_offsets.0;
        let end = self.base_offset + meta.data_offsets.1;

        if end > self._mmap.len() {
            bail!("Tensor '{name}' offset out of file bounds");
        }

        let slice = &self._mmap[start..end];
        let mut floats = Vec::new();

        match meta.dtype.as_str() {
            "BF16" => {
                floats.reserve(slice.len() / 2);
                for chunk in slice.chunks_exact(2) {
                    let u = u16::from_le_bytes(chunk.try_into().unwrap());
                    // bfloat16 to f32: shift to upper 16 bits of IEEE 754 float
                    let f = f32::from_bits((u as u32) << 16);
                    floats.push(f);
                }
            }
            "F32" => {
                floats.reserve(slice.len() / 4);
                for chunk in slice.chunks_exact(4) {
                    let f = f32::from_le_bytes(chunk.try_into().unwrap());
                    floats.push(f);
                }
            }
            "F16" => {
                floats.reserve(slice.len() / 2);
                for chunk in slice.chunks_exact(2) {
                    let u = u16::from_le_bytes(chunk.try_into().unwrap());
                    floats.push(f16_to_f32(u));
                }
            }
            other => bail!("Unsupported safetensors dtype '{other}' for tensor '{name}'"),
        }

        Ok((floats, meta.shape.clone()))
    }

    /// Read 2D weight matrix and transpose from PyTorch `[out_features, in_features]`
    /// to Burn's expected `[in_features, out_features]`.
    pub fn get_2d_transposed(&self, name: &str) -> Result<(Vec<f32>, [usize; 2])> {
        let (raw, shape) = self.get_tensor(name)?;
        if shape.len() != 2 {
            bail!("Expected 2D tensor for '{name}', found shape {:?}", shape);
        }

        let out_dim = shape[0];
        let in_dim = shape[1];
        let mut transposed = vec![0.0f32; out_dim * in_dim];

        for i in 0..out_dim {
            for j in 0..in_dim {
                transposed[j * out_dim + i] = raw[i * in_dim + j];
            }
        }

        Ok((transposed, [in_dim, out_dim]))
    }

    /// Read 2D KV projection and expand GQA (Grouped Query Attention) if kv_heads < q_heads.
    pub fn get_kv_proj_transposed(
        &self,
        name: &str,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<(Vec<f32>, [usize; 2])> {
        let (raw, shape) = self.get_tensor(name)?;
        if shape.len() != 2 {
            bail!("Expected 2D tensor for '{name}', found shape {:?}", shape);
        }
        let out_dim = shape[0];
        let in_dim = shape[1];

        if n_kv_heads == n_heads || out_dim == d_model {
            return self.get_2d_transposed(name);
        }

        let head_dim = d_model / n_heads;
        let groups = n_heads / n_kv_heads.max(1);
        let expanded_out_dim = d_model;
        let mut expanded_raw = vec![0.0f32; expanded_out_dim * in_dim];

        for kv_h in 0..n_kv_heads {
            for g in 0..groups {
                let target_head = kv_h * groups + g;
                for row in 0..head_dim {
                    let src_row = kv_h * head_dim + row;
                    let dst_row = target_head * head_dim + row;
                    let src_offset = src_row * in_dim;
                    let dst_offset = dst_row * in_dim;
                    if src_offset + in_dim <= raw.len() && dst_offset + in_dim <= expanded_raw.len() {
                        expanded_raw[dst_offset..dst_offset + in_dim]
                            .copy_from_slice(&raw[src_offset..src_offset + in_dim]);
                    }
                }
            }
        }

        let mut transposed = vec![0.0f32; expanded_out_dim * in_dim];
        for i in 0..expanded_out_dim {
            for j in 0..in_dim {
                transposed[j * expanded_out_dim + i] = expanded_raw[i * in_dim + j];
            }
        }

        Ok((transposed, [in_dim, expanded_out_dim]))
    }
}

/// Convert IEEE 754 half-precision float (`f16`) bits to single-precision `f32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp32 = (127 - 15 - e + 1) as u32;
            let mant32 = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23)) // Inf
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13)) // NaN
        }
    } else {
        let exp32 = (exp + (127 - 15)) << 23;
        let mant32 = mant << 13;
        f32::from_bits((sign << 31) | exp32 | mant32)
    }
}

/// HuggingFace LLaMA `config.json` schema.
#[derive(Debug, Clone, Deserialize)]
pub struct HfLlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_seq_len")]
    pub max_position_embeddings: usize,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_max_seq_len() -> usize {
    2048
}

impl HfLlamaConfig {
    /// Load from a model directory containing `config.json`.
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Could not read config at {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config at {}", path.display()))?;
        Ok(cfg)
    }

    /// Convert to Nexus [`LlamaConfig`].
    pub fn to_nexus_config(&self) -> LlamaConfig {
        LlamaConfig::new(
            self.vocab_size,
            self.hidden_size,
            self.num_attention_heads,
            self.num_hidden_layers,
        )
        .with_max_seq_len(self.max_position_embeddings.min(2048))
        .with_d_ff(self.intermediate_size)
        .with_rms_norm_eps(self.rms_norm_eps)
        .with_rope_theta(self.rope_theta)
    }
}

/// Import a full HuggingFace model into an initialized [`Llama`] instance.
pub fn import_hf_to_llama<B: Backend>(model_dir: &Path, device: &B::Device) -> Result<Llama<B>> {
    import_hf_to_llama_scaled(model_dir, device, None)
}

/// Import a HuggingFace model with an optional upper bound on the number of transformer layers.
/// This allows scaling foundation models to fit constrained VRAM (e.g. 4GB GPUs) during training.
pub fn import_hf_to_llama_scaled<B: Backend>(
    model_dir: &Path,
    device: &B::Device,
    max_layers: Option<usize>,
) -> Result<Llama<B>> {
    let hf_cfg = HfLlamaConfig::load_from_dir(model_dir)?;
    let mut nexus_cfg = hf_cfg.to_nexus_config();
    if let Some(layers) = max_layers {
        nexus_cfg.n_layers = layers.min(hf_cfg.num_hidden_layers);
    }
    let mut model = nexus_cfg.init::<B>(device);

    let weights_path = model_dir.join("model.safetensors");
    let reader = SafetensorsReader::open(&weights_path)?;

    // 1. Token Embeddings
    if let Ok((embed_data, shape)) = reader.get_tensor("model.embed_tokens.weight") {
        let t = Tensor::<B, 2>::from_data(TensorData::new(embed_data, shape), device);
        model.token_embed.weight = Param::from_tensor(t);
    }

    // 2. Final Norm & LM Head
    if let Ok((norm_data, shape)) = reader.get_tensor("model.norm.weight") {
        let t = Tensor::<B, 1>::from_data(TensorData::new(norm_data, shape), device);
        model.final_norm.gamma = Param::from_tensor(t);
    }

    if let Ok((head_data, shape)) = reader.get_2d_transposed("lm_head.weight") {
        let t = Tensor::<B, 2>::from_data(TensorData::new(head_data, shape), device);
        model.lm_head.weight = Param::from_tensor(t);
    } else if let Ok((embed_data, shape)) = reader.get_2d_transposed("model.embed_tokens.weight") {
        // Tied weights fallback
        let t = Tensor::<B, 2>::from_data(TensorData::new(embed_data, shape), device);
        model.lm_head.weight = Param::from_tensor(t);
    }

    // 3. Layer Blocks
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");

        // Attention Norm & FFN Norm
        if let Ok((d, s)) = reader.get_tensor(&format!("{prefix}.input_layernorm.weight")) {
            block.attn_norm.gamma = Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(d, s), device));
        }
        if let Ok((d, s)) = reader.get_tensor(&format!("{prefix}.post_attention_layernorm.weight")) {
            block.ffn_norm.gamma = Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(d, s), device));
        }

        // Self-Attention Projections
        let n_heads = hf_cfg.num_attention_heads;
        let n_kv_heads = hf_cfg.num_key_value_heads.unwrap_or(n_heads);
        let d_model = hf_cfg.hidden_size;

        if let Ok((d, s)) = reader.get_2d_transposed(&format!("{prefix}.self_attn.q_proj.weight")) {
            block.attn.query.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(d, s), device));
        }
        if let Ok((d, s)) = reader.get_kv_proj_transposed(&format!("{prefix}.self_attn.k_proj.weight"), d_model, n_heads, n_kv_heads) {
            block.attn.key.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(d, s), device));
        }
        if let Ok((d, s)) = reader.get_kv_proj_transposed(&format!("{prefix}.self_attn.v_proj.weight"), d_model, n_heads, n_kv_heads) {
            block.attn.value.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(d, s), device));
        }
        if let Ok((d, s)) = reader.get_2d_transposed(&format!("{prefix}.self_attn.o_proj.weight")) {
            block.attn.output.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(d, s), device));
        }

        // SwiGLU FFN Projections
        if let Ok((d, s)) = reader.get_2d_transposed(&format!("{prefix}.mlp.gate_proj.weight")) {
            block.ffn_gate_up.linear_inner.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(d, s), device));
        }
        if let Ok((d, s)) = reader.get_2d_transposed(&format!("{prefix}.mlp.up_proj.weight")) {
            block.ffn_gate_up.linear_outer.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(d, s), device));
        }
        if let Ok((d, s)) = reader.get_2d_transposed(&format!("{prefix}.mlp.down_proj.weight")) {
            block.ffn_down.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(d, s), device));
        }
    }

    Ok(model)
}

/// Ingest FFN weights from HuggingFace directly into an [`ExpertWarehouse`] (L1/L2/L3).
///
/// Each layer's pretrained FFN is cloned across `n_experts` slots per block.
pub fn import_hf_to_warehouse<B: Backend>(
    model_dir: &Path,
    warehouse: &ExpertWarehouse<B>,
    n_experts_per_block: usize,
) -> Result<usize> {
    let hf_cfg = HfLlamaConfig::load_from_dir(model_dir)?;
    let weights_path = model_dir.join("model.safetensors");
    let reader = SafetensorsReader::open(&weights_path)?;

    let mut total_experts = 0;

    for blk_idx in 0..hf_cfg.num_hidden_layers {
        let prefix = format!("model.layers.{blk_idx}");

        let (gate_data, gate_shape) = reader.get_2d_transposed(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let (up_data, up_shape) = reader.get_2d_transposed(&format!("{prefix}.mlp.up_proj.weight"))?;
        let (down_data, down_shape) = reader.get_2d_transposed(&format!("{prefix}.mlp.down_proj.weight"))?;

        let inner_st = SerializedTensor {
            shape: gate_shape.to_vec(),
            data: gate_data,
        };
        let outer_st = SerializedTensor {
            shape: up_shape.to_vec(),
            data: up_data,
        };
        let down_st = SerializedTensor {
            shape: down_shape.to_vec(),
            data: down_data,
        };

        for exp_idx in 0..n_experts_per_block {
            let id = crate::tiered::global_expert_id(blk_idx, exp_idx);
            let expert = SerializedExpert {
                id,
                inner_weight: inner_st.clone(),
                outer_weight: outer_st.clone(),
                down_weight: down_st.clone(),
            };
            warehouse.put_l2(expert.clone())?;
            warehouse.persist_to_l3(&expert)?;
            total_experts += 1;
        }
    }

    Ok(total_experts)
}
