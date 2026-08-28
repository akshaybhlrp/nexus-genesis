//! HuggingFace safetensors import integration tests.
//!
//! Validates:
//! - Safetensors binary header parsing (little-endian u64 + JSON metadata).
//! - Data dtype extraction (BF16, F32, F16).
//! - 2D matrix transposition from PyTorch `[out, in]` to Burn `[in, out]`.
//! - Configuration conversion from `HfLlamaConfig` to `LlamaConfig`.
//! - Warehouse expert injection from open weight checkpoints.

mod common;

use common::*;
use nexus_core::import::{HfLlamaConfig, SafetensorsReader, import_hf_to_warehouse};
use nexus_memory::{ExpertWarehouse, WarehouseConfig};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

struct SafetensorsFixture {
    pub dir: TempDir,
    pub path: PathBuf,
}

impl SafetensorsFixture {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        Self { dir, path }
    }

    /// Write a synthetic `.safetensors` file with specified tensors.
    pub fn write_synthetic(&self, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header_map = serde_json::Map::new();
        let mut total_offset = 0usize;
        let mut payload = Vec::new();

        for (name, dtype, shape, data) in tensors {
            let start = total_offset;
            let end = total_offset + data.len();
            total_offset = end;
            payload.extend_from_slice(data);

            let mut meta = serde_json::Map::new();
            meta.insert("dtype".to_string(), serde_json::Value::String(dtype.to_string()));
            meta.insert(
                "shape".to_string(),
                serde_json::Value::Array(shape.iter().map(|&s| serde_json::Value::from(s as u64)).collect()),
            );
            meta.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(vec![
                    serde_json::Value::from(start as u64),
                    serde_json::Value::from(end as u64),
                ]),
            );

            header_map.insert(name.to_string(), serde_json::Value::Object(meta));
        }

        let header_str = serde_json::to_string(&serde_json::Value::Object(header_map)).unwrap();
        let header_bytes = header_str.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut f = File::create(&self.path).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(&payload).unwrap();
        f.sync_all().unwrap();
    }
}

#[test]
fn test_safetensors_reader_f32_extraction() {
    let fixture = SafetensorsFixture::new();
    let original_floats = vec![1.0f32, 2.5, -3.125, 4.0];
    let mut raw_bytes = Vec::new();
    for f in &original_floats {
        raw_bytes.extend_from_slice(&f.to_le_bytes());
    }

    fixture.write_synthetic(&[("test.weight", "F32", vec![2, 2], raw_bytes)]);

    let reader = SafetensorsReader::open(&fixture.path).expect("open safetensors");
    let (floats, shape) = reader.get_tensor("test.weight").expect("read tensor");

    assert_eq!(shape, vec![2, 2]);
    assert_eq!(floats, original_floats);
}

#[test]
fn test_safetensors_reader_bf16_conversion() {
    let fixture = SafetensorsFixture::new();
    // bfloat16: 1.0f32 -> bits 0x3F800000 -> upper 16 bits = 0x3F80
    // 2.0f32 -> bits 0x40000000 -> upper 16 bits = 0x4000
    let bf16_vals = vec![0x3F80u16, 0x4000u16];
    let mut raw_bytes = Vec::new();
    for u in &bf16_vals {
        raw_bytes.extend_from_slice(&u.to_le_bytes());
    }

    fixture.write_synthetic(&[("bf16.tensor", "BF16", vec![2], raw_bytes)]);

    let reader = SafetensorsReader::open(&fixture.path).expect("open safetensors");
    let (floats, shape) = reader.get_tensor("bf16.tensor").expect("read bf16 tensor");

    assert_eq!(shape, vec![2]);
    assert_eq!(floats, vec![1.0f32, 2.0f32]);
}

#[test]
fn test_safetensors_2d_transposition_math() {
    let fixture = SafetensorsFixture::new();
    // PyTorch tensor shape [out=2, in=3]:
    // Row 0: [1, 2, 3]
    // Row 1: [4, 5, 6]
    let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut raw_bytes = Vec::new();
    for f in &original {
        raw_bytes.extend_from_slice(&f.to_le_bytes());
    }

    fixture.write_synthetic(&[("linear.weight", "F32", vec![2, 3], raw_bytes)]);

    let reader = SafetensorsReader::open(&fixture.path).expect("open safetensors");
    let (transposed, shape) = reader.get_2d_transposed("linear.weight").expect("transpose 2d");

    // Burn shape [in=3, out=2]:
    // Row 0: [1, 4]
    // Row 1: [2, 5]
    // Row 2: [3, 6]
    assert_eq!(shape, [3, 2]);
    assert_eq!(transposed, vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_hf_llama_config_conversion() {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = dir.path().join("config.json");

    let json_content = r#"{
        "vocab_size": 49152,
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "rms_norm_eps": 0.00001,
        "rope_theta": 100000.0,
        "max_position_embeddings": 8192
    }"#;

    std::fs::write(&cfg_path, json_content).unwrap();

    let hf_cfg = HfLlamaConfig::load_from_dir(dir.path()).expect("parse hf config");
    assert_eq!(hf_cfg.vocab_size, 49152);
    assert_eq!(hf_cfg.hidden_size, 576);
    assert_eq!(hf_cfg.intermediate_size, 1536);
    assert_eq!(hf_cfg.num_hidden_layers, 30);
    assert_eq!(hf_cfg.num_attention_heads, 9);

    let nexus_cfg = hf_cfg.to_nexus_config();
    assert_eq!(nexus_cfg.vocab_size, 49152);
    assert_eq!(nexus_cfg.d_model, 576);
    assert_eq!(nexus_cfg.d_ff, 1536);
    assert_eq!(nexus_cfg.n_layers, 30);
    assert_eq!(nexus_cfg.n_heads, 9);
}

#[test]
fn test_import_hf_to_warehouse_synthetic_pipeline() -> anyhow::Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let cfg_path = dir.path().join("config.json");
    let weights_path = dir.path().join("model.safetensors");

    let json_content = r#"{
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 64
    }"#;
    std::fs::write(&cfg_path, json_content)?;

    // Create synthetic FFN weights for 2 layers
    let fixture = SafetensorsFixture::new();
    let gate_raw: Vec<u8> = vec![0u8; 32 * 16 * 4]; // [32, 16] F32
    let up_raw: Vec<u8> = vec![0u8; 32 * 16 * 4];   // [32, 16] F32
    let down_raw: Vec<u8> = vec![0u8; 16 * 32 * 4]; // [16, 32] F32

    fixture.write_synthetic(&[
        ("model.layers.0.mlp.gate_proj.weight", "F32", vec![32, 16], gate_raw.clone()),
        ("model.layers.0.mlp.up_proj.weight", "F32", vec![32, 16], up_raw.clone()),
        ("model.layers.0.mlp.down_proj.weight", "F32", vec![16, 32], down_raw.clone()),
        ("model.layers.1.mlp.gate_proj.weight", "F32", vec![32, 16], gate_raw),
        ("model.layers.1.mlp.up_proj.weight", "F32", vec![32, 16], up_raw),
        ("model.layers.1.mlp.down_proj.weight", "F32", vec![16, 32], down_raw),
    ]);

    std::fs::copy(&fixture.path, &weights_path)?;

    let wh_cfg = WarehouseConfig {
        l1_capacity: 4,
        l2_capacity: 8,
        ssd_dir: dir.path().join("warehouse"),
    };
    let warehouse = ExpertWarehouse::<TB>::new(wh_cfg)?;

    // 2 layers * 4 experts = 8 experts total
    let total = import_hf_to_warehouse::<TB>(dir.path(), &warehouse, 4)?;
    assert_eq!(total, 8);

    // Verify all 8 experts exist in warehouse
    for blk in 0..2 {
        for exp in 0..4 {
            let id = nexus_core::tiered::global_expert_id(blk, exp);
            assert!(warehouse.contains_expert(id));
        }
    }

    Ok(())
}
