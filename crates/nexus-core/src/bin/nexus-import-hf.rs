//! CLI binary to import and inspect HuggingFace open weights.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-import-hf -- [model_dir] [--upcycle] [--warehouse]
//!
//! Example:
//!   cargo run --release -p nexus-core --bin nexus-import-hf -- data/models/smollm2-135m --upcycle --warehouse

use nexus_core::import::{import_hf_to_llama, import_hf_to_warehouse, HfLlamaConfig, SafetensorsReader};
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_memory::{ExpertWarehouse, WarehouseConfig};
use std::path::Path;

#[cfg(feature = "cuda")]
type Backend = burn::backend::Cuda;
#[cfg(not(feature = "cuda"))]
type Backend = burn::backend::Wgpu;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_dir_str = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "data/models/smollm2-135m".to_string());

    let model_dir = Path::new(&model_dir_str);
    if !model_dir.exists() {
        eprintln!("Model directory '{}' does not exist.", model_dir.display());
        std::process::exit(1);
    }

    println!("=== Nexus HuggingFace Importer ===");
    println!("Model Path: {}", model_dir.display());

    // 1. Inspect config
    let hf_cfg = HfLlamaConfig::load_from_dir(model_dir)?;
    println!("\n[Config]");
    println!("- Vocab Size: {}", hf_cfg.vocab_size);
    println!("- Hidden Size (d_model): {}", hf_cfg.hidden_size);
    println!("- FFN Intermediate (d_ff): {}", hf_cfg.intermediate_size);
    println!("- Layers: {}", hf_cfg.num_hidden_layers);
    println!("- Heads: {}", hf_cfg.num_attention_heads);
    println!("- RoPE Theta: {}", hf_cfg.rope_theta);

    // 2. Inspect Safetensors
    let weights_path = model_dir.join("model.safetensors");
    let reader = SafetensorsReader::open(&weights_path)?;
    println!("\n[Safetensors Container]");
    println!("- Total Tensor Keys: {}", reader.tensor_names().len());

    #[cfg(feature = "cuda")]
    let device = burn::backend::cuda::CudaDevice::default();
    #[cfg(not(feature = "cuda"))]
    let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);

    // 3. Load into Nexus LLaMA
    println!("\n[Loading Dense Model into Nexus Engine on NVIDIA T500...]");
    let llama = import_hf_to_llama::<Backend>(model_dir, &device)?;
    println!("✓ Successfully loaded {} LLaMA blocks!", llama.n_blocks());

    // 4. Optionally Upcycle to MoE
    if args.iter().any(|a| a == "--upcycle") {
        println!("\n[Upcycling into MoE Brain...]");
        let router_cfg = RouterConfig::new(4);
        let moe = upcycle_dense(&llama, &router_cfg);
        println!(
            "✓ Upcycled to MoE with {} blocks and {} experts each!",
            moe.blocks.len(),
            moe.blocks[0].experts.len()
        );
    }

    // 5. Optionally Persist to Tiered Warehouse
    if args.iter().any(|a| a == "--warehouse") {
        println!("\n[Exporting Experts into Tiered Warehouse (L1/L2/L3 SSD)...]");
        let wh_cfg = WarehouseConfig::default();
        let warehouse = ExpertWarehouse::<Backend>::new(wh_cfg)?;
        let count = import_hf_to_warehouse::<Backend>(model_dir, &warehouse, 4)?;
        println!("✓ Saved {count} experts into compressed L3 SSD storage!");
    }

    println!("\n✓ Import complete and validated without errors.");
    Ok(())
}
