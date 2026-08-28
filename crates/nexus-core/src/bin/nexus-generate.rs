//! Autoregressive text generation and token sampling binary with HuggingFace weights support.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-generate -- [prompt] [--model data/models/smollm2-135m] [--tokens 30] [--temperature 0.7]

use burn::tensor::Tensor;
use nexus_core::import::import_hf_to_llama;
use nexus_core::model::{LlamaConfig, check_token_ids};
use std::path::Path;
use tokenizers::Tokenizer;

type Backend = burn::backend::Wgpu;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut prompt = "The future of artificial intelligence is".to_string();
    let mut model_dir_str = "data/models/smollm2-135m".to_string();
    let mut max_tokens = 30usize;
    let mut temperature = 0.7f32;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--tokens" && i + 1 < args.len() {
            i += 1;
            if let Ok(t) = args[i].parse() {
                max_tokens = t;
            }
        } else if arg == "--model" && i + 1 < args.len() {
            i += 1;
            model_dir_str = args[i].clone();
        } else if arg == "--temperature" && i + 1 < args.len() {
            i += 1;
            if let Ok(temp) = args[i].parse() {
                temperature = temp;
            }
        } else if !arg.starts_with("--") {
            prompt = arg.clone();
        }
        i += 1;
    }

    println!("=== Nexus Autoregressive Inference ===");
    println!("Prompt: \"{prompt}\"");
    println!("Model Path: {model_dir_str}");
    println!("Max Tokens: {max_tokens} | Temperature: {temperature:.2}");

    let model_dir = Path::new(&model_dir_str);
    let tok_path = if model_dir.join("tokenizer.json").exists() {
        model_dir.join("tokenizer.json")
    } else {
        Path::new("data/tokenizer.json").to_path_buf()
    };

    if !tok_path.exists() {
        eprintln!("Tokenizer file '{}' not found.", tok_path.display());
        std::process::exit(1);
    }
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let encoding = tokenizer.encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        token_ids.push(1); // default BOS token
    }

    println!("Initial token count: {}", token_ids.len());

    let device = Default::default();

    let (model, vocab_size) = if model_dir.exists() && model_dir.join("model.safetensors").exists() {
        println!("\n[Loading Pretrained SmolLM2 Open Weights into Nexus Engine...]");
        let m = import_hf_to_llama::<Backend>(model_dir, &device)?;
        let v = m.vocab_size;
        println!("✓ Successfully loaded {} LLaMA blocks (vocab={v})!", m.n_blocks());
        (m, v)
    } else {
        println!("\n[Initializing Random Weight Baseline...]");
        let v = 50_257usize;
        let cfg = LlamaConfig::new(v, 256, 8, 4)
            .with_max_seq_len(256)
            .with_d_ff(512);
        let m = cfg.init::<Backend>(&device);
        (m, v)
    };

    println!("\n[Generating text...]");
    print!("{prompt}");
    std::io::Write::flush(&mut std::io::stdout())?;

    for _ in 0..max_tokens {
        let cur_len = token_ids.len();
        let input_slice = if cur_len > 128 {
            &token_ids[cur_len - 128..]
        } else {
            &token_ids[..]
        };

        let seq_len = input_slice.len();
        let raw_i64: Vec<i64> = input_slice.iter().map(|&x| (x as usize % vocab_size) as i64).collect();
        let tensor_data = burn::tensor::TensorData::new(raw_i64, [1, seq_len]);
        let input_tensor = Tensor::<Backend, 2, burn::tensor::Int>::from_data(tensor_data, &device);

        check_token_ids(&input_tensor, vocab_size);
        let logits = model.forward(input_tensor);

        // Extract logits for last position: [1, vocab_size]
        let last_logits = logits.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
        let raw_logits = last_logits.into_data();
        let slice: Vec<f32> = raw_logits.iter::<f32>().collect();

        // Sample next token (greedy or temperature scaled)
        let next_token_id = if temperature < 1e-4 {
            slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
        } else {
            // Temperature scaled sampling with top-k filtering (k=50)
            let max_logit = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut pairs: Vec<(usize, f32)> = slice
                .iter()
                .enumerate()
                .map(|(idx, &l)| (idx, ((l - max_logit) / temperature).exp()))
                .collect();

            // Top-50 filtering
            pairs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            pairs.truncate(50);

            let sum_exp: f32 = pairs.iter().map(|(_, e)| e).sum();
            let r: f32 = rand::random::<f32>() * sum_exp;
            let mut acc = 0.0f32;
            let mut picked = pairs.first().map(|(idx, _)| *idx as u32).unwrap_or(0);
            for &(idx, e) in &pairs {
                acc += e;
                if r <= acc {
                    picked = idx as u32;
                    break;
                }
            }
            picked
        };

        token_ids.push(next_token_id);
        if let Ok(decoded_chunk) = tokenizer.decode(&[next_token_id], false) {
            print!("{decoded_chunk}");
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }

    println!("\n\n✓ Generation complete.");
    Ok(())
}
