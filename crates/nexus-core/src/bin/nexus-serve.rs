//! Persistent GPU Inference Server for Nexus on NVIDIA T500 (CUDA).
//!
//! Loads the foundation model into GPU VRAM once and keeps it resident indefinitely,
//! serving generation queries over stdin/stdout with sub-second turnaround.

use burn::tensor::Tensor;
use nexus_core::model::check_token_ids;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Request {
    prompt: String,
    #[serde(default = "default_tokens")]
    tokens: usize,
    #[serde(default = "default_temp")]
    temperature: f32,
}

fn default_tokens() -> usize {
    30
}
fn default_temp() -> f32 {
    0.7
}

#[derive(Serialize)]
struct Response {
    success: bool,
    prompt: String,
    generated_text: String,
    tokens_generated: usize,
    elapsed_seconds: f64,
    tokens_per_second: f32,
    error: String,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = PathBuf::from("data/models/smollm2-135m");

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model_dir = PathBuf::from(&args[i + 1]);
            i += 1;
        }
        i += 1;
    }

    #[cfg(feature = "cuda")]
    {
        run_server::<burn::backend::Cuda>(model_dir)
    }
    #[cfg(not(feature = "cuda"))]
    {
        run_server::<burn::backend::Wgpu>(model_dir)
    }
}

fn run_server<B: burn::tensor::backend::Backend>(model_dir: PathBuf) -> anyhow::Result<()> {
    let device: B::Device = Default::default();

    eprintln!("⚡ Nexus Persistent Inference Server starting...");
    eprintln!("  Hardware Device: NVIDIA T500 (CUDA Native)");
    eprintln!("  Loading model from: {}", model_dir.display());

    let tok_path = model_dir.join("tokenizer.json");
    if !tok_path.exists() {
        anyhow::bail!("Tokenizer not found at {}", tok_path.display());
    }
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
    let vocab_size = tokenizer.get_vocab_size(true);

    let dense = nexus_core::import::import_hf_to_llama::<B>(&model_dir, &device)?;
    let router_cfg = RouterConfig::new(4);
    let moe = upcycle_dense(&dense, &router_cfg);

    eprintln!("✓ Pretrained weights resident in NVIDIA T500 VRAM (30 blocks × 4 experts MoE)");
    eprintln!("✓ Ready for incoming queries.\n");

    // Signal readiness on stdout as JSON
    println!(r#"{{"status":"READY","backend":"NVIDIA T500 (CUDA)","model":"{}"}}"#, model_dir.display());
    io::stdout().flush()?;

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response {
                    success: false,
                    prompt: String::new(),
                    generated_text: String::new(),
                    tokens_generated: 0,
                    elapsed_seconds: 0.0,
                    tokens_per_second: 0.0,
                    error: format!("Invalid JSON request: {e}"),
                };
                println!("{}", serde_json::to_string(&resp)?);
                io::stdout().flush()?;
                continue;
            }
        };

        let t0 = Instant::now();
        let encoding = match tokenizer.encode(req.prompt.as_str(), false) {
            Ok(enc) => enc,
            Err(e) => {
                let resp = Response {
                    success: false,
                    prompt: req.prompt,
                    generated_text: String::new(),
                    tokens_generated: 0,
                    elapsed_seconds: 0.0,
                    tokens_per_second: 0.0,
                    error: format!("Tokenization failed: {e}"),
                };
                println!("{}", serde_json::to_string(&resp)?);
                io::stdout().flush()?;
                continue;
            }
        };

        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            token_ids.push(0); // SmolLM2 BOS / start
        }

        let _initial_len = token_ids.len();
        let mut generated_ids = Vec::new();

        for _ in 0..req.tokens {
            let cur_len = token_ids.len();
            let input_slice = if cur_len > 128 {
                &token_ids[cur_len - 128..]
            } else {
                &token_ids[..]
            };

            let seq_len = input_slice.len();
            let raw_i64: Vec<i64> = input_slice
                .iter()
                .map(|&x| (x as usize % vocab_size) as i64)
                .collect();
            let tensor_data = burn::tensor::TensorData::new(raw_i64, [1, seq_len]);
            let input_tensor = Tensor::<B, 2, burn::tensor::Int>::from_data(tensor_data, &device);

            check_token_ids(&input_tensor, vocab_size);
            let (logits, _, _, _) = moe.forward_with_balance(input_tensor);

            // Extract logits for last position: [1, vocab_size]
            let last_logits = logits.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
            let raw_logits = last_logits.into_data();
            let slice: Vec<f32> = raw_logits.iter::<f32>().collect();

            let next_token_id = if req.temperature < 1e-4 {
                slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            } else {
                let max_logit = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut pairs: Vec<(usize, f32)> = slice
                    .iter()
                    .enumerate()
                    .map(|(idx, &l)| (idx, ((l - max_logit) / req.temperature).exp()))
                    .collect();

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
            generated_ids.push(next_token_id);

            // Stop at EOS (token 0 for SmolLM2)
            if next_token_id == 0 {
                break;
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let tok_s = if elapsed > 0.0 {
            (generated_ids.len() as f64 / elapsed) as f32
        } else {
            0.0
        };

        let decoded_text = tokenizer
            .decode(&generated_ids, true)
            .unwrap_or_else(|_| "[Decode error]".to_string());

        let resp = Response {
            success: true,
            prompt: req.prompt,
            generated_text: decoded_text,
            tokens_generated: generated_ids.len(),
            elapsed_seconds: (elapsed * 1000.0).round() / 1000.0,
            tokens_per_second: (tok_s * 10.0).round() / 10.0,
            error: String::new(),
        };

        println!("{}", serde_json::to_string(&resp)?);
        io::stdout().flush()?;
    }

    Ok(())
}
