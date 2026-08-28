//! Interactive terminal REPL / chat interface for Nexus digital organism.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-chat -- [--model data/models/smollm2-135m] [--temperature 0.7]

use burn::tensor::Tensor;
use nexus_core::import::import_hf_to_llama;
use nexus_core::model::{LlamaConfig, check_token_ids};
use nexus_core::moe::{upcycle_dense, RouterConfig};
use std::io::{self, BufRead, Write};
use std::path::Path;
use tokenizers::Tokenizer;

type Backend = burn::backend::Wgpu;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir_str = "data/models/smollm2-135m".to_string();
    let mut temperature = 0.7f32;
    let max_tokens = 64usize;

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            i += 1;
            model_dir_str = args[i].clone();
        } else if args[i] == "--temperature" && i + 1 < args.len() {
            i += 1;
            if let Ok(temp) = args[i].parse() {
                temperature = temp;
            }
        }
        i += 1;
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                   NEXUS INTERACTIVE REPL                       ║");
    println!("║       Hardware-Agnostic Self-Evolving MoE Brain Terminal       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Type 'exit', 'quit', or Ctrl-D to leave.\n");

    let model_dir = Path::new(&model_dir_str);
    let tok_path = if model_dir.join("tokenizer.json").exists() {
        model_dir.join("tokenizer.json")
    } else {
        Path::new("data/tokenizer.json").to_path_buf()
    };

    if !tok_path.exists() {
        eprintln!("Tokenizer '{}' not found.", tok_path.display());
        std::process::exit(1);
    }
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    let device = Default::default();

    println!("[1/2] Loading Brain Weights into GPU Engine...");
    let (dense_model, vocab_size) = if model_dir.exists() && model_dir.join("model.safetensors").exists() {
        let m = import_hf_to_llama::<Backend>(model_dir, &device)?;
        let v = m.vocab_size;
        println!("  ✓ Loaded {} LLaMA blocks from {} (vocab={v})", m.n_blocks(), model_dir.display());
        (m, v)
    } else {
        let v = 50_257usize;
        let cfg = LlamaConfig::new(v, 256, 8, 4)
            .with_max_seq_len(256)
            .with_d_ff(512);
        let m = cfg.init::<Backend>(&device);
        println!("  ✓ Initialized default baseline model (vocab={v})");
        (m, v)
    };

    println!("[2/2] Upcycling into Hierarchical MoE Structure...");
    let router_cfg = RouterConfig::new(4);
    let moe_model = upcycle_dense(&dense_model, &router_cfg);
    println!("  ✓ MoE Brain Ready: {} blocks × {} experts each (Total: {} experts)",
        moe_model.blocks.len(),
        moe_model.blocks[0].experts.len(),
        moe_model.blocks.len() * moe_model.blocks[0].experts.len()
    );

    println!("\nNexus is ready for interaction. Enter your prompt below:\n");

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("nexus> ");
        io::stdout().flush()?;

        let line = match lines.next() {
            Some(Ok(l)) => l.trim().to_string(),
            Some(Err(e)) => {
                eprintln!("Error reading stdin: {e}");
                break;
            }
            None => break, // EOF
        };

        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            println!("Shutting down Nexus session. Goodbye.");
            break;
        }

        let encoding = match tokenizer.encode(line.as_str(), false) {
            Ok(enc) => enc,
            Err(e) => {
                eprintln!("Tokenization error: {e}");
                continue;
            }
        };

        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            token_ids.push(1);
        }

        print!("Nexus: ");
        io::stdout().flush()?;

        let start_time = std::time::Instant::now();
        let mut generated_count = 0usize;
        let mut last_entropy = 0.0f32;
        let mut top_experts: Vec<usize> = Vec::new();

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
            let (logits, _balance, entropy, routes) = moe_model.forward_with_balance(input_tensor);
            last_entropy = entropy;

            if let Some(first_route) = routes.first() {
                top_experts = first_route
                    .expert_mass
                    .iter()
                    .enumerate()
                    .filter(|(_, &m)| m > 0.1)
                    .map(|(idx, _)| idx)
                    .collect();
            }

            // Extract last logits
            let last_logits = logits.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
            let raw_logits = last_logits.into_data();
            let slice: Vec<f32> = raw_logits.iter::<f32>().collect();

            let max_logit = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut pairs: Vec<(usize, f32)> = slice
                .iter()
                .enumerate()
                .map(|(idx, &l)| (idx, ((l - max_logit) / temperature).exp()))
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

            token_ids.push(picked);
            generated_count += 1;

            if let Ok(decoded) = tokenizer.decode(&[picked], false) {
                print!("{decoded}");
                io::stdout().flush()?;
            }

            // Stop at EOS / newline
            if picked == 2 || picked == 0 || (generated_count > 10 && picked == 13) {
                break;
            }
        }

        let elapsed = start_time.elapsed().as_secs_f32();
        let tok_per_sec = if elapsed > 0.0 { generated_count as f32 / elapsed } else { 0.0 };

        println!();
        println!("  └─ [Telemetry] Tokens: {generated_count} | Speed: {tok_per_sec:.1} tok/s | Entropy: {last_entropy:.3} | Active Experts: {:?}",
            top_experts
        );
        println!();
    }

    Ok(())
}
