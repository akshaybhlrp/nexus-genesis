//! Interactive terminal REPL / chat interface for native Scratch-Trained Nexus MoE digital organism.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-chat -- [--temperature 0.7] [--cpu]

use burn::tensor::Tensor;
use nexus_core::model::{LlamaConfig, check_token_ids};
use nexus_core::moe::{upcycle_dense, RouterConfig};
use std::io::{self, BufRead, Write};
use std::path::Path;
use tokenizers::Tokenizer;

fn run_chat<B: burn::tensor::backend::Backend>(temperature: f32, tokenizer: Tokenizer) -> anyhow::Result<()> {
    let device = Default::default();
    let vocab_size = 50_257usize;
    let max_tokens = 64usize;

    println!("[1/2] Initializing Scaled Scratch LLaMA Core (85M MoE Target)...");
    let cfg = LlamaConfig::new(vocab_size, 384, 12, 8)
        .with_max_seq_len(256)
        .with_d_ff(1024);
    let dense_model = cfg.init::<B>(&device);
    println!("  ✓ Initialized 8 blocks (vocab={vocab_size}, d_model=384, d_ff=1024, heads=12)");

    println!("[2/2] Upcycling into Hierarchical MoE Structure (64 Experts)...");
    let router_cfg = RouterConfig::new(8);
    let mut moe_model = upcycle_dense(&dense_model, &router_cfg);

    // Check for trained evolved experts in L3 SSD Warehouse
    let wh_cfg = nexus_memory::WarehouseConfig::default();
    if let Ok(warehouse) = nexus_memory::ExpertWarehouse::<B>::new(wh_cfg) {
        if let Ok(count) = nexus_core::tiered::load_model_from_warehouse(&mut moe_model, &warehouse, &device) {
            if count > 0 {
                println!("  ✓ Successfully loaded {count} evolved experts from L3 SSD Warehouse!");
            } else {
                println!("  ℹ No evolved experts in warehouse yet. Starting fresh.");
            }
        }
    }

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
        let mut active_experts: Vec<usize> = Vec::new();

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
            let input_tensor = Tensor::<B, 2, burn::tensor::Int>::from_data(tensor_data, &device);

            check_token_ids(&input_tensor, vocab_size);
            let (logits, _balance, entropy, routes) = moe_model.forward_with_balance(input_tensor);
            last_entropy = entropy;

            if let Some(first_route) = routes.first() {
                active_experts = first_route
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

            // Stop at EOS
            if picked == 2 || picked == 0 {
                break;
            }
        }

        let elapsed = start_time.elapsed().as_secs_f32();
        let tok_per_sec = if elapsed > 0.0 { generated_count as f32 / elapsed } else { 0.0 };

        println!();
        println!("  └─ [Telemetry] Tokens: {generated_count} | Speed: {tok_per_sec:.1} tok/s | Entropy: {last_entropy:.3} | Active Experts: {:?}",
            active_experts
        );
        println!();
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut temperature = 0.7f32;
    let mut use_cpu = false;

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--temperature" && i + 1 < args.len() {
            i += 1;
            if let Ok(temp) = args[i].parse() {
                temperature = temp;
            }
        } else if args[i] == "--cpu" {
            use_cpu = true;
        }
        i += 1;
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              NEXUS SCRATCH-TRAINED MOE REPL                    ║");
    println!("║       Hardware-Agnostic Self-Evolving MoE Brain Terminal       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Backend: {} | Temperature: {temperature:.2}", if use_cpu { "CPU (NdArray)" } else { "GPU (Wgpu)" });
    println!("Type 'exit', 'quit', or Ctrl-D to leave.\n");

    let tok_path = Path::new("data/tokenizer.json");
    if !tok_path.exists() {
        eprintln!("Tokenizer '{}' not found.", tok_path.display());
        std::process::exit(1);
    }
    let tokenizer = Tokenizer::from_file(tok_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    if use_cpu {
        run_chat::<burn::backend::NdArray>(temperature, tokenizer)
    } else {
        run_chat::<burn::backend::Wgpu>(temperature, tokenizer)
    }
}
