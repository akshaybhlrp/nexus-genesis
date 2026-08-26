//! Pack FineWeb-Edu parquet shards into a binary token dataset.
//!
//! Usage:
//!   cargo run --release -p nexus-core --bin nexus-pack-data -- \
//!     --tokenizer tokenizer.json --out data/fineweb.bin \
//!     [--seq-len 512] [--max-tokens 50000000] <shard1.parquet> [shard2.parquet ...]

use anyhow::Context;
use nexus_core::data::pack_shards;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut args = std::env::args().skip(1);
    let mut tokenizer = PathBuf::from("tokenizer.json");
    let mut out = PathBuf::from("data/fineweb.bin");
    let mut seq_len = 512usize;
    let mut max_tokens = 50_000_000usize;
    let mut shards: Vec<PathBuf> = Vec::new();

    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokenizer" => tokenizer = args.next().context("--tokenizer needs value")?.into(),
            "--out" => out = args.next().context("--out needs value")?.into(),
            "--seq-len" => seq_len = args.next().context("--seq-len needs value")?.parse()?,
            "--max-tokens" => max_tokens = args.next().context("--max-tokens needs value")?.parse()?,
            s if s.starts_with('-') => anyhow::bail!("unknown flag {s}"),
            s => shards.push(s.into()),
        }
    }
    anyhow::ensure!(!shards.is_empty(), "no parquet shards given");
    for s in &shards {
        anyhow::ensure!(s.exists(), "missing shard {}", s.display());
    }

    pack_shards(&shards, &tokenizer, &out, seq_len, max_tokens)?;
    println!("wrote {}", out.display());
    Ok(())
}
