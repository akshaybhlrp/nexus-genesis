//! Shared test helpers: tiny configs, deterministic token batches, packed
//! dataset fixtures written straight to disk in the on-disk binary layout.

#![allow(dead_code)]

use nexus_core::data::{HEADER_LEN, PackedDataset};
use nexus_core::model::LlamaConfig;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

/// CPU backend used by tests on headless CI runners.
pub type TB = burn::backend::NdArray;

/// Tiny config: fast init + forward, small enough for many iterations.
pub fn tiny_cfg() -> LlamaConfig {
    LlamaConfig::new(64, 32, 4, 2)
        .with_max_seq_len(16)
        .with_d_ff(64)
}

pub fn device() -> burn::backend::ndarray::NdArrayDevice {
    Default::default()
}

/// Deterministic `[b, s]` int batch, values in `1..vocab`.
pub fn tokens(vocab: usize, b: usize, s: usize) -> burn::tensor::Tensor<TB, 2, burn::tensor::Int> {
    // Ids stay in [0, vocab): the forward() OOV guard rejects id == vocab.
    let data: Vec<u32> = (0..(b * s) as u32).map(|i| i % (vocab as u32).max(1)).collect();
    burn::tensor::Tensor::<TB, 1, burn::tensor::Int>::from_ints(data.as_slice(), &device())
        .reshape([b, s])
}

pub struct TempBin(pub tempfile::TempDir);

impl TempBin {
    /// Empty dir under the cargo target tree; auto-removed at drop.
    pub fn new() -> Self {
        Self(tempfile::tempdir().expect("tempdir"))
    }
    pub fn path(&self) -> PathBuf {
        self.0.path().join("tokens.bin")
    }

    /// Write a well-formed packed file with `n` sequences of length `seq_len`,
    /// token values drawn deterministically from `0..vocab`.
    pub fn write_valid(&self, n_seqs: usize, seq_len: usize, vocab: u32) -> PathBuf {
        let mut f = File::create(self.path()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // vocab sentinel
        f.write_all(&(seq_len as u32).to_le_bytes()).unwrap();
        f.write_all(&(n_seqs as u32).to_le_bytes()).unwrap();
        for i in 0..(n_seqs * seq_len) as u32 {
            let v = ((i as u64 * 2654435761) % vocab.max(1) as u64) as u32;
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.sync_all().unwrap();
        self.path()
    }

    /// Write raw bytes verbatim — for corrupt-header / truncated-file cases.
    pub fn write_raw(&self, bytes: &[u8]) -> PathBuf {
        std::fs::write(self.path(), bytes).unwrap();
        self.path()
    }
}

impl Default for TempBin {
    fn default() -> Self {
        Self::new()
    }
}

/// Open helper returning Arc (eval API takes Arc).
pub fn open_arc(p: &std::path::Path) -> Arc<PackedDataset> {
    Arc::new(PackedDataset::open(p).unwrap())
}

pub const HDR: usize = HEADER_LEN;
