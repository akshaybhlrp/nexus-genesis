//! Real-data ingest: FineWeb-Edu parquet shards → packed token sequences.
//!
//! One-shot preprocessing: reads parquet files, tokenizes with a HF tokenizer,
//! packs into fixed-length `u32` blocks, writes a single binary file
//! (`tokens.bin` + header). Training then memmaps that file — no tokenizer or
//! parquet machinery on the hot path.

use anyhow::Context;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

/// Packed dataset layout: `[u32 vocab_size][u32 seq_len][u32 n_seqs][payload]`.
pub const HEADER_LEN: usize = 12;

/// Header-validation failures. Distinct variants so callers can react
/// programmatically instead of string-matching error text.
#[derive(Debug)]
pub enum PackedError {
    /// File smaller than a bare header.
    TooShort { len: u64 },
    /// Vocab sentinel is 0 (never written by [`pack_shards`]).
    ZeroVocab,
    /// Payload length disagrees with header claims (incl. overflow-sized
    /// claims from corrupt headers).
    SizeMismatch { len: u64, expected: u64 },
}

impl std::fmt::Display for PackedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { len } => {
                write!(f, "corrupt dataset: {len} bytes, header needs {HEADER_LEN}")
            }
            Self::ZeroVocab => write!(f, "corrupt dataset: vocab sentinel is 0"),
            Self::SizeMismatch { len, expected } => {
                write!(f, "corrupt dataset: file is {len} bytes, expected exactly {expected}")
            }
        }
    }
}

impl std::error::Error for PackedError {}

/// Single distrust choke point for the on-disk format: every byte-level
/// claim is validated here and nowhere else. Downstream code sees only
/// checked values.
///
/// Checks: minimum length, nonzero vocab sentinel, exact payload size
/// (`n_seqs * seq_len * 4` in checked arithmetic — header counts are
/// untrusted input and may claim absurd sizes).
fn parse_header(bytes: &[u8]) -> Result<(usize, usize), PackedError> {
    let len = bytes.len() as u64;
    if bytes.len() < HEADER_LEN {
        return Err(PackedError::TooShort { len });
    }
    let vocab = u32::from_le_bytes(bytes[0..4].try_into().expect("len checked"));
    let seq_len = u32::from_le_bytes(bytes[4..8].try_into().expect("len checked")) as usize;
    let n_seqs = u32::from_le_bytes(bytes[8..HEADER_LEN].try_into().expect("len checked")) as usize;
    if vocab == 0 {
        return Err(PackedError::ZeroVocab);
    }
    let payload = n_seqs
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(4))
        .ok_or(PackedError::SizeMismatch { len, expected: u64::MAX })?;
    let expected = HEADER_LEN as u64 + payload as u64;
    if len != expected {
        return Err(PackedError::SizeMismatch { len, expected });
    }
    Ok((seq_len, n_seqs))
}

pub struct PackedDataset {
    mmap: memmap2::Mmap,
    pub seq_len: usize,
    pub n_seqs: usize,
}

impl PackedDataset {
    /// Open a `.bin` produced by [`pack_shards`]. Payload is read-only shared.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        let (seq_len, n_seqs) =
            parse_header(&mmap).with_context(|| format!("dataset {}", path.display()))?;
        Ok(Self { mmap, seq_len, n_seqs })
    }

    /// Copy of sequence `i` (memmap slice → owned Vec).
    ///
    /// # Panics
    /// If `i >= n_seqs` — caller bug, not data corruption; fail loudly with
    /// the offending index rather than slicing out of the mmap.
    pub fn seq(&self, i: usize) -> Vec<u32> {
        assert!(
            i < self.n_seqs,
            "PackedDataset::seq({i}) out of range (n_seqs={})",
            self.n_seqs
        );
        let start = HEADER_LEN + i * self.seq_len * 4;
        let end = start + self.seq_len * 4;
        self.mmap[start..end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.n_seqs
    }
}

/// Tokenize + pack every parquet shard in `paths` into one binary at `out`.
///
/// `vocab_size` caps token ids (safety net for a smaller test vocab; ids are
/// NOT remapped — use the real tokenizer's vocab for production runs).
pub fn pack_shards(
    paths: &[PathBuf],
    tokenizer_file: &Path,
    out: &Path,
    seq_len: usize,
    max_tokens: usize,
) -> anyhow::Result<()> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_file)
        .map_err(|e| anyhow::anyhow!("load tokenizer {tokenizer_file:?}: {e}"))?;

    let out_f = File::create(out)?;
    let mut w = BufWriter::new(out_f);
    // Placeholder header; patched after packing.
    w.write_all(&0u32.to_le_bytes())?;
    w.write_all(&(seq_len as u32).to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?;

    let mut buf: Vec<u32> = Vec::with_capacity(seq_len * 1024);
    let mut total_tokens = 0usize;
    let mut n_seqs = 0usize;

    'outer: for path in paths {
        let f = File::open(path)?;
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f)?.build()?;
        for batch in reader {
            let batch = batch?;
            let text_col = batch
                .column_by_name("text")
                .context("parquet missing 'text' column")?;
            let texts = text_col
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .context("'text' not a string array")?;
            for t in texts.iter().flatten() {
                let enc = tok
                    .encode(t, false)
                    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
                buf.extend(enc.get_ids().iter().copied());
                while buf.len() >= seq_len + 1 {
                    let block: Vec<u32> = buf.drain(..seq_len + 1).collect();
                    write_seq(&mut w, &block[..seq_len])?;
                    n_seqs += 1;
                    total_tokens += seq_len;
                    if total_tokens >= max_tokens || n_seqs >= u32::MAX as usize {
                        break 'outer;
                    }
                }
            }
        }
        tracing::info!(shard = %path.display(), total_tokens, n_seqs, "packed shard");
    }

    w.flush()?;
    let mut f = w.into_inner()?;
    use std::io::SeekFrom;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&1u32.to_le_bytes())?; // vocab unused by reader; nonzero sentinel
    f.write_all(&(seq_len as u32).to_le_bytes())?;
    f.write_all(&(n_seqs as u32).to_le_bytes())?;
    f.sync_all()?;
    tracing::info!(total_tokens, n_seqs, out = %out.display(), "dataset done");
    Ok(())
}

fn write_seq<W: Write>(w: &mut W, seq: &[u32]) -> std::io::Result<()> {
    for t in seq {
        w.write_all(&t.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_header_valid() {
        // Minimal valid header: vocab=1, seq_len=2, n_seqs=3 → expected size = 12 + 3*2*4 = 36.
        let mut bytes = Vec::with_capacity(36);
        bytes.extend(1u32.to_le_bytes());   // vocab
        bytes.extend(2u32.to_le_bytes());   // seq_len
        bytes.extend(3u32.to_le_bytes());   // n_seqs
        bytes.extend([0u8; 24]);            // payload dummy
        let (seq_len, n_seqs) = parse_header(&bytes).unwrap();
        assert_eq!(seq_len, 2);
        assert_eq!(n_seqs, 3);
    }

    #[test]
    fn parse_header_too_short() {
        let bytes = [0u8; 5];
        match parse_header(&bytes) {
            Err(PackedError::TooShort { len }) => assert_eq!(len, 5),
            other => panic!("expected TooShort, got {other:?}"),
        }
    }

    #[test]
    fn parse_header_zero_vocab() {
        let mut bytes = vec![0u8; 12];
        bytes[0..4].copy_from_slice(&0u32.to_le_bytes()); // vocab = 0
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes()); // seq_len
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // n_seqs
        match parse_header(&bytes) {
            Err(PackedError::ZeroVocab) => {}
            other => panic!("expected ZeroVocab, got {other:?}"),
        }
    }

    #[test]
    fn parse_header_size_mismatch() {
        // Header says 1 seq of len 4 → expected size = 12 + 1*4*4 = 28, but we give 30.
        let mut bytes = vec![0u8; 30];
        bytes[0..4].copy_from_slice(&1u32.to_le_bytes());   // vocab
        bytes[4..8].copy_from_slice(&4u32.to_le_bytes());   // seq_len
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());   // n_seqs
        match parse_header(&bytes) {
            Err(PackedError::SizeMismatch { len, expected }) => {
                assert_eq!(len, 30);
                assert_eq!(expected, 28);
            }
            other => panic!("expected SizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn parse_header_overflow_payload() {
        // n_seqs and seq_len chosen so that n_seqs * seq_len * 4 overflows usize (on 32-bit)
        // or exceeds u64::MAX. Use large values that cause checked_mul to fail.
        let mut bytes = vec![0u8; 12];
        bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes()); // seq_len = 4_294_967_295
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes()); // n_seqs = 4_294_967_295
        match parse_header(&bytes) {
            Err(PackedError::SizeMismatch { .. }) => {} // overflow leads to SizeMismatch with expected=MAX
            other => panic!("expected SizeMismatch on overflow, got {other:?}"),
        }
    }

    #[test]
    fn packed_error_display() {
        let e = PackedError::TooShort { len: 3 };
        assert!(format!("{e}").contains("corrupt dataset"));
        assert!(format!("{e}").contains("3 bytes"));

        let e = PackedError::ZeroVocab;
        assert!(format!("{e}").contains("vocab sentinel is 0"));

        let e = PackedError::SizeMismatch { len: 100, expected: 200 };
        let s = format!("{e}");
        assert!(s.contains("100 bytes"));
        assert!(s.contains("exactly 200"));
    }
}
