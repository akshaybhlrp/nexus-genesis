//! data.rs coverage: PackedDataset parsing (valid + corrupt), pack_shards.

mod common;

use common::*;
use nexus_core::data::{HEADER_LEN, PackedDataset, pack_shards};
use std::fs::File;

// ---------- PackedDataset::open — valid files ----------

#[test]
fn open_valid_dataset_reads_header() {
    let t = TempBin::new();
    let p = t.write_valid(5, 12, 100);
    let ds = PackedDataset::open(&p).unwrap();
    assert_eq!(ds.len(), 5);
    assert_eq!(ds.seq_len, 12);
}

#[test]
fn seq_roundtrips_written_tokens() {
    let t = TempBin::new();
    // Same generator as write_valid: tok(i) = (i*2654435761) % vocab.
    let (n, sl, vocab) = (3usize, 4usize, 50u32);
    let p = t.write_valid(n, sl, vocab);
    let ds = PackedDataset::open(&p).unwrap();
    for i in 0..n {
        for j in 0..sl {
            let gidx = (i * sl + j) as u64;
            let expect = ((gidx * 2654435761) % vocab.max(1) as u64) as u32;
            assert_eq!(ds.seq(i)[j], expect, "seq {i} pos {j}");
        }
    }
}

#[test]
fn seq_returns_owned_copy_not_alias() {
    let t = TempBin::new();
    let ds = PackedDataset::open(&t.write_valid(2, 4, 8)).unwrap();
    let a = ds.seq(0);
    let b = ds.seq(0);
    assert_eq!(a, b);
    // Two independent allocations.
    assert_ne!(a.as_ptr(), b.as_ptr());
}

#[test]
fn len_zero_seqs_is_zero() {
    let t = TempBin::new();
    let ds = PackedDataset::open(&t.write_valid(0, 8, 16)).unwrap();
    assert_eq!(ds.len(), 0);
}

#[test]
fn open_last_seq_fully_in_bounds() {
    let t = TempBin::new();
    let n = 3usize;
    let ds = PackedDataset::open(&t.write_valid(n, 6, 32)).unwrap();
    assert_eq!(ds.seq(n - 1).len(), 6);
}

#[test]
fn header_len_constant_is_twelve() {
    assert_eq!(HEADER_LEN, 12);
}

#[test]
fn open_missing_file_is_error_with_context() {
    let msg = match PackedDataset::open(std::path::Path::new("/nonexistent/tokens.bin")) {
        Ok(_) => panic!("missing file must not open"),
        Err(e) => format!("{e:#}"),
    };
    assert!(msg.contains("open"), "error should name the path: {msg}");
    assert!(msg.contains("/nonexistent"), "{msg}");
}

// ---------- PackedDataset::open — corrupt / hostile files ----------

#[test]
fn open_empty_file_errors() {
    let t = TempBin::new();
    let p = t.write_raw(&[]);
    assert!(PackedDataset::open(&p).is_err(), "empty file must not open");
}

#[test]
fn open_truncated_header_errors() {
    let t = TempBin::new();
    let p = t.write_raw(&[1u8, 0, 0, 0, 0]);
    assert!(PackedDataset::open(&p).is_err());
}

#[test]
fn open_truncated_payload_errors() {
    let t = TempBin::new();
    // Header claims 10×8-seq but payload has only 1 token.
    let mut b = Vec::new();
    b.extend(1u32.to_le_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend(10u32.to_le_bytes());
    b.extend(42u32.to_le_bytes()); // 1 token of 80 expected
    let p = t.write_raw(&b);
    assert!(PackedDataset::open(&p).is_err(), "short payload must be rejected");
}

#[test]
fn open_exact_size_passes_boundary_check() {
    let t = TempBin::new();
    // Payload exactly HEADER_LEN + n*s*4 — the `>=` boundary itself.
    let (n, sl) = (2usize, 4usize);
    let mut b = Vec::new();
    b.extend(1u32.to_le_bytes());
    b.extend((sl as u32).to_le_bytes());
    b.extend((n as u32).to_le_bytes());
    for i in 0..(n * sl) as u32 {
        b.extend(i.to_le_bytes());
    }
    let p = t.write_raw(&b);
    let ds = PackedDataset::open(&p).unwrap();
    assert_eq!(ds.len(), n);
}

#[test]
fn open_zero_vocab_rejected() {
    let t = TempBin::new();
    let mut b = Vec::new();
    b.extend(0u32.to_le_bytes()); // vocab sentinel 0 → invalid per ensure!
    b.extend(4u32.to_le_bytes());
    b.extend(1u32.to_le_bytes());
    b.extend([0u8; 16]);
    let p = t.write_raw(&b);
    assert!(PackedDataset::open(&p).is_err(), "vocab=0 sentinel must be rejected");
}

#[test]
fn open_huge_n_seqs_overflow_claim_rejected() {
    let t = TempBin::new();
    // Header claims ~4 billion seqs; file is tiny. Must reject, not OOM/hang.
    let mut b = Vec::new();
    b.extend(1u32.to_le_bytes());
    b.extend(512u32.to_le_bytes());
    b.extend(u32::MAX.to_le_bytes());
    b.extend([0u8; 32]);
    let p = t.write_raw(&b);
    assert!(
        PackedDataset::open(&p).is_err(),
        "absurd n_seqs claim must fail the size check"
    );
}

#[test]
fn open_directory_path_errors_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    assert!(PackedDataset::open(dir.path()).is_err());
}

#[test]
fn corrupt_file_error_mentions_corruption() {
    let t = TempBin::new();
    let p = t.write_raw(&[9u8; 4]); // valid-ish start, wrong total length
    let msg = match PackedDataset::open(&p) {
        Ok(_) => panic!("corrupt file must not open"),
        Err(e) => format!("{:#}", anyhow::Chain::new(e.as_ref()).map(|c| c.to_string()).collect::<Vec<_>>().join(": ")),
    };
    assert!(msg.contains("corrupt"), "got: {msg}");
}

// ---------- pack_shards ----------

/// Minimal single-row parquet with one string column named "text".
fn write_parquet(path: &std::path::Path, texts: &[&str]) -> anyhow::Result<()> {
    use arrow::array::StringArray;
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "text",
        arrow::datatypes::DataType::Utf8,
        false,
    )]);
    let arr = StringArray::from(texts.to_vec());
    let batch = RecordBatch::try_new(std::sync::Arc::new(schema), vec![std::sync::Arc::new(arr)])?;
    let f = File::create(path)?;
    let mut w = ArrowWriter::try_new(f, batch.schema(), None)?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

/// Tiny byte-pair-free tokenizer.json (word-level model over our words).
const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": { "type": "WordLevel", "vocab": {"hello": 0, "world": 1, "foo": 2, "bar": 3}, "unk_token": "[UNK]" }
}"#;

fn setup_pack() -> (TempBin, std::path::PathBuf, std::path::PathBuf) {
    let t = TempBin::new();
    let tok_path = t.0.path().join("tokenizer.json");
    std::fs::write(&tok_path, TOKENIZER_JSON).unwrap();
    let shard = t.0.path().join("shard.parquet");
    write_parquet(
        &shard,
        &["hello world foo bar", "hello hello world", "foo bar bar bar"],
    )
    .unwrap();
    (t, tok_path, shard)
}

#[test]
fn pack_shards_writes_parseable_dataset() {
    let (t, tok, shard) = setup_pack();
    let out = t.path();
    pack_shards(&[shard], &tok, &out, 4, 100).unwrap();
    let ds = PackedDataset::open(&out).unwrap();
    assert_eq!(ds.seq_len, 4);
    assert!(ds.len() > 0);
    // All tokens within the 4-word vocab.
    for i in 0..ds.len() {
        assert!(ds.seq(i).iter().all(|&x| x < 4));
    }
}

#[test]
fn pack_shards_respects_max_tokens_cap() {
    let (t, tok, shard) = setup_pack();
    let out = t.path();
    // Cap at fewer tokens than the corpus holds.
    pack_shards(&[shard], &tok, &out, 2, 4).unwrap();
    let ds = PackedDataset::open(&out).unwrap();
    assert_eq!(ds.len() * ds.seq_len, 4, "must stop at exactly max_tokens");
}

#[test]
fn pack_shards_multiple_shards_concatenate() {
    let (t, tok, shard) = setup_pack();
    let shard2 = t.0.path().join("shard2.parquet");
    write_parquet(&shard2, &["bar foo"]).unwrap();
    let out = t.path();
    pack_shards(&[shard, shard2], &tok, &out, 3, 1000).unwrap();
    assert!(PackedDataset::open(&out).unwrap().len() > 0);
}

#[test]
fn pack_shards_missing_tokenizer_errors() {
    let (t, _tok, shard) = setup_pack();
    let bogus = t.0.path().join("nope.json");
    assert!(pack_shards(&[shard], &bogus, &t.path(), 4, 100).is_err());
}

#[test]
fn pack_shards_missing_shard_errors() {
    let (t, tok, _shard) = setup_pack();
    let bogus = t.0.path().join("nope.parquet");
    assert!(pack_shards(&[bogus], &tok, &t.path(), 4, 100).is_err());
}

#[test]
fn pack_shards_parquet_without_text_column_errors() {
    let t = TempBin::new();
    let tok = {
        let tok_path = t.0.path().join("tok.json");
        std::fs::write(&tok_path, TOKENIZER_JSON).unwrap();
        tok_path
    };
    use arrow::array::Int64Array;
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "value", arrow::datatypes::DataType::Int64, false,
    )]);
    let arr = Int64Array::from(vec![1i64, 2]);
    let batch = RecordBatch::try_new(std::sync::Arc::new(schema), vec![std::sync::Arc::new(arr)]).unwrap();
    let shard = t.0.path().join("wrong_col.parquet");
    let f = File::create(&shard).unwrap();
    let mut w = ArrowWriter::try_new(f, batch.schema(), None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    let err = pack_shards(&[shard.clone()], &tok, &t.path(), 4, 100).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("text") || msg.contains("column"),
        "error should mention missing text column: {msg}"
    );
    let _ = shard; // keep alive
}

#[test]
fn pack_shards_output_header_matches_args() {
    let (t, tok, shard) = setup_pack();
    let out = t.path();
    pack_shards(&[shard], &tok, &out, 7, 10_000).unwrap();
    let raw = std::fs::read(&out).unwrap();
    assert_eq!(u32::from_le_bytes(raw[0..4].try_into().unwrap()), 1, "vocab sentinel");
    assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), 7, "seq_len");
    let n = u32::from_le_bytes(raw[8..HEADER_LEN].try_into().unwrap()) as usize;
    assert_eq!(raw.len(), HEADER_LEN + n * 7 * 4, "file length exact");
}

#[test]
fn pack_shards_overwrites_existing_out_file() {
    let (t, tok, shard) = setup_pack();
    let out = t.path();
    std::fs::write(&out, &[0xFFu8; 4096]).unwrap(); // stale garbage
    pack_shards(&[shard], &tok, &out, 4, 100).unwrap();
    assert!(PackedDataset::open(&out).is_ok(), "rerun must produce clean file");
}

#[test]
fn pack_shards_seq_longer_than_corpus_yields_zero_seqs() {
    // Corpus smaller than seq_len+1 → no complete block → empty dataset.
    let (t, tok, shard) = setup_pack();
    let out = t.path();
    pack_shards(&[shard], &tok, &out, 10_000, 100).unwrap();
    let ds = PackedDataset::open(&out).unwrap();
    assert_eq!(ds.len(), 0, "no complete blocks possible");
}

#[test]
fn pack_shards_null_text_rows_skipped_without_panic() {
    let t = TempBin::new();
    let tok = {
        let tok_path = t.0.path().join("tok.json");
        std::fs::write(&tok_path, TOKENIZER_JSON).unwrap();
        tok_path
    };
    // Build nullable string column containing nulls.
    use arrow::array::StringArray;
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "text", arrow::datatypes::DataType::Utf8, true,
    )]);
    let arr = StringArray::from(vec![Some("hello world"), None, Some("foo bar")]);
    let batch = RecordBatch::try_new(std::sync::Arc::new(schema), vec![std::sync::Arc::new(arr)]).unwrap();
    let shard = t.0.path().join("nulls.parquet");
    let f = File::create(&shard).unwrap();
    let mut w = ArrowWriter::try_new(f, batch.schema(), None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    pack_shards(&[shard], &tok, &t.path(), 2, 100).unwrap(); // must not panic
}
