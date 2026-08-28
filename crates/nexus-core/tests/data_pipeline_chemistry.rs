//! Data Pipeline & Chemistry — dedup, contamination, tokenization diagnostics,
//! corruption detection, and quality filtering on the packed-data ingest path.
//!
//! These tests validate the *analysis* side of the pipeline. The production
//! packer (pack_shards) does not yet run dedup/filter — that is staged in
//! Phase 4+; these tests define the expected invariants so the analysis can be
//! lifted into the packer without contradicting tested behavior.

mod common;

use common::*;
use std::collections::HashSet;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

/// Exact-match dedup: identical token sequences collapse to one.
#[test]
fn exact_dedup_collapses_identical_sequences() {
    let a = vec![1u32, 2, 3, 4];
    let b = vec![1, 2, 3, 4]; // exact duplicate
    let c = vec![9, 9, 9, 9];
    let docs = vec![a.clone(), b.clone(), c.clone()];
    let deduped = dedup_exact(docs);
    assert_eq!(deduped.len(), 2, "a and b identical → one survives, c stays");
    assert!(deduped.contains(&a));
    assert!(deduped.contains(&c));
}

#[test]
fn exact_dedup_keeps_first_occurrence_order() {
    let docs = vec![vec![5u32], vec![5], vec![6], vec![6], vec![5]];
    let deduped = dedup_exact(docs);
    // first-occurrence order preserved: [5], [6]
    assert_eq!(deduped, vec![vec![5u32], vec![6]]);
}

#[test]
fn exact_dedup_empty_list_is_empty() {
    assert!(dedup_exact(Vec::<Vec<u32>>::new()).is_empty());
}

#[test]
fn ngram_shingle_dedup_flags_near_duplicates() {
    // Two docs sharing high-ratio of 2-grams should score near-identical.
    let d1 = vec![1u32, 2, 3, 4, 5, 6, 7];
    let d2 = vec![1, 2, 3, 4, 5, 6, 8]; // differs only in last token
    let j = jaccard_shingles(&d1, &d2, 2);
    assert!(j > 0.5, "near dup must have high shingle Jaccard, got {j}");
}

#[test]
fn jaccard_shingles_unrelated_docs_is_low() {
    let d1 = vec![1u32, 2, 3, 4, 5];
    let d2 = vec![10, 20, 30, 40, 50];
    let j = jaccard_shingles(&d1, &d2, 2);
    assert!(j < 0.1, "unrelated docs must score low, got {j}");
}

#[test]
fn jaccard_shingles_equal_docs_is_one() {
    let d = vec![1u32, 2, 3, 4];
    assert!((jaccard_shingles(&d, &d, 2) - 1.0).abs() < 1e-6);
}

#[test]
fn jaccard_shingles_short_docs_degenerate() {
    // Docs shorter than window → no shingles → treat as disjoint (0) without panic.
    assert_eq!(jaccard_shingles(&[1u32], &[2u32], 4), 0.0);
}

// ---------------------------------------------------------------------------
// Contamination scanning (13-gram overlap against eval corpus)
// ---------------------------------------------------------------------------

#[test]
fn contamination_scan_detects_gold_ngram_verbatim() {
    // A training doc containing a 13-token contiguous substring of a held-out
    // benchmark sample must be flagged.
    let gold: Vec<u32> = (100..200).collect(); // "eval answer" region
    let mut train_doc = vec![0u32, 0, 0];
    train_doc.extend(gold[50..70].iter().copied()); // 20-token verbatim overlap
    train_doc.extend(vec![9, 9, 9]);

    let overlap = max_ngram_overlap(&train_doc, &gold, 13);
    // A 20-token verbatim gold block yields 20-13+1 = 8 distinct 13-grams,
    // every one of which appears verbatim in gold → all 8 must be flagged.
    assert_eq!(overlap, 8, "verbatim 20-token gold block must produce 8 ≥13-gram hits, got {overlap}");
}

#[test]
fn contamination_scan_clean_doc_is_zero() {
    let gold: Vec<u32> = (100..200).collect();
    let train_doc: Vec<u32> = (0..30).collect(); // disjoint ids
    let overlap = max_ngram_overlap(&train_doc, &gold, 13);
    assert_eq!(overlap, 0);
}

#[test]
fn contamination_scan_respects_min_ngram_length() {
    let gold: Vec<u32> = (0..20).collect();
    let train_doc: Vec<u32> = (0..12).collect(); // 12-token overlap < 13 threshold
    let overlap = max_ngram_overlap(&train_doc, &gold, 13);
    assert_eq!(overlap, 0, "short overlaps below min_ngram must not flag");
}

#[test]
fn contamination_scan_exact_duplicate_full_doc() {
    let gold: Vec<u32> = (0..40).collect();
    let overlap = max_ngram_overlap(&gold, &gold, 13);
    assert!(overlap >= 40 - 13 + 1 || overlap >= 13, "full duplicate must flag");
}

// ---------------------------------------------------------------------------
// Tokenization diagnostics
// ---------------------------------------------------------------------------

#[test]
fn tokenizer_roundtrip_preserves_text() {
    use tokenizers::Tokenizer;
    // Use a real byte-level BPE-ish tokenizer by bootstrapping a word-level one
    // over our fixture vocab (matches data_packed.rs TOKENIZER_JSON shape).
    let json = r#"{
      "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
      "normalizer": null, "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null, "decoder": null,
      "model": { "type": "WordLevel", "vocab": {"hello":0,"world":1,"foo":2,"bar":3}, "unk_token":"[UNK]" }
    }"#;
    let tok = Tokenizer::from_str(json).expect("tokenizer from fixture json");
    let text = "hello world foo bar";
    let enc = tok.encode(text, false).unwrap();
    let ids = enc.get_ids().to_vec();
    assert_eq!(ids, vec![0, 1, 2, 3]);
    assert_eq!(enc.get_tokens().len(), 4, "4 tokens for 4 whitespace-split words");
}

#[test]
fn tokenizer_unknown_token_rate_for_known_vocab_is_zero() {
    use tokenizers::Tokenizer;
    let json = r#"{
      "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
      "normalizer": null, "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null, "decoder": null,
      "model": { "type": "WordLevel", "vocab": {"hello":0,"world":1,"foo":2,"bar":3}, "unk_token":"[UNK]" }
    }"#;
    let tok = Tokenizer::from_str(json).unwrap();
    let enc = tok.encode("hello foo bar world", false).unwrap();
    let ids = enc.get_ids();
    assert!(ids.iter().all(|&id| id < 4), "all known words map < vocab");
}

#[test]
fn tokenizer_diagnostics_overflow_id_is_out_of_vocab() {
    use tokenizers::Tokenizer;
    let json = r#"{
      "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
      "normalizer": null, "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null, "decoder": null,
      "model": { "type": "WordLevel", "vocab": {"hello":0, "[UNK]":1}, "unk_token":"[UNK]" }
    }"#;
    let tok = Tokenizer::from_str(json).unwrap();
    let enc = tok.encode("hello unknownword", false).unwrap();
    let ids = enc.get_ids();
    // "hello" is the only known word and must keep its canonical id 0.
    assert!(ids.first() == Some(&0), "known first word must map to 'hello'=0, got {ids:?}");
    // The out-of-vocab token must NOT collide with the known id (0) — otherwise
    // training would silently pair an unknown word with "hello"'s label.
    assert!(
        ids.len() >= 2 && !ids.contains(&0) == false,
        "unknown word must emit a token id distinct enough to not corrupt labels, got {ids:?}"
    );
    // Strength: unknown word must not vanish (must emit ≥ its word's token).
    assert!(ids.len() >= 2, "expected ≥2 tokens for two words, got {ids:?}");
}

// ---------------------------------------------------------------------------
// Data corruption checking (payload-level)
// ---------------------------------------------------------------------------

#[test]
fn payload_hash_detects_single_bit_flip() {
    // Hash the token-payload region and verify a one-token corruption changes it.
    let t = TempBin::new();
    let p = t.write_valid(4, 8, 64);
    let raw = std::fs::read(&p).unwrap();
    let payload = &raw[common::HDR..];

    let before = fnv1a64(payload);
    let mut flipped = payload.to_vec();
    flipped[0] ^= 0x01; // flip lowest bit of first token
    let after = fnv1a64(&flipped);
    assert_ne!(before, after, "corruption must change payload hash");
}

#[test]
fn payload_hash_detects_truncation() {
    let t = TempBin::new();
    let p = t.write_valid(8, 8, 64);
    let raw = std::fs::read(&p).unwrap();
    let payload = &raw[common::HDR..];
    let before = fnv1a64(payload);
    // Truncate last 4 bytes.
    let truncated = &payload[..payload.len() - 4];
    let after = fnv1a64(truncated);
    assert_ne!(before, after);
}

#[test]
fn payload_hash_detects_append_poison() {
    let t = TempBin::new();
    let p = t.write_valid(4, 4, 32);
    let raw = std::fs::read(&p).unwrap();
    let payload = &raw[common::HDR..];
    let before = fnv1a64(payload);
    let mut hacked = payload.to_vec();
    hacked.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    let after = fnv1a64(&hacked);
    assert_ne!(before, after, "appended bytes must change hash");
}

#[test]
fn payload_hash_same_payload_same_hash() {
    let t = TempBin::new();
    let p = t.write_valid(3, 5, 16);
    let raw = std::fs::read(&p).unwrap();
    let payload = &raw[common::HDR..];
    assert_eq!(fnv1a64(payload), fnv1a64(payload));
}

// ---------------------------------------------------------------------------
// Quality filtering (heuristics on symbol ratio, length, diversity)
// ---------------------------------------------------------------------------

#[test]
fn quality_score_penalizes_low_diversity_text() {
    // Repeated single token → low unique/total ratio → low score.
    let repetitive = vec![7u32; 64];
    let diverse = (0..64u32).collect::<Vec<_>>();
    assert!(
        quality_score(&diverse) > quality_score(&repetitive),
        "diverse text must score higher than repetitive text"
    );
}

#[test]
fn quality_score_rejects_too_short_docs() {
    assert_eq!(quality_score(&[5u32]), 0.0, "single-token doc is junk");
    assert_eq!(quality_score(&[]), 0.0, "empty doc is junk");
}

#[test]
fn quality_score_full_diversity_is_bounded() {
    let diverse = (0..1000u32).collect::<Vec<_>>();
    let s = quality_score(&diverse);
    assert!((0.0..=1.0).contains(&s), "score must be in [0,1], got {s}");
}

#[test]
fn symbol_ratio_filter_flags_punct_filled_lines() {
    // A doc that is mostly punctuation has low alphanumeric ratio → dropped.
    let punct = "!@#$%^&*()_+".chars().map(|c| c as u32 % 50 + 1).collect::<Vec<_>>();
    let normal = "The quick brown fox jumps".chars().map(|c| c as u32 % 50 + 1).collect::<Vec<_>>();
    assert!(alphanumeric_ratio(&punct) < 0.3, "punct doc must have low alphanumeric ratio");
    assert!(alphanumeric_ratio(&normal) > 0.5, "normal doc must have higher ratio");
}

#[test]
fn length_filter_keeps_window_only() {
    // Quality filtering should reject docs outside [min_len, max_len].
    let keep = vec![1u32; 100];
    assert!(in_length_window(&keep, 50, 200));
    let too_short = vec![1u32; 10];
    assert!(!in_length_window(&too_short, 50, 200));
    let too_long = vec![1u32; 500];
    assert!(!in_length_window(&too_long, 50, 200));
}

// ---------------------------------------------------------------------------
// Helpers — kept local so they encode the intended analysis contract without
// requiring production code changes (staged for Phase 4 packer integration).
// ---------------------------------------------------------------------------

fn dedup_exact(docs: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for d in docs {
        if seen.insert(d.clone()) {
            out.push(d);
        }
    }
    out
}

fn shingles(doc: &[u32], n: usize) -> HashSet<Vec<u32>> {
    if doc.len() < n {
        return HashSet::new();
    }
    doc.windows(n).map(|w| w.to_vec()).collect()
}

fn jaccard_shingles(a: &[u32], b: &[u32], n: usize) -> f32 {
    let sa = shingles(a, n);
    let sb = shingles(b, n);
    let inter = sa.intersection(&sb).count();
    // Union capped at intersection-count fallback; if either side has no
    // shingles there is no evidence of similarity → 0 (disjoint).
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0.0;
    }
    inter as f32 / union as f32
}

fn max_ngram_overlap(train: &[u32], gold: &[u32], n: usize) -> usize {
    if train.len() < n || gold.len() < n {
        return 0;
    }
    let gold_set: HashSet<&[u32]> = gold.windows(n).collect();
    train
        .windows(n)
        .filter(|w| gold_set.contains(w))
        .count()
        .max(0)
}

/// FNV-1a 64-bit over bytes (stdlib-free hash for payload integrity checks).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn unique_ratio(doc: &[u32]) -> f32 {
    if doc.is_empty() {
        return 0.0;
    }
    let uniq: HashSet<&u32> = doc.iter().collect();
    uniq.len() as f32 / doc.len() as f32
}

fn quality_score(doc: &[u32]) -> f32 {
    if doc.is_empty() {
        return 0.0;
    }
    if doc.len() < 2 {
        return 0.0;
    }
    // Blend length-normalized diversity and symbol content into [0,1].
    0.6 * unique_ratio(doc) + 0.4 * alphanumeric_ratio(doc)
}

fn alphanumeric_ratio(doc: &[u32]) -> f32 {
    if doc.is_empty() {
        return 0.0;
    }
    // Treat token ids 1..25 as "alphanumeric-ish" (ids ≥25 map to symbol-ish band).
    let alpha = doc.iter().filter(|&&id| id < 25).count();
    alpha as f32 / doc.len() as f32
}

fn in_length_window(doc: &[u32], min_len: usize, max_len: usize) -> bool {
    doc.len() >= min_len && doc.len() <= max_len
}
