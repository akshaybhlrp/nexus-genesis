//! stream.rs coverage: synthetic_stream, packed_stream, LmBatcher, lm_loss.

mod common;

use common::*;
use nexus_core::data::PackedDataset;
use nexus_core::stream::{LmBatcher, Sequence, lm_loss, packed_stream, synthetic_stream};
use burn::data::dataloader::batcher::Batcher;

// ---------- synthetic_stream ----------

#[test]
fn synthetic_stream_yields_requested_len() {
    assert_eq!(synthetic_stream(10, 256).count(), 10);
    assert_eq!(synthetic_stream(0, 256).count(), 0);
    assert_eq!(synthetic_stream(1000, 64).count(), 1000);
}

#[test]
fn synthetic_stream_sequence_length_is_128() {
    for s in synthetic_stream(3, 256) {
        assert_eq!(s.tokens.len(), 128);
    }
}

#[test]
fn synthetic_stream_tokens_in_vocab_range() {
    let vocab = 16u32;
    for s in synthetic_stream(5, vocab) {
        assert!(s.tokens.iter().all(|&t| t < vocab), "token out of range: {:?}", s.tokens);
    }
}

#[test]
fn synthetic_stream_deterministic() {
    let a: Vec<u32> = synthetic_stream(2, 32).flat_map(|s| s.tokens).collect();
    let b: Vec<u32> = synthetic_stream(2, 32).flat_map(|s| s.tokens).collect();
    assert_eq!(a, b);
}

#[test]
fn synthetic_stream_vocab_one_never_panics() {
    // Degenerate vocab=1 exercises the .max(1) guard.
    for s in synthetic_stream(2, 1) {
        assert_eq!(s.tokens.len(), 128);
        assert!(s.tokens.iter().all(|&t| t < 1));
    }
}

// ---------- packed_stream ----------

#[test]
fn packed_stream_len_matches_dataset() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(7, 8, 32));
    assert_eq!(packed_stream(ds.clone()).count(), 7);
}

#[test]
fn packed_stream_contents_equal_dataset_seqs() {
    let t = TempBin::new();
    let p = t.write_valid(4, 6, 16);
    let ds = PackedDataset::open(&p).unwrap();
    let streamed: Vec<Vec<u32>> = packed_stream(std::sync::Arc::new(PackedDataset::open(&p).unwrap()))
        .map(|s| s.tokens)
        .collect();
    for i in 0..ds.len() {
        assert_eq!(streamed[i], ds.seq(i));
    }
}

#[test]
fn packed_stream_empty_dataset_is_empty() {
    let t = TempBin::new();
    let ds = open_arc(&t.write_valid(0, 4, 8));
    assert_eq!(packed_stream(ds).count(), 0);
}

// ---------- LmBatcher ----------

#[test]
fn batch_shapes_and_shift() {
    let batcher = LmBatcher::<TB>::new();
    // Two identical sequences of 5 → inputs [:-1], targets [1:] per row.
    let seq = || Sequence { tokens: vec![10, 20, 30, 40, 50] };
    let batch = batcher.batch(vec![seq(), seq()], &device());
    assert_eq!(batch.inputs.dims(), [2, 4]);
    assert_eq!(batch.targets.dims(), [2, 4]);
    let ins: Vec<u32> = batch.inputs.into_data().iter().collect();
    let tg: Vec<u32> = batch.targets.into_data().iter().collect();
    assert_eq!(ins, vec![10, 20, 30, 40, 10, 20, 30, 40]);
    assert_eq!(tg, vec![20, 30, 40, 50, 20, 30, 40, 50]);
}

#[test]
fn batch_single_item_minimal_shape() {
    let batcher = LmBatcher::<TB>::new();
    let batch = batcher.batch(
        vec![Sequence { tokens: vec![1, 2] }],
        &device(),
    );
    assert_eq!(batch.inputs.dims(), [1, 1]);
    assert_eq!(batch.targets.dims(), [1, 1]);
    let tg: Vec<u32> = batch.targets.into_data().iter().collect();
    assert_eq!(tg, vec![2]);
}

#[test]
#[should_panic]
fn batch_empty_items_panics_indexing_first_seq() {
    let batcher = LmBatcher::<TB>::new();
    let _ = batcher.batch(Vec::new(), &device());
}

#[test]
#[should_panic]
fn batch_reshape_panics_on_ragged_lengths() {
    // Sequences of different lengths flatten to a non-[b*s] count → reshape fails.
    let batcher = LmBatcher::<TB>::new();
    let _ = batcher.batch(
        vec![
            Sequence { tokens: vec![1, 2, 3, 4] },
            Sequence { tokens: vec![5, 6] },
        ],
        &device(),
    );
}

#[test]
fn batch_large_batch_roundtrip() {
    let batcher = LmBatcher::<TB>::new();
    let seqs: Vec<Sequence> = (0..128)
        .map(|i| Sequence { tokens: (0..32u32).map(|j| j + i as u32 % 7 + 1).collect() })
        .collect();
    let batch = batcher.batch(seqs, &device());
    assert_eq!(batch.inputs.dims(), [128, 31]);
    assert_eq!(batch.targets.dims(), [128, 31]);
}

// ---------- lm_loss ----------

#[test]
fn loss_of_perfect_prediction_is_near_zero() {
    use burn::tensor::{Tensor, TensorData};
    // logits one-hot on target class → CE ≈ 0.
    let v = 4usize;
    // targets: [0, 3, 1, 2] flattened over B*S=4
    let targets = Tensor::<TB, 2, burn::tensor::Int>::from_ints([[0u32, 3], [1, 2]], &device());
    let mut flat = Vec::new();
    for t in [0u32, 3, 1, 2] {
        let mut row = vec![0f32; v];
        row[t as usize] = 100.0;
        flat.extend(row);
    }
    let logits =
        Tensor::<TB, 2>::from_data(TensorData::new(flat, [4, v]), &device()).reshape([2, 2, v]);
    let l = lm_loss(logits, targets);
    let val = l.into_data().iter::<f32>().next().unwrap();
    assert!(val.abs() < 1e-2, "perfect-prediction CE should be ~0, got {val}");
}

#[test]
fn loss_of_uniform_logits_equals_ln_vocab() {
    use burn::tensor::{Tensor, TensorData};
    let (b, s, v) = (2usize, 2usize, 8usize);
    let logits = Tensor::<TB, 2>::from_data(
        TensorData::new(vec![0f32; b * s * v], [b * s, v]),
        &device(),
    )
    .reshape([b, s, v]);
    let targets =
        Tensor::<TB, 2, burn::tensor::Int>::from_ints([[0u32, 1], [2, 3]], &device());
    let l = lm_loss(logits, targets).into_data().iter::<f32>().next().unwrap();
    assert!((l - (v as f32).ln()).abs() < 1e-3, "uniform CE={l}, want ln({v})={}", (v as f32).ln());
}

#[test]
fn loss_worse_prediction_gives_higher_loss() {
    use burn::tensor::{Tensor, TensorData};
    let mk = |good: bool| {
        // Target always class 0. Good: mass on 0. Bad: mass on 1.
        let row: Vec<f32> = if good {
            vec![5.0, 0.0, 0.0, 0.0]
        } else {
            vec![0.0, 5.0, 0.0, 0.0]
        };
        let mut flat = Vec::new();
        for _ in 0..4 {
            flat.extend(row.iter().copied());
        }
        (
            Tensor::<TB, 2>::from_data(TensorData::new(flat, [4, 4]), &device()).reshape([2, 2, 4]),
            Tensor::<TB, 2, burn::tensor::Int>::from_ints([[0u32, 0], [0, 0]], &device()),
        )
    };
    let (lg_good, tg) = mk(true);
    let good = lm_loss(lg_good, tg.clone()).into_data().iter::<f32>().next().unwrap();
    let (lg_bad, _) = mk(false);
    let bad = lm_loss(lg_bad, tg).into_data().iter::<f32>().next().unwrap();
    assert!(bad > good, "bad={bad} must exceed good={good}");
}

#[test]
fn loss_finite_on_extreme_logits() {
    use burn::tensor::{Tensor, TensorData};
    let (b, s, v) = (1usize, 2usize, 4usize);
    // ±1000 logits — softmax overflow territory; log_softmax must stay finite.
    let mut flat = Vec::new();
    for _ in 0..b * s {
        flat.extend([1000f32, -1000.0, 0.0, 500.0]);
    }
    let logits =
        Tensor::<TB, 2>::from_data(TensorData::new(flat, [b * s, v]), &device()).reshape([b, s, v]);
    let targets = Tensor::<TB, 2, burn::tensor::Int>::from_ints([[0u32, 3]], &device());
    let l = lm_loss(logits, targets).into_data().iter::<f32>().next().unwrap();
    assert!(l.is_finite(), "loss must stay finite on extreme logits, got {l}");
}
