//! Batch stream: sequences from the packed dataset (or synthetic fallback)
//! into shifted (inputs, targets) device batches.

pub use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use std::marker::PhantomData;

/// One fixed-length token sequence ready for LM training.
#[derive(Clone, Debug)]
pub struct Sequence {
    pub tokens: Vec<u32>,
}

/// A batch of `[B, S]` input tokens plus `[B, S]` targets (next-token shifted).
#[derive(Clone, Debug)]
pub struct LmBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

/// Deterministic pseudo-text stream. Learnable structure so loss must drop —
/// used for smoke tests only.
///
/// ponytail: keep until real-data path is exercised in CI; delete after.
pub fn synthetic_stream(len: usize, vocab_size: u32) -> impl Iterator<Item = Sequence> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    (0..len).map(move |_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let seq_len = 128usize;
        let mut tokens = Vec::with_capacity(seq_len);
        let base = (state % vocab_size.max(1) as u64) as u32;
        for i in 0..seq_len {
            // Arithmetic walk mod vocab → predictable next token.
            tokens.push((base + (i * 7) as u32) % vocab_size);
        }
        Sequence { tokens }
    })
}

/// Iterator over a [`PackedDataset`](crate::data::PackedDataset).
pub fn packed_stream(dataset: std::sync::Arc<crate::data::PackedDataset>) -> impl Iterator<Item = Sequence> + 'static {
    packed_stream_from(dataset, 0)
}

/// Iterator over a [`PackedDataset`](crate::data::PackedDataset) starting at a cursor position.
/// Wraps circularly around `dataset.len()` for infinite continuous streaming.
pub fn packed_stream_from(
    dataset: std::sync::Arc<crate::data::PackedDataset>,
    start_idx: usize,
) -> impl Iterator<Item = Sequence> + 'static {
    let n = dataset.len().max(1);
    (0..n).map(move |offset| {
        let i = (start_idx + offset) % n;
        Sequence { tokens: dataset.seq(i) }
    })
}

/// Batches sequences into shifted (inputs, targets) pairs on the device the
/// dataloader passes in. Stateless — no stored device needed.
#[derive(Clone, Default)]
pub struct LmBatcher<B: Backend> {
    _b: PhantomData<B>,
}

impl<B: Backend> LmBatcher<B> {
    pub fn new() -> Self {
        Self { _b: PhantomData }
    }
}

impl<B: Backend> Batcher<B, Sequence, LmBatch<B>> for LmBatcher<B> {
    fn batch(&self, items: Vec<Sequence>, device: &B::Device) -> LmBatch<B> {
        let flat: Vec<u32> = items.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let b = items.len();
        let s = items[0].tokens.len();

        let all = Tensor::<B, 1, Int>::from_data(TensorData::new(flat.clone(), [flat.len()]), device)
            .reshape([b, s]);

        // Next-token shift: inputs [:-1], targets [1:], same shape.
        let inputs = all.clone().slice([0..b, 0..s - 1]);
        let targets = all.slice([0..b, 1..s]);
        LmBatch { inputs, targets }
    }
}

/// Mean cross-entropy over flattened [B*S, V] logits against [B*S] targets.
pub fn lm_loss<B: Backend>(logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
    let [b, s, v] = logits.dims();
    let log_probs = burn::tensor::activation::log_softmax(logits.reshape([b * s, v]), 1);
    let picked = log_probs.gather(1, targets.reshape([b * s, 1]));
    -picked.mean()
}
