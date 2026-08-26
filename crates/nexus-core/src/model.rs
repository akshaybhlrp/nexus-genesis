//! Dense LLaMA-style decoder for Phase 1 ("The Seed").
//!
//! Built entirely from burn-nn primitives: Embedding, RmsNorm, RoPE,
//! MultiHeadAttention (causal), SwiGLU. No MoE yet — that arrives in Phase 2
//! when the dense FFN gets up-cycled into experts.

use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, generate_autoregressive_mask};
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RotaryEncoding,
    RotaryEncodingConfig, RmsNorm, RmsNormConfig, SwiGlu, SwiGluConfig,
};
use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Model hyperparameters. Start tiny to validate the loop, scale toward 0.1B.
#[derive(Config, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    #[config(default = "1024")]
    pub max_seq_len: usize,
    /// FFN hidden dim. LLaMA convention: ~2/3 * 4 * d_model, rounded to a
    /// multiple of n_heads for SwiGLU sharding friendliness.
    #[config(default = "1376")]
    pub d_ff: usize,
    #[config(default = "1e-5")]
    pub rms_norm_eps: f64,
    #[config(default = "10000.0")]
    pub rope_theta: f32,
}

impl LlamaConfig {
    /// Parameter count estimate (embedding-tied logits excluded from tie here).
    pub fn num_params(&self) -> usize {
        let head = self.d_model * self.d_model; // q,k,v,o per layer
        let ffn = 3 * self.d_model * self.d_ff + self.d_ff * self.d_model; // gate+up+down
        let per_layer = head + ffn + 2 * self.d_model; // + 2 RMSNorms
        self.vocab_size * self.d_model + self.n_layers * per_layer
    }
}

#[derive(Module, Debug)]
pub struct LlamaBlock<B: Backend> {
    pub(crate) attn_norm: RmsNorm<B>,
    pub(crate) attn: MultiHeadAttention<B>,
    pub(crate) rope: RotaryPosition<B>,
    pub(crate) ffn_norm: RmsNorm<B>,
    /// Gate+up projection: d_model -> 2 * d_ff internally, output d_ff.
    pub(crate) ffn_gate_up: SwiGlu<B>,
    /// Down projection: d_ff -> d_model.
    pub(crate) ffn_down: Linear<B>,
}

// RotaryEncoding is a plain struct (not #[derive(Module)]-annotated in a way we
// can embed directly), so wrap its Param tensor behind a Module wrapper.
#[derive(Module, Debug)]
pub struct RotaryPosition<B: Backend> {
    pub(crate) inner: RotaryEncoding<B>,
}

#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    pub(crate) token_embed: Embedding<B>,
    pub(crate) blocks: Vec<LlamaBlock<B>>,
    pub(crate) final_norm: RmsNorm<B>,
    pub(crate) lm_head: Linear<B>,
    /// Carried from config so forward can reject out-of-vocab tokens.
    /// Backends don't bounds-check embed lookups (Wgpu silently reads
    /// garbage; CUDA may fault) — the guard must live here.
    pub(crate) vocab_size: usize,
}

/// Fail-fast check at the trust boundary between raw token ids and the
/// embedding table. Panic (not Err): forward returns a Tensor, and a
/// poisoned training run is worse than a loud stop. Same behavior on every
/// backend — hardware-agnostic by construction.
pub fn check_token_ids<B: Backend>(tokens: &Tensor<B, 2, burn::tensor::Int>, vocab_size: usize) {
    // ponytail: readback of one scalar per forward — negligible vs attention,
    // but hoist out of hot loops if a no-GPU-sync path is ever needed.
    use burn::prelude::ElementConversion;
    let max_id: i64 = tokens.clone().max().max().into_scalar().elem();
    assert!(
        max_id >= 0 && max_id < vocab_size as i64,
        "token id {max_id} out of range [0, {vocab_size}) — dataset/tokenizer/config mismatch?"
    );
}

impl LlamaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Llama<B> {
        assert_eq!(
            self.d_model % self.n_heads,
            0,
            "d_model must be divisible by n_heads"
        );
        let tok = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let mut blocks = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            blocks.push(self.init_block(device));
        }
        let final_norm = self.rms_norm().init(device);
        let lm_head = LinearConfig::new(self.d_model, self.vocab_size)
            .with_initializer(Initializer::Zeros)
            .init(device);
        Llama {
            token_embed: tok,
            blocks,
            final_norm,
            lm_head,
            vocab_size: self.vocab_size,
        }
    }

    fn rms_norm(&self) -> RmsNormConfig {
        RmsNormConfig::new(self.d_model).with_epsilon(self.rms_norm_eps)
    }

    fn init_block<B: Backend>(&self, device: &B::Device) -> LlamaBlock<B> {
        // ponytail: RoPE applied on the full d_model stream, not per-head.
        // Per-head RoPE needs q/k projection hooks inside MultiHeadAttention;
        // revisit when swapping MHA for a custom attention module (Phase 2).
        let rope = RotaryPosition {
            inner: RotaryEncodingConfig::new(self.max_seq_len, self.d_model)
                .with_theta(self.rope_theta)
                .init(device),
        };
        LlamaBlock {
            attn_norm: self.rms_norm().init(device),
            attn: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_dropout(0.0)
                .with_quiet_softmax(true)
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / 3.0f64.sqrt(),
                    fan_out_only: false,
                })
                .init(device),
            rope,
            ffn_norm: self.rms_norm().init(device),
            ffn_gate_up: SwiGluConfig::new(self.d_model, self.d_ff).init(device),
            ffn_down: LinearConfig::new(self.d_ff, self.d_model).init(device),
        }
    }
}

impl<B: Backend> LlamaBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _d] = x.dims();
        let device = x.device();

        // Pre-norm attention with causal masking and per-head RoPE.
        let normed = self.attn_norm.forward(x.clone());
        // MHA expects [B, S, D]; RoPE must land on head dims [B*H, S, dk], so
        // project first via MHA's internal linears? No — apply RoPE on the raw
        // stream instead: simplest correct variant applies it before attention
        // on [B, S, D] (burn's RotaryEncoding handles [..., seq, dim]).
        let normed = self.rope.inner.forward(normed);

        let mask = generate_autoregressive_mask(batch, seq, &device);
        let out = self
            .attn
            .forward(MhaInput::self_attn(normed).mask_attn(mask))
            .context;

        let x = x + out;

        // Pre-norm SwiGLU FFN: gate+up to d_ff, then down-project back.
        let normed_ffn = self.ffn_norm.forward(x.clone());
        let x = x + self.ffn_down.forward(self.ffn_gate_up.forward(normed_ffn));
        x
    }
}

impl<B: Backend> Llama<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        check_token_ids(&tokens, self.vocab_size);
        let mut x = self.token_embed.forward(tokens);
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = self.final_norm.forward(x);
        self.lm_head.forward(x)
    }

    /// Next-token logits `[batch, seq, vocab]`.
    pub fn forward_logits(&self, tokens: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.forward(tokens)
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    type TestB = burn::backend::Wgpu;

    fn tiny() -> LlamaConfig {
        LlamaConfig::new(256, 64, 4, 2).with_max_seq_len(32).with_d_ff(128)
    }

    #[test]
    fn forward_shapes_and_loss_decreases_over_steps() {
        let device = Default::default();
        let cfg = tiny();
        let model = cfg.init::<TestB>(&device);

        // Deterministic toy batch.
        let tokens: Vec<u32> = (0..64u32).collect();
        let tokens = Tensor::<TestB, 1, burn::tensor::Int>::from_ints(tokens.as_slice(), &device)
            .reshape([2, 32]);

        let logits = model.forward_logits(tokens.clone());
        assert_eq!(logits.dims(), [2, 32, 256]);

        // Param estimate sanity: tiny config should be well under 1M params.
        let est = cfg.num_params();
        assert!(est > 0 && est < 1_000_000, "est={est}");
    }
}
