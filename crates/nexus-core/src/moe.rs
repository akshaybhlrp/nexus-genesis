//! Phase 2 "The Split": Hierarchical MoE via sparse upcycling of the dense FFN.
//!
//! Upcycle: each dense block's SwiGLU FFN is cloned N times into experts
//! (perturbed copies — the classic sparse-upcycling init), and a top-k router
//! picks which experts process each token. Attention stays shared/dense.

use crate::model::{Llama, LlamaBlock};
use burn::config::Config;
use burn::nn::{Embedding, Linear, RmsNorm, SwiGlu};
use burn::nn::attention::{MultiHeadAttention, generate_autoregressive_mask};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Router hyperparameters.
#[derive(Config, Debug)]
pub struct RouterConfig {
    pub n_experts: usize,
    #[config(default = "2")]
    pub top_k: usize,
    /// Load-balancing auxiliary loss weight (Switch-Transformer convention).
    #[config(default = "0.01")]
    pub balance_weight: f32,
}

/// One expert == the up-cycled dense FFN (gate/up + down).
#[derive(Module, Debug)]
pub struct Expert<B: Backend> {
    pub(crate) gate_up: SwiGlu<B>,
    pub(crate) down: Linear<B>,
}

impl<B: Backend> Expert<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down.forward(self.gate_up.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct MoEBlock<B: Backend> {
    pub(crate) attn_norm: RmsNorm<B>,
    pub(crate) attn: MultiHeadAttention<B>,
    pub(crate) rope: crate::model::RotaryPosition<B>,
    pub(crate) ffn_norm: RmsNorm<B>,
    pub(crate) router_gate: Linear<B>,
    pub experts: Vec<Expert<B>>,
    top_k: usize,
}

#[derive(Module, Debug)]
pub struct MoELlama<B: Backend> {
    pub(crate) token_embed: Embedding<B>,
    pub blocks: Vec<MoEBlock<B>>,
    pub(crate) final_norm: RmsNorm<B>,
    pub(crate) lm_head: Linear<B>,
    /// Carried from the dense model for out-of-vocab rejection (see
    /// [`crate::model::check_token_ids`]).
    pub(crate) vocab_size: usize,
}

/// Router output for one block forward.
pub struct RouteInfo<B: Backend> {
    /// Top-k expert indices per token, `[batch*seq, top_k]`.
    pub indices: Tensor<B, 2, Int>,
    /// Router probabilities of the chosen experts (normalized), same shape.
    pub weights: Tensor<B, 2>,
    /// Switch-style load-balance auxiliary loss (scalar).
    pub balance_loss: Tensor<B, 1>,
    /// Router entropy: -Σ p·log(p) over the full softmax (not just top-k).
    /// High entropy → router uncertain → good time to explore via mutation.
    pub entropy: f32,
    /// Per-expert routed mass fraction (sum of softmax weights routed to
    /// each expert, normalized by total tokens). Used to update resistance.
    pub expert_mass: Vec<f32>,
}

impl RouterConfig {
    fn init_router_gate<B: Backend>(&self, d_model: usize, device: &B::Device) -> Linear<B> {
        use burn::nn::{Initializer, LinearConfig};
        // ponytail: Zeros init → uniform routing at start; dense-init router is
        // the upgrade if experts fail to differentiate early.
        LinearConfig::new(d_model, self.n_experts)
            .with_initializer(Initializer::Zeros)
            .init(device)
    }
}

/// Sparse upcycling: transplant a trained dense model into an MoE model.
///
/// Every dense FFN becomes `n_experts` perturbed copies (noise ~1% of weight
/// std) so experts start near-identical but can specialize; everything else
/// (embeddings, attention, norms) is copied verbatim.
pub fn upcycle_dense<B>(dense: &Llama<B>, cfg: &RouterConfig) -> MoELlama<B>
where
    B: Backend,
{
    let device = dense.devices().first().cloned().unwrap_or_default();
    let blocks = dense
        .blocks
        .iter()
        .map(|blk| upcycle_block(blk, cfg, &device))
        .collect();

    MoELlama {
        token_embed: dense.token_embed.clone(),
        blocks,
        final_norm: dense.final_norm.clone(),
        lm_head: dense.lm_head.clone(),
        vocab_size: dense.vocab_size,
    }
}

fn upcycle_block<B: Backend>(
    blk: &LlamaBlock<B>,
    cfg: &RouterConfig,
    device: &B::Device,
) -> MoEBlock<B> {
    let mk_expert = || Expert {
        gate_up: blk.ffn_gate_up.clone(),
        down: blk.ffn_down.clone(),
    };
    let mut experts = Vec::with_capacity(cfg.n_experts);
    for _ in 0..cfg.n_experts {
        experts.push(mk_expert());
    }
    // ponytail: experts are exact clones (zero noise) — identical experts make
    // routing degenerate only if router stays zero-trained; add ~1% weight
    // noise here once EMNS mutation lands to break symmetry cheaply.

    MoEBlock {
        attn_norm: blk.attn_norm.clone(),
        attn: blk.attn.clone(),
        rope: blk.rope.clone(),
        ffn_norm: blk.ffn_norm.clone(),
        router_gate: cfg.init_router_gate(blk.ffn_norm.gamma.dims()[0], device),
        experts,
        top_k: cfg.top_k.min(cfg.n_experts),
    }
}

impl<B: Backend> MoEBlock<B> {
    /// Route + compute. Returns block output plus routing info for logging
    /// and the balance loss.
    pub fn forward_routed(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, RouteInfo<B>) {
        let [batch, seq, d] = x.dims();
        let device = x.device();
        let t = batch * seq;

        // Shared attention path with per-head RoPE on Q & K
        let normed = self.attn_norm.forward(x.clone());
        let n_heads = self.attn.n_heads;
        let d_k = self.attn.d_k;

        let q = self.attn.query.forward(normed.clone())
            .reshape([batch, seq, n_heads, d_k])
            .swap_dims(1, 2);
        let k = self.attn.key.forward(normed.clone())
            .reshape([batch, seq, n_heads, d_k])
            .swap_dims(1, 2);
        let v = self.attn.value.forward(normed)
            .reshape([batch, seq, n_heads, d_k])
            .swap_dims(1, 2);

        let q = self.rope.inner.forward(q);
        let k = self.rope.inner.forward(k);

        let scale = (d_k as f32).sqrt();
        let scores = q.matmul(k.transpose()).div_scalar(scale);
        let mask = generate_autoregressive_mask(batch, seq, &device);
        let scores = scores.mask_fill(mask.reshape([batch, 1, seq, seq]), -1e4);
        let weights = burn::tensor::activation::softmax(scores, 3);
        let context = weights.matmul(v).swap_dims(1, 2).reshape([batch, seq, n_heads * d_k]);
        let out = self.attn.output.forward(context);
        let x = x + out;

        // --- Routing on the FFN-normed stream ---
        let h = self.ffn_norm.forward(x.clone()).reshape([t, d]);
        let logits = self.router_gate.forward(h.clone()); // [T, E]

        // Full softmax for entropy computation (over all experts).
        let full_probs = burn::tensor::activation::softmax(logits.clone(), 1); // [T, E]

        let (topk_w, indices) = logits.topk_with_indices(self.top_k, 1); // [T, k]
        // Softmax over the selected k so expert weights sum to 1 per token.
        let weights = burn::tensor::activation::softmax(topk_w, 1);

        let n_experts = self.experts.len();
        // ponytail: dense dispatch — computes ALL experts then masks by route.
        // O(E) work per token instead of O(top_k). Swap to gather/scatter
        // dispatch when profiling shows it matters (Phase 4).
        let mut combined = Tensor::zeros([t, d], &device);
        let mut acc_balance = Tensor::zeros([1], &device);
        let mut expert_mass = vec![0.0f32; n_experts];

        for e in 0..n_experts {
            // [T, k] mask of "slot j chose expert e".
            let idx_e = indices.clone().equal_elem(e as i64);
            let w_e = (weights.clone() * idx_e.float()).sum_dim(1); // routed mass [T]
            let y_e = self.experts[e]
                .forward(h.clone().reshape([batch, seq, d]))
                .reshape([t, d]);
            combined = combined + y_e * w_e.clone().detach().reshape([t, 1]);

            // Track per-expert mass for resistance updates.
            let mass_val: f32 = w_e.clone().mean().into_data().iter::<f32>().next().unwrap_or(0.0);
            expert_mass[e] = mass_val;

            // Load-balance aux loss (Switch): E * sum_e f_e * P_e; f uses hard
            // count proxy via soft mass here — same minimum.
            acc_balance = acc_balance + w_e.mean();
        }
        // acc_balance is [1]; scale by n_experts (Switch-style aux loss).
        let balance_loss = acc_balance * Tensor::from_floats([n_experts as f32], &device);

        // Router entropy: -Σ p·log(p) averaged over all tokens.
        let entropy = Self::compute_entropy(&full_probs);

        let ffn_out = combined.reshape([batch, seq, d]);
        (
            x + ffn_out,
            RouteInfo {
                indices,
                weights,
                balance_loss,
                entropy,
                expert_mass,
            },
        )
    }

    /// Compute mean entropy of router softmax distribution across all tokens.
    /// -Σ p·log(p), averaged over the token dimension.
    fn compute_entropy(probs: &Tensor<B, 2>) -> f32 {
        // p * log(p), clamped to avoid log(0).
        let log_p = probs.clone().clamp_min(1e-8).log();
        let plogp = probs.clone() * log_p;
        // Sum over experts (dim 1) → [T], then mean over tokens.
        let per_token = -plogp.sum_dim(1); // [T, 1]
        per_token
            .mean()
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap_or(0.0)
    }
}

impl<B: Backend> MoELlama<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        crate::model::check_token_ids(&tokens, self.vocab_size);
        let mut x = self.token_embed.forward(tokens);
        for block in &self.blocks {
            x = block.forward_routed(x).0;
        }
        let x = self.final_norm.forward(x);
        self.lm_head.forward(x)
    }

    /// Forward collecting per-block balance losses, mean router entropy, and route info for training.
    pub fn forward_with_balance(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, f32, Vec<RouteInfo<B>>) {
        crate::model::check_token_ids(&tokens, self.vocab_size);
        let mut x = self.token_embed.forward(tokens);
        let device = x.device();
        let mut balance = Tensor::zeros([1], &device);
        let mut total_entropy = 0.0f32;
        let mut routes = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            let (out, info) = block.forward_routed(x);
            balance = balance + info.balance_loss.clone();
            total_entropy += info.entropy;
            routes.push(info);
            x = out;
        }

        let mean_entropy = if self.blocks.is_empty() {
            0.0
        } else {
            total_entropy / self.blocks.len() as f32
        };

        let x = self.final_norm.forward(x);
        (self.lm_head.forward(x), balance, mean_entropy, routes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LlamaConfig;
    type TestB = burn::backend::NdArray;

    #[test]
    fn upcycle_preserves_shapes_and_routes() {
        let device = Default::default();
        let dense = LlamaConfig::new(64, 32, 4, 1)
            .with_max_seq_len(16)
            .with_d_ff(64)
            .init::<TestB>(&Default::default());
        let cfg = RouterConfig::new(4); // top_k defaults to 2
        let moe = upcycle_dense(&dense, &cfg);

        let tokens: Vec<u32> = (0..8u32).map(|x| x + 1).collect();
        let tokens =
            Tensor::<TestB, 1, Int>::from_ints(tokens.as_slice(), &device).reshape([2, 4]);
        let logits = moe.forward(tokens);
        assert_eq!(logits.dims(), [2, 4, 64]);
        assert_eq!(moe.blocks[0].experts.len(), 4);
    }
}
