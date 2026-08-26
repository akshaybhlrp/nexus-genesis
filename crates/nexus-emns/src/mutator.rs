//! EMNS (Evolutionary Mutation with Natural Selection) — CPU-side mutator.
//!
//! Safe `detach()` → `to_data()` → mutate → `from_data()` round-trip.
//! No unsafe GPU mutation. CubeCL kernel is a stretch goal once this is proven.
//!
//! Per-expert macro-resistance (one f32 each) instead of per-parameter (6.4GB
//! overhead killed). Noise via XORSHIFT32 + Box-Muller for deterministic,
//! thread-unique Gaussian draws.

use burn::module::Param;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use rayon::prelude::*;

/// Mutation hyperparameters.
#[derive(Debug, Clone)]
pub struct MutationConfig {
    /// Base mutation rate. Scaled by `(1 - resistance)` per expert.
    pub mu: f32,
    pub mu_min: f32,
    pub mu_max: f32,
    /// How fast resistance grows per step for routed experts.
    /// Applied as `r = r + decay * (1 - r)` — asymptotic approach to 1.0.
    pub resistance_decay: f32,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            mu: 0.01,
            mu_min: 0.001,
            mu_max: 0.1,
            resistance_decay: 0.001,
        }
    }
}

impl MutationConfig {
    pub fn clamp_mu(&mut self) {
        self.mu = self.mu.clamp(self.mu_min, self.mu_max);
    }
}

/// Per-expert resistance. Established experts resist mutation; fresh ones
/// are more malleable. One f32 per expert — the macro-resistance fix from
/// the plan (not per-param).
#[derive(Debug, Clone)]
pub struct ExpertResistance {
    pub values: Vec<f32>,
}

impl ExpertResistance {
    pub fn new(n_experts: usize) -> Self {
        Self {
            values: vec![0.0; n_experts],
        }
    }

    /// Nudge resistance toward 1.0 for experts that were routed this step.
    /// `routed_mass[i]` ∈ [0, 1] = fraction of tokens routed to expert i.
    pub fn update(&mut self, routed_mass: &[f32], decay: f32) {
        for (r, &mass) in self.values.iter_mut().zip(routed_mass.iter()) {
            // Only grow resistance for experts that got traffic.
            if mass > 0.0 {
                *r += decay * (1.0 - *r);
            }
        }
    }
}

/// The mutator. Holds config + per-block resistance vectors.
#[derive(Debug, Clone)]
pub struct Mutator {
    pub config: MutationConfig,
    /// One resistance vector per MoE block (each block has its own experts).
    pub resistances: Vec<ExpertResistance>,
    pub step: u32,
}

impl Mutator {
    /// Create for `n_blocks` MoE blocks, each with `n_experts` experts.
    pub fn new(config: MutationConfig, n_blocks: usize, n_experts: usize) -> Self {
        Self {
            config,
            resistances: (0..n_blocks)
                .map(|_| ExpertResistance::new(n_experts))
                .collect(),
            step: 0,
        }
    }

    /// Mutate one expert's weight parameter via CPU round-trip.
    ///
    /// Generic over any AutodiffBackend. Unwraps Param, detaches, copies to
    /// CPU, applies XORSHIFT+Box-Muller noise, pushes back, re-wraps as Param.
    pub fn mutate_weight<B: AutodiffBackend>(
        &self,
        param: Param<Tensor<B, 2>>,
        expert_idx: usize,
        resistance: f32,
    ) -> Param<Tensor<B, 2>> {
        let mu = self.config.mu;
        let step = self.step;
        let effective_mu = mu * (1.0 - resistance);
        if effective_mu < 1e-8 {
            return param;
        }

        let tensor = param.val();
        let device = tensor.device();
        let inner = tensor.inner();
        let data = inner.to_data();
        let shape = data.shape.clone();
        let mut raw: Vec<f32> = data.iter::<f32>().collect();

        // Rayon parallel mutation — unique seed per (expert, step, element).
        let seed_base = (expert_idx as u32).wrapping_mul(0x9E3779B9).wrapping_add(step);
        raw.par_iter_mut().enumerate().for_each(|(i, w)| {
            let noise = xorshift_gaussian(i, seed_base);
            *w += noise * effective_mu;
        });

        let new_data = TensorData::new(raw, shape);
        let new_inner = Tensor::<B::InnerBackend, 2>::from_data(new_data, &device);
        let new_tensor = Tensor::from_inner(new_inner).require_grad();
        Param::from_tensor(new_tensor)
    }

    /// Bump the step counter. Call once per training step.
    pub fn advance(&mut self) {
        self.step = self.step.wrapping_add(1);
    }
}

/// XORSHIFT32 + Box-Muller: deterministic Gaussian N(0,1) from (idx, seed).
///
/// Two 3-shift xorshift passes → two uniform u32 → Box-Muller polar form
/// → one Gaussian sample. Identical to PLAN.md kernel logic.
#[inline]
fn xorshift_gaussian(idx: usize, seed: u32) -> f32 {
    let mut s: u32 = (idx as u32) ^ seed.wrapping_mul(2654435761);
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    let mut s2: u32 = s ^ 0xDEADBEEF;
    s2 ^= s2 << 13;
    s2 ^= s2 >> 17;
    s2 ^= s2 << 5;
    let u1 = (s as f32) / 4294967295.0;
    let u2 = (s2 as f32) / 4294967295.0;
    (-2.0 * u1.max(1e-7).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MutationConfig ───────────────────────────────────────────────

    #[test]
    fn mutation_config_default_values() {
        let c = MutationConfig::default();
        assert_eq!(c.mu, 0.01);
        assert_eq!(c.mu_min, 0.001);
        assert_eq!(c.mu_max, 0.1);
        assert_eq!(c.resistance_decay, 0.001);
    }

    #[test]
    fn clamp_mu_enforces_bounds() {
        let mut c = MutationConfig::default();

        // Above max
        c.mu = 5.0;
        c.clamp_mu();
        assert_eq!(c.mu, 0.1);

        // Below min
        c.mu = 0.0;
        c.clamp_mu();
        assert_eq!(c.mu, 0.001);

        // Within range — unchanged
        c.mu = 0.05;
        c.clamp_mu();
        assert_eq!(c.mu, 0.05);
    }

    #[test]
    fn clamp_mu_exact_boundary() {
        let mut c = MutationConfig::default();
        c.mu = c.mu_min;
        c.clamp_mu();
        assert_eq!(c.mu, c.mu_min);
        c.mu = c.mu_max;
        c.clamp_mu();
        assert_eq!(c.mu, c.mu_max);
    }

    // ── ExpertResistance ─────────────────────────────────────────────

    #[test]
    fn resistance_new_starts_at_zero() {
        let r = ExpertResistance::new(8);
        assert_eq!(r.values.len(), 8);
        assert!(r.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn resistance_zero_experts() {
        let r = ExpertResistance::new(0);
        assert!(r.values.is_empty());
    }

    #[test]
    fn resistance_update_only_routed_grow() {
        let mut r = ExpertResistance::new(4);
        let mass = [1.0, 0.0, 0.5, 0.0];
        r.update(&mass, 0.1);
        // Expert 0: routed → 0 + 0.1*(1-0) = 0.1
        assert!((r.values[0] - 0.1).abs() < 1e-6);
        // Expert 1: unrouted → stays 0
        assert_eq!(r.values[1], 0.0);
        // Expert 2: routed → 0 + 0.1*(1-0) = 0.1
        assert!((r.values[2] - 0.1).abs() < 1e-6);
        // Expert 3: unrouted → stays 0
        assert_eq!(r.values[3], 0.0);
    }

    #[test]
    fn resistance_update_asymptotic_to_one() {
        let mut r = ExpertResistance::new(2);
        let mass = [1.0, 0.5];
        let decay = 0.01;
        for _ in 0..2000 {
            r.update(&mass, decay);
        }
        // Both routed experts should converge close to 1.0.
        assert!(r.values[0] > 0.99, "r[0]={}", r.values[0]);
        assert!(r.values[1] > 0.99, "r[1]={}", r.values[1]);
    }

    #[test]
    fn resistance_unrouted_stays_zero() {
        let mut r = ExpertResistance::new(3);
        let mass = [0.0, 0.0, 0.0];
        for _ in 0..5000 {
            r.update(&mass, 0.05);
        }
        assert!(r.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn resistance_does_not_exceed_one() {
        let mut r = ExpertResistance::new(1);
        let mass = [1.0];
        for _ in 0..10000 {
            r.update(&mass, 0.5);
        }
        assert!(r.values[0] <= 1.0, "r={}", r.values[0]);
    }

    #[test]
    fn resistance_decay_zero_no_growth() {
        let mut r = ExpertResistance::new(2);
        let mass = [1.0, 1.0];
        r.update(&mass, 0.0);
        assert!(r.values.iter().all(|&v| v == 0.0));
    }

    // ── xorshift_gaussian ────────────────────────────────────────────

    #[test]
    fn xorshift_deterministic() {
        // Same inputs → same output.
        let a = xorshift_gaussian(42, 7);
        let b = xorshift_gaussian(42, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn xorshift_varies_with_idx() {
        let a = xorshift_gaussian(0, 1);
        let b = xorshift_gaussian(1, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn xorshift_varies_with_seed() {
        let a = xorshift_gaussian(0, 1);
        let b = xorshift_gaussian(0, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn xorshift_no_nan_or_inf() {
        let vals: Vec<f32> = (0..10000).map(|i| xorshift_gaussian(i, 42)).collect();
        assert!(vals.iter().all(|v| v.is_finite()), "found NaN or Inf");
    }

    #[test]
    fn xorshift_mean_near_zero() {
        let n = 10000;
        let vals: Vec<f32> = (0..n).map(|i| xorshift_gaussian(i, 99)).collect();
        let mean = vals.iter().sum::<f32>() / n as f32;
        // For N(0,1), std of sample mean ≈ 1/sqrt(n) ≈ 0.01. 3σ bound = 0.03.
        assert!(mean.abs() < 0.05, "mean={mean}, expected near 0");
    }

    #[test]
    fn xorshift_std_near_one() {
        let n = 10000;
        let vals: Vec<f32> = (0..n).map(|i| xorshift_gaussian(i, 123)).collect();
        let mean = vals.iter().sum::<f32>() / n as f32;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt();
        // Std of N(0,1) = 1. Allow generous margin for finite sample + xorshift quality.
        assert!(std > 0.8 && std < 1.2, "std={std}, expected ~1.0");
    }

    #[test]
    fn xorshift_range_plausible() {
        // Gaussian should occasionally produce values beyond ±2 but rarely ±4.
        let vals: Vec<f32> = (0..100000).map(|i| xorshift_gaussian(i, 55)).collect();
        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 3.0, "max_abs={max_abs}, expected >3 for 100k samples");
        assert!(max_abs < 8.0, "max_abs={max_abs}, implausible for N(0,1)");
    }

    #[test]
    fn xorshift_skew_near_zero() {
        // Skewness of N(0,1) is 0. Check gross asymmetry is absent.
        let n = 50000;
        let vals: Vec<f32> = (0..n).map(|i| xorshift_gaussian(i, 77)).collect();
        let mean = vals.iter().sum::<f32>() / n as f32;
        let m3 = vals.iter().map(|v| (v - mean).powi(3)).sum::<f32>() / n as f32;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let skew = m3 / variance.powf(1.5);
        assert!(skew.abs() < 0.2, "skew={skew}, expected near 0");
    }

    // ── Mutator ──────────────────────────────────────────────────────

    #[test]
    fn mutator_new_structure() {
        let cfg = MutationConfig::default();
        let m = Mutator::new(cfg, 3, 4);
        assert_eq!(m.resistances.len(), 3);
        assert!(m.resistances.iter().all(|r| r.values.len() == 4));
        assert_eq!(m.step, 0);
    }

    #[test]
    fn mutator_advance_increments() {
        let mut m = Mutator::new(MutationConfig::default(), 1, 1);
        assert_eq!(m.step, 0);
        m.advance();
        assert_eq!(m.step, 1);
        m.advance();
        assert_eq!(m.step, 2);
    }

    #[test]
    fn mutator_advance_wraps() {
        let mut m = Mutator::new(MutationConfig::default(), 1, 1);
        m.step = u32::MAX;
        m.advance();
        assert_eq!(m.step, 0); // wrapping_add
    }

    #[test]
    fn mutator_full_resistance_no_change() {
        type TestB = burn::backend::Autodiff<burn::backend::Wgpu>;

        let cfg = MutationConfig { mu: 0.1, ..Default::default() };
        let m = Mutator::new(cfg, 1, 1);

        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let tensor = Tensor::<TestB, 2>::from_data(
            TensorData::new(vec![1.0f32; 16], [4, 4]),
            &device,
        );
        let param = Param::from_tensor(tensor);
        let original: Vec<f32> = param.val().into_data().iter::<f32>().collect();

        // resistance = 1.0 → effective_mu = 0 → early return.
        let result = m.mutate_weight::<TestB>(param.clone(), 0, 1.0);
        let after: Vec<f32> = result.val().into_data().iter::<f32>().collect();
        assert_eq!(original, after, "full resistance should prevent mutation");
    }

    #[test]
    fn mutator_zero_resistance_changes_weights() {
        type TestB = burn::backend::Autodiff<burn::backend::Wgpu>;

        let cfg = MutationConfig { mu: 0.05, ..Default::default() };
        let m = Mutator::new(cfg, 1, 1);

        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let tensor = Tensor::<TestB, 2>::from_data(
            TensorData::new(vec![0.0f32; 16], [4, 4]),
            &device,
        );
        let param = Param::from_tensor(tensor);

        let result = m.mutate_weight::<TestB>(param, 0, 0.0);
        let after: Vec<f32> = result.val().into_data().iter::<f32>().collect();
        // With mu=0.05, resistance=0, effective_mu=0.05, noise should move weights.
        let changed = after.iter().any(|v| v.abs() > 1e-6);
        assert!(changed, "weights should change with zero resistance");
    }

    #[test]
    fn mutator_preserves_tensor_shape() {
        type TestB = burn::backend::Autodiff<burn::backend::Wgpu>;

        let cfg = MutationConfig { mu: 0.1, ..Default::default() };
        let m = Mutator::new(cfg, 1, 1);

        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let tensor = Tensor::<TestB, 2>::from_data(
            TensorData::new(vec![0.5f32; 24], [3, 8]),
            &device,
        );
        let param = Param::from_tensor(tensor);

        let result = m.mutate_weight::<TestB>(param, 0, 0.0);
        assert_eq!(result.val().dims(), [3, 8]);
    }

    #[test]
    fn mutator_deterministic_same_step() {
        type TestB = burn::backend::Autodiff<burn::backend::Wgpu>;

        let cfg = MutationConfig { mu: 0.01, ..Default::default() };
        let m = Mutator::new(cfg, 1, 1);

        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let make_param = || {
            let t = Tensor::<TestB, 2>::from_data(
                TensorData::new(vec![1.0f32; 8], [2, 4]),
                &device,
            );
            Param::from_tensor(t)
        };

        let r1 = m.mutate_weight::<TestB>(make_param(), 0, 0.0);
        let r2 = m.mutate_weight::<TestB>(make_param(), 0, 0.0);
        let v1: Vec<f32> = r1.val().into_data().iter::<f32>().collect();
        let v2: Vec<f32> = r2.val().into_data().iter::<f32>().collect();
        assert_eq!(v1, v2, "same step+expert → same mutation");
    }

    #[test]
    fn mutator_different_step_different_output() {
        type TestB = burn::backend::Autodiff<burn::backend::Wgpu>;

        let cfg = MutationConfig { mu: 0.05, ..Default::default() };
        let mut m1 = Mutator::new(cfg.clone(), 1, 1);
        let mut m2 = Mutator::new(cfg, 1, 1);
        m2.advance();

        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let make_param = || {
            let t = Tensor::<TestB, 2>::from_data(
                TensorData::new(vec![0.0f32; 8], [2, 4]),
                &device,
            );
            Param::from_tensor(t)
        };

        let v1: Vec<f32> = m1.mutate_weight::<TestB>(make_param(), 0, 0.0)
            .val().into_data().iter::<f32>().collect();
        let v2: Vec<f32> = m2.mutate_weight::<TestB>(make_param(), 0, 0.0)
            .val().into_data().iter::<f32>().collect();
        assert_ne!(v1, v2, "different steps should produce different mutations");
    }

    #[test]
    fn mutator_different_expert_different_output() {
        type TestB = burn::backend::Autodiff<burn::backend::Wgpu>;

        let cfg = MutationConfig { mu: 0.05, ..Default::default() };
        let m = Mutator::new(cfg, 1, 4);

        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let make_param = || {
            let t = Tensor::<TestB, 2>::from_data(
                TensorData::new(vec![0.0f32; 8], [2, 4]),
                &device,
            );
            Param::from_tensor(t)
        };

        let v0: Vec<f32> = m.mutate_weight::<TestB>(make_param(), 0, 0.0)
            .val().into_data().iter::<f32>().collect();
        let v1: Vec<f32> = m.mutate_weight::<TestB>(make_param(), 1, 0.0)
            .val().into_data().iter::<f32>().collect();
        assert_ne!(v0, v1, "different experts should produce different mutations");
    }

    #[test]
    fn mutator_mutation_magnitude_scales_with_mu() {
        type TestB = burn::backend::Autodiff<burn::backend::Wgpu>;

        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let make_param = || {
            let t = Tensor::<TestB, 2>::from_data(
                TensorData::new(vec![0.0f32; 64], [8, 8]),
                &device,
            );
            Param::from_tensor(t)
        };

        let cfg_small = MutationConfig { mu: 0.001, ..Default::default() };
        let m_small = Mutator::new(cfg_small, 1, 1);
        let r_small: Vec<f32> = m_small.mutate_weight::<TestB>(make_param(), 0, 0.0)
            .val().into_data().iter::<f32>().collect();
        let dist_small: f32 = r_small.iter().map(|v| v * v).sum();

        let cfg_large = MutationConfig { mu: 0.1, ..Default::default() };
        let m_large = Mutator::new(cfg_large, 1, 1);
        let r_large: Vec<f32> = m_large.mutate_weight::<TestB>(make_param(), 0, 0.0)
            .val().into_data().iter::<f32>().collect();
        let dist_large: f32 = r_large.iter().map(|v| v * v).sum();

        assert!(dist_large > dist_small, "larger mu should produce larger perturbation");
    }

    #[test]
    fn mutator_resistance_reduces_mutation_magnitude() {
        type TestB = burn::backend::Autodiff<burn::backend::Wgpu>;

        let cfg = MutationConfig { mu: 0.1, ..Default::default() };
        let m = Mutator::new(cfg, 1, 1);

        let device: burn::backend::wgpu::WgpuDevice = Default::default();
        let make_param = || {
            let t = Tensor::<TestB, 2>::from_data(
                TensorData::new(vec![0.0f32; 64], [8, 8]),
                &device,
            );
            Param::from_tensor(t)
        };

        let r_low: Vec<f32> = m.mutate_weight::<TestB>(make_param(), 0, 0.0)
            .val().into_data().iter::<f32>().collect();
        let dist_low: f32 = r_low.iter().map(|v| v * v).sum();

        let r_high: Vec<f32> = m.mutate_weight::<TestB>(make_param(), 0, 0.9)
            .val().into_data().iter::<f32>().collect();
        let dist_high: f32 = r_high.iter().map(|v| v * v).sum();

        assert!(dist_low > dist_high, "higher resistance should reduce mutation");
    }

    // ── Integration: Mutator + Resistance lifecycle ──────────────────

    #[test]
    fn resistance_and_mu_interact_correctly() {
        let mut cfg = MutationConfig::default();
        let mut r = ExpertResistance::new(4);

        // Simulate 100 steps of routing to experts 0,1,3 (not 2).
        for _ in 0..100 {
            r.update(&[0.25, 0.25, 0.0, 0.25], cfg.resistance_decay);
        }

        // Experts 0,1,3 should have high resistance; 2 should be 0.
        assert!(r.values[0] > 0.05, "r[0]={}", r.values[0]);
        assert!(r.values[2] == 0.0, "r[2]={}", r.values[2]);

        // effective_mu for expert 0 should be less than for expert 2.
        let eff_0 = cfg.mu * (1.0 - r.values[0]);
        let eff_2 = cfg.mu * (1.0 - r.values[2]);
        assert!(eff_2 > eff_0, "unrouted expert should have higher effective mu");
    }

    #[test]
    fn adaptive_mu_boost_and_decay() {
        let mut cfg = MutationConfig::default();

        // Simulate high entropy → boost
        let entropy_high = 0.8;
        let threshold = 0.7;
        if entropy_high > threshold {
            cfg.mu *= 1.2;
        }
        cfg.clamp_mu();
        assert!((cfg.mu - 0.012).abs() < 1e-6);

        // Simulate low entropy → decay
        let entropy_low = 0.5;
        if entropy_low > threshold {
            cfg.mu *= 1.2;
        } else {
            cfg.mu *= 0.95;
        }
        cfg.clamp_mu();
        assert!((cfg.mu - (0.012 * 0.95)).abs() < 1e-6);

        // Repeated decay clamps to min
        for _ in 0..1000 {
            cfg.mu *= 0.95;
            cfg.clamp_mu();
        }
        assert_eq!(cfg.mu, cfg.mu_min);
    }

    #[test]
    fn resistance_update_short_mass_vector() {
        // Mass vector shorter than resistance → only first elements updated.
        let mut r = ExpertResistance::new(8);
        let mass = [0.5, 0.5]; // only 2 elements for 8 experts
        r.update(&mass, 0.1);
        // First two routed, rest stay 0.
        assert!((r.values[0] - 0.1).abs() < 1e-6);
        assert!((r.values[1] - 0.1).abs() < 1e-6);
        assert_eq!(r.values[2], 0.0);
        assert_eq!(r.values[7], 0.0);
    }
}
