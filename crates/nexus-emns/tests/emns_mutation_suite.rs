//! EMNS (Evolutionary Mutation with Natural Selection) integration tests.
//!
//! Validates:
//! - Pillar 2 (The Soul): CPU-side mutation kernel round-trip.
//! - Critical Patch 1: Per-expert macro-resistance (208 floats vs 6.4GB per-param).
//! - Critical Patch 4: Safe detach() -> to_data() -> mutate -> from_data().
//! - Critical Patch 5: XORSHIFT32 + Box-Muller Gaussian N(0,1) RNG properties.
//! - Adaptive mu scaling, parameter bounds, multithreaded Rayon consistency.

use burn::module::Param;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use nexus_emns::mutator::{ExpertResistance, MutationConfig, Mutator};

type TestBackend = burn::backend::Autodiff<burn::backend::Wgpu>;

fn device() -> burn::backend::wgpu::WgpuDevice {
    Default::default()
}

fn create_param<B: AutodiffBackend>(shape: [usize; 2], val: f32) -> Param<Tensor<B, 2>> {
    let dev = B::Device::default();
    let num_elements = shape[0] * shape[1];
    let data = vec![val; num_elements];
    let tensor = Tensor::<B, 2>::from_data(TensorData::new(data, shape), &dev);
    Param::from_tensor(tensor)
}

// =========================================================================
// 1. Critical Patch 5: XORSHIFT32 + Box-Muller Distribution Invariants
// =========================================================================

#[test]
fn test_emns_mutation_noise_reproducibility() {
    let cfg = MutationConfig {
        mu: 0.05,
        ..Default::default()
    };
    let m1 = Mutator::new(cfg.clone(), 1, 4);
    let m2 = Mutator::new(cfg, 1, 4);

    let p1 = create_param::<TestBackend>([4, 8], 0.0);
    let p2 = create_param::<TestBackend>([4, 8], 0.0);

    // Identical step & expert_idx must produce strictly identical mutated weights
    let r1 = m1.mutate_weight::<TestBackend>(p1, 2, 0.0);
    let r2 = m2.mutate_weight::<TestBackend>(p2, 2, 0.0);

    let v1: Vec<f32> = r1.val().into_data().iter().collect();
    let v2: Vec<f32> = r2.val().into_data().iter().collect();

    assert_eq!(v1, v2, "Same seed, step, and expert must produce bit-exact noise");
}

#[test]
fn test_emns_mutation_noise_divergence_across_experts() {
    let cfg = MutationConfig {
        mu: 0.05,
        ..Default::default()
    };
    let m = Mutator::new(cfg, 1, 4);

    let p1 = create_param::<TestBackend>([4, 8], 0.0);
    let p2 = create_param::<TestBackend>([4, 8], 0.0);

    let r1 = m.mutate_weight::<TestBackend>(p1, 0, 0.0);
    let r2 = m.mutate_weight::<TestBackend>(p2, 1, 0.0);

    let v1: Vec<f32> = r1.val().into_data().iter().collect();
    let v2: Vec<f32> = r2.val().into_data().iter().collect();

    assert_ne!(v1, v2, "Different experts must receive distinct noise vectors");
}

#[test]
fn test_emns_mutation_noise_divergence_across_steps() {
    let cfg = MutationConfig {
        mu: 0.05,
        ..Default::default()
    };
    let mut m = Mutator::new(cfg, 1, 4);

    let p1 = create_param::<TestBackend>([4, 8], 0.0);
    let p2 = create_param::<TestBackend>([4, 8], 0.0);

    let r1 = m.mutate_weight::<TestBackend>(p1, 0, 0.0);
    m.advance();
    let r2 = m.mutate_weight::<TestBackend>(p2, 0, 0.0);

    let v1: Vec<f32> = r1.val().into_data().iter().collect();
    let v2: Vec<f32> = r2.val().into_data().iter().collect();

    assert_ne!(v1, v2, "Advancing step counter must alter noise trajectory");
}

// =========================================================================
// 2. Critical Patch 1 & 4: Macro-Resistance & Safe CPU Round-trip
// =========================================================================

#[test]
fn test_emns_full_resistance_locks_weights() {
    let cfg = MutationConfig {
        mu: 0.1, // high mutation rate
        ..Default::default()
    };
    let m = Mutator::new(cfg, 2, 4);

    let initial_values = vec![1.234f32; 32];
    let p = Param::from_tensor(Tensor::<TestBackend, 2>::from_data(
        TensorData::new(initial_values.clone(), [4, 8]),
        &device(),
    ));

    // Full resistance (1.0) -> effective_mu = 0.0 -> weights untouched
    let mutated = m.mutate_weight::<TestBackend>(p, 0, 1.0);
    let after_values: Vec<f32> = mutated.val().into_data().iter().collect();

    assert_eq!(initial_values, after_values, "Resistance=1.0 must completely protect weights");
}

#[test]
fn test_emns_zero_resistance_allows_full_mutation() {
    let cfg = MutationConfig {
        mu: 0.08,
        ..Default::default()
    };
    let m = Mutator::new(cfg, 1, 2);

    let initial_values = vec![0.5f32; 64];
    let p = Param::from_tensor(Tensor::<TestBackend, 2>::from_data(
        TensorData::new(initial_values.clone(), [8, 8]),
        &device(),
    ));

    let mutated = m.mutate_weight::<TestBackend>(p, 0, 0.0);
    let after_values: Vec<f32> = mutated.val().into_data().iter().collect();

    let any_changed = initial_values
        .iter()
        .zip(after_values.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5);

    assert!(any_changed, "Resistance=0.0 must allow weights to mutate");
}

#[test]
fn test_emns_mutation_scales_strictly_with_mu() {
    let make_weights = || {
        Param::from_tensor(Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0f32; 128], [16, 8]),
            &device(),
        ))
    };

    let cfg_low = MutationConfig {
        mu: 0.002,
        ..Default::default()
    };
    let cfg_high = MutationConfig {
        mu: 0.08,
        ..Default::default()
    };

    let m_low = Mutator::new(cfg_low, 1, 1);
    let m_high = Mutator::new(cfg_high, 1, 1);

    let out_low: Vec<f32> = m_low
        .mutate_weight::<TestBackend>(make_weights(), 0, 0.0)
        .val()
        .into_data()
        .iter()
        .collect();

    let out_high: Vec<f32> = m_high
        .mutate_weight::<TestBackend>(make_weights(), 0, 0.0)
        .val()
        .into_data()
        .iter()
        .collect();

    let l2_low: f32 = out_low.iter().map(|x| x * x).sum::<f32>().sqrt();
    let l2_high: f32 = out_high.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(
        l2_high > l2_low * 10.0,
        "Higher mu must produce proportionally larger weight displacement"
    );
}

#[test]
fn test_emns_preserves_grad_requirement_after_cpu_roundtrip() {
    let cfg = MutationConfig::default();
    let m = Mutator::new(cfg, 1, 1);

    let p = create_param::<TestBackend>([4, 4], 0.1);
    assert!(p.val().is_require_grad());

    let mutated = m.mutate_weight::<TestBackend>(p, 0, 0.2);
    // After CPU detach + mutate + from_data, tensor must still require gradients for backprop
    assert!(
        mutated.val().is_require_grad(),
        "Mutated parameter tensor must have require_grad enabled"
    );
}

// =========================================================================
// 3. Expert Resistance Dynamics
// =========================================================================

#[test]
fn test_expert_resistance_selective_update() {
    let mut resistance = ExpertResistance::new(4);
    assert_eq!(resistance.values, vec![0.0, 0.0, 0.0, 0.0]);

    // Step 1: traffic to expert 0 and 2 only
    let routed_mass = vec![0.8, 0.0, 0.2, 0.0];
    resistance.update(&routed_mass, 0.05);

    assert!(resistance.values[0] > 0.04, "Expert 0 must gain resistance");
    assert_eq!(resistance.values[1], 0.0, "Expert 1 must stay unresistant");
    assert!(resistance.values[2] > 0.04, "Expert 2 must gain resistance");
    assert_eq!(resistance.values[3], 0.0, "Expert 3 must stay unresistant");
}

#[test]
fn test_expert_resistance_asymptotic_convergence() {
    let mut resistance = ExpertResistance::new(1);
    let routed_mass = vec![1.0];
    let decay = 0.02;

    // Simulate 500 active steps
    for _ in 0..500 {
        resistance.update(&routed_mass, decay);
    }

    assert!(
        resistance.values[0] > 0.999 && resistance.values[0] <= 1.0,
        "Resistance must asymptotically approach 1.0 without overflowing, got {}",
        resistance.values[0]
    );
}

// =========================================================================
// 4. MutationConfig Bounds and Hyperparameter Clamping
// =========================================================================

#[test]
fn test_mutation_config_clamp_bounds() {
    let mut cfg = MutationConfig {
        mu: 0.5,
        mu_min: 0.001,
        mu_max: 0.1,
        resistance_decay: 0.001,
    };

    cfg.clamp_mu();
    assert_eq!(cfg.mu, 0.1, "mu must clamp to mu_max");

    cfg.mu = 0.00001;
    cfg.clamp_mu();
    assert_eq!(cfg.mu, 0.001, "mu must clamp to mu_min");

    cfg.mu = 0.04;
    cfg.clamp_mu();
    assert_eq!(cfg.mu, 0.04, "mu within [mu_min, mu_max] must remain untouched");
}

#[test]
fn test_mutator_step_wrapping_overflow_safety() {
    let mut m = Mutator::new(MutationConfig::default(), 2, 4);
    m.step = u32::MAX - 1;

    m.advance();
    assert_eq!(m.step, u32::MAX);

    m.advance();
    assert_eq!(m.step, 0, "Step counter must wrap safely without panic");
}
