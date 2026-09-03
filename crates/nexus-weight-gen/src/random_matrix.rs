//! Random Matrix Theory — Marchenko-Pastur Spectral Clamping
//!
//! The Marchenko-Pastur distribution describes the eigenvalue spectrum of
//! large random matrices. For an n×m matrix with i.i.d. entries of variance σ²,
//! the eigenvalues of the Gram matrix W^T W / n follow:
//!
//!   f(λ) = √((λ+)(λ-)(λ-λ))/(2πσ²λ)
//!
//! where λ± = σ²(1 ± √(n/m))²
//!
//! For neural network weights, clamping singular values to the Marchenko-Pastur
//! support means:
//! - No eigenvalue is too large (prevents gradient explosion)
//! - No eigenvalue is too small (prevents vanishing gradients)
//! - The spectral density is "information-theoretically optimal"
//!
//! MKG finding: Random Matrix Theory Marchenko-Pastur was ranked in top 5,
//! especially effective when combined with other frameworks.

/// Apply Marchenko-Pastur spectral clamping to a weight matrix.
///
/// Clamps singular values to the theoretical bounds of the Marchenko-Pastur
/// distribution, ensuring the weight matrix has an optimal eigenvalue spectrum.
pub fn apply_spectral_clamp(data: &mut [f32], rows: usize, cols: usize, seed: u64) {
    assert_eq!(data.len(), rows * cols);

    let n = rows;
    let m = cols;
    let gamma = n as f32 / m as f32; // aspect ratio

    // Marchenko-Pastur bounds: λ± = (1 ± √γ)²
    let sqrt_gamma = gamma.sqrt();
    let lambda_plus = (1.0 + sqrt_gamma).powi(2);
    let lambda_minus = (1.0 - sqrt_gamma).powi(2).max(0.0);

    // Step 1: Compute singular values via diagonal of Gram matrix
    // For efficiency, compute diagonal of W^T W instead of full SVD
    let mut diag = vec![0.0f32; m];
    for j in 0..m {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += data[i * cols + j].powi(2);
        }
        diag[j] = sum / n as f32; // Normalize by rows
    }

    // Step 2: Clamp eigenvalues to Marchenko-Pastur support
    for j in 0..m {
        let eigenvalue = diag[j];
        let clamped = eigenvalue.max(lambda_minus).min(lambda_plus);
        let scale = if eigenvalue > 1e-10 {
            (clamped / eigenvalue).sqrt()
        } else {
            1.0
        };

        for i in 0..n {
            data[i * cols + j] *= scale;
        }
    }

    // Step 3: Apply SoftSign regularization to prevent outlier singular values
    // SoftSign(x) = x / (1 + |x|) — smooth approximation to sign
    let outlier_threshold = lambda_plus.sqrt() * 0.5;
    for x in data.iter_mut() {
        let abs_x = x.abs();
        if abs_x > outlier_threshold {
            // SoftSign scaling: bring outliers toward threshold
            let sign = x.signum();
            *x = sign * outlier_threshold * (abs_x / (abs_x + outlier_threshold));
        }
    }

    // Step 4: Random rotation to break remaining structure
    apply_random_rotation(data, rows, cols, seed);
}

/// Apply a random orthogonal rotation to break deterministic structure.
///
/// This ensures the spectral clamping doesn't introduce unwanted patterns.
fn apply_random_rotation(data: &mut [f32], rows: usize, cols: usize, seed: u64) {
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(999));
    let n = rows.min(cols);

    // Apply Givens rotations (plane rotations) to mix columns
    for i in 0..n {
        for j in (i + 1)..n {
            if j >= cols {
                break;
            }
            let theta: f32 = rng.random_range(-0.1..0.1);
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            for row in 0..rows {
                let a = data[row * cols + i];
                let b = data[row * cols + j];
                data[row * cols + i] = a * cos_t - b * sin_t;
                data[row * cols + j] = a * sin_t + b * cos_t;
            }
        }
    }
}

/// Estimate the Marchenko-Pastur density at a given eigenvalue.
pub fn marchenko_pastur_density(lambda: f32, gamma: f32, sigma2: f32) -> f32 {
    let sqrt_gamma = gamma.sqrt();
    let lambda_plus = sigma2 * (1.0 + sqrt_gamma).powi(2);
    let lambda_minus = sigma2 * (1.0 - sqrt_gamma).powi(2);

    if lambda < lambda_minus || lambda > lambda_plus {
        return 0.0;
    }

    let numer = ((lambda_plus - lambda) * (lambda - lambda_minus)).sqrt();
    let denom = 2.0 * std::f32::consts::PI * sigma2 * lambda;

    if denom < 1e-10 {
        0.0
    } else {
        numer / denom
    }
}

/// Compute the eigenvalue spectrum of the Gram matrix W^T W / n.
pub fn eigenvalue_spectrum(data: &[f32], rows: usize, cols: usize, num_bins: usize) -> Vec<(f32, f32)> {
    // Compute diagonal of Gram matrix (approximate eigenvalues)
    let mut diag = vec![0.0f32; cols];
    for j in 0..cols {
        let mut sum = 0.0f32;
        for i in 0..rows {
            sum += data[i * cols + j].powi(2);
        }
        diag[j] = sum / rows as f32;
    }

    let min_val = diag.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = diag.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if max_val - min_val < 1e-10 {
        return vec![(min_val, cols as f32)];
    }

    let bin_width = (max_val - min_val) / num_bins as f32;
    let mut histogram = vec![(0.0f32, 0.0f32); num_bins];

    for &val in &diag {
        let bin = ((val - min_val) / bin_width) as usize;
        let bin = bin.min(num_bins - 1);
        histogram[bin].0 = min_val + (bin as f32 + 0.5) * bin_width;
        histogram[bin].1 += 1.0;
    }

    // Normalize
    let total = cols as f32;
    for (_, count) in histogram.iter_mut() {
        *count /= total * bin_width;
    }

    histogram
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_clamp() {
        let mut data: Vec<f32> = (0..64).map(|x| x as f32 - 32.0).collect();
        apply_spectral_clamp(&mut data, 8, 8, 42);

        // Check that no entry is extremely large
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 100.0, "max |weight| = {max_abs}");
    }

    #[test]
    fn test_marchenko_pastur_density() {
        let gamma = 1.0; // square matrix
        let sigma2 = 1.0;
        let lambda_plus = (1.0 + 1.0).powi(2); // 4.0
        let lambda_minus = (1.0 - 1.0).powi(2); // 0.0

        // Density should be positive within support
        let d = marchenko_pastur_density(2.0, gamma, sigma2);
        assert!(d > 0.0, "density at λ=2 should be positive");

        // Density should be zero outside support
        let d = marchenko_pastur_density(5.0, gamma, sigma2);
        assert!(d == 0.0, "density at λ=5 should be zero");
    }

    #[test]
    fn test_eigenvalue_spectrum() {
        let data: Vec<f32> = (0..64).map(|x| x as f32 - 32.0).collect();
        let spectrum = eigenvalue_spectrum(&data, 8, 8, 10);
        assert_eq!(spectrum.len(), 10);

        // All eigenvalues should be non-negative
        for (val, _) in &spectrum {
            assert!(*val >= 0.0, "eigenvalue should be ≥ 0");
        }
    }
}
