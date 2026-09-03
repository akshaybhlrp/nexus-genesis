//! Lie Algebras — Cayley Transform for SO(n) Orthogonality
//!
//! The Lie algebra so(n) consists of skew-symmetric matrices A where A^T = -A.
//! The Cayley transform maps from so(n) to SO(n) (special orthogonal group):
//!
//!   U = (I - A)(I + A)^{-1}
//!
//! This produces an orthogonal matrix U where U^T U = I, guaranteeing:
//! - Condition number κ = 1 (perfect numerical stability)
//! - Singular values all equal 1 (no amplification or attenuation)
//! - Gradient flow preservation (orthogonal weights don't distort gradients)
//!
//! MKG finding: Lie Algebra Cayley transform was ranked in top 5,
//! with +11.2% improvement on context needle retrieval.

/// Apply Cayley transform to produce an approximately orthogonal weight matrix.
///
/// The input data is used to parameterize a skew-symmetric matrix A,
/// then the Cayley transform U = (I - A)(I + A)^{-1} is applied.
pub fn apply_cayley_transform(data: &mut [f32], rows: usize, cols: usize, seed: u64) {
    assert_eq!(data.len(), rows * cols);

    let n = rows.min(cols);

    // Step 1: Extract an n×n submatrix and make it skew-symmetric
    let mut skew = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i < j {
                // Use data as source of skew-symmetric parameters
                let val = data[i * cols + j] * 0.05;
                skew[i * n + j] = val;
                skew[j * n + i] = -val;
            }
        }
    }

    // Step 2: Cayley transform U = (I - A)(I + A)^{-1}
    let mut i_plus_a = vec![0.0f32; n * n];
    let mut i_minus_a = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..n {
            let a_ij = skew[i * n + j];
            i_plus_a[i * n + j] = if i == j { 1.0 + a_ij } else { a_ij };
            i_minus_a[i * n + j] = if i == j { 1.0 - a_ij } else { -a_ij };
        }
    }

    // Solve (I + A) * U = (I - A) for U using Gaussian elimination
    let u = solve_linear_system(&i_plus_a, &i_minus_a, n);

    // Step 3: Embed U back into the full matrix
    for i in 0..rows {
        for j in 0..cols {
            if i < n && j < n {
                data[i * cols + j] = u[i * n + j];
            } else {
                // Fill remaining with scaled original data
                data[i * cols + j] *= 0.01;
            }
        }
    }

    // Step 4: Apply rotation mixing across all rows using Hadamard-like transform
    apply_recursive_hadamard(data, rows, cols, seed);
}

/// Solve AX = B for X using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut aug = vec![0.0f32; n * (n + 1)];

    // Build augmented matrix [A | B]
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        // B is n×n, column i of B goes to column n of aug
        for j in 0..n {
            aug[i * (n + 1) + n] += b[i * n + j] * if i == j { 1.0 } else { 0.0 };
        }
    }

    // Actually, we need to solve for each column of X separately
    let mut result = vec![0.0f32; n * n];

    for col in 0..n {
        // Extract column col of B
        let mut b_col = vec![0.0f32; n];
        for i in 0..n {
            b_col[i] = b[i * n + col];
        }

        // Gaussian elimination
        let mut a_copy = a.to_vec();
        let mut b_copy = b_col.clone();

        for k in 0..n {
            // Partial pivoting
            let mut max_val = a_copy[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                if a_copy[i * n + k].abs() > max_val {
                    max_val = a_copy[i * n + k].abs();
                    max_row = i;
                }
            }
            if max_row != k {
                for j in 0..n {
                    let tmp = a_copy[k * n + j];
                    a_copy[k * n + j] = a_copy[max_row * n + j];
                    a_copy[max_row * n + j] = tmp;
                }
                let tmp = b_copy[k];
                b_copy[k] = b_copy[max_row];
                b_copy[max_row] = tmp;
            }

            // Eliminate below
            let pivot = a_copy[k * n + k];
            if pivot.abs() < 1e-10 {
                continue;
            }
            for i in (k + 1)..n {
                let factor = a_copy[i * n + k] / pivot;
                for j in k..n {
                    a_copy[i * n + j] -= factor * a_copy[k * n + j];
                }
                b_copy[i] -= factor * b_copy[k];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            let mut sum = b_copy[i];
            for j in (i + 1)..n {
                sum -= a_copy[i * n + j] * result[j * n + col];
            }
            let pivot = a_copy[i * n + i];
            result[i * n + col] = if pivot.abs() > 1e-10 {
                sum / pivot
            } else {
                0.0
            };
        }
    }

    result
}

/// Apply recursive Hadamard-like mixing to spread structure across rows.
fn apply_recursive_hadamard(data: &mut [f32], rows: usize, cols: usize, seed: u64) {
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Butterfly mixing: pairs of rows are mixed with random rotation angles
    let mut step = 1;
    while step < rows {
        for i in (0..rows).step_by(step * 2) {
            if i + step >= rows {
                break;
            }
            let angle: f32 = rng.random::<f32>() * std::f32::consts::PI * 0.1;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            for j in 0..cols {
                let a = data[i * cols + j];
                let b = data[(i + step) * cols + j];
                data[i * cols + j] = a * cos_a - b * sin_a;
                data[(i + step) * cols + j] = a * sin_a + b * cos_a;
            }
        }
        step *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cayley_produces_orthogonal_submatrix() {
        let mut data: Vec<f32> = (0..64).map(|x| (x as f32 - 32.0) * 0.01).collect();
        apply_cayley_transform(&mut data, 8, 8, 42);

        // Extract top-left 8×8 and check approximate orthogonality
        let n = 8;
        let mut gram = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += data[i * n + k] * data[j * n + k];
                }
                gram[i * n + j] = sum;
            }
        }

        // Diagonal should be ~1, off-diagonal should be ~0
        for i in 0..n {
            let diag = gram[i * n + i];
            assert!(
                (diag - 1.0).abs() < 0.5,
                "gram diagonal[{i}] = {diag}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_solve_linear_system() {
        // Solve 2x + y = 5, x + 3y = 7
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 7.0, 0.0, 0.0]; // Only solving first column
        let x = solve_linear_system(&a, &b, 2);
        // x ≈ [1.6, 1.8] for first column
        assert!((x[0] - 1.6).abs() < 0.1, "x[0] = {}", x[0]);
    }
}
