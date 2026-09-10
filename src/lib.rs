//! Stable distribution propagation through neural network layers.
//!
//! Implements moment-matching propagation of Gaussian distributions through
//! affine (linear) and ReLU layers. The linear case is exact. For a Gaussian
//! input, the ReLU step evaluates the closed-form univariate moments from
//! Frey & Hinton (1999) with numerical CDF and tail approximations, then drops
//! off-diagonal covariance before the next layer.
//!
//! The ReLU step in this module is the Frey & Hinton (1999) Gaussian
//! moment calculation with off-diagonal covariance dropped (diagonal
//! assumption). The feature-gated `burn_sdp` module also provides a
//! third-order full-covariance ReLU approximation and local-linear Cauchy
//! propagation. Those paths are related to, but do not reproduce, Petersen et
//! al.'s stable distribution propagation algorithm.
//!
//! The `burn_sdp` module (feature `burn`) provides the propagation on
//! Burn tensors: batched, differentiable, and composable with Burn models.

#[cfg(feature = "burn")]
pub mod burn_sdp;

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// First two moments of a multivariate Gaussian (mean + full covariance).
///
/// Inputs must be nonempty and finite. Covariance must be symmetric and
/// positive semidefinite; propagation checks dimensions and finiteness but
/// does not test symmetry or positive semidefiniteness.
#[derive(Debug, Clone)]
pub struct Moments {
    pub mean: Vec<f64>,
    /// Row-major `n x n` covariance matrix stored as `Vec<Vec<f64>>`.
    pub cov: Vec<Vec<f64>>,
}

/// A single neural-network layer.
#[derive(Debug, Clone)]
pub enum Layer {
    Linear {
        /// Row-major weight matrix, shape `[out, in]`.
        weight: Vec<Vec<f64>>,
        /// Bias vector, length `out`.
        bias: Vec<f64>,
    },
    ReLU,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Standard normal CDF via the error function.
fn std_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

/// Standard normal PDF.
fn std_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Error function approximation (Abramowitz & Stegun 7.1.26, max error < 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/// Matrix-vector product: A (m x n) * v (n) -> (m).
fn mat_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

fn validate_moments(moments: &Moments) {
    let n = moments.mean.len();
    assert!(n > 0, "mean must be non-empty");
    assert!(
        moments.mean.iter().all(|x| x.is_finite()),
        "mean must be finite"
    );
    assert_eq!(
        moments.cov.len(),
        n,
        "covariance must have one row per mean element"
    );
    for row in &moments.cov {
        assert_eq!(row.len(), n, "covariance must be square");
        assert!(
            row.iter().all(|x| x.is_finite()),
            "covariance must be finite"
        );
    }
}

/// Matrix multiply: A (m x k) * B (k x n) -> (m x n).
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = b[0].len();
    let k = b.len();
    a.iter()
        .map(|row_a| {
            (0..n)
                .map(|j| (0..k).map(|l| row_a[l] * b[l][j]).sum())
                .collect()
        })
        .collect()
}

/// Transpose of a matrix.
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

// ---------------------------------------------------------------------------
// Propagation
// ---------------------------------------------------------------------------

/// Propagate Gaussian moments through an affine (linear) layer.
///
/// ```text
/// mean' = W * mean + bias
/// cov'  = W * cov * W^T
/// ```
///
/// Finite inputs can still produce an infinite or `NaN` output when the exact
/// affine arithmetic exceeds the representable `f64` range.
///
/// # Panics
/// Panics on empty inputs, inconsistent dimensions, or non-finite values.
/// The covariance requirements on [`Moments`] also apply.
pub fn propagate_linear(moments: &Moments, weight: &[Vec<f64>], bias: &[f64]) -> Moments {
    validate_moments(moments);
    assert!(
        !weight.is_empty(),
        "weight must have at least one output row"
    );
    assert_eq!(
        bias.len(),
        weight.len(),
        "bias length must match weight rows"
    );
    for row in weight {
        assert_eq!(
            row.len(),
            moments.mean.len(),
            "weight columns must match mean length"
        );
        assert!(row.iter().all(|x| x.is_finite()), "weight must be finite");
    }
    assert!(bias.iter().all(|x| x.is_finite()), "bias must be finite");
    let new_mean: Vec<f64> = mat_vec(weight, &moments.mean)
        .iter()
        .zip(bias)
        .map(|(m, b)| m + b)
        .collect();

    // W * cov
    let wc = mat_mul(weight, &moments.cov);
    // (W * cov) * W^T
    let wt = transpose(weight);
    let new_cov = mat_mul(&wc, &wt);

    Moments {
        mean: new_mean,
        cov: new_cov,
    }
}

/// Propagate Gaussian moments through an element-wise ReLU using Frey & Hinton
/// (1999) moment matching.
///
/// For each dimension independently (diagonal approximation):
///
/// ```text
/// alpha  = mu / sigma
/// Phi    = std_normal_cdf(alpha)
/// phi    = std_normal_pdf(alpha)
/// mu'    = mu * Phi + sigma * phi
/// sigma' = sqrt( (mu^2 + sigma^2) * Phi + mu * sigma * phi - mu'^2 )
/// ```
///
/// The implementation evaluates variance without the second-moment subtraction
/// shown above, approximates the CDF, and uses linear tail limits beyond eight
/// standard deviations. Off-diagonal covariances are zeroed.
///
/// # Panics
/// Panics on empty inputs, inconsistent dimensions, non-finite values, or
/// negative diagonal variances. The covariance requirements on [`Moments`]
/// also apply.
pub fn propagate_relu(moments: &Moments) -> Moments {
    validate_moments(moments);
    let n = moments.mean.len();
    let mut new_mean = vec![0.0; n];
    let mut new_cov = vec![vec![0.0; n]; n];

    for i in 0..n {
        let mu = moments.mean[i];
        let var = moments.cov[i][i];

        assert!(
            var.is_finite() && var >= 0.0,
            "variance must be finite and non-negative"
        );
        if var == 0.0 {
            // Deterministic: apply ReLU to the mean exactly.
            let relu_mu = mu.max(0.0);
            new_mean[i] = relu_mu;
            // Variance stays zero.
            continue;
        }

        let sigma = var.sqrt();
        let alpha = mu / sigma;
        // In the far tails, use the numerically stable linear-limit
        // approximation rather than forming products that can overflow.
        if alpha >= 8.0 {
            new_mean[i] = mu;
            new_cov[i][i] = var;
            continue;
        }
        if alpha <= -8.0 {
            continue;
        }
        let phi = std_normal_pdf(alpha);
        let big_phi = std_normal_cdf(alpha);

        let mu_out = mu * big_phi + sigma * phi;
        // Algebraically equivalent to E[X_+^2] - E[X_+]^2, but does not
        // subtract two O(mu^2) values when ReLU is nearly linear.
        let normalized_var = alpha * alpha * big_phi * (1.0 - big_phi)
            + big_phi
            + alpha * phi * (1.0 - 2.0 * big_phi)
            - phi * phi;
        let var_out = var * normalized_var;

        new_mean[i] = mu_out;
        new_cov[i][i] = var_out.max(0.0); // clamp numerical noise
    }

    Moments {
        mean: new_mean,
        cov: new_cov,
    }
}

/// Propagate moments through a sequence of layers.
///
/// Inputs are independent Gaussian features described by their means and
/// standard deviations. Affine layers retain covariance; ReLU drops its
/// off-diagonal entries. An empty layer sequence returns the input moments.
///
/// # Panics
/// Panics on empty or mismatched input vectors, non-finite inputs, negative
/// standard deviations, standard deviations whose nonzero squared variance is
/// not representable as `f64`, or invalid layer dimensions or values.
pub fn propagate_sequential(layers: &[Layer], input_mean: &[f64], input_std: &[f64]) -> Moments {
    assert!(!input_mean.is_empty(), "input mean must be non-empty");
    assert_eq!(
        input_std.len(),
        input_mean.len(),
        "input std length must match mean length"
    );
    assert!(
        input_mean.iter().all(|x| x.is_finite()),
        "input mean must be finite"
    );
    assert!(
        input_std.iter().all(|x| {
            let variance = x * x;
            x.is_finite() && *x >= 0.0 && variance.is_finite() && (*x == 0.0 || variance != 0.0)
        }),
        "input std must be finite and non-negative, with a representable squared variance"
    );
    let n = input_mean.len();
    let mut moments = Moments {
        mean: input_mean.to_vec(),
        cov: (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = input_std[i] * input_std[i];
                row
            })
            .collect(),
    };

    for layer in layers {
        moments = match layer {
            Layer::Linear { weight, bias } => propagate_linear(&moments, weight, bias),
            Layer::ReLU => propagate_relu(&moments),
        };
    }

    moments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "representable squared variance")]
    fn sequential_rejects_variance_overflow_even_without_layers() {
        let _ = propagate_sequential(&[], &[0.0], &[f64::MAX]);
    }

    #[test]
    #[should_panic(expected = "representable squared variance")]
    fn sequential_rejects_variance_underflow_even_when_relu_mean_is_representable() {
        let _ = propagate_sequential(&[Layer::ReLU], &[0.0], &[1.0e-200]);
    }

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "{a} != {b} (diff = {}, tol = {tol})",
            (a - b).abs()
        );
    }

    #[test]
    fn linear_propagation_matches_analytical() {
        // 2D input, 2D output
        let moments = Moments {
            mean: vec![1.0, 2.0],
            cov: vec![vec![0.5, 0.1], vec![0.1, 0.3]],
        };
        let w = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let b = vec![1.0, -1.0];

        let out = propagate_linear(&moments, &w, &b);

        // mean' = W*mean + b = [2*1+0*2+1, 0*1+3*2-1] = [3, 5]
        approx_eq(out.mean[0], 3.0, 1e-12);
        approx_eq(out.mean[1], 5.0, 1e-12);

        // cov' = W * cov * W^T
        // W*cov = [[2*0.5, 2*0.1], [3*0.1, 3*0.3]] = [[1.0, 0.2], [0.3, 0.9]]
        // (W*cov)*W^T = [[1.0*2+0.2*0, 1.0*0+0.2*3], [0.3*2+0.9*0, 0.3*0+0.9*3]]
        //             = [[2.0, 0.6], [0.6, 2.7]]
        approx_eq(out.cov[0][0], 2.0, 1e-12);
        approx_eq(out.cov[0][1], 0.6, 1e-12);
        approx_eq(out.cov[1][0], 0.6, 1e-12);
        approx_eq(out.cov[1][1], 2.7, 1e-12);
    }

    #[test]
    fn relu_reduces_variance() {
        // Input with positive mean -- ReLU should pass most through but reduce variance.
        let moments = Moments {
            mean: vec![1.0, -1.0, 0.0],
            cov: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        };

        let out = propagate_relu(&moments);

        // Post-ReLU variance <= pre-ReLU variance for each dimension.
        for i in 0..3 {
            assert!(
                out.cov[i][i] <= moments.cov[i][i] + 1e-12,
                "dim {i}: post-ReLU var {} > pre-ReLU var {}",
                out.cov[i][i],
                moments.cov[i][i]
            );
        }

        // Positive mean: most mass passes through, mean should still be positive.
        assert!(out.mean[0] > 0.5);

        // Negative mean: ReLU clips most mass, mean should be small positive.
        assert!(out.mean[1] > 0.0);
        assert!(out.mean[1] < 0.5);

        // Zero mean: symmetric case, mean = sigma / sqrt(2*pi) ~ 0.3989
        approx_eq(out.mean[2], 1.0 / (2.0 * PI).sqrt(), 1e-4);
    }

    #[test]
    fn relu_handles_zero_tiny_and_large_signal_variances() {
        let moments = Moments {
            mean: vec![1.0e10, -2.0, 0.0],
            cov: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0e-20],
            ],
        };

        let out = propagate_relu(&moments);
        approx_eq(out.mean[0], 1.0e10, 1e-6);
        approx_eq(out.cov[0][0], 1.0, 1e-12);
        assert_eq!(out.mean[1], 0.0);
        assert_eq!(out.cov[1][1], 0.0);
        approx_eq(out.mean[2], 1.0e-10 / (2.0 * PI).sqrt(), 1e-18);
        approx_eq(out.cov[2][2], (0.5 - 1.0 / (2.0 * PI)) * 1.0e-20, 1e-28);
    }

    #[test]
    #[should_panic(expected = "bias length")]
    fn linear_rejects_mismatched_bias() {
        let moments = Moments {
            mean: vec![1.0, 2.0],
            cov: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let _ = propagate_linear(&moments, &[vec![3.0, 4.0]], &[]);
    }

    #[test]
    #[should_panic(expected = "input std length")]
    fn sequential_rejects_extra_std() {
        let _ = propagate_sequential(&[], &[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "input std must")]
    fn sequential_rejects_negative_std() {
        let _ = propagate_sequential(&[], &[1.0], &[-1.0]);
    }

    #[test]
    fn sequential_three_layer_network() {
        // input(2) -> Linear(2->3) -> ReLU -> Linear(3->2) -> ReLU -> Linear(2->1)
        let layers = vec![
            Layer::Linear {
                weight: vec![vec![1.0, 0.5], vec![-0.5, 1.0], vec![0.3, -0.3]],
                bias: vec![0.1, 0.0, -0.1],
            },
            Layer::ReLU,
            Layer::Linear {
                weight: vec![vec![1.0, -1.0, 0.5], vec![0.5, 1.0, -0.5]],
                bias: vec![0.0, 0.0],
            },
            Layer::ReLU,
            Layer::Linear {
                weight: vec![vec![1.0, 1.0]],
                bias: vec![0.0],
            },
        ];

        let input_mean = vec![1.0, 0.5];
        let input_std = vec![0.3, 0.2];

        let out = propagate_sequential(&layers, &input_mean, &input_std);

        // Output is 1-dimensional.
        assert_eq!(out.mean.len(), 1);
        assert_eq!(out.cov.len(), 1);
        assert_eq!(out.cov[0].len(), 1);

        // After ReLU layers, mean should be non-negative.
        assert!(out.mean[0] >= 0.0, "output mean = {}", out.mean[0]);
        // Variance should be non-negative.
        assert!(out.cov[0][0] >= 0.0, "output var = {}", out.cov[0][0]);
        // Should have finite, reasonable values.
        assert!(out.mean[0].is_finite());
        assert!(out.cov[0][0].is_finite());
        // Propagation should produce non-zero output for this input.
        assert!(
            out.mean[0] > 0.01,
            "output mean suspiciously small: {}",
            out.mean[0]
        );
    }
}
