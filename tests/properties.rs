//! Property-based tests for the f64 reference propagation: invariants that must
//! hold for generated diagonal-Gaussian inputs, checked with proptest.

use proptest::prelude::*;
use stableprop::{propagate_linear, propagate_relu, Moments};

/// A diagonal-Gaussian `Moments`: mean in [-5, 5]^d, variance from std in (0, 3].
fn moments() -> impl Strategy<Value = Moments> {
    (1usize..6).prop_flat_map(|d| {
        (
            prop::collection::vec(-5.0f64..5.0, d),
            prop::collection::vec(0.01f64..3.0, d),
        )
            .prop_map(move |(mean, std)| {
                let cov = (0..d)
                    .map(|i| {
                        let mut row = vec![0.0; d];
                        row[i] = std[i] * std[i];
                        row
                    })
                    .collect();
                Moments { mean, cov }
            })
    })
}

/// A weight matrix `[d_out, d_in]` and bias `[d_out]` for the given input dim.
fn linear(d_in: usize) -> impl Strategy<Value = (Vec<Vec<f64>>, Vec<f64>)> {
    (1usize..5).prop_flat_map(move |d_out| {
        (
            prop::collection::vec(prop::collection::vec(-3.0f64..3.0, d_in), d_out),
            prop::collection::vec(-3.0f64..3.0, d_out),
        )
    })
}

/// A full PSD covariance formed as `A * A^T`. Its first off-diagonal is
/// deliberately nonzero and signed, so affine covariance tests exercise the
/// cross terms rather than only a diagonal special case.
fn full_moments() -> impl Strategy<Value = Moments> {
    (2usize..6).prop_flat_map(|d| {
        (
            Just(d),
            prop::collection::vec(-3.0f64..3.0, d),
            0.25f64..2.0,
            prop_oneof![Just(-1.0f64), Just(1.0f64)],
            0.25f64..2.0,
            prop::collection::vec(-2.0f64..2.0, d * d),
        )
            .prop_map(move |(d, mean, first, sign, second, entries)| {
                let mut factor: Vec<Vec<f64>> = entries.chunks(d).map(|row| row.to_vec()).collect();
                factor[0] = vec![0.0; d];
                factor[0][0] = first;
                factor[1][0] = sign * second;

                let cov = (0..d)
                    .map(|i| {
                        (0..d)
                            .map(|j| (0..d).map(|k| factor[i][k] * factor[j][k]).sum())
                            .collect()
                    })
                    .collect();
                Moments { mean, cov }
            })
    })
}

/// A rectangular affine layer whose nonzero bias makes mean and covariance
/// propagation distinguishable in the same generated case.
fn rectangular_linear(d_in: usize) -> impl Strategy<Value = (Vec<Vec<f64>>, Vec<f64>)> {
    (1usize..6).prop_flat_map(move |d_out| {
        (
            prop::collection::vec(
                prop::collection::vec(prop_oneof![Just(0.0f64), -3.0f64..3.0], d_in),
                d_out,
            ),
            prop::collection::vec(prop_oneof![(-3.0f64..-0.1), (0.1f64..3.0)], d_out),
        )
    })
}

proptest! {
    /// ReLU output mean is non-negative (relu(x) >= 0).
    #[test]
    fn relu_mean_nonnegative(m in moments()) {
        let out = propagate_relu(&m);
        for &x in &out.mean {
            prop_assert!(x >= -1e-9, "negative relu mean {x}");
        }
    }

    /// ReLU output variance is non-negative.
    #[test]
    fn relu_variance_nonnegative(m in moments()) {
        let out = propagate_relu(&m);
        for i in 0..out.mean.len() {
            prop_assert!(out.cov[i][i] >= -1e-9, "negative relu var {}", out.cov[i][i]);
        }
    }

    /// ReLU does not increase per-feature variance: Var(relu(X)) <= Var(X).
    #[test]
    fn relu_reduces_variance(m in moments()) {
        let out = propagate_relu(&m);
        for i in 0..out.mean.len() {
            prop_assert!(
                out.cov[i][i] <= m.cov[i][i] + 1e-9,
                "relu raised variance: {} -> {}",
                m.cov[i][i],
                out.cov[i][i]
            );
        }
    }

    /// Linear mean is exact: out.mean[o] = bias[o] + sum_i W[o][i] * mean[i].
    #[test]
    fn linear_mean_is_exact((m, (w, b)) in moments().prop_flat_map(|m| {
        let d_in = m.mean.len();
        (Just(m), linear(d_in))
    })) {
        let out = propagate_linear(&m, &w, &b);
        for o in 0..b.len() {
            let expect = b[o] + (0..m.mean.len()).map(|i| w[o][i] * m.mean[i]).sum::<f64>();
            prop_assert!((out.mean[o] - expect).abs() < 1e-9, "{} vs {expect}", out.mean[o]);
        }
    }

    /// Affine covariance is the scalar quadratic form W C W^T, including
    /// signed off-diagonal covariance terms from a generated PSD matrix.
    #[test]
    fn linear_covariance_matches_independent_quadratic_form((m, (w, b)) in full_moments().prop_flat_map(|m| {
        let d_in = m.mean.len();
        (Just(m), rectangular_linear(d_in))
    })) {
        prop_assert!(m.cov[0][1].abs() > 0.0, "generator lost signed cross term");
        let out = propagate_linear(&m, &w, &b);
        for o in 0..w.len() {
            for p in 0..w.len() {
                let mut expected = 0.0;
                let mut term_scale = 0.0;
                for i in 0..m.mean.len() {
                    for j in 0..m.mean.len() {
                        let term = w[o][i] * m.cov[i][j] * w[p][j];
                        expected += term;
                        term_scale += term.abs();
                    }
                }
                prop_assert!(
                    (out.cov[o][p] - expected).abs() <= 1e-12 * term_scale.max(1e-12),
                    "covariance [{o}][{p}] {} vs {expected}",
                    out.cov[o][p],
                );
            }
        }
    }

    /// ReLU and reflected ReLU partition the first and second raw moments.
    #[test]
    fn relu_reflection_preserves_raw_moments((mean, variance) in (
        prop_oneof![Just(0.0f64), -5.0f64..5.0],
        prop_oneof![Just(0.0f64), 0.01f64..9.0],
    )) {
        let m = Moments { mean: vec![mean], cov: vec![vec![variance]] };
        let reflected = Moments { mean: vec![-mean], cov: vec![vec![variance]] };
        let positive = propagate_relu(&m);
        let negative = propagate_relu(&reflected);

        let mean_error = (positive.mean[0] - negative.mean[0] - mean).abs();
        let mean_scale = mean.abs().max(variance.sqrt()).max(1e-12);
        prop_assert!(mean_error <= 3e-6 * mean_scale, "mean reflection error {mean_error}");

        let actual_second = positive.mean[0].powi(2)
            + positive.cov[0][0]
            + negative.mean[0].powi(2)
            + negative.cov[0][0];
        let expected_second = mean.powi(2) + variance;
        let second_scale = expected_second.max(1e-12);
        prop_assert!(
            (actual_second - expected_second).abs() <= 3e-6 * second_scale,
            "second-moment reflection {} vs {expected_second}",
            actual_second,
        );
    }

    /// Positive rescaling commutes with ReLU moment propagation. Restricting
    /// alpha to [-4, 4] avoids tail zeros, so relative checks retain power.
    #[test]
    fn relu_is_positively_homogeneous((mean, std, scale) in (
        -2.0f64..2.0,
        0.5f64..3.0,
        0.25f64..4.0,
    )) {
        let variance = std * std;
        let base = Moments { mean: vec![mean], cov: vec![vec![variance]] };
        let scaled = Moments {
            mean: vec![scale * mean],
            cov: vec![vec![scale * scale * variance]],
        };
        let base_out = propagate_relu(&base);
        let scaled_out = propagate_relu(&scaled);
        let expected_mean = scale * base_out.mean[0];
        let expected_variance = scale * scale * base_out.cov[0][0];

        prop_assert!(
            (scaled_out.mean[0] - expected_mean).abs() / expected_mean.abs() <= 2e-12,
            "scaled mean {} vs {expected_mean}",
            scaled_out.mean[0],
        );
        prop_assert!(
            (scaled_out.cov[0][0] - expected_variance).abs() / expected_variance.abs() <= 2e-12,
            "scaled variance {} vs {expected_variance}",
            scaled_out.cov[0][0],
        );
    }
}
