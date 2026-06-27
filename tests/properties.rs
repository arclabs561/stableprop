//! Property-based tests for the f64 reference propagation: invariants that must
//! hold for ANY diagonal-Gaussian input, checked with proptest.

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
}
