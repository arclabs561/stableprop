#![cfg(feature = "burn")]

use burn::tensor::{DType, Tensor, TensorData};
use burn_ndarray::NdArray;
use proptest::prelude::*;
use stableprop::burn_sdp::{propagate_relu_full, MomentsFull};

#[test]
fn centered_relu_covariance_stays_within_series_remainder() {
    let device = Default::default();
    let pi = std::f64::consts::PI;
    // The exact centered bivariate Gaussian ReLU kernel has a closed form.
    // Its omitted even powers have nonnegative coefficients; their sum at
    // |rho| = 1 bounds the third-order remainder on [-1, 1].
    let max_remainder = (pi - 3.0) / (4.0 * pi);
    for rho in [-1.0f64, -0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 1.0] {
        let input = MomentsFull::new(
            Tensor::<NdArray<f32>, 2>::zeros([1, 2], &device),
            Tensor::from_data([[[1.0, rho as f32], [rho as f32, 1.0]]], &device),
        );
        let cov = propagate_relu_full(&input)
            .cov
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let exact = ((1.0 - rho * rho).sqrt() + (pi - rho.acos()) * rho - 1.0) / (2.0 * pi);
        let error = exact - cov[1] as f64;
        assert!(
            (-2e-6..=max_remainder + 2e-6).contains(&error),
            "rho={rho}: covariance={} reference={exact}",
            cov[1]
        );
        assert!((cov[0] as f64 - (0.5 - 1.0 / (2.0 * pi))).abs() < 2e-6);
        assert_eq!(cov[1], cov[2]);
        assert_eq!(cov[0], cov[3]);
        if rho == 1.0 {
            // Identical inputs have identical rectified outputs. Keeping exact
            // diagonals with truncated cross-covariance leaves a spurious
            // positive margin variance: a known approximation limit.
            let margin_var = cov[0] as f64 + cov[3] as f64 - 2.0 * cov[1] as f64;
            assert!((margin_var - 2.0 * max_remainder).abs() < 4e-6);
        }
    }
}

proptest! {
    // The exact pair kernel is independent from the implementation's Hermite
    // series.  At zero mean, c_3 vanishes; keeping three orders is therefore
    // the same as retaining the first two nonzero coefficients.  Parseval
    // gives the remaining marginal energy below.
    #[test]
    fn zero_mean_relu_k3_error_obeys_hermite_remainder(
        rho in prop_oneof![Just(-1.0f64), Just(1.0f64), -1.0f64..1.0f64],
        left_exponent in -6i32..=6,
        right_exponent in -6i32..=6,
    ) {
        let sigma_left = 10.0f64.powi(left_exponent);
        let sigma_right = 10.0f64.powi(right_exponent);
        let sigma_product = sigma_left * sigma_right;
        let device = Default::default();
        let input = MomentsFull::new(
            Tensor::<NdArray<f64>, 2>::zeros([1, 2], (&device, DType::F64)),
            Tensor::<NdArray<f64>, 3>::from_data(
                TensorData::new(
                    vec![
                        sigma_left * sigma_left,
                        rho * sigma_product,
                        rho * sigma_product,
                        sigma_right * sigma_right,
                    ],
                    [1, 2, 2],
                ),
                (&device, DType::F64),
            ),
        );
        let implemented = propagate_relu_full(&input)
            .cov
            .into_data()
            .to_vec::<f64>()
            .unwrap()[1];

        let pi = std::f64::consts::PI;
        let exact_centered = sigma_product
            * ((1.0 - rho * rho).sqrt() + (pi - rho.acos()) * rho - 1.0)
            / (2.0 * pi);
        let remaining_marginal_energy = 0.25 - 3.0 / (4.0 * pi);
        let remainder_bound = rho.abs().powi(4) * sigma_product * remaining_marginal_energy;
        // The pair oracle subtracts close terms near rho = 0, and the tensor
        // path obtains rho by two divisions. Scale the floor with the natural
        // covariance unit rather than imposing an absolute tolerance.
        let roundoff_floor = 512.0 * f64::EPSILON * sigma_product;

        prop_assert!(
            (exact_centered - implemented).abs() <= remainder_bound + roundoff_floor,
            "rho={rho}, sigmas=({sigma_left}, {sigma_right}), exact={exact_centered}, \
             implemented={implemented}, bound={remainder_bound}, floor={roundoff_floor}",
        );
    }
}
