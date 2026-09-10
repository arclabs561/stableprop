#![cfg(feature = "burn")]

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_leaky_relu, propagate_relu, propagate_relu_cross_covariance, propagate_relu_full,
    Moments, MomentsFull,
};

type Ad = Autodiff<NdArray<f32>>;

#[test]
fn finite_gradients_for_representable_subnormal_variances() {
    let device = Default::default();
    for variance in [f32::MIN_POSITIVE, 1e-40f32] {
        let sigma = variance.sqrt();
        for mode in ["relu", "leaky", "full", "cross"] {
            let mean =
                Tensor::<Ad, 2>::from_data([[0.0, sigma, 9.0 * sigma, -9.0 * sigma]], &device)
                    .require_grad();
            let var = Tensor::<Ad, 2>::from_data([[variance; 4]], &device).require_grad();
            let moments = Moments::new(mean.clone(), var.clone());
            let loss = match mode {
                "relu" => {
                    let out = propagate_relu(&moments);
                    out.mean.sum() + out.var.sum()
                }
                "leaky" => {
                    let out = propagate_leaky_relu(&moments, 0.3);
                    out.mean.sum() + out.var.sum()
                }
                "full" => {
                    let out =
                        propagate_relu_full(&MomentsFull::from_diagonal(mean.clone(), var.clone()));
                    out.mean.sum() + out.cov.sum()
                }
                "cross" => {
                    let cross = Tensor::<Ad, 3>::from_data([[[0.5 * variance; 4]]], &device);
                    propagate_relu_cross_covariance(cross, &moments).sum()
                }
                _ => unreachable!(),
            };
            let gradients = loss.backward();
            for gradient in [
                mean.grad(&gradients).unwrap(),
                var.grad(&gradients).unwrap(),
            ] {
                let values = gradient.into_data().to_vec::<f32>().unwrap();
                assert!(
                    values.iter().all(|x| x.is_finite()),
                    "{mode}, var={variance:e}: {values:?}"
                );
            }
        }
    }
}

#[test]
fn distant_means_with_subnormal_variance_have_finite_tail_gradients() {
    let device = Default::default();
    for mode in ["relu", "leaky", "full", "cross"] {
        let mean =
            Tensor::<Ad, 2>::from_data([[1.0, -1.0, f32::MAX, -f32::MAX]], &device).require_grad();
        let variance = 1e-40f32;
        let var = Tensor::<Ad, 2>::from_data([[variance; 4]], &device).require_grad();
        let moments = Moments::new(mean.clone(), var.clone());
        let (output_mean, output_var) = match mode {
            "relu" => {
                let out = propagate_relu(&moments);
                (out.mean, out.var)
            }
            "leaky" => {
                let out = propagate_leaky_relu(&moments, 0.25);
                (out.mean, out.var)
            }
            "full" => {
                let out =
                    propagate_relu_full(&MomentsFull::from_diagonal(mean.clone(), var.clone()));
                let variance = out.variance();
                (out.mean, variance)
            }
            "cross" => {
                let cross = Tensor::<Ad, 3>::from_data([[[0.5 * variance; 4]]], &device);
                let out = propagate_relu_cross_covariance(cross, &moments).reshape([1, 4]);
                (out.clone(), out)
            }
            _ => unreachable!(),
        };
        let slope = if mode == "leaky" { 0.25 } else { 0.0 };
        let (expected_mean, expected_var) = if mode == "cross" {
            let cross = vec![0.5 * variance, 0.0, 0.5 * variance, 0.0];
            (cross.clone(), cross)
        } else {
            (
                vec![1.0, -slope, f32::MAX, -slope * f32::MAX],
                vec![
                    variance,
                    slope * slope * variance,
                    variance,
                    slope * slope * variance,
                ],
            )
        };
        assert_eq!(
            output_mean.clone().into_data().to_vec::<f32>().unwrap(),
            expected_mean,
            "{mode} mean"
        );
        assert_eq!(
            output_var.clone().into_data().to_vec::<f32>().unwrap(),
            expected_var,
            "{mode} variance"
        );
        // Scale before reduction so even the f32::MAX output gives a finite loss.
        let loss = output_mean.mul_scalar(0.25).sum() + output_var.sum();
        assert!(
            loss.clone().into_scalar().is_finite(),
            "{mode}: loss overflow"
        );
        let gradients = loss.backward();
        let expected_mean_grad = if mode == "cross" {
            vec![0.0; 4]
        } else {
            vec![0.25, 0.25 * slope, 0.25, 0.25 * slope]
        };
        let expected_var_grad = if mode == "cross" {
            vec![0.0; 4]
        } else {
            vec![1.0, slope * slope, 1.0, slope * slope]
        };
        assert_eq!(
            mean.grad(&gradients)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            expected_mean_grad,
            "{mode} d/dmean"
        );
        assert_eq!(
            var.grad(&gradients)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            expected_var_grad,
            "{mode} d/dvariance"
        );
    }
}

#[test]
fn mixed_scale_relu_covariance_gradients_match_centered_series() {
    let device = Default::default();
    let v0 = f32::from_bits(1);
    let v1 = 1e38f32;
    let cross = 1e-5f32;
    assert!((cross as f64).powi(2) < v0 as f64 * v1 as f64);
    let cov = Tensor::<Ad, 3>::from_data([[[v0, cross], [cross, v1]]], &device).require_grad();
    let out = propagate_relu_full(&MomentsFull::new(
        Tensor::<Ad, 2>::zeros([1, 2], &device),
        cov.clone(),
    ));
    let gradients = out.cov.sum().backward();
    let actual = cov
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // At zero means, the implemented series is C/4 + C²/(4*pi*sqrt(v0*v1)).
    // Differentiate the sum of both off-diagonals and the marginal variances.
    let scale = (v0 as f64 * v1 as f64).sqrt();
    let correction = (cross as f64).powi(2) / (4.0 * std::f64::consts::PI * scale);
    let marginal = 0.5 - 0.5 / std::f64::consts::PI;
    let cross_grad = 0.25 + cross as f64 / (2.0 * std::f64::consts::PI * scale);
    let expected = [
        marginal - correction / v0 as f64,
        cross_grad,
        cross_grad,
        marginal - correction / v1 as f64,
    ];
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(expected.is_finite() && expected.abs() < f32::MAX as f64);
        let relative_error = (actual as f64 - expected).abs() / expected.abs();
        assert!(
            relative_error < 2e-4,
            "{actual:e} vs {expected:e}, relative error {relative_error:e}"
        );
    }
}

#[test]
fn equal_scale_relu_cross_covariance_depends_on_both_variances() {
    let device = Default::default();
    let cov = Tensor::<Ad, 3>::from_data([[[1.0, 0.5], [0.5, 1.0]]], &device).require_grad();
    let out = propagate_relu_full(&MomentsFull::new(
        Tensor::<Ad, 2>::zeros([1, 2], &device),
        cov.clone(),
    ));
    // Use one off-diagonal so symmetry of the loss cannot hide an asymmetric
    // derivative when the two normalization scales compare equal.
    let gradients = out.cov.slice([0..1, 0..1, 1..2]).sum().backward();
    let actual = cov
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let variance_grad = -0.25 / (8.0 * std::f32::consts::PI);
    let expected = [
        variance_grad,
        0.25 + 0.25 / std::f32::consts::PI,
        0.0,
        variance_grad,
    ];
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 2e-7, "{actual} vs {expected}");
    }
}
