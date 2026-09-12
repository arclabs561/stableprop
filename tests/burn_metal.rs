#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used)]

//! Local Apple-GPU parity and timing checks. They are deliberately ignored:
//! the runtime checks require an Apple GPU, while CI only compiles this harness.
//! Burn 0.22 chooses WGSL/SPIR-V/MSL at runtime, so this suite selects the real
//! `Device::metal` path and compares it with Flex rather than naming compiler aliases.

use std::{env, hint::black_box, time::Instant};

use burn::tensor::{DType, Device, DeviceKind, Tensor, TensorData};
use stableprop::burn_sdp::{
    propagate_conv2d, propagate_leaky_relu, propagate_linear, propagate_linear_bayes,
    propagate_linear_cauchy, propagate_linear_cross_covariance, propagate_linear_full,
    propagate_matmul_left, propagate_relu, propagate_relu_cauchy, propagate_relu_cross_covariance,
    propagate_relu_full, propagate_residual_add_correlated, Cauchy, Moments, MomentsFull,
};

fn metal_device() -> Device {
    Device::metal(DeviceKind::DefaultDevice)
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2e-4 + 2e-4 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}]: GPU={actual}, CPU={expected}, tolerance={tolerance}"
        );
    }
}

fn data<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().unwrap()
}

fn assert_large_cauchy_interval(device: &Device, dtype: DType, label: &str) {
    let cauchy = Cauchy::new(
        Tensor::<2>::zeros([1, 1], (device, dtype)),
        Tensor::<2>::ones([1, 1], (device, dtype)),
    );
    assert_eq!(
        cauchy.scale.dtype(),
        dtype,
        "{label} must preserve the requested factory dtype"
    );

    // tan(pi * p / 2) = 100_000 by construction. This is representable in
    // f32, although it exceeds the legacy Flex32 finfo maximum.
    let p = 2.0 * 100_000.0_f64.atan() / std::f64::consts::PI;
    let halfwidth = cauchy.interval_halfwidth(p);
    assert_eq!(halfwidth.dtype(), dtype, "{label} result dtype");
    let actual = halfwidth
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap()[0];
    let relative_error = (actual - 100_000.0).abs() / 100_000.0;
    assert!(
        relative_error <= 2e-5,
        "{label} Cauchy interval half-width: {actual} vs 100000, relative error {relative_error}"
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_cauchy_interval_uses_f32_range_beyond_legacy_finfo() {
    let flex = Device::flex();
    let metal = metal_device();
    assert_large_cauchy_interval(&flex, DType::F32, "Flex F32 control");
    assert_large_cauchy_interval(&metal, DType::Flex32, "Metal Flex32 target");
    metal.sync().unwrap();
}

#[derive(Clone, Copy)]
struct TailReference {
    alpha: f32,
    p: f32,
    phi: f32,
    mean: f32,
    var: f32,
}

// Subset of tests/relu_tail_accuracy.rs's mpmath-90-digit erfc references.
const TAILS: [TailReference; 4] = [
    TailReference {
        alpha: -3.0,
        p: 1.349_898e-3,
        phi: 4.431_848_5e-3,
        mean: 3.821_543e-4,
        var: 2.032_890_4e-4,
    },
    TailReference {
        alpha: -5.0,
        p: 2.866_515_8e-7,
        phi: 1.486_719_5e-6,
        mean: 5.346_165_5e-8,
        var: 1.934_329_2e-8,
    },
    TailReference {
        alpha: -7.0,
        p: 1.279_812_5e-12,
        phi: 9.134_72e-12,
        mean: 1.760_326e-13,
        var: 4.758_433_6e-14,
    },
    TailReference {
        alpha: -7.5,
        p: 3.190_891_7e-14,
        phi: 2.434_320_5e-13,
        mean: 4.115_178e-15,
        var: 1.045_083e-15,
    },
];

fn relative(actual: f32, expected: f32, label: &str) {
    assert!(expected != 0.0, "{label} reference must be nonzero");
    let error = (actual - expected).abs() / expected.abs();
    assert!(
        error <= 3e-4,
        "{label}: {actual:e} vs {expected:e}, rel={error:e}"
    );
}

fn tail_values(device: &Device, alpha: f32, variance: f32) -> [f32; 5] {
    let mean = alpha * variance.sqrt();
    let moments = Moments::new(
        Tensor::<2>::from_data([[mean]], (device, DType::F32)),
        Tensor::<2>::from_data([[variance]], (device, DType::F32)),
    );
    let diagonal = propagate_relu(&moments);
    let full = propagate_relu_full(&MomentsFull::from_diagonal(
        moments.mean.clone(),
        moments.var.clone(),
    ));
    let full_var = data(full.variance())[0];
    let full_mean = data(full.mean)[0];
    let cross = propagate_relu_cross_covariance(
        Tensor::<3>::from_data([[[variance]]], (device, DType::F32)),
        &moments,
    );
    [
        data(diagonal.mean)[0],
        data(diagonal.var)[0],
        full_mean,
        full_var,
        data(cross)[0],
    ]
}

fn tail_gradients(device: &Device, alpha: f32, variance: f32, mean_loss: bool) -> (f32, f32) {
    let device = device.clone().autodiff();
    let mean =
        Tensor::<2>::from_data([[alpha * variance.sqrt()]], (&device, DType::F32)).require_grad();
    let var = Tensor::<2>::from_data([[variance]], (&device, DType::F32)).require_grad();
    let out = propagate_relu(&Moments::new(mean.clone(), var.clone()));
    let loss = if mean_loss {
        out.mean.sum()
    } else {
        out.var.sum()
    };
    let gradients = loss.backward();
    (
        data(mean.grad(&gradients).unwrap())[0],
        data(var.grad(&gradients).unwrap())[0],
    )
}

fn assert_tail_values(device: &Device, label: &str) {
    for tail in TAILS {
        for variance in [2f32.powi(-40), 1.0, 2f32.powi(40)] {
            let sigma = variance.sqrt();
            let actual = tail_values(device, tail.alpha, variance);
            let expected_mean = sigma * tail.mean;
            let expected_var = variance * tail.var;
            relative(actual[0], expected_mean, &format!("{label} diagonal mean"));
            relative(
                actual[1],
                expected_var,
                &format!("{label} diagonal variance"),
            );
            relative(actual[2], expected_mean, &format!("{label} full mean"));
            relative(actual[3], expected_var, &format!("{label} full variance"));
            relative(actual[4], variance * tail.p, &format!("{label} cross gate"));
        }
    }
}

fn assert_tail_gradients(device: &Device, label: &str) {
    for tail in TAILS {
        for variance in [2f32.powi(-40), 1.0, 2f32.powi(40)] {
            let sigma = variance.sqrt();
            let mean = sigma * tail.mean;
            for (mean_loss, expected) in [
                (true, [tail.p, tail.phi / (2.0 * sigma)]),
                (
                    false,
                    [2.0 * mean * (1.0 - tail.p), tail.p - tail.mean * tail.phi],
                ),
            ] {
                let actual = tail_gradients(device, tail.alpha, variance, mean_loss);
                relative(actual.0, expected[0], &format!("{label} d/dmean"));
                relative(actual.1, expected[1], &format!("{label} d/dvariance"));
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_tail_relu_values_and_gradients_match_erfc_references() {
    let device = metal_device();
    assert_tail_values(&device, "Metal");
    assert_tail_gradients(&device, "Metal");
    device.sync().unwrap();
}

fn assert_distant_tail_gradients(device: &Device) {
    let device = device.clone().autodiff();
    // All inputs are normal f32 values. The old division backward still
    // overflows: mean / variance is 1e50 even though the selected slope is 0 or 1.
    for mode in ["relu", "leaky", "full", "cross"] {
        let mean = Tensor::<2>::from_data([[1e20, -1e20]], (&device, DType::F32)).require_grad();
        let var = Tensor::<2>::from_data([[1e-30; 2]], (&device, DType::F32)).require_grad();
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
                let cross = Tensor::<3>::from_data([[[5e-31; 2]]], (&device, DType::F32));
                let out = propagate_relu_cross_covariance(cross, &moments).reshape([1, 2]);
                (out.clone(), out)
            }
            _ => unreachable!(),
        };
        let slope = if mode == "leaky" { 0.25 } else { 0.0 };
        let (expected_mean, expected_var, mean_grad, var_grad) = if mode == "cross" {
            ([5e-31, 0.0], [5e-31, 0.0], [0.0; 2], [0.0; 2])
        } else {
            (
                [1e20, -slope * 1e20],
                [1e-30, slope * slope * 1e-30],
                [1.0, slope],
                [1.0, slope * slope],
            )
        };
        assert_eq!(data(output_mean.clone()), expected_mean, "{mode} mean");
        assert_eq!(data(output_var.clone()), expected_var, "{mode} variance");
        let gradients = (output_mean.sum() + output_var.sum()).backward();
        assert_eq!(
            data(mean.grad(&gradients).unwrap()),
            mean_grad,
            "{mode} d/dmean"
        );
        assert_eq!(
            data(var.grad(&gradients).unwrap()),
            var_grad,
            "{mode} d/dvariance"
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_distant_relu_tails_have_linear_gradients() {
    let device = metal_device();
    assert_distant_tail_gradients(&device);
}

fn assert_mixed_scale_covariance_gradients(device: &Device) {
    let device = device.clone().autodiff();
    let v0 = 1e-30f32;
    let v1 = 1e30f32;
    let cross = 0.25f32;
    let cov =
        Tensor::<3>::from_data([[[v0, cross], [cross, v1]]], (&device, DType::F32)).require_grad();
    let out = propagate_relu_full(&MomentsFull::new(
        Tensor::<2>::zeros([1, 2], (&device, DType::F32)),
        cov.clone(),
    ));
    let gradients = out.cov.sum().backward();
    let actual = data(cov.grad(&gradients).unwrap());
    // Centered third-order series: C/4 + C²/(4*pi*sqrt(v0*v1)).
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
        relative(actual, expected as f32, "mixed-scale covariance gradient");
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_mixed_scale_covariance_gradients_match_centered_series() {
    let device = metal_device();
    assert_mixed_scale_covariance_gradients(&device);
}

fn fixture(device: &Device) -> Vec<Vec<f32>> {
    // Includes zero variance, a central input, tiny scale, and linear tails.
    let mean = Tensor::<2>::from_data([[0.0, 1e-12, 9.0, -9.0]], (device, DType::F32));
    let var = Tensor::<2>::from_data([[0.0, 1e-24, 0.49, 1.0]], (device, DType::F32));
    let moments = Moments::new(mean.clone(), var.clone());
    let leaky = propagate_leaky_relu(&moments, 0.1);
    let weight = Tensor::<2>::from_data(
        [
            [0.5, -1.0, 0.25],
            [1.0, 0.5, -0.5],
            [-0.25, 0.75, 1.0],
            [0.5, -0.25, 0.5],
        ],
        (device, DType::F32),
    );
    let w_var = Tensor::<2>::from_data(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.01, 0.02],
            [0.02, 0.03, 0.01],
            [0.01, 0.02, 0.04],
        ],
        (device, DType::F32),
    );
    let bayes = propagate_linear_bayes(
        &moments,
        weight.clone(),
        w_var,
        Some((
            Tensor::from_data([0.1, -0.2, 0.3], (device, DType::F32)),
            Tensor::from_data([0.01, 0.02, 0.03], (device, DType::F32)),
        )),
    );
    let left = propagate_matmul_left(
        Tensor::from_data([[1.0], [-0.25]], (device, DType::F32)),
        &bayes,
    );
    let residual =
        propagate_residual_add_correlated(&left, &left, left.var.clone().mul_scalar(0.25));
    let full = MomentsFull::new(
        mean.clone(),
        Tensor::from_data(
            [[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1e-24, 0.0, 0.0],
                [0.0, 0.0, 0.49, 0.21],
                [0.0, 0.0, 0.21, 1.0],
            ]],
            (device, DType::F32),
        ),
    );
    let full = propagate_relu_full(&propagate_linear_full(&full, weight, None));
    let cross = propagate_relu_cross_covariance(
        propagate_linear_cross_covariance(
            // The deterministic first margin has a zero covariance column.
            Tensor::from_data([[[0.0, 1e-24, 0.2, -0.2]]], (device, DType::F32)),
            Tensor::from_data(
                [
                    [0.5, -1.0, 0.25],
                    [1.0, 0.5, -0.5],
                    [-0.25, 0.75, 1.0],
                    [0.5, -0.25, 0.5],
                ],
                (device, DType::F32),
            ),
        ),
        &bayes,
    );
    let (conv_mean, conv_var) = propagate_conv2d(
        Tensor::<4>::from_data([[[[0.0, 1.0], [2.0, -1.0]]]], (device, DType::F32)),
        Tensor::<4>::from_data([[[[1e-24, 0.25], [0.5, 1.0]]]], (device, DType::F32)),
        Tensor::<4>::from_data([[[[0.5, -1.0], [0.25, 0.75]]]], (device, DType::F32)),
        Some(Tensor::<1>::from_data([0.1], (device, DType::F32))),
        burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );
    let cauchy = propagate_relu_cauchy(&propagate_linear_cauchy(
        &Cauchy::new(mean, var.sqrt()),
        Tensor::from_data(
            [[1.0, -0.5], [-0.25, 0.75], [0.5, 1.0], [-1.0, 0.25]],
            (device, DType::F32),
        ),
        Some(Tensor::from_data([0.2, -0.1], (device, DType::F32))),
    ));
    vec![
        data(propagate_relu(&moments).mean),
        data(propagate_relu(&moments).var),
        data(leaky.mean),
        data(leaky.var),
        data(bayes.mean),
        data(bayes.var),
        data(residual.mean),
        data(residual.var),
        data(full.mean),
        data(full.cov),
        data(cross),
        data(conv_mean),
        data(conv_var),
        data(cauchy.location),
        data(cauchy.scale),
    ]
}

#[derive(Clone, Copy, Debug)]
enum GradientContribution {
    BayesLeftResidual,
    Cross,
    Conv,
    Cauchy,
    Full,
    ResidualCross,
    ResidualCrossConv,
    ResidualCrossConvCauchy,
    All,
}

fn gradient_fixture(device: &Device, contribution: GradientContribution) -> Vec<Option<Vec<f32>>> {
    let device = device.clone().autodiff();
    // CPU/GPU gradient parity stays off exact kinks; tiny correlated boundary
    // gradients have their own finite-value check below.
    let mean =
        Tensor::<2>::from_data([[0.1, 1e-4, 9.0, -9.0]], (&device, DType::F32)).require_grad();
    let var =
        Tensor::<2>::from_data([[1e-6, 1e-8, 0.49, 1.0]], (&device, DType::F32)).require_grad();
    let cross =
        Tensor::<3>::from_data([[[1e-5, 1e-6, 0.2, -0.2]]], (&device, DType::F32)).require_grad();
    let moments = Moments::new(mean.clone(), var.clone());
    let leaky = propagate_leaky_relu(&moments, 0.1);
    let weight = Tensor::<2>::from_data(
        [[0.5, -1.0], [1.0, 0.5], [-0.25, 0.75], [0.5, -0.25]],
        (&device, DType::F32),
    )
    .require_grad();
    let bayes = propagate_linear_bayes(
        &moments,
        weight.clone(),
        weight.clone() * weight.clone().mul_scalar(0.02),
        None,
    );
    let left = propagate_matmul_left(
        Tensor::from_data([[1.0], [0.5]], (&device, DType::F32)),
        &bayes,
    );
    let residual =
        propagate_residual_add_correlated(&left, &left, left.var.clone().mul_scalar(0.2));
    let full_cov = Tensor::<3>::from_data(
        [[
            [1e-6, 5e-8, 0.0, 0.0],
            [5e-8, 1e-8, 0.0, 0.0],
            [0.0, 0.0, 0.49, 0.21],
            [0.0, 0.0, 0.21, 1.0],
        ]],
        (&device, DType::F32),
    )
    .require_grad();
    let full = propagate_relu_full(&propagate_linear_full(
        &MomentsFull::new(mean.clone(), full_cov.clone()),
        weight.clone(),
        None,
    ));
    let relu_cross = propagate_relu_cross_covariance(
        propagate_linear_cross_covariance(cross.clone(), weight.clone()),
        &bayes,
    );
    let (conv_mean, conv_var) = propagate_conv2d(
        mean.clone().reshape([1, 1, 2, 2]),
        var.clone().reshape([1, 1, 2, 2]),
        Tensor::from_data([[[[0.5, -1.0], [0.25, 0.75]]]], (&device, DType::F32)),
        None,
        burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );
    let cauchy = propagate_relu_cauchy(&propagate_linear_cauchy(
        // Keep this differentiability probe away from the local Cauchy gate's
        // zero subgradient and a zero scale; the forward fixture covers both.
        &Cauchy::new(
            mean.clone().add_scalar(0.17),
            var.clone().add_scalar(1e-4).sqrt(),
        ),
        weight.clone(),
        None,
    ));
    let loss = match contribution {
        GradientContribution::BayesLeftResidual => residual.mean.sum() + residual.var.sum(),
        GradientContribution::Cross => relu_cross.sum(),
        GradientContribution::Conv => conv_mean.sum() + conv_var.sum(),
        GradientContribution::Cauchy => cauchy.location.sum() + cauchy.scale.sum(),
        GradientContribution::Full => full.mean.sum() + full.cov.sum(),
        GradientContribution::ResidualCross => {
            residual.mean.sum() + residual.var.sum() + relu_cross.sum()
        }
        GradientContribution::ResidualCrossConv => {
            residual.mean.sum()
                + residual.var.sum()
                + relu_cross.sum()
                + conv_mean.sum()
                + conv_var.sum()
        }
        GradientContribution::ResidualCrossConvCauchy => {
            residual.mean.sum()
                + residual.var.sum()
                + relu_cross.sum()
                + conv_mean.sum()
                + conv_var.sum()
                + cauchy.location.sum()
                + cauchy.scale.sum()
        }
        GradientContribution::All => {
            residual.mean.sum()
                + residual.var.sum()
                + leaky.mean.sum()
                + leaky.var.sum()
                + full.mean.sum()
                + full.cov.sum()
                + relu_cross.sum()
                + conv_mean.sum()
                + conv_var.sum()
                + cauchy.location.sum()
                + cauchy.scale.sum()
        }
    };
    let gradients = loss.backward();
    vec![
        mean.grad(&gradients).map(data),
        var.grad(&gradients).map(data),
        cross.grad(&gradients).map(data),
        weight.grad(&gradients).map(data),
        full_cov.grad(&gradients).map(data),
    ]
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_matches_flex_forward_and_autodiff() {
    let cpu_device = Device::flex();
    let gpu_device = metal_device();
    let expected = fixture(&cpu_device);
    for (index, (actual, expected)) in fixture(&gpu_device).iter().zip(&expected).enumerate() {
        close(actual, expected, &format!("Metal forward fixture {index}"));
    }
    // Absolute tolerance alone would accept losing the 1e-24 variance.
    let gpu_tiny = fixture(&gpu_device)[1][1] / 1e-24;
    let cpu_tiny = expected[1][1] / 1e-24;
    assert!(
        (gpu_tiny - cpu_tiny).abs() < 0.02,
        "tiny variance GPU={gpu_tiny} CPU={cpu_tiny}"
    );
    for contribution in [
        GradientContribution::BayesLeftResidual,
        GradientContribution::Cross,
        GradientContribution::Cauchy,
        GradientContribution::Full,
        GradientContribution::ResidualCross,
    ] {
        let expected_gradients = gradient_fixture(&cpu_device, contribution);
        let actual = gradient_fixture(&gpu_device, contribution);
        gpu_device.sync().unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected_gradients).enumerate() {
            match (actual, expected) {
                (Some(actual), Some(expected)) => close(
                    actual,
                    expected,
                    &format!("Metal {contribution:?} gradient {index}"),
                ),
                (None, None) => {}
                _ => panic!("Metal {contribution:?} gradient presence differs at {index}"),
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_accumulates_shared_branch_gradients() {
    let cpu_device = Device::flex();
    let gpu_device = metal_device();
    let contribution = GradientContribution::ResidualCross;
    let expected = gradient_fixture(&cpu_device, contribution);
    let actual = gradient_fixture(&gpu_device, contribution);
    gpu_device.sync().unwrap();
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        match (actual, expected) {
            (Some(actual), Some(expected)) => close(
                actual,
                expected,
                &format!("Metal {contribution:?} gradient {index}"),
            ),
            (None, None) => {}
            _ => panic!("Metal {contribution:?} gradient presence differs at {index}"),
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_extended_fixture_matches_flex_forward_and_autodiff() {
    let cpu_device = Device::flex();
    let gpu_device = metal_device();
    for (index, (actual, expected)) in fixture(&gpu_device)
        .iter()
        .zip(fixture(&cpu_device))
        .enumerate()
    {
        close(actual, &expected, &format!("Metal forward fixture {index}"));
    }
    for contribution in [
        GradientContribution::BayesLeftResidual,
        GradientContribution::Cross,
        GradientContribution::Conv,
        GradientContribution::Cauchy,
        GradientContribution::Full,
        GradientContribution::ResidualCross,
        GradientContribution::ResidualCrossConv,
        GradientContribution::ResidualCrossConvCauchy,
        GradientContribution::All,
    ] {
        let expected = gradient_fixture(&cpu_device, contribution);
        let actual = gradient_fixture(&gpu_device, contribution);
        gpu_device.sync().unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            match (actual, expected) {
                (Some(actual), Some(expected)) => close(
                    actual,
                    expected,
                    &format!("Metal {contribution:?} gradient {index}"),
                ),
                (None, None) => {}
                _ => panic!("Metal {contribution:?} gradient presence differs at {index}"),
            }
        }
    }
}

fn direct_conv_gradients(device: &Device) -> (Vec<f32>, Vec<f32>) {
    let device = device.clone().autodiff();
    let input =
        Tensor::<4>::from_data([[[[0.0, 1.0], [2.0, -1.0]]]], (&device, DType::F32)).require_grad();
    let weight = Tensor::<4>::from_data([[[[0.5, -1.0], [0.25, 0.75]]]], (&device, DType::F32))
        .require_grad();
    let output = burn::tensor::module::conv2d(
        input.clone(),
        weight.clone(),
        None,
        burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );
    let gradients = output.sum().backward();
    (
        data(input.grad(&gradients).unwrap()),
        data(weight.grad(&gradients).unwrap()),
    )
}

fn shared_weight_conv_gradients(device: &Device) -> (Vec<f32>, Vec<f32>) {
    let device = device.clone().autodiff();
    let input =
        Tensor::<4>::from_data([[[[0.0, 1.0], [2.0, -1.0]]]], (&device, DType::F32)).require_grad();
    let weight = Tensor::<4>::from_data([[[[0.5, -1.0], [0.25, 0.75]]]], (&device, DType::F32))
        .require_grad();
    let options = burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let output = burn::tensor::module::conv2d(input.clone(), weight.clone(), None, options.clone())
        + burn::tensor::module::conv2d(input.clone(), weight.clone(), None, options);
    let gradients = output.sum().backward();
    (
        data(input.grad(&gradients).unwrap()),
        data(weight.grad(&gradients).unwrap()),
    )
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_direct_conv2d_gradients_match_flex_analytic() {
    let cpu_device = Device::flex();
    let gpu_device = metal_device();
    let (cpu_input, cpu_weight) = direct_conv_gradients(&cpu_device);
    let (gpu_input, gpu_weight) = direct_conv_gradients(&gpu_device);
    gpu_device.sync().unwrap();
    close(&gpu_input, &cpu_input, "Metal direct conv input gradient");
    close(
        &gpu_weight,
        &cpu_weight,
        "Metal direct conv weight gradient",
    );
    close(
        &gpu_input,
        &[0.5, -1.0, 0.25, 0.75],
        "Metal direct conv input analytic gradient",
    );
    close(
        &gpu_weight,
        &[0.0, 1.0, 2.0, -1.0],
        "Metal direct conv weight analytic gradient",
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_direct_conv2d_gradients_match_flex() {
    let cpu_device = Device::flex();
    let gpu_device = metal_device();
    let (cpu_input, cpu_weight) = direct_conv_gradients(&cpu_device);
    let (gpu_input, gpu_weight) = direct_conv_gradients(&gpu_device);
    gpu_device.sync().unwrap();
    close(&gpu_input, &cpu_input, "Metal direct conv input gradient");
    close(
        &gpu_weight,
        &cpu_weight,
        "Metal direct conv weight gradient",
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_shared_weight_conv2d_gradients_match_flex() {
    let cpu_device = Device::flex();
    let gpu_device = metal_device();
    let (cpu_input, cpu_weight) = shared_weight_conv_gradients(&cpu_device);
    let (gpu_input, gpu_weight) = shared_weight_conv_gradients(&gpu_device);
    gpu_device.sync().unwrap();
    close(&gpu_input, &cpu_input, "Metal shared conv input gradient");
    close(
        &gpu_weight,
        &cpu_weight,
        "Metal shared conv weight gradient",
    );
}

fn diagonal_work(device: &Device, batch: usize, width: usize) -> Moments {
    let mean = Tensor::<2>::full([batch, width], 0.2, (device, DType::F32));
    let var = Tensor::<2>::full([batch, width], 0.3, (device, DType::F32));
    let weight = Tensor::<2>::full([width, width], 0.01, (device, DType::F32));
    propagate_relu(&propagate_linear_bayes(
        &Moments::new(mean, var),
        weight.clone(),
        weight.mul_scalar(0.1),
        None,
    ))
}

fn full_work(device: &Device, batch: usize, width: usize) -> MomentsFull {
    let mean = Tensor::<2>::full([batch, width], 0.2, (device, DType::F32));
    let var = Tensor::<2>::full([batch, width], 0.3, (device, DType::F32));
    let weight = Tensor::<2>::full([width, width], 0.01, (device, DType::F32));
    propagate_relu_full(&propagate_linear_full(
        &MomentsFull::from_diagonal(mean, var),
        weight,
        None,
    ))
}

fn diagonal_training_work(device: &Device, batch: usize, width: usize) -> Tensor<2> {
    let device = device.clone().autodiff();
    let mean = Tensor::<2>::full([batch, width], 0.2, (&device, DType::F32)).require_grad();
    let var = Tensor::<2>::full([batch, width], 0.3, (&device, DType::F32)).require_grad();
    let weight = Tensor::<2>::full([width, width], 0.01, (&device, DType::F32)).require_grad();
    let output = propagate_relu(&propagate_linear_bayes(
        &Moments::new(mean.clone(), var),
        weight.clone(),
        weight.mul_scalar(0.1),
        None,
    ));
    mean.grad(&output.mean.sum().backward()).unwrap()
}

struct FullBackwardFixture {
    mean: Tensor<2>,
    cov: Tensor<3>,
    weight: Tensor<2>,
}

struct FullBackwardGradients {
    mean: Tensor<2>,
    cov: Tensor<3>,
    weight: Tensor<2>,
}

fn full_backward_fixture(device: &Device, batch: usize, width: usize) -> FullBackwardFixture {
    let device = device.clone().autodiff();
    let cov: Vec<f32> = (0..batch)
        .flat_map(|_| {
            (0..width).flat_map(move |i| (0..width).map(move |j| if i == j { 0.3 } else { 0.001 }))
        })
        .collect();
    FullBackwardFixture {
        mean: Tensor::<2>::from_data(
            TensorData::new(vec![0.2; batch * width], [batch, width]),
            (&device, DType::F32),
        ),
        cov: Tensor::<3>::from_data(
            TensorData::new(cov, [batch, width, width]),
            (&device, DType::F32),
        ),
        weight: Tensor::<2>::from_data(
            TensorData::new(vec![0.01; width * width], [width, width]),
            (&device, DType::F32),
        ),
    }
}

fn full_backward_work(fixture: &FullBackwardFixture) -> FullBackwardGradients {
    let mean = fixture.mean.clone().require_grad();
    let cov = fixture.cov.clone().require_grad();
    let weight = fixture.weight.clone().require_grad();
    let output = propagate_relu_full(&propagate_linear_full(
        &MomentsFull::new(mean.clone(), cov.clone()),
        weight.clone(),
        None,
    ));
    let gradients = (output.mean.sum() + output.cov.sum()).backward();
    FullBackwardGradients {
        mean: mean.grad(&gradients).unwrap(),
        cov: cov.grad(&gradients).unwrap(),
        weight: weight.grad(&gradients).unwrap(),
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn metal_rank_one_full_backward_matches_flex() {
    // Several rows and features expose partially evaluated broadcast selection
    // kernels that tiny matrices can miss. Constant weight columns produce a
    // rank-one output covariance and equal feature standard deviations.
    fn compare(device: &Device, width: usize, cpu: &FullBackwardGradients) {
        let fixture = full_backward_fixture(device, 8, width);
        let gradients = full_backward_work(&fixture);
        close(
            &data(gradients.mean),
            &data(cpu.mean.clone()),
            "rank-one mean gradient",
        );
        close(
            &data(gradients.cov),
            &data(cpu.cov.clone()),
            "rank-one covariance gradient",
        );
        close(
            &data(gradients.weight),
            &data(cpu.weight.clone()),
            "rank-one weight gradient",
        );
    }
    let dev = metal_device();
    for width in [4, 16] {
        let cpu = full_backward_work(&full_backward_fixture(&Device::flex(), 8, width));
        compare(&dev, width, &cpu);
    }
}

fn timed<T>(label: &str, device: &Device, mut work: impl FnMut() -> T) -> T {
    let warmup = work();
    device.sync().unwrap();
    black_box(warmup);
    let start = Instant::now();
    let result = work();
    device.sync().unwrap();
    let elapsed = start.elapsed();
    eprintln!("{label}: {elapsed:?}");
    black_box(result)
}

fn timed_repeated<T>(label: &str, device: &Device, mut work: impl FnMut() -> T) -> T {
    let warmup = work();
    device.sync().unwrap();
    black_box(warmup);
    let mut last = None;
    for repeat in 1..=3 {
        let start = Instant::now();
        let result = work();
        device.sync().unwrap();
        eprintln!("{label} repeat={repeat}: {:?}", start.elapsed());
        last = Some(result);
    }
    black_box(last.unwrap())
}

#[test]
#[ignore = "requires a Metal GPU and is an informational local benchmark"]
fn metal_full_covariance_backward_timing_matches_flex_and_series_control() {
    let cpu_device = Device::flex();
    let gpu_device = metal_device();
    eprintln!("Warmed f32 full backward; tensor-resident timing includes graph allocation, host validation excluded.");
    for &(batch, width, iterations) in &[(8, 16, 8), (64, 64, 1)] {
        let cpu_fixture = full_backward_fixture(&cpu_device, batch, width);
        let gpu_fixture = full_backward_fixture(&gpu_device, batch, width);
        let cpu = timed_repeated(
            &format!("CPU full backward b={batch} w={width} n={iterations}"),
            &cpu_device,
            || {
                let mut last = full_backward_work(&cpu_fixture);
                for _ in 1..iterations {
                    last = full_backward_work(&cpu_fixture);
                }
                last
            },
        );
        let gpu = timed_repeated(
            &format!("Metal full backward b={batch} w={width} n={iterations}"),
            &gpu_device,
            || {
                let mut last = full_backward_work(&gpu_fixture);
                for _ in 1..iterations {
                    last = full_backward_work(&gpu_fixture);
                }
                last
            },
        );
        close(
            &data(gpu.mean),
            &data(cpu.mean),
            "timed full backward mean gradient",
        );
        close(
            &data(gpu.cov),
            &data(cpu.cov),
            "timed full backward covariance gradient",
        );
        close(
            &data(gpu.weight),
            &data(cpu.weight),
            "timed full backward weight gradient",
        );
    }
    assert_mixed_scale_covariance_gradients(&cpu_device);
    assert_mixed_scale_covariance_gradients(&gpu_device);
}

#[test]
#[ignore = "requires a Metal GPU and is an informational local benchmark"]
fn metal_synchronized_diagonal_and_full_timings() {
    let cpu_device = Device::flex();
    let gpu_device = metal_device();
    eprintln!("Warmed f32 workloads; allocation included, host validation excluded.");
    for &(batch, width, iterations) in &[(8, 16, 32), (64, 64, 8), (256, 256, 1)] {
        let cpu = timed(
            &format!("CPU diagonal b={batch} w={width} n={iterations}"),
            &cpu_device,
            || {
                let mut last = diagonal_work(&cpu_device, batch, width);
                for _ in 1..iterations {
                    last = diagonal_work(&cpu_device, batch, width);
                }
                last
            },
        );
        let gpu = timed(
            &format!("Metal diagonal b={batch} w={width} n={iterations}"),
            &gpu_device,
            || {
                let mut last = diagonal_work(&gpu_device, batch, width);
                for _ in 1..iterations {
                    last = diagonal_work(&gpu_device, batch, width);
                }
                last
            },
        );
        close(&data(gpu.mean), &data(cpu.mean), "timed diagonal mean");
        close(&data(gpu.var), &data(cpu.var), "timed diagonal variance");
        if width <= 64 {
            let cpu = timed(
                &format!("CPU full b={batch} w={width} n={iterations}"),
                &cpu_device,
                || {
                    let mut last = full_work(&cpu_device, batch, width);
                    for _ in 1..iterations {
                        last = full_work(&cpu_device, batch, width);
                    }
                    last
                },
            );
            let gpu = timed(
                &format!("Metal full b={batch} w={width} n={iterations}"),
                &gpu_device,
                || {
                    let mut last = full_work(&gpu_device, batch, width);
                    for _ in 1..iterations {
                        last = full_work(&gpu_device, batch, width);
                    }
                    last
                },
            );
            close(&data(gpu.mean), &data(cpu.mean), "timed full mean");
            close(&data(gpu.cov), &data(cpu.cov), "timed full covariance");
        }
        let cpu = timed(
            &format!("CPU diagonal training b={batch} w={width} n={iterations}"),
            &cpu_device,
            || {
                let mut last = diagonal_training_work(&cpu_device, batch, width);
                for _ in 1..iterations {
                    last = diagonal_training_work(&cpu_device, batch, width);
                }
                last
            },
        );
        let gpu = timed(
            &format!("Metal diagonal training b={batch} w={width} n={iterations}"),
            &gpu_device,
            || {
                let mut last = diagonal_training_work(&gpu_device, batch, width);
                for _ in 1..iterations {
                    last = diagonal_training_work(&gpu_device, batch, width);
                }
                last
            },
        );
        close(&data(gpu), &data(cpu), "timed input gradient");
    }
}

struct MatchedHostFixture {
    mean: Vec<f32>,
    variance: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    output_std: Vec<f32>,
    batch: usize,
    width: usize,
}

fn matched_host_fixture(batch: usize, width: usize, regime: &str) -> MatchedHostFixture {
    let mean = vec![0.0; batch * width];
    let variance = vec![0.3; batch * width];
    let weight: Vec<f32> = (0..width * width)
        .map(|index| 0.025 * ((index * 5 + index / width * 3) % 11) as f32 - 0.125)
        .collect();
    let output_std: Vec<f32> = (0..width)
        .map(|output| {
            (0..width)
                .map(|input| {
                    let weight = weight[input * width + output];
                    0.3 * weight * weight
                })
                .sum::<f32>()
                .sqrt()
        })
        .collect();
    let bias: Vec<f32> = output_std
        .iter()
        .enumerate()
        .map(|(output, std)| {
            let z = match regime {
                "central" => [-1.0, 0.0, 1.0][output % 3],
                "tail" => -7.0,
                _ => unreachable!("validated profile regime"),
            };
            z * std
        })
        .collect();
    let z: Vec<f32> = bias
        .iter()
        .zip(&output_std)
        .map(|(bias, std)| bias / std)
        .collect();
    match regime {
        "central" => assert!(z.iter().all(|z| (-1.0..=1.0).contains(z))),
        "tail" => assert!(z.iter().all(|z| (*z + 7.0).abs() < 1e-6)),
        _ => unreachable!("validated profile regime"),
    }
    MatchedHostFixture {
        mean,
        variance,
        weight,
        bias,
        output_std,
        batch,
        width,
    }
}

fn full_diagonal_covariance(host: &MatchedHostFixture) -> Vec<f32> {
    (0..host.batch)
        .flat_map(|batch| {
            (0..host.width).flat_map(move |row| {
                (0..host.width).map(move |column| {
                    if row == column {
                        host.variance[batch * host.width + row]
                    } else {
                        0.0
                    }
                })
            })
        })
        .collect()
}

struct MatchedDiagonalForwardFixture {
    mean: Tensor<2>,
    variance: Tensor<2>,
    weight: Tensor<2>,
    bias: Tensor<1>,
}

fn matched_diagonal_forward_fixture(
    device: &Device,
    host: &MatchedHostFixture,
) -> MatchedDiagonalForwardFixture {
    MatchedDiagonalForwardFixture {
        mean: Tensor::from_data(
            TensorData::new(host.mean.clone(), [host.batch, host.width]),
            (device, DType::F32),
        ),
        variance: Tensor::from_data(
            TensorData::new(host.variance.clone(), [host.batch, host.width]),
            (device, DType::F32),
        ),
        weight: Tensor::from_data(
            TensorData::new(host.weight.clone(), [host.width, host.width]),
            (device, DType::F32),
        ),
        bias: Tensor::from_data(
            TensorData::new(host.bias.clone(), [host.width]),
            (device, DType::F32),
        ),
    }
}

fn matched_diagonal_forward(fixture: &MatchedDiagonalForwardFixture) -> Moments {
    let input = Moments::new(fixture.mean.clone(), fixture.variance.clone());
    propagate_relu(&propagate_linear(
        &input,
        fixture.weight.clone(),
        Some(fixture.bias.clone()),
    ))
}

struct MatchedFullForwardFixture {
    mean: Tensor<2>,
    covariance: Tensor<3>,
    weight: Tensor<2>,
    bias: Tensor<1>,
}

fn matched_full_forward_fixture(
    device: &Device,
    host: &MatchedHostFixture,
) -> MatchedFullForwardFixture {
    MatchedFullForwardFixture {
        mean: Tensor::from_data(
            TensorData::new(host.mean.clone(), [host.batch, host.width]),
            (device, DType::F32),
        ),
        covariance: Tensor::from_data(
            TensorData::new(
                full_diagonal_covariance(host),
                [host.batch, host.width, host.width],
            ),
            (device, DType::F32),
        ),
        weight: Tensor::from_data(
            TensorData::new(host.weight.clone(), [host.width, host.width]),
            (device, DType::F32),
        ),
        bias: Tensor::from_data(
            TensorData::new(host.bias.clone(), [host.width]),
            (device, DType::F32),
        ),
    }
}

fn matched_full_forward(fixture: &MatchedFullForwardFixture) -> MomentsFull {
    let input = MomentsFull::new(fixture.mean.clone(), fixture.covariance.clone());
    propagate_relu_full(&propagate_linear_full(
        &input,
        fixture.weight.clone(),
        Some(fixture.bias.clone()),
    ))
}

struct MatchedDiagonalBackwardFixture {
    mean: Tensor<2>,
    variance: Tensor<2>,
    weight: Tensor<2>,
    bias: Tensor<1>,
}

fn matched_diagonal_backward_fixture(
    device: &Device,
    host: &MatchedHostFixture,
) -> MatchedDiagonalBackwardFixture {
    let device = device.clone().autodiff();
    MatchedDiagonalBackwardFixture {
        mean: Tensor::from_data(
            TensorData::new(host.mean.clone(), [host.batch, host.width]),
            (&device, DType::F32),
        ),
        variance: Tensor::from_data(
            TensorData::new(host.variance.clone(), [host.batch, host.width]),
            (&device, DType::F32),
        ),
        weight: Tensor::from_data(
            TensorData::new(host.weight.clone(), [host.width, host.width]),
            (&device, DType::F32),
        ),
        bias: Tensor::from_data(
            TensorData::new(host.bias.clone(), [host.width]),
            (&device, DType::F32),
        ),
    }
}

struct MatchedDiagonalGradients {
    mean: Tensor<2>,
    variance: Tensor<2>,
    weight: Tensor<2>,
    bias: Tensor<1>,
}

fn matched_diagonal_backward(fixture: &MatchedDiagonalBackwardFixture) -> MatchedDiagonalGradients {
    let mean = fixture.mean.clone().require_grad();
    let variance = fixture.variance.clone().require_grad();
    let weight = fixture.weight.clone().require_grad();
    let bias = fixture.bias.clone().require_grad();
    let output = propagate_relu(&propagate_linear(
        &Moments::new(mean.clone(), variance.clone()),
        weight.clone(),
        Some(bias.clone()),
    ));
    let gradients = (output.mean.sum() + output.var.sum()).backward();
    MatchedDiagonalGradients {
        mean: mean.grad(&gradients).unwrap(),
        variance: variance.grad(&gradients).unwrap(),
        weight: weight.grad(&gradients).unwrap(),
        bias: bias.grad(&gradients).unwrap(),
    }
}

struct MatchedFullBackwardFixture {
    mean: Tensor<2>,
    covariance: Tensor<3>,
    weight: Tensor<2>,
    bias: Tensor<1>,
}

fn matched_full_backward_fixture(
    device: &Device,
    host: &MatchedHostFixture,
) -> MatchedFullBackwardFixture {
    let device = device.clone().autodiff();
    MatchedFullBackwardFixture {
        mean: Tensor::from_data(
            TensorData::new(host.mean.clone(), [host.batch, host.width]),
            (&device, DType::F32),
        ),
        covariance: Tensor::from_data(
            TensorData::new(
                full_diagonal_covariance(host),
                [host.batch, host.width, host.width],
            ),
            (&device, DType::F32),
        ),
        weight: Tensor::from_data(
            TensorData::new(host.weight.clone(), [host.width, host.width]),
            (&device, DType::F32),
        ),
        bias: Tensor::from_data(
            TensorData::new(host.bias.clone(), [host.width]),
            (&device, DType::F32),
        ),
    }
}

struct MatchedFullGradients {
    mean: Tensor<2>,
    covariance: Tensor<3>,
    weight: Tensor<2>,
    bias: Tensor<1>,
}

fn matched_full_backward(fixture: &MatchedFullBackwardFixture) -> MatchedFullGradients {
    let mean = fixture.mean.clone().require_grad();
    let covariance = fixture.covariance.clone().require_grad();
    let weight = fixture.weight.clone().require_grad();
    let bias = fixture.bias.clone().require_grad();
    let output = propagate_relu_full(&propagate_linear_full(
        &MomentsFull::new(mean.clone(), covariance.clone()),
        weight.clone(),
        Some(bias.clone()),
    ));
    let variance = output.variance();
    let gradients = (output.mean.sum() + variance.sum()).backward();
    MatchedFullGradients {
        mean: mean.grad(&gradients).unwrap(),
        covariance: covariance.grad(&gradients).unwrap(),
        weight: weight.grad(&gradients).unwrap(),
        bias: bias.grad(&gradients).unwrap(),
    }
}

fn warm<T>(device: &Device, work: &mut impl FnMut() -> T) {
    let warmup = work();
    device.sync().unwrap();
    black_box(warmup);
}

fn timed_once<T>(label: &str, repeat: usize, device: &Device, work: &mut impl FnMut() -> T) -> T {
    let start = Instant::now();
    let result = work();
    device.sync().unwrap();
    eprintln!("{label} repeat={repeat}: {:?}", start.elapsed());
    result
}

fn timed_interleaved<C, G>(
    label: &str,
    cpu_device: &Device,
    gpu_device: &Device,
    mut cpu_work: impl FnMut() -> C,
    mut gpu_work: impl FnMut() -> G,
) -> (C, G) {
    warm(cpu_device, &mut cpu_work);
    warm(gpu_device, &mut gpu_work);
    let mut cpu_last = None;
    let mut gpu_last = None;
    for repeat in 1..=3 {
        if repeat % 2 == 1 {
            cpu_last = Some(timed_once(
                &format!("CPU {label}"),
                repeat,
                cpu_device,
                &mut cpu_work,
            ));
            gpu_last = Some(timed_once(
                &format!("Metal {label}"),
                repeat,
                gpu_device,
                &mut gpu_work,
            ));
        } else {
            gpu_last = Some(timed_once(
                &format!("Metal {label}"),
                repeat,
                gpu_device,
                &mut gpu_work,
            ));
            cpu_last = Some(timed_once(
                &format!("CPU {label}"),
                repeat,
                cpu_device,
                &mut cpu_work,
            ));
        }
    }
    (black_box(cpu_last.unwrap()), black_box(gpu_last.unwrap()))
}

fn profile_setting(name: &str, default: &str, accepted: &[&str]) -> String {
    let setting = env::var(name).unwrap_or_else(|_| default.to_owned());
    assert!(
        accepted.contains(&setting.as_str()),
        "{name} must be one of {}; got {setting:?}",
        accepted.join(", ")
    );
    setting
}

fn profile_includes(setting: &str, value: &str) -> bool {
    setting == "both" || setting == value
}

fn assert_finite(values: &[f32], label: &str) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} is not finite"
    );
}

fn validate_values(values: &[Vec<f32>], label: &str) {
    for (index, values) in values.iter().enumerate() {
        assert_finite(values, &format!("{label} result {index}"));
    }
}

fn close_matched_values(
    gpu: &[Vec<f32>],
    cpu: &[Vec<f32>],
    label: &str,
    _host: &MatchedHostFixture,
) {
    assert_eq!(gpu.len(), cpu.len(), "{label} result count");
    for (result, (gpu, cpu)) in gpu.iter().zip(cpu).enumerate() {
        assert_eq!(gpu.len(), cpu.len(), "{label} result {result} length");
        let max_abs = cpu.iter().map(|value| value.abs()).fold(0.0, f32::max);
        assert!(
            max_abs > 0.0,
            "{label} result {result} unexpectedly has no nonzero reference"
        );
        let max_error = gpu
            .iter()
            .zip(cpu)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0, f32::max);
        eprintln!(
            "{label} result {result}: relative max error={:e}",
            max_error / max_abs
        );
        for (index, (&gpu, &cpu)) in gpu.iter().zip(cpu).enumerate() {
            // Affine gradient reductions can cancel. Scale the absolute term
            // by this tensor, so cancellation is allowed without accepting an
            // all-zero tail tensor under a fixed absolute tolerance.
            let tolerance = 1e-3 * cpu.abs() + 1e-5 * max_abs;
            assert!(
                (gpu - cpu).abs() <= tolerance,
                "{label} result {result}[{index}]: GPU={gpu:e}, CPU={cpu:e}, \
                 tolerance={tolerance:e}, scale={max_abs:e}"
            );
        }
    }
}

fn close_matched_tail_backward_values(
    gpu: &[Vec<f32>],
    cpu: &[Vec<f32>],
    label: &str,
    host: &MatchedHostFixture,
) {
    if !label.contains("tail") {
        close_matched_values(gpu, cpu, label, host);
        return;
    }

    assert_eq!(gpu.len(), 4, "{label} result count");
    assert_eq!(cpu.len(), 4, "{label} result count");
    assert_eq!(gpu[0].len(), host.batch * host.width, "{label} mean length");
    assert_eq!(cpu[0].len(), host.batch * host.width, "{label} mean length");

    let tail = TAILS
        .iter()
        .find(|tail| tail.alpha == -7.0)
        .expect("matched tail fixture has an alpha=-7 reference");
    // For L = sum_j (M_j + V_j), dL/dy_j is p + 2 sigma_j m (1-p).
    // dL/dmu_i is its signed W column sum. Its absolute-sum chain scale is
    // independent of that cancellation and uses the frozen erfc reference.
    let output_slopes: Vec<f32> = host
        .output_std
        .iter()
        .map(|&std| tail.p + 2.0 * std * tail.mean * (1.0 - tail.p))
        .collect();
    for batch in 0..host.batch {
        for input in 0..host.width {
            let index = batch * host.width + input;
            let chain_scale = (0..host.width)
                .map(|output| {
                    host.weight[input * host.width + output].abs() * output_slopes[output].abs()
                })
                .sum::<f32>();
            assert!(
                chain_scale > 0.0,
                "{label} mean[{index}] has zero chain scale"
            );
            // The direct scalar tail check above is strict to 3e-4. Apply the
            // same bound to the non-cancelling affine chain, not to the signed
            // reduction result, then retain a relative check for large values.
            let tolerance = 1e-3 * cpu[0][index].abs() + 3e-4 * chain_scale;
            assert!(
                (gpu[0][index] - cpu[0][index]).abs() <= tolerance,
                "{label} mean[{index}]: GPU={:e}, CPU={:e}, tolerance={tolerance:e}, \
                 chain_scale={chain_scale:e}",
                gpu[0][index],
                cpu[0][index],
            );
        }
    }
    close_matched_values(&gpu[1..], &cpu[1..], label, host);
}

fn diagonal_forward_values(output: Moments) -> Vec<Vec<f32>> {
    vec![data(output.mean), data(output.var)]
}

fn full_forward_values(output: MomentsFull) -> Vec<Vec<f32>> {
    vec![data(output.mean), data(output.cov)]
}

fn diagonal_backward_values(output: MatchedDiagonalGradients) -> Vec<Vec<f32>> {
    vec![
        data(output.mean),
        data(output.variance),
        data(output.weight),
        data(output.bias),
    ]
}

fn full_backward_values(output: MatchedFullGradients) -> Vec<Vec<f32>> {
    vec![
        data(output.mean),
        data(output.covariance),
        data(output.weight),
        data(output.bias),
    ]
}

macro_rules! profile_matched_case {
    ($backend:expr, $label:expr, $host:expr, $fixture:ident, $work:ident, $values:ident, $compare:ident) => {{
        match $backend {
            "cpu" => {
                let device = Device::flex();
                let fixture = $fixture(&device, $host);
                let values = ($values)(timed_repeated($label, &device, || $work(&fixture)));
                validate_values(&values, $label);
            }
            "metal" => {
                let device = metal_device();
                let fixture = $fixture(&device, $host);
                let values = ($values)(timed_repeated($label, &device, || $work(&fixture)));
                validate_values(&values, $label);
            }
            "both" => {
                let cpu_device = Device::flex();
                let gpu_device = metal_device();
                let cpu_fixture = $fixture(&cpu_device, $host);
                let gpu_fixture = $fixture(&gpu_device, $host);
                let (cpu, gpu) = timed_interleaved(
                    $label,
                    &cpu_device,
                    &gpu_device,
                    || $work(&cpu_fixture),
                    || $work(&gpu_fixture),
                );
                let cpu = ($values)(cpu);
                let gpu = ($values)(gpu);
                validate_values(&cpu, &format!("CPU {}", $label));
                validate_values(&gpu, &format!("Metal {}", $label));
                $compare(&gpu, &cpu, $label, $host);
            }
            _ => unreachable!("validated profile backend"),
        }
    }};
}

/// Matched CPU/Metal timing with host uploads and validation outside the timer.
///
/// The default checks both backends, representations, passes, central/tail
/// regimes, and 8x16/64x64 shapes. For process-isolated peak-RSS measurement,
/// choose one backend; that path never constructs the other backend or device:
///
/// ```text
/// STABLEPROP_PROFILE_BACKEND=cpu|metal|both       (default both)
/// STABLEPROP_PROFILE_SHAPE=8x16|64x64|both       (default both)
/// STABLEPROP_PROFILE_PASS=forward|backward|both   (default both)
/// STABLEPROP_PROFILE_REPRESENTATION=diagonal|full|both (default both)
/// STABLEPROP_PROFILE_REGIME=central|tail|both     (default both)
/// ```
#[test]
#[ignore = "requires a Metal GPU and is an informational matched timing benchmark"]
fn metal_matched_diagonal_full_forward_backward_timings() {
    let backend = profile_setting(
        "STABLEPROP_PROFILE_BACKEND",
        "both",
        &["cpu", "metal", "both"],
    );
    let shape = profile_setting(
        "STABLEPROP_PROFILE_SHAPE",
        "both",
        &["8x16", "64x64", "both"],
    );
    let pass = profile_setting(
        "STABLEPROP_PROFILE_PASS",
        "both",
        &["forward", "backward", "both"],
    );
    let representation = profile_setting(
        "STABLEPROP_PROFILE_REPRESENTATION",
        "both",
        &["diagonal", "full", "both"],
    );
    let regime = profile_setting(
        "STABLEPROP_PROFILE_REGIME",
        "both",
        &["central", "tail", "both"],
    );

    eprintln!(
        "Matched f32 timings: host uploads and readback validation are outside the timer; \
         each timed work creates fresh autodiff leaves for backward. backend={backend}, \
         shape={shape}, pass={pass}, representation={representation}, regime={regime}"
    );
    for (shape_name, batch, width) in [("8x16", 8, 16), ("64x64", 64, 64)] {
        if !profile_includes(&shape, shape_name) {
            continue;
        }
        for regime_name in ["central", "tail"] {
            if !profile_includes(&regime, regime_name) {
                continue;
            }
            let host = matched_host_fixture(batch, width, regime_name);
            for representation_name in ["diagonal", "full"] {
                if !profile_includes(&representation, representation_name) {
                    continue;
                }
                let label = format!("{representation_name} {regime_name} b={batch} w={width}");
                if profile_includes(&pass, "forward") {
                    let timing_label = format!("{label} forward");
                    match representation_name {
                        "diagonal" => profile_matched_case!(
                            backend.as_str(),
                            &timing_label,
                            &host,
                            matched_diagonal_forward_fixture,
                            matched_diagonal_forward,
                            diagonal_forward_values,
                            close_matched_values
                        ),
                        "full" => profile_matched_case!(
                            backend.as_str(),
                            &timing_label,
                            &host,
                            matched_full_forward_fixture,
                            matched_full_forward,
                            full_forward_values,
                            close_matched_values
                        ),
                        _ => unreachable!("validated representation"),
                    }
                }
                if profile_includes(&pass, "backward") {
                    let timing_label = format!("{label} backward");
                    match representation_name {
                        "diagonal" => profile_matched_case!(
                            backend.as_str(),
                            &timing_label,
                            &host,
                            matched_diagonal_backward_fixture,
                            matched_diagonal_backward,
                            diagonal_backward_values,
                            close_matched_tail_backward_values
                        ),
                        "full" => profile_matched_case!(
                            backend.as_str(),
                            &timing_label,
                            &host,
                            matched_full_backward_fixture,
                            matched_full_backward,
                            full_backward_values,
                            close_matched_tail_backward_values
                        ),
                        _ => unreachable!("validated representation"),
                    }
                }
            }
        }
    }
}
