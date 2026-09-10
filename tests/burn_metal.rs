#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used)]

//! Local Apple-GPU parity and timing checks. They are deliberately ignored:
//! the runtime checks require an Apple GPU, while CI only compiles this harness.

use std::{hint::black_box, sync::OnceLock, time::Instant};

use burn::{
    backend::wgpu::{
        graphics::{GraphicsApi, Metal as MetalApi},
        init_setup, CubeBackend, RuntimeOptions, WgpuDevice, WgpuRuntime,
    },
    backend::{Autodiff, Metal as MetalBackend, Wgpu},
    tensor::{backend::Backend, Tensor},
};
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_conv2d, propagate_leaky_relu, propagate_linear_bayes, propagate_linear_cauchy,
    propagate_linear_cross_covariance, propagate_linear_full, propagate_matmul_left,
    propagate_relu, propagate_relu_cauchy, propagate_relu_cross_covariance, propagate_relu_full,
    propagate_residual_add_correlated, Cauchy, Moments, MomentsFull,
};

type Cpu = NdArray<f32>;
type GpuDefault = Wgpu<f32>;
type GpuMsl = MetalBackend<f32>;
type GpuEager = CubeBackend<WgpuRuntime, f32, i32, u8>;

fn metal_device() -> WgpuDevice {
    static DEVICE: OnceLock<WgpuDevice> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let device = WgpuDevice::DefaultDevice;
            let setup = init_setup::<MetalApi>(&device, RuntimeOptions::default());
            assert_eq!(
                setup.backend,
                MetalApi::backend(),
                "Metal setup must not fall back to CPU"
            );
            eprintln!(
                "Metal adapter={:?}; backend={:?}; Wgpu default runtime={}; Metal alias runtime={}",
                setup.adapter.get_info(),
                setup.backend,
                GpuDefault::name(&device),
                GpuMsl::name(&device),
            );
            device
        })
        .clone()
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

fn data<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
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

fn tail_values<B: Backend>(device: &B::Device, alpha: f32, variance: f32) -> [f32; 5] {
    let mean = alpha * variance.sqrt();
    let moments = Moments::new(
        Tensor::<B, 2>::from_data([[mean]], device),
        Tensor::<B, 2>::from_data([[variance]], device),
    );
    let diagonal = propagate_relu(&moments);
    let full = propagate_relu_full(&MomentsFull::from_diagonal(
        moments.mean.clone(),
        moments.var.clone(),
    ));
    let full_var = data(full.variance())[0];
    let full_mean = data(full.mean)[0];
    let cross = propagate_relu_cross_covariance(
        Tensor::<B, 3>::from_data([[[variance]]], device),
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

fn tail_gradients<B: Backend>(
    device: &B::Device,
    alpha: f32,
    variance: f32,
    mean_loss: bool,
) -> (f32, f32) {
    let mean =
        Tensor::<Autodiff<B>, 2>::from_data([[alpha * variance.sqrt()]], device).require_grad();
    let var = Tensor::<Autodiff<B>, 2>::from_data([[variance]], device).require_grad();
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

fn assert_tail_values<B: Backend>(device: &B::Device, label: &str) {
    for tail in TAILS {
        for variance in [2f32.powi(-40), 1.0, 2f32.powi(40)] {
            let sigma = variance.sqrt();
            let actual = tail_values::<B>(device, tail.alpha, variance);
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

fn assert_tail_gradients<B: Backend>(device: &B::Device, label: &str) {
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
                let actual = tail_gradients::<B>(device, tail.alpha, variance, mean_loss);
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
    assert_tail_values::<GpuMsl>(&device, "Metal fused");
    assert_tail_values::<GpuEager>(&device, "Metal eager");
    assert_tail_gradients::<GpuMsl>(&device, "Metal fused");
    <Autodiff<GpuMsl> as Backend>::sync(&device).unwrap();
    assert_tail_gradients::<GpuEager>(&device, "Metal eager");
    <Autodiff<GpuEager> as Backend>::sync(&device).unwrap();
}

fn assert_distant_tail_gradients<B: Backend>(device: &B::Device) {
    // All inputs are normal f32 values. The old division backward still
    // overflows: mean / variance is 1e50 even though the selected slope is 0 or 1.
    for mode in ["relu", "leaky", "full", "cross"] {
        let mean = Tensor::<Autodiff<B>, 2>::from_data([[1e20, -1e20]], device).require_grad();
        let var = Tensor::<Autodiff<B>, 2>::from_data([[1e-30; 2]], device).require_grad();
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
                let cross = Tensor::<Autodiff<B>, 3>::from_data([[[5e-31; 2]]], device);
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
    assert_distant_tail_gradients::<GpuMsl>(&device);
    assert_distant_tail_gradients::<GpuEager>(&device);
}

fn assert_mixed_scale_covariance_gradients<B: Backend>(device: &B::Device) {
    let v0 = 1e-30f32;
    let v1 = 1e30f32;
    let cross = 0.25f32;
    let cov =
        Tensor::<Autodiff<B>, 3>::from_data([[[v0, cross], [cross, v1]]], device).require_grad();
    let out = propagate_relu_full(&MomentsFull::new(
        Tensor::<Autodiff<B>, 2>::zeros([1, 2], device),
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
    assert_mixed_scale_covariance_gradients::<GpuMsl>(&device);
    assert_mixed_scale_covariance_gradients::<GpuEager>(&device);
}

fn fixture<B: Backend>(device: &B::Device) -> Vec<Vec<f32>> {
    // Includes zero variance, a central input, tiny scale, and linear tails.
    let mean = Tensor::<B, 2>::from_data([[0.0, 1e-12, 9.0, -9.0]], device);
    let var = Tensor::<B, 2>::from_data([[0.0, 1e-24, 0.49, 1.0]], device);
    let moments = Moments::new(mean.clone(), var.clone());
    let leaky = propagate_leaky_relu(&moments, 0.1);
    let weight = Tensor::<B, 2>::from_data(
        [
            [0.5, -1.0, 0.25],
            [1.0, 0.5, -0.5],
            [-0.25, 0.75, 1.0],
            [0.5, -0.25, 0.5],
        ],
        device,
    );
    let w_var = Tensor::<B, 2>::from_data(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.01, 0.02],
            [0.02, 0.03, 0.01],
            [0.01, 0.02, 0.04],
        ],
        device,
    );
    let bayes = propagate_linear_bayes(
        &moments,
        weight.clone(),
        w_var,
        Some((
            Tensor::from_data([0.1, -0.2, 0.3], device),
            Tensor::from_data([0.01, 0.02, 0.03], device),
        )),
    );
    let left = propagate_matmul_left(Tensor::from_data([[1.0], [-0.25]], device), &bayes);
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
            device,
        ),
    );
    let full = propagate_relu_full(&propagate_linear_full(&full, weight, None));
    let cross = propagate_relu_cross_covariance(
        propagate_linear_cross_covariance(
            // The deterministic first margin has a zero covariance column.
            Tensor::from_data([[[0.0, 1e-24, 0.2, -0.2]]], device),
            Tensor::from_data(
                [
                    [0.5, -1.0, 0.25],
                    [1.0, 0.5, -0.5],
                    [-0.25, 0.75, 1.0],
                    [0.5, -0.25, 0.5],
                ],
                device,
            ),
        ),
        &bayes,
    );
    let (conv_mean, conv_var) = propagate_conv2d(
        Tensor::<B, 4>::from_data([[[[0.0, 1.0], [2.0, -1.0]]]], device),
        Tensor::<B, 4>::from_data([[[[1e-24, 0.25], [0.5, 1.0]]]], device),
        Tensor::<B, 4>::from_data([[[[0.5, -1.0], [0.25, 0.75]]]], device),
        Some(Tensor::<B, 1>::from_data([0.1], device)),
        burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );
    let cauchy = propagate_relu_cauchy(&propagate_linear_cauchy(
        &Cauchy::new(mean, var.sqrt()),
        Tensor::from_data(
            [[1.0, -0.5], [-0.25, 0.75], [0.5, 1.0], [-1.0, 0.25]],
            device,
        ),
        Some(Tensor::from_data([0.2, -0.1], device)),
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

fn gradient_fixture<B: Backend>(
    device: &B::Device,
    contribution: GradientContribution,
) -> Vec<Option<Vec<f32>>> {
    type Ad<B> = Autodiff<B>;
    // CPU/GPU gradient parity stays off exact kinks; tiny correlated boundary
    // gradients have their own finite-value check below.
    let mean = Tensor::<Ad<B>, 2>::from_data([[0.1, 1e-4, 9.0, -9.0]], device).require_grad();
    let var = Tensor::<Ad<B>, 2>::from_data([[1e-6, 1e-8, 0.49, 1.0]], device).require_grad();
    let cross = Tensor::<Ad<B>, 3>::from_data([[[1e-5, 1e-6, 0.2, -0.2]]], device).require_grad();
    let moments = Moments::new(mean.clone(), var.clone());
    let leaky = propagate_leaky_relu(&moments, 0.1);
    let weight = Tensor::<Ad<B>, 2>::from_data(
        [[0.5, -1.0], [1.0, 0.5], [-0.25, 0.75], [0.5, -0.25]],
        device,
    )
    .require_grad();
    let bayes = propagate_linear_bayes(
        &moments,
        weight.clone(),
        weight.clone() * weight.clone().mul_scalar(0.02),
        None,
    );
    let left = propagate_matmul_left(Tensor::from_data([[1.0], [0.5]], device), &bayes);
    let residual =
        propagate_residual_add_correlated(&left, &left, left.var.clone().mul_scalar(0.2));
    let full_cov = Tensor::<Ad<B>, 3>::from_data(
        [[
            [1e-6, 5e-8, 0.0, 0.0],
            [5e-8, 1e-8, 0.0, 0.0],
            [0.0, 0.0, 0.49, 0.21],
            [0.0, 0.0, 0.21, 1.0],
        ]],
        device,
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
        Tensor::from_data([[[[0.5, -1.0], [0.25, 0.75]]]], device),
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
fn metal_wgpu_and_msl_match_ndarray_forward_and_autodiff() {
    let cpu_device = Default::default();
    let gpu_device = metal_device();
    let expected = fixture::<Cpu>(&cpu_device);
    for (label, actual) in [
        ("Wgpu default", fixture::<GpuDefault>(&gpu_device)),
        ("Metal alias", fixture::<GpuMsl>(&gpu_device)),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            close(
                actual,
                expected,
                &format!("{label} forward fixture {index}"),
            );
        }
    }
    // Absolute tolerance alone would accept losing the 1e-24 variance.
    let gpu_tiny = fixture::<GpuMsl>(&gpu_device)[1][1] / 1e-24;
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
        let expected_gradients = gradient_fixture::<Cpu>(&cpu_device, contribution);
        for (label, actual) in [
            (
                "Metal alias",
                gradient_fixture::<GpuMsl>(&gpu_device, contribution),
            ),
            (
                "Wgpu default",
                gradient_fixture::<GpuDefault>(&gpu_device, contribution),
            ),
        ] {
            <Autodiff<GpuDefault> as Backend>::sync(&gpu_device).unwrap();
            for (index, (actual, expected)) in actual.iter().zip(&expected_gradients).enumerate() {
                match (actual, expected) {
                    (Some(actual), Some(expected)) => close(
                        actual,
                        expected,
                        &format!("{label} {contribution:?} gradient {index}"),
                    ),
                    (None, None) => {}
                    _ => panic!("{label} {contribution:?} gradient presence differs at {index}"),
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn eager_metal_accumulates_shared_branch_gradients() {
    let cpu_device = Default::default();
    let gpu_device = metal_device();
    let contribution = GradientContribution::ResidualCross;
    let expected = gradient_fixture::<Cpu>(&cpu_device, contribution);
    let actual = gradient_fixture::<GpuEager>(&gpu_device, contribution);
    <Autodiff<GpuEager> as Backend>::sync(&gpu_device).unwrap();
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        match (actual, expected) {
            (Some(actual), Some(expected)) => close(
                actual,
                expected,
                &format!("eager {contribution:?} gradient {index}"),
            ),
            (None, None) => {}
            _ => panic!("eager {contribution:?} gradient presence differs at {index}"),
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn eager_metal_matches_ndarray_forward_and_autodiff() {
    let cpu_device = Default::default();
    let gpu_device = metal_device();
    for (index, (actual, expected)) in fixture::<GpuEager>(&gpu_device)
        .iter()
        .zip(fixture::<Cpu>(&cpu_device))
        .enumerate()
    {
        close(actual, &expected, &format!("eager forward fixture {index}"));
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
        let expected = gradient_fixture::<Cpu>(&cpu_device, contribution);
        let actual = gradient_fixture::<GpuEager>(&gpu_device, contribution);
        <Autodiff<GpuEager> as Backend>::sync(&gpu_device).unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            match (actual, expected) {
                (Some(actual), Some(expected)) => close(
                    actual,
                    expected,
                    &format!("eager {contribution:?} gradient {index}"),
                ),
                (None, None) => {}
                _ => panic!("eager {contribution:?} gradient presence differs at {index}"),
            }
        }
    }
}

fn direct_conv_gradients<B: Backend>(device: &B::Device) -> (Vec<f32>, Vec<f32>) {
    type Ad<B> = Autodiff<B>;
    let input = Tensor::<Ad<B>, 4>::from_data([[[[0.0, 1.0], [2.0, -1.0]]]], device).require_grad();
    let weight =
        Tensor::<Ad<B>, 4>::from_data([[[[0.5, -1.0], [0.25, 0.75]]]], device).require_grad();
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

fn shared_weight_conv_gradients<B: Backend>(device: &B::Device) -> (Vec<f32>, Vec<f32>) {
    type Ad<B> = Autodiff<B>;
    let input = Tensor::<Ad<B>, 4>::from_data([[[[0.0, 1.0], [2.0, -1.0]]]], device).require_grad();
    let weight =
        Tensor::<Ad<B>, 4>::from_data([[[[0.5, -1.0], [0.25, 0.75]]]], device).require_grad();
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
fn eager_metal_direct_conv2d_gradients_match_ndarray() {
    let cpu_device = Default::default();
    let gpu_device = metal_device();
    let (cpu_input, cpu_weight) = direct_conv_gradients::<Cpu>(&cpu_device);
    let (gpu_input, gpu_weight) = direct_conv_gradients::<GpuEager>(&gpu_device);
    <Autodiff<GpuEager> as Backend>::sync(&gpu_device).unwrap();
    close(&gpu_input, &cpu_input, "eager direct conv input gradient");
    close(
        &gpu_weight,
        &cpu_weight,
        "eager direct conv weight gradient",
    );
    close(
        &gpu_input,
        &[0.5, -1.0, 0.25, 0.75],
        "eager direct conv input analytic gradient",
    );
    close(
        &gpu_weight,
        &[0.0, 1.0, 2.0, -1.0],
        "eager direct conv weight analytic gradient",
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn fused_metal_direct_conv2d_gradients_match_ndarray() {
    let cpu_device = Default::default();
    let gpu_device = metal_device();
    let (cpu_input, cpu_weight) = direct_conv_gradients::<Cpu>(&cpu_device);
    let (gpu_input, gpu_weight) = direct_conv_gradients::<GpuMsl>(&gpu_device);
    <Autodiff<GpuMsl> as Backend>::sync(&gpu_device).unwrap();
    close(&gpu_input, &cpu_input, "fused direct conv input gradient");
    close(
        &gpu_weight,
        &cpu_weight,
        "fused direct conv weight gradient",
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn fused_metal_shared_weight_conv2d_gradients_match_ndarray() {
    let cpu_device = Default::default();
    let gpu_device = metal_device();
    let (cpu_input, cpu_weight) = shared_weight_conv_gradients::<Cpu>(&cpu_device);
    let (gpu_input, gpu_weight) = shared_weight_conv_gradients::<GpuMsl>(&gpu_device);
    <Autodiff<GpuMsl> as Backend>::sync(&gpu_device).unwrap();
    close(&gpu_input, &cpu_input, "fused shared conv input gradient");
    close(
        &gpu_weight,
        &cpu_weight,
        "fused shared conv weight gradient",
    );
}

fn diagonal_work<B: Backend>(device: &B::Device, batch: usize, width: usize) -> Moments<B> {
    let mean = Tensor::<B, 2>::full([batch, width], 0.2, device);
    let var = Tensor::<B, 2>::full([batch, width], 0.3, device);
    let weight = Tensor::<B, 2>::full([width, width], 0.01, device);
    propagate_relu(&propagate_linear_bayes(
        &Moments::new(mean, var),
        weight.clone(),
        weight.mul_scalar(0.1),
        None,
    ))
}

fn full_work<B: Backend>(device: &B::Device, batch: usize, width: usize) -> MomentsFull<B> {
    let mean = Tensor::<B, 2>::full([batch, width], 0.2, device);
    let var = Tensor::<B, 2>::full([batch, width], 0.3, device);
    let weight = Tensor::<B, 2>::full([width, width], 0.01, device);
    propagate_relu_full(&propagate_linear_full(
        &MomentsFull::from_diagonal(mean, var),
        weight,
        None,
    ))
}

fn diagonal_training_work<B: Backend>(
    device: &B::Device,
    batch: usize,
    width: usize,
) -> Tensor<B, 2> {
    type Ad<B> = Autodiff<B>;
    let mean = Tensor::<Ad<B>, 2>::full([batch, width], 0.2, device).require_grad();
    let var = Tensor::<Ad<B>, 2>::full([batch, width], 0.3, device).require_grad();
    let weight = Tensor::<Ad<B>, 2>::full([width, width], 0.01, device).require_grad();
    let output = propagate_relu(&propagate_linear_bayes(
        &Moments::new(mean.clone(), var),
        weight.clone(),
        weight.mul_scalar(0.1),
        None,
    ));
    mean.grad(&output.mean.sum().backward()).unwrap()
}

fn timed<B: Backend, T>(label: &str, device: &B::Device, mut work: impl FnMut() -> T) -> T {
    let warmup = work();
    B::sync(device).unwrap();
    black_box(warmup);
    let start = Instant::now();
    let result = work();
    B::sync(device).unwrap();
    let elapsed = start.elapsed();
    eprintln!("{label}: {elapsed:?}");
    black_box(result)
}

#[test]
#[ignore = "requires a Metal GPU and is an informational local benchmark"]
fn metal_synchronized_diagonal_and_full_timings() {
    let cpu_device = Default::default();
    let gpu_device = metal_device();
    eprintln!("Warmed f32 workloads; allocation included, host validation excluded.");
    for &(batch, width, iterations) in &[(8, 16, 32), (64, 64, 8), (256, 256, 1)] {
        let cpu = timed::<Cpu, _>(
            &format!("CPU diagonal b={batch} w={width} n={iterations}"),
            &cpu_device,
            || {
                let mut last = diagonal_work::<Cpu>(&cpu_device, batch, width);
                for _ in 1..iterations {
                    last = diagonal_work::<Cpu>(&cpu_device, batch, width);
                }
                last
            },
        );
        let gpu = timed::<GpuMsl, _>(
            &format!("Metal diagonal b={batch} w={width} n={iterations}"),
            &gpu_device,
            || {
                let mut last = diagonal_work::<GpuMsl>(&gpu_device, batch, width);
                for _ in 1..iterations {
                    last = diagonal_work::<GpuMsl>(&gpu_device, batch, width);
                }
                last
            },
        );
        close(&data(gpu.mean), &data(cpu.mean), "timed diagonal mean");
        close(&data(gpu.var), &data(cpu.var), "timed diagonal variance");
        if width <= 64 {
            let cpu = timed::<Cpu, _>(
                &format!("CPU full b={batch} w={width} n={iterations}"),
                &cpu_device,
                || {
                    let mut last = full_work::<Cpu>(&cpu_device, batch, width);
                    for _ in 1..iterations {
                        last = full_work::<Cpu>(&cpu_device, batch, width);
                    }
                    last
                },
            );
            let gpu = timed::<GpuMsl, _>(
                &format!("Metal full b={batch} w={width} n={iterations}"),
                &gpu_device,
                || {
                    let mut last = full_work::<GpuMsl>(&gpu_device, batch, width);
                    for _ in 1..iterations {
                        last = full_work::<GpuMsl>(&gpu_device, batch, width);
                    }
                    last
                },
            );
            close(&data(gpu.mean), &data(cpu.mean), "timed full mean");
            close(&data(gpu.cov), &data(cpu.cov), "timed full covariance");
        }
        let cpu = timed::<Cpu, _>(
            &format!("CPU diagonal training b={batch} w={width} n={iterations}"),
            &cpu_device,
            || {
                let mut last = diagonal_training_work::<Cpu>(&cpu_device, batch, width);
                for _ in 1..iterations {
                    last = diagonal_training_work::<Cpu>(&cpu_device, batch, width);
                }
                last
            },
        );
        let gpu = timed::<GpuMsl, _>(
            &format!("Metal diagonal training b={batch} w={width} n={iterations}"),
            &gpu_device,
            || {
                let mut last = diagonal_training_work::<GpuMsl>(&gpu_device, batch, width);
                for _ in 1..iterations {
                    last = diagonal_training_work::<GpuMsl>(&gpu_device, batch, width);
                }
                last
            },
        );
        close(&data(gpu), &data(cpu), "timed input gradient");
    }
}
