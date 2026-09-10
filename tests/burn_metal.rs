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
