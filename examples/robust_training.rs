//! Train with a propagated-variance penalty under input noise.
//!
//! Because stableprop's propagation is differentiable, the analytic output
//! variance under input noise can go straight into the loss. Penalizing it
//! `loss = MSE + lambda * mean(output_variance)` changes the training objective.
//! This controlled comparison includes clean MSE, Gaussian input augmentation,
//! and the variance penalty. The latter is not generally expected noisy squared
//! loss: its point-prediction term is evaluated at the clean input.
//!
//! All nets start from the same weights and use the same noisy test draws, so
//! only the loss differs. Input perturbations retain the clean target: this
//! measures label-preserving measurement noise, not target or label noise.
//! Augmentation uses `TRAIN_STD`; test RMSE is reported at both that matched
//! noise level and a separately stated shifted level.
//! Each backend has its own RNG stream, so metrics from separate backend runs
//! need not match.
//!
//! Run on CPU: `cargo run --release --example robust_training --features burn`
//! Run on macOS Metal: `cargo run --release --example robust_training --features metal -- --metal`

use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{activation, Device, Distribution, Tensor, TensorData};
use std::time::Instant;

use stableprop::burn_sdp::{propagate_linear, propagate_relu, Moments};

const D_IN: usize = 6;
const HIDDEN: usize = 64;
const N_TRAIN: usize = 3000;
const N_TEST: usize = 1000;
const TRAIN_STEPS: usize = 800;
const TRAIN_STD: f64 = 0.2;
const TEST_STD: f64 = 0.3;
const EVALUATION_DRAWS: usize = 20;

const DATA_SEED: u64 = 0xA0B5_7001;
const INIT_SEED: u64 = 0xA0B5_7002;
const AUGMENTATION_SEED: u64 = 0xA0B5_7003;
const EVALUATION_SEED: u64 = 0xA0B5_7004;

#[derive(Module, Debug)]
struct Mlp {
    lin1: Linear,
    lin2: Linear,
}

impl Mlp {
    fn init(device: &Device) -> Self {
        let model = Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, 1).init(device),
        };
        // Burn parameters initialize lazily; materialize before cloning a baseline.
        for layer in [&model.lin1, &model.lin2] {
            drop(layer.weight.val());
            if let Some(bias) = &layer.bias {
                drop(bias.val());
            }
        }
        model
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.lin2.forward(activation::relu(self.lin1.forward(x)))
    }
    /// Point prediction and propagated variance under input noise `std`.
    /// The first output is not the mean over perturbed inputs.
    fn forward_with_var(&self, x: Tensor<2>, std: f64) -> (Tensor<2>, Tensor<2>) {
        let [n, d] = x.dims();
        let var0 = Tensor::<2>::full([n, d], std * std, &x.device());
        let w1 = self.lin1.weight.val();
        let b1 = self.lin1.bias.as_ref().map(|p| p.val());
        let w2 = self.lin2.weight.val();
        let b2 = self.lin2.bias.as_ref().map(|p| p.val());
        let m1 = propagate_relu(&propagate_linear(&Moments::new(x.clone(), var0), w1, b1));
        let m2 = propagate_linear(&m1, w2, b2);
        (self.forward(x), m2.var)
    }
}

fn target(x: &[f32]) -> f32 {
    let s: f32 = x.iter().sum();
    (s * 0.6).sin() + 0.5 * x[0] * x[1] - 0.3 * x[2] * x[2]
}

fn mse(model: &Mlp, inputs: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
    MseLoss::new().forward(model.forward(inputs), targets, Reduction::Mean)
}

fn augmented_mse(
    model: &Mlp,
    clean_inputs: Tensor<2>,
    targets: Tensor<2>,
    noise: Tensor<2>,
) -> Tensor<1> {
    mse(model, clean_inputs + noise, targets)
}

#[derive(Clone, Copy)]
enum Objective {
    CleanMse,
    GaussianAugmentation,
    VariancePenalty,
}

fn train(
    mut model: Mlp,
    objective: Objective,
    inputs: &Tensor<2>,
    targets: &Tensor<2>,
    device: &Device,
) -> Mlp {
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..TRAIN_STEPS {
        let loss = match objective {
            Objective::CleanMse => mse(&model, inputs.clone(), targets.clone()),
            Objective::GaussianAugmentation => {
                let noise = Tensor::<2>::random(
                    inputs.dims(),
                    Distribution::Normal(0.0, TRAIN_STD),
                    device,
                );
                augmented_mse(&model, inputs.clone(), targets.clone(), noise)
            }
            Objective::VariancePenalty => {
                let (prediction, variance) = model.forward_with_var(inputs.clone(), TRAIN_STD);
                MseLoss::new().forward(prediction, targets.clone(), Reduction::Mean)
                    + variance.mean().mul_scalar(3.0)
            }
        };
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(1e-3, model, gradients);
    }
    model
}

fn run(dev: &Device, backend: &str) {
    dev.seed(DATA_SEED);
    let make = |n: usize| -> (Tensor<2>, Tensor<2>) {
        let xt = Tensor::<2>::random([n, D_IN], Distribution::Normal(0.0, 1.0), dev);
        let xv = xt.to_data().try_to_vec::<f32>().unwrap();
        let yv: Vec<f32> = (0..n)
            .map(|i| target(&xv[i * D_IN..(i + 1) * D_IN]))
            .collect();
        (xt, Tensor::from_data(TensorData::new(yv, [n, 1]), dev))
    };
    let (x_tr, y_tr) = make(N_TRAIN);
    let (x_te, y_te) = make(N_TEST);

    // Materialization in `Mlp::init` makes every clone a shared baseline.
    dev.seed(INIT_SEED);
    let init = Mlp::init(dev);
    dev.sync().expect("training backend synchronization failed");
    let started = Instant::now();
    let plain = train(init.clone(), Objective::CleanMse, &x_tr, &y_tr, dev);
    dev.seed(AUGMENTATION_SEED);
    let augmented = train(
        init.clone(),
        Objective::GaussianAugmentation,
        &x_tr,
        &y_tr,
        dev,
    );
    let robust = train(init, Objective::VariancePenalty, &x_tr, &y_tr, dev);
    dev.sync().expect("training backend synchronization failed");
    let elapsed = started.elapsed();

    // Evaluation noise does not depend on any random draws used while training.
    dev.seed(EVALUATION_SEED);
    let clean = vec![x_te.clone()];
    let noisy_inputs = |std| {
        (0..EVALUATION_DRAWS)
            .map(|_| {
                x_te.clone()
                    + Tensor::<2>::random([N_TEST, D_IN], Distribution::Normal(0.0, std), dev)
            })
            .collect::<Vec<_>>()
    };
    let matched_noise = noisy_inputs(TRAIN_STD);
    let shifted_noise = noisy_inputs(TEST_STD);
    let y = y_te.into_data().try_to_vec::<f32>().unwrap();
    let rmse = |model: &Mlp, inputs: &[Tensor<2>]| -> f64 {
        let mut total = 0.0;
        for x in inputs {
            let p = model
                .forward(x.clone())
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            total += (0..N_TEST)
                .map(|i| (p[i] - y[i]).powi(2) as f64)
                .sum::<f64>()
                / N_TEST as f64;
        }
        (total / inputs.len() as f64).sqrt()
    };

    println!("Backend: {backend}");
    println!(
        "Training elapsed for all three objectives (includes first-use kernel compilation/autotuning): {:.3}s",
        elapsed.as_secs_f64()
    );
    println!("RMSE (lower = better); shared initialization and shared evaluation draws:");
    println!(
        "  {:<28} {:>8} {:>14} {:>14}",
        "net",
        "clean",
        format!("noise std={TRAIN_STD}"),
        format!("noise std={TEST_STD}")
    );
    println!(
        "  {:<28} {:>8.4} {:>14.4} {:>14.4}",
        "plain MSE",
        rmse(&plain, &clean),
        rmse(&plain, &matched_noise),
        rmse(&plain, &shifted_noise)
    );
    println!(
        "  {:<28} {:>8.4} {:>14.4} {:>14.4}",
        "Gaussian augmentation",
        rmse(&augmented, &clean),
        rmse(&augmented, &matched_noise),
        rmse(&augmented, &shifted_noise)
    );
    println!(
        "  {:<28} {:>8.4} {:>14.4} {:>14.4}",
        "MSE + variance penalty",
        rmse(&robust, &clean),
        rmse(&robust, &matched_noise),
        rmse(&robust, &shifted_noise)
    );
    println!("\nNoisy inputs retain clean targets: this is label-preserving measurement noise.");
    println!("Augmentation and the penalty use train noise std={TRAIN_STD}; std={TEST_STD} is a shifted test condition.");
    println!("Compare all three columns; neither training objective establishes a general robustness result.");
    println!("Different backend RNG streams can produce different trained metrics.");
}

fn run_cpu() {
    run(&Device::flex().autodiff(), "CPU");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_metal() {
    use burn::backend::wgpu::{
        graphics::{GraphicsApi, Metal},
        init_setup, RuntimeOptions, WgpuDevice,
    };

    let setup = init_setup::<Metal>(&WgpuDevice::DefaultDevice, RuntimeOptions::default());
    assert_eq!(setup.backend, Metal::backend());
    println!("Device: {}", setup.adapter.get_info().name);
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice).autodiff();
    run(&device, "Metal");
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_metal() -> Result<(), String> {
    #[cfg(not(feature = "metal"))]
    return Err(
        "--metal requires the `metal` feature on macOS; run with `--features metal`".into(),
    );

    #[cfg(all(feature = "metal", not(target_os = "macos")))]
    Err("--metal is supported only on macOS".into())
}

fn main() -> Result<(), String> {
    let mut metal = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--metal" => metal = true,
            _ => {
                return Err(format!(
                    "unknown argument `{arg}`; usage: robust_training [--metal]"
                ))
            }
        }
    }

    if metal {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        run_metal();
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        run_metal()?;
    } else {
        run_cpu();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_baseline_has_identical_forward_predictions() {
        let device = Device::flex().autodiff();
        device.seed(0xA0B5_7E57);
        let baseline = Mlp::init(&device);
        let left = baseline.clone();
        let right = baseline.clone();
        let probe = Tensor::<2>::from_data(
            TensorData::new(
                vec![
                    -1.0, -0.5, 0.25, 0.75, 1.25, 1.5, 0.4, -0.8, 1.2, -1.6, 2.0, -2.4,
                ],
                [2, D_IN],
            ),
            &device,
        );

        let left = left
            .forward(probe.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let right = right
            .forward(probe)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(
            left, right,
            "cloned baselines must share initialized weights"
        );
    }

    #[test]
    fn augmentation_matches_affine_noisy_risk_and_weight_gradient() {
        use burn::module::Param;
        use burn::tensor::DType;

        let device = Device::flex().autodiff();
        let options = (&device, DType::F32);
        // Positive inputs keep ReLU linear: f(x) = w*x, with w = 2.
        let weight = Tensor::<2>::from_data([[2.0]], options).require_grad();
        let model = Mlp {
            lin1: Linear {
                weight: Param::from_tensor(Tensor::from_data([[1.0]], options)),
                bias: None,
            },
            lin2: Linear {
                weight: Param::from_tensor(weight.clone()),
                bias: None,
            },
        };
        let inputs = Tensor::<2>::from_data([[2.0], [2.0]], options);
        let targets = Tensor::<2>::from_data([[3.0], [3.0]], options);
        // Antithetic draws integrate this quadratic loss exactly for a
        // zero-mean noise law with variance 1/16, including a Gaussian.
        let noise = Tensor::<2>::from_data([[0.25], [-0.25]], options);
        let loss = augmented_mse(&model, inputs, targets, noise);
        let value = loss.clone().into_scalar::<f32>();
        let gradients = loss.backward();
        let derivative = weight.grad(&gradients).unwrap().into_scalar::<f32>();
        // Risk = (2*w - 3)^2 + w^2/16; derivative at w=2 is 4.25.
        assert!((value - 1.25).abs() < 1e-6, "risk = {value}");
        assert!((derivative - 4.25).abs() < 1e-6, "gradient = {derivative}");
    }
}
