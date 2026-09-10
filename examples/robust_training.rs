//! Train with a propagated-variance penalty under input noise.
//!
//! Because stableprop's propagation is differentiable, the analytic output
//! variance under input noise can go straight into the loss. Penalizing it
//! `loss = MSE + lambda * mean(output_variance)` changes the training objective.
//! The example reports clean and noisy test RMSE for that objective and plain MSE.
//!
//! Both nets start from the same weights and use the same noisy test draws,
//! so only the loss differs. Each backend has its own RNG stream, so metrics
//! from separate backend runs need not match.
//!
//! Run on CPU: `cargo run --release --example robust_training --features burn`
//! Run on macOS Metal: `cargo run --release --example robust_training --features metal -- --metal`

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{activation, Device, Distribution, Tensor, TensorData};
use burn_ndarray::NdArray;
use std::time::Instant;

use stableprop::burn_sdp::{propagate_linear, propagate_relu, Moments};

type Ad = Autodiff<NdArray<f32>>;

const D_IN: usize = 6;
const HIDDEN: usize = 64;
const N_TRAIN: usize = 3000;
const N_TEST: usize = 1000;
const TRAIN_STD: f64 = 0.2;
const TEST_STD: f64 = 0.3;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.lin2.forward(activation::relu(self.lin1.forward(x)))
    }
    /// Mean prediction and analytic output variance under input noise `std`.
    fn forward_with_var(&self, x: Tensor<B, 2>, std: f64) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [n, d] = x.dims();
        let var0 = Tensor::<B, 2>::full([n, d], std * std, &x.device());
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

fn run<B: AutodiffBackend>(dev: B::Device, backend: &str) {
    B::seed(&dev, 0xA0B5_7001);
    let make = |n: usize| -> (Tensor<B, 2>, Tensor<B, 2>) {
        let xt = Tensor::<B, 2>::random([n, D_IN], Distribution::Normal(0.0, 1.0), &dev);
        let xv = xt.to_data().to_vec::<f32>().unwrap();
        let yv: Vec<f32> = (0..n)
            .map(|i| target(&xv[i * D_IN..(i + 1) * D_IN]))
            .collect();
        (xt, Tensor::from_data(TensorData::new(yv, [n, 1]), &dev))
    };
    let (x_tr, y_tr) = make(N_TRAIN);
    let (x_te, y_te) = make(N_TEST);

    // Same starting weights for both nets: only the loss differs.
    let init = Mlp::<B>::init(&dev);
    let train = |mut model: Mlp<B>, lambda: f64| -> Mlp<B> {
        let mut optim = AdamConfig::new().init();
        for _ in 0..800 {
            let (pred, var) = model.forward_with_var(x_tr.clone(), TRAIN_STD);
            let mut loss = MseLoss::new().forward(pred, y_tr.clone(), Reduction::Mean);
            if lambda > 0.0 {
                loss = loss + var.mean().mul_scalar(lambda);
            }
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-3, model, grads);
        }
        model
    };
    B::sync(&dev).expect("training backend synchronization failed");
    let started = Instant::now();
    let plain = train(init.clone(), 0.0);
    let robust = train(init, 3.0);
    B::sync(&dev).expect("training backend synchronization failed");
    let elapsed = started.elapsed();

    // Same test-noise draws for both nets.
    let clean = vec![x_te.clone()];
    let noisy: Vec<Tensor<B, 2>> = (0..20)
        .map(|_| {
            x_te.clone()
                + Tensor::<B, 2>::random([N_TEST, D_IN], Distribution::Normal(0.0, TEST_STD), &dev)
        })
        .collect();
    let y = y_te.into_data().to_vec::<f32>().unwrap();
    let rmse = |model: &Mlp<B>, inputs: &[Tensor<B, 2>]| -> f64 {
        let mut total = 0.0;
        for x in inputs {
            let p = model
                .forward(x.clone())
                .into_data()
                .to_vec::<f32>()
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
        "Training elapsed (includes first-use kernel compilation/autotuning): {:.3}s",
        elapsed.as_secs_f64()
    );
    println!("RMSE (lower = better), shared init + shared test noise:");
    println!("  {:<28} {:>8} {:>8}", "net", "clean", "noisy");
    println!(
        "  {:<28} {:>8.4} {:>8.4}",
        "plain MSE",
        rmse(&plain, &clean),
        rmse(&plain, &noisy)
    );
    println!(
        "  {:<28} {:>8.4} {:>8.4}",
        "MSE + variance penalty",
        rmse(&robust, &clean),
        rmse(&robust, &noisy)
    );
    println!("\nCompare clean and noisy RMSE; the variance penalty can change either metric.");
    println!("Different backend RNG streams can produce different trained metrics.");
}

fn run_ndarray() {
    run::<Ad>(Device::<Ad>::default(), "NdArray CPU");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_metal() {
    use burn::backend::wgpu::{
        graphics::{GraphicsApi, Metal},
        init_setup, RuntimeOptions, WgpuDevice,
    };

    type MetalAd = Autodiff<burn::backend::Metal<f32>>;
    let device = WgpuDevice::DefaultDevice;
    let setup = init_setup::<Metal>(&device, RuntimeOptions::default());
    assert_eq!(setup.backend, Metal::backend());
    println!("Device: {}", setup.adapter.get_info().name);
    run::<MetalAd>(device, "Metal");
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
        run_ndarray();
    }

    Ok(())
}
