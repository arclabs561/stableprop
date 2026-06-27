//! The useful case: sampling-free prediction intervals for a regressor under
//! *known* input uncertainty.
//!
//! When a regression network's inputs carry known noise (sensor error, an
//! upstream model's variance), you want output error bars. The usual way is
//! Monte Carlo: push K noisy copies through and measure the spread. momentprop
//! gives the same mean and variance in ONE analytic pass. Unlike classification
//! (where softmax confidence is a strong free baseline), regression has no such
//! baseline -- MC or ensembles are the alternatives, and momentprop replaces
//! them cheaply.
//!
//! This demo trains an MLP regressor, then on a test set with known input noise
//! compares momentprop's analytic (mean, std) against K-sample Monte Carlo:
//! agreement of the error bars, and empirical coverage of the 95% interval.
//!
//! Run: `cargo run --release --example regression_intervals --features burn`

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Distribution, Tensor, TensorData};
use burn_ndarray::NdArray;

use momentprop::burn_sdp::{propagate_linear, propagate_relu, Moments};

type Ad = Autodiff<NdArray<f32>>;
type Nd = NdArray<f32>;

const D_IN: usize = 8;
const HIDDEN: usize = 64;
const N_TRAIN: usize = 3000;
const N_TEST: usize = 1000;
const MC_SAMPLES: usize = 200;

/// Single-hidden-layer MLP regressor: Linear -> ReLU -> Linear -> scalar.
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
        let h = activation::relu(self.lin1.forward(x));
        self.lin2.forward(h)
    }
}

/// Target function (nonlinear, with interactions) so ReLU moment-matching matters.
fn target(x: &[f32]) -> f32 {
    let s: f32 = x.iter().sum();
    (s * 0.7).sin() + 0.5 * x[0] * x[1] - 0.3 * x[2] * x[2] + 0.4 * (x[3] - x[4]).abs()
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    cov / (va.sqrt() * vb.sqrt())
}

fn main() {
    let dev = <Ad as Backend>::Device::default();

    let make = |n: usize| -> (Vec<f32>, Vec<f32>) {
        let xt = Tensor::<Ad, 2>::random([n, D_IN], Distribution::Normal(0.0, 1.0), &dev);
        let xv = xt.to_data().to_vec::<f32>().unwrap();
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            yv.push(target(&xv[i * D_IN..(i + 1) * D_IN]));
        }
        (xv, yv)
    };
    let (xtr, ytr) = make(N_TRAIN);
    let (xte, _yte) = make(N_TEST);
    let x_train = Tensor::<Ad, 2>::from_data(TensorData::new(xtr, [N_TRAIN, D_IN]), &dev);
    let y_train = Tensor::<Ad, 2>::from_data(TensorData::new(ytr, [N_TRAIN, 1]), &dev);

    let mut model = Mlp::<Ad>::init(&dev);
    let mut optim = AdamConfig::new().init();
    println!("training MLP regressor ({N_TRAIN} samples, 800 epochs)...");
    for _ in 0..800 {
        let pred = model.forward(x_train.clone());
        let loss = MseLoss::new().forward(pred, y_train.clone(), Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-3, model, grads);
    }
    let train_rmse = {
        let p = model
            .forward(x_train.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let y = y_train.into_data().to_vec::<f32>().unwrap();
        (p.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / N_TRAIN as f32).sqrt()
    };
    println!("train RMSE: {train_rmse:.4}\n");

    // Inner-backend weights for analytic propagation.
    let idev = <Nd as Backend>::Device::default();
    let x_test = Tensor::<Nd, 2>::from_data(TensorData::new(xte, [N_TEST, D_IN]), &idev);
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val().inner());
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val().inner());

    // momentprop: analytic (mean, var) in ONE pass.
    // Heteroscedastic: each test point carries its own known input-noise std
    // (the realistic case -- different measurements have different uncertainty).
    let sigma = Tensor::<Nd, 2>::random([N_TEST, 1], Distribution::Uniform(0.05, 0.4), &idev);
    let var0 = (sigma.clone() * sigma.clone()).expand([N_TEST, D_IN]);
    let m0 = Moments::new(x_test.clone(), var0);
    let m1 = propagate_relu(&propagate_linear(&m0, w1.clone(), b1.clone()));
    let m2 = propagate_linear(&m1, w2.clone(), b2.clone());
    let mp_mean = m2.mean.to_data().to_vec::<f32>().unwrap();
    let mp_std: Vec<f64> = m2
        .var
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| (*v as f64).max(0.0).sqrt())
        .collect();

    // Monte Carlo: K noisy copies through the deterministic net.
    let mut sums = vec![0.0f64; N_TEST];
    let mut sumsq = vec![0.0f64; N_TEST];
    let mut within = 0usize;
    let mut total = 0usize;
    for _ in 0..MC_SAMPLES {
        let z = Tensor::<Nd, 2>::random([N_TEST, D_IN], Distribution::Normal(0.0, 1.0), &idev);
        let xk = x_test.clone() + z * sigma.clone();
        let h = activation::relu(xk.matmul(w1.clone()) + b1.clone().unwrap().reshape([1, HIDDEN]));
        let yk = (h.matmul(w2.clone()) + b2.clone().unwrap().reshape([1, 1]))
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        for i in 0..N_TEST {
            let v = yk[i] as f64;
            sums[i] += v;
            sumsq[i] += v * v;
            let lo = mp_mean[i] as f64 - 1.96 * mp_std[i];
            let hi = mp_mean[i] as f64 + 1.96 * mp_std[i];
            if v >= lo && v <= hi {
                within += 1;
            }
            total += 1;
        }
    }
    let kf = MC_SAMPLES as f64;
    let mc_std: Vec<f64> = (0..N_TEST)
        .map(|i| {
            ((sumsq[i] - sums[i] * sums[i] / kf) / (kf - 1.0))
                .max(0.0)
                .sqrt()
        })
        .collect();

    let r = pearson(&mp_std, &mc_std);
    let ratios: Vec<f64> = mp_std
        .iter()
        .zip(&mc_std)
        .filter(|(_, m)| **m > 1e-6)
        .map(|(s, m)| s / m)
        .collect();
    let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let coverage = within as f64 / total as f64;

    println!("sampling-free error bars vs {MC_SAMPLES}-sample Monte Carlo:");
    println!("  std agreement (Pearson r) = {r:.4}   (1.0 = identical error bars)");
    println!("  std mean ratio (mp / MC)  = {mean_ratio:.3}  (1.0 = unbiased magnitude)");
    println!("  95% interval coverage     = {coverage:.3}   (target ~0.95 = calibrated)");
    println!("\ncost: momentprop = 1 forward pass, Monte Carlo = {MC_SAMPLES} passes");
}
