//! Analytic estimate of a classifier's misclassification probability under
//! Gaussian input noise, with no sampling.
//!
//! Propagating the input noise with FULL covariance gives a joint Gaussian over
//! the logits. For the true class `t`, competitor `j` wins when the margin
//! `logit_t - logit_j` (Gaussian, variance `S_tt + S_jj - 2 S_tj` from the
//! propagated covariance) goes negative; a union bound over competitors estimates
//! `P(wrong) ~ sum_j Phi(-margin_mean_j / margin_std_j)`.
//!
//! Honest status: this is an ESTIMATE, not a sound certificate. The propagation
//! and the Gaussian-logit assumption are approximations, so per input the value
//! can land just above or below the true rate. It is NOT a guaranteed upper bound
//! like PROVEN (Weng et al., ICML 2019), which needs interval/Lipschitz methods.
//! What it gives cheaply is an estimate that tracks the Monte-Carlo rate closely
//! on average. We report both.
//!
//! Run: `cargo run --release --example certified_robustness --features burn`

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Int, Tensor, TensorData};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{propagate_linear_full, propagate_relu_full, MomentsFull};

type Ad = Autodiff<NdArray<f32>>;
type Nd = NdArray<f32>;

const D_IN: usize = 4;
const HIDDEN: usize = 32;
const N_CLASS: usize = 3;
const N_TRAIN: usize = 3000;
const N_TEST: usize = 400;
const INPUT_STD: f64 = 0.25;
const MC_SAMPLES: usize = 400;

#[derive(Module, Debug)]
struct Net<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> Net<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, N_CLASS).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.lin2.forward(activation::relu(self.lin1.forward(x)))
    }
}

fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}
fn phi_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Class-conditional Gaussian blobs: balanced classes, blob `c` shifted on two
/// features so the classes separate.
fn make(n: usize, dev: &<Ad as Backend>::Device) -> (Vec<f32>, Vec<i32>) {
    let mut x =
        Tensor::<Ad, 2>::random([n, D_IN], burn::tensor::Distribution::Normal(0.0, 0.6), dev)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
    let lab: Vec<i32> = (0..n).map(|i| (i % N_CLASS) as i32).collect();
    for i in 0..n {
        let c = lab[i] as f32;
        x[i * D_IN] += 1.6 * c;
        x[i * D_IN + 1] -= 1.2 * c;
    }
    (x, lab)
}

fn main() {
    let dev = <Ad as Backend>::Device::default();
    let idev = <Nd as Backend>::Device::default();
    let (xtr, ytr) = make(N_TRAIN, &dev);
    let (xte, yte) = make(N_TEST, &dev);

    let x_train = Tensor::<Ad, 2>::from_data(TensorData::new(xtr, [N_TRAIN, D_IN]), &dev);
    let y_train = Tensor::<Ad, 1, Int>::from_data(TensorData::new(ytr, [N_TRAIN]), &dev);

    let mut model = Net::<Ad>::init(&dev);
    let mut optim = AdamConfig::new().init();
    println!("training classifier...");
    for _ in 0..400 {
        let logits = model.forward(x_train.clone());
        let loss = CrossEntropyLoss::new(None, &dev).forward(logits, y_train.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-2, model, grads);
    }

    // Full-covariance propagation of input noise -> joint logit Gaussian.
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val().inner());
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val().inner());
    let x_te = Tensor::<Nd, 2>::from_data(TensorData::new(xte.clone(), [N_TEST, D_IN]), &idev);
    let var0 = Tensor::<Nd, 2>::full([N_TEST, D_IN], INPUT_STD * INPUT_STD, &idev);
    let m0 = MomentsFull::from_diagonal(x_te.clone(), var0);
    let m1 = propagate_relu_full(&propagate_linear_full(&m0, w1.clone(), b1.clone()));
    let m2 = propagate_linear_full(&m1, w2.clone(), b2.clone());
    let mean = m2.mean.to_data().to_vec::<f32>().unwrap(); // [N_TEST * C]
    let cov = m2.cov.to_data().to_vec::<f32>().unwrap(); // [N_TEST * C * C]

    // Certified per-input misclassification bound (union over competitor margins).
    let c = N_CLASS;
    let mut bound = vec![0.0f64; N_TEST];
    for i in 0..N_TEST {
        let t = yte[i] as usize;
        let mu = |k: usize| mean[i * c + k] as f64;
        let s = |a: usize, b: usize| cov[i * c * c + a * c + b] as f64;
        let mut p = 0.0;
        for j in 0..c {
            if j == t {
                continue;
            }
            let mm = mu(t) - mu(j);
            let mv = (s(t, t) + s(j, j) - 2.0 * s(t, j)).max(1e-9);
            p += phi_cdf(-mm / mv.sqrt());
        }
        bound[i] = p.min(1.0);
    }

    // Monte-Carlo misclassification rate per input.
    let mut mc = vec![0.0f64; N_TEST];
    for _ in 0..MC_SAMPLES {
        let noise = Tensor::<Nd, 2>::random(
            [N_TEST, D_IN],
            burn::tensor::Distribution::Normal(0.0, INPUT_STD),
            &idev,
        );
        let logits = (activation::relu(
            (x_te.clone() + noise).matmul(w1.clone()) + b1.clone().unwrap().reshape([1, HIDDEN]),
        )
        .matmul(w2.clone())
            + b2.clone().unwrap().reshape([1, N_CLASS]))
        .to_data()
        .to_vec::<f32>()
        .unwrap();
        for i in 0..N_TEST {
            let row = &logits[i * c..(i + 1) * c];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred != yte[i] as usize {
                mc[i] += 1.0;
            }
        }
    }
    for v in mc.iter_mut() {
        *v /= MC_SAMPLES as f64;
    }

    let mean_bound = bound.iter().sum::<f64>() / N_TEST as f64;
    let mean_mc = mc.iter().sum::<f64>() / N_TEST as f64;
    let valid = (0..N_TEST).filter(|&i| bound[i] + 1e-3 >= mc[i]).count() as f64 / N_TEST as f64;

    println!("\nanalytic misclassification-risk estimate under input noise std {INPUT_STD}:");
    println!("  mean analytic estimate = {mean_bound:.4}");
    println!("  mean MC rate           = {mean_mc:.4}  ({MC_SAMPLES} samples)");
    println!(
        "  estimate within {:.4} of MC on average; lands above the per-input rate {:.1}% of the time (an estimate, not a guaranteed bound).",
        (mean_bound - mean_mc).abs(),
        100.0 * valid
    );
}
