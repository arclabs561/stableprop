//! Compares full and diagonal covariance propagation through a two-layer MLP
//! against Monte Carlo.
//!
//! A second affine layer recombines correlated hidden units. Full covariance
//! retains those cross-terms; the diagonal path drops them. ReLU covariance
//! uses a third-order approximation, so the seeded result does not establish
//! a general ordering of the methods.
//! With one hidden ReLU layer, no repeated Gaussian approximation is needed.
//! The Monte Carlo reference still has sampling error.
//!
//! Run: `cargo run --release --example full_covariance --features burn`

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Device, Distribution, Tensor};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_full, propagate_relu, propagate_relu_full, Moments,
    MomentsFull,
};

type Nd = NdArray<f32>;

const D_IN: usize = 8;
const HIDDEN: usize = 24;
const D_OUT: usize = 4;
const N: usize = 1000;
const INPUT_STD: f64 = 0.4;
const MC_SAMPLES: usize = 400;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, D_OUT).init(device),
        }
    }
}

fn mean_ratio(est: &[f64], mc: &[f64]) -> f64 {
    let r: Vec<f64> = est
        .iter()
        .zip(mc)
        .filter(|(_, m)| **m > 1e-6)
        .map(|(e, m)| e / m)
        .collect();
    r.iter().sum::<f64>() / r.len() as f64
}

fn mean_abs_relative_error(est: &[f64], mc: &[f64]) -> f64 {
    let errors: Vec<f64> = est
        .iter()
        .zip(mc)
        .filter(|(_, m)| **m > 1e-6)
        .map(|(e, m)| (e - m).abs() / m)
        .collect();
    errors.iter().sum::<f64>() / errors.len() as f64
}

fn normalized_frobenius_error(est: &[f64], mc: &[f64]) -> f64 {
    let squared_error: f64 = est
        .iter()
        .zip(mc)
        .map(|(estimate, reference)| (estimate - reference).powi(2))
        .sum();
    let reference_norm: f64 = mc.iter().map(|value| value.powi(2)).sum();
    squared_error.sqrt() / reference_norm.sqrt().max(1e-12)
}

fn main() {
    let dev = Device::<Nd>::default();
    <Nd as Backend>::seed(&dev, 0xF011_C0A1);
    let model = Mlp::<Nd>::init(&dev);
    let w1 = model.lin1.weight.val();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val());
    let w2 = model.lin2.weight.val();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val());

    let x = Tensor::<Nd, 2>::random([N, D_IN], Distribution::Normal(0.0, 1.0), &dev);
    let var0 = Tensor::<Nd, 2>::full([N, D_IN], INPUT_STD * INPUT_STD, &dev);

    // Diagonal propagation.
    let d1 = propagate_relu(&propagate_linear(
        &Moments::new(x.clone(), var0.clone()),
        w1.clone(),
        b1.clone(),
    ));
    let d2 = propagate_linear(&d1, w2.clone(), b2.clone());
    let diag_var: Vec<f64> = d2
        .var
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| *v as f64)
        .collect();
    let diag_std: Vec<f64> = diag_var.iter().map(|v| v.max(0.0).sqrt()).collect();

    // Full-covariance propagation.
    let f1 = propagate_relu_full(&propagate_linear_full(
        &MomentsFull::from_diagonal(x.clone(), var0),
        w1.clone(),
        b1.clone(),
    ));
    let f2 = propagate_linear_full(&f1, w2.clone(), b2.clone());
    let full_std: Vec<f64> = f2
        .variance()
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| (*v as f64).max(0.0).sqrt())
        .collect();
    let full_cov: Vec<f64> = f2
        .cov
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| *v as f64)
        .collect();

    // Monte Carlo.
    let len = N * D_OUT;
    let mut sums = vec![0.0f64; len];
    let mut sumsq = vec![0.0f64; len];
    let mut sum_outer = vec![0.0f64; N * D_OUT * D_OUT];
    for _ in 0..MC_SAMPLES {
        let noise = Tensor::<Nd, 2>::random([N, D_IN], Distribution::Normal(0.0, INPUT_STD), &dev);
        let h = activation::relu(
            (x.clone() + noise).matmul(w1.clone()) + b1.clone().unwrap().reshape([1, HIDDEN]),
        );
        let y = (h.matmul(w2.clone()) + b2.clone().unwrap().reshape([1, D_OUT]))
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        for i in 0..len {
            sums[i] += y[i] as f64;
            sumsq[i] += (y[i] as f64).powi(2);
        }
        for row in 0..N {
            for i in 0..D_OUT {
                for j in 0..D_OUT {
                    sum_outer[(row * D_OUT + i) * D_OUT + j] +=
                        y[row * D_OUT + i] as f64 * y[row * D_OUT + j] as f64;
                }
            }
        }
    }
    let kf = MC_SAMPLES as f64;
    let mc_std: Vec<f64> = (0..len)
        .map(|i| {
            ((sumsq[i] - sums[i] * sums[i] / kf) / (kf - 1.0))
                .max(0.0)
                .sqrt()
        })
        .collect();
    let mut mc_cov = vec![0.0; N * D_OUT * D_OUT];
    let mut diag_cov = vec![0.0; N * D_OUT * D_OUT];
    for row in 0..N {
        for i in 0..D_OUT {
            for j in 0..D_OUT {
                let index = (row * D_OUT + i) * D_OUT + j;
                mc_cov[index] = (sum_outer[index]
                    - sums[row * D_OUT + i] * sums[row * D_OUT + j] / kf)
                    / (kf - 1.0);
                if i == j {
                    diag_cov[index] = diag_var[row * D_OUT + i];
                }
            }
        }
    }

    println!("output std vs {MC_SAMPLES}-sample Monte Carlo:");
    println!(
        "  {:<17} {:>11} {:>11}",
        "method", "mean ratio", "mean abs rel err"
    );
    println!(
        "  {:<17} {:>11.3} {:>11.3}",
        "diagonal",
        mean_ratio(&diag_std, &mc_std),
        mean_abs_relative_error(&diag_std, &mc_std)
    );
    println!(
        "  {:<17} {:>11.3} {:>11.3}",
        "full covariance",
        mean_ratio(&full_std, &mc_std),
        mean_abs_relative_error(&full_std, &mc_std)
    );
    println!(
        "\nRatios are analytic / MC per output; lower relative error is closer on this seeded comparison."
    );
    println!("\nwithin-row output covariance vs {MC_SAMPLES}-sample Monte Carlo:");
    println!("  {:<17} {:>28}", "method", "normalized Frobenius error");
    println!(
        "  {:<17} {:>28.3}",
        "diagonal",
        normalized_frobenius_error(&diag_cov, &mc_cov)
    );
    println!(
        "  {:<17} {:>28.3}",
        "full covariance",
        normalized_frobenius_error(&full_cov, &mc_cov)
    );
    println!(
        "\nOne hidden ReLU layer avoids repeated Gaussian approximation; Monte Carlo still has sampling error."
    );
}
