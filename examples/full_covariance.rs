//! Full-covariance vs diagonal propagation: keeping the cross-feature
//! correlations a layer introduces makes the output variance more accurate.
//!
//! Diagonal propagation drops the off-diagonal covariance after each layer, so a
//! second linear layer (which recombines correlated hidden units) gets the
//! variance wrong. Full-covariance propagation (`MomentsFull`) keeps it. This
//! pushes input noise through a 2-layer MLP both ways and compares each to Monte
//! Carlo; the full-covariance output std is closer.
//!
//! Run: `cargo run --release --example full_covariance --features burn`

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Distribution, Tensor, TensorData};
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

fn main() {
    let dev = <Nd as Backend>::Device::default();
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
    let diag_std: Vec<f64> = d2
        .var
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| (*v as f64).max(0.0).sqrt())
        .collect();

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

    // Monte Carlo.
    let len = N * D_OUT;
    let mut sums = vec![0.0f64; len];
    let mut sumsq = vec![0.0f64; len];
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
    }
    let kf = MC_SAMPLES as f64;
    let mc_std: Vec<f64> = (0..len)
        .map(|i| {
            ((sumsq[i] - sums[i] * sums[i] / kf) / (kf - 1.0))
                .max(0.0)
                .sqrt()
        })
        .collect();

    println!("output std vs {MC_SAMPLES}-sample Monte Carlo (mean ratio, 1.0 = unbiased):");
    println!("  diagonal       {:.3}", mean_ratio(&diag_std, &mc_std));
    println!("  full covariance {:.3}", mean_ratio(&full_std, &mc_std));
    println!("\nfull covariance keeps the cross-feature correlations the diagonal drops.");
}
