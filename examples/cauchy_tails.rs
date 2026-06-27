//! Why Cauchy: under heavy-tailed input noise, Gaussian intervals under-cover
//! and Cauchy (stable) intervals do not.
//!
//! When the true input perturbation is heavy-tailed (occasional large outliers),
//! propagating it as a Gaussian gives intervals that are too narrow -- the tail
//! events escape. Propagating it as a Cauchy keeps the heavy tails. This pushes
//! the same net both ways and measures interval coverage of Cauchy-perturbed
//! outputs.
//!
//! Run: `cargo run --release --example cauchy_tails --features burn`

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Distribution, Tensor, TensorData};
use burn_ndarray::NdArray;
use std::f64::consts::PI;

use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_cauchy, propagate_relu, propagate_relu_cauchy, Cauchy,
    Moments,
};

type Nd = NdArray<f32>;

const D_IN: usize = 4;
const HIDDEN: usize = 16;
const D_OUT: usize = 2;
const N: usize = 1000;
const GAMMA: f64 = 0.2; // Cauchy scale of the true input noise
const MC_SAMPLES: usize = 4000;

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

fn main() {
    let dev = <Nd as Backend>::Device::default();
    let model = Mlp::<Nd>::init(&dev);
    let w1 = model.lin1.weight.val();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val());
    let w2 = model.lin2.weight.val();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val());

    let x = Tensor::<Nd, 2>::random([N, D_IN], Distribution::Normal(0.0, 1.0), &dev);
    let len = N * D_OUT;

    // Cauchy propagation: location + scale, then a 90% interval half-width.
    let c0 = Cauchy::new(x.clone(), Tensor::<Nd, 2>::full([N, D_IN], GAMMA, &dev));
    let c1 = propagate_relu_cauchy(&propagate_linear_cauchy(&c0, w1.clone(), b1.clone()));
    let c2 = propagate_linear_cauchy(&c1, w2.clone(), b2.clone());
    let c_loc = c2.location.to_data().to_vec::<f32>().unwrap();
    let c_hw = c2
        .interval_halfwidth(0.9)
        .to_data()
        .to_vec::<f32>()
        .unwrap();

    // Gaussian propagation with variance = gamma^2 (a naive match), 90% interval.
    let m0 = Moments::new(
        x.clone(),
        Tensor::<Nd, 2>::full([N, D_IN], GAMMA * GAMMA, &dev),
    );
    let m1 = propagate_relu(&propagate_linear(&m0, w1.clone(), b1.clone()));
    let m2 = propagate_linear(&m1, w2.clone(), b2.clone());
    let g_mean = m2.mean.to_data().to_vec::<f32>().unwrap();
    let g_std: Vec<f64> = m2
        .var
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| (*v as f64).max(0.0).sqrt())
        .collect();

    // True outputs under Cauchy input noise; measure coverage of each interval.
    let mut rng = 0x0CA0_C1A0_u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
    };
    let xv = x.to_data().to_vec::<f32>().unwrap();
    let w1v = w1.to_data().to_vec::<f32>().unwrap();
    let b1v = b1.clone().unwrap().to_data().to_vec::<f32>().unwrap();
    let w2v = w2.to_data().to_vec::<f32>().unwrap();
    let b2v = b2.clone().unwrap().to_data().to_vec::<f32>().unwrap();

    let (mut g_cov, mut c_cov, mut total) = (0usize, 0usize, 0usize);
    for _ in 0..MC_SAMPLES {
        for i in 0..N {
            // Cauchy-perturbed input row.
            let xi: Vec<f64> = (0..D_IN)
                .map(|k| xv[i * D_IN + k] as f64 + GAMMA * (PI * (next() - 0.5)).tan())
                .collect();
            let h: Vec<f64> = (0..HIDDEN)
                .map(|j| {
                    (b1v[j] as f64
                        + (0..D_IN)
                            .map(|k| xi[k] * w1v[k * HIDDEN + j] as f64)
                            .sum::<f64>())
                    .max(0.0)
                })
                .collect();
            for o in 0..D_OUT {
                let y = b2v[o] as f64
                    + (0..HIDDEN)
                        .map(|j| h[j] * w2v[j * D_OUT + o] as f64)
                        .sum::<f64>();
                let idx = i * D_OUT + o;
                let g_lo = g_mean[idx] as f64 - 1.645 * g_std[idx];
                let g_hi = g_mean[idx] as f64 + 1.645 * g_std[idx];
                if y >= g_lo && y <= g_hi {
                    g_cov += 1;
                }
                let c_lo = c_loc[idx] as f64 - c_hw[idx] as f64;
                let c_hi = c_loc[idx] as f64 + c_hw[idx] as f64;
                if y >= c_lo && y <= c_hi {
                    c_cov += 1;
                }
                total += 1;
            }
        }
    }

    println!("90% interval coverage under heavy-tailed (Cauchy) input noise:");
    println!("  Gaussian propagation  {:.3}", g_cov as f64 / total as f64);
    println!("  Cauchy propagation    {:.3}", c_cov as f64 / total as f64);
    println!("\nGaussian intervals are too narrow for heavy tails; Cauchy keeps them.");
}
