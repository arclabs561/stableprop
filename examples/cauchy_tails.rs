//! Compare Gaussian propagation with a local-linear Cauchy approximation under
//! heavy-tailed input noise.
//!
//! The synthetic inputs are Cauchy-perturbed. The example propagates Gaussian
//! moments and Cauchy location/scale through the same network, then reports
//! observed interval coverage for both descriptions in one seeded synthetic
//! run. Affine Cauchy propagation is exact for independent marginals, but the
//! ReLU step is a local-linear approximation: its output is not generally
//! Cauchy, so its nominal interval has no coverage guarantee.
//!
//! Run: `cargo run --release --example cauchy_tails --features burn`

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Distribution, Tensor};
use std::f64::consts::PI;

use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_cauchy, propagate_relu, propagate_relu_cauchy, Cauchy,
    Moments,
};

const D_IN: usize = 4;
const HIDDEN: usize = 16;
const D_OUT: usize = 2;
const N: usize = 1000;
const GAMMA: f64 = 0.2; // Cauchy scale of the true input noise
const MC_SAMPLES: usize = 4000;

#[derive(Module, Debug)]
struct Mlp {
    lin1: Linear,
    lin2: Linear,
}

impl Mlp {
    fn init(device: &Device) -> Self {
        Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, D_OUT).init(device),
        }
    }
}

fn main() {
    let dev = Device::flex();
    dev.seed(0xCA0C_0001);
    let model = Mlp::init(&dev);
    let w1 = model.lin1.weight.val();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val());
    let w2 = model.lin2.weight.val();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val());

    let x = Tensor::<2>::random([N, D_IN], Distribution::Normal(0.0, 1.0), &dev);
    // Exact affine Cauchy propagation followed by the local-linear ReLU
    // approximation; use its location + scale for a nominal 90% half-width.
    let c0 = Cauchy::new(x.clone(), Tensor::<2>::full([N, D_IN], GAMMA, &dev));
    let c1 = propagate_relu_cauchy(&propagate_linear_cauchy(&c0, w1.clone(), b1.clone()));
    let c2 = propagate_linear_cauchy(&c1, w2.clone(), b2.clone());
    let c_loc = c2.location.to_data().try_to_vec::<f32>().unwrap();
    let c_hw = c2
        .interval_halfwidth(0.9)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();

    // Gaussian propagation with variance = gamma^2 (a naive match), 90% interval.
    let m0 = Moments::new(x.clone(), Tensor::<2>::full([N, D_IN], GAMMA * GAMMA, &dev));
    let m1 = propagate_relu(&propagate_linear(&m0, w1.clone(), b1.clone()));
    let m2 = propagate_linear(&m1, w2.clone(), b2.clone());
    let g_mean = m2.mean.to_data().try_to_vec::<f32>().unwrap();
    let g_std: Vec<f64> = m2
        .var
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| {
            let variance = f64::from(*v);
            assert!(
                variance.is_finite() && variance >= 0.0,
                "propagated variance must be finite and nonnegative"
            );
            variance.sqrt()
        })
        .collect();

    // True outputs under Cauchy input noise; record observed coverage of each
    // nominal interval in this synthetic draw.
    let mut rng = 0x0CA0_C1A0_u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
    };
    let xv = x.to_data().try_to_vec::<f32>().unwrap();
    let w1v = w1.to_data().try_to_vec::<f32>().unwrap();
    let b1v = b1.clone().unwrap().to_data().try_to_vec::<f32>().unwrap();
    let w2v = w2.to_data().try_to_vec::<f32>().unwrap();
    let b2v = b2.clone().unwrap().to_data().try_to_vec::<f32>().unwrap();

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

    println!("observed coverage of nominal 90% intervals under Cauchy input noise:");
    println!("  Gaussian propagation  {:.3}", g_cov as f64 / total as f64);
    println!(
        "  local-linear Cauchy    {:.3}",
        c_cov as f64 / total as f64
    );
    println!(
        "\nThe Cauchy ReLU approximation is not a calibrated interval construction; compare this run with 0.90, not as a guarantee."
    );
}
