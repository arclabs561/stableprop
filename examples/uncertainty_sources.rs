//! Separate input-noise sensitivity from supplied parameter uncertainty.
//!
//! For a scalar linear prediction `y = x^T beta`, this example propagates
//! independent diagonal-Gaussian input and parameter distributions. It compares
//! input-only, parameter-only, and combined uncertainty with seeded Monte Carlo.
//! The parameters are fixed synthetic distributions: `stableprop` propagates a
//! supplied distribution but does not fit a posterior.
//!
//! Run: `cargo run --release --example uncertainty_sources --features burn`

use burn::tensor::{Device, Tensor, TensorData};

use stableprop::burn_sdp::{propagate_linear_bayes, Moments};

const D: usize = 3;
const MC_SAMPLES: usize = 100_000;

fn next_uniform(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 2.0)
}

fn normal(state: &mut u64) -> f64 {
    let radius = (-2.0 * next_uniform(state).ln()).sqrt();
    radius * (core::f64::consts::TAU * next_uniform(state)).cos()
}

fn monte_carlo(
    x_mean: &[f64; D],
    x_var: &[f64; D],
    beta_mean: &[f64; D],
    beta_var: &[f64; D],
    seed: u64,
) -> (f64, f64) {
    let mut state = seed;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..MC_SAMPLES {
        let y = (0..D)
            .map(|i| {
                let x = x_mean[i] + x_var[i].sqrt() * normal(&mut state);
                let beta = beta_mean[i] + beta_var[i].sqrt() * normal(&mut state);
                x * beta
            })
            .sum::<f64>();
        sum += y;
        sum_sq += y * y;
    }
    let n = MC_SAMPLES as f64;
    let mean = sum / n;
    (mean, (sum_sq - sum * sum / n) / (n - 1.0))
}

fn propagated_variance(
    dev: &Device,
    x_mean: &[f64; D],
    x_var: &[f64; D],
    beta_mean: &[f64; D],
    beta_var: &[f64; D],
) -> (f64, f64) {
    let to_f32 = |xs: &[f64; D]| xs.iter().map(|&x| x as f32).collect();
    let x_mean = Tensor::<2>::from_data(TensorData::new(to_f32(x_mean), [1, D]), dev);
    let x_var = Tensor::<2>::from_data(TensorData::new(to_f32(x_var), [1, D]), dev);
    let beta_mean = Tensor::<2>::from_data(TensorData::new(to_f32(beta_mean), [D, 1]), dev);
    let beta_var = Tensor::<2>::from_data(TensorData::new(to_f32(beta_var), [D, 1]), dev);
    let out = propagate_linear_bayes(&Moments::new(x_mean, x_var), beta_mean, beta_var, None);
    let mean = out.mean.to_data().try_to_vec::<f32>().unwrap()[0] as f64;
    let variance = out.var.to_data().try_to_vec::<f32>().unwrap()[0] as f64;
    (mean, variance)
}

fn main() {
    let dev = Device::flex();
    let x_mean = [1.25, -0.75, 0.50];
    let x_var = [0.16, 0.09, 0.04];
    let beta_mean = [0.80, -1.10, 0.50];
    let beta_var = [0.04, 0.01, 0.09];
    let zero = [0.0; D];

    let input = x_var
        .iter()
        .zip(beta_mean)
        .map(|(vx, beta)| vx * beta * beta)
        .sum::<f64>();
    let parameter = x_mean
        .iter()
        .zip(beta_var)
        .map(|(x, vb)| x * x * vb)
        .sum::<f64>();
    let product = x_var
        .iter()
        .zip(beta_var)
        .map(|(vx, vb)| vx * vb)
        .sum::<f64>();

    println!("uncertainty source                  propagated       Monte Carlo");
    for (name, xv, bv, expected) in [
        ("input only", x_var, zero, input),
        ("parameter only", zero, beta_var, parameter),
        ("both", x_var, beta_var, input + parameter + product),
    ] {
        let (mean, variance) = propagated_variance(&dev, &x_mean, &xv, &beta_mean, &bv);
        let (mc_mean, mc_variance) = monte_carlo(&x_mean, &xv, &beta_mean, &bv, 0x51A7_E000);
        assert!((variance - expected).abs() < 2e-6);
        assert!((mean - mc_mean).abs() < 0.015);
        assert!((variance - mc_variance).abs() < 0.02);
        println!("  {name:<28} {variance:>10.5}       {mc_variance:>10.5}");
    }
    println!("\ndecomposition for both: input {input:.5} + parameter {parameter:.5} + product {product:.5}");
    println!(
        "the product term means combined uncertainty is not the naive sum of the first two rows."
    );
}
