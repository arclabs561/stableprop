//! Shows why a residual branch needs its covariance with the skip path.
//!
//! `Y = X + ReLU(X W + b) V` has a branch driven by the same uncertain input
//! as its skip. Treating the two paths as independent therefore misses
//! `2 Cov(X, branch)`. The one-dimensional hidden state makes this example's
//! propagated marginal moments exact apart from the ReLU CDF tail numerics.
//!
//! Run: `cargo run --release --example correlated_residual --features burn`

use burn::tensor::linalg;
use burn::tensor::{Device, Tensor, TensorData};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_cross_covariance, propagate_relu,
    propagate_relu_cross_covariance, propagate_residual_add, propagate_residual_add_correlated,
    Moments, MomentsFull,
};

type Nd = NdArray<f32>;

const D_IN: usize = 2;
const D_OUT: usize = 2;
const MC_SAMPLES: usize = 100_000;

fn next_uniform(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 2.0)
}

fn normal(state: &mut u64) -> f64 {
    (-2.0 * next_uniform(state).ln()).sqrt() * (core::f64::consts::TAU * next_uniform(state)).cos()
}

fn monte_carlo(
    mean: [f64; D_IN],
    std: [f64; D_IN],
    weight: [[f64; 1]; D_IN],
    bias: f64,
    value: [[f64; D_OUT]; 1],
) -> ([f64; D_OUT], [f64; D_OUT]) {
    let mut state = 0xC0_7E_1A_7Eu64;
    let mut sum = [0.0; D_OUT];
    let mut sum_sq = [0.0; D_OUT];
    for _ in 0..MC_SAMPLES {
        let x = [
            mean[0] + std[0] * normal(&mut state),
            mean[1] + std[1] * normal(&mut state),
        ];
        let hidden = (x[0] * weight[0][0] + x[1] * weight[1][0] + bias).max(0.0);
        for output in 0..D_OUT {
            let y = x[output] + hidden * value[0][output];
            sum[output] += y;
            sum_sq[output] += y * y;
        }
    }
    let n = MC_SAMPLES as f64;
    let mut sample_mean = [0.0; D_OUT];
    let mut sample_var = [0.0; D_OUT];
    for output in 0..D_OUT {
        sample_mean[output] = sum[output] / n;
        sample_var[output] = (sum_sq[output] - sum[output] * sum[output] / n) / (n - 1.0);
    }
    (sample_mean, sample_var)
}

fn main() {
    let dev = Device::<Nd>::default();
    let x_mean = [0.2, -0.1];
    let x_std = [0.5, 0.4];
    let weight = [[1.0], [-0.75]];
    let bias = 0.1;
    let value = [[0.8, 0.6]];

    let tensor2 =
        |values: Vec<f32>, shape| Tensor::<Nd, 2>::from_data(TensorData::new(values, shape), &dev);
    let mean = tensor2(x_mean.map(|x| x as f32).to_vec(), [1, D_IN]);
    let var = tensor2(x_std.map(|x| (x * x) as f32).to_vec(), [1, D_IN]);
    let w = tensor2(
        weight.into_iter().flatten().map(|x| x as f32).collect(),
        [D_IN, 1],
    );
    let v = tensor2(
        value.into_iter().flatten().map(|x| x as f32).collect(),
        [1, D_OUT],
    );
    let b = Tensor::<Nd, 1>::from_data([bias as f32], &dev);

    let skip = Moments::new(mean.clone(), var.clone());
    let hidden_pre = propagate_linear(&skip, w.clone(), Some(b));
    let hidden = propagate_relu(&hidden_pre);
    let branch = propagate_linear(&hidden, v.clone(), None);
    let independent = propagate_residual_add(&skip, &branch);

    // Cxx is Cov(X, X); carry its right variable through the branch.
    let cxx = MomentsFull::from_diagonal(mean, var).cov;
    let c_x_hidden_pre = propagate_linear_cross_covariance(cxx, w);
    let c_x_hidden = propagate_relu_cross_covariance(c_x_hidden_pre, &hidden_pre);
    let c_x_branch = propagate_linear_cross_covariance(c_x_hidden, v);
    let skip_branch_cov: Tensor<Nd, 2> = linalg::diag(c_x_branch);
    let correlated = propagate_residual_add_correlated(&skip, &branch, skip_branch_cov);

    let propagated_mean = correlated.mean.to_data().to_vec::<f32>().unwrap();
    let propagated_var = correlated.var.to_data().to_vec::<f32>().unwrap();
    let independent_var = independent.var.to_data().to_vec::<f32>().unwrap();
    let (mc_mean, mc_var) = monte_carlo(x_mean, x_std, weight, bias, value);

    println!("residual output moments vs {MC_SAMPLES}-sample seeded Monte Carlo:");
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "output", "mean", "corr var", "MC var", "ind var", "corr err", "ind err"
    );
    for output in 0..D_OUT {
        let mean_error = (propagated_mean[output] as f64 - mc_mean[output]).abs();
        println!(
            "  {output:>6} {:>10.5} {:>10.5} {:>10.5} {:>10.5} {:>10.5} {:>10.5}",
            propagated_mean[output],
            propagated_var[output],
            mc_var[output],
            independent_var[output],
            (propagated_var[output] as f64 - mc_var[output]).abs(),
            (independent_var[output] as f64 - mc_var[output]).abs(),
        );
        println!(
            "           MC mean {:>10.5}, absolute mean error {mean_error:.5}",
            mc_mean[output]
        );
    }
    println!("The shared branch raises one output variance and lowers the other for these fixed weights.");
}
