//! Estimates pairwise ranking-flip risk from propagated score covariance.
//!
//! A small query MLP scores two fixed candidate embeddings. Gaussian feature
//! noise changes the query representation; a flip is when the noisy winner
//! differs from the point-network winner. Full covariance gives the Gaussian
//! margin variance `S_00 + S_11 - 2 S_01`; the comparison that drops `S_01`
//! shows why independent score variances are insufficient for this question.
//!
//! The ReLU Gaussian closure is approximate, so the nonlinear estimate is
//! evaluated against shared-noise Monte Carlo. Here the score distribution
//! comes from feature noise. A fitted reward posterior can supply joint scores
//! too; valuing exploration additionally requires an observation model and
//! posterior updates. See docs/sensitivity-and-selection.md for that connection.
//!
//! Run: `cargo run --release --example pairwise_ranking_risk --features burn`

use burn::tensor::backend::Backend;
use burn::tensor::{activation, Device, Distribution, Tensor, TensorData};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{propagate_linear_full, propagate_relu_full, MomentsFull};

type Nd = NdArray<f32>;

const N_QUERY: usize = 128;
const D_QUERY: usize = 2;
const HIDDEN: usize = 4;
const N_CANDIDATE: usize = 2;
const INPUT_STD: f64 = 0.30;
const MC_SAMPLES: usize = 2048;

fn tensor2(data: Vec<f32>, shape: [usize; 2], dev: &Device<Nd>) -> Tensor<Nd, 2> {
    Tensor::from_data(TensorData::new(data, shape), dev)
}

fn cdf(z: Vec<f32>, dev: &Device<Nd>) -> Vec<f64> {
    let n = z.len();
    Tensor::<Nd, 1>::from_data(TensorData::new(z, [n]), dev)
        .mul_scalar(std::f64::consts::FRAC_1_SQRT_2)
        .erf()
        .add_scalar(1.0)
        .mul_scalar(0.5)
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

fn flip_probability(
    mean: &[f32],
    cov: &[f32],
    point_scores: &[f32],
    discard_covariance: bool,
    dev: &Device<Nd>,
) -> Vec<f64> {
    let mut probability = vec![0.0; N_QUERY];
    let mut indices = Vec::with_capacity(N_QUERY);
    let mut z = Vec::with_capacity(N_QUERY);
    for i in 0..N_QUERY {
        let winner = usize::from(point_scores[i * N_CANDIDATE] < point_scores[i * N_CANDIDATE + 1]);
        let loser = 1 - winner;
        let margin_mean = mean[i * N_CANDIDATE + winner] - mean[i * N_CANDIDATE + loser];
        let base = i * N_CANDIDATE * N_CANDIDATE;
        let mut margin_var =
            cov[base + winner * N_CANDIDATE + winner] + cov[base + loser * N_CANDIDATE + loser];
        if !discard_covariance {
            margin_var -= 2.0 * cov[base + winner * N_CANDIDATE + loser];
        }
        let scale = (cov[base + winner * N_CANDIDATE + winner].abs()
            + cov[base + loser * N_CANDIDATE + loser].abs())
        .max(1.0);
        assert!(
            margin_var >= -1e-5 * scale,
            "score-margin variance is materially negative: {margin_var}"
        );
        if margin_var <= 0.0 {
            probability[i] = if margin_mean < 0.0 {
                1.0
            } else if margin_mean > 0.0 {
                0.0
            } else {
                0.5
            };
        } else {
            indices.push(i);
            z.push(-margin_mean / margin_var.sqrt());
        }
    }
    for (i, p) in indices.into_iter().zip(cdf(z, dev)) {
        probability[i] = p;
    }
    probability
}

fn mean_abs_error(estimate: &[f64], observed: &[f64]) -> f64 {
    estimate
        .iter()
        .zip(observed)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / N_QUERY as f64
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn main() {
    let dev = Device::<Nd>::default();
    <Nd as Backend>::seed(&dev, 0xFA1F_0001);

    // A two-dimensional query grid and fixed two-tower-style scoring weights.
    let mut query = Vec::with_capacity(N_QUERY * D_QUERY);
    for i in 0..N_QUERY {
        let t = i as f32 / (N_QUERY - 1) as f32;
        query.extend([2.0 * t - 1.0, (6.0 * t - 3.0).sin()]);
    }
    let x = tensor2(query, [N_QUERY, D_QUERY], &dev);
    let query_weight = tensor2(
        vec![1.10, -0.70, 0.55, 0.30, -0.35, 0.90, 0.65, -1.00],
        [D_QUERY, HIDDEN],
        &dev,
    );
    let query_bias = Tensor::<Nd, 1>::from_data(
        TensorData::new(vec![0.10, -0.20, 0.15, 0.05], [HIDDEN]),
        &dev,
    );
    // Columns are fixed candidate embeddings in the query-representation space.
    let candidate_embedding = tensor2(
        vec![0.9, -0.6, -0.4, 0.8, 0.7, 0.2, -0.5, -0.9],
        [HIDDEN, N_CANDIDATE],
        &dev,
    );

    let point_scores = (activation::relu(
        x.clone().matmul(query_weight.clone()) + query_bias.clone().reshape([1, HIDDEN]),
    )
    .matmul(candidate_embedding.clone()))
    .to_data()
    .to_vec::<f32>()
    .unwrap();

    let input_var = Tensor::<Nd, 2>::full([N_QUERY, D_QUERY], INPUT_STD * INPUT_STD, &dev);
    let input = MomentsFull::from_diagonal(x.clone(), input_var);
    let nonlinear = propagate_linear_full(
        &propagate_relu_full(&propagate_linear_full(
            &input,
            query_weight.clone(),
            Some(query_bias.clone()),
        )),
        candidate_embedding.clone(),
        None,
    );
    let analytic_full = flip_probability(
        &nonlinear.mean.to_data().to_vec::<f32>().unwrap(),
        &nonlinear.cov.to_data().to_vec::<f32>().unwrap(),
        &point_scores,
        false,
        &dev,
    );
    let analytic_diagonal = flip_probability(
        &nonlinear.mean.to_data().to_vec::<f32>().unwrap(),
        &nonlinear.cov.to_data().to_vec::<f32>().unwrap(),
        &point_scores,
        true,
        &dev,
    );

    // This affine control has an exactly Gaussian score margin, isolating the
    // pairwise covariance calculation from the ReLU approximation above.
    let affine_hidden =
        propagate_linear_full(&input, query_weight.clone(), Some(query_bias.clone()));
    let affine = propagate_linear_full(&affine_hidden, candidate_embedding.clone(), None);
    let affine_point = (x.clone().matmul(query_weight.clone())
        + query_bias.clone().reshape([1, HIDDEN]))
    .matmul(candidate_embedding.clone())
    .to_data()
    .to_vec::<f32>()
    .unwrap();
    let affine_analytic = flip_probability(
        &affine.mean.to_data().to_vec::<f32>().unwrap(),
        &affine.cov.to_data().to_vec::<f32>().unwrap(),
        &affine_point,
        false,
        &dev,
    );

    let mut observed = vec![0.0; N_QUERY];
    let mut affine_observed = vec![0.0; N_QUERY];
    for _ in 0..MC_SAMPLES {
        let noisy_query = x.clone()
            + Tensor::<Nd, 2>::random(
                [N_QUERY, D_QUERY],
                Distribution::Normal(0.0, INPUT_STD),
                &dev,
            );
        let score = activation::relu(
            noisy_query.clone().matmul(query_weight.clone())
                + query_bias.clone().reshape([1, HIDDEN]),
        )
        .matmul(candidate_embedding.clone())
        .to_data()
        .to_vec::<f32>()
        .unwrap();
        let affine_score = (noisy_query.matmul(query_weight.clone())
            + query_bias.clone().reshape([1, HIDDEN]))
        .matmul(candidate_embedding.clone())
        .to_data()
        .to_vec::<f32>()
        .unwrap();
        for i in 0..N_QUERY {
            let winner =
                usize::from(point_scores[i * N_CANDIDATE] < point_scores[i * N_CANDIDATE + 1]);
            let affine_winner =
                usize::from(affine_point[i * N_CANDIDATE] < affine_point[i * N_CANDIDATE + 1]);
            observed[i] += if score[i * N_CANDIDATE + winner] < score[i * N_CANDIDATE + 1 - winner]
            {
                1.0
            } else {
                0.0
            };
            affine_observed[i] += if affine_score[i * N_CANDIDATE + affine_winner]
                < affine_score[i * N_CANDIDATE + 1 - affine_winner]
            {
                1.0
            } else {
                0.0
            };
        }
    }
    for value in observed.iter_mut().chain(affine_observed.iter_mut()) {
        *value /= MC_SAMPLES as f64;
    }

    let cov = nonlinear.cov.to_data().to_vec::<f32>().unwrap();
    let max_asymmetry = (0..N_QUERY)
        .map(|i| (cov[i * 4 + 1] - cov[i * 4 + 2]).abs() as f64)
        .fold(0.0, f64::max);
    assert!(analytic_full
        .iter()
        .all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    assert!(analytic_diagonal
        .iter()
        .all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    assert!(max_asymmetry < 1e-5, "output covariance must be symmetric");

    println!("pairwise ranking flips under feature noise std {INPUT_STD}:");
    println!(
        "  nonlinear full covariance MAE versus MC = {:.4}",
        mean_abs_error(&analytic_full, &observed)
    );
    println!(
        "  nonlinear diagonal-score MAE versus MC  = {:.4}",
        mean_abs_error(&analytic_diagonal, &observed)
    );
    println!(
        "  mean full / diagonal estimate  = {:.4} / {:.4}",
        mean(&analytic_full),
        mean(&analytic_diagonal)
    );
    println!(
        "  affine control MAE versus MC (exact Gaussian margin) = {:.4}",
        mean_abs_error(&affine_analytic, &affine_observed)
    );
    println!("  {MC_SAMPLES} shared-noise draws; max covariance asymmetry = {max_asymmetry:.2e}");
    println!("  MAEs include Monte Carlo sampling error.");
}
