//! Compare matched abstention policies for pairwise ranking under Gaussian input noise.
//!
//! The outcome is a noisy winner differing from the point-network winner. This
//! is sensitivity under a supplied perturbation model, not relevance, reward,
//! or an acquisition value. Full score covariance, dropped score covariance,
//! a rank-only point margin, a local Jacobian, and sampled scores defer the same
//! query fraction. Reference, held-out, and sampled-policy draws are independent.
//!
//! Run: `cargo run --release --example pairwise_ranking_risk --features burn`
//! Add `-- --study` for 30 fixed-model fixtures and larger Monte Carlo budgets.
//! Add `-- --generalize` for independently generated, untrained networks and
//! candidate/query fixtures; `-- --generalize --quick` is a three-model
//! descriptive workflow check.

use burn::tensor::{Device, Tensor, TensorData};
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{propagate_linear_full, propagate_relu_full, MomentsFull};
use std::time::Instant;

type Nd = NdArray<f32>;
const D: usize = 2;
const H: usize = 4;
const Q: usize = 96;
const C: usize = 2;
const STD: f64 = 0.30;
const DEFER: [f64; 2] = [0.10, 0.25];
const AFFINE_AGREEMENT_TOLERANCE: f64 = 2e-5;
const W: [[f64; H]; D] = [[1.10, -0.70, 0.55, 0.30], [-0.35, 0.90, 0.65, -1.00]];
const B: [f64; H] = [0.10, -0.20, 0.15, 0.05];
const E: [[f64; C]; H] = [[0.9, -0.6], [-0.4, 0.8], [0.7, 0.2], [-0.5, -0.9]];

#[derive(Clone, Copy)]
struct Config {
    reps: usize,
    reference: usize,
    heldout: usize,
    sampled: usize,
}
const DEFAULT: Config = Config {
    reps: 8,
    reference: 2048,
    heldout: 64,
    sampled: 32,
};
const STUDY: Config = Config {
    reps: 30,
    reference: 16384,
    heldout: 256,
    sampled: 128,
};
const GENERALIZE: Config = Config {
    reps: 3,
    reference: 4096,
    heldout: 192,
    sampled: 96,
};
const GENERALIZE_QUICK: Config = Config {
    reps: 1,
    reference: 1024,
    heldout: 64,
    sampled: 32,
};
const GENERALIZE_MODELS: usize = 30;
const GENERALIZE_QUICK_MODELS: usize = 3;
const GENERALIZE_STDS: [f64; 3] = [0.15, 0.30, 0.45];

/// Host RNG keeps all evaluation streams independent of Burn's RNG.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
}

/// Assigns each replicate/role pair a distinct 64-bit stream identifier. The
/// upper 32 bits name the role and the lower 32 bits name the fixture.
fn stream_seed(role: u32, replicate: usize) -> u64 {
    assert!(
        replicate <= u32::MAX as usize,
        "replicate index exceeds seed space"
    );
    ((role as u64) << 32) | replicate as u64
}

fn tensor2(data: Vec<f32>, shape: [usize; 2], dev: &Device<Nd>) -> Tensor<Nd, 2> {
    Tensor::from_data(TensorData::new(data, shape), dev)
}
fn tensor1(data: Vec<f32>, dev: &Device<Nd>) -> Tensor<Nd, 1> {
    let n = data.len();
    Tensor::from_data(TensorData::new(data, [n]), dev)
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

/// The point network breaks score ties in favor of candidate zero.
fn deterministic_flip_probability(point_winner: usize, margin: f64) -> f64 {
    if margin < 0.0 {
        1.0
    } else if margin > 0.0 {
        0.0
    } else {
        f64::from(point_winner != 0)
    }
}

fn scores(x: [f64; D], relu: bool) -> [f64; C] {
    let mut h = [0.0; H];
    for j in 0..H {
        h[j] = B[j] + (0..D).map(|d| x[d] * W[d][j]).sum::<f64>();
        if relu {
            h[j] = h[j].max(0.0);
        }
    }
    let mut out = [0.0; C];
    for c in 0..C {
        out[c] = (0..H).map(|j| h[j] * E[j][c]).sum();
    }
    out
}
fn winner(scores: [f64; C]) -> (usize, f64) {
    let w = usize::from(scores[0] < scores[1]);
    (w, scores[w] - scores[1 - w])
}
fn queries(rep: usize) -> Vec<[f64; D]> {
    let mut rng = Rng::new(stream_seed(1, rep));
    (0..Q)
        .map(|i| {
            let t = (i as f64 + 0.5) / Q as f64;
            [
                2.0 * t - 1.0 + 0.12 * rng.normal(),
                (6.0 * t - 3.0).sin() + 0.12 * rng.normal(),
            ]
        })
        .collect()
}
fn sampled_rates(query: &[[f64; D]], relu: bool, draws: usize, seed: u64) -> Vec<f64> {
    let point: Vec<_> = query.iter().map(|&x| winner(scores(x, relu)).0).collect();
    let mut count = vec![0usize; Q];
    let mut rng = Rng::new(seed);
    for _ in 0..draws {
        for (i, &x) in query.iter().enumerate() {
            let noisy = [x[0] + STD * rng.normal(), x[1] + STD * rng.normal()];
            count[i] += usize::from(winner(scores(noisy, relu)).0 != point[i]);
        }
    }
    count.into_iter().map(|n| n as f64 / draws as f64).collect()
}

fn moment_probabilities(
    query: &[[f64; D]],
    relu: bool,
    dev: &Device<Nd>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let x = tensor2(
        query
            .iter()
            .flat_map(|v| v.iter().map(|&x| x as f32))
            .collect(),
        [Q, D],
        dev,
    );
    let w = tensor2(W.iter().flatten().map(|&x| x as f32).collect(), [D, H], dev);
    let b = tensor1(B.iter().map(|&x| x as f32).collect(), dev);
    let e = tensor2(E.iter().flatten().map(|&x| x as f32).collect(), [H, C], dev);
    let input = MomentsFull::from_diagonal(x, Tensor::<Nd, 2>::full([Q, D], STD * STD, dev));
    let hidden = propagate_linear_full(&input, w, Some(b));
    let output = if relu {
        propagate_linear_full(&propagate_relu_full(&hidden), e, None)
    } else {
        propagate_linear_full(&hidden, e, None)
    };
    let mean: Vec<f64> = output
        .mean
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let cov: Vec<f64> = output
        .cov
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    validate_score_covariance(&cov);
    let point: Vec<_> = query.iter().map(|&x| winner(scores(x, relu))).collect();
    let winners: Vec<_> = point.iter().map(|p| p.0).collect();
    let rank_only: Vec<_> = point.iter().map(|p| -p.1.abs()).collect();
    let full = margin_probability(&mean, &cov, &winners, false, dev);
    let dropped = margin_probability(&mean, &cov, &winners, true, dev);
    let local = local_probability(query, &winners, relu, dev);
    for (name, probability) in [
        ("full-covariance", &full),
        ("dropped-covariance", &dropped),
        ("local-Jacobian", &local),
    ] {
        validate_probability(name, probability);
    }
    (full, dropped, local, rank_only)
}

fn validate_score_covariance(cov: &[f64]) {
    assert_eq!(cov.len(), Q * C * C, "score covariance shape changed");
    assert!(
        cov.iter().all(|value| value.is_finite()),
        "score covariance must be finite"
    );
    let max_asymmetry = (0..Q)
        .map(|i| {
            let base = i * C * C;
            (cov[base + 1] - cov[base + C]).abs()
        })
        .fold(0.0, f64::max);
    assert!(
        max_asymmetry < 1e-5,
        "output covariance must be symmetric; max asymmetry {max_asymmetry}"
    );
}

fn validate_probability(name: &str, probability: &[f64]) {
    assert!(
        probability
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
        "{name} produced an invalid flip probability"
    );
}
fn margin_probability(
    mean: &[f64],
    cov: &[f64],
    winners: &[usize],
    drop_cross: bool,
    dev: &Device<Nd>,
) -> Vec<f64> {
    let mut out = vec![0.0; Q];
    let mut ix = Vec::new();
    let mut z = Vec::new();
    for (i, &w) in winners.iter().enumerate() {
        let l = 1 - w;
        let base = i * C * C;
        let m = mean[i * C + w] - mean[i * C + l];
        let mut v = cov[base + w * C + w] + cov[base + l * C + l];
        if !drop_cross {
            v -= cov[base + w * C + l] + cov[base + l * C + w];
        }
        let scale = (cov[base + w * C + w].abs() + cov[base + l * C + l].abs()).max(1.0);
        assert!(
            v >= -1e-5 * scale,
            "materially negative score-margin variance: {v}"
        );
        if v <= 0.0 {
            out[i] = deterministic_flip_probability(w, m);
        } else {
            ix.push(i);
            z.push((-m / v.sqrt()) as f32);
        }
    }
    if !ix.is_empty() {
        for (i, p) in ix.into_iter().zip(cdf(z, dev)) {
            out[i] = p;
        }
    }
    out
}

/// A one-unit ranking fixture whose propagated moments are exact, while its
/// ReLU margin is intentionally non-Gaussian. It separates the final Gaussian
/// margin-tail approximation from covariance-series truncation.
struct ExactReluMarginControl {
    relu_mean: f64,
    relu_variance: f64,
    margin_mean: f64,
    margin_variance: f64,
    exact_flip: f64,
    gaussian_proxy: f64,
}

fn exact_relu_margin_control(dev: &Device<Nd>) -> ExactReluMarginControl {
    let bias = 0.1f64;
    let input = MomentsFull::from_diagonal(
        tensor2(vec![0.0], [1, 1], dev),
        Tensor::<Nd, 2>::full([1, 1], 1.0, dev),
    );
    let rectified = propagate_relu_full(&input);
    let relu_mean = f64::from(rectified.mean.clone().to_data().to_vec::<f32>().unwrap()[0]);
    let relu_variance = f64::from(rectified.cov.clone().to_data().to_vec::<f32>().unwrap()[0]);
    // Scores are (bias, ReLU(X)), so the point winner is candidate zero.
    let scores = propagate_linear_full(
        &rectified,
        tensor2(vec![0.0, 1.0], [1, 2], dev),
        Some(tensor1(vec![bias as f32, 0.0], dev)),
    );
    let mean = scores.mean.to_data().to_vec::<f32>().unwrap();
    let covariance = scores.cov.to_data().to_vec::<f32>().unwrap();
    let margin_mean = f64::from(mean[0] - mean[1]);
    let margin_variance = f64::from(covariance[0] + covariance[3] - covariance[1] - covariance[2]);
    ExactReluMarginControl {
        relu_mean,
        relu_variance,
        margin_mean,
        margin_variance,
        // ReLU(X) > bias iff X > bias. bias > 0 avoids a threshold tie.
        exact_flip: cdf(vec![-bias as f32], dev)[0],
        gaussian_proxy: cdf(vec![(-margin_mean / margin_variance.sqrt()) as f32], dev)[0],
    }
}

fn report_exact_relu_margin_control(dev: &Device<Nd>) {
    let control = exact_relu_margin_control(dev);
    let gap = control.gaussian_proxy - control.exact_flip;
    println!("ReLU margin with exact moments:");
    println!(
        "  exact flip {:.4}, Gaussian margin proxy {:.4}, proxy gap {gap:+.4}",
        control.exact_flip, control.gaussian_proxy
    );
    println!(
        "  ReLU mean {:.4}, variance {:.4}; margin mean {:.4}, variance {:.4}",
        control.relu_mean, control.relu_variance, control.margin_mean, control.margin_variance
    );
    println!(
        "  one hidden coordinate, one ReLU: exact moments do not determine this non-Gaussian tail"
    );
}

fn local_probability(
    query: &[[f64; D]],
    winners: &[usize],
    relu: bool,
    dev: &Device<Nd>,
) -> Vec<f64> {
    let mut out = vec![0.0; Q];
    let mut ix = Vec::new();
    let mut z = Vec::with_capacity(Q);
    for (i, (&x, &w)) in query.iter().zip(winners).enumerate() {
        let (point_winner, margin) = winner(scores(x, relu));
        assert_eq!(
            w, point_winner,
            "point winner changed during local propagation"
        );
        let l = 1 - w;
        let mut variance = 0.0;
        for weight_row in &W {
            let mut derivative = 0.0;
            for h in 0..H {
                let pre = B[h] + (0..D).map(|j| x[j] * W[j][h]).sum::<f64>();
                derivative += weight_row[h]
                    * if relu && pre <= 0.0 { 0.0 } else { 1.0 }
                    * (E[h][w] - E[h][l]);
            }
            variance += STD * STD * derivative * derivative;
        }
        if variance <= 0.0 {
            out[i] = deterministic_flip_probability(w, margin);
        } else {
            ix.push(i);
            z.push((-margin / variance.sqrt()) as f32);
        }
    }
    if !ix.is_empty() {
        for (i, probability) in ix.into_iter().zip(cdf(z, dev)) {
            out[i] = probability;
        }
    }
    out
}

/// A generated fixture has the same small architecture as the walkthrough,
/// but its weights, candidates and query locations are independently seeded.
/// These are untrained random networks: the study measures propagation
/// sensitivity under the stated feature-noise laws, not learned ranking value.
#[derive(Clone)]
struct Model {
    w: [[f64; H]; D],
    b: [f64; H],
    e: [[f64; C]; H],
}

fn generated_model(fixture: usize) -> Model {
    let mut rng = Rng::new(stream_seed(20, fixture));
    let mut model = Model {
        w: [[0.0; H]; D],
        b: [0.0; H],
        e: [[0.0; C]; H],
    };
    for row in &mut model.w {
        for value in row {
            *value = 0.75 * rng.normal();
        }
    }
    for value in &mut model.b {
        *value = 0.20 * rng.normal();
    }
    for row in &mut model.e {
        for value in row {
            *value = 0.75 * rng.normal();
        }
    }
    model
}

fn model_scores(model: &Model, x: [f64; D], relu: bool) -> [f64; C] {
    let mut hidden = [0.0; H];
    for (j, value) in hidden.iter_mut().enumerate() {
        *value = model.b[j] + (0..D).map(|d| x[d] * model.w[d][j]).sum::<f64>();
        if relu {
            *value = value.max(0.0);
        }
    }
    std::array::from_fn(|c| (0..H).map(|j| hidden[j] * model.e[j][c]).sum())
}

fn generated_queries(fixture: usize, set: usize) -> Vec<[f64; D]> {
    let mut rng = Rng::new(stream_seed(21 + set as u32, fixture));
    (0..Q)
        .map(|i| {
            let t = (i as f64 + 0.5) / Q as f64;
            [
                2.4 * t - 1.2 + 0.20 * rng.normal(),
                (7.0 * t - 3.5).sin() + 0.20 * rng.normal(),
            ]
        })
        .collect()
}

fn sampled_rates_model(
    model: &Model,
    query: &[[f64; D]],
    relu: bool,
    std: f64,
    draws: usize,
    seed: u64,
) -> Vec<f64> {
    let point: Vec<_> = query
        .iter()
        .map(|&x| winner(model_scores(model, x, relu)).0)
        .collect();
    let mut count = vec![0usize; Q];
    let mut rng = Rng::new(seed);
    for _ in 0..draws {
        for (i, &x) in query.iter().enumerate() {
            let noisy = [x[0] + std * rng.normal(), x[1] + std * rng.normal()];
            count[i] += usize::from(winner(model_scores(model, noisy, relu)).0 != point[i]);
        }
    }
    count.into_iter().map(|n| n as f64 / draws as f64).collect()
}

fn moment_probabilities_model(
    model: &Model,
    query: &[[f64; D]],
    relu: bool,
    std: f64,
    dev: &Device<Nd>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let x = tensor2(
        query
            .iter()
            .flat_map(|v| v.iter().map(|&v| v as f32))
            .collect(),
        [Q, D],
        dev,
    );
    let w = tensor2(
        model.w.iter().flatten().map(|&v| v as f32).collect(),
        [D, H],
        dev,
    );
    let b = tensor1(model.b.iter().map(|&v| v as f32).collect(), dev);
    let e = tensor2(
        model.e.iter().flatten().map(|&v| v as f32).collect(),
        [H, C],
        dev,
    );
    let input = MomentsFull::from_diagonal(x, Tensor::<Nd, 2>::full([Q, D], std * std, dev));
    let hidden = propagate_linear_full(&input, w, Some(b));
    let output = if relu {
        propagate_linear_full(&propagate_relu_full(&hidden), e, None)
    } else {
        propagate_linear_full(&hidden, e, None)
    };
    let mean: Vec<f64> = output
        .mean
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let cov: Vec<f64> = output
        .cov
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    validate_score_covariance(&cov);
    let point: Vec<_> = query
        .iter()
        .map(|&x| winner(model_scores(model, x, relu)))
        .collect();
    let winners: Vec<_> = point.iter().map(|point| point.0).collect();
    let rank_only: Vec<_> = point.iter().map(|point| -point.1.abs()).collect();
    let full = margin_probability(&mean, &cov, &winners, false, dev);
    let dropped = margin_probability(&mean, &cov, &winners, true, dev);
    let local = local_probability_model(model, query, &winners, relu, std, dev);
    for (name, probability) in [
        ("full-covariance", &full),
        ("dropped-covariance", &dropped),
        ("local-Jacobian", &local),
    ] {
        validate_probability(name, probability);
    }
    (full, dropped, local, rank_only)
}

fn local_probability_model(
    model: &Model,
    query: &[[f64; D]],
    winners: &[usize],
    relu: bool,
    std: f64,
    dev: &Device<Nd>,
) -> Vec<f64> {
    let mut out = vec![0.0; Q];
    let mut ix = Vec::new();
    let mut z = Vec::with_capacity(Q);
    for (i, (&x, &w)) in query.iter().zip(winners).enumerate() {
        let (point_winner, margin) = winner(model_scores(model, x, relu));
        assert_eq!(
            w, point_winner,
            "point winner changed during local propagation"
        );
        let l = 1 - w;
        let mut variance = 0.0;
        for weight_row in &model.w {
            let mut derivative = 0.0;
            for (h, &weight) in weight_row.iter().enumerate() {
                let pre = model.b[h] + (0..D).map(|j| x[j] * model.w[j][h]).sum::<f64>();
                derivative += weight
                    * if relu && pre <= 0.0 { 0.0 } else { 1.0 }
                    * (model.e[h][w] - model.e[h][l]);
            }
            variance += std * std * derivative * derivative;
        }
        if variance <= 0.0 {
            out[i] = deterministic_flip_probability(w, margin);
        } else {
            ix.push(i);
            z.push((-margin / variance.sqrt()) as f32);
        }
    }
    for (i, probability) in ix.into_iter().zip(cdf(z, dev)) {
        out[i] = probability;
    }
    out
}

fn mae(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / Q as f64
}
/// Averaged binary Brier scores over held-out draws, expressed through the rate.
fn brier(p: &[f64], heldout: &[f64]) -> f64 {
    p.iter()
        .zip(heldout)
        .map(|(&p, &y)| p * p * (1.0 - y) + (1.0 - p) * (1.0 - p) * y)
        .sum::<f64>()
        / Q as f64
}
fn policy(risk: &[f64], heldout: &[f64]) -> ([f64; 2], [f64; 2]) {
    let mut retained = [0.0; 2];
    let mut avoided = [0.0; 2];
    for (slot, rate) in DEFER.iter().enumerate() {
        let n = deferred_count(*rate);
        let mut order: Vec<_> = (0..Q).collect();
        order.sort_by(|&a, &b| {
            risk[b]
                .partial_cmp(&risk[a])
                .unwrap()
                .then_with(|| a.cmp(&b))
        });
        let mut defer = [false; Q];
        for &i in order.iter().take(n) {
            defer[i] = true;
        }
        retained[slot] = heldout
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| (!defer[i]).then_some(x))
            .sum::<f64>()
            / (Q - n) as f64;
        avoided[slot] = heldout
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| defer[i].then_some(x))
            .sum::<f64>()
            / n as f64;
    }
    (retained, avoided)
}

fn deferred_count(rate: f64) -> usize {
    (rate * Q as f64).round() as usize
}
fn mean_se(values: &[f64]) -> (f64, f64) {
    let m = values.iter().sum::<f64>() / values.len() as f64;
    let v = values.iter().map(|x| (x - m).powi(2)).sum::<f64>()
        / values.len().saturating_sub(1).max(1) as f64;
    (m, (v / values.len() as f64).sqrt())
}
#[derive(Default)]
struct Rows {
    mae: Vec<Option<f64>>,
    brier: Vec<Option<f64>>,
    retained: [Vec<f64>; 2],
    avoided: [Vec<f64>; 2],
}
fn report(name: &str, row: &Rows) {
    let m: Vec<_> = row.mae.iter().flatten().copied().collect();
    let b: Vec<_> = row.brier.iter().flatten().copied().collect();
    let mae = if m.is_empty() {
        "rank only".to_owned()
    } else {
        let (x, se) = mean_se(&m);
        format!("{x:.4} ± {se:.4}")
    };
    let brier = if b.is_empty() {
        "not probabilistic".to_owned()
    } else {
        let (x, se) = mean_se(&b);
        format!("{x:.4} ± {se:.4}")
    };
    println!("  {name:<25} {mae:>15} {brier:>18}");
    for (i, &rate) in DEFER.iter().enumerate() {
        let (r, rs) = mean_se(&row.retained[i]);
        let (a, as_) = mean_se(&row.avoided[i]);
        println!(
            "    defer {}/{} ({:.1}%): retained {r:.4} ± {rs:.4}; avoided/defer {a:.4} ± {as_:.4}",
            deferred_count(rate),
            Q,
            100.0 * deferred_count(rate) as f64 / Q as f64,
        );
    }
}

fn report_generalized(name: &str, row: &Rows, uncertainty: bool) {
    let show = |values: &[f64]| {
        let (mean, se) = mean_se(values);
        if uncertainty {
            format!("{mean:.4} ± {se:.4}")
        } else {
            format!("{mean:.4}")
        }
    };
    let mae: Vec<_> = row.mae.iter().flatten().copied().collect();
    let brier: Vec<_> = row.brier.iter().flatten().copied().collect();
    println!(
        "  {name:<25} {:>15} {:>18}",
        if mae.is_empty() {
            "rank only".into()
        } else {
            show(&mae)
        },
        if brier.is_empty() {
            "not probabilistic".into()
        } else {
            show(&brier)
        }
    );
    for (i, &rate) in DEFER.iter().enumerate() {
        println!(
            "    defer {}/{} ({:.1}%): retained {}; avoided/defer {}",
            deferred_count(rate),
            Q,
            100.0 * deferred_count(rate) as f64 / Q as f64,
            show(&row.retained[i]),
            show(&row.avoided[i])
        );
    }
}

fn report_contrast(name: &str, full: &Rows, other: &Rows, uncertainty: bool) {
    let difference = |left: &[f64], right: &[f64]| -> Vec<f64> {
        left.iter().zip(right).map(|(a, b)| a - b).collect()
    };
    let show = |values: Vec<f64>| {
        let (mean, se) = mean_se(&values);
        if uncertainty {
            format!("{mean:+.4} ± {se:.4}")
        } else {
            format!("{mean:+.4}")
        }
    };
    let full_mae: Vec<_> = full.mae.iter().flatten().copied().collect();
    let other_mae: Vec<_> = other.mae.iter().flatten().copied().collect();
    let full_brier: Vec<_> = full.brier.iter().flatten().copied().collect();
    let other_brier: Vec<_> = other.brier.iter().flatten().copied().collect();
    let mae = if full_mae.is_empty() || other_mae.is_empty() {
        "n/a".into()
    } else {
        show(difference(&full_mae, &other_mae))
    };
    let brier = if full_brier.is_empty() || other_brier.is_empty() {
        "n/a".into()
    } else {
        show(difference(&full_brier, &other_brier))
    };
    let primary = 1;
    println!("  full minus {name:<20} MAE {mae:>14}; Brier {brier:>14}; retained {:+.4}{}; avoided {:+.4}{}",
        mean_se(&difference(&full.retained[primary], &other.retained[primary])).0,
        if uncertainty { format!(" ± {:.4}", mean_se(&difference(&full.retained[primary], &other.retained[primary])).1) } else { String::new() },
        mean_se(&difference(&full.avoided[primary], &other.avoided[primary])).0,
        if uncertainty { format!(" ± {:.4}", mean_se(&difference(&full.avoided[primary], &other.avoided[primary])).1) } else { String::new() },
    );
}

fn append_fixture_mean(destination: &mut Rows, source: &Rows) {
    destination
        .mae
        .push(if source.mae.iter().all(Option::is_none) {
            None
        } else {
            Some(mean_se(&source.mae.iter().flatten().copied().collect::<Vec<_>>()).0)
        });
    destination
        .brier
        .push(if source.brier.iter().all(Option::is_none) {
            None
        } else {
            Some(mean_se(&source.brier.iter().flatten().copied().collect::<Vec<_>>()).0)
        });
    for i in 0..DEFER.len() {
        destination.retained[i].push(mean_se(&source.retained[i]).0);
        destination.avoided[i].push(mean_se(&source.avoided[i]).0);
    }
}

fn report_regime_policy_contrasts(rows: &[Rows; 5], std: f64, uncertainty: bool) {
    let difference = |left: &[f64], right: &[f64]| -> (f64, f64) {
        mean_se(
            &left
                .iter()
                .zip(right)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        )
    };
    println!("  std {std:.2}:");
    for (name, other) in [
        "dropped covariance",
        "local Jacobian",
        "sampled score",
        "point margin",
    ]
    .iter()
    .zip(rows.iter().skip(1))
    {
        let (retained, retained_se) = difference(&rows[0].retained[1], &other.retained[1]);
        let (avoided, avoided_se) = difference(&rows[0].avoided[1], &other.avoided[1]);
        if uncertainty {
            println!("    full minus {name:<20}: retained {retained:+.4} ± {retained_se:.4}; avoided {avoided:+.4} ± {avoided_se:.4}");
        } else {
            println!("    full minus {name:<20}: retained {retained:+.4}; avoided {avoided:+.4}");
        }
    }
}

fn generalization_study(config: Config, models: usize, quick: bool, dev: &Device<Nd>) {
    let started = Instant::now();
    let mut rows: [Rows; 5] = std::array::from_fn(|_| Rows::default());
    let mut rows_by_regime: Vec<[Rows; 5]> = (0..GENERALIZE_STDS.len())
        .map(|_| std::array::from_fn(|_| Rows::default()))
        .collect();
    let mut affine_error = Vec::new();
    for fixture in 0..models {
        let model = generated_model(fixture);
        let mut within: [Rows; 5] = std::array::from_fn(|_| Rows::default());
        let mut within_by_regime: Vec<[Rows; 5]> = (0..GENERALIZE_STDS.len())
            .map(|_| std::array::from_fn(|_| Rows::default()))
            .collect();
        for set in 0..config.reps {
            let query = generated_queries(fixture, set);
            for (regime, &std) in GENERALIZE_STDS.iter().enumerate() {
                let (full, dropped, local, margin) =
                    moment_probabilities_model(&model, &query, true, std, dev);
                let role = 30 + (set * GENERALIZE_STDS.len() + regime) as u32 * 3;
                let reference = sampled_rates_model(
                    &model,
                    &query,
                    true,
                    std,
                    config.reference,
                    stream_seed(role, fixture),
                );
                let heldout = sampled_rates_model(
                    &model,
                    &query,
                    true,
                    std,
                    config.heldout,
                    stream_seed(role + 1, fixture),
                );
                let sampled = sampled_rates_model(
                    &model,
                    &query,
                    true,
                    std,
                    config.sampled,
                    stream_seed(role + 2, fixture),
                );
                for (name, probability) in [
                    ("reference", &reference),
                    ("held-out", &heldout),
                    ("sampled", &sampled),
                ] {
                    validate_probability(name, probability);
                }
                for (index, (probability, probabilistic)) in [
                    (full, true),
                    (dropped, true),
                    (local, true),
                    (sampled, true),
                    (margin, false),
                ]
                .into_iter()
                .enumerate()
                {
                    let (retained, avoided) = policy(&probability, &heldout);
                    within[index]
                        .mae
                        .push(probabilistic.then(|| mae(&probability, &reference)));
                    within[index]
                        .brier
                        .push(probabilistic.then(|| brier(&probability, &heldout)));
                    for i in 0..DEFER.len() {
                        within[index].retained[i].push(retained[i]);
                        within[index].avoided[i].push(avoided[i]);
                        within_by_regime[regime][index].retained[i].push(retained[i]);
                        within_by_regime[regime][index].avoided[i].push(avoided[i]);
                    }
                    within_by_regime[regime][index]
                        .mae
                        .push(probabilistic.then(|| mae(&probability, &reference)));
                    within_by_regime[regime][index]
                        .brier
                        .push(probabilistic.then(|| brier(&probability, &heldout)));
                }
            }
        }
        // This affine control is meaningful for each independently generated
        // model: full propagation and its local Jacobian must agree exactly.
        let query = generated_queries(fixture, 0);
        let (affine_full, _, affine_local, _) =
            moment_probabilities_model(&model, &query, false, GENERALIZE_STDS[1], dev);
        let max_difference = affine_full
            .iter()
            .zip(affine_local)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_difference <= AFFINE_AGREEMENT_TOLERANCE,
            "generated affine full/local probabilities disagree by {max_difference}"
        );
        affine_error.push(max_difference);
        for (destination, source) in rows.iter_mut().zip(&within) {
            append_fixture_mean(destination, source);
        }
        for (destination_set, source_set) in rows_by_regime.iter_mut().zip(&within_by_regime) {
            for (destination, source) in destination_set.iter_mut().zip(source_set) {
                append_fixture_mean(destination, source);
            }
        }
    }
    println!("pairwise ranking generalization under Gaussian feature noise:");
    println!("  {models} independently generated untrained model/candidate fixtures; {} query sets/model; std regimes {:?}", config.reps, GENERALIZE_STDS);
    println!(
        "  reference {}, held-out {}, sampled-policy {} draws/query; all streams are independent",
        config.reference, config.heldout, config.sampled
    );
    println!("  each displayed model-fixture value is an equal-weight average over the three fixed noise regimes and nested query sets; the 25% budget is primary and 10% is secondary");
    println!(
        "  {:<25} {:>15} {:>18}",
        "policy", "probability MAE", "Brier"
    );
    for (name, row) in [
        "full score covariance",
        "dropped score covariance",
        "local Jacobian",
        "sampled score",
        "point margin rank",
    ]
    .iter()
    .zip(&rows)
    {
        report_generalized(name, row, !quick);
    }
    println!("\nper-regime paired policy contrasts at the primary 25% deferral budget (full minus comparator):");
    for (std, regime_rows) in GENERALIZE_STDS.iter().zip(&rows_by_regime) {
        report_regime_policy_contrasts(regime_rows, *std, !quick);
    }
    println!("\npaired contrasts at the model-fixture unit, 25% deferral (full minus comparator):");
    for (name, row) in [
        "dropped covariance",
        "local Jacobian",
        "sampled score",
        "point margin",
    ]
    .iter()
    .zip(rows.iter().skip(1))
    {
        report_contrast(name, &rows[0], row, !quick);
    }
    let (affine_mean, affine_se) = mean_se(&affine_error);
    if quick {
        println!("\naffine control: mean per-model maximum full/local difference {affine_mean:.2e} (descriptive quick run)");
    } else {
        println!("\naffine control: mean per-model maximum full/local difference {affine_mean:.2e} ± {affine_se:.2e} (SE across model fixtures)");
    }
    println!("MC reference and held-out draws still add finite-sampling error; model-fixture SEs do not isolate it. These untrained generated networks establish only synthetic input-noise sensitivity, not trained-model ranking performance.");
    let elapsed = started.elapsed();
    println!("end-to-end wall time {:.2}s ({:.2}s/model): fixture generation, analytic propagation, and all MC streams are included; compilation is excluded, so this is workload context rather than a per-method speed comparison.", elapsed.as_secs_f64(), elapsed.as_secs_f64() / models as f64);
}

fn main() {
    let mut study = false;
    let mut generalize = false;
    let mut quick = false;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--study" => study = true,
            "--generalize" => generalize = true,
            "--quick" => quick = true,
            _ => {
                panic!("unknown argument {argument:?}; expected --study, --generalize, or --quick")
            }
        }
    }
    assert!(!quick || generalize, "--quick requires --generalize");
    let config = if study { STUDY } else { DEFAULT };
    let dev = Device::<Nd>::default();
    report_exact_relu_margin_control(&dev);
    if generalize {
        assert!(
            !study,
            "--study is the fixed-model protocol; --generalize already runs its 30-model study"
        );
        let config = if quick { GENERALIZE_QUICK } else { GENERALIZE };
        let models = if quick {
            GENERALIZE_QUICK_MODELS
        } else {
            GENERALIZE_MODELS
        };
        generalization_study(config, models, quick, &dev);
        return;
    }
    let mut rows: [Rows; 5] = std::array::from_fn(|_| Rows::default());
    let mut affine_full = Vec::new();
    let mut affine_repeat = Vec::new();
    let mut affine_local = Vec::new();
    for rep in 0..config.reps {
        let query = queries(rep);
        let (full, dropped, local, margin) = moment_probabilities(&query, true, &dev);
        let reference = sampled_rates(&query, true, config.reference, stream_seed(2, rep));
        let heldout = sampled_rates(&query, true, config.heldout, stream_seed(3, rep));
        let sampled = sampled_rates(&query, true, config.sampled, stream_seed(4, rep));
        for (name, probability) in [
            ("reference", &reference),
            ("held-out", &heldout),
            ("sampled", &sampled),
        ] {
            validate_probability(name, probability);
        }
        for (i, (probability, probabilistic)) in [
            (full, true),
            (dropped, true),
            (local, true),
            (sampled, true),
            (margin, false),
        ]
        .into_iter()
        .enumerate()
        {
            let risk = probability.clone();
            let (retained, avoided) = policy(&risk, &heldout);
            rows[i]
                .mae
                .push(probabilistic.then(|| mae(&probability, &reference)));
            rows[i]
                .brier
                .push(probabilistic.then(|| brier(&probability, &heldout)));
            for j in 0..2 {
                rows[i].retained[j].push(retained[j]);
                rows[i].avoided[j].push(avoided[j]);
            }
        }
        let (a_full, _, a_local, _) = moment_probabilities(&query, false, &dev);
        let a_ref = sampled_rates(&query, false, config.reference, stream_seed(5, rep));
        let a_repeat = sampled_rates(&query, false, config.reference, stream_seed(6, rep));
        validate_probability("affine reference", &a_ref);
        validate_probability("affine repeat", &a_repeat);
        affine_full.push(mae(&a_full, &a_ref));
        affine_repeat.push(mae(&a_ref, &a_repeat));
        let local_difference = a_full
            .iter()
            .zip(a_local)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(
            local_difference <= AFFINE_AGREEMENT_TOLERANCE,
            "affine full/local probabilities disagree by {local_difference}"
        );
        affine_local.push(local_difference);
    }
    println!("pairwise ranking abstention under Gaussian feature noise std {STD}:");
    println!(
        "  {} independent query fixtures; reference {}, held-out {}, sampled-policy {} draws/query",
        config.reps, config.reference, config.heldout, config.sampled
    );
    println!("  probability MAE uses the independent high-draw reference; Brier and policy outcomes use held-out draws");
    println!(
        "  {:<25} {:>15} {:>18}",
        "policy", "probability MAE", "Brier"
    );
    for (name, row) in [
        "full score covariance",
        "dropped score covariance",
        "local Jacobian",
        "sampled score",
        "point margin rank",
    ]
    .iter()
    .zip(&rows)
    {
        report(name, row);
    }
    let (fm, fs) = mean_se(&affine_full);
    let (rm, rs) = mean_se(&affine_repeat);
    let (lm, ls) = mean_se(&affine_local);
    println!("\naffine exact-Gaussian control:");
    println!("  full analytic MAE vs reference = {fm:.4} ± {fs:.4}");
    println!("  independent reference-repeat MAE = {rm:.4} ± {rs:.4}");
    println!("  max full/local probability difference = {lm:.2e} ± {ls:.2e}");
    println!("\nUncertainty is mean ± SE over query fixtures, not over Monte-Carlo draws.");
}

#[cfg(test)]
mod tests {
    use super::{
        deferred_count, deterministic_flip_probability, exact_relu_margin_control,
        margin_probability, Device, Nd, DEFER, Q,
    };

    #[test]
    fn deterministic_tie_uses_candidate_zero() {
        assert_eq!(deterministic_flip_probability(0, 0.0), 0.0);
        assert_eq!(deterministic_flip_probability(1, 0.0), 1.0);
        assert_eq!(deterministic_flip_probability(0, -1.0), 1.0);
        assert_eq!(deterministic_flip_probability(1, 1.0), 0.0);
    }

    #[test]
    fn matched_defer_counts_use_the_reported_rounding() {
        assert_eq!(deferred_count(DEFER[0]), 10);
        assert_eq!(deferred_count(DEFER[1]), 24);
    }

    #[test]
    fn exact_relu_margin_control_isolates_non_gaussian_tail_shape() {
        let dev = Device::<Nd>::default();
        let control = exact_relu_margin_control(&dev);
        let expected_relu_mean = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let expected_relu_variance = 0.5 - 1.0 / (2.0 * std::f64::consts::PI);
        assert!((control.relu_mean - expected_relu_mean).abs() < 2e-6);
        assert!((control.relu_variance - expected_relu_variance).abs() < 2e-6);
        assert!((control.margin_mean - (0.1 - expected_relu_mean)).abs() < 2e-6);
        assert!((control.margin_variance - expected_relu_variance).abs() < 2e-6);
        assert!((control.exact_flip - 0.460_172_16).abs() < 2e-6);

        // Unlike a tail probability, expected squared deviation needs only
        // these two moments. Direct integration gives E[X_+^2] = 1/2.
        let squared_loss = control.margin_mean.powi(2) + control.margin_variance;
        let integrated_loss = 0.5 - 0.2 * expected_relu_mean + 0.01;
        assert!((squared_loss - integrated_loss).abs() < 2e-6);

        // Repeat the same real score moments through the ranking estimator.
        let mut mean = Vec::with_capacity(Q * 2);
        let mut covariance = Vec::with_capacity(Q * 4);
        for _ in 0..Q {
            mean.extend([0.1, control.relu_mean]);
            covariance.extend([0.0, 0.0, 0.0, control.relu_variance]);
        }
        let proxy = margin_probability(&mean, &covariance, &vec![0; Q], false, &dev)[0];
        assert!((proxy - control.gaussian_proxy).abs() < 2e-6);
        assert!((proxy - 0.695_690_55).abs() < 2e-6);
        assert!(proxy - control.exact_flip > 0.20);
    }
}
