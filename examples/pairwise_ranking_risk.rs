//! Compare matched abstention policies for pairwise ranking under Gaussian input noise.
//!
//! The outcome is a noisy winner differing from the point-network winner. This
//! is sensitivity under a supplied perturbation model, not relevance, reward,
//! or an acquisition value. Full score covariance, dropped score covariance,
//! a rank-only point margin, a local Jacobian, and sampled scores defer the same
//! query fraction. Reference, held-out, and sampled-policy draws are independent.
//!
//! Run: `cargo run --release --example pairwise_ranking_risk --features burn`
//! Add `-- --study` for 30 fixtures and larger Monte Carlo budgets.

use burn::tensor::{Device, Tensor, TensorData};
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{propagate_linear_full, propagate_relu_full, MomentsFull};

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

fn main() {
    let mut study = false;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--study" => study = true,
            _ => panic!("unknown argument {argument:?}; expected optional --study"),
        }
    }
    let config = if study { STUDY } else { DEFAULT };
    let dev = Device::<Nd>::default();
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
    use super::{deferred_count, deterministic_flip_probability, DEFER};

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
}
