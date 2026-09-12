//! Select among fixed actions when execution inputs have correlated Gaussian noise.
//!
//! A one-hidden-layer ReLU surrogate is trained on a deterministic quadratic
//! simulator.  Each method estimates the surrogate risk of every action under
//! an assumed covariance, then selects one action for each target.  Selection
//! is evaluated with the simulator's exact Gaussian quadratic-loss formula,
//! never with the samples used to rank candidates.
//!
//! The ratios below multiply a covariance, not a standard deviation.  Thus
//! `1.25` means 25% more covariance than the true execution law.  A method's
//! error against independent surrogate Monte Carlo, covariance misspecification,
//! and surrogate error against the simulator are printed separately.
//!
//! Run: `cargo run --release --features burn --example robust_selection`
//! Quick: `cargo run --release --features burn --example robust_selection -- --quick`
//! Paired fit-quality control: append `--fit-study` (fixed 500/2,000 epochs;
//! 80/320 with `--quick`).

use std::env;

use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{activation, Device, Tensor, TensorData};
use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_full, propagate_relu, propagate_relu_full, Moments,
    MomentsFull,
};

const HIDDEN: usize = 32;
const TRAIN_ROWS: usize = 1_024;
const HELDOUT_ROWS: usize = 256;
const TARGETS: [f64; 3] = [0.25, 0.75, 1.5];
const RATIOS: [f64; 3] = [0.75, 1.0, 1.25];
const TRUE_L: [[f64; 2]; 2] = [[0.20, 0.0], [0.12, 0.16]];

#[derive(Module, Debug)]
struct Mlp {
    first: Linear,
    last: Linear,
}
impl Mlp {
    fn init(device: &Device) -> Self {
        Self {
            first: LinearConfig::new(2, HIDDEN).init(device),
            last: LinearConfig::new(HIDDEN, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.last.forward(activation::relu(self.first.forward(x)))
    }
}

#[derive(Clone, Copy)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 2.0)
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
}

fn q(x: [f64; 2]) -> f64 {
    x[0] * x[0] + 0.5 * x[0] * x[1] + 0.5 * x[1] * x[1] + 0.2 * x[0] - 0.1 * x[1]
}
fn covariance(l: [[f64; 2]; 2], ratio: f64) -> [[f64; 2]; 2] {
    [
        [
            ratio * (l[0][0] * l[0][0] + l[0][1] * l[0][1]),
            ratio * (l[0][0] * l[1][0] + l[0][1] * l[1][1]),
        ],
        [
            ratio * (l[1][0] * l[0][0] + l[1][1] * l[0][1]),
            ratio * (l[1][0] * l[1][0] + l[1][1] * l[1][1]),
        ],
    ]
}
fn scaled_l(ratio: f64) -> [[f64; 2]; 2] {
    let scale = ratio.sqrt();
    [
        [TRUE_L[0][0] * scale, 0.0],
        [TRUE_L[1][0] * scale, TRUE_L[1][1] * scale],
    ]
}

/// Exact E[(h(X)-target)^2] for the analytic quadratic simulator, X~N(mu,Sigma).
fn oracle(mu: [f64; 2], sigma: [[f64; 2]; 2], target: f64) -> f64 {
    let qmu = q(mu);
    let trace_q_sigma = sigma[0][0] + 0.25 * (sigma[0][1] + sigma[1][0]) + 0.5 * sigma[1][1];
    let mean = qmu + trace_q_sigma;
    let g = [2.0 * mu[0] + 0.5 * mu[1] + 0.2, 0.5 * mu[0] + mu[1] - 0.1];
    let qs = [
        [
            sigma[0][0] + 0.25 * sigma[1][0],
            sigma[0][1] + 0.25 * sigma[1][1],
        ],
        [
            0.25 * sigma[0][0] + 0.5 * sigma[1][0],
            0.25 * sigma[0][1] + 0.5 * sigma[1][1],
        ],
    ];
    let trace_qsqs =
        qs[0][0] * qs[0][0] + qs[0][1] * qs[1][0] + qs[1][0] * qs[0][1] + qs[1][1] * qs[1][1];
    let var = 2.0 * trace_qsqs
        + g[0] * (sigma[0][0] * g[0] + sigma[0][1] * g[1])
        + g[1] * (sigma[1][0] * g[0] + sigma[1][1] * g[1]);
    (mean - target).powi(2) + var
}

fn candidates() -> Vec<[f64; 2]> {
    (-6..=6)
        .flat_map(|a| (-6..=6).map(move |b| [a as f64 * 0.2, b as f64 * 0.2]))
        .collect()
}
fn flat(points: &[[f64; 2]]) -> Vec<f32> {
    points
        .iter()
        .flat_map(|p| [p[0] as f32, p[1] as f32])
        .collect()
}

fn train_trajectory(
    repeat: usize,
    checkpoint: Option<usize>,
    epochs: usize,
    device: &Device,
) -> (Option<Mlp>, Mlp) {
    assert!(checkpoint.map_or(true, |epoch| epoch < epochs));
    let mut rng = Rng::new(0xF17_0000 + repeat as u64);
    let points: Vec<[f64; 2]> = (0..TRAIN_ROWS)
        .map(|_| [4.0 * rng.uniform() - 2.0, 4.0 * rng.uniform() - 2.0])
        .collect();
    let labels: Vec<f32> = points.iter().map(|&x| q(x) as f32).collect();
    let x = Tensor::<2>::from_data(TensorData::new(flat(&points), [TRAIN_ROWS, 2]), device);
    let y = Tensor::<2>::from_data(TensorData::new(labels, [TRAIN_ROWS, 1]), device);
    device.seed(0xA0D3_0000 + repeat as u64);
    let mut model = Mlp::init(device);
    let mut optimizer = AdamConfig::new().init();
    let mut snapshot = None;
    for epoch in 1..=epochs {
        let loss = MseLoss::new().forward(model.forward(x.clone()), y.clone(), Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(1e-3, model, grads);
        if checkpoint == Some(epoch) {
            snapshot = Some(model.clone().valid());
        }
    }
    (snapshot, model.valid())
}

fn train(repeat: usize, epochs: usize, device: &Device) -> Mlp {
    train_trajectory(repeat, None, epochs, device).1
}

/// Two frozen inference snapshots from one optimizer trajectory.  The epochs
/// are fixed before execution; neither checkpoint is selected by its results.
fn train_snapshots(
    repeat: usize,
    first_epoch: usize,
    last_epoch: usize,
    device: &Device,
) -> (Mlp, Mlp) {
    let (first, last) = train_trajectory(repeat, Some(first_epoch), last_epoch, device);
    (first.expect("first checkpoint"), last)
}

fn clean_rmse(model: &Mlp, repeat: usize, device: &Device) -> f64 {
    let mut rng = Rng::new(0xB31D_0000 + repeat as u64);
    let points: Vec<[f64; 2]> = (0..HELDOUT_ROWS)
        .map(|_| [4.0 * rng.uniform() - 2.0, 4.0 * rng.uniform() - 2.0])
        .collect();
    let output = model
        .forward(Tensor::<2>::from_data(
            TensorData::new(flat(&points), [HELDOUT_ROWS, 2]),
            device,
        ))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    (points
        .iter()
        .zip(output)
        .map(|(x, y)| (q(*x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / HELDOUT_ROWS as f64)
        .sqrt()
}

#[derive(Clone)]
struct Evaluation {
    loss: Vec<[f64; 3]>,
}
fn from_moments(mean: Vec<f64>, variance: Vec<f64>) -> Evaluation {
    assert_eq!(mean.len(), variance.len(), "moment shapes must match");
    assert!(
        !mean.is_empty()
            && mean.iter().all(|value| value.is_finite())
            && variance
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
        "method moments must be finite with nonnegative variance"
    );
    let loss = mean
        .iter()
        .zip(&variance)
        .map(|(m, v)| TARGETS.map(|t| (m - t).powi(2) + v))
        .collect();
    Evaluation { loss }
}
fn checked_variance(values: Vec<f32>) -> Vec<f64> {
    let values: Vec<f64> = values.into_iter().map(f64::from).collect();
    assert!(
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "propagated variances must be finite and nonnegative"
    );
    values
}
fn point_mean(model: &Mlp, candidates: &[[f64; 2]], device: &Device) -> Vec<f64> {
    model
        .forward(Tensor::<2>::from_data(
            TensorData::new(flat(candidates), [candidates.len(), 2]),
            device,
        ))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}
fn point(model: &Mlp, candidates: &[[f64; 2]], device: &Device) -> Evaluation {
    from_moments(
        point_mean(model, candidates, device),
        vec![0.0; candidates.len()],
    )
}
fn local(
    model: &Mlp,
    candidates: &[[f64; 2]],
    sigma: [[f64; 2]; 2],
    device: &Device,
) -> Evaluation {
    let mean = point_mean(model, candidates, device);
    let w1 = model
        .first
        .weight
        .val()
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let b1 = model
        .first
        .bias
        .as_ref()
        .unwrap()
        .val()
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let w2 = model
        .last
        .weight
        .val()
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let variance = candidates
        .iter()
        .map(|x| {
            let mut g = [0.0; 2];
            for h in 0..HIDDEN {
                if w1[h] as f64 * x[0] + w1[HIDDEN + h] as f64 * x[1] + b1[h] as f64 > 0.0 {
                    g[0] += w2[h] as f64 * w1[h] as f64;
                    g[1] += w2[h] as f64 * w1[HIDDEN + h] as f64;
                }
            }
            g[0] * (sigma[0][0] * g[0] + sigma[0][1] * g[1])
                + g[1] * (sigma[1][0] * g[0] + sigma[1][1] * g[1])
        })
        .collect();
    from_moments(mean, variance)
}

fn propagation(
    model: &Mlp,
    candidates: &[[f64; 2]],
    sigma: [[f64; 2]; 2],
    full: bool,
    device: &Device,
) -> Evaluation {
    let n = candidates.len();
    let mean = Tensor::<2>::from_data(TensorData::new(flat(candidates), [n, 2]), device);
    let covariance = Tensor::<3>::from_data(
        TensorData::new(
            (0..n)
                .flat_map(|_| {
                    [
                        sigma[0][0] as f32,
                        sigma[0][1] as f32,
                        sigma[1][0] as f32,
                        sigma[1][1] as f32,
                    ]
                })
                .collect(),
            [n, 2, 2],
        ),
        device,
    );
    let w1 = model.first.weight.val();
    let b1 = model.first.bias.as_ref().map(|b| b.val());
    let w2 = model.last.weight.val();
    let b2 = model.last.bias.as_ref().map(|b| b.val());
    if full {
        let end = propagate_linear_full(
            &propagate_relu_full(&propagate_linear_full(
                &MomentsFull::new(mean, covariance),
                w1,
                b1,
            )),
            w2,
            b2,
        );
        let variance = checked_variance(end.variance().to_data().try_to_vec::<f32>().unwrap());
        let mean = end
            .mean
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
        from_moments(mean, variance)
    } else {
        // Input correlations affect the first affine marginal variance; hidden
        // cross-covariances are deliberately dropped before the ReLU.
        let first = propagate_linear_full(&MomentsFull::new(mean, covariance), w1, b1);
        let first_variance = first.variance();
        let end = propagate_linear(
            &propagate_relu(&Moments::new(first.mean, first_variance)),
            w2,
            b2,
        );
        let mean = end
            .mean
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
        let variance = checked_variance(end.var.to_data().try_to_vec::<f32>().unwrap());
        from_moments(mean, variance)
    }
}

fn sigma_points(
    model: &Mlp,
    candidates: &[[f64; 2]],
    l: [[f64; 2]; 2],
    device: &Device,
) -> Evaluation {
    let mut all = Vec::with_capacity(candidates.len() * 4);
    for &mu in candidates {
        all.extend(sigma_rule(mu, l));
    }
    let y = model
        .forward(Tensor::<2>::from_data(
            TensorData::new(flat(&all), [all.len(), 2]),
            device,
        ))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let mut mean = Vec::with_capacity(candidates.len());
    let mut variance = Vec::with_capacity(candidates.len());
    for values in y.chunks_exact(4) {
        let m = values.iter().map(|v| f64::from(*v)).sum::<f64>() / 4.0;
        mean.push(m);
        variance.push(
            values
                .iter()
                .map(|v| (f64::from(*v) - m).powi(2))
                .sum::<f64>()
                / 4.0,
        );
    }
    from_moments(mean, variance)
}

/// Four equally weighted, positive spherical-radial points.  They match the
/// Gaussian mean and covariance, but not quadratic-loss expectations in general.
fn sigma_rule(mu: [f64; 2], l: [[f64; 2]; 2]) -> [[f64; 2]; 4] {
    [
        [
            mu[0] - 2.0_f64.sqrt() * l[0][0],
            mu[1] - 2.0_f64.sqrt() * l[1][0],
        ],
        [
            mu[0] + 2.0_f64.sqrt() * l[0][0],
            mu[1] + 2.0_f64.sqrt() * l[1][0],
        ],
        [
            mu[0] - 2.0_f64.sqrt() * l[0][1],
            mu[1] - 2.0_f64.sqrt() * l[1][1],
        ],
        [
            mu[0] + 2.0_f64.sqrt() * l[0][1],
            mu[1] + 2.0_f64.sqrt() * l[1][1],
        ],
    ]
}

fn mc(
    model: &Mlp,
    candidates: &[[f64; 2]],
    l: [[f64; 2]; 2],
    draws: usize,
    seed: u64,
    device: &Device,
) -> Evaluation {
    let mut rng = Rng::new(seed);
    let n = candidates.len();
    let mut loss = vec![[0.0; 3]; n];
    for _ in 0..draws {
        let points: Vec<[f64; 2]> = candidates
            .iter()
            .map(|mu| {
                let z = [rng.normal(), rng.normal()];
                [
                    mu[0] + l[0][0] * z[0] + l[0][1] * z[1],
                    mu[1] + l[1][0] * z[0] + l[1][1] * z[1],
                ]
            })
            .collect();
        let y = model
            .forward(Tensor::<2>::from_data(
                TensorData::new(flat(&points), [n, 2]),
                device,
            ))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (i, value) in y.into_iter().enumerate() {
            let value = f64::from(value);
            for (j, target) in TARGETS.iter().enumerate() {
                loss[i][j] += (value - target).powi(2);
            }
        }
    }
    for row in &mut loss {
        for value in row {
            *value /= draws as f64;
        }
    }
    Evaluation { loss }
}

fn choose(loss: &[f64]) -> usize {
    assert!(
        !loss.is_empty() && loss.iter().all(|value| value.is_finite()),
        "candidate losses must be finite and nonempty"
    );
    loss.iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| a.total_cmp(b).then(i.cmp(j)))
        .unwrap()
        .0
}
fn mae(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "MAE inputs must have matching lengths");
    assert!(
        !a.is_empty() && a.iter().chain(b).all(|value| value.is_finite()),
        "MAE inputs must be finite and nonempty"
    );
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
fn se(values: &[f64]) -> f64 {
    if values.len() < 2 {
        0.0
    } else {
        let m = mean(values);
        (values.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64
            / values.len() as f64)
            .sqrt()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    Point,
    Local,
    Sigma,
    Diag,
    Full,
    Mc,
}
impl Method {
    const ALL: [Self; 6] = [
        Self::Point,
        Self::Local,
        Self::Sigma,
        Self::Diag,
        Self::Full,
        Self::Mc,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Local => "local Jacobian",
            Self::Sigma => "4-point",
            Self::Diag => "diag hidden",
            Self::Full => "full K3",
            Self::Mc => "MC",
        }
    }
}

struct Record {
    repeat: usize,
    method: Method,
    ratio: usize,
    target: usize,
    method_mae: f64,
    selected_loss: f64,
    regret: f64,
    covariance_gap: f64,
    surrogate_gap: f64,
}

struct Study {
    epochs: usize,
    rmses: Vec<f64>,
    records: Vec<Record>,
}

fn evaluate_model(
    repeat: usize,
    model: &Mlp,
    candidates: &[[f64; 2]],
    method_draws: usize,
    reference_draws: usize,
    device: &Device,
) -> (f64, Vec<Record>) {
    let true_sigma = covariance(TRUE_L, 1.0);
    let rmse = clean_rmse(model, repeat, device);
    let reference_true = mc(
        model,
        candidates,
        TRUE_L,
        reference_draws,
        0xE3F0_0000 + repeat as u64,
        device,
    );
    let assumed_references: [Evaluation; 3] = std::array::from_fn(|ratio_index| {
        mc(
            model,
            candidates,
            scaled_l(RATIOS[ratio_index]),
            reference_draws,
            0xE3F1_0000 + repeat as u64 * 3 + ratio_index as u64,
            device,
        )
    });
    let mut records = Vec::new();
    for (ratio_index, ratio) in RATIOS.iter().enumerate() {
        let l = scaled_l(*ratio);
        let assumed = covariance(TRUE_L, *ratio);
        let reference = &assumed_references[ratio_index];
        let evaluations = [
            point(model, candidates, device),
            local(model, candidates, assumed, device),
            sigma_points(model, candidates, l, device),
            propagation(model, candidates, assumed, false, device),
            propagation(model, candidates, assumed, true, device),
            mc(
                model,
                candidates,
                l,
                method_draws,
                0xAC00_0000 + repeat as u64 * 3 + ratio_index as u64,
                device,
            ),
        ];
        for (method, estimate) in Method::ALL.into_iter().zip(evaluations) {
            for target in 0..TARGETS.len() {
                let estimated: Vec<f64> = estimate.loss.iter().map(|row| row[target]).collect();
                let assumed_reference: Vec<f64> =
                    reference.loss.iter().map(|row| row[target]).collect();
                let true_reference: Vec<f64> =
                    reference_true.loss.iter().map(|row| row[target]).collect();
                let exact: Vec<f64> = candidates
                    .iter()
                    .map(|&x| oracle(x, true_sigma, TARGETS[target]))
                    .collect();
                let selected = choose(&estimated);
                let exact_loss = exact[selected];
                let exact_best = exact.iter().copied().fold(f64::INFINITY, f64::min);
                records.push(Record {
                    repeat,
                    method,
                    ratio: ratio_index,
                    target,
                    method_mae: mae(&estimated, &assumed_reference),
                    selected_loss: exact_loss,
                    regret: exact_loss - exact_best,
                    covariance_gap: mae(&assumed_reference, &true_reference),
                    surrogate_gap: mae(&true_reference, &exact),
                });
            }
        }
    }
    (rmse, records)
}

fn run_fit_study(
    quick: bool,
    method_draws: usize,
    reference_draws: usize,
    dev: &Device,
    inner: &Device,
    candidates: &[[f64; 2]],
) {
    let repeats = if quick { 3 } else { 20 };
    let (early, late) = if quick { (80, 320) } else { (500, 2_000) };
    print_protocol(repeats, candidates.len(), method_draws, reference_draws);
    println!("paired fixed checkpoints {early}/{late} epochs from one trajectory per fit");
    let mut first = Study {
        epochs: early,
        rmses: Vec::new(),
        records: Vec::new(),
    };
    let mut last = Study {
        epochs: late,
        rmses: Vec::new(),
        records: Vec::new(),
    };
    for repeat in 0..repeats {
        let (first_model, last_model) = train_snapshots(repeat, early, late, dev);
        let (rmse, records) = evaluate_model(
            repeat,
            &first_model,
            candidates,
            method_draws,
            reference_draws,
            inner,
        );
        first.rmses.push(rmse);
        first.records.extend(records);
        let (rmse, records) = evaluate_model(
            repeat,
            &last_model,
            candidates,
            method_draws,
            reference_draws,
            inner,
        );
        last.rmses.push(rmse);
        last.records.extend(records);
    }
    print_fit_table(&first, repeats);
    print_fit_table(&last, repeats);
    println!(
        "\npaired {} minus {} epochs (SE over whole trained fits{}):",
        late,
        early,
        if quick { "; descriptive quick run" } else { "" }
    );
    println!(
        "  clean RMSE {:+.4} ± {:.4}",
        mean(&last.rmses) - mean(&first.rmses),
        paired_se(&last.rmses, &first.rmses)
    );
    let true_mae = |study: &Study| {
        (0..study.rmses.len())
            .map(|repeat| {
                mean(
                    &study
                        .records
                        .iter()
                        .filter(|x| x.repeat == repeat)
                        .map(|x| x.surrogate_gap)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    let late_true = true_mae(&last);
    let early_true = true_mae(&first);
    println!(
        "  true-law surrogate-vs-simulator MAE {:+.4} ± {:.4}",
        mean(&late_true) - mean(&early_true),
        paired_se(&late_true, &early_true)
    );
    for method in Method::ALL {
        let left = condition_means(&last, method, true);
        let right = condition_means(&first, method, true);
        let left_regret = condition_means(&last, method, false);
        let right_regret = condition_means(&first, method, false);
        println!(
            "  {:<14} selected loss {:+.4} ± {:.4}; regret {:+.4} ± {:.4}",
            method.name(),
            mean(&left) - mean(&right),
            paired_se(&left, &right),
            mean(&left_regret) - mean(&right_regret),
            paired_se(&left_regret, &right_regret)
        );
    }
}

fn print_protocol(repeats: usize, candidates: usize, method_draws: usize, reference_draws: usize) {
    println!(
        "{} repeats; {} candidates; covariance ratios {:?}; method MC={} draws; reference MC={} draws",
        repeats, candidates, RATIOS, method_draws, reference_draws
    );
}

fn condition_means(study: &Study, method: Method, loss: bool) -> Vec<f64> {
    (0..study.rmses.len())
        .map(|repeat| {
            let values: Vec<f64> = study
                .records
                .iter()
                .filter(|x| x.repeat == repeat && x.method == method)
                .map(|x| if loss { x.selected_loss } else { x.regret })
                .collect();
            mean(&values)
        })
        .collect()
}
fn paired_se(left: &[f64], right: &[f64]) -> f64 {
    se(&left
        .iter()
        .zip(right)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>())
}
fn print_fit_table(study: &Study, repeats: usize) {
    println!(
        "\nfit-quality checkpoint: {} epochs; clean RMSE {:.4} ± {:.4} (SE over {} fits)",
        study.epochs,
        mean(&study.rmses),
        se(&study.rmses),
        repeats
    );
    for (ratio, value) in RATIOS.iter().enumerate() {
        println!("  covariance ratio {value:.2}:");
        for method in Method::ALL {
            let rows: Vec<_> = study
                .records
                .iter()
                .filter(|x| x.ratio == ratio && x.method == method)
                .collect();
            println!(
                "    {:<14} risk MAE {:.4} loss {:.4} regret {:.4}",
                method.name(),
                mean(&rows.iter().map(|x| x.method_mae).collect::<Vec<_>>()),
                mean(&rows.iter().map(|x| x.selected_loss).collect::<Vec<_>>()),
                mean(&rows.iter().map(|x| x.regret).collect::<Vec<_>>())
            );
        }
        let rows: Vec<_> = study.records.iter().filter(|x| x.ratio == ratio).collect();
        println!(
            "    reference discrepancy: assumed-vs-true surrogate {:.4}; true-surrogate-vs-exact simulator {:.4}",
            mean(&rows.iter().map(|x| x.covariance_gap).collect::<Vec<_>>()),
            mean(&rows.iter().map(|x| x.surrogate_gap).collect::<Vec<_>>())
        );
    }
    for (target, value) in TARGETS.iter().enumerate() {
        println!("  target {value:.2}:");
        for method in Method::ALL {
            let rows: Vec<_> = study
                .records
                .iter()
                .filter(|x| x.target == target && x.method == method)
                .collect();
            println!(
                "    {:<14} loss {:.4} regret {:.4}",
                method.name(),
                mean(&rows.iter().map(|x| x.selected_loss).collect::<Vec<_>>()),
                mean(&rows.iter().map(|x| x.regret).collect::<Vec<_>>())
            );
        }
    }
}

fn main() {
    let mut quick = false;
    let mut fit_study = false;
    for argument in env::args().skip(1) {
        match argument.as_str() {
            "--quick" if !quick => quick = true,
            "--fit-study" if !fit_study => fit_study = true,
            _ => {
                eprintln!("usage: robust_selection [--quick] [--fit-study]");
                std::process::exit(2);
            }
        }
    }
    let repeats = if quick { 3 } else { 20 };
    let epochs = if quick { 80 } else { 500 };
    let method_draws = if quick { 32 } else { 64 };
    let reference_draws = if quick { 256 } else { 2_048 };
    let candidates = candidates();
    let dev = Device::flex().autodiff();
    let inner = Device::flex();
    let mut records = Vec::new();
    let mut rmses = Vec::new();
    if fit_study {
        run_fit_study(
            quick,
            method_draws,
            reference_draws,
            &dev,
            &inner,
            &candidates,
        );
        return;
    }
    print_protocol(repeats, candidates.len(), method_draws, reference_draws);
    println!("fixed {epochs}-epoch run: a descriptive fit-quality diagnostic, not an epoch-selected result");
    for repeat in 0..repeats {
        let model = train(repeat, epochs, &dev);
        let (rmse, values) = evaluate_model(
            repeat,
            &model,
            &candidates,
            method_draws,
            reference_draws,
            &inner,
        );
        rmses.push(rmse);
        records.extend(values);
    }
    let study = Study {
        epochs,
        rmses,
        records,
    };
    print_fit_table(&study, repeats);
    for comparison in [Method::Mc, Method::Point, Method::Diag] {
        let differences: Vec<f64> = (0..repeats)
            .map(|repeat| {
                let full = study
                    .records
                    .iter()
                    .filter(|x| x.method == Method::Full && x.repeat == repeat)
                    .map(|x| x.selected_loss)
                    .sum::<f64>()
                    / (RATIOS.len() * TARGETS.len()) as f64;
                let other = study
                    .records
                    .iter()
                    .filter(|x| x.method == comparison && x.repeat == repeat)
                    .map(|x| x.selected_loss)
                    .sum::<f64>()
                    / (RATIOS.len() * TARGETS.len()) as f64;
                full - other
            })
            .collect();
        println!(
            "paired full K3 minus {} selected loss: {:.4} ± {:.4} SE over repeats{}",
            comparison.name(),
            mean(&differences),
            se(&differences),
            if quick {
                " (descriptive quick run)"
            } else {
                ""
            }
        );
    }
    println!("Methods rank fixed candidates using a surrogate; exact regret always uses the true simulator covariance. The diagonal-hidden path preserves input correlation through its first affine marginals, then drops hidden cross-covariances.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;

    fn hand_model(device: &Device) -> Mlp {
        let mut first = vec![0.0; 2 * HIDDEN];
        first[0] = 1.0;
        first[HIDDEN + 1] = 1.0;
        let mut last = vec![0.0; HIDDEN];
        last[0] = 0.6;
        last[1] = -0.2;
        Mlp {
            first: Linear {
                weight: Param::from_tensor(Tensor::from_data(
                    TensorData::new(first, [2, HIDDEN]),
                    device,
                )),
                bias: Some(Param::from_tensor(Tensor::from_data(
                    TensorData::new(vec![0.5; HIDDEN], [HIDDEN]),
                    device,
                ))),
            },
            last: Linear {
                weight: Param::from_tensor(Tensor::from_data(
                    TensorData::new(last, [HIDDEN, 1]),
                    device,
                )),
                bias: Some(Param::from_tensor(Tensor::from_data(
                    TensorData::new(vec![0.1], [1]),
                    device,
                ))),
            },
        }
    }

    #[test]
    fn frozen_checkpoint_matches_its_epoch_and_not_the_later_epoch() {
        let device = Device::flex().autodiff();
        let inner = Device::flex();
        let (snapshot_one, snapshot_two) = train_snapshots(0, 1, 2, &device);
        let alone_one = train(0, 1, &device);
        let alone_two = train(0, 2, &device);
        let input = Tensor::<2>::from_data(TensorData::new(vec![0.2, -0.1], [1, 2]), &inner);
        let one = snapshot_one
            .forward(input.clone())
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let two = snapshot_two
            .forward(input.clone())
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(
            one,
            alone_one
                .forward(input.clone())
                .to_data()
                .try_to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            two,
            alone_two
                .forward(input)
                .to_data()
                .try_to_vec::<f32>()
                .unwrap()
        );
        assert_ne!(
            one, two,
            "one and two optimizer steps must produce distinct snapshots"
        );
    }

    #[test]
    fn zero_noise_actual_methods_match_deterministic_hand_model() {
        let device = Device::flex();
        let model = hand_model(&device);
        let candidates = [[0.2, -0.1], [0.7, 0.4]];
        let point = point(&model, &candidates, &device);
        let zero = [[0.0; 2]; 2];
        for estimate in [
            local(&model, &candidates, zero, &device),
            sigma_points(&model, &candidates, zero, &device),
            propagation(&model, &candidates, zero, false, &device),
            propagation(&model, &candidates, zero, true, &device),
            mc(&model, &candidates, zero, 4, 1, &device),
        ] {
            for (actual, expected) in estimate.loss.iter().zip(&point.loss) {
                for (actual, expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn local_gradient_matches_finite_difference_for_asymmetric_hand_model() {
        let device = Device::flex();
        let model = hand_model(&device);
        let x = [0.2, -0.1];
        let epsilon = 1e-3;
        let gradient: [f64; 2] = std::array::from_fn(|axis| {
            let mut up = x;
            let mut down = x;
            up[axis] += epsilon;
            down[axis] -= epsilon;
            (point_mean(&model, &[up], &device)[0] - point_mean(&model, &[down], &device)[0])
                / (2.0 * epsilon)
        });
        let sigma = covariance(TRUE_L, 1.0);
        let expected_variance = (0..2)
            .flat_map(|i| (0..2).map(move |j| gradient[i] * sigma[i][j] * gradient[j]))
            .sum::<f64>();
        let point_loss = point(&model, &[x], &device).loss[0][0];
        let local_loss = local(&model, &[x], sigma, &device).loss[0][0];
        assert!((local_loss - point_loss - expected_variance).abs() < 2e-5);
    }
    #[test]
    fn oracle_matches_expanded_one_dimensional_special_case() {
        let mu: [f64; 2] = [0.7, 0.0];
        let sigma = [[0.3, 0.0], [0.0, 0.0]];
        let target = 0.4;
        let m = mu[0] * mu[0] + 0.2 * mu[0] + 0.3;
        let v = 2.0 * 0.3_f64.powi(2) + (2.0 * mu[0] + 0.2).powi(2) * 0.3;
        assert!((oracle(mu, sigma, target) - ((m - target).powi(2) + v)).abs() < 1e-12);
    }
    #[test]
    fn sigma_points_match_mean_and_covariance() {
        let l = [[0.2, 0.0], [0.12, 0.16]];
        let points = sigma_rule([0.0, 0.0], l);
        let mean = [
            points.iter().map(|p| p[0]).sum::<f64>() / 4.0,
            points.iter().map(|p| p[1]).sum::<f64>() / 4.0,
        ];
        let cov00 = points.iter().map(|p| p[0] * p[0]).sum::<f64>() / 4.0;
        let cov01 = points.iter().map(|p| p[0] * p[1]).sum::<f64>() / 4.0;
        let cov11 = points.iter().map(|p| p[1] * p[1]).sum::<f64>() / 4.0;
        assert!(mean[0].abs() < 1e-14 && mean[1].abs() < 1e-14);
        assert!(
            (cov00 - covariance(l, 1.0)[0][0]).abs() < 1e-14
                && (cov01 - covariance(l, 1.0)[0][1]).abs() < 1e-14
                && (cov11 - covariance(l, 1.0)[1][1]).abs() < 1e-14
        );
    }
    #[test]
    fn zero_noise_oracle_and_ties_are_deterministic() {
        let x = [0.3, -0.2];
        assert_eq!(oracle(x, [[0.0; 2]; 2], 1.0), (q(x) - 1.0).powi(2));
        assert_eq!(choose(&[1.0, 1.0, 2.0]), 0);
        let candidates = [[0.0, 0.0], [1.0, 0.0]];
        let losses: Vec<f64> = candidates
            .iter()
            .map(|&x| oracle(x, [[0.0; 2]; 2], 0.0))
            .collect();
        let selected = choose(&losses);
        let best = losses.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(losses[selected] - best, 0.0);
    }
    proptest::proptest! {
        #[test]
        fn oracle_matches_independent_gauss_hermite(
            a in -1.0f64..1.0, b in -1.0f64..1.0, c in -1.0f64..1.0,
            x in -2.0f64..2.0, y in -2.0f64..2.0, target in -2.0f64..2.0,
        ) {
            let l = [[a, 0.0], [b, c]];
            let nodes = [-3_f64.sqrt(), 0.0, 3_f64.sqrt()];
            let weights = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
            let direct = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| {
                    let z = [nodes[i], nodes[j]];
                    let value = q([
                        x + l[0][0] * z[0] + l[0][1] * z[1],
                        y + l[1][0] * z[0] + l[1][1] * z[1],
                    ]);
                    weights[i] * weights[j] * (value - target).powi(2)
                })
                .sum::<f64>();
            let exact = oracle([x, y], covariance(l, 1.0), target);
            proptest::prop_assert!((direct - exact).abs() <= 2e-12 * exact.abs().max(1.0));
        }
    }
}
