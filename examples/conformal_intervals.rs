//! Conformalize stableprop's analytic error bars.
//!
//! The default run is one small illustrative split. `--study` repeats a
//! heteroscedastic synthetic study. It trains on clean feature centers, then
//! uses separate calibration/test splits whose targets include one draw of
//! independent Gaussian noise in each feature, plus label noise. The feature
//! noise scale varies by input, is supplied to stableprop, and is never fitted
//! from residuals. It reports marginal coverage and width across repeats;
//! low/high noise-bin summaries are descriptive, not coverage guarantees.
//!
//! stableprop's propagated standard deviation is an input-noise sensitivity
//! scale, not a calibrated residual model. Split conformal calibrates held-out
//! residuals scaled by that quantity. It targets marginal coverage under
//! exchangeability; neither mode guarantees coverage for every input or after
//! distribution shift.
//!
//! Run: `cargo run --release --example conformal_intervals --features burn`
//! Study: `cargo run --release --example conformal_intervals --features burn -- --study`
//! Diagnose: `cargo run --release --example conformal_intervals --features burn -- --diagnose`
//! Full diagnostic study: `cargo run --release --example conformal_intervals --features burn -- --diagnose-study`

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Device, Tensor, TensorData};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_full, propagate_relu, propagate_relu_full, Moments,
    MomentsFull,
};

type Ad = Autodiff<NdArray<f32>>;
type Nd = NdArray<f32>;

const D_IN: usize = 6;
const HIDDEN: usize = 64;
const N_TRAIN: usize = 3000;
const N_CAL: usize = 1000;
const N_TEST: usize = 1000;
const EPOCHS: usize = 800;
const INPUT_STD: f64 = 0.1;
const LABEL_STD: f64 = 0.2;
const ALPHA: f64 = 0.1; // target miscoverage -> 90% intervals

const STUDY_REPEATS: usize = 30;
const STUDY_TRAIN: usize = 800;
const STUDY_CAL: usize = 400;
const STUDY_TEST: usize = 800;
const STUDY_EPOCHS: usize = 300;
const STUDY_T_CRITICAL: f64 = 2.045; // t_{0.975,29}
const COVERAGE_TOLERANCE: f64 = 0.02;
const MIN_WIDTH_REDUCTION: f64 = 0.02;
const NOISE_BIN_SPLIT: f64 = 0.14;
const STUDY_SEED_BASE: u64 = 0xC0A1_3000;

const DIAGNOSTIC_QUICK_REPEATS: usize = 4;
const DIAGNOSTIC_QUICK_CENTERS: usize = 12;
const DIAGNOSTIC_STUDY_CENTERS: usize = 48;
const DIAGNOSTIC_MC_BATCHES: usize = 4;
const DIAGNOSTIC_MC_DRAWS: usize = 512;
const DIAGNOSTIC_SEED_BASE: u64 = 0xD1A6_0000;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.lin1.forward(x));
        self.lin2.forward(h)
    }
}

#[derive(Clone)]
struct Data {
    x: Vec<f32>,
    y: Vec<f32>,
    sigma: Vec<f32>,
}

#[derive(Clone, Copy)]
struct IntervalMetrics {
    coverage: f64,
    width: f64,
    low_noise_coverage: f64,
    high_noise_coverage: f64,
}

struct Trial {
    raw: IntervalMetrics,
    scaled: IntervalMetrics,
    constant: IntervalMetrics,
}

#[derive(Clone, Copy)]
struct MeanCi {
    mean: f64,
    lower: f64,
    upper: f64,
}

#[derive(Clone, Copy, Default)]
struct DiagnosticMetrics {
    diagonal_mean_error: f64,
    full_mean_error: f64,
    diagonal_variance_error: f64,
    full_variance_error: f64,
    target_variance: f64,
    target_signal_variance: f64,
    target_noise_variance: f64,
    sampled_model_variance: f64,
    target_model_mean_shift: f64,
    center_squared_bias: f64,
    target_center_mse: f64,
    model_mean_mc_batch_spread: f64,
    model_variance_mc_batch_spread: f64,
    target_variance_mc_batch_spread: f64,
    target_mean_mc_batch_spread: f64,
}

/// Small local RNG keeps split construction independent of Burn's backend RNG.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        radius * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
}

fn target(x: &[f32]) -> f32 {
    let s: f32 = x.iter().sum();
    (s * 0.6).sin() + 0.5 * x[0] * x[1] - 0.3 * x[2] * x[2]
}

/// A supplied feature-noise scale. It is defined before labels are sampled and
/// is not a fitted estimate of residual variance.
fn known_feature_std(x0: f64) -> f64 {
    0.04 + 0.20 / (1.0 + (-x0).exp())
}

/// SplitMix64 maps distinct structured study IDs to well-mixed RNG seeds.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Roles 1--4 are train, calibration, test, and model initialization.
fn study_seed(repeat: usize, role: u64) -> u64 {
    assert!(repeat < STUDY_REPEATS, "unknown study repeat");
    assert!((1..=4).contains(&role), "unknown study seed role");
    splitmix64(STUDY_SEED_BASE + repeat as u64 * 4 + role)
}

fn make_data(
    n: usize,
    seed: u64,
    heteroscedastic_features: bool,
    perturb_target_features: bool,
) -> Data {
    assert!(
        !perturb_target_features || heteroscedastic_features,
        "a target feature perturbation requires a declared feature-noise scale"
    );
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n * D_IN);
    let mut y = Vec::with_capacity(n);
    let mut sigma = Vec::with_capacity(n);
    for _ in 0..n {
        let center: Vec<f32> = (0..D_IN).map(|_| rng.normal() as f32).collect();
        let feature_std = if heteroscedastic_features {
            known_feature_std(f64::from(center[0]))
        } else {
            INPUT_STD
        };
        let perturbed: Vec<f32> = center
            .iter()
            .map(|value| {
                let perturbation = if perturb_target_features {
                    feature_std * rng.normal()
                } else {
                    0.0
                };
                *value + perturbation as f32
            })
            .collect();
        x.extend(center);
        y.push(target(&perturbed) + (LABEL_STD * rng.normal()) as f32);
        sigma.push(feature_std as f32);
    }
    Data { x, y, sigma }
}

fn train_model(data: &Data, epochs: usize, model_seed: u64, dev: &Device<Ad>) -> Mlp<Ad> {
    <Ad as Backend>::seed(dev, model_seed);
    let x = Tensor::<Ad, 2>::from_data(TensorData::new(data.x.clone(), [data.y.len(), D_IN]), dev);
    let y = Tensor::<Ad, 2>::from_data(TensorData::new(data.y.clone(), [data.y.len(), 1]), dev);
    let mut model = Mlp::<Ad>::init(dev);
    let mut optim = AdamConfig::new().init();
    for _ in 0..epochs {
        let pred = model.forward(x.clone());
        let loss = MseLoss::new().forward(pred, y.clone(), Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-3, model, grads);
    }
    model
}

/// Propagated mean and standard deviation under supplied per-example feature
/// uncertainty. The variance floor is a conformal-score normalization floor,
/// not an uncertainty estimate.
fn predict(model: &Mlp<Ad>, x: &[f32], sigma: &[f32], dev: &Device<Nd>) -> (Vec<f32>, Vec<f64>) {
    let n = sigma.len();
    assert_eq!(x.len(), n * D_IN, "input shape must match feature scales");
    let input = Tensor::<Nd, 2>::from_data(TensorData::new(x.to_vec(), [n, D_IN]), dev);
    let input_variance = Tensor::<Nd, 2>::from_data(
        TensorData::new(sigma.iter().map(|s| s * s).collect::<Vec<_>>(), [n, 1]),
        dev,
    )
    .expand([n, D_IN]);
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val().inner());
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val().inner());
    let m1 = propagate_relu(&propagate_linear(
        &Moments::new(input.clone(), input_variance),
        w1,
        b1,
    ));
    let m2 = propagate_linear(&m1, w2, b2);
    let mean = m2.mean.to_data().to_vec::<f32>().unwrap();
    let std = m2
        .var
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|value| {
            let variance = f64::from(*value);
            assert!(
                variance.is_finite() && variance >= 0.0,
                "propagated variance must be finite and nonnegative"
            );
            variance.max(1e-12).sqrt()
        })
        .collect();
    (mean, std)
}

/// Diagonal and full-K3 moments for the same one-hidden-ReLU model and input
/// law. There is no repeated Gaussian closure in this architecture: the only
/// nonlinear moment approximation is the hidden full-covariance K3 step.
fn predict_diagnostic_moments(
    model: &Mlp<Ad>,
    x: &[f32],
    sigma: &[f32],
    dev: &Device<Nd>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = sigma.len();
    assert_eq!(x.len(), n * D_IN, "input shape must match feature scales");
    let input = Tensor::<Nd, 2>::from_data(TensorData::new(x.to_vec(), [n, D_IN]), dev);
    let input_variance = Tensor::<Nd, 2>::from_data(
        TensorData::new(sigma.iter().map(|s| s * s).collect::<Vec<_>>(), [n, 1]),
        dev,
    )
    .expand([n, D_IN]);
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val().inner());
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val().inner());

    let diagonal = propagate_linear(
        &propagate_relu(&propagate_linear(
            &Moments::new(input.clone(), input_variance.clone()),
            w1.clone(),
            b1.clone(),
        )),
        w2.clone(),
        b2.clone(),
    );
    let full = propagate_linear_full(
        &propagate_relu_full(&propagate_linear_full(
            &MomentsFull::from_diagonal(input, input_variance),
            w1,
            b1,
        )),
        w2,
        b2,
    );
    let diagonal_mean = diagonal
        .mean
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let diagonal_variance = diagonal
        .var
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let full_mean = full
        .mean
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let full_variance = full
        .variance()
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    (diagonal_mean, diagonal_variance, full_mean, full_variance)
}

fn sample_mean_variance(values: &[f64]) -> (f64, f64) {
    assert!(values.len() > 1, "sample variance needs at least two draws");
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, variance)
}

fn sample_model_outputs(
    model: &Mlp<Ad>,
    center: &[f32],
    sigma: f64,
    draws: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut input = Vec::with_capacity(draws * D_IN);
    for _ in 0..draws {
        input.extend(
            center
                .iter()
                .map(|value| *value + (sigma * rng.normal()) as f32),
        );
    }
    let dev = Device::<Nd>::default();
    let input = Tensor::<Nd, 2>::from_data(TensorData::new(input, [draws, D_IN]), &dev);
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val().inner());
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val().inner());
    let mut hidden = input.matmul(w1);
    if let Some(bias) = b1 {
        hidden = hidden + bias.reshape([1, HIDDEN]);
    }
    let mut output = activation::relu(hidden).matmul(w2);
    if let Some(bias) = b2 {
        output = output + bias.reshape([1, 1]);
    }
    output
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

fn sample_target_outputs(center: &[f32], sigma: f64, draws: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..draws)
        .map(|_| {
            let perturbed: Vec<_> = center
                .iter()
                .map(|value| *value + (sigma * rng.normal()) as f32)
                .collect();
            f64::from(target(&perturbed))
        })
        .collect()
}

/// An unbiased estimate of `(E[Y | x] - c)^2` from independent target-MC
/// batch means. It can be negative at finite MC budget; that does not mean the
/// population squared bias is negative.
fn corrected_squared_bias(batch_means: &[f64], center: f64) -> f64 {
    let (mean, variance) = sample_mean_variance(batch_means);
    (mean - center).powi(2) - variance / batch_means.len() as f64
}

#[cfg(test)]
fn cross_batch_squared_bias(batch_means: &[f64], center: f64) -> f64 {
    assert!(
        batch_means.len() > 1,
        "cross-batch estimate needs two batches"
    );
    let mut sum = 0.0;
    for (i, left) in batch_means.iter().enumerate() {
        for (j, right) in batch_means.iter().enumerate() {
            if i != j {
                sum += (left - center) * (right - center);
            }
        }
    }
    sum / (batch_means.len() * (batch_means.len() - 1)) as f64
}

fn target_center_mse(target_variance: f64, corrected_squared_bias: f64) -> f64 {
    target_variance + corrected_squared_bias
}

fn conformal_quantile(mut scores: Vec<f64>) -> f64 {
    scores.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let rank = (((scores.len() + 1) as f64 * (1.0 - ALPHA)).ceil() as usize).min(scores.len()) - 1;
    scores[rank]
}

fn interval_metrics(
    y: &[f32],
    mean: &[f32],
    sigma: &[f64],
    known_sigma: &[f32],
    multiplier: impl Fn(usize) -> f64,
) -> IntervalMetrics {
    let (mut covered, mut low_covered, mut high_covered) = (0usize, 0usize, 0usize);
    let (mut low_count, mut high_count) = (0usize, 0usize);
    let mut width = 0.0;
    for i in 0..y.len() {
        let half_width = multiplier(i) * sigma[i];
        let hit = (f64::from(y[i]) - f64::from(mean[i])).abs() <= half_width;
        covered += usize::from(hit);
        width += 2.0 * half_width;
        if f64::from(known_sigma[i]) <= NOISE_BIN_SPLIT {
            low_count += 1;
            low_covered += usize::from(hit);
        } else {
            high_count += 1;
            high_covered += usize::from(hit);
        }
    }
    IntervalMetrics {
        coverage: covered as f64 / y.len() as f64,
        width: width / y.len() as f64,
        low_noise_coverage: if low_count > 0 {
            low_covered as f64 / low_count as f64
        } else {
            f64::NAN
        },
        high_noise_coverage: if high_count > 0 {
            high_covered as f64 / high_count as f64
        } else {
            f64::NAN
        },
    }
}

fn evaluate(
    train: Data,
    calibration: Data,
    test: Data,
    epochs: usize,
    model_seed: u64,
    dev: &Device<Ad>,
) -> Trial {
    let model = train_model(&train, epochs, model_seed, dev);
    let inner = Device::<Nd>::default();
    let (cal_mean, cal_sigma) = predict(&model, &calibration.x, &calibration.sigma, &inner);
    let (test_mean, test_sigma) = predict(&model, &test.x, &test.sigma, &inner);
    let scaled_quantile = conformal_quantile(
        (0..calibration.y.len())
            .map(|i| (f64::from(calibration.y[i]) - f64::from(cal_mean[i])).abs() / cal_sigma[i])
            .collect(),
    );
    let constant_quantile = conformal_quantile(
        (0..calibration.y.len())
            .map(|i| (f64::from(calibration.y[i]) - f64::from(cal_mean[i])).abs())
            .collect(),
    );
    Trial {
        raw: interval_metrics(&test.y, &test_mean, &test_sigma, &test.sigma, |_| 1.645),
        scaled: interval_metrics(&test.y, &test_mean, &test_sigma, &test.sigma, |_| {
            scaled_quantile
        }),
        constant: interval_metrics(
            &test.y,
            &test_mean,
            &vec![1.0; test.y.len()],
            &test.sigma,
            |_| constant_quantile,
        ),
    }
}

fn mean_ci(values: &[f64]) -> MeanCi {
    assert_eq!(
        values.len(),
        STUDY_REPEATS,
        "study intervals require {STUDY_REPEATS} repeats"
    );
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let half_width = STUDY_T_CRITICAL * (variance / values.len() as f64).sqrt();
    MeanCi {
        mean,
        lower: mean - half_width,
        upper: mean + half_width,
    }
}

fn print_single(trial: Trial) {
    println!(
        "conformal calibration target coverage = {:.2} (not a target for raw sensitivity)\n",
        1.0 - ALPHA
    );
    println!("  {:<34} {:>8} {:>10}", "method", "coverage", "avg width");
    for (name, metric) in [
        ("raw stableprop sensitivity (uncalibrated)", trial.raw),
        ("conformalized stableprop (adaptive)", trial.scaled),
        ("constant-width conformal", trial.constant),
    ] {
        println!(
            "  {name:<34} {:>8.3} {:>10.3}",
            metric.coverage, metric.width
        );
    }
    println!("\nThis is one split. Run with --study for repeated heteroscedastic evidence.");
}

fn print_summary(name: &str, coverage: MeanCi, width: MeanCi) {
    println!(
        "  {name:<28} coverage {:.3} [{:.3}, {:.3}]  width {:.3} [{:.3}, {:.3}]",
        coverage.mean, coverage.lower, coverage.upper, width.mean, width.lower, width.upper
    );
}

fn run_study(dev: &Device<Ad>) {
    let mut trials = Vec::with_capacity(STUDY_REPEATS);
    for repeat in 0..STUDY_REPEATS {
        println!("  study repeat {}/{}", repeat + 1, STUDY_REPEATS);
        trials.push(evaluate(
            // Train on clean centers so the network approximates f. Only
            // calibration/test targets include independent feature perturbations.
            make_data(STUDY_TRAIN, study_seed(repeat, 1), true, false),
            make_data(STUDY_CAL, study_seed(repeat, 2), true, true),
            make_data(STUDY_TEST, study_seed(repeat, 3), true, true),
            STUDY_EPOCHS,
            study_seed(repeat, 4),
            dev,
        ));
    }
    let summary = |metric: fn(&Trial) -> IntervalMetrics| {
        let coverage = mean_ci(
            &trials
                .iter()
                .map(|trial| metric(trial).coverage)
                .collect::<Vec<_>>(),
        );
        let width = mean_ci(
            &trials
                .iter()
                .map(|trial| metric(trial).width)
                .collect::<Vec<_>>(),
        );
        (coverage, width)
    };
    let (raw_coverage, raw_width) = summary(|trial| trial.raw);
    let (scaled_coverage, scaled_width) = summary(|trial| trial.scaled);
    let (constant_coverage, constant_width) = summary(|trial| trial.constant);
    let paired_coverage = mean_ci(
        &trials
            .iter()
            .map(|trial| trial.scaled.coverage - trial.constant.coverage)
            .collect::<Vec<_>>(),
    );
    let paired_width = mean_ci(
        &trials
            .iter()
            .map(|trial| trial.scaled.width - trial.constant.width)
            .collect::<Vec<_>>(),
    );
    let relative_width_reduction = -paired_width.mean / constant_width.mean;
    let low_scaled = mean_ci(
        &trials
            .iter()
            .map(|trial| trial.scaled.low_noise_coverage)
            .collect::<Vec<_>>(),
    );
    let high_scaled = mean_ci(
        &trials
            .iter()
            .map(|trial| trial.scaled.high_noise_coverage)
            .collect::<Vec<_>>(),
    );
    let coverage_screen = (scaled_coverage.mean - (1.0 - ALPHA)).abs() <= COVERAGE_TOLERANCE
        && !(scaled_coverage.upper < 1.0 - ALPHA - COVERAGE_TOLERANCE
            || scaled_coverage.lower > 1.0 - ALPHA + COVERAGE_TOLERANCE);
    let efficiency_screen =
        relative_width_reduction >= MIN_WIDTH_REDUCTION && paired_width.upper < 0.0;

    println!(
        "heteroscedastic split-conformal pilot: {STUDY_REPEATS} independent repeats; train/cal/test = {STUDY_TRAIN}/{STUDY_CAL}/{STUDY_TEST}; {STUDY_EPOCHS} epochs"
    );
    println!(
        "mean [t_29-approximate 95% interval across repeats]; calibrated-row target marginal coverage = {:.2}",
        1.0 - ALPHA
    );
    print_summary(
        "raw stableprop sensitivity (uncalibrated)",
        raw_coverage,
        raw_width,
    );
    print_summary("scaled conformal", scaled_coverage, scaled_width);
    print_summary("constant conformal", constant_coverage, constant_width);
    println!(
        "  scaled - constant           coverage {:.3} [{:.3}, {:.3}]  width {:.3} [{:.3}, {:.3}]",
        paired_coverage.mean,
        paired_coverage.lower,
        paired_coverage.upper,
        paired_width.mean,
        paired_width.lower,
        paired_width.upper
    );
    println!(
        "  scaled conformal bins        low {:.3} [{:.3}, {:.3}]  high {:.3} [{:.3}, {:.3}] (descriptive only)",
        low_scaled.mean,
        low_scaled.lower,
        low_scaled.upper,
        high_scaled.mean,
        high_scaled.lower,
        high_scaled.upper
    );
    println!(
        "coverage screen (0.90 ± {COVERAGE_TOLERANCE:.2}): {}",
        if coverage_screen { "passed" } else { "not met" }
    );
    println!(
        "efficiency screen (>= {:.0}% lower paired width and upper CI < 0): {}",
        100.0 * MIN_WIDTH_REDUCTION,
        if efficiency_screen {
            "passed"
        } else {
            "not met; no superiority claim"
        }
    );
}

/// Roles are training data, initialization, diagnostic centers, model-output
/// noise, and target-output noise. They remain independent within each outer
/// fitted-model/data repeat.
fn diagnostic_seed(repeat: usize, role: u64) -> u64 {
    assert!(repeat < STUDY_REPEATS, "unknown diagnostic repeat");
    assert!((1..=5).contains(&role), "unknown diagnostic seed role");
    splitmix64(DIAGNOSTIC_SEED_BASE + repeat as u64 * 5 + role)
}

fn diagnostic_centers(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n * D_IN).map(|_| rng.normal() as f32).collect()
}

fn diagnostic_trial(repeat: usize, centers: usize, dev: &Device<Ad>) -> DiagnosticMetrics {
    let train = make_data(STUDY_TRAIN, diagnostic_seed(repeat, 1), true, false);
    let model = train_model(&train, STUDY_EPOCHS, diagnostic_seed(repeat, 2), dev);
    let centers_x = diagnostic_centers(centers, diagnostic_seed(repeat, 3));
    let center_sigma: Vec<f32> = centers_x
        .chunks_exact(D_IN)
        .map(|center| known_feature_std(f64::from(center[0])) as f32)
        .collect();
    let inner = Device::<Nd>::default();
    let (diagonal_mean, diagonal_variance, full_mean, full_variance) =
        predict_diagnostic_moments(&model, &centers_x, &center_sigma, &inner);
    let mut metric = DiagnosticMetrics::default();

    for center_index in 0..centers {
        let center = &centers_x[center_index * D_IN..(center_index + 1) * D_IN];
        let sigma = f64::from(center_sigma[center_index]);
        let mut model_batch_means = Vec::with_capacity(DIAGNOSTIC_MC_BATCHES);
        let mut model_batch_variances = Vec::with_capacity(DIAGNOSTIC_MC_BATCHES);
        let mut target_batch_means = Vec::with_capacity(DIAGNOSTIC_MC_BATCHES);
        let mut target_batch_variances = Vec::with_capacity(DIAGNOSTIC_MC_BATCHES);
        for batch in 0..DIAGNOSTIC_MC_BATCHES {
            let model_seed = splitmix64(
                diagnostic_seed(repeat, 4) ^ ((center_index as u64) << 16) ^ batch as u64,
            );
            let target_seed = splitmix64(
                diagnostic_seed(repeat, 5) ^ ((center_index as u64) << 16) ^ batch as u64,
            );
            let (model_mean, model_variance) = sample_mean_variance(&sample_model_outputs(
                &model,
                center,
                sigma,
                DIAGNOSTIC_MC_DRAWS,
                model_seed,
            ));
            let (target_mean, target_variance) = sample_mean_variance(&sample_target_outputs(
                center,
                sigma,
                DIAGNOSTIC_MC_DRAWS,
                target_seed,
            ));
            model_batch_means.push(model_mean);
            model_batch_variances.push(model_variance);
            target_batch_means.push(target_mean);
            target_batch_variances.push(target_variance);
        }
        let sampled_mean = model_batch_means.iter().sum::<f64>() / DIAGNOSTIC_MC_BATCHES as f64;
        let sampled_variance =
            model_batch_variances.iter().sum::<f64>() / DIAGNOSTIC_MC_BATCHES as f64;
        let target_mean = target_batch_means.iter().sum::<f64>() / DIAGNOSTIC_MC_BATCHES as f64;
        // This is a synthetic oracle: the generator's independent label-noise
        // variance is known, rather than estimated from residuals.
        let target_noise_variance = LABEL_STD * LABEL_STD;
        let target_variance = target_batch_variances.iter().sum::<f64>()
            / DIAGNOSTIC_MC_BATCHES as f64
            + target_noise_variance;
        let corrected_bias =
            corrected_squared_bias(&target_batch_means, diagonal_mean[center_index]);
        metric.diagonal_mean_error += (diagonal_mean[center_index] - sampled_mean).abs();
        metric.full_mean_error += (full_mean[center_index] - sampled_mean).abs();
        metric.diagonal_variance_error +=
            (diagonal_variance[center_index] - sampled_variance).abs();
        metric.full_variance_error += (full_variance[center_index] - sampled_variance).abs();
        metric.target_variance += target_variance;
        metric.target_signal_variance += target_variance - target_noise_variance;
        metric.target_noise_variance += target_noise_variance;
        metric.sampled_model_variance += sampled_variance;
        metric.target_model_mean_shift += target_mean - sampled_mean;
        metric.center_squared_bias += corrected_bias;
        metric.target_center_mse += target_center_mse(target_variance, corrected_bias);
        // These common-reference spreads quantify the MC component of both
        // paired diagonal and full comparisons; subtracting either fixed
        // analytic estimate leaves the batch standard deviation unchanged.
        metric.model_mean_mc_batch_spread += sample_mean_variance(&model_batch_means).1.sqrt();
        metric.model_variance_mc_batch_spread +=
            sample_mean_variance(&model_batch_variances).1.sqrt();
        metric.target_variance_mc_batch_spread += sample_mean_variance(
            &target_batch_variances
                .iter()
                .map(|variance| variance + target_noise_variance)
                .collect::<Vec<_>>(),
        )
        .1
        .sqrt();
        metric.target_mean_mc_batch_spread += sample_mean_variance(&target_batch_means).1.sqrt();
    }
    let divisor = centers as f64;
    DiagnosticMetrics {
        diagonal_mean_error: metric.diagonal_mean_error / divisor,
        full_mean_error: metric.full_mean_error / divisor,
        diagonal_variance_error: metric.diagonal_variance_error / divisor,
        full_variance_error: metric.full_variance_error / divisor,
        target_variance: metric.target_variance / divisor,
        target_signal_variance: metric.target_signal_variance / divisor,
        target_noise_variance: metric.target_noise_variance / divisor,
        sampled_model_variance: metric.sampled_model_variance / divisor,
        target_model_mean_shift: metric.target_model_mean_shift / divisor,
        center_squared_bias: metric.center_squared_bias / divisor,
        target_center_mse: metric.target_center_mse / divisor,
        model_mean_mc_batch_spread: metric.model_mean_mc_batch_spread / divisor,
        model_variance_mc_batch_spread: metric.model_variance_mc_batch_spread / divisor,
        target_variance_mc_batch_spread: metric.target_variance_mc_batch_spread / divisor,
        target_mean_mc_batch_spread: metric.target_mean_mc_batch_spread / divisor,
    }
}

fn diagnostic_mean(trials: &[DiagnosticMetrics], value: fn(DiagnosticMetrics) -> f64) -> f64 {
    trials.iter().copied().map(value).sum::<f64>() / trials.len() as f64
}

fn print_diagnostic_metric(
    label: &str,
    trials: &[DiagnosticMetrics],
    value: fn(DiagnosticMetrics) -> f64,
    full_study: bool,
) {
    let values: Vec<_> = trials.iter().copied().map(value).collect();
    if full_study {
        let ci = mean_ci(&values);
        println!(
            "  {label:<42} {:.5} [{:.5}, {:.5}]",
            ci.mean, ci.lower, ci.upper
        );
    } else {
        println!(
            "  {label:<42} {:.5} (quick descriptive mean)",
            diagnostic_mean(trials, value)
        );
    }
}

fn run_diagnostic(dev: &Device<Ad>, repeats: usize, centers: usize, full_study: bool) {
    println!(
        "conformal diagnostic: repeats={repeats}; train={STUDY_TRAIN}; epochs={STUDY_EPOCHS}; centers/model={centers}; MC={DIAGNOSTIC_MC_BATCHES} independent batches x {DIAGNOSTIC_MC_DRAWS} draws"
    );
    let mut trials = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        println!(
            "  diagnostic fitted-model/data repeat {}/{}",
            repeat + 1,
            repeats
        );
        trials.push(diagnostic_trial(repeat, centers, dev));
    }
    println!("same fitted one-hidden-ReLU model and feature-noise law per center:");
    println!(
        "  diagonal and full rows are paired against the same sampled model-output reference."
    );
    println!("  Full K3 uses a third-order covariance series; its discrepancy from MC combines truncation, numerical, and sampling error.");
    print_diagnostic_metric(
        "diagonal |mean - MC mean|",
        &trials,
        |m| m.diagonal_mean_error,
        full_study,
    );
    print_diagnostic_metric(
        "full K3 |mean - MC mean|",
        &trials,
        |m| m.full_mean_error,
        full_study,
    );
    print_diagnostic_metric(
        "diagonal |variance - MC variance|",
        &trials,
        |m| m.diagonal_variance_error,
        full_study,
    );
    print_diagnostic_metric(
        "full K3 |variance - MC variance|",
        &trials,
        |m| m.full_variance_error,
        full_study,
    );
    print_diagnostic_metric(
        "MC Var(M | x)",
        &trials,
        |m| m.sampled_model_variance,
        full_study,
    );
    println!("  MC batch spreads are standard deviations across the four independent batches, averaged over centers:");
    print_diagnostic_metric(
        "shared model-mean batch spread",
        &trials,
        |m| m.model_mean_mc_batch_spread,
        full_study,
    );
    print_diagnostic_metric(
        "shared model-variance batch spread",
        &trials,
        |m| m.model_variance_mc_batch_spread,
        full_study,
    );
    println!("synthetic target oracle, separate from propagated model sensitivity; c is the diagonal propagated interval center:");
    print_diagnostic_metric(
        "Var(f(x + epsilon) + eta | x)",
        &trials,
        |m| m.target_variance,
        full_study,
    );
    print_diagnostic_metric(
        "Var(f(x + epsilon) | x)",
        &trials,
        |m| m.target_signal_variance,
        full_study,
    );
    print_diagnostic_metric(
        "target variance batch spread",
        &trials,
        |m| m.target_variance_mc_batch_spread,
        full_study,
    );
    print_diagnostic_metric(
        "target mean batch spread",
        &trials,
        |m| m.target_mean_mc_batch_spread,
        full_study,
    );
    print_diagnostic_metric(
        "known independent label-noise variance",
        &trials,
        |m| m.target_noise_variance,
        full_study,
    );
    print_diagnostic_metric(
        "E[Y | x] - E[M | x]",
        &trials,
        |m| m.target_model_mean_shift,
        full_study,
    );
    print_diagnostic_metric(
        "MC-corrected squared bias of c",
        &trials,
        |m| m.center_squared_bias,
        full_study,
    );
    print_diagnostic_metric(
        "MC-corrected E[(Y - c)^2 | x]",
        &trials,
        |m| m.target_center_mse,
        full_study,
    );
    println!("  The MC-corrected estimates can be negative at finite draw budgets; the population squared bias cannot.");
    if full_study {
        println!(
            "mean [t_29-approximate 95% interval across independent fitted-model/data repeats]"
        );
    } else {
        println!("quick mode is descriptive; its four fitted-model/data repeats do not print inferential intervals.");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        corrected_squared_bias, cross_batch_squared_bias, diagnostic_seed, sample_mean_variance,
        study_seed, target_center_mse, STUDY_REPEATS,
    };
    use std::collections::BTreeSet;

    #[test]
    fn study_split_and_model_seeds_are_distinct() {
        let mut all = BTreeSet::new();
        for repeat in 0..STUDY_REPEATS {
            let roles: BTreeSet<u64> = (1..=4).map(|role| study_seed(repeat, role)).collect();
            assert_eq!(roles.len(), 4, "seed roles collided in repeat {repeat}");
            all.extend(roles);
        }
        assert_eq!(all.len(), STUDY_REPEATS * 4);
    }

    #[test]
    fn diagnostic_streams_are_independent() {
        let mut all = BTreeSet::new();
        for repeat in 0..STUDY_REPEATS {
            let roles: BTreeSet<u64> = (1..=5).map(|role| diagnostic_seed(repeat, role)).collect();
            assert_eq!(
                roles.len(),
                5,
                "diagnostic roles collided in repeat {repeat}"
            );
            all.extend(roles);
        }
        assert_eq!(all.len(), STUDY_REPEATS * 5);
    }

    #[test]
    fn constructed_metric_control_has_exact_moments_and_decomposition() {
        let (mean, variance) = sample_mean_variance(&[1.0, 3.0, 5.0]);
        assert_eq!(mean, 3.0);
        assert_eq!(variance, 4.0);
        assert_eq!(target_center_mse(variance, (mean - 1.0).powi(2)), 8.0);
    }

    #[test]
    fn corrected_bias_matches_independent_cross_batch_control() {
        // Zero-mean independent batch estimates around c=0 make the unbiased
        // finite-MC estimate negative. The cross-batch product is an
        // independent construction of the same estimand.
        let batches = [-1.0, 1.0];
        let corrected = corrected_squared_bias(&batches, 0.0);
        let cross_batch = cross_batch_squared_bias(&batches, 0.0);
        assert_eq!(corrected, -1.0);
        assert_eq!(cross_batch, -1.0);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dev = Device::<Ad>::default();
    match args.as_slice() {
        [] => {
            let trial = evaluate(
                make_data(N_TRAIN, 0xC0A1_0001, false, false),
                make_data(N_CAL, 0xC0A1_0002, false, false),
                make_data(N_TEST, 0xC0A1_0003, false, false),
                EPOCHS,
                0xC0A1_1000,
                &dev,
            );
            print_single(trial);
        }
        [flag] if flag == "--study" => run_study(&dev),
        [flag] if flag == "--diagnose" => run_diagnostic(
            &dev,
            DIAGNOSTIC_QUICK_REPEATS,
            DIAGNOSTIC_QUICK_CENTERS,
            false,
        ),
        [flag] if flag == "--diagnose-study" => {
            run_diagnostic(&dev, STUDY_REPEATS, DIAGNOSTIC_STUDY_CENTERS, true)
        }
        _ => panic!("usage: conformal_intervals [--study|--diagnose|--diagnose-study]"),
    }
}
