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

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Device, Tensor, TensorData};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{propagate_linear, propagate_relu, Moments};

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
    println!("target coverage = {:.2}\n", 1.0 - ALPHA);
    println!("  {:<34} {:>8} {:>10}", "method", "coverage", "avg width");
    for (name, metric) in [
        ("raw stableprop (1.645*sigma)", trial.raw),
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
            // calibration/test targets contain one feature perturbation.
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
        "mean [t_29-approximate 95% interval across repeats]; target marginal coverage = {:.2}",
        1.0 - ALPHA
    );
    print_summary("raw stableprop", raw_coverage, raw_width);
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

#[cfg(test)]
mod tests {
    use super::{study_seed, STUDY_REPEATS};
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
        _ => panic!("usage: conformal_intervals [--study]"),
    }
}
