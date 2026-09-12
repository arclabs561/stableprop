//! Estimates classifier error under Gaussian input noise from propagated
//! logit covariance and compares it with Monte Carlo.
//!
//! For true class `t`, competitor `j` wins when `logit_t - logit_j` is negative.
//! Its variance is `S_tt + S_jj - S_tj - S_jt`. Summing Gaussian margin-tail
//! probabilities gives the risk estimate.
//! A deterministic tie contributes zero to this strict-margin estimate;
//! classifier tie-breaking can differ.
//! Competing margin events can overlap, so their summed estimate is not an
//! exact calibration quantity even with exact Gaussian margin probabilities.
//!
//! Nonlinear propagation and the Gaussian-logit assumption are approximate,
//! so this estimate can fall above or below the true per-input error rate.
//! It is not a robustness certificate. Evaluation uses each test point's true
//! class, which is unavailable for deployment-time ranking.
//!
//! Run: `cargo run --release --example misclassification_risk --features burn`

use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{activation, Device, Int, Tensor, TensorData};

use stableprop::burn_sdp::{propagate_linear_full, propagate_relu_full, MomentsFull};

const D_IN: usize = 4;
const HIDDEN: usize = 32;
const N_CLASS: usize = 3;
const N_TRAIN: usize = 3000;
const N_TEST: usize = 400;
const INPUT_STD: f64 = 0.25;
const MC_SAMPLES: usize = 400;
const RISK_BIN_UPPER: [f64; 5] = [0.02, 0.05, 0.10, 0.20, 1.0];

#[derive(Module, Debug)]
struct Net {
    lin1: Linear,
    lin2: Linear,
}

impl Net {
    fn init(device: &Device) -> Self {
        Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, N_CLASS).init(device),
        }
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.lin2.forward(activation::relu(self.lin1.forward(x)))
    }
}

fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}
fn phi_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// `Pr[margin < 0]` under a Gaussian margin approximation.
///
/// A zero-variance margin is deterministic. Small negative values can arise
/// when f32 covariance terms cancel, but a material negative variance signals
/// invalid propagated moments and must not be turned into a plausible risk.
fn gaussian_margin_loss_probability(mean: f64, covariance_terms: [f64; 4]) -> f64 {
    assert!(mean.is_finite(), "margin mean must be finite");
    let variance =
        covariance_terms[0] + covariance_terms[1] - covariance_terms[2] - covariance_terms[3];
    assert!(variance.is_finite(), "margin variance must be finite");
    let scale = covariance_terms.iter().map(|term| term.abs()).sum::<f64>();
    let roundoff_tolerance = 64.0 * f64::from(f32::EPSILON) * scale;
    assert!(
        variance >= -roundoff_tolerance,
        "margin variance {variance:e} is materially negative (tolerance {roundoff_tolerance:e})"
    );
    let variance = variance.max(0.0);
    if variance == 0.0 {
        return if mean < 0.0 { 1.0 } else { 0.0 };
    }
    phi_cdf(-mean / variance.sqrt())
}

fn risk_bin_index(risk: f64) -> usize {
    assert!((0.0..=1.0).contains(&risk), "risk must lie in [0, 1]");
    RISK_BIN_UPPER
        .iter()
        .position(|upper| risk < *upper)
        .unwrap_or(RISK_BIN_UPPER.len() - 1)
}

/// Estimated Monte Carlo standard error of a mean across fixed inputs.
///
/// Each input has its own Bernoulli error probability, so pooling all outcomes
/// would include between-input variation that is irrelevant to repeated draws
/// at these fixed centers. The per-input sample variance divided by the number
/// of draws estimates each sampled-rate variance.
fn fixed_input_mc_standard_error(error_counts: &[usize], draws_per_input: usize) -> f64 {
    assert!(
        !error_counts.is_empty(),
        "a fixed-input Monte Carlo bin needs at least one input"
    );
    assert!(
        draws_per_input > 1,
        "fixed-input Monte Carlo needs at least two draws"
    );
    let variance = error_counts
        .iter()
        .map(|errors| {
            assert!(
                *errors <= draws_per_input,
                "errors cannot exceed draws per input"
            );
            let probability = *errors as f64 / draws_per_input as f64;
            probability * (1.0 - probability) / (draws_per_input - 1) as f64
        })
        .sum::<f64>()
        / (error_counts.len() as f64).powi(2);
    variance.sqrt()
}

#[derive(Default)]
struct RiskBinSummary {
    count: usize,
    predicted_sum: f64,
    mc_errors: usize,
    input_error_counts: Vec<usize>,
}

/// Class-conditional Gaussian blobs: balanced classes, blob `c` shifted on two
/// features so the classes separate.
fn make(n: usize, dev: &Device) -> (Vec<f32>, Vec<i32>) {
    let mut x = Tensor::<2>::random([n, D_IN], burn::tensor::Distribution::Normal(0.0, 0.6), dev)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let lab: Vec<i32> = (0..n).map(|i| (i % N_CLASS) as i32).collect();
    for i in 0..n {
        let c = lab[i] as f32;
        x[i * D_IN] += 1.6 * c;
        x[i * D_IN + 1] -= 1.2 * c;
    }
    (x, lab)
}

fn main() {
    let dev = Device::flex().autodiff();
    dev.seed(0xA115_C1A5);
    let idev = Device::flex();
    let (xtr, ytr) = make(N_TRAIN, &dev);
    let (xte, yte) = make(N_TEST, &dev);

    let x_train = Tensor::<2>::from_data(TensorData::new(xtr, [N_TRAIN, D_IN]), &dev);
    let y_train = Tensor::<1, Int>::from_data(TensorData::new(ytr, [N_TRAIN]), &dev);

    let mut model = Net::init(&dev);
    let mut optim = AdamConfig::new().init();
    println!("training classifier...");
    for _ in 0..400 {
        let logits = model.forward(x_train.clone());
        let loss = CrossEntropyLoss::new(None, &dev).forward(logits, y_train.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-2, model, grads);
    }

    // Full-covariance propagation of input noise -> joint logit Gaussian.
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val().inner());
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val().inner());
    let x_te = Tensor::<2>::from_data(TensorData::new(xte.clone(), [N_TEST, D_IN]), &idev);
    let var0 = Tensor::<2>::full([N_TEST, D_IN], INPUT_STD * INPUT_STD, &idev);
    let m0 = MomentsFull::from_diagonal(x_te.clone(), var0);
    let m1 = propagate_relu_full(&propagate_linear_full(&m0, w1.clone(), b1.clone()));
    let m2 = propagate_linear_full(&m1, w2.clone(), b2.clone());
    let mean = m2.mean.to_data().try_to_vec::<f32>().unwrap(); // [N_TEST * C]
    let cov = m2.cov.to_data().try_to_vec::<f32>().unwrap(); // [N_TEST * C * C]

    // Test-only per-input risk estimate: a sum of approximate Gaussian margin
    // probabilities for the known true class, capped at one. Competing margin
    // events can overlap, so this construction is not exact calibration even
    // when every Gaussian margin probability were exact.
    let c = N_CLASS;
    let mut bound = vec![0.0f64; N_TEST];
    for i in 0..N_TEST {
        let t = yte[i] as usize;
        let mu = |k: usize| mean[i * c + k] as f64;
        let s = |a: usize, b: usize| cov[i * c * c + a * c + b] as f64;
        let mut p = 0.0;
        for j in 0..c {
            if j == t {
                continue;
            }
            p += gaussian_margin_loss_probability(
                mu(t) - mu(j),
                [s(t, t), s(j, j), s(t, j), s(j, t)],
            );
        }
        bound[i] = p.min(1.0);
    }

    // Monte-Carlo misclassification rate per input.
    let mut mc_errors = vec![0usize; N_TEST];
    for _ in 0..MC_SAMPLES {
        let noise = Tensor::<2>::random(
            [N_TEST, D_IN],
            burn::tensor::Distribution::Normal(0.0, INPUT_STD),
            &idev,
        );
        let logits = (activation::relu(
            (x_te.clone() + noise).matmul(w1.clone()) + b1.clone().unwrap().reshape([1, HIDDEN]),
        )
        .matmul(w2.clone())
            + b2.clone().unwrap().reshape([1, N_CLASS]))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
        for i in 0..N_TEST {
            let row = &logits[i * c..(i + 1) * c];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred != yte[i] as usize {
                mc_errors[i] += 1;
            }
        }
    }
    let mc: Vec<f64> = mc_errors
        .iter()
        .map(|errors| *errors as f64 / MC_SAMPLES as f64)
        .collect();

    let mean_bound = bound.iter().sum::<f64>() / N_TEST as f64;
    let mean_mc = mc.iter().sum::<f64>() / N_TEST as f64;
    let mean_abs_error = (0..N_TEST).map(|i| (bound[i] - mc[i]).abs()).sum::<f64>() / N_TEST as f64;

    println!("\nanalytic misclassification-risk estimate under input noise std {INPUT_STD}:");
    println!("  mean analytic estimate = {mean_bound:.4}");
    println!("  mean MC estimate       = {mean_mc:.4}  ({MC_SAMPLES} samples per input)");
    println!("  mean absolute difference from MC = {mean_abs_error:.4}");
    println!("  The comparison includes Monte Carlo sampling error.");

    let mut bins = (0..RISK_BIN_UPPER.len())
        .map(|_| RiskBinSummary::default())
        .collect::<Vec<_>>();
    for (&risk, &errors) in bound.iter().zip(&mc_errors) {
        let summary = &mut bins[risk_bin_index(risk)];
        summary.count += 1;
        summary.predicted_sum += risk;
        summary.mc_errors += errors;
        summary.input_error_counts.push(errors);
    }
    println!("\nfixed risk bins (known labels are evaluation-only):");
    println!(
        "  {:>11} {:>6} {:>12} {:>12} {:>10}",
        "predicted", "count", "mean risk", "MC error", "fixed MC SE"
    );
    let mut lower = 0.0;
    for (index, summary) in bins.iter().enumerate() {
        let upper = RISK_BIN_UPPER[index];
        let closing = if index + 1 == RISK_BIN_UPPER.len() {
            "]"
        } else {
            ")"
        };
        if summary.count == 0 {
            println!(
                "  [{lower:.2}, {upper:.2}{closing} {:>6} {:>12} {:>12} {:>10}",
                0, "n/a", "n/a", "n/a"
            );
        } else {
            let trials = summary.count * MC_SAMPLES;
            let predicted = summary.predicted_sum / summary.count as f64;
            let sampled = summary.mc_errors as f64 / trials as f64;
            assert_eq!(summary.input_error_counts.len(), summary.count);
            let mc_se = fixed_input_mc_standard_error(&summary.input_error_counts, MC_SAMPLES);
            println!(
                "  [{lower:.2}, {upper:.2}{closing} {:>6} {predicted:>12.4} {sampled:>12.4} {mc_se:>10.4}",
                summary.count
            );
        }
        lower = upper;
    }
    println!(
        "  Fixed MC SE estimates repeated-draw uncertainty at these input centers; it excludes dataset and model uncertainty."
    );
    println!(
        "  It is not a confidence bound and can be zero when observed per-input errors are all zero or one."
    );
    println!(
        "  Summed margin events can overlap, so these rows are not an exact calibration check."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_affine_scores_have_deterministic_margin_risk() {
        let device = Device::flex();
        let moments = MomentsFull::from_diagonal(
            Tensor::from_data([[0.5, -0.25]], &device),
            Tensor::from_data([[0.25, 0.5]], &device),
        );
        let identical_scores = Tensor::from_data([[1.0, 1.0], [-2.0, -2.0]], &device);
        let propagated = propagate_linear_full(&moments, identical_scores, None);
        let covariance = propagated.cov.to_data().try_to_vec::<f32>().unwrap();
        let terms = [
            f64::from(covariance[0]),
            f64::from(covariance[3]),
            f64::from(covariance[1]),
            f64::from(covariance[2]),
        ];

        assert_eq!(gaussian_margin_loss_probability(0.25, terms), 0.0);
        assert_eq!(gaussian_margin_loss_probability(-0.25, terms), 1.0);
        assert_eq!(gaussian_margin_loss_probability(0.0, terms), 0.0);
    }

    #[test]
    fn one_standard_deviation_margin_has_gaussian_tail_probability() {
        let probability = gaussian_margin_loss_probability(1.0, [1.0, 0.0, 0.0, 0.0]);
        assert!((probability - 0.158_655).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "margin variance must be finite")]
    fn rejects_nonfinite_margin_variance() {
        gaussian_margin_loss_probability(0.0, [f64::NAN, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "materially negative")]
    fn rejects_materially_negative_margin_variance() {
        gaussian_margin_loss_probability(0.0, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn fixed_risk_bins_cover_the_unit_interval() {
        assert_eq!(risk_bin_index(0.0), 0);
        assert_eq!(risk_bin_index(0.02), 1);
        assert_eq!(risk_bin_index(0.20), RISK_BIN_UPPER.len() - 1);
        assert_eq!(risk_bin_index(1.0), RISK_BIN_UPPER.len() - 1);
    }

    #[test]
    fn fixed_input_mc_uncertainty_drops_between_input_variation() {
        // Fixed inputs with deterministic but different error probabilities
        // have no repeated-draw uncertainty. A pooled Bernoulli calculation
        // would incorrectly report a nonzero value here.
        assert_eq!(fixed_input_mc_standard_error(&[0, 4], 4), 0.0);
    }

    #[test]
    fn fixed_input_mc_uncertainty_matches_indicator_sample_variances() {
        let indicators = [[true, false, false, false], [true, true, false, false]];
        let counts = indicators.map(|draws| draws.into_iter().filter(|error| *error).count());
        // The two per-input sample variances are 1/4 and 1/3. Dividing each
        // by four draws and averaging two independent input rates gives 7/192.
        assert!(
            (fixed_input_mc_standard_error(&counts, 4) - (7.0f64 / 192.0).sqrt()).abs() < 1e-12
        );
    }
}
