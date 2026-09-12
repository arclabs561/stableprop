//! Measures covariance propagation error as ReLU depth increases.
//!
//! In the full-covariance path, depth one tests the finite ReLU covariance
//! series without repeated Gaussian closure. Each deeper network also
//! approximates the preceding ReLU output as Gaussian. Its extra
//! error therefore combines a new series truncation with repeated Gaussian
//! closure; the depth sweep does not attribute that gap to closure alone.
//!
//! A scalar control below isolates closure: `ReLU(ReLU(X)) == ReLU(X)` for
//! every sample, while repeated Gaussian moment propagation need not preserve
//! those moments after the first ReLU.
//!
//! Run: `cargo run --release --example full_covariance --features burn`

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, Distribution, Tensor};

use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_full, propagate_relu, propagate_relu_full, Moments,
    MomentsFull,
};

const D_IN: usize = 8;
const HIDDEN: usize = 24;
const D_OUT: usize = 4;
const N: usize = 128;
const INPUT_STD: f64 = 0.4;
const MC_SAMPLES: usize = 2_048;
const DEPTHS: [usize; 3] = [1, 2, 3];
const SEEDS: [u64; 3] = [0xF011_C0A1, 0xF011_C0A2, 0xF011_C0A3];
const MODEL_SEED: u64 = 0x4D4F_4445;
const INPUT_SEED: u64 = 0x494E_5054;
const MC_SEED: u64 = 0x4D43_4E4F;
const REPEAT_SEED: u64 = 0x5245_5045;

#[derive(Module, Debug)]
struct Mlp {
    input: Linear,
    hidden1: Linear,
    hidden2: Linear,
    output: Linear,
}

impl Mlp {
    fn init(device: &Device) -> Self {
        let model = Self {
            input: LinearConfig::new(D_IN, HIDDEN).init(device),
            hidden1: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            hidden2: LinearConfig::new(HIDDEN, HIDDEN).init(device),
            output: LinearConfig::new(HIDDEN, D_OUT).init(device),
        };
        // Burn parameters are lazy. Fix every weight before later RNG resets,
        // including layers first used at greater depth.
        for layer in model.hidden_layers(3).chain(std::iter::once(&model.output)) {
            drop(weights(layer));
        }
        model
    }

    fn hidden_layers(&self, depth: usize) -> impl Iterator<Item = &Linear> {
        assert!(DEPTHS.contains(&depth), "unsupported ReLU depth {depth}");
        [&self.input, &self.hidden1, &self.hidden2]
            .into_iter()
            .take(depth)
    }
}

struct Estimate {
    mean: Vec<f64>,
    cov: Vec<f64>,
}

#[derive(Clone, Copy)]
struct ErrorMetrics {
    mean: f64,
    covariance: f64,
    margin: f64,
}

/// Per-row, within-output covariance accumulated in f64 without storing every
/// Monte Carlo output. Each `push` contains one independent draw per input row.
struct OnlineMoments {
    count: usize,
    mean: Vec<f64>,
    cov_m2: Vec<f64>,
}

impl OnlineMoments {
    fn new() -> Self {
        Self {
            count: 0,
            mean: vec![0.0; N * D_OUT],
            cov_m2: vec![0.0; N * D_OUT * D_OUT],
        }
    }

    fn push(&mut self, values: &[f32]) {
        assert_eq!(values.len(), self.mean.len(), "unexpected MC output shape");
        self.count += 1;
        let n = self.count as f64;
        for row in 0..N {
            let mut delta = [0.0; D_OUT];
            for (feature, difference) in delta.iter_mut().enumerate() {
                let index = row * D_OUT + feature;
                *difference = f64::from(values[index]) - self.mean[index];
                self.mean[index] += *difference / n;
            }
            for (left, difference) in delta.iter().enumerate() {
                for right in 0..D_OUT {
                    let right_index = row * D_OUT + right;
                    let covariance_index = (row * D_OUT + left) * D_OUT + right;
                    self.cov_m2[covariance_index] +=
                        difference * (f64::from(values[right_index]) - self.mean[right_index]);
                }
            }
        }
    }

    fn finish(self) -> Estimate {
        assert!(
            self.count > 1,
            "Monte Carlo covariance needs at least two draws"
        );
        let divisor = (self.count - 1) as f64;
        Estimate {
            mean: self.mean,
            cov: self
                .cov_m2
                .into_iter()
                .map(|value| value / divisor)
                .collect(),
        }
    }
}

fn weights(layer: &Linear) -> (Tensor<2>, Tensor<1>) {
    (
        layer.weight.val(),
        layer
            .bias
            .as_ref()
            .expect("LinearConfig enables a bias")
            .val(),
    )
}

fn full_estimate(model: &Mlp, x: Tensor<2>, var: Tensor<2>, depth: usize) -> Estimate {
    let mut hidden = MomentsFull::from_diagonal(x, var);
    for layer in model.hidden_layers(depth) {
        let (w, b) = weights(layer);
        hidden = propagate_relu_full(&propagate_linear_full(&hidden, w, Some(b)));
    }
    let (w_out, b_out) = weights(&model.output);
    let output = propagate_linear_full(&hidden, w_out, Some(b_out));
    Estimate {
        mean: output
            .mean
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect(),
        cov: output
            .cov
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect(),
    }
}

fn diagonal_estimate(model: &Mlp, x: Tensor<2>, var: Tensor<2>, depth: usize) -> Estimate {
    let mut hidden = Moments::new(x, var);
    for layer in model.hidden_layers(depth) {
        let (w, b) = weights(layer);
        hidden = propagate_relu(&propagate_linear(&hidden, w, Some(b)));
    }
    let (w_out, b_out) = weights(&model.output);
    let output = propagate_linear(&hidden, w_out, Some(b_out));
    let mean = output
        .mean
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let variance: Vec<f64> = output
        .var
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let mut cov = vec![0.0; N * D_OUT * D_OUT];
    for row in 0..N {
        for feature in 0..D_OUT {
            cov[(row * D_OUT + feature) * D_OUT + feature] = variance[row * D_OUT + feature];
        }
    }
    Estimate { mean, cov }
}

fn forward(model: &Mlp, x: Tensor<2>, depth: usize) -> Tensor<2> {
    let mut hidden = x;
    for layer in model.hidden_layers(depth) {
        hidden = activation::relu(layer.forward(hidden));
    }
    let (w_out, b_out) = weights(&model.output);
    hidden.matmul(w_out) + b_out.reshape([1, D_OUT])
}

fn monte_carlo(model: &Mlp, x: &Tensor<2>, depth: usize, dev: &Device) -> Estimate {
    let mut moments = OnlineMoments::new();
    for _ in 0..MC_SAMPLES {
        let noise = Tensor::<2>::random([N, D_IN], Distribution::Normal(0.0, INPUT_STD), dev);
        let values = forward(model, x.clone() + noise, depth)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        moments.push(&values);
    }
    moments.finish()
}

fn normalized_covariance_error(estimate: &[f64], reference: &[f64]) -> f64 {
    assert_eq!(
        estimate.len(),
        reference.len(),
        "covariance shapes must match"
    );
    let squared_error = estimate
        .iter()
        .zip(reference)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>();
    let reference_norm = reference.iter().map(|value| value.powi(2)).sum::<f64>();
    assert!(reference_norm > 0.0, "Monte Carlo covariance norm is zero");
    squared_error.sqrt() / reference_norm.sqrt()
}

fn normalized_mean_error(estimate: &Estimate, reference: &Estimate) -> f64 {
    let squared_error = estimate
        .mean
        .iter()
        .zip(&reference.mean)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>();
    let output_scale = (0..N)
        .flat_map(|row| (0..D_OUT).map(move |feature| (row * D_OUT + feature) * D_OUT + feature))
        .map(|index| reference.cov[index])
        .sum::<f64>();
    assert!(output_scale > 0.0, "Monte Carlo output variance is zero");
    squared_error.sqrt() / output_scale.sqrt()
}

fn margin_standard_deviations(cov: &[f64]) -> Vec<f64> {
    (0..N)
        .map(|row| {
            let base = row * D_OUT * D_OUT;
            // Compute w^T Sigma w for w = [1, -1] from both stored
            // off-diagonal entries. Their low-bit difference is representational
            // roundoff, but choosing only Sigma_01 would make the diagnostic
            // depend on matrix orientation.
            let variance = cov[base] + cov[base + D_OUT + 1] - cov[base + 1] - cov[base + D_OUT];
            assert!(
                variance >= 0.0,
                "output margin variance is negative: {variance:e}"
            );
            variance.sqrt()
        })
        .collect()
}

fn normalized_margin_std_error(estimate: &Estimate, reference: &Estimate) -> f64 {
    let estimate = margin_standard_deviations(&estimate.cov);
    let reference = margin_standard_deviations(&reference.cov);
    let squared_error = estimate
        .iter()
        .zip(&reference)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>();
    let reference_scale = reference.iter().map(|value| value.powi(2)).sum::<f64>();
    assert!(
        reference_scale > 0.0,
        "Monte Carlo margin standard deviation is zero"
    );
    squared_error.sqrt() / reference_scale.sqrt()
}

fn error_metrics(estimate: &Estimate, reference: &Estimate) -> ErrorMetrics {
    ErrorMetrics {
        mean: normalized_mean_error(estimate, reference),
        covariance: normalized_covariance_error(&estimate.cov, &reference.cov),
        margin: normalized_margin_std_error(estimate, reference),
    }
}

fn descriptive_summary(values: &[f64]) -> (f64, f64, f64) {
    assert!(
        !values.is_empty(),
        "a depth summary needs at least one seed"
    );
    assert!(values.iter().all(|value| value.is_finite()));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

fn scalar_closure_control(dev: &Device) {
    let expected_mean = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let expected_var = 0.5 - 1.0 / (2.0 * std::f64::consts::PI);
    let mut moments = Moments::new(Tensor::zeros([1, 1], dev), Tensor::ones([1, 1], dev));
    println!("scalar ReLU closure control (exact mean {expected_mean:.6}, var {expected_var:.6}):");
    for step in 1..=3 {
        moments = propagate_relu(&moments);
        let mean = f64::from(moments.mean.to_data().try_to_vec::<f32>().unwrap()[0]);
        let variance = f64::from(moments.var.to_data().try_to_vec::<f32>().unwrap()[0]);
        if step == 1 {
            assert!(
                (mean - expected_mean).abs() < 2e-6,
                "first ReLU mean must be exact"
            );
            assert!(
                (variance - expected_var).abs() < 2e-6,
                "first ReLU variance must be exact"
            );
        }
        println!("  propagated step {step}: mean {mean:.6}, var {variance:.6}");
    }
}

fn main() {
    let dev = Device::flex();
    scalar_closure_control(&dev);
    println!("\ndepth sweep: {N} centers, {MC_SAMPLES} draws per Monte Carlo estimate");
    println!(
        "  {:<10} {:>5} {:<9} {:>10} {:>10} {:>10}",
        "seed", "depth", "method", "mean nRMS", "cov nFrob", "margin nRMS"
    );
    // Each inner vector holds [full, diagonal, MC repeat] errors for one depth.
    let mut depth_rows = (0..DEPTHS.len())
        .map(|_| Vec::<[ErrorMetrics; 3]>::with_capacity(SEEDS.len()))
        .collect::<Vec<_>>();
    for &seed in &SEEDS {
        // All depths receive the same initialized prefix and the same output map.
        dev.seed(seed ^ MODEL_SEED);
        let model = Mlp::init(&dev);
        // Input centers are independent of model initialization and shared across depths.
        dev.seed(seed ^ INPUT_SEED);
        let x = Tensor::<2>::random([N, D_IN], Distribution::Normal(0.0, 1.0), &dev);
        for (depth_index, &depth) in DEPTHS.iter().enumerate() {
            let variance = Tensor::<2>::full([N, D_IN], INPUT_STD * INPUT_STD, &dev);
            let full = full_estimate(&model, x.clone(), variance.clone(), depth);
            let diagonal = diagonal_estimate(&model, x.clone(), variance, depth);
            // Re-seeding makes the perturbation sequence identical at every depth.
            dev.seed(seed ^ MC_SEED);
            let reference = monte_carlo(&model, &x, depth, &dev);
            // An independent, equally sized estimate shows sampling variability.
            // This stream is also shared across depths, not across input rows.
            dev.seed(seed ^ REPEAT_SEED);
            let repeat = monte_carlo(&model, &x, depth, &dev);
            let rows = [
                ("full", error_metrics(&full, &reference)),
                ("diagonal", error_metrics(&diagonal, &reference)),
                ("MC repeat", error_metrics(&repeat, &reference)),
            ];
            for (name, metrics) in rows {
                println!(
                    "  {seed:08x} {depth:>5} {name:<9} {:>10.4} {:>10.4} {:>10.4}",
                    metrics.mean, metrics.covariance, metrics.margin,
                );
            }
            depth_rows[depth_index].push([rows[0].1, rows[1].1, rows[2].1]);
        }
    }
    println!("\ndescriptive depth summaries across the three fixed seeds (mean [min, max]):");
    println!(
        "  {:>5} {:<9} {:>22} {:>22} {:>22}",
        "depth", "method", "mean nRMS", "cov nFrob", "margin nRMS"
    );
    for (depth_index, rows) in depth_rows.iter().enumerate() {
        for (method_index, method) in ["full", "diagonal", "MC repeat"].iter().enumerate() {
            let mean = descriptive_summary(
                &rows
                    .iter()
                    .map(|metrics| metrics[method_index].mean)
                    .collect::<Vec<_>>(),
            );
            let covariance = descriptive_summary(
                &rows
                    .iter()
                    .map(|metrics| metrics[method_index].covariance)
                    .collect::<Vec<_>>(),
            );
            let margin = descriptive_summary(
                &rows
                    .iter()
                    .map(|metrics| metrics[method_index].margin)
                    .collect::<Vec<_>>(),
            );
            println!(
                "  {:>5} {method:<9} {:>7.4} [{:>6.4}, {:>6.4}] {:>7.4} [{:>6.4}, {:>6.4}] {:>7.4} [{:>6.4}, {:>6.4}]",
                DEPTHS[depth_index],
                mean.0,
                mean.1,
                mean.2,
                covariance.0,
                covariance.1,
                covariance.2,
                margin.0,
                margin.1,
                margin.2,
            );
        }
    }
    println!("\nmean nRMS scales output-mean error by aggregate MC output standard deviation.");
    println!("margin nRMS uses the fixed output-0 minus output-1 standard deviation.");
    println!("Compare full and diagonal with the MC-repeat row at the same depth.");
    println!(
        "The three fixed-seed summaries are descriptive, not population confidence intervals."
    );
    println!(
        "MC repeat compares two independent estimates; it is a sampling diagnostic, not an error bound."
    );
    println!(
        "Depth 1 has no repeated Gaussian closure; deeper rows include closure and further series approximations."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_seeds_do_not_change_the_initialized_network() {
        let dev = Device::flex();
        let predictions = [17, 29].map(|sampling_seed| {
            dev.seed(MODEL_SEED);
            let model = Mlp::init(&dev);
            dev.seed(sampling_seed);
            DEPTHS.map(|depth| {
                forward(&model, Tensor::from_data([[0.25; D_IN]], &dev), depth)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
            })
        });
        assert_eq!(predictions[0], predictions[1]);
    }

    #[test]
    fn margin_uses_the_full_quadratic_form_under_roundoff_asymmetry() {
        // A symmetric positive-semidefinite covariance with off-diagonal
        // correlation just below one can acquire different low bits in its two
        // stored entries after f32 transport. The quadratic form must use both
        // entries, and must be invariant to transposition.
        let epsilon = f64::from(f32::EPSILON);
        let mut covariance = vec![0.0; N * D_OUT * D_OUT];
        for row in 0..N {
            let base = row * D_OUT * D_OUT;
            covariance[base] = 1.0;
            covariance[base + D_OUT + 1] = 1.0;
            covariance[base + 1] = 1.0 - 4.0 * epsilon;
            covariance[base + D_OUT] = 1.0 - 2.0 * epsilon;
        }
        let expected_variance = 6.0 * epsilon;
        let standard_deviation = margin_standard_deviations(&covariance)[0];
        assert_eq!(standard_deviation, expected_variance.sqrt());

        let mut transpose = covariance.clone();
        for row in 0..N {
            let base = row * D_OUT * D_OUT;
            transpose.swap(base + 1, base + D_OUT);
        }
        assert_eq!(
            margin_standard_deviations(&transpose)[0],
            standard_deviation
        );

        let w = [1.0, -1.0];
        let direct = w[0] * covariance[0] * w[0]
            + w[0] * covariance[1] * w[1]
            + w[1] * covariance[D_OUT] * w[0]
            + w[1] * covariance[D_OUT + 1] * w[1];
        assert_eq!(direct, expected_variance);
    }

    #[test]
    fn sampled_affine_outputs_have_known_covariance_and_margin_error() {
        let slopes = [1.0, -2.0, 0.5, 3.0];
        let mut accumulator = OnlineMoments::new();
        // Four centered scalar observations have sample variance 20/3.
        // Offsets test centered accumulation; signed slopes test cross terms.
        for t in [-3.0, -1.0, 1.0, 3.0] {
            let values = (0..N)
                .flat_map(|row| {
                    slopes
                        .iter()
                        .map(move |slope| 10_000.0 + 16.0 * row as f32 + slope * t)
                })
                .collect::<Vec<_>>();
            accumulator.push(&values);
        }
        let reference = accumulator.finish();
        for row in 0..N {
            for left in 0..D_OUT {
                assert!(
                    (reference.mean[row * D_OUT + left] - (10_000.0 + 16.0 * row as f64)).abs()
                        < 1e-10
                );
                for right in 0..D_OUT {
                    let expected = f64::from(slopes[left] * slopes[right]) * 20.0 / 3.0;
                    assert!(
                        (reference.cov[(row * D_OUT + left) * D_OUT + right] - expected).abs()
                            < 1e-10
                    );
                }
            }
        }
        // Scaling covariance by four doubles every margin standard deviation.
        let estimate = Estimate {
            mean: reference.mean.iter().map(|mean| mean + 1.0).collect(),
            cov: reference.cov.iter().map(|value| 4.0 * value).collect(),
        };
        assert!((normalized_covariance_error(&estimate.cov, &reference.cov) - 3.0).abs() < 1e-12);
        assert!((normalized_margin_std_error(&estimate, &reference) - 1.0).abs() < 1e-12);
        // Sum of marginal variances = (1 + 4 + 1/4 + 9) * 20/3 = 95 per row.
        assert!(
            (normalized_mean_error(&estimate, &reference) - (4.0f64 / 95.0).sqrt()).abs() < 1e-12
        );
        assert!((margin_standard_deviations(&reference.cov)[0] - 60.0f64.sqrt()).abs() < 1e-10);
    }
}
