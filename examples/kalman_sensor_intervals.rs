//! Calibrated target intervals from a known two-channel sensor model.
//!
//! This is a controlled state-estimation example.  A latent two-dimensional
//! state follows a linear Gaussian transition and is observed by two sensors.
//! The sensor covariance is known because it is part of the simulator: a shared
//! per-reading error produces an off-diagonal covariance in addition to each
//! channel's independent noise.
//!
//! A one-hidden-layer ReLU surrogate learns a quadratic terminal quantity from
//! clean simulated states.  At evaluation time, the Kalman posterior for each
//! independently simulated trajectory supplies the input mean and covariance.
//! The quadratic simulator has exact posterior target moments, so this example
//! separates (1) propagated surrogate moments from matched surrogate Monte
//! Carlo (with a second stream to show sampling variation), and (2) surrogate
//! discrepancy from the simulator oracle, including Monte Carlo error. It compares
//! constant, propagated-scale, and oracle-center-and-scale split-conformal target
//! intervals.  Raw Gaussian moment bands are descriptive only: the quadratic
//! target distribution is not Gaussian.
//!
//! Run a small descriptive study:
//! `cargo run --release --features burn --example kalman_sensor_intervals -- --quick`
//! Run 20 independent fitted-model/split studies:
//! `cargo run --release --features burn --example kalman_sensor_intervals -- --study`

use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{activation, Device, Tensor, TensorData};
use statskit::conformal::{calibrate_in_place, Coverage, Threshold};

use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_full, propagate_relu, propagate_relu_full, Moments,
    MomentsFull,
};

const DIM: usize = 2;
const HIDDEN: usize = 24;
const OBSERVATIONS: usize = 6;
const HORIZON: usize = 2;
const ALPHA: f64 = 0.1;
const LABEL_VARIANCE: f64 = 0.04;
const SCALE_FLOOR: f64 = 1e-6;

const QUICK_REPEATS: usize = 3;
const FULL_REPEATS: usize = 20;
const QUICK_TRAIN: usize = 256;
const QUICK_CAL: usize = 96;
const QUICK_TEST: usize = 192;
const QUICK_EPOCHS: usize = 80;
const FULL_TRAIN: usize = 768;
const FULL_CAL: usize = 256;
const FULL_TEST: usize = 512;
const FULL_EPOCHS: usize = 300;
const QUICK_MC_DRAWS: usize = 512;
const FULL_MC_DRAWS: usize = 1024;

/// The latent transition and process covariance are known simulator inputs.
const F: [[f64; DIM]; DIM] = [[0.85, 0.20], [0.0, 0.75]];
const Q: [[f64; DIM]; DIM] = [[0.12, 0.03], [0.03, 0.16]];
const P0: [[f64; DIM]; DIM] = [[1.0, 0.25], [0.25, 0.70]];

/// `R = common per-reading covariance + independent channel covariance`.
/// It is the covariance of one two-channel reading, not a covariance divided
/// by a number of readings.  The Kalman update consumes one reading at a time.
const R: [[f64; DIM]; DIM] = [[0.08, 0.02], [0.02, 0.11]];

/// Symmetric A gives `terminal(x) = x^T A x`.
const A: [[f64; DIM]; DIM] = [[0.90, 0.25], [0.25, 0.50]];

#[derive(Module, Debug)]
struct Mlp {
    first: Linear,
    last: Linear,
}

impl Mlp {
    fn init(device: &Device) -> Self {
        Self {
            first: LinearConfig::new(DIM, HIDDEN).init(device),
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

#[derive(Clone, Copy)]
struct Posterior {
    mean: [f64; DIM],
    covariance: [[f64; DIM]; DIM],
}

#[derive(Clone, Copy)]
struct Trajectory {
    posterior: Posterior,
    target: f64,
}

#[derive(Clone, Copy, Default)]
struct MomentsEstimate {
    mean: f64,
    variance: f64,
}

#[derive(Clone, Copy, Default)]
struct IntervalMetrics {
    coverage: f64,
    width: f64,
}

#[derive(Clone, Copy, Default)]
struct Trial {
    full_mean_mc_mae: f64,
    full_variance_mc_mae: f64,
    mc_repeat_mean_mae: f64,
    mc_repeat_variance_mae: f64,
    diagonal_mean_mc_mae: f64,
    diagonal_variance_mc_mae: f64,
    model_oracle_mean_mae: f64,
    model_oracle_variance_mae: f64,
    raw: IntervalMetrics,
    constant: IntervalMetrics,
    propagated: IntervalMetrics,
    oracle: IntervalMetrics,
}

fn add(left: [[f64; DIM]; DIM], right: [[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    [
        [left[0][0] + right[0][0], left[0][1] + right[0][1]],
        [left[1][0] + right[1][0], left[1][1] + right[1][1]],
    ]
}

fn mat_vec(matrix: [[f64; DIM]; DIM], vector: [f64; DIM]) -> [f64; DIM] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

fn mat_mul(left: [[f64; DIM]; DIM], right: [[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]
}

fn transpose(matrix: [[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]
}

fn inverse(matrix: [[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    assert!(
        determinant.is_finite() && determinant > 0.0,
        "matrix must be positive definite"
    );
    [
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ]
}

fn cholesky(matrix: [[f64; DIM]; DIM]) -> [[f64; DIM]; DIM] {
    assert!(
        (matrix[0][1] - matrix[1][0]).abs() < 1e-12,
        "covariance must be symmetric"
    );
    let l00 = matrix[0][0].sqrt();
    let l10 = matrix[1][0] / l00;
    let l11_squared = matrix[1][1] - l10 * l10;
    assert!(l00.is_finite() && l11_squared.is_finite() && l00 > 0.0 && l11_squared > 0.0);
    [[l00, 0.0], [l10, l11_squared.sqrt()]]
}

fn sample_gaussian(mean: [f64; DIM], covariance: [[f64; DIM]; DIM], rng: &mut Rng) -> [f64; DIM] {
    let l = cholesky(covariance);
    let z = [rng.normal(), rng.normal()];
    [
        mean[0] + l[0][0] * z[0],
        mean[1] + l[1][0] * z[0] + l[1][1] * z[1],
    ]
}

fn terminal(state: [f64; DIM]) -> f64 {
    state[0] * (A[0][0] * state[0] + A[0][1] * state[1])
        + state[1] * (A[1][0] * state[0] + A[1][1] * state[1])
}

fn kalman_predict(posterior: Posterior) -> Posterior {
    Posterior {
        mean: mat_vec(F, posterior.mean),
        covariance: add(mat_mul(mat_mul(F, posterior.covariance), transpose(F)), Q),
    }
}

fn kalman_update_with_noise(
    prediction: Posterior,
    reading: [f64; DIM],
    reading_covariance: [[f64; DIM]; DIM],
) -> Posterior {
    let innovation_covariance = add(prediction.covariance, reading_covariance);
    let gain = mat_mul(prediction.covariance, inverse(innovation_covariance));
    let innovation = [
        reading[0] - prediction.mean[0],
        reading[1] - prediction.mean[1],
    ];
    let correction = mat_vec(gain, innovation);
    let identity_minus_gain = [
        [1.0 - gain[0][0], -gain[0][1]],
        [-gain[1][0], 1.0 - gain[1][1]],
    ];
    let covariance = mat_mul(identity_minus_gain, prediction.covariance);
    Posterior {
        mean: [
            prediction.mean[0] + correction[0],
            prediction.mean[1] + correction[1],
        ],
        covariance: [
            [
                covariance[0][0],
                0.5 * (covariance[0][1] + covariance[1][0]),
            ],
            [
                0.5 * (covariance[0][1] + covariance[1][0]),
                covariance[1][1],
            ],
        ],
    }
}

fn kalman_update(prediction: Posterior, reading: [f64; DIM]) -> Posterior {
    kalman_update_with_noise(prediction, reading, R)
}

fn horizon_posterior(mut posterior: Posterior) -> Posterior {
    for _ in 0..HORIZON {
        posterior = kalman_predict(posterior);
    }
    posterior
}

fn terminal_prior() -> Posterior {
    let mut prior = Posterior {
        mean: [0.0; DIM],
        covariance: P0,
    };
    for _ in 0..(OBSERVATIONS + HORIZON) {
        prior = kalman_predict(prior);
    }
    prior
}

fn oracle_moments(posterior: Posterior) -> MomentsEstimate {
    let ap = mat_mul(A, posterior.covariance);
    let apa = mat_mul(ap, A);
    let mean = terminal(posterior.mean) + ap[0][0] + ap[1][1];
    let quadratic_variance = 4.0
        * (posterior.mean[0] * (apa[0][0] * posterior.mean[0] + apa[0][1] * posterior.mean[1])
            + posterior.mean[1] * (apa[1][0] * posterior.mean[0] + apa[1][1] * posterior.mean[1]))
        + 2.0 * (ap[0][0] * ap[0][0] + 2.0 * ap[0][1] * ap[1][0] + ap[1][1] * ap[1][1]);
    MomentsEstimate {
        mean,
        variance: quadratic_variance + LABEL_VARIANCE,
    }
}

fn simulate_trajectory(rng: &mut Rng) -> Trajectory {
    let mut state = sample_gaussian([0.0; DIM], P0, rng);
    let mut posterior = Posterior {
        mean: [0.0; DIM],
        covariance: P0,
    };
    for _ in 0..OBSERVATIONS {
        state = mat_vec(F, state);
        let process = sample_gaussian([0.0; DIM], Q, rng);
        state[0] += process[0];
        state[1] += process[1];
        let noise = sample_gaussian([0.0; DIM], R, rng);
        let reading = [state[0] + noise[0], state[1] + noise[1]];
        posterior = kalman_update(kalman_predict(posterior), reading);
    }
    for _ in 0..HORIZON {
        state = mat_vec(F, state);
        let process = sample_gaussian([0.0; DIM], Q, rng);
        state[0] += process[0];
        state[1] += process[1];
    }
    Trajectory {
        posterior: horizon_posterior(posterior),
        target: terminal(state) + LABEL_VARIANCE.sqrt() * rng.normal(),
    }
}

fn trajectories(count: usize, seed: u64) -> Vec<Trajectory> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| simulate_trajectory(&mut rng)).collect()
}

fn clean_training_data(count: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut inputs = Vec::with_capacity(count * DIM);
    let mut labels = Vec::with_capacity(count);
    let prior = terminal_prior();
    for _ in 0..count {
        let state = sample_gaussian(prior.mean, prior.covariance, &mut rng);
        inputs.extend(state.map(|value| value as f32));
        labels.push(terminal(state) as f32);
    }
    (inputs, labels)
}

fn train_model(rows: usize, epochs: usize, seed: u64, device: &Device) -> Mlp {
    let (inputs, labels) = clean_training_data(rows, seed ^ 0xA11C_E001);
    let x = Tensor::<2>::from_data(TensorData::new(inputs, [rows, DIM]), device);
    let y = Tensor::<2>::from_data(TensorData::new(labels, [rows, 1]), device);
    device.seed(seed ^ 0xA11C_E002);
    let mut model = Mlp::init(device);
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..epochs {
        let loss = MseLoss::new().forward(model.forward(x.clone()), y.clone(), Reduction::Mean);
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(1e-3, model, gradients);
    }
    model
}

fn flatten_means(rows: &[Trajectory]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| row.posterior.mean.map(|value| value as f32))
        .collect()
}

fn flatten_covariances(rows: &[Trajectory]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| {
            let p = row.posterior.covariance;
            [
                p[0][0] as f32,
                p[0][1] as f32,
                p[1][0] as f32,
                p[1][1] as f32,
            ]
        })
        .collect()
}

fn finite_moments(mean: Vec<f32>, variance: Vec<f32>) -> Vec<MomentsEstimate> {
    assert_eq!(mean.len(), variance.len());
    mean.into_iter()
        .zip(variance)
        .map(|(mean, variance)| {
            assert!(mean.is_finite(), "mean must be finite");
            let variance = f64::from(variance);
            assert!(
                variance.is_finite() && variance >= 0.0,
                "variance must be finite and nonnegative"
            );
            MomentsEstimate {
                mean: f64::from(mean),
                variance,
            }
        })
        .collect()
}

fn propagate(
    model: &Mlp,
    rows: &[Trajectory],
    full: bool,
    device: &Device,
) -> Vec<MomentsEstimate> {
    let count = rows.len();
    let mean = Tensor::<2>::from_data(TensorData::new(flatten_means(rows), [count, DIM]), device);
    let covariance = Tensor::<3>::from_data(
        TensorData::new(flatten_covariances(rows), [count, DIM, DIM]),
        device,
    );
    let w1 = model.first.weight.val().inner();
    let b1 = model.first.bias.as_ref().map(|value| value.val().inner());
    let w2 = model.last.weight.val().inner();
    let b2 = model.last.bias.as_ref().map(|value| value.val().inner());
    if full {
        let output = propagate_linear_full(
            &propagate_relu_full(&propagate_linear_full(
                &MomentsFull::new(mean, covariance),
                w1,
                b1,
            )),
            w2,
            b2,
        );
        let variance = output.variance().to_data().try_to_vec::<f32>().unwrap();
        let mean = output.mean.to_data().try_to_vec::<f32>().unwrap();
        finite_moments(mean, variance)
    } else {
        // Retain the supplied input covariance in the first affine marginal,
        // then drop hidden-feature covariance at the diagonal representation.
        let first = propagate_linear_full(&MomentsFull::new(mean, covariance), w1, b1);
        let first_variance = first.variance();
        let output = propagate_linear(
            &propagate_relu(&Moments::new(first.mean, first_variance)),
            w2,
            b2,
        );
        finite_moments(
            output.mean.to_data().try_to_vec::<f32>().unwrap(),
            output.var.to_data().try_to_vec::<f32>().unwrap(),
        )
    }
}

fn forward_inner(model: &Mlp, inputs: Vec<f32>, rows: usize, device: &Device) -> Vec<f64> {
    let x = Tensor::<2>::from_data(TensorData::new(inputs, [rows, DIM]), device);
    let w1 = model.first.weight.val().inner();
    let b1 = model
        .first
        .bias
        .as_ref()
        .map(|value| value.val().inner())
        .expect("configured linear layer has a bias");
    let w2 = model.last.weight.val().inner();
    let b2 = model
        .last
        .bias
        .as_ref()
        .map(|value| value.val().inner())
        .expect("configured linear layer has a bias");
    (activation::relu(x.matmul(w1) + b1.reshape([1, HIDDEN])).matmul(w2) + b2.reshape([1, 1]))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

fn matched_mc(
    model: &Mlp,
    rows: &[Trajectory],
    draws: usize,
    seed: u64,
    device: &Device,
) -> Vec<MomentsEstimate> {
    assert!(draws > 1, "matched Monte Carlo needs two draws");
    let mut rng = Rng::new(seed);
    let mut inputs = Vec::with_capacity(rows.len() * draws * DIM);
    for row in rows {
        for _ in 0..draws {
            inputs.extend(
                sample_gaussian(row.posterior.mean, row.posterior.covariance, &mut rng)
                    .map(|x| x as f32),
            );
        }
    }
    forward_inner(model, inputs, rows.len() * draws, device)
        .chunks_exact(draws)
        .map(|draws| {
            let mean = draws.iter().sum::<f64>() / draws.len() as f64;
            let variance = draws
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (draws.len() - 1) as f64;
            MomentsEstimate { mean, variance }
        })
        .collect()
}

fn conformal_quantile(mut scores: Vec<f64>) -> f64 {
    assert!(scores
        .iter()
        .all(|score| score.is_finite() && *score >= 0.0));
    let coverage = Coverage::from_miscoverage(ALPHA).expect("ALPHA is a valid miscoverage");
    match calibrate_in_place(&mut scores, coverage).expect("finite scores calibrate") {
        Threshold::Finite(value) => value,
        Threshold::Unbounded => f64::INFINITY,
    }
}

fn interval_metrics(
    rows: &[Trajectory],
    estimates: &[MomentsEstimate],
    multiplier: f64,
) -> IntervalMetrics {
    assert_eq!(rows.len(), estimates.len());
    let mut hits = 0usize;
    let mut width = 0.0;
    for (row, estimate) in rows.iter().zip(estimates) {
        let half_width = multiplier * estimate.variance.max(SCALE_FLOOR).sqrt();
        hits += usize::from((row.target - estimate.mean).abs() <= half_width);
        width += 2.0 * half_width;
    }
    IntervalMetrics {
        coverage: hits as f64 / rows.len() as f64,
        width: width / rows.len() as f64,
    }
}

fn constant_metrics(
    rows: &[Trajectory],
    estimates: &[MomentsEstimate],
    half_width: f64,
) -> IntervalMetrics {
    assert_eq!(rows.len(), estimates.len());
    let hits = rows
        .iter()
        .zip(estimates)
        .filter(|(row, estimate)| (row.target - estimate.mean).abs() <= half_width)
        .count();
    IntervalMetrics {
        coverage: hits as f64 / rows.len() as f64,
        width: 2.0 * half_width,
    }
}

fn mean_abs_difference(
    left: &[MomentsEstimate],
    right: &[MomentsEstimate],
    select: impl Fn(MomentsEstimate) -> f64,
) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (select(*left) - select(*right)).abs())
        .sum::<f64>()
        / left.len() as f64
}

fn trial(repeat: usize, quick: bool, device: &Device) -> Trial {
    let (train_rows, calibration_rows, test_rows, epochs, draws) = if quick {
        (
            QUICK_TRAIN,
            QUICK_CAL,
            QUICK_TEST,
            QUICK_EPOCHS,
            QUICK_MC_DRAWS,
        )
    } else {
        (FULL_TRAIN, FULL_CAL, FULL_TEST, FULL_EPOCHS, FULL_MC_DRAWS)
    };
    let base = 0xA11C_9000 + repeat as u64 * 8;
    let model = train_model(train_rows, epochs, base + 1, device);
    let calibration = trajectories(calibration_rows, base + 2);
    let test = trajectories(test_rows, base + 3);
    let inner = Device::flex();
    let full_calibration = propagate(&model, &calibration, true, &inner);
    let full_test = propagate(&model, &test, true, &inner);
    let diagonal_test = propagate(&model, &test, false, &inner);
    let sampled_test = matched_mc(&model, &test, draws, base + 4, &inner);
    let sampled_test_repeat = matched_mc(&model, &test, draws, base + 5, &inner);
    let oracle_calibration: Vec<_> = calibration
        .iter()
        .map(|row| oracle_moments(row.posterior))
        .collect();
    let oracle_test: Vec<_> = test
        .iter()
        .map(|row| oracle_moments(row.posterior))
        .collect();

    let constant = conformal_quantile(
        calibration
            .iter()
            .zip(&full_calibration)
            .map(|(row, estimate)| (row.target - estimate.mean).abs())
            .collect(),
    );
    let propagated = conformal_quantile(
        calibration
            .iter()
            .zip(&full_calibration)
            .map(|(row, estimate)| {
                (row.target - estimate.mean).abs()
                    / (estimate.variance + LABEL_VARIANCE).max(SCALE_FLOOR).sqrt()
            })
            .collect(),
    );
    let oracle = conformal_quantile(
        calibration
            .iter()
            .zip(&oracle_calibration)
            .map(|(row, estimate)| {
                (row.target - estimate.mean).abs() / estimate.variance.max(SCALE_FLOOR).sqrt()
            })
            .collect(),
    );
    let propagated_interval: Vec<_> = full_test
        .iter()
        .map(|estimate| MomentsEstimate {
            mean: estimate.mean,
            variance: estimate.variance + LABEL_VARIANCE,
        })
        .collect();
    let sampled_target_test: Vec<_> = sampled_test
        .iter()
        .map(|estimate| MomentsEstimate {
            mean: estimate.mean,
            variance: estimate.variance + LABEL_VARIANCE,
        })
        .collect();

    Trial {
        full_mean_mc_mae: mean_abs_difference(&full_test, &sampled_test, |m| m.mean),
        full_variance_mc_mae: mean_abs_difference(&full_test, &sampled_test, |m| m.variance),
        mc_repeat_mean_mae: mean_abs_difference(&sampled_test, &sampled_test_repeat, |m| m.mean),
        mc_repeat_variance_mae: mean_abs_difference(&sampled_test, &sampled_test_repeat, |m| {
            m.variance
        }),
        diagonal_mean_mc_mae: mean_abs_difference(&diagonal_test, &sampled_test, |m| m.mean),
        diagonal_variance_mc_mae: mean_abs_difference(&diagonal_test, &sampled_test, |m| {
            m.variance
        }),
        model_oracle_mean_mae: mean_abs_difference(&sampled_target_test, &oracle_test, |m| m.mean),
        model_oracle_variance_mae: mean_abs_difference(&sampled_target_test, &oracle_test, |m| {
            m.variance
        }),
        raw: interval_metrics(&test, &propagated_interval, 1.645),
        constant: constant_metrics(&test, &full_test, constant),
        propagated: interval_metrics(&test, &propagated_interval, propagated),
        oracle: interval_metrics(&test, &oracle_test, oracle),
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<_> = values.collect();
    values.iter().sum::<f64>() / values.len() as f64
}

fn se(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<_> = values.collect();
    if values.len() < 2 {
        return f64::NAN;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (variance / values.len() as f64).sqrt()
}

fn print_metric(name: &str, trials: &[Trial], metric: impl Fn(Trial) -> f64, quick: bool) {
    let values: Vec<_> = trials.iter().copied().map(metric).collect();
    if quick {
        println!(
            "  {name}: {:.5} (three-repeat descriptive mean)",
            mean(values.into_iter())
        );
    } else {
        println!(
            "  {name}: {:.5} +/- {:.5} SE across independent fits/splits",
            mean(values.iter().copied()),
            se(values.into_iter())
        );
    }
}

fn main() {
    let quick = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => true,
        [flag] if flag == "--quick" => true,
        [flag] if flag == "--study" => false,
        _ => panic!("usage: kalman_sensor_intervals [--quick|--study]"),
    };
    let repeats = if quick { QUICK_REPEATS } else { FULL_REPEATS };
    let device = Device::flex().autodiff();
    println!("two-channel linear-Gaussian state estimator; shared-plus-independent reading covariance R={R:?}");
    println!("{repeats} independent trajectory splits; one terminal target per trajectory; observation steps={OBSERVATIONS}, horizon={HORIZON}");
    println!(
        "clean-state surrogate: hidden={HIDDEN}; matched surrogate MC={}",
        if quick { QUICK_MC_DRAWS } else { FULL_MC_DRAWS }
    );
    let trials: Vec<_> = (0..repeats)
        .map(|repeat| trial(repeat, quick, &device))
        .collect();
    println!("\npropagation error against matched surrogate Monte Carlo:");
    print_metric("full mean MAE", &trials, |t| t.full_mean_mc_mae, quick);
    print_metric(
        "full variance MAE",
        &trials,
        |t| t.full_variance_mc_mae,
        quick,
    );
    print_metric(
        "diagonal mean MAE",
        &trials,
        |t| t.diagonal_mean_mc_mae,
        quick,
    );
    print_metric(
        "diagonal variance MAE",
        &trials,
        |t| t.diagonal_variance_mc_mae,
        quick,
    );
    println!("\nMonte Carlo sampling difference (two independent streams per fitted model):");
    print_metric("mean MAE", &trials, |t| t.mc_repeat_mean_mae, quick);
    print_metric("variance MAE", &trials, |t| t.mc_repeat_variance_mae, quick);
    println!("\nsampled surrogate versus exact simulator oracle (includes Monte Carlo error):");
    print_metric(
        "sampled-model mean MAE",
        &trials,
        |t| t.model_oracle_mean_mae,
        quick,
    );
    print_metric(
        "sampled-model variance MAE",
        &trials,
        |t| t.model_oracle_variance_mae,
        quick,
    );
    println!("\ntarget intervals (coverage and width are separate; raw Gaussian bands are uncalibrated):");
    for (name, metric) in [
        (
            "raw propagated Gaussian",
            (|t: Trial| t.raw) as fn(Trial) -> IntervalMetrics,
        ),
        ("constant split conformal", |t: Trial| t.constant),
        ("propagated-scale conformal", |t: Trial| t.propagated),
        (
            "oracle center-and-scale conformal diagnostic",
            |t: Trial| t.oracle,
        ),
    ] {
        let coverage = mean(trials.iter().copied().map(|trial| metric(trial).coverage));
        let width = mean(trials.iter().copied().map(|trial| metric(trial).width));
        if quick {
            println!("  {name}: coverage={coverage:.3}, mean width={width:.3}");
        } else {
            let coverage_se = se(trials.iter().copied().map(|trial| metric(trial).coverage));
            let width_se = se(trials.iter().copied().map(|trial| metric(trial).width));
            println!("  {name}: coverage={coverage:.3} +/- {coverage_se:.3} SE, mean width={width:.3} +/- {width_se:.3} SE (fits/splits)");
        }
    }
    if !quick {
        println!("\npaired propagated-scale minus constant split-conformal metrics:");
        print_metric(
            "coverage difference",
            &trials,
            |t| t.propagated.coverage - t.constant.coverage,
            false,
        );
        print_metric(
            "mean-width difference",
            &trials,
            |t| t.propagated.width - t.constant.width,
            false,
        );
    }
    if quick {
        println!(
            "quick mode is descriptive. Run --study for 20 independent fitted-model/split repeats."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "left={left}, right={right}"
        );
    }

    #[test]
    fn diagonal_kalman_update_matches_scalar_conditioning() {
        let prediction = Posterior {
            mean: [1.2, -0.4],
            covariance: [[0.8, 0.0], [0.0, 0.3]],
        };
        let reading = [0.7, -0.1];
        let diagonal_noise = [[0.08, 0.0], [0.0, 0.11]];
        let updated = kalman_update_with_noise(prediction, reading, diagonal_noise);
        for i in 0..DIM {
            let gain =
                prediction.covariance[i][i] / (prediction.covariance[i][i] + diagonal_noise[i][i]);
            close(
                updated.mean[i],
                prediction.mean[i] + gain * (reading[i] - prediction.mean[i]),
                1e-12,
            );
            close(
                updated.covariance[i][i],
                (1.0 - gain) * prediction.covariance[i][i],
                1e-12,
            );
        }
    }

    #[test]
    fn correlated_kalman_update_matches_information_form() {
        let prediction = Posterior {
            mean: [0.3, -0.7],
            covariance: [[0.6, 0.15], [0.15, 0.4]],
        };
        let reading = [-0.2, 0.9];
        let updated = kalman_update(prediction, reading);
        let posterior_covariance = inverse(add(inverse(prediction.covariance), inverse(R)));
        let precision_weighted_mean = mat_vec(posterior_covariance, {
            let prior_term = mat_vec(inverse(prediction.covariance), prediction.mean);
            let reading_term = mat_vec(inverse(R), reading);
            [
                prior_term[0] + reading_term[0],
                prior_term[1] + reading_term[1],
            ]
        });
        for i in 0..DIM {
            close(updated.mean[i], precision_weighted_mean[i], 1e-12);
            for j in 0..DIM {
                close(updated.covariance[i][j], posterior_covariance[i][j], 1e-12);
            }
        }
    }

    fn gauss_hermite_quadratic_moments(posterior: Posterior) -> MomentsEstimate {
        let nodes = [-3.0_f64.sqrt(), 0.0, 3.0_f64.sqrt()];
        let weights = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
        let l = cholesky(posterior.covariance);
        let mut mean = 0.0;
        let mut second = 0.0;
        for (i, &zi) in nodes.iter().enumerate() {
            for (j, &zj) in nodes.iter().enumerate() {
                let state = [
                    posterior.mean[0] + l[0][0] * zi,
                    posterior.mean[1] + l[1][0] * zi + l[1][1] * zj,
                ];
                let value = terminal(state);
                mean += weights[i] * weights[j] * value;
                second += weights[i] * weights[j] * value * value;
            }
        }
        MomentsEstimate {
            mean,
            variance: second - mean * mean + LABEL_VARIANCE,
        }
    }

    proptest::proptest! {
        #[test]
        fn quadratic_oracle_matches_gauss_hermite_for_generated_gaussians(
            a in 0.1f64..2.0, b in -2.0f64..2.0, c in 0.1f64..2.0,
            x in -3.0f64..3.0, y in -3.0f64..3.0,
        ) {
            let lower = [[a, 0.0], [b, c]];
            let posterior = Posterior {
                mean: [x, y],
                covariance: mat_mul(lower, transpose(lower)),
            };
            let oracle = oracle_moments(posterior);
            let quadrature = gauss_hermite_quadratic_moments(posterior);
            proptest::prop_assert!((oracle.mean - quadrature.mean).abs() <= 1e-12 * oracle.mean.abs().max(1.0));
            proptest::prop_assert!((oracle.variance - quadrature.variance).abs() <= 1e-12 * oracle.variance.max(1.0));
        }
    }

    #[test]
    fn actual_propagation_has_zero_variance_for_zero_covariance() {
        let device = Device::flex();
        let mean = Tensor::<2>::from_data(TensorData::new(vec![0.4_f32, -0.2], [1, DIM]), &device);
        let covariance = Tensor::<3>::zeros([1, DIM, DIM], &device);
        let w1 = Tensor::<2>::from_data(
            TensorData::new(vec![1.0_f32, -0.5, -0.3, 0.8], [DIM, DIM]),
            &device,
        );
        let b1 = Tensor::<1>::from_data(TensorData::new(vec![0.1_f32, 0.2], [DIM]), &device);
        let w2 = Tensor::<2>::from_data(TensorData::new(vec![0.7_f32, -0.4], [DIM, 1]), &device);
        let b2 = Tensor::<1>::from_data(TensorData::new(vec![0.05_f32], [1]), &device);
        let output = propagate_linear_full(
            &propagate_relu_full(&propagate_linear_full(
                &MomentsFull::new(mean.clone(), covariance),
                w1.clone(),
                Some(b1.clone()),
            )),
            w2.clone(),
            Some(b2.clone()),
        );
        let variance = output.variance().to_data().try_to_vec::<f32>().unwrap();
        let propagated = output.mean.to_data().try_to_vec::<f32>().unwrap();
        let point = (activation::relu(mean.matmul(w1) + b1.reshape([1, DIM])).matmul(w2)
            + b2.reshape([1, 1]))
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
        assert_eq!(propagated, point);
        assert_eq!(variance, vec![0.0]);
    }
}
