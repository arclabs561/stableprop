//! Grouped conformal intervals on two public tabular regression data sets.
//!
//! This example keeps all recordings from one reconstructed condition (Airfoil)
//! or one person (Parkinsons) in exactly one split.  It fits an MLP only on fit
//! groups, calibrates one maximum residual score per calibration group, and
//! evaluates complete held-out groups.  The point predictor is shared by both
//! interval arms: a constant-width group conformal interval and one scaled by
//! stableprop's diagonal sensitivity to a supplied input-noise stress.
//!
//! The supplied 0.05 standardized-feature noise is an experimental stress, not
//! a measurement-error estimate.  Propagated variance therefore supplies a
//! score scale; it is not target uncertainty.  With exchangeable complete
//! groups conditional on fitting data, under the same recording scheme and
//! group-size distribution, split conformal targets coverage of every observed
//! row in a new complete group.  This fixed split reports only empirical panels
//! and does not attach a row-independence confidence interval.
//!
//! Obtain the files from UCI using the commands in `examples/README.md`, then run:
//!
//! `cargo run --release --features burn --example grouped_intervals -- airfoil PATH`
//! `cargo run --release --features burn --example grouped_intervals -- parkinsons PATH --quick`

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Device, Tensor, TensorData};
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{propagate_linear, propagate_relu, Moments};
use statskit::conformal::{calibrate_in_place, Coverage, Threshold};

type Ad = Autodiff<NdArray<f32>>;
type Nd = NdArray<f32>;

const HIDDEN: usize = 16;
const INPUT_STRESS_STD: f32 = 0.05;
const VARIANCE_FLOOR: f64 = 1e-4;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    first: Linear<B>,
    last: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn init(inputs: usize, device: &B::Device) -> Self {
        Self {
            first: LinearConfig::new(inputs, HIDDEN).init(device),
            last: LinearConfig::new(HIDDEN, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.last.forward(activation::relu(self.first.forward(x)))
    }
}

#[derive(Clone)]
struct Rows {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    group: Vec<usize>,
    inputs: usize,
}

impl Rows {
    fn len(&self) -> usize {
        self.y.len()
    }
}

#[derive(Clone, Copy)]
enum DataSet {
    Airfoil,
    Parkinsons,
}

impl DataSet {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "airfoil" => Ok(Self::Airfoil),
            "parkinsons" => Ok(Self::Parkinsons),
            _ => Err("dataset must be `airfoil` or `parkinsons`".into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Airfoil => "Airfoil Self-Noise",
            Self::Parkinsons => "Parkinsons Telemonitoring",
        }
    }
}

/// Keep group splits and Monte Carlo draws independent of the backend RNG.
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

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = (self.uniform() * (i + 1) as f64) as usize;
            values.swap(i, j);
        }
    }
}

fn parse_finite(text: &str, context: &str) -> Result<f64, String> {
    let value = text
        .parse::<f64>()
        .map_err(|_| format!("{context}: expected a finite number, found `{text}`"))?;
    if !value.is_finite() || !((value as f32).is_finite()) {
        return Err(format!(
            "{context}: value `{text}` is non-finite or outside f32 range"
        ));
    }
    Ok(value)
}

fn canonical_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn group_index<K: std::hash::Hash + Eq>(key: K, groups: &mut HashMap<K, usize>) -> usize {
    let next = groups.len();
    *groups.entry(key).or_insert(next)
}

fn load_airfoil(text: &str) -> Result<Rows, String> {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut group = Vec::new();
    let mut groups = HashMap::<[u64; 4], usize>::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!(
                "Airfoil line {}: empty records are not allowed",
                line_number + 1
            ));
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(format!(
                "Airfoil line {}: expected 6 whitespace columns",
                line_number + 1
            ));
        }
        let values: Vec<f64> = fields
            .iter()
            .enumerate()
            .map(|(column, value)| {
                parse_finite(
                    value,
                    &format!("Airfoil line {}, column {}", line_number + 1, column + 1),
                )
            })
            .collect::<Result<_, _>>()?;
        let features = values[..5].to_vec();
        // Frequency is the first feature.  Equal raw values for the remaining
        // measured-condition fields reconstruct a panel; this is not a verified
        // physical configuration identifier.
        let key = [
            canonical_bits(values[1]),
            canonical_bits(values[2]),
            canonical_bits(values[3]),
            canonical_bits(values[4]),
        ];
        x.push(features);
        y.push(values[5]);
        group.push(group_index(key, &mut groups));
    }
    build_rows(x, y, group, "Airfoil")
}

const PARKINSONS_HEADER: &str = "subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP,Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA,NHR,HNR,RPDE,DFA,PPE";

fn load_parkinsons(text: &str) -> Result<Rows, String> {
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or("Parkinsons file is empty")?
        .trim_end_matches('\r');
    if header != PARKINSONS_HEADER {
        return Err("Parkinsons header does not match the UCI 22-column schema".into());
    }
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut group = Vec::new();
    let mut groups = HashMap::<u64, usize>::new();
    for (offset, line) in lines.enumerate() {
        if line.trim().is_empty() {
            return Err(format!(
                "Parkinsons line {}: empty records are not allowed",
                offset + 2
            ));
        }
        let fields: Vec<_> = line.trim_end_matches('\r').split(',').collect();
        if fields.len() != 22 {
            return Err(format!(
                "Parkinsons line {}: expected 22 CSV columns",
                offset + 2
            ));
        }
        let subject = fields[0].parse::<u64>().map_err(|_| {
            format!(
                "Parkinsons line {}: subject# must be a positive integer",
                offset + 2
            )
        })?;
        if subject == 0 {
            return Err(format!(
                "Parkinsons line {}: subject# must be positive",
                offset + 2
            ));
        }
        let values: Vec<f64> = fields[1..]
            .iter()
            .enumerate()
            .map(|(column, value)| {
                parse_finite(
                    value,
                    &format!(
                        "Parkinsons line {}, numeric column {}",
                        offset + 2,
                        column + 2
                    ),
                )
            })
            .collect::<Result<_, _>>()?;
        let target = values[3];
        let features = values[5..].to_vec();
        // The 16 voice columns are the only Parkinsons features.  Metadata and
        // both UPDRS targets never enter the model input.
        x.push(features);
        y.push(target);
        group.push(group_index(subject, &mut groups));
    }
    build_rows(x, y, group, "Parkinsons")
}

fn build_rows(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    group: Vec<usize>,
    name: &str,
) -> Result<Rows, String> {
    if x.is_empty()
        || x.len() != y.len()
        || y.len() != group.len()
        || x.iter().any(|row| row.len() != x[0].len())
    {
        return Err(format!("{name}: no usable complete rows"));
    }
    Ok(Rows {
        inputs: x[0].len(),
        x,
        y,
        group,
    })
}

fn load(dataset: DataSet, path: &str) -> Result<Rows, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("cannot read `{path}`: {error}"))?;
    match dataset {
        DataSet::Airfoil => load_airfoil(&text),
        DataSet::Parkinsons => load_parkinsons(&text),
    }
}

#[derive(Clone)]
struct Split {
    fit: Vec<usize>,
    calibration: Vec<usize>,
    test: Vec<usize>,
}

fn split_groups(rows: &Rows, seed: u64) -> Split {
    let count = rows.group.iter().copied().max().expect("nonempty rows") + 1;
    let mut groups: Vec<usize> = (0..count).collect();
    Rng::new(seed).shuffle(&mut groups);
    let fit_groups = count / 2;
    let calibration_groups = count / 4;
    assert!(
        fit_groups > 0 && calibration_groups > 0 && count - fit_groups - calibration_groups > 0,
        "need at least four groups"
    );
    let mut assignment = vec![2_u8; count];
    for &id in &groups[..fit_groups] {
        assignment[id] = 0;
    }
    for &id in &groups[fit_groups..fit_groups + calibration_groups] {
        assignment[id] = 1;
    }
    let mut split = Split {
        fit: Vec::new(),
        calibration: Vec::new(),
        test: Vec::new(),
    };
    for (row, &id) in rows.group.iter().enumerate() {
        match assignment[id] {
            0 => split.fit.push(row),
            1 => split.calibration.push(row),
            _ => split.test.push(row),
        }
    }
    split
}

#[derive(Clone)]
struct Standardizer {
    x_mean: Vec<f64>,
    x_scale: Vec<f64>,
    y_mean: f64,
    y_scale: f64,
}

fn fit_standardizer(rows: &Rows, fit: &[usize]) -> Standardizer {
    let mut x_mean = vec![0.0; rows.inputs];
    let mut y_mean = 0.0;
    for &row in fit {
        for (mean, value) in x_mean.iter_mut().zip(&rows.x[row]) {
            *mean += value;
        }
        y_mean += rows.y[row];
    }
    for feature in &mut x_mean {
        *feature /= fit.len() as f64;
    }
    y_mean /= fit.len() as f64;
    let mut x_scale = vec![0.0; rows.inputs];
    let mut y_scale = 0.0;
    for &row in fit {
        for feature in 0..rows.inputs {
            x_scale[feature] += (rows.x[row][feature] - x_mean[feature]).powi(2);
        }
        y_scale += (rows.y[row] - y_mean).powi(2);
    }
    for scale in &mut x_scale {
        *scale = (*scale / fit.len() as f64).sqrt();
        if *scale == 0.0 {
            *scale = 1.0;
        }
    }
    y_scale = (y_scale / fit.len() as f64).sqrt();
    if y_scale == 0.0 {
        y_scale = 1.0;
    }
    Standardizer {
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    }
}

impl Standardizer {
    fn x(&self, row: &[f64]) -> Vec<f32> {
        row.iter()
            .enumerate()
            .map(|(feature, value)| {
                ((*value - self.x_mean[feature]) / self.x_scale[feature]) as f32
            })
            .collect()
    }
    fn y(&self, value: f64) -> f32 {
        ((value - self.y_mean) / self.y_scale) as f32
    }
    fn raw_y(&self, value: f32) -> f64 {
        self.y_mean + self.y_scale * f64::from(value)
    }
}

fn train(
    rows: &Rows,
    fit: &[usize],
    standardizer: &Standardizer,
    epochs: usize,
    device: &Device<Ad>,
) -> Mlp<Ad> {
    <Ad as Backend>::seed(device, 0xA11C_E001);
    let x: Vec<f32> = fit
        .iter()
        .flat_map(|&row| standardizer.x(&rows.x[row]))
        .collect();
    let y: Vec<f32> = fit.iter().map(|&row| standardizer.y(rows.y[row])).collect();
    let x = Tensor::<Ad, 2>::from_data(TensorData::new(x, [fit.len(), rows.inputs]), device);
    let y = Tensor::<Ad, 2>::from_data(TensorData::new(y, [fit.len(), 1]), device);
    let mut model = Mlp::init(rows.inputs, device);
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..epochs {
        let loss = MseLoss::new().forward(model.forward(x.clone()), y.clone(), Reduction::Mean);
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(1e-3, model, gradients);
    }
    model
}

fn propagated(
    model: &Mlp<Ad>,
    x: &[f32],
    inputs: usize,
    device: &Device<Nd>,
) -> (Vec<f32>, Vec<f64>) {
    let n = x.len() / inputs;
    let input = Tensor::<Nd, 2>::from_data(TensorData::new(x.to_vec(), [n, inputs]), device);
    let variance =
        Tensor::<Nd, 2>::ones([n, inputs], device) * (INPUT_STRESS_STD * INPUT_STRESS_STD);
    let first = propagate_relu(&propagate_linear(
        &Moments::new(input, variance),
        model.first.weight.val().inner(),
        model.first.bias.as_ref().map(|bias| bias.val().inner()),
    ));
    let output = propagate_linear(
        &first,
        model.last.weight.val().inner(),
        model.last.bias.as_ref().map(|bias| bias.val().inner()),
    );
    let mean = output.mean.to_data().to_vec::<f32>().unwrap();
    assert!(
        mean.iter().all(|value| value.is_finite()),
        "propagated means must be finite"
    );
    let variance = output
        .var
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    if variance
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        panic!("propagated variance must be finite and nonnegative");
    }
    (mean, variance)
}

/// The interval center is the deterministic network evaluated at the observed
/// feature center.  It intentionally does not use the propagated mean under
/// the experimental input-noise law, so both calibration arms share one point
/// predictor.
fn point_centers(model: &Mlp<Ad>, x: &[f32], inputs: usize, device: &Device<Ad>) -> Vec<f32> {
    let n = x.len() / inputs;
    let centers = model
        .forward(Tensor::<Ad, 2>::from_data(
            TensorData::new(x.to_vec(), [n, inputs]),
            device,
        ))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        centers.iter().all(|value| value.is_finite()),
        "point predictions must be finite"
    );
    centers
}

fn quantile(scores: &mut [f64]) -> f64 {
    if scores
        .iter()
        .any(|score| !score.is_finite() || *score < 0.0)
    {
        panic!("conformal scores must be finite and nonnegative");
    }
    // Request exactly 90% group coverage.
    let coverage = Coverage::from_ratio(9, 10).expect("9/10 must be valid coverage");
    if scores.is_empty() {
        return f64::INFINITY;
    }
    match calibrate_in_place(scores, coverage)
        .expect("validated finite nonnegative scores must calibrate")
    {
        Threshold::Finite(value) => value,
        Threshold::Unbounded => f64::INFINITY,
    }
}

fn group_scores(
    indices: &[usize],
    rows: &Rows,
    centers: &[f64],
    scales: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut constant = HashMap::<usize, f64>::new();
    let mut scaled = HashMap::<usize, f64>::new();
    for (offset, &row) in indices.iter().enumerate() {
        let residual = (rows.y[row] - centers[offset]).abs();
        constant
            .entry(rows.group[row])
            .and_modify(|value| *value = value.max(residual))
            .or_insert(residual);
        let score = residual / scales[offset];
        scaled
            .entry(rows.group[row])
            .and_modify(|value| *value = value.max(score))
            .or_insert(score);
    }
    (
        constant.into_values().collect(),
        scaled.into_values().collect(),
    )
}

#[derive(Clone, Copy)]
struct Metrics {
    complete: f64,
    macro_coverage: f64,
    macro_width: f64,
    rmse: f64,
}

fn metrics(indices: &[usize], rows: &Rows, centers: &[f64], half_widths: &[f64]) -> Metrics {
    let mut panels = HashMap::<usize, (usize, usize, f64, f64)>::new();
    for (offset, &row) in indices.iter().enumerate() {
        let covered = ((rows.y[row] - centers[offset]).abs() <= half_widths[offset]) as usize;
        let entry = panels.entry(rows.group[row]).or_insert((0, 0, 0.0, 0.0));
        entry.0 += 1;
        entry.1 += covered;
        entry.2 += 2.0 * half_widths[offset];
        entry.3 += (rows.y[row] - centers[offset]).powi(2);
    }
    let groups = panels.len() as f64;
    let complete = panels
        .values()
        .filter(|(n, covered, _, _)| n == covered)
        .count() as f64
        / groups;
    let macro_coverage = panels
        .values()
        .map(|(n, covered, _, _)| *covered as f64 / *n as f64)
        .sum::<f64>()
        / groups;
    let macro_width = panels
        .values()
        .map(|(n, _, width, _)| width / *n as f64)
        .sum::<f64>()
        / groups;
    let rmse = (panels
        .values()
        .map(|(n, _, _, squared)| squared / *n as f64)
        .sum::<f64>()
        / groups)
        .sqrt();
    Metrics {
        complete,
        macro_coverage,
        macro_width,
        rmse,
    }
}

fn mc_check(
    model: &Mlp<Ad>,
    x: &[f32],
    inputs: usize,
    analytic_mean: &[f32],
    analytic_variance: &[f64],
    device: &Device<Nd>,
    quick: bool,
) {
    let n = x.len() / inputs;
    let draws = if quick { 64 } else { 256 };
    let mut means = vec![0.0; n];
    let mut m2 = vec![0.0; n];
    let mut rng = Rng::new(0xAC00_0001);
    for draw in 1..=draws {
        let noisy: Vec<f32> = x
            .iter()
            .map(|value| (*value as f64 + INPUT_STRESS_STD as f64 * rng.normal()) as f32)
            .collect();
        let input = Tensor::<Nd, 2>::from_data(TensorData::new(noisy, [n, inputs]), device);
        let hidden = activation::relu(
            input.matmul(model.first.weight.val().inner())
                + model
                    .first
                    .bias
                    .as_ref()
                    .unwrap()
                    .val()
                    .inner()
                    .reshape([1, HIDDEN]),
        );
        let output = (hidden.matmul(model.last.weight.val().inner())
            + model
                .last
                .bias
                .as_ref()
                .unwrap()
                .val()
                .inner()
                .reshape([1, 1]))
        .to_data()
        .to_vec::<f32>()
        .unwrap();
        for (index, value) in output.into_iter().enumerate() {
            let delta = f64::from(value) - means[index];
            means[index] += delta / draw as f64;
            m2[index] += delta * (f64::from(value) - means[index]);
        }
    }
    let mean_mae = means
        .iter()
        .zip(analytic_mean)
        .map(|(mc, analytic)| (mc - f64::from(*analytic)).abs())
        .sum::<f64>()
        / n as f64;
    if m2.iter().any(|value| !value.is_finite() || *value < 0.0) {
        panic!("Monte Carlo variances must be finite and nonnegative");
    }
    let variance_mae = m2
        .iter()
        .zip(analytic_variance)
        .map(|(m2, analytic)| ((m2 / (draws - 1) as f64) - analytic).abs())
        .sum::<f64>()
        / n as f64;
    println!("diagonal moments versus MC ({draws} draws, first {n} test rows, standardized outputs): mean MAE {mean_mae:.5}, variance MAE {variance_mae:.5}");
}

fn run(dataset: DataSet, path: &str, quick: bool) -> Result<(), String> {
    let rows = load(dataset, path)?;
    let groups = rows.group.iter().copied().max().unwrap() + 1;
    if groups < 4 {
        return Err("need at least four distinct groups for fit/calibration/test splits".into());
    }
    let split = split_groups(&rows, 0x5A11_7001);
    let standardizer = fit_standardizer(&rows, &split.fit);
    if rows
        .x
        .iter()
        .flat_map(|row| standardizer.x(row))
        .any(|value| !value.is_finite())
        || split
            .fit
            .iter()
            .any(|&row| !standardizer.y(rows.y[row]).is_finite())
    {
        return Err(
            "fit-standardized inputs or training targets exceed the finite f32 range".into(),
        );
    }
    let epochs = if quick { 30 } else { 300 };
    println!("{}: {} rows, {} groups; fit/calibration/test groups = {}/{}/{}; {epochs} full-batch epochs", dataset.name(), rows.len(), groups, split.fit.iter().map(|&row| rows.group[row]).collect::<HashSet<_>>().len(), split.calibration.iter().map(|&row| rows.group[row]).collect::<HashSet<_>>().len(), split.test.iter().map(|&row| rows.group[row]).collect::<HashSet<_>>().len());
    let mut group_sizes = vec![0; groups];
    for &group in &rows.group {
        group_sizes[group] += 1;
    }
    println!(
        "rows per group: {}..{}; one fixed split, complete-group target at 90%",
        group_sizes.iter().min().unwrap(),
        group_sizes.iter().max().unwrap()
    );
    println!("feature stress std {INPUT_STRESS_STD}, output variance floor {VARIANCE_FLOOR} in fit-standardized units");
    println!("coverage assumes exchangeable complete groups under the same observation scheme; the stress is not measured sensor noise");
    let device = Device::<Ad>::default();
    let model = train(&rows, &split.fit, &standardizer, epochs, &device);
    let inner = Device::<Nd>::default();
    let normalized = |indices: &[usize]| {
        indices
            .iter()
            .flat_map(|&row| standardizer.x(&rows.x[row]))
            .collect::<Vec<_>>()
    };
    let calibration_x = normalized(&split.calibration);
    let test_x = normalized(&split.test);
    let (_, calibration_variance) = propagated(&model, &calibration_x, rows.inputs, &inner);
    let (test_mean, test_variance) = propagated(&model, &test_x, rows.inputs, &inner);
    let calibration_point = point_centers(&model, &calibration_x, rows.inputs, &device);
    let test_point = point_centers(&model, &test_x, rows.inputs, &device);
    let calibration_center: Vec<f64> = calibration_point
        .iter()
        .map(|&value| standardizer.raw_y(value))
        .collect();
    let test_center: Vec<f64> = test_point
        .iter()
        .map(|&value| standardizer.raw_y(value))
        .collect();
    let calibration_scale: Vec<f64> = calibration_variance
        .iter()
        .map(|&value| standardizer.y_scale * value.max(VARIANCE_FLOOR).sqrt())
        .collect();
    let test_scale: Vec<f64> = test_variance
        .iter()
        .map(|&value| standardizer.y_scale * value.max(VARIANCE_FLOOR).sqrt())
        .collect();
    let (mut constant_scores, mut scaled_scores) = group_scores(
        &split.calibration,
        &rows,
        &calibration_center,
        &calibration_scale,
    );
    let constant_q = quantile(&mut constant_scores);
    let scaled_q = quantile(&mut scaled_scores);
    if !constant_q.is_finite() || !scaled_q.is_finite() {
        println!("calibration has too few groups for the finite-sample 90% rank: intervals are unbounded");
    }
    let constant = metrics(
        &split.test,
        &rows,
        &test_center,
        &vec![constant_q; split.test.len()],
    );
    let scaled = metrics(
        &split.test,
        &rows,
        &test_center,
        &test_scale
            .iter()
            .map(|scale| scaled_q * scale)
            .collect::<Vec<_>>(),
    );
    println!(
        "group-macro point RMSE {:.3}; widths and RMSE are in original target units",
        constant.rmse
    );
    println!("constant score: complete-group coverage {:.3}, macro row coverage {:.3}, macro width {:.3}", constant.complete, constant.macro_coverage, constant.macro_width);
    println!("scaled sensitivity score: complete-group coverage {:.3}, macro row coverage {:.3}, macro width {:.3}", scaled.complete, scaled.macro_coverage, scaled.macro_width);
    let mc_rows = split.test.len().min(32);
    mc_check(
        &model,
        &test_x[..mc_rows * rows.inputs],
        rows.inputs,
        &test_mean[..mc_rows],
        &test_variance[..mc_rows],
        &inner,
        quick,
    );
    Ok(())
}

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let usage = "usage: grouped_intervals <airfoil|parkinsons> PATH [--quick]";
    let (dataset, path, quick) = match args.as_slice() {
        [dataset, path] => (DataSet::parse(dataset), path.as_str(), false),
        [dataset, path, flag] if flag == "--quick" => {
            (DataSet::parse(dataset), path.as_str(), true)
        }
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    match dataset.and_then(|dataset| run(dataset, path, quick)) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn group_quantile_covers_at_least_ninety_percent_of_exchangeable_ranks(
            scores in prop::collection::vec(0_u16..1000, 2..50)
        ) {
            // Exhaust every choice of the held-out group, conditional on this
            // unordered multiset. This checks the rank argument without MC.
            let covered = (0..scores.len()).filter(|&held_out| {
                let mut calibration: Vec<f64> = scores.iter().enumerate()
                    .filter(|(index, _)| *index != held_out)
                    .map(|(_, &score)| f64::from(score)).collect();
                f64::from(scores[held_out]) <= quantile(&mut calibration)
            }).count();
            prop_assert!(covered * 10 >= scores.len() * 9);
        }
    }

    #[test]
    fn airfoil_rejects_bad_fields_and_nonfinite_values() {
        assert!(load_airfoil("1 2 3 4 5\n").is_err());
        assert!(load_airfoil("1 2 NaN 4 5 6\n").is_err());
    }

    #[test]
    fn parkinsons_uses_voice_columns_not_motor_target() {
        let line = "1,1,0,0,99,100,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16";
        let rows = load_parkinsons(&format!("{PARKINSONS_HEADER}\n{line}\n")).unwrap();
        assert_eq!(rows.y, vec![99.0]);
        assert_eq!(&rows.x[0][..5], [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(
            &rows.x[0][5..],
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        );
    }

    #[test]
    fn group_split_and_fit_statistics_do_not_use_held_out_rows() {
        let rows = Rows {
            x: (1..=8).map(|value| vec![f64::from(value); 2]).collect(),
            y: (1..=8).map(f64::from).collect(),
            group: vec![0, 0, 1, 1, 2, 2, 3, 3],
            inputs: 2,
        };
        let split = split_groups(&rows, 7);
        let before = fit_standardizer(&rows, &split.fit);
        let mut tainted = rows.clone();
        for &row in split.test.iter().chain(&split.calibration) {
            tainted.x[row] = vec![9e99; 2];
            tainted.y[row] = 9e99;
        }
        let after = fit_standardizer(&tainted, &split.fit);
        assert_eq!(before.x_mean, after.x_mean);
        assert_eq!(before.y_mean, after.y_mean);
        assert_eq!(before.x_scale, after.x_scale);
        assert_eq!(before.y_scale, after.y_scale);
        for group in 0..4 {
            let in_fit = split.fit.iter().any(|&row| rows.group[row] == group);
            let in_cal = split
                .calibration
                .iter()
                .any(|&row| rows.group[row] == group);
            let in_test = split.test.iter().any(|&row| rows.group[row] == group);
            assert_eq!(in_fit as u8 + in_cal as u8 + in_test as u8, 1);
        }
    }

    #[test]
    fn group_quantile_has_unbounded_eight_group_boundary_and_handles_ties() {
        assert!(quantile(&mut []).is_infinite());
        assert!(quantile(&mut vec![1.0; 8]).is_infinite());
        assert_eq!(quantile(&mut vec![1.0; 9]), 1.0);
        let rows = Rows {
            x: vec![vec![0.0; 1]; 3],
            y: vec![2.0, 2.0, 1.0],
            group: vec![0, 0, 1],
            inputs: 1,
        };
        let (mut once, _) = group_scores(&[0, 2], &rows, &[0.0, 0.0], &[1.0, 1.0]);
        let (mut duplicated, _) =
            group_scores(&[0, 1, 2], &rows, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        once.sort_by(f64::total_cmp);
        duplicated.sort_by(f64::total_cmp);
        assert_eq!(once, duplicated);
    }

    #[test]
    #[should_panic(expected = "conformal scores must be finite and nonnegative")]
    fn group_quantile_adapter_rejects_signed_residual_scores() {
        let _ = quantile(&mut [0.0, -0.01]);
    }
}
