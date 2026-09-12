//! Compare input-noise disagreement with ordinary active-learning scores.
//!
//! This is a small controlled two-class pool experiment, not evidence that
//! input sensitivity estimates the value of a label.  Every policy sees the
//! same fixed pool, initial labels, initial weights, training recipe, and
//! held-out test set for each seed.  Labels are read only after an index has
//! been selected.  Both disagreement scores center logits with
//! `P = I - 11^T / K`: the analytic score is `2 * trace(P Cov(logits | x) P)`
//! and the Monte Carlo score is the centered-view equivalent.  This removes
//! common-logit shifts that do not change a classifier decision.  They describe
//! the same fixed-model perturbation quantity, approximately in the analytic
//! path.
//! Input-space farthest-first is a separate geometric coverage control: each
//! new pick accounts for earlier picks in the same batch. Its pool covering
//! radius describes raw feature-space coverage, not expected label value.
//!
//! Each budget is evaluated after retraining from the seed's original model
//! weights.  There is no variance penalty or consistency loss.  The printed
//! toy learning curves and score agreement are descriptive only.
//!
//! Run: `cargo run --release --example active_selection --features burn`

use std::time::{Duration, Instant};

use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{activation, Device, Int, Tensor, TensorData};

use stableprop::burn_sdp::{propagate_linear_full, propagate_relu_full, MomentsFull};

const D_IN: usize = 2;
const HIDDEN: usize = 16;
const N_CLASS: usize = 2;
const N_POOL: usize = 256;
const N_TEST: usize = 512;
const INITIAL: usize = 16;
const BUDGETS: [usize; 4] = [16, 32, 64, 96];
const EPOCHS: usize = 250;
const INPUT_STD: f64 = 0.12;
const MC_DRAWS: usize = 64;
const EVAL_DRAWS: usize = 32;
const SEEDS: [u64; 3] = [0x51EC_0001, 0x51EC_0002, 0x51EC_0003];
/// Only round-off at this scale is projected to zero; larger negative output
/// variances signal an invalid propagated covariance or a numerical failure.
const VARIANCE_TOLERANCE: f64 = 1e-7;

#[derive(Module, Debug)]
struct Net {
    lin1: Linear,
    lin2: Linear,
}

impl Net {
    fn init(device: &Device) -> Self {
        let model = Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, N_CLASS).init(device),
        };
        // Burn parameters initialize lazily; materialize before cloning a baseline.
        for layer in [&model.lin1, &model.lin2] {
            drop(layer.weight.val());
            if let Some(bias) = &layer.bias {
                drop(bias.val());
            }
        }
        model
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.lin2.forward(activation::relu(self.lin1.forward(x)))
    }
}

#[derive(Clone)]
struct Data {
    x: Vec<f32>,
    y: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
enum Policy {
    Random,
    Entropy,
    Analytic,
    MonteCarlo,
    Diversity,
}

impl Policy {
    const ALL: [Self; 5] = [
        Self::Random,
        Self::Entropy,
        Self::Analytic,
        Self::MonteCarlo,
        Self::Diversity,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Entropy => "entropy",
            Self::Analytic => "analytic 2trPCP",
            Self::MonteCarlo => "MC disagreement",
            Self::Diversity => "input farthest-first",
        }
    }
}

struct Row {
    policy: Policy,
    budget: usize,
    clean: f64,
    noisy: f64,
    selected_classes: [usize; N_CLASS],
    pool_cover_radius: f64,
    acquisition: Duration,
    pearson: Option<f64>,
    spearman: Option<f64>,
}

type Agreement = (Option<f64>, Option<f64>);

/// Small local RNG: data, random acquisition, and every noise view are
/// independent of Burn's backend RNG.
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
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
    fn index(&mut self, upper: usize) -> usize {
        (self.uniform() * upper as f64) as usize
    }
}

/// Two interleaved half-moons.  Generator order deliberately makes the first
/// sixteen labels class-balanced.  This fixed warm start is not
/// label-structure-free; later acquisition never reads labels.
fn make_data(n: usize, rng: &mut Rng) -> Data {
    let mut x = Vec::with_capacity(n * D_IN);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let class = (i % N_CLASS) as i32;
        let angle = rng.uniform() * std::f64::consts::PI;
        let (mut a, mut b) = (angle.cos(), angle.sin());
        if class == 1 {
            a = 1.0 - a;
            b = 0.45 - b;
        }
        x.push((a + 0.14 * rng.normal()) as f32);
        x.push((b + 0.14 * rng.normal()) as f32);
        y.push(class);
    }
    Data { x, y }
}

fn train(init: &Net, data: &Data, chosen: &[usize], device: &Device) -> Net {
    let mut xv = Vec::with_capacity(chosen.len() * D_IN);
    let mut yv = Vec::with_capacity(chosen.len());
    for &i in chosen {
        xv.extend_from_slice(&data.x[i * D_IN..(i + 1) * D_IN]);
        yv.push(data.y[i]); // labels are used only once this index is selected.
    }
    let x = Tensor::<2>::from_data(TensorData::new(xv, [chosen.len(), D_IN]), device);
    let y = Tensor::<1, Int>::from_data(TensorData::new(yv, [chosen.len()]), device);
    let mut model = init.clone();
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..EPOCHS {
        let loss = CrossEntropyLoss::new(None, device).forward(model.forward(x.clone()), y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(3e-2, model, grads);
    }
    model
}

#[derive(Clone)]
struct Weights {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

fn weights(model: &Net) -> Weights {
    Weights {
        w1: model
            .lin1
            .weight
            .val()
            .inner()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
        b1: model
            .lin1
            .bias
            .as_ref()
            .unwrap()
            .val()
            .inner()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
        w2: model
            .lin2
            .weight
            .val()
            .inner()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
        b2: model
            .lin2
            .bias
            .as_ref()
            .unwrap()
            .val()
            .inner()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
    }
}

fn logits(w: &Weights, x0: f32, x1: f32) -> [f64; N_CLASS] {
    let mut h = [0.0f64; HIDDEN];
    for (j, value) in h.iter_mut().enumerate() {
        *value =
            (x0 as f64 * w.w1[j] as f64 + x1 as f64 * w.w1[HIDDEN + j] as f64 + w.b1[j] as f64)
                .max(0.0);
    }
    let mut out = [0.0; N_CLASS];
    for (k, value) in out.iter_mut().enumerate() {
        *value = w.b2[k] as f64
            + (0..HIDDEN)
                .map(|j| h[j] * w.w2[j * N_CLASS + k] as f64)
                .sum::<f64>();
    }
    out
}

fn entropy(w: &Weights, x: &[f32]) -> f64 {
    let z = logits(w, x[0], x[1]);
    let top = z[0].max(z[1]);
    let p0 = (z[0] - top).exp() / ((z[0] - top).exp() + (z[1] - top).exp());
    if p0 <= 0.0 || p0 >= 1.0 {
        return 0.0;
    }
    -p0 * p0.ln() - (1.0 - p0) * (1.0 - p0).ln()
}

fn analytic_scores(model: &Net, x: &[f32], n: usize, device: &Device) -> Vec<f64> {
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val().inner());
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val().inner());
    let input = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [n, D_IN]), device);
    let variance = Tensor::<2>::full([n, D_IN], INPUT_STD * INPUT_STD, device);
    let m1 = propagate_relu_full(&propagate_linear_full(
        &MomentsFull::from_diagonal(input, variance),
        w1,
        b1,
    ));
    let m2 = propagate_linear_full(&m1, w2, b2);
    let cov = m2.cov.to_data().try_to_vec::<f32>().unwrap();
    (0..n)
        .map(|i| {
            let block = &cov[i * N_CLASS * N_CLASS..(i + 1) * N_CLASS * N_CLASS];
            centered_disagreement(&block.iter().map(|&value| value as f64).collect::<Vec<_>>())
        })
        .collect()
}

fn checked_variance(value: f64, source: &str) -> f64 {
    assert!(value.is_finite(), "{source} variance is non-finite");
    assert!(
        value >= -VARIANCE_TOLERANCE,
        "{source} variance {value:e} is materially negative"
    );
    value.max(0.0)
}

/// `2 * trace(P Cov P)` for `P = I - 11^T / K`, retaining only shifts that
/// can change relative logits.
fn centered_disagreement(cov: &[f64]) -> f64 {
    let trace = (0..N_CLASS).map(|i| cov[i * N_CLASS + i]).sum::<f64>();
    let total = cov.iter().sum::<f64>();
    checked_variance(
        2.0 * (trace - total / N_CLASS as f64),
        "analytic centered-logit",
    )
}

/// `2 * sum_c Var[(P f(x + eps))_c]`, estimated from 64 iid views.  This
/// equals expected squared disagreement of two independent centered views.
fn mc_scores(model: &Net, x: &[f32], n: usize, rng: &mut Rng, device: &Device) -> Vec<f64> {
    let mut views = Vec::with_capacity(n * MC_DRAWS * D_IN);
    for i in 0..n {
        let p = &x[i * D_IN..(i + 1) * D_IN];
        for _ in 0..MC_DRAWS {
            views.push(p[0] + (INPUT_STD * rng.normal()) as f32);
            views.push(p[1] + (INPUT_STD * rng.normal()) as f32);
        }
    }
    // This is the same Flex affine/ReLU forward path as the analytic
    // propagator's fixed model, with explicitly seeded host-side input noise.
    let input = Tensor::<2>::from_data(TensorData::new(views, [n * MC_DRAWS, D_IN]), device);
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().unwrap().val().inner();
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().unwrap().val().inner();
    let output = (activation::relu(input.matmul(w1) + b1.reshape([1, HIDDEN])).matmul(w2)
        + b2.reshape([1, N_CLASS]))
    .to_data()
    .try_to_vec::<f32>()
    .unwrap();
    let mut scores = vec![0.0; n];
    for (i, score) in scores.iter_mut().enumerate() {
        for c in 0..N_CLASS {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for draw in 0..MC_DRAWS {
                let base = (i * MC_DRAWS + draw) * N_CLASS;
                let mean =
                    (0..N_CLASS).map(|k| output[base + k] as f64).sum::<f64>() / N_CLASS as f64;
                let value = output[base + c] as f64 - mean;
                sum += value;
                sum_sq += value * value;
            }
            *score += checked_variance(
                2.0 * (sum_sq - sum * sum / MC_DRAWS as f64) / (MC_DRAWS - 1) as f64,
                "MC centered-logit",
            );
        }
    }
    scores
}

fn correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        ab += (x - ma) * (y - mb);
        aa += (x - ma).powi(2);
        bb += (y - mb).powi(2);
    }
    (aa > 0.0 && bb > 0.0).then(|| ab / (aa * bb).sqrt())
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
    let mut out = vec![0.0; values.len()];
    let mut first = 0;
    while first < order.len() {
        let mut end = first + 1;
        while end < order.len() && values[order[first]] == values[order[end]] {
            end += 1;
        }
        let average_rank = (first + end - 1) as f64 / 2.0;
        for &i in &order[first..end] {
            out[i] = average_rank;
        }
        first = end;
    }
    out
}

fn select_top(indices: &[usize], scores: &[f64], count: usize, tie_rng: &mut Rng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..indices.len()).collect();
    let mut tie_rank: Vec<usize> = (0..indices.len()).collect();
    for i in (1..tie_rank.len()).rev() {
        tie_rank.swap(i, tie_rng.index(i + 1));
    }
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap()
            .then_with(|| tie_rank[a].cmp(&tie_rank[b]))
    });
    order.into_iter().take(count).map(|i| indices[i]).collect()
}

/// Greedily cover raw two-dimensional input space without using labels or a
/// learned representation. Each pick maximizes its nearest squared distance to
/// already chosen pool points and earlier picks in this acquisition chunk.
fn farthest_first(
    features: &[f32],
    chosen: &[usize],
    count: usize,
    tie_rng: &mut Rng,
) -> Vec<usize> {
    assert_eq!(
        features.len() % D_IN,
        0,
        "features must contain complete rows"
    );
    let n = features.len() / D_IN;
    assert!(
        !chosen.is_empty(),
        "farthest-first requires an initial chosen set"
    );
    assert!(
        chosen.iter().all(|&index| index < n),
        "chosen index is outside the pool"
    );
    assert!(
        chosen
            .iter()
            .enumerate()
            .all(|(i, index)| chosen[..i].iter().all(|earlier| earlier != index)),
        "chosen indices must be distinct"
    );
    assert!(
        count <= n - chosen.len(),
        "acquisition exceeds remaining pool"
    );

    let mut used = vec![false; n];
    for &index in chosen {
        used[index] = true;
    }
    let mut nearest: Vec<f64> = (0..n)
        .map(|point| {
            chosen
                .iter()
                .map(|&center| squared_distance(features, point, center))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let mut selected = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best_distance = f64::NEG_INFINITY;
        let mut ties = Vec::new();
        for candidate in 0..n {
            if used[candidate] {
                continue;
            }
            let distance = nearest[candidate];
            if distance > best_distance {
                best_distance = distance;
                ties.clear();
                ties.push(candidate);
            } else if distance == best_distance {
                ties.push(candidate);
            }
        }
        assert!(
            !ties.is_empty(),
            "farthest-first had no remaining candidate"
        );
        let next = ties[tie_rng.index(ties.len())];
        used[next] = true;
        selected.push(next);
        for candidate in 0..n {
            if !used[candidate] {
                nearest[candidate] =
                    nearest[candidate].min(squared_distance(features, candidate, next));
            }
        }
    }
    selected
}

fn squared_distance(features: &[f32], left: usize, right: usize) -> f64 {
    let dx = features[left * D_IN] as f64 - features[right * D_IN] as f64;
    let dy = features[left * D_IN + 1] as f64 - features[right * D_IN + 1] as f64;
    dx * dx + dy * dy
}

/// Largest nearest-neighbor distance from a pool point to the selected set.
/// This post-hoc geometry diagnostic is independent of labels and acquisition
/// scores.
fn pool_cover_radius(features: &[f32], chosen: &[usize]) -> f64 {
    assert!(
        !chosen.is_empty(),
        "coverage requires at least one chosen point"
    );
    let n = features.len() / D_IN;
    assert_eq!(
        features.len() % D_IN,
        0,
        "features must contain complete rows"
    );
    assert!(
        chosen.iter().all(|&index| index < n),
        "chosen index is outside the pool"
    );
    (0..n)
        .map(|point| {
            chosen
                .iter()
                .map(|&center| squared_distance(features, point, center))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .fold(0.0, f64::max)
}

fn acquire(
    policy: Policy,
    model: &Net,
    pool_features: &[f32],
    chosen: &[usize],
    count: usize,
    seed: u64,
    nd_device: &Device,
) -> (Vec<usize>, Duration, Option<Agreement>) {
    let started = Instant::now();
    let mut candidates: Vec<usize> = (0..N_POOL).filter(|i| !chosen.contains(i)).collect();
    let mut score_rng = Rng::new(seed ^ 0x5C0E_0001);
    let mut tie_rng = Rng::new(seed ^ 0x71E8_0002);
    if matches!(policy, Policy::Random) {
        for i in (1..candidates.len()).rev() {
            candidates.swap(i, tie_rng.index(i + 1));
        }
        return (
            candidates.into_iter().take(count).collect(),
            started.elapsed(),
            None,
        );
    }
    if matches!(policy, Policy::Diversity) {
        return (
            farthest_first(pool_features, chosen, count, &mut tie_rng),
            started.elapsed(),
            None,
        );
    }
    let mut x = Vec::with_capacity(candidates.len() * D_IN);
    for &i in &candidates {
        x.extend_from_slice(&pool_features[i * D_IN..(i + 1) * D_IN]);
    }
    let scores = match policy {
        Policy::Entropy => {
            let w = weights(model);
            (0..candidates.len())
                .map(|i| entropy(&w, &x[i * D_IN..]))
                .collect()
        }
        Policy::Analytic => analytic_scores(model, &x, candidates.len(), nd_device),
        Policy::MonteCarlo => mc_scores(model, &x, candidates.len(), &mut score_rng, nd_device),
        Policy::Random | Policy::Diversity => unreachable!(),
    };
    assert!(
        scores.iter().all(|score| score.is_finite()),
        "{policy:?} produced a non-finite acquisition score"
    );
    let selected = select_top(&candidates, &scores, count, &mut tie_rng);
    let acquisition = started.elapsed();
    // Agreement is diagnostic only: it is deliberately outside the policy's
    // ranking time and only needs to be measured for the MC selector.
    let agreement = matches!(policy, Policy::MonteCarlo).then(|| {
        let analytic = analytic_scores(model, &x, candidates.len(), nd_device);
        (
            correlation(&analytic, &scores),
            correlation(&ranks(&analytic), &ranks(&scores)),
        )
    });
    (selected, acquisition, agreement)
}

/// Count labels only after acquisition has returned fixed indices. Keeping the
/// labels out of `acquire` makes this diagnostic unable to affect a policy.
fn selected_class_counts(selected: &[usize], labels: &[i32]) -> [usize; N_CLASS] {
    let mut counts = [0; N_CLASS];
    for &index in selected {
        let class = labels[index] as usize;
        assert!(class < N_CLASS, "selected label is outside the class range");
        counts[class] += 1;
    }
    counts
}

fn accuracy(w: &Weights, data: &Data) -> f64 {
    let mut correct = 0usize;
    for i in 0..data.y.len() {
        let p = &data.x[i * D_IN..(i + 1) * D_IN];
        let z = logits(w, p[0], p[1]);
        if (z[1] > z[0]) as i32 == data.y[i] {
            correct += 1;
        }
    }
    correct as f64 / data.y.len() as f64
}

fn noisy_accuracy(w: &Weights, data: &Data, rng: &mut Rng) -> f64 {
    let mut correct = 0usize;
    for _ in 0..EVAL_DRAWS {
        for i in 0..data.y.len() {
            let p = &data.x[i * D_IN..(i + 1) * D_IN];
            let z = logits(
                w,
                p[0] + (INPUT_STD * rng.normal()) as f32,
                p[1] + (INPUT_STD * rng.normal()) as f32,
            );
            if (z[1] > z[0]) as i32 == data.y[i] {
                correct += 1;
            }
        }
    }
    correct as f64 / (data.y.len() * EVAL_DRAWS) as f64
}

fn prediction_bytes(model: &Net, data: &Data, device: &Device) -> Vec<f32> {
    let input = Tensor::<2>::from_data(
        TensorData::new(data.x.clone(), [data.y.len(), D_IN]),
        device,
    );
    model.forward(input).to_data().try_to_vec::<f32>().unwrap()
}

fn main() {
    let ad_device = Device::flex().autodiff();
    let nd_device = Device::flex();
    let mut rows = Vec::new();
    for &seed in &SEEDS {
        let mut pool_rng = Rng::new(seed ^ 0xDADA_0001);
        let mut test_rng = Rng::new(seed ^ 0xDADA_0002);
        let pool = make_data(N_POOL, &mut pool_rng);
        let test = make_data(N_TEST, &mut test_rng);
        ad_device.seed(seed ^ 0x1A17_0000);
        // The model is initialized once per seed and cloned for every fit.
        // The following prediction equality assertion makes this shared-init
        // control observable instead of relying on that implementation detail.
        let init = Net::init(&ad_device);
        let mut initial_prediction = None;
        for policy in Policy::ALL {
            let mut chosen: Vec<usize> = (0..INITIAL).collect();
            for &budget in &BUDGETS {
                let scorer = train(&init, &pool, &chosen, &ad_device);
                if budget == INITIAL {
                    let prediction = prediction_bytes(&scorer, &test, &ad_device);
                    if let Some(reference) = &initial_prediction {
                        assert_eq!(reference, &prediction, "shared initial control diverged");
                    } else {
                        initial_prediction = Some(prediction);
                    }
                }
                let (fitted, acquisition, agreement) = if budget == INITIAL {
                    (scorer, Duration::ZERO, None)
                } else {
                    let (new_indices, acquisition, agreement) = acquire(
                        policy,
                        &scorer,
                        &pool.x,
                        &chosen,
                        budget - chosen.len(),
                        seed ^ budget as u64,
                        &nd_device,
                    );
                    chosen.extend(new_indices);
                    (
                        train(&init, &pool, &chosen, &ad_device),
                        acquisition,
                        agreement,
                    )
                };
                let w = weights(&fitted);
                let mut noise_rng = Rng::new(seed ^ ((budget as u64) << 24) ^ 0xEAA1_0000);
                rows.push(Row {
                    policy,
                    budget,
                    clean: accuracy(&w, &test),
                    noisy: noisy_accuracy(&w, &test, &mut noise_rng),
                    selected_classes: selected_class_counts(&chosen, &pool.y),
                    pool_cover_radius: pool_cover_radius(&pool.x, &chosen),
                    acquisition,
                    pearson: agreement.and_then(|x| x.0),
                    spearman: agreement.and_then(|x| x.1),
                });
            }
        }
    }
    println!(
        "controlled selection toy: {} seeds, pool {}, test {}, {} epochs; input std {:.2}",
        SEEDS.len(),
        N_POOL,
        N_TEST,
        EPOCHS,
        INPUT_STD
    );
    println!("each row averages over seeds; acquisition excludes retraining and uses {MC_DRAWS} iid MC views.");
    println!("\nlearning curve (accuracy; higher is better):");
    println!(
        "  {:<22} {:>6} {:>9} {:>9}",
        "policy", "labels", "clean", "noisy"
    );
    for policy in Policy::ALL {
        for &budget in &BUDGETS {
            let group: Vec<&Row> = rows
                .iter()
                .filter(|r| r.policy as u8 == policy as u8 && r.budget == budget)
                .collect();
            let mean =
                |f: fn(&Row) -> f64| group.iter().map(|r| f(r)).sum::<f64>() / group.len() as f64;
            println!(
                "  {:<22} {:>6} {:>9.3} {:>9.3}",
                policy.name(),
                budget,
                mean(|r| r.clean),
                mean(|r| r.noisy)
            );
        }
    }
    println!("\nselected-set diagnostics (mean per seed; labels read after acquisition):");
    println!(
        "  {:<22} {:>6} {:>10} {:>10} {:>11}",
        "policy", "labels", "class 0", "class 1", "pool radius"
    );
    for policy in Policy::ALL {
        for &budget in &BUDGETS {
            let group: Vec<&Row> = rows
                .iter()
                .filter(|r| r.policy as u8 == policy as u8 && r.budget == budget)
                .collect();
            let mean_count = |class: usize| {
                group
                    .iter()
                    .map(|r| r.selected_classes[class] as f64)
                    .sum::<f64>()
                    / group.len() as f64
            };
            println!(
                "  {:<22} {:>6} {:>10.1} {:>10.1} {:>11.3}",
                policy.name(),
                budget,
                mean_count(0),
                mean_count(1),
                group.iter().map(|r| r.pool_cover_radius).sum::<f64>() / group.len() as f64,
            );
        }
    }
    println!("pool radius: largest Euclidean distance from a pool point to the selected set; lower is better coverage.");
    println!("\nacquisition time only (mean milliseconds; lower is cheaper):");
    println!("  {:<22} {:>6} {:>10}", "policy", "labels", "ms");
    for policy in Policy::ALL {
        for &budget in &BUDGETS {
            let group: Vec<&Row> = rows
                .iter()
                .filter(|r| r.policy as u8 == policy as u8 && r.budget == budget)
                .collect();
            let ms = group
                .iter()
                .map(|r| r.acquisition.as_secs_f64() * 1e3)
                .sum::<f64>()
                / group.len() as f64;
            println!("  {:<22} {:>6} {:>10.3}", policy.name(), budget, ms);
        }
    }
    println!("\nanalytic versus MC score agreement during acquisition (Pearson / Spearman):");
    for &budget in &BUDGETS {
        let group: Vec<&Row> = rows
            .iter()
            .filter(|r| r.policy as u8 == Policy::MonteCarlo as u8 && r.budget == budget)
            .collect();
        if group
            .iter()
            .all(|r| r.pearson.is_none() && r.spearman.is_none())
        {
            println!("  labels {budget:>3}: baseline shared control (no acquisition)");
            continue;
        }
        if group
            .iter()
            .any(|r| r.pearson.is_none() || r.spearman.is_none())
        {
            println!("  labels {budget:>3}: N/A / N/A (zero score variance)");
            continue;
        }
        let p = group.iter().map(|r| r.pearson.unwrap()).sum::<f64>() / group.len() as f64;
        let s = group.iter().map(|r| r.spearman.unwrap()).sum::<f64>() / group.len() as f64;
        println!("  labels {budget:>3}: {p:.3} / {s:.3}");
    }
    println!("\nThis clean synthetic result does not establish label value or robustness on noisy, irrelevant, or non-invariant pools.");
}

#[cfg(test)]
mod tests {
    use super::{
        centered_disagreement, correlation, farthest_first, pool_cover_radius, ranks, Device, Net,
        Rng, Tensor, TensorData, D_IN,
    };
    use proptest::prelude::*;

    fn squared_cover_radius(points: &[(i16, i16)], chosen: &[usize]) -> i64 {
        points
            .iter()
            .map(|&(x, y)| {
                chosen
                    .iter()
                    .map(|&center| {
                        let dx = i64::from(x) - i64::from(points[center].0);
                        let dy = i64::from(y) - i64::from(points[center].1);
                        dx * dx + dy * dy
                    })
                    .min()
                    .unwrap()
            })
            .max()
            .unwrap()
    }

    #[test]
    fn centered_disagreement_ignores_common_logit_shift_covariance() {
        // Adding v * 11^T models a random shift shared by both logits.  It
        // changes neither the centered covariance nor the class margin.
        let base = [4.0, 1.0, 1.0, 3.0];
        let shifted = [13.0, 10.0, 10.0, 12.0];
        let expected_margin_variance = 5.0;
        assert!((centered_disagreement(&base) - expected_margin_variance).abs() < 1e-12);
        assert!((centered_disagreement(&shifted) - expected_margin_variance).abs() < 1e-12);
    }

    #[test]
    fn spearman_helpers_average_ties_and_reject_degenerate_scores() {
        let left = ranks(&[2.0, 2.0, 1.0, 0.0]);
        let right = ranks(&[20.0, 20.0, 10.0, 0.0]);
        assert_eq!(left, vec![2.5, 2.5, 1.0, 0.0]);
        assert_eq!(right, vec![2.5, 2.5, 1.0, 0.0]);
        assert!((correlation(&left, &right).unwrap() - 1.0).abs() < 1e-12);

        let tied = ranks(&[7.0, 7.0, 7.0]);
        assert_eq!(tied, vec![1.0, 1.0, 1.0]);
        assert_eq!(correlation(&tied, &[0.0, 1.0, 2.0]), None);
        assert_eq!(correlation(&[1.0, 1.0], &[2.0, 3.0]), None);
    }

    #[test]
    fn farthest_first_improves_raw_pool_coverage_over_static_top_scores() {
        let points = vec![0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 9.0, 0.0, 10.0, 0.0];
        let mut tie_rng = Rng::new(0xD1FE_0001);
        let selected = farthest_first(&points, &[0], 2, &mut tie_rng);
        assert_eq!(selected, vec![4, 2]);

        let mut greedy_chosen = vec![0];
        greedy_chosen.extend(&selected);
        assert!((pool_cover_radius(&points, &greedy_chosen) - 1.0).abs() < 1e-12);

        let static_selected = [4, 3];
        let mut static_chosen = vec![0];
        static_chosen.extend(static_selected);
        assert!((pool_cover_radius(&points, &static_chosen) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn farthest_first_uses_seeded_ties_without_reselecting_points() {
        let points = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut first_rng = Rng::new(0xD1FE_0002);
        let mut second_rng = Rng::new(0xD1FE_0002);
        let first = farthest_first(&points, &[0], 3, &mut first_rng);
        let second = farthest_first(&points, &[0], 3, &mut second_rng);
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert!(first.iter().all(|&index| index != 0));
        assert!(first
            .iter()
            .enumerate()
            .all(|(i, index)| { first[..i].iter().all(|earlier| earlier != index) }));
    }

    proptest! {
        #[test]
        fn farthest_first_is_a_two_approximation_on_tiny_integer_pools(
            points in prop::collection::vec((-8i16..9, -8i16..9), 4..9),
            initial_count in 1usize..4,
            seed in any::<u64>(),
        ) {
            let features: Vec<f32> = points
                .iter()
                .flat_map(|&(x, y)| [x as f32, y as f32])
                .collect();
            let mut tie_rng = Rng::new(seed);
            let chosen: Vec<usize> = (0..initial_count.min(points.len() - 2)).collect();
            let selected = farthest_first(&features, &chosen, 2, &mut tie_rng);
            prop_assert_eq!(selected.len(), 2);
            prop_assert!(selected.iter().all(|&index| index >= chosen.len() && index < points.len()));
            prop_assert_ne!(selected[0], selected[1]);

            let mut greedy_chosen = chosen.clone();
            greedy_chosen.extend(&selected);
            let greedy_radius_squared = squared_cover_radius(&points, &greedy_chosen);
            let mut optimal_radius_squared = i64::MAX;
            for left in chosen.len()..points.len() {
                for right in left + 1..points.len() {
                    let mut candidate = chosen.clone();
                    candidate.extend([left, right]);
                    optimal_radius_squared = optimal_radius_squared
                        .min(squared_cover_radius(&points, &candidate));
                }
            }
            prop_assert!(greedy_radius_squared <= 4 * optimal_radius_squared);
        }
    }

    #[test]
    fn cloned_baseline_has_identical_forward_predictions() {
        let device = Device::flex().autodiff();
        device.seed(0x51EC_7E57);
        let baseline = Net::init(&device);
        let left = baseline.clone();
        let right = baseline.clone();
        let probe = Tensor::<2>::from_data(
            TensorData::new(vec![-1.0, 0.5, 1.25, -0.75], [2, D_IN]),
            &device,
        );

        let left = left
            .forward(probe.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let right = right
            .forward(probe)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(
            left, right,
            "cloned baselines must share initialized weights"
        );
    }
}
