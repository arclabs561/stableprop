//! Real-data calibration eval: does momentprop's analytic uncertainty actually
//! flag the nodes a GCN gets wrong on Cora?
//!
//! Trains a 2-layer GCN on the Cora citation graph (Kipf & Welling 2017), then
//! puts a Gaussian over the input features and propagates it through the trained
//! net with momentprop to get a per-node predictive variance in one pass. The test
//! that risks the claim: rank test nodes by that variance and **abstain on the
//! most-uncertain ones** -- if the uncertainty is meaningful, accuracy on the
//! retained nodes climbs above the random-abstention baseline. We also check that
//! the analytic ranking agrees with Monte-Carlo sampling (Spearman).
//!
//! Data is reused from propago's example (gitignored). Run:
//! `cargo run --release --example cora_uncertainty --features burn`
//! (optionally pass a path to a dir holding `cora.content` + `cora.cites`).

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{activation, Int, Tensor, TensorData};
use burn_ndarray::NdArray;

use burn::nn::{Linear, LinearConfig};
use momentprop::burn_sdp::{propagate_linear, propagate_matmul_left, propagate_relu, Moments};

const HIDDEN: usize = 16;
const INPUT_STD: f64 = 0.1;
const MC_SAMPLES: usize = 200;

struct Graph {
    n: usize,
    n_features: usize,
    n_classes: usize,
    features: Vec<f32>,
    labels: Vec<i32>,
    adj_norm: Vec<f32>,
}

/// LBC-format loader (same parser propago's cora example uses).
fn load_planetoid(dir: &Path, name: &str) -> std::io::Result<Graph> {
    let content = std::fs::read_to_string(dir.join(format!("{name}.content")))?;
    let cites = std::fs::read_to_string(dir.join(format!("{name}.cites")))?;

    let n_features = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').count().saturating_sub(2))
        .unwrap_or(0);

    let mut label_names: Vec<&str> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.rsplit('\t').next().unwrap())
        .collect();
    label_names.sort_unstable();
    label_names.dedup();
    let n_classes = label_names.len();
    let class_id: HashMap<&str, i32> = label_names
        .iter()
        .enumerate()
        .map(|(i, &nm)| (nm, i as i32))
        .collect();

    let mut id_to_idx: HashMap<String, usize> = HashMap::new();
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for line in content.lines().filter(|l| !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let idx = id_to_idx.len();
        id_to_idx.insert(cols[0].to_string(), idx);
        for f in &cols[1..=n_features] {
            features.push(f.parse::<f32>().unwrap_or(0.0));
        }
        labels.push(class_id[cols[n_features + 1]]);
    }
    let n = labels.len();

    let mut adj = vec![0.0f32; n * n];
    for i in 0..n {
        adj[i * n + i] = 1.0;
    }
    for line in cites.lines().filter(|l| !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let (a, b) = (it.next().unwrap(), it.next().unwrap());
        if let (Some(&i), Some(&j)) = (id_to_idx.get(a), id_to_idx.get(b)) {
            adj[i * n + j] = 1.0;
            adj[j * n + i] = 1.0;
        }
    }
    let mut deg = vec![0.0f32; n];
    for i in 0..n {
        deg[i] = (0..n).map(|j| adj[i * n + j]).sum();
    }
    let inv_sqrt: Vec<f32> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    let mut adj_norm = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let a = adj[i * n + j];
            if a != 0.0 {
                adj_norm[i * n + j] = inv_sqrt[i] * a * inv_sqrt[j];
            }
        }
    }
    Ok(Graph {
        n,
        n_features,
        n_classes,
        features,
        labels,
        adj_norm,
    })
}

/// A GCN layer is `adj @ (x W + b)`; built from `burn::nn::Linear` so the model
/// is trainable without depending on a specific propago version (the spike in
/// `gcn_uncertainty.rs` exercises propago's `GCNConv` directly).
#[derive(Module, Debug)]
struct Gcn<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> Gcn<B> {
    fn init(n_features: usize, n_classes: usize, device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(n_features, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, n_classes).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = adj.clone().matmul(self.lin1.forward(x));
        let h = activation::relu(h);
        adj.matmul(self.lin2.forward(h))
    }

    /// SDP through one GCN layer: linear then adjacency aggregation.
    fn sdp_gcn(m: &Moments<B>, lin: &Linear<B>, adj: Tensor<B, 2>) -> Moments<B> {
        let w = lin.weight.val();
        let b = lin.bias.as_ref().map(|p| p.val());
        propagate_matmul_left(adj, &propagate_linear(m, w, b))
    }

    /// Per-node total predictive variance from input-feature noise (one pass).
    fn sdp_node_variance(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>, input_std: f64) -> Vec<f64> {
        let [n, d] = x.dims();
        let var0 = Tensor::<B, 2>::full([n, d], input_std * input_std, &x.device());
        let m0 = Moments::new(x, var0);
        let m1 = propagate_relu(&Self::sdp_gcn(&m0, &self.lin1, adj.clone()));
        let m2 = Self::sdp_gcn(&m1, &self.lin2, adj);
        let [_, c] = m2.var.dims();
        let v = m2.var.to_data().to_vec::<f32>().unwrap();
        (0..n)
            .map(|i| (0..c).map(|j| v[i * c + j] as f64).sum())
            .collect()
    }
}

fn argmax_correct(logits: &[f32], labels: &[i32], i: usize, c: usize) -> bool {
    let row = &logits[i * c..(i + 1) * c];
    let pred = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32;
    pred == labels[i]
}

fn shuffle(v: &mut [usize], state: &mut u64) {
    for i in (1..v.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        v.swap(i, (*state % (i as u64 + 1)) as usize);
    }
}

fn split(labels: &[i32], n_classes: usize) -> (Vec<usize>, Vec<usize>) {
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &c) in labels.iter().enumerate() {
        by_class[c as usize].push(i);
    }
    let mut train = Vec::new();
    for bucket in &mut by_class {
        shuffle(bucket, &mut rng);
        train.extend(bucket.iter().take(20).copied());
    }
    let train_set: std::collections::HashSet<usize> = train.iter().copied().collect();
    let mut rest: Vec<usize> = (0..labels.len())
        .filter(|i| !train_set.contains(i))
        .collect();
    shuffle(&mut rest, &mut rng);
    let test = rest.into_iter().take(1000).collect();
    (train, test)
}

/// Accuracy over `idx` after retaining the `coverage` fraction with the lowest
/// uncertainty (uncertainty[k] aligns with idx[k]). `coverage = 1.0` keeps all.
fn accuracy_at_coverage(
    logits: &[f32],
    labels: &[i32],
    idx: &[usize],
    uncertainty: &[f64],
    c: usize,
    coverage: f64,
) -> f64 {
    let mut order: Vec<usize> = (0..idx.len()).collect();
    order.sort_by(|&a, &b| uncertainty[a].partial_cmp(&uncertainty[b]).unwrap());
    let keep = ((idx.len() as f64) * coverage).round() as usize;
    let kept = &order[..keep.max(1)];
    let correct = kept
        .iter()
        .filter(|&&k| argmax_correct(logits, labels, idx[k], c))
        .count();
    correct as f64 / kept.len() as f64
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0.0; v.len()];
        for (rank, &i) in idx.iter().enumerate() {
            r[i] = rank as f64;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in ra.iter().zip(&rb) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// AUROC of `score` as a detector of the boolean `positive` label (here:
/// node is misclassified). 0.5 = uninformative, 1.0 = uncertainty perfectly
/// ranks every wrong prediction above every correct one. Mann-Whitney U with
/// tie-corrected average ranks.
fn auroc(score: &[f64], positive: &[bool]) -> f64 {
    let n = score.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| score[i].partial_cmp(&score[j]).unwrap());
    let mut rank = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && score[order[j + 1]] == score[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0; // 1-based mid-rank for ties
        for k in i..=j {
            rank[order[k]] = avg;
        }
        i = j + 1;
    }
    let n_pos = positive.iter().filter(|&&p| p).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    let sum_pos: f64 = (0..n).filter(|&k| positive[k]).map(|k| rank[k]).sum();
    (sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos as f64 * n_neg as f64)
}

fn run<B: AutodiffBackend>(device: B::Device, dir: &Path, name: &str) {
    let g = load_planetoid(dir, name).unwrap();
    let (train_idx, test_idx) = split(&g.labels, g.n_classes);
    println!(
        "dataset: {name}  nodes: {}  features: {}  classes: {}  test: {}",
        g.n,
        g.n_features,
        g.n_classes,
        test_idx.len()
    );

    let x = Tensor::<B, 2>::from_data(
        TensorData::new(g.features.clone(), [g.n, g.n_features]),
        &device,
    );
    let adj = Tensor::<B, 2>::from_data(TensorData::new(g.adj_norm.clone(), [g.n, g.n]), &device);
    let targets = Tensor::<B, 1, Int>::from_data(TensorData::new(g.labels.clone(), [g.n]), &device);
    let train_sel = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            train_idx.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            [train_idx.len()],
        ),
        &device,
    );

    let mut model = Gcn::<B>::init(g.n_features, g.n_classes, &device);
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(5e-4)))
        .init();
    println!("training 2-layer GCN (200 epochs)...");
    for epoch in 1..=200 {
        let logits = model.forward(x.clone(), adj.clone());
        let train_logits = logits.select(0, train_sel.clone());
        let train_targets = targets.clone().select(0, train_sel.clone());
        let loss = CrossEntropyLoss::new(None, &device).forward(train_logits, train_targets);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(0.01, model, grads);
        let _ = epoch;
    }

    let logits = model.forward(x.clone(), adj.clone());
    let logits_v = logits.into_data().to_vec::<f32>().unwrap();
    let base_acc = {
        let c = test_idx
            .iter()
            .filter(|&&i| argmax_correct(&logits_v, &g.labels, i, g.n_classes))
            .count();
        c as f64 / test_idx.len() as f64
    };
    println!("test accuracy (full coverage): {base_acc:.4}\n");

    // --- SDP per-node uncertainty (one analytic pass) ---
    let node_var = model.sdp_node_variance(x.clone(), adj.clone(), INPUT_STD);
    let u_sdp: Vec<f64> = test_idx.iter().map(|&i| node_var[i]).collect();

    // --- MC per-node uncertainty (oracle) ---
    let len = g.n * g.n_classes;
    let mut acc_mean = vec![0.0f64; len];
    let mut acc_sq = vec![0.0f64; len];
    for _ in 0..MC_SAMPLES {
        let noise = Tensor::<B, 2>::random(
            [g.n, g.n_features],
            burn::tensor::Distribution::Normal(0.0, INPUT_STD),
            &device,
        );
        let yk = model
            .forward(x.clone() + noise, adj.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for i in 0..len {
            acc_mean[i] += yk[i] as f64;
            acc_sq[i] += (yk[i] as f64).powi(2);
        }
    }
    let kf = MC_SAMPLES as f64;
    let mc_node_var: Vec<f64> = (0..g.n)
        .map(|node| {
            (0..g.n_classes)
                .map(|c| {
                    let idx = node * g.n_classes + c;
                    (acc_sq[idx] - acc_mean[idx].powi(2) / kf) / (kf - 1.0)
                })
                .sum()
        })
        .collect();
    let u_mc: Vec<f64> = test_idx.iter().map(|&i| mc_node_var[i]).collect();

    // --- Accuracy-coverage: abstain on most-uncertain by SDP vs random ---
    let mut rand_state = 0xACE1u64;
    let mut rand_u: Vec<f64> = (0..test_idx.len())
        .map(|_| {
            rand_state ^= rand_state << 13;
            rand_state ^= rand_state >> 7;
            rand_state ^= rand_state << 17;
            (rand_state >> 11) as f64
        })
        .collect();
    // normalize random key magnitude irrelevant; used only for ordering
    rand_u.iter_mut().for_each(|v| *v = v.fract());

    println!("accuracy vs coverage (abstain on most-uncertain):");
    println!(
        "  {:>9}  {:>10}  {:>10}",
        "coverage", "sdp", "random"
    );
    for &cov in &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5] {
        let a_sdp =
            accuracy_at_coverage(&logits_v, &g.labels, &test_idx, &u_sdp, g.n_classes, cov);
        let a_rand =
            accuracy_at_coverage(&logits_v, &g.labels, &test_idx, &rand_u, g.n_classes, cov);
        println!("  {cov:>9.2}  {a_sdp:>10.4}  {a_rand:>10.4}");
    }

    // --- Misclassification detection: AUROC of uncertainty vs error ---
    let errors: Vec<bool> = test_idx
        .iter()
        .map(|&i| !argmax_correct(&logits_v, &g.labels, i, g.n_classes))
        .collect();
    println!("\nmisclassification detection (AUROC of uncertainty vs error):");
    println!("  SDP  AUROC = {:.4}  (one pass)", auroc(&u_sdp, &errors));
    println!(
        "  MC   AUROC = {:.4}  ({MC_SAMPLES} samples)",
        auroc(&u_mc, &errors)
    );
    println!("  (0.5 = uninformative, 1.0 = flags every wrong prediction)");

    let rho = spearman(&u_sdp, &u_mc);
    println!("\nSDP vs MC per-node uncertainty (test nodes): Spearman rho = {rho:.4}");
}

fn main() -> ExitCode {
    let arg = std::env::args().nth(1);
    let dir: PathBuf = match arg {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../propago/data/cora"),
    };
    if !dir.join("cora.content").exists() {
        eprintln!(
            "cora data not found at {}\nfetch it via propago: (cd ../propago && ./scripts/fetch_cora.sh)",
            dir.display()
        );
        return ExitCode::SUCCESS;
    }
    run::<Autodiff<NdArray<f32>>>(Default::default(), &dir, "cora");
    ExitCode::SUCCESS
}
