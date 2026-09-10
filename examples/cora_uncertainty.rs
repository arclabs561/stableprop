//! Real-data evaluation of propagated uncertainty on Cora nodes.
//!
//! Trains a 2-layer GCN on the Cora citation graph (Kipf & Welling 2017), then
//! propagates Gaussian input-feature noise through the trained net to obtain a
//! per-node variance in one pass. It compares accuracy after uncertainty-based
//! abstention with its exact random-ordering expectation and compares analytic
//! with Monte-Carlo uncertainty rankings. The diagonal path drops feature
//! covariance and cross-node covariance introduced by graph aggregation, so the
//! second GCN layer is approximate even when the input features are independent.
//!
//! Provide a directory containing raw `cora.content` and `cora.cites` files. Run:
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
use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_bayes, propagate_matmul_left, propagate_relu, Moments,
};

const HIDDEN: usize = 16;
const INPUT_STD: f64 = 0.1;
const MC_SAMPLES: usize = 200;
const FISHER_PRIOR_PREC: f64 = 1e-2;

struct Graph {
    n: usize,
    n_features: usize,
    n_classes: usize,
    features: Vec<f32>,
    labels: Vec<i32>,
    adj_norm: Vec<f32>,
}

/// Load tab-separated raw Cora content and citation files.
fn load_planetoid(dir: &Path, name: &str) -> std::io::Result<Graph> {
    let content = std::fs::read_to_string(dir.join(format!("{name}.content")))?;
    let cites = std::fs::read_to_string(dir.join(format!("{name}.cites")))?;
    let invalid = |message: String| std::io::Error::new(std::io::ErrorKind::InvalidData, message);

    let n_features = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').count().saturating_sub(2))
        .ok_or_else(|| invalid(format!("{} has no nonempty rows", dir.display())))?;
    if n_features == 0 {
        return Err(invalid(
            "Cora content rows need an id, at least one feature, and a label".into(),
        ));
    }

    let mut label_names: Vec<&str> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.rsplit('\t')
                .next()
                .ok_or_else(|| invalid("Cora content row has no label".into()))
        })
        .collect::<Result<_, _>>()?;
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
        if cols.len() != n_features + 2 || cols[0].is_empty() {
            return Err(invalid(format!("malformed Cora content row: {line}")));
        }
        let idx = id_to_idx.len();
        if id_to_idx.insert(cols[0].to_string(), idx).is_some() {
            return Err(invalid(format!("duplicate Cora node id: {}", cols[0])));
        }
        for f in &cols[1..=n_features] {
            let value = f
                .parse::<f32>()
                .map_err(|_| invalid(format!("non-numeric Cora feature: {f}")))?;
            if !value.is_finite() {
                return Err(invalid(format!("non-finite Cora feature: {f}")));
            }
            features.push(value);
        }
        labels.push(
            *class_id
                .get(cols[n_features + 1])
                .ok_or_else(|| invalid("unknown Cora label".into()))?,
        );
    }
    let n = labels.len();

    let mut adj = vec![0.0f32; n * n];
    for i in 0..n {
        adj[i * n + i] = 1.0;
    }
    for line in cites.lines().filter(|l| !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let (a, b) = match (it.next(), it.next(), it.next()) {
            (Some(a), Some(b), None) => (a, b),
            _ => return Err(invalid(format!("malformed Cora citation row: {line}"))),
        };
        let i = id_to_idx
            .get(a)
            .ok_or_else(|| invalid(format!("unknown Cora citation node id: {a}")))?;
        let j = id_to_idx
            .get(b)
            .ok_or_else(|| invalid(format!("unknown Cora citation node id: {b}")))?;
        adj[i * n + j] = 1.0;
        adj[j * n + i] = 1.0;
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

/// A GCN layer is `adj @ (x W + b)`. This example trains Burn linear layers;
/// `gcn_uncertainty.rs` exercises ricci's `GCNConv`.
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

    /// Per-node centered-logit disagreement from input-feature noise (one pass).
    ///
    /// The final weight is right-multiplied by `P = I - 11^T / C`, so summing
    /// the propagated output variances is exactly `trace(P Cov(logits) P)` for
    /// the represented diagonal hidden moments. This ignores the same feature
    /// and cross-node covariance as the ordinary diagonal path, but does not
    /// count a random offset shared by every class as classification uncertainty.
    fn sdp_centered_logit_variance(
        &self,
        x: Tensor<B, 2>,
        adj: Tensor<B, 2>,
        input_std: f64,
    ) -> Vec<f64> {
        let [n, d] = x.dims();
        let var0 = Tensor::<B, 2>::full([n, d], input_std * input_std, &x.device());
        let m0 = Moments::new(x, var0);
        let m1 = propagate_relu(&Self::sdp_gcn(&m0, &self.lin1, adj.clone()));
        let row_trace = centered_linear_variance(&m1, self.lin2.weight.val(), None);
        let node_trace = (adj.clone() * adj).matmul(row_trace);
        let v = node_trace.to_data().to_vec::<f32>().unwrap();
        (0..n).map(|i| v[i] as f64).collect()
    }
}

/// Centered-logit variance after an affine head, one scalar trace per row.
///
/// With `P = I - 11^T / C`, this returns `trace(P Cov(Y) P)` for diagonal
/// input moments. `weight_var` represents independent weight elements: its
/// contribution remains diagonal before projection, so it contributes
/// `(1 - 1/C) * sum(q)` rather than treating the centered weights as independent.
fn centered_linear_variance<B: Backend>(
    m: &Moments<B>,
    weight_mean: Tensor<B, 2>,
    weight_var: Option<Tensor<B, 2>>,
) -> Tensor<B, 2> {
    let c = weight_mean.dims()[1];
    let centered_weight_mean = weight_mean.clone() - weight_mean.mean_dim(1);
    let deterministic_trace = propagate_linear(m, centered_weight_mean, None)
        .var
        .sum_dim(1);
    match weight_var {
        None => deterministic_trace,
        Some(weight_var) => {
            let weight_noise = (m.mean.clone() * m.mean.clone() + m.var.clone()).matmul(weight_var);
            deterministic_trace + weight_noise.sum_dim(1).mul_scalar(1.0 - 1.0 / c as f64)
        }
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

/// Map labels for a classifier trained without `held_out`. Held-out labels are
/// assigned a harmless placeholder because they are never selected for training
/// or Fisher estimation.
fn remap_known_labels(labels: &[i32], held_out: i32, n_classes: usize) -> Vec<i32> {
    assert!(
        n_classes > 1,
        "novel-class scoring needs at least two classes"
    );
    assert!(
        (0..n_classes as i32).contains(&held_out),
        "held-out class must be in the dataset"
    );
    labels
        .iter()
        .map(|&label| {
            assert!(
                (0..n_classes as i32).contains(&label),
                "dataset label must be in the declared class range"
            );
            if label == held_out {
                0
            } else if label < held_out {
                label
            } else {
                label - 1
            }
        })
        .collect()
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
        let mut start = 0;
        while start < idx.len() {
            let mut end = start;
            while end + 1 < idx.len() && v[idx[end + 1]] == v[idx[start]] {
                end += 1;
            }
            // Zero-based average rank for tied values.
            let average = (start + end) as f64 / 2.0;
            for &i in &idx[start..=end] {
                r[i] = average;
            }
            start = end + 1;
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

/// Apply the class-centering projection `P = I - 11^T / C` to each logit row.
fn center_logits(logits: &mut [f64], c: usize) {
    assert!(c > 0, "a logit row needs at least one class");
    assert_eq!(logits.len() % c, 0, "logits must contain whole class rows");
    for row in logits.chunks_exact_mut(c) {
        let mean = row.iter().sum::<f64>() / c as f64;
        for value in row {
            *value -= mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        center_logits, centered_linear_variance, load_planetoid, remap_known_labels, spearman,
        Moments,
    };
    use burn::tensor::{backend::Backend, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn spearman_uses_average_ranks_for_ties() {
        let a = [1.0, 1.0, 2.0, 3.0];
        let b = [1.0, 1.0, 3.0, 2.0];
        assert!((spearman(&a, &b) - 7.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn known_class_labels_close_the_held_out_gap() {
        assert_eq!(remap_known_labels(&[0, 1, 2, 3], 1, 4), vec![0, 0, 1, 2]);
    }

    #[test]
    fn rejects_citation_endpoint_missing_from_content() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("stableprop-cora-{nonce}"));
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("cora.content"), "a\t1\tA\nb\t0\tB\n").unwrap();
        std::fs::write(dir.join("cora.cites"), "a\tmissing\n").unwrap();

        let result = load_planetoid(&dir, "cora");
        std::fs::remove_dir_all(&dir).unwrap();
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("citation with an unknown endpoint was accepted"),
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error
            .to_string()
            .contains("unknown Cora citation node id: missing"));
    }

    #[test]
    fn centering_removes_a_per_row_common_logit_shift() {
        let mut logits = vec![1.0, -2.0, 4.0, 0.5, 3.0, -1.0];
        let mut shifted = vec![8.0, 5.0, 11.0, -4.5, -2.0, -6.0];
        center_logits(&mut logits, 3);
        center_logits(&mut shifted, 3);
        assert!(logits
            .iter()
            .zip(shifted)
            .all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn centered_final_head_matches_projected_covariance_with_weight_noise() {
        // For diagonal hidden covariance D and independent output weights,
        // Cov(Y) = W^T D W + diag(q). Centering W gives the first projected
        // trace term, while the diagonal q term contributes (1 - 1/C) sum(q).
        let hidden_var = [2.0, 3.0];
        let hidden_mean = [1.5, -2.0];
        let w = [[1.0, 2.0, 5.0], [-1.0, 4.0, 0.0]];
        let w_var = [[0.5, 0.25, 1.0], [0.75, 0.125, 0.5]];
        let c = 3.0;
        let deterministic_trace: f64 = w
            .iter()
            .zip(hidden_var)
            .map(|(row, variance)| {
                let mean = row.iter().sum::<f64>() / c;
                variance
                    * row
                        .iter()
                        .map(|weight| (weight - mean).powi(2))
                        .sum::<f64>()
            })
            .sum();
        let q: [f64; 3] = std::array::from_fn(|class| {
            hidden_mean
                .iter()
                .zip(hidden_var)
                .zip(w_var.iter())
                .map(|((&mean, variance), row)| (mean * mean + variance) * row[class])
                .sum()
        });
        let covariance: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let deterministic = w
                    .iter()
                    .zip(hidden_var)
                    .map(|(row, variance)| variance * row[i] * row[j])
                    .sum::<f64>();
                deterministic + if i == j { q[i] } else { 0.0 }
            })
        });
        let trace_p_cov_p = (0..3).map(|i| covariance[i][i]).sum::<f64>()
            - covariance.iter().flatten().sum::<f64>() / c;
        let trace_p_deterministic_cov_p = deterministic_trace;

        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();
        let m = Moments::new(
            Tensor::<B, 2>::from_data(
                TensorData::new(hidden_mean.map(|x| x as f32).to_vec(), [1, 2]),
                &device,
            ),
            Tensor::<B, 2>::from_data(
                TensorData::new(hidden_var.map(|x| x as f32).to_vec(), [1, 2]),
                &device,
            ),
        );
        let weight_mean = Tensor::<B, 2>::from_data(
            TensorData::new(w.into_iter().flatten().map(|x| x as f32).collect(), [2, 3]),
            &device,
        );
        let weight_var = Tensor::<B, 2>::from_data(
            TensorData::new(
                w_var.into_iter().flatten().map(|x| x as f32).collect(),
                [2, 3],
            ),
            &device,
        );
        let no_weight_noise = centered_linear_variance(&m, weight_mean.clone(), None)
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0] as f64;
        let with_weight_noise = centered_linear_variance(&m, weight_mean, Some(weight_var))
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0] as f64;
        assert!((no_weight_noise - trace_p_deterministic_cov_p).abs() < 1e-5);
        assert!((with_weight_noise - trace_p_cov_p).abs() < 1e-5);
    }
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

/// Centered-logit diagonal empirical-Fisher weight-uncertainty proxy: invert
/// the per-weight Fisher diagonal plus a chosen prior precision, then propagate
/// it with zero input noise. This is neither a calibrated posterior nor a full
/// Laplace approximation; it omits bias uncertainty, parameter correlations,
/// shared-weight cross-node covariance. Its scale is not calibrated.
fn epistemic_centered_logit_variance<B: AutodiffBackend>(
    model: &Gcn<B>,
    x: &Tensor<B, 2>,
    adj: &Tensor<B, 2>,
    targets: &Tensor<B, 1, Int>,
    train_idx: &[usize],
    device: &B::Device,
    prior_prec: f64,
) -> Vec<f64> {
    // Empirical Fisher diagonal for the two weight matrices.
    let mut f1: Option<Tensor<B::InnerBackend, 2>> = None;
    let mut f2: Option<Tensor<B::InnerBackend, 2>> = None;
    for &node in train_idx {
        let logits = model.forward(x.clone(), adj.clone());
        let sel = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![node as i32], [1]), device);
        let nl = logits.select(0, sel.clone());
        let nt = targets.clone().select(0, sel);
        let loss = CrossEntropyLoss::new(None, device).forward(nl, nt);
        let grads = loss.backward();
        let g1 = model.lin1.weight.val().grad(&grads).unwrap();
        let g2 = model.lin2.weight.val().grad(&grads).unwrap();
        f1 = Some(match f1 {
            None => g1.clone() * g1,
            Some(a) => a + g1.clone() * g1,
        });
        f2 = Some(match f2 {
            None => g2.clone() * g2,
            Some(a) => a + g2.clone() * g2,
        });
    }
    // Mean-field variance proxy = 1 / (chosen prior precision + Fisher).
    let wvar1 = f1.unwrap().add_scalar(prior_prec).recip();
    let wvar2 = f2.unwrap().add_scalar(prior_prec).recip();
    let wmean1 = model.lin1.weight.val().inner();
    let wmean2 = model.lin2.weight.val().inner();
    let zeros_like = |t: &Tensor<B::InnerBackend, 1>| t.clone().zeros_like();
    let bias1 = model
        .lin1
        .bias
        .as_ref()
        .map(|b| (b.val().inner(), zeros_like(&b.val().inner())));
    // Propagate the weight-uncertainty proxy with zero input noise.
    let xi = x.clone().inner();
    let adji = adj.clone().inner();
    let [n, _] = xi.dims();
    let var0 = xi.clone().zeros_like();
    let m0 = Moments::new(xi, var0);
    let m1 = propagate_relu(&propagate_matmul_left(
        adji.clone(),
        &propagate_linear_bayes(&m0, wmean1, wvar1, bias1),
    ));
    let row_trace = centered_linear_variance(&m1, wmean2, Some(wvar2));
    let node_trace = (adji.clone() * adji).matmul(row_trace);
    let v = node_trace.to_data().to_vec::<f32>().unwrap();
    (0..n).map(|i| v[i] as f64).collect()
}

fn run<B: AutodiffBackend>(device: B::Device, dir: &Path, name: &str) -> std::io::Result<()> {
    <B as Backend>::seed(&device, 0xC0A0_0001);
    let g = load_planetoid(dir, name)?;
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

    // --- SDP per-node centered-logit disagreement (one analytic pass) ---
    let node_var = model.sdp_centered_logit_variance(x.clone(), adj.clone(), INPUT_STD);
    let u_sdp: Vec<f64> = test_idx.iter().map(|&i| node_var[i]).collect();

    // --- Monte Carlo reference under the same input-noise model ---
    let len = g.n * g.n_classes;
    let mut acc_mean = vec![0.0f64; len];
    let mut acc_sq = vec![0.0f64; len];
    for _ in 0..MC_SAMPLES {
        let noise = Tensor::<B, 2>::random(
            [g.n, g.n_features],
            burn::tensor::Distribution::Normal(0.0, INPUT_STD),
            &device,
        );
        let mut yk: Vec<f64> = model
            .forward(x.clone() + noise, adj.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();
        center_logits(&mut yk, g.n_classes);
        for i in 0..len {
            acc_mean[i] += yk[i];
            acc_sq[i] += yk[i].powi(2);
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

    // A uniform random retained set has expected accuracy equal to the full-set
    // accuracy at every retained count, avoiding a noisy single permutation.
    println!("accuracy vs coverage (abstain on largest centered-logit variance):");
    println!("  {:>9}  {:>10}  {:>10}", "coverage", "sdp", "random E");
    for &cov in &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5] {
        let a_sdp = accuracy_at_coverage(&logits_v, &g.labels, &test_idx, &u_sdp, g.n_classes, cov);
        println!("  {cov:>9.2}  {a_sdp:>10.4}  {base_acc:>10.4}");
    }

    // --- Misclassification detection: AUROC of uncertainty vs error ---
    let errors: Vec<bool> = test_idx
        .iter()
        .map(|&i| !argmax_correct(&logits_v, &g.labels, i, g.n_classes))
        .collect();
    // Centered-logit weight-uncertainty signal via the diagonal empirical-Fisher proxy.
    let node_epi = epistemic_centered_logit_variance(
        &model,
        &x,
        &adj,
        &targets,
        &train_idx,
        &device,
        FISHER_PRIOR_PREC,
    );
    let u_epi: Vec<f64> = test_idx.iter().map(|&i| node_epi[i]).collect();

    println!("\nmisclassification detection (AUROC of score vs error):");
    println!(
        "  centered-logit input-noise AUROC = {:.4}",
        auroc(&u_sdp, &errors)
    );
    println!(
        "  centered-logit empirical-Fisher proxy AUROC = {:.4}",
        auroc(&u_epi, &errors)
    );
    println!(
        "  MC centered-logit input-noise AUROC = {:.4}  ({MC_SAMPLES} samples)",
        auroc(&u_mc, &errors)
    );
    println!("  (0.5 = uninformative, 1.0 = flags every wrong prediction)");

    // Accuracy-coverage for the centered-logit empirical-Fisher proxy too.
    println!(
        "\ncentered-logit empirical-Fisher proxy accuracy vs coverage (abstain on most-uncertain):"
    );
    for &cov in &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5] {
        let a = accuracy_at_coverage(&logits_v, &g.labels, &test_idx, &u_epi, g.n_classes, cov);
        println!("  {cov:>9.2}  {a:>10.4}");
    }

    let rho = spearman(&u_sdp, &u_mc);
    println!("\nSDP vs MC centered-logit disagreement (test nodes): Spearman rho = {rho:.4}");
    Ok(())
}

/// Transductive novel-class scoring: train without labels from `held_out`, then
/// score its nodes against in-distribution nodes. The full graph, including
/// held-out node features and edges, remains visible to message passing, so this
/// is not an inductive OOD evaluation. Compares input-noise, the empirical-Fisher
/// weight-uncertainty proxy, and max-softmax probability (MSP).
fn ood_eval<B: AutodiffBackend>(
    device: B::Device,
    dir: &Path,
    name: &str,
    held_out: i32,
) -> std::io::Result<()> {
    <B as Backend>::seed(&device, 0xC0A0_0002);
    let g = load_planetoid(dir, name)?;
    if g.n_classes < 2 || !(0..g.n_classes as i32).contains(&held_out) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "held-out class must be one of at least two dataset classes",
        ));
    }
    let known_classes = g.n_classes - 1;
    let known_labels = remap_known_labels(&g.labels, held_out, g.n_classes);
    let mut rng = 0x51ce_2026_0bad_f00du64;
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); g.n_classes];
    for (i, &c) in g.labels.iter().enumerate() {
        by_class[c as usize].push(i);
    }
    let mut train_idx = Vec::new();
    for (c, bucket) in by_class.iter_mut().enumerate() {
        if c as i32 == held_out {
            continue;
        }
        shuffle(bucket, &mut rng);
        train_idx.extend(bucket.iter().take(20).copied());
    }
    let train_set: std::collections::HashSet<usize> = train_idx.iter().copied().collect();
    let mut id_test: Vec<usize> = (0..g.n)
        .filter(|&i| g.labels[i] != held_out && !train_set.contains(&i))
        .collect();
    shuffle(&mut id_test, &mut rng);
    id_test.truncate(1000);
    let ood: Vec<usize> = (0..g.n).filter(|&i| g.labels[i] == held_out).collect();
    println!(
        "\n=== transductive novel-class scoring: class {held_out} unseen in a {known_classes}-class head ===\nID train: {}  ID test: {}  novel-class nodes: {}",
        train_idx.len(),
        id_test.len(),
        ood.len()
    );

    let x = Tensor::<B, 2>::from_data(
        TensorData::new(g.features.clone(), [g.n, g.n_features]),
        &device,
    );
    let adj = Tensor::<B, 2>::from_data(TensorData::new(g.adj_norm.clone(), [g.n, g.n]), &device);
    let targets = Tensor::<B, 1, Int>::from_data(TensorData::new(known_labels, [g.n]), &device);
    let train_sel = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            train_idx.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            [train_idx.len()],
        ),
        &device,
    );

    let mut model = Gcn::<B>::init(g.n_features, known_classes, &device);
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(5e-4)))
        .init();
    for _ in 1..=200 {
        let logits = model.forward(x.clone(), adj.clone());
        let tl = logits.select(0, train_sel.clone());
        let tt = targets.clone().select(0, train_sel.clone());
        let loss = CrossEntropyLoss::new(None, &device).forward(tl, tt);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(0.01, model, grads);
    }

    let logits_v = model
        .forward(x.clone(), adj.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let node_var = model.sdp_centered_logit_variance(x.clone(), adj.clone(), INPUT_STD);
    let node_epi = epistemic_centered_logit_variance(
        &model,
        &x,
        &adj,
        &targets,
        &train_idx,
        &device,
        FISHER_PRIOR_PREC,
    );
    // MSP novel-class score = 1 - max softmax probability (higher = more novel).
    let c = known_classes;
    let msp: Vec<f64> = (0..g.n)
        .map(|i| {
            let row = &logits_v[i * c..(i + 1) * c];
            let m = row.iter().cloned().fold(f32::MIN, f32::max);
            let denom: f32 = row.iter().map(|v| (v - m).exp()).sum();
            let maxp = 1.0 / denom; // exp(max-m)/sum = 1/denom
            1.0 - maxp as f64
        })
        .collect();

    let eval: Vec<usize> = id_test.iter().chain(ood.iter()).copied().collect();
    let labels: Vec<bool> = eval.iter().map(|&i| g.labels[i] == held_out).collect();
    let pick = |src: &[f64]| -> Vec<f64> { eval.iter().map(|&i| src[i]).collect() };
    println!("transductive novel-class AUROC (1.0 = score perfectly separates classes):");
    println!(
        "  centered-logit input-noise = {:.4}",
        auroc(&pick(&node_var), &labels)
    );
    println!(
        "  centered-logit empirical-Fisher proxy = {:.4}",
        auroc(&pick(&node_epi), &labels)
    );
    println!(
        "  max-softmax-prob (baseline) = {:.4}",
        auroc(&pick(&msp), &labels)
    );
    Ok(())
}

fn main() -> ExitCode {
    let (dir, explicit_dir): (PathBuf, bool) = match std::env::args().nth(1) {
        Some(p) => (PathBuf::from(p), true),
        None => match std::env::var_os("STABLEPROP_CORA_DIR") {
            Some(p) => (PathBuf::from(p), true),
            None => (
                Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cora"),
                false,
            ),
        },
    };
    if !dir.join("cora.content").exists() || !dir.join("cora.cites").exists() {
        eprintln!(
            "Cora data not found at {}\npass a data directory as the first argument or set STABLEPROP_CORA_DIR; it must contain cora.content and cora.cites.",
            dir.display()
        );
        return if explicit_dir {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        };
    }
    if let Err(err) = run::<Autodiff<NdArray<f32>>>(Default::default(), &dir, "cora") {
        eprintln!("could not run Cora evaluation: {err}");
        return ExitCode::FAILURE;
    }
    if let Err(err) = ood_eval::<Autodiff<NdArray<f32>>>(Default::default(), &dir, "cora", 0) {
        eprintln!("could not run Cora transductive novel-class evaluation: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
