//! Compares diagonal Gaussian propagation with Monte Carlo output variance
//! in a two-layer ricci GCN (`GCNConv -> ReLU -> GCNConv`).
//!
//! The output reports variance correlation and scale agreement. The diagonal
//! path drops feature covariance after each layer and node covariance after
//! adjacency aggregation, so the second layer can disagree with Monte Carlo.
//!
//! Run: `cargo run --release --example gcn_uncertainty --features burn`

use burn::tensor::{Device, Distribution, Tensor, TensorData};
#[cfg(test)]
use burn::{module::Param, nn::Linear};
use ricci::GCNConv;
use stableprop::burn_sdp::{propagate_linear, propagate_matmul_left, propagate_relu, Moments};

/// Pull `(weight, bias)` tensors out of a ricci GCN layer's linear.
fn lin_params(layer: &GCNConv) -> (Tensor<2>, Option<Tensor<1>>) {
    let w = layer.linear().weight.val();
    let b = layer.linear().bias.as_ref().map(|p| p.val());
    (w, b)
}

/// SDP through one GCN layer: transform and aggregate, then add the bias.
/// This matches `GCNConv::forward`: `adj @ (x @ W) + b`.
fn sdp_gcn(m: &Moments, layer: &GCNConv, adj: Tensor<2>) -> Moments {
    let (w, b) = lin_params(layer);
    let mut output = propagate_matmul_left(adj, &propagate_linear(m, w, None));
    if let Some(bias) = b {
        let [features] = bias.dims();
        output.mean = output.mean + bias.reshape([1, features]);
    }
    output
}

fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "Pearson inputs must have matching lengths"
    );
    assert!(a.iter().chain(b).all(|value| value.is_finite()));
    if a.len() < 2 {
        return None;
    }
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    (va > 0.0 && vb > 0.0).then(|| cov / (va.sqrt() * vb.sqrt()))
}

fn pearson_undefined_reason(a: &[f64], b: &[f64]) -> &'static str {
    if a.len() < 2 || b.len() < 2 {
        "fewer than two samples"
    } else {
        "zero spread"
    }
}

fn main() {
    let dev = Device::flex();
    dev.seed(0x6C6E_0001);
    let (n, d_in, d_hid, d_out) = (32usize, 8usize, 8usize, 4usize);
    let input_std = 0.3f64;
    let k = 4000usize;

    // Ring-graph normalized adjacency (each node aggregates itself + 2 neighbors).
    let mut adj_v = vec![0.0f32; n * n];
    for i in 0..n {
        for &j in &[i, (i + n - 1) % n, (i + 1) % n] {
            adj_v[i * n + j] = 1.0 / 3.0;
        }
    }
    let adj = Tensor::<2>::from_data(TensorData::new(adj_v, [n, n]), &dev);

    // Seeded input means and layers make this comparison repeatable.
    let x_mean = Tensor::<2>::random([n, d_in], Distribution::Normal(0.0, 1.0), &dev);
    let layer1 = GCNConv::init(d_in, d_hid, &dev);
    let layer2 = GCNConv::init(d_hid, d_out, &dev);

    // --- SDP: one analytic forward pass over moments ---
    let var0 = Tensor::<2>::full([n, d_in], input_std * input_std, &dev);
    let m0 = Moments::new(x_mean.clone(), var0);
    let m1 = propagate_relu(&sdp_gcn(&m0, &layer1, adj.clone()));
    let m2 = sdp_gcn(&m1, &layer2, adj.clone());
    let sdp_var = m2.var.to_data().try_to_vec::<f32>().unwrap();
    assert!(
        sdp_var
            .iter()
            .all(|variance| variance.is_finite() && *variance >= 0.0),
        "propagated variances must be finite and nonnegative"
    );

    // --- MC: K noisy inputs through the deterministic GCN ---
    let len = n * d_out;
    let mut acc_mean = vec![0.0f64; len];
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        let noise = Tensor::<2>::random([n, d_in], Distribution::Normal(0.0, input_std), &dev);
        let xk = x_mean.clone() + noise;
        let h = layer1.forward(xk, adj.clone()).clamp_min(0.0);
        let yk = layer2.forward(h, adj.clone());
        let v: Vec<f64> = yk
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|x| *x as f64)
            .collect();
        for i in 0..len {
            acc_mean[i] += v[i];
        }
        samples.push(v);
    }
    for m in acc_mean.iter_mut() {
        *m /= k as f64;
    }
    let mut mc_var = vec![0.0f64; len];
    for s in &samples {
        for i in 0..len {
            mc_var[i] += (s[i] - acc_mean[i]).powi(2);
        }
    }
    for v in mc_var.iter_mut() {
        *v /= (k - 1) as f64;
    }
    assert!(mc_var
        .iter()
        .all(|variance| variance.is_finite() && *variance >= 0.0));

    // --- Compare ---
    let sdp_var_f: Vec<f64> = sdp_var.iter().map(|x| *x as f64).collect();
    let r = pearson(&sdp_var_f, &mc_var);
    let ratios: Vec<f64> = sdp_var_f
        .iter()
        .zip(&mc_var)
        .filter(|(_, m)| **m > 1e-9)
        .map(|(s, m)| s / m)
        .collect();
    let mean_ratio = (!ratios.is_empty()).then(|| ratios.iter().sum::<f64>() / ratios.len() as f64);

    println!("2-layer GCN (GCNConv -> ReLU -> GCNConv), n={n} nodes, d_out={d_out}");
    println!("input noise std = {input_std}, MC samples = {k}\n");
    println!("SDP var vs MC var:");
    match r {
        Some(r) => println!("  Pearson r   = {r:.4}   (1.0 = perfect agreement)"),
        None => println!(
            "  Pearson r   = undefined ({})",
            pearson_undefined_reason(&sdp_var_f, &mc_var)
        ),
    }
    match mean_ratio {
        Some(mean_ratio) => println!(
            "  mean ratio  = {mean_ratio:.3}  (SDP / MC; {} of {len} outputs with MC variance > 1e-9)\n",
            ratios.len(),
        ),
        None => println!("  mean ratio  = undefined (no outputs with MC variance > 1e-9; 0 of {len} included)\n"),
    }

    println!("per-output predictive std (sqrt var), first 8 of {len}:");
    println!("  {:>10}  {:>10}  {:>8}", "sdp_std", "mc_std", "ratio");
    for i in 0..len.min(8) {
        let s = sdp_var_f[i].sqrt();
        let m = mc_var[i].sqrt();
        if m > 0.0 {
            println!("  {s:>10.4}  {m:>10.4}  {:>8.3}", s / m);
        } else {
            println!("  {s:>10.4}  {m:>10.4}  undefined (MC std is zero)");
        }
    }

    // --- Thresholding mechanics only: this random, unlabeled graph cannot
    // establish whether deferring improves decisions. ---
    let mut node_std: Vec<(usize, f64)> = (0..n)
        .map(|node| {
            let mean_v =
                (0..d_out).map(|c| sdp_var_f[node * d_out + c]).sum::<f64>() / d_out as f64;
            (node, mean_v.sqrt())
        })
        .collect();
    node_std.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let thresh = node_std.iter().map(|(_, s)| s).sum::<f64>() / n as f64;
    println!(
        "\nrelative-uncertainty thresholding mechanics (mean std {thresh:.4}; no task metric):"
    );
    for (node, s) in &node_std {
        let band = if *s > thresh {
            "above mean"
        } else {
            "at or below mean"
        };
        println!("  node {node}: std={s:.4}  -> {band}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: Vec<f32>, expected: Vec<f32>) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn sdp_gcn_adds_bias_after_non_normalized_aggregation() {
        let device = Device::flex();
        let weight = Tensor::<2>::from_data([[2.0, -1.0], [0.5, 3.0]], &device);
        let bias = Tensor::<1>::from_data([0.75, -1.25], &device);
        let layer = GCNConv::new(Linear {
            weight: Param::from_tensor(weight.clone()),
            bias: Some(Param::from_tensor(bias.clone())),
        });
        let mean = Tensor::<2>::from_data([[1.0, -2.0], [0.5, 3.0]], &device);
        let var = Tensor::<2>::from_data([[0.25, 4.0], [1.0, 0.5]], &device);
        let adj = Tensor::<2>::from_data([[2.0, 1.0], [-1.0, 3.0]], &device);

        let actual = sdp_gcn(
            &Moments::new(mean.clone(), var.clone()),
            &layer,
            adj.clone(),
        );
        close(
            actual.mean.into_data().try_to_vec::<f32>().unwrap(),
            vec![5.25, -6.75, 7.25, 31.25],
        );
        close(
            layer
                .forward(mean, adj)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![5.25, -6.75, 7.25, 31.25],
        );
        close(
            actual.var.into_data().try_to_vec::<f32>().unwrap(),
            vec![12.125, 150.5, 39.125, 85.75],
        );
    }
}
