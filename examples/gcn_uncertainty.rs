//! Analytic uncertainty on a propago GCN vs Monte-Carlo input-noise sampling.
//!
//! This deliberately risks the claim that diagonal Gaussian moment propagation
//! (momentprop) matches the true output uncertainty. We build a 2-layer GCN
//! (`GCNConv -> ReLU -> GCNConv`) from propago, put a Gaussian over the input
//! node features, then compare two ways of getting the output variance:
//!
//! 1. SDP -- propagate (mean, var) analytically through the layers, one pass.
//! 2. MC -- sample K noisy inputs, run the deterministic GCN on each, take the
//!    empirical per-output variance.
//!
//! If diagonal SDP is faithful, the two variances agree (high correlation,
//! ratio ~1). The diagonal assumption drops cross-feature covariance after the
//! ReLU, so the second layer is where any disagreement shows up -- that is the
//! honest part of the test.
//!
//! Run: `cargo run --example gcn_uncertainty --features burn`

use burn::tensor::{backend::Backend, Distribution, Tensor, TensorData};
use burn_ndarray::NdArray;
use momentprop::burn_sdp::{propagate_linear, propagate_matmul_left, propagate_relu, Moments};
use propago::GCNConv;

type B = NdArray<f32>;

/// Pull `(weight, bias)` tensors out of a propago GCN layer's linear.
fn lin_params(layer: &GCNConv<B>) -> (Tensor<B, 2>, Option<Tensor<B, 1>>) {
    let w = layer.linear().weight.val();
    let b = layer.linear().bias.as_ref().map(|p| p.val());
    (w, b)
}

/// SDP through one GCN layer: linear then adjacency aggregation (matches
/// `GCNConv::forward`, which is `adj @ (x @ W + b)`).
fn sdp_gcn(m: &Moments<B>, layer: &GCNConv<B>, adj: Tensor<B, 2>) -> Moments<B> {
    let (w, b) = lin_params(layer);
    let after_linear = propagate_linear(m, w, b);
    propagate_matmul_left(adj, &after_linear)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
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
    cov / (va.sqrt() * vb.sqrt())
}

fn main() {
    let dev = <B as Backend>::Device::default();
    let (n, d_in, d_hid, d_out) = (6usize, 8usize, 8usize, 4usize);
    let input_std = 0.3f64;
    let k = 4000usize;

    // Ring-graph normalized adjacency (each node aggregates itself + 2 neighbors).
    let mut adj_v = vec![0.0f32; n * n];
    for i in 0..n {
        for &j in &[i, (i + n - 1) % n, (i + 1) % n] {
            adj_v[i * n + j] = 1.0 / 3.0;
        }
    }
    let adj = Tensor::<B, 2>::from_data(TensorData::new(adj_v, [n, n]), &dev);

    // Deterministic input means; random-init layers (the SDP-vs-MC agreement is
    // intrinsic to whatever model gets initialized, so a fixed seed is not needed).
    let x_mean = Tensor::<B, 2>::random([n, d_in], Distribution::Normal(0.0, 1.0), &dev);
    let layer1 = GCNConv::<B>::init(d_in, d_hid, &dev);
    let layer2 = GCNConv::<B>::init(d_hid, d_out, &dev);

    // --- SDP: one analytic forward pass over moments ---
    let var0 = Tensor::<B, 2>::full([n, d_in], input_std * input_std, &dev);
    let m0 = Moments::new(x_mean.clone(), var0);
    let m1 = propagate_relu(&sdp_gcn(&m0, &layer1, adj.clone()));
    let m2 = sdp_gcn(&m1, &layer2, adj.clone());
    let sdp_var = m2.var.to_data().to_vec::<f32>().unwrap();

    // --- MC: K noisy inputs through the deterministic GCN ---
    let len = n * d_out;
    let mut acc_mean = vec![0.0f64; len];
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        let noise = Tensor::<B, 2>::random([n, d_in], Distribution::Normal(0.0, input_std), &dev);
        let xk = x_mean.clone() + noise;
        let h = layer1.forward(xk, adj.clone()).clamp_min(0.0);
        let yk = layer2.forward(h, adj.clone());
        let v: Vec<f64> = yk
            .to_data()
            .to_vec::<f32>()
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

    // --- Compare ---
    let sdp_var_f: Vec<f64> = sdp_var.iter().map(|x| *x as f64).collect();
    let r = pearson(&sdp_var_f, &mc_var);
    let ratios: Vec<f64> = sdp_var_f
        .iter()
        .zip(&mc_var)
        .filter(|(_, m)| **m > 1e-9)
        .map(|(s, m)| s / m)
        .collect();
    let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;

    println!("2-layer GCN (GCNConv -> ReLU -> GCNConv), n={n} nodes, d_out={d_out}");
    println!("input noise std = {input_std}, MC samples = {k}\n");
    println!("SDP var vs MC var:");
    println!("  Pearson r   = {r:.4}   (1.0 = perfect agreement)");
    println!("  mean ratio  = {mean_ratio:.3}  (SDP / MC; 1.0 = unbiased)\n");

    println!("per-output predictive std (sqrt var), first 8 of {len}:");
    println!("  {:>10}  {:>10}  {:>8}", "sdp_std", "mc_std", "ratio");
    for i in 0..len.min(8) {
        let s = sdp_var_f[i].sqrt();
        let m = mc_var[i].sqrt();
        println!("  {:>10.4}  {:>10.4}  {:>8.3}", s, m, s / m.max(1e-9));
    }

    // --- Abstention demo: flag the highest-uncertainty nodes by SDP std ---
    let mut node_std: Vec<(usize, f64)> = (0..n)
        .map(|node| {
            let mean_v =
                (0..d_out).map(|c| sdp_var_f[node * d_out + c]).sum::<f64>() / d_out as f64;
            (node, mean_v.sqrt())
        })
        .collect();
    node_std.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let thresh = node_std.iter().map(|(_, s)| s).sum::<f64>() / n as f64;
    println!("\nabstention (SDP node std > mean {thresh:.4} => defer):");
    for (node, s) in &node_std {
        let act = if *s > thresh { "ABSTAIN" } else { "predict" };
        println!("  node {node}: std={s:.4}  -> {act}");
    }
}
