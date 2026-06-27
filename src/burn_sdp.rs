//! Distribution propagation on Burn tensors (diagonal-Gaussian / assumed-density
//! filtering).
//!
//! Tracks a per-feature mean and variance for a batch of independent Gaussians
//! and pushes them through linear, fixed-matmul (e.g. a GCN adjacency), and ReLU
//! layers. Linear and matmul propagate variance exactly under the diagonal
//! assumption; ReLU uses Frey & Hinton (1999) moment matching. All ops are Burn
//! tensor ops, so the propagation is differentiable and runs on any backend.
//!
//! Covariance is approximated as diagonal: cross-feature correlations introduced
//! by a layer are dropped before the next layer. This is the cheap inference-time
//! variant; the off-diagonal terms are what the full "stable distribution
//! propagation" of Petersen et al. (ICLR 2024) keeps.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use core::f64::consts::{FRAC_1_SQRT_2, PI};

/// Mean and per-feature variance of a batch of independent Gaussians.
///
/// Both tensors are shape `[n, d]` (n rows, d features). Variance is the
/// diagonal of the covariance; off-diagonal terms are not tracked.
#[derive(Clone, Debug)]
pub struct Moments<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub var: Tensor<B, 2>,
}

impl<B: Backend> Moments<B> {
    pub fn new(mean: Tensor<B, 2>, var: Tensor<B, 2>) -> Self {
        Self { mean, var }
    }
}

/// Propagate through an affine map `y = x @ weight + bias`.
///
/// `weight` is `[d_in, d_out]` (Burn's `Linear` layout); `bias` is `[d_out]`.
/// Mean is exact; variance is `var @ weight^2`, which is the exact marginal
/// variance of each output when the input covariance is diagonal.
pub fn propagate_linear<B: Backend>(
    m: &Moments<B>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Moments<B> {
    let mut mean = m.mean.clone().matmul(weight.clone());
    if let Some(b) = bias {
        let d = b.dims()[0];
        mean = mean + b.reshape([1, d]);
    }
    let w2 = weight.clone() * weight;
    let var = m.var.clone().matmul(w2);
    Moments { mean, var }
}

/// Propagate through `y = x @ W + b` where BOTH inputs and weights are uncertain
/// (mean-field: all elements independent). This is the linear step of
/// Probabilistic Backpropagation (Hernandez-Lobato 2015) / Deterministic
/// Variational Inference (Wu 2019): the weight variance `w_var` is what turns
/// input sensitivity into *epistemic* uncertainty. Reduces to
/// [`propagate_linear`] when `w_var` is zero.
///
/// `var_out = mean_x^2 @ w_var  +  var_x @ mean_W^2  +  var_x @ w_var  +  b_var`
/// (the first term is the new epistemic contribution, the second is the existing
/// input-variance propagation, the third is the cross term).
pub fn propagate_linear_bayes<B: Backend>(
    m: &Moments<B>,
    w_mean: Tensor<B, 2>,
    w_var: Tensor<B, 2>,
    bias: Option<(Tensor<B, 1>, Tensor<B, 1>)>,
) -> Moments<B> {
    let mut mean = m.mean.clone().matmul(w_mean.clone());
    let wm2 = w_mean.clone() * w_mean;
    let mx2 = m.mean.clone() * m.mean.clone();
    let mut var =
        mx2.matmul(w_var.clone()) + m.var.clone().matmul(wm2) + m.var.clone().matmul(w_var);
    if let Some((bm, bv)) = bias {
        let d = bm.dims()[0];
        mean = mean + bm.reshape([1, d]);
        var = var + bv.reshape([1, d]);
    }
    Moments { mean, var }
}

/// Propagate through left multiplication by a fixed matrix `y = a @ x`
/// (e.g. a GCN message-passing step `A_hat @ H`).
///
/// For independent rows with diagonal variance, the output variance is
/// `(a ∘ a) @ var` (cross-row correlations are dropped, matching the diagonal
/// assumption).
pub fn propagate_matmul_left<B: Backend>(a: Tensor<B, 2>, m: &Moments<B>) -> Moments<B> {
    let mean = a.clone().matmul(m.mean.clone());
    let a2 = a.clone() * a;
    let var = a2.matmul(m.var.clone());
    Moments { mean, var }
}

/// Propagate through an element-wise ReLU via Frey & Hinton (1999) moment
/// matching.
///
/// Per element with mean `mu`, std `sigma`, `alpha = mu / sigma`:
/// `mu'    = mu * Phi(alpha) + sigma * phi(alpha)`
/// `var'   = (mu^2 + var) * Phi(alpha) + mu * sigma * phi(alpha) - mu'^2`
/// where `Phi` / `phi` are the standard normal CDF / PDF.
pub fn propagate_relu<B: Backend>(m: &Moments<B>) -> Moments<B> {
    let eps = 1e-12;
    let var = m.var.clone().clamp_min(eps);
    let sigma = var.clone().sqrt();
    let mu = m.mean.clone();
    let alpha = mu.clone() / sigma.clone();

    // Phi(alpha) = 0.5 * (1 + erf(alpha / sqrt(2)))
    let big_phi = alpha
        .clone()
        .mul_scalar(FRAC_1_SQRT_2)
        .erf()
        .add_scalar(1.0)
        .mul_scalar(0.5);
    // phi(alpha) = exp(-alpha^2 / 2) / sqrt(2 pi)
    let phi = (alpha.clone() * alpha.clone())
        .mul_scalar(-0.5)
        .exp()
        .mul_scalar(1.0 / (2.0 * PI).sqrt());

    let mu_out = mu.clone() * big_phi.clone() + sigma.clone() * phi.clone();
    let var_out = (mu.clone() * mu.clone() + var) * big_phi + mu * sigma * phi
        - mu_out.clone() * mu_out.clone();

    Moments {
        mean: mu_out,
        var: var_out.clamp_min(0.0),
    }
}

/// `d x d` identity on the given backend/device.
fn eye<B: Backend>(d: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut v = vec![0.0f32; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    Tensor::<B, 2>::from_data(TensorData::new(v, [d, d]), device)
}

/// Mean and FULL covariance of a batch of `n` independent Gaussians.
///
/// `mean` is `[n, d]`, `cov` is `[n, d, d]`. Unlike [`Moments`], this keeps the
/// cross-feature correlations a layer introduces, which is the accuracy diagonal
/// propagation drops (Petersen et al., ICLR 2024). Cost is `O(n d^2)` memory and
/// `O(n d^3)` per linear layer, so it suits small-to-medium feature dimensions.
#[derive(Clone, Debug)]
pub struct MomentsFull<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub cov: Tensor<B, 3>,
}

impl<B: Backend> MomentsFull<B> {
    pub fn new(mean: Tensor<B, 2>, cov: Tensor<B, 3>) -> Self {
        Self { mean, cov }
    }

    /// Build from a diagonal variance `[n, d]` (independent input features):
    /// `cov = diag(var)` per row.
    pub fn from_diagonal(mean: Tensor<B, 2>, var: Tensor<B, 2>) -> Self {
        let [n, d] = var.dims();
        let eye_d = eye::<B>(d, &var.device());
        let cov = var.unsqueeze_dim::<3>(2).expand([n, d, d]) * eye_d.unsqueeze::<3>();
        Self { mean, cov }
    }

    /// Per-feature variance (the diagonal of the covariance), shape `[n, d]`.
    pub fn variance(&self) -> Tensor<B, 2> {
        let [n, d, _] = self.cov.dims();
        let eye_d = eye::<B>(d, &self.cov.device());
        (self.cov.clone() * eye_d.unsqueeze::<3>())
            .sum_dim(2)
            .reshape([n, d])
    }
}

/// Full-covariance affine map `y = x W + b`: `Sigma_out = W^T Sigma_in W` (exact).
pub fn propagate_linear_full<B: Backend>(
    m: &MomentsFull<B>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> MomentsFull<B> {
    let [n, _] = m.mean.dims();
    let [d_in, d_out] = weight.dims();
    let mut mean = m.mean.clone().matmul(weight.clone());
    if let Some(b) = bias {
        mean = mean + b.reshape([1, d_out]);
    }
    let w3 = weight.clone().unsqueeze::<3>().expand([n, d_in, d_out]);
    let wt3 = weight
        .swap_dims(0, 1)
        .unsqueeze::<3>()
        .expand([n, d_out, d_in]);
    let cov = wt3.matmul(m.cov.clone().matmul(w3));
    MomentsFull { mean, cov }
}

/// Full-covariance ReLU: exact Frey-Hinton moments on the diagonal, smooth-gated
/// (`g_i = Phi(alpha_i)`) cross-terms off-diagonal.
///
/// The smooth gate `Phi(alpha)` is the expected ReLU derivative, which fixes the
/// decision-boundary brittleness of the hard 0/1 Jacobian gate that the local-
/// linearization method of Petersen et al. (2024) lists as a limitation.
pub fn propagate_relu_full<B: Backend>(m: &MomentsFull<B>) -> MomentsFull<B> {
    let [n, d, _] = m.cov.dims();
    let dev = m.cov.device();
    let eye_d = eye::<B>(d, &dev);

    let var = (m.cov.clone() * eye_d.clone().unsqueeze::<3>())
        .sum_dim(2)
        .reshape([n, d])
        .clamp_min(1e-12);
    let sigma = var.clone().sqrt();
    let mu = m.mean.clone();
    let alpha = mu.clone() / sigma.clone();
    let big_phi = alpha
        .clone()
        .mul_scalar(FRAC_1_SQRT_2)
        .erf()
        .add_scalar(1.0)
        .mul_scalar(0.5);
    let phi = (alpha.clone() * alpha.clone())
        .mul_scalar(-0.5)
        .exp()
        .mul_scalar(1.0 / (2.0 * PI).sqrt());
    let mu_out = mu.clone() * big_phi.clone() + sigma.clone() * phi.clone();
    let var_out = ((mu.clone() * mu.clone() + var) * big_phi.clone() + mu * sigma * phi
        - mu_out.clone() * mu_out.clone())
    .clamp_min(0.0);

    // Off-diagonal: g_i g_j Sigma_ij; diagonal then overwritten with var_out.
    let g = big_phi;
    let gg = g.clone().unsqueeze_dim::<3>(2) * g.unsqueeze_dim::<3>(1);
    let off_mask = eye_d
        .clone()
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .unsqueeze::<3>();
    let off = m.cov.clone() * gg * off_mask;
    let diag = var_out.unsqueeze_dim::<3>(2).expand([n, d, d]) * eye_d.unsqueeze::<3>();
    MomentsFull {
        mean: mu_out,
        cov: off + diag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn mc_moments(samples: &[Vec<f64>], len: usize) -> (Vec<f64>, Vec<f64>) {
        let k = samples.len() as f64;
        let mut mean = vec![0.0; len];
        for s in samples {
            for i in 0..len {
                mean[i] += s[i];
            }
        }
        for m in mean.iter_mut() {
            *m /= k;
        }
        let mut var = vec![0.0; len];
        for s in samples {
            for i in 0..len {
                var[i] += (s[i] - mean[i]).powi(2);
            }
        }
        for v in var.iter_mut() {
            *v /= k - 1.0;
        }
        (mean, var)
    }

    /// One linear map of a diagonal Gaussian: the marginal output variance is
    /// exact (`var @ W^2`), so SDP must match Monte Carlo to within sampling
    /// noise. This is the load-bearing exactness claim of `propagate_linear`.
    #[test]
    fn linear_variance_matches_monte_carlo() {
        let dev = <B as Backend>::Device::default();
        let (n, d_in, d_out, k) = (3usize, 5usize, 4usize, 40_000usize);
        let std = 0.25f64;

        let w = Tensor::<B, 2>::random([d_in, d_out], Distribution::Normal(0.0, 1.0), &dev);
        let mean = Tensor::<B, 2>::random([n, d_in], Distribution::Normal(0.0, 1.0), &dev);
        let var = Tensor::<B, 2>::full([n, d_in], std * std, &dev);

        let out = propagate_linear(&Moments::new(mean.clone(), var), w.clone(), None);
        let sdp_var = out.var.to_data().to_vec::<f32>().unwrap();

        let len = n * d_out;
        let mut samples = Vec::with_capacity(k);
        for _ in 0..k {
            let noise = Tensor::<B, 2>::random([n, d_in], Distribution::Normal(0.0, std), &dev);
            let y = (mean.clone() + noise).matmul(w.clone());
            samples.push(
                y.to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| *x as f64)
                    .collect(),
            );
        }
        let (_, mc_var) = mc_moments(&samples, len);
        for i in 0..len {
            let rel = (sdp_var[i] as f64 - mc_var[i]).abs() / mc_var[i].max(1e-9);
            assert!(rel < 0.10, "output {i}: sdp={} mc={mc_var:?}", sdp_var[i]);
        }
    }

    /// A single ReLU on a Gaussian: Frey-Hinton gives the *exact* moments of
    /// `max(0, X)`, so SDP mean and variance must match Monte Carlo tightly.
    #[test]
    fn relu_moments_match_monte_carlo() {
        let dev = <B as Backend>::Device::default();
        let (n, d, k) = (2usize, 3usize, 60_000usize);
        let std = 0.8f64;

        let mean = Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, 0.5), &dev);
        let var = Tensor::<B, 2>::full([n, d], std * std, &dev);
        let out = propagate_relu(&Moments::new(mean.clone(), var));
        let sdp_mean = out.mean.to_data().to_vec::<f32>().unwrap();
        let sdp_var = out.var.to_data().to_vec::<f32>().unwrap();

        let len = n * d;
        let mut samples = Vec::with_capacity(k);
        for _ in 0..k {
            let noise = Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, std), &dev);
            let y = (mean.clone() + noise).clamp_min(0.0);
            samples.push(
                y.to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| *x as f64)
                    .collect(),
            );
        }
        let (mc_mean, mc_var) = mc_moments(&samples, len);
        for i in 0..len {
            assert!(
                (sdp_mean[i] as f64 - mc_mean[i]).abs() < 0.02,
                "mean {i}: sdp={} mc={}",
                sdp_mean[i],
                mc_mean[i]
            );
            let rel = (sdp_var[i] as f64 - mc_var[i]).abs() / mc_var[i].max(1e-9);
            assert!(rel < 0.08, "var {i}: sdp={} mc={}", sdp_var[i], mc_var[i]);
        }
    }

    /// Linear -> ReLU -> Linear: full-covariance output variance must match Monte
    /// Carlo AND be closer than diagonal propagation (which drops the ReLU's
    /// cross-correlations the second linear layer recombines).
    #[test]
    fn full_cov_beats_diagonal_vs_monte_carlo() {
        let dev = <B as Backend>::Device::default();
        let (n, d_in, h, d_out, k) = (4usize, 5usize, 8usize, 3usize, 80_000usize);
        let sig = 0.3f64;

        let w1 = Tensor::<B, 2>::random([d_in, h], Distribution::Normal(0.0, 1.0), &dev);
        let b1 = Tensor::<B, 1>::random([h], Distribution::Normal(0.0, 0.3), &dev);
        let w2 = Tensor::<B, 2>::random([h, d_out], Distribution::Normal(0.0, 1.0), &dev);
        let b2 = Tensor::<B, 1>::random([d_out], Distribution::Normal(0.0, 0.3), &dev);
        let mean = Tensor::<B, 2>::random([n, d_in], Distribution::Normal(0.0, 1.0), &dev);
        let var = Tensor::<B, 2>::full([n, d_in], sig * sig, &dev);

        // Full covariance.
        let m0 = MomentsFull::from_diagonal(mean.clone(), var.clone());
        let m1 = propagate_relu_full(&propagate_linear_full(&m0, w1.clone(), Some(b1.clone())));
        let m2 = propagate_linear_full(&m1, w2.clone(), Some(b2.clone()));
        let f_var = m2.variance().to_data().to_vec::<f32>().unwrap();

        // Diagonal.
        let d0 = Moments::new(mean.clone(), var.clone());
        let d1 = propagate_relu(&propagate_linear(&d0, w1.clone(), Some(b1.clone())));
        let d2 = propagate_linear(&d1, w2.clone(), Some(b2.clone()));
        let d_var = d2.var.to_data().to_vec::<f32>().unwrap();

        // Monte Carlo oracle.
        let len = n * d_out;
        let mut samples = Vec::with_capacity(k);
        for _ in 0..k {
            let noise = Tensor::<B, 2>::random([n, d_in], Distribution::Normal(0.0, sig), &dev);
            let xk = mean.clone() + noise;
            let hk = (xk.matmul(w1.clone()) + b1.clone().reshape([1, h])).clamp_min(0.0);
            let yk = hk.matmul(w2.clone()) + b2.clone().reshape([1, d_out]);
            samples.push(
                yk.to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| *x as f64)
                    .collect(),
            );
        }
        let (_, mc_var) = mc_moments(&samples, len);

        let rel_err = |est: &[f32]| -> f64 {
            (0..len)
                .map(|i| (est[i] as f64 - mc_var[i]).abs() / mc_var[i].max(1e-9))
                .sum::<f64>()
                / len as f64
        };
        let (fe, de) = (rel_err(&f_var), rel_err(&d_var));
        assert!(fe < 0.12, "full-cov mean rel err {fe} vs MC too high");
        assert!(
            fe < de,
            "full-cov ({fe}) should beat diagonal ({de}) against MC"
        );
    }
}
