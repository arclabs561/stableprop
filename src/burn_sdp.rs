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

/// Propagate through leaky ReLU `max(x, alpha * x)` via exact Gaussian moments.
///
/// Uses `leaky(x) = alpha*x + (1-alpha)*relu(x)`, so the moments combine the raw
/// and rectified-Gaussian moments. Reduces to [`propagate_relu`] at `alpha = 0`.
pub fn propagate_leaky_relu<B: Backend>(m: &Moments<B>, alpha: f64) -> Moments<B> {
    let var = m.var.clone().clamp_min(1e-12);
    let sigma = var.clone().sqrt();
    let mu = m.mean.clone();
    let a = mu.clone() / sigma.clone();
    let big_phi = a
        .clone()
        .mul_scalar(FRAC_1_SQRT_2)
        .erf()
        .add_scalar(1.0)
        .mul_scalar(0.5);
    let phi = (a.clone() * a.clone())
        .mul_scalar(-0.5)
        .exp()
        .mul_scalar(1.0 / (2.0 * PI).sqrt());

    let mu2_plus_var = mu.clone() * mu.clone() + var.clone();
    // E[relu] and E[relu^2].
    let e_r = mu.clone() * big_phi.clone() + sigma.clone() * phi.clone();
    let e_r2 = mu2_plus_var.clone() * big_phi + mu * sigma * phi;

    let mean = e_r.mul_scalar(1.0 - alpha) + m.mean.clone().mul_scalar(alpha);
    let e_y2 = mu2_plus_var.mul_scalar(alpha * alpha) + e_r2.mul_scalar(1.0 - alpha * alpha);
    let var_out = (e_y2 - mean.clone() * mean.clone()).clamp_min(0.0);
    Moments { mean, var: var_out }
}

/// Combine a residual skip and a branch `y = skip + branch` under the
/// independence approximation: `mean = skip.mean + branch.mean`,
/// `var = skip.var + branch.var`.
///
/// This IGNORES the skip-branch covariance (the branch is a function of the
/// skip's input, so they are correlated). It is accurate when the branch is
/// small relative to the skip (the usual residual-block regime) and approximate
/// otherwise. A common simplification for residual networks.
pub fn propagate_residual_add<B: Backend>(skip: &Moments<B>, branch: &Moments<B>) -> Moments<B> {
    Moments {
        mean: skip.mean.clone() + branch.mean.clone(),
        var: skip.var.clone() + branch.var.clone(),
    }
}

/// Propagate a diagonal Gaussian through a 2-D convolution. Convolution is a
/// linear map, so under the diagonal-covariance assumption the moments are
/// exact: `mean_out = conv(mean, w) + b`, `var_out = conv(var, w^2)`.
///
/// `mean` / `var` are `[N, C_in, H, W]`, `weight` is `[C_out, C_in, kh, kw]`.
pub fn propagate_conv2d<B: Backend>(
    mean: Tensor<B, 4>,
    var: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    options: burn::tensor::ops::ConvOptions<2>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let mean_out = burn::tensor::module::conv2d(mean, weight.clone(), bias, options.clone());
    let var_out = burn::tensor::module::conv2d(var, weight.clone() * weight, None, options);
    (mean_out, var_out)
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

/// Independent Cauchy distributions per feature: `location` and `scale`, `[n, d]`.
///
/// Cauchy is the heavy-tailed stable distribution. It has NO mean or variance
/// (both integrals diverge), so we propagate its location (median) and scale
/// (half-width) rather than moments. The heavy tails keep predictions
/// appropriately uncertain far from the training data, which is the basis of the
/// Cauchy mode of Petersen et al. (2024) for OOD robustness. Like Gaussians,
/// Cauchys are closed under linear maps, so the linear step is exact.
#[derive(Clone, Debug)]
pub struct Cauchy<B: Backend> {
    pub location: Tensor<B, 2>,
    pub scale: Tensor<B, 2>,
}

impl<B: Backend> Cauchy<B> {
    pub fn new(location: Tensor<B, 2>, scale: Tensor<B, 2>) -> Self {
        Self { location, scale }
    }

    /// Half-width of the symmetric central interval of probability mass `p`:
    /// `scale * tan(pi p / 2)` (e.g. p=0.9 -> scale * 6.31). Far wider than the
    /// Gaussian `1.64 * sigma` -- the heavy-tail signature.
    pub fn interval_halfwidth(&self, p: f64) -> Tensor<B, 2> {
        self.scale.clone().mul_scalar((PI * p / 2.0).tan())
    }
}

/// Cauchy propagation through `y = x W + b`. Location maps linearly; scale adds
/// under ABSOLUTE weights: `scale_out = scale @ |W|` (vs `var @ W^2` for
/// Gaussians -- the `|.|` and the lack of squaring are the heavy-tail signature).
/// Exact: Cauchy is closed under linear maps.
pub fn propagate_linear_cauchy<B: Backend>(
    c: &Cauchy<B>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Cauchy<B> {
    let d_out = weight.dims()[1];
    let mut location = c.location.clone().matmul(weight.clone());
    if let Some(b) = bias {
        location = location + b.reshape([1, d_out]);
    }
    let scale = c.scale.clone().matmul(weight.abs());
    Cauchy { location, scale }
}

/// Cauchy propagation through ReLU via local linearization (Petersen 2024): the
/// gate is 1 where the location is positive, 0 otherwise; the location is
/// rectified and the scale is gated.
pub fn propagate_relu_cauchy<B: Backend>(c: &Cauchy<B>) -> Cauchy<B> {
    let gate = c.location.clone().clamp_min(0.0).sign();
    Cauchy {
        location: c.location.clone().clamp_min(0.0),
        scale: c.scale.clone() * gate,
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

    /// Cauchy is closed under linear maps, so `propagate_linear_cauchy` is exact.
    /// Validate location (median) and scale (half-IQR) against Cauchy-input MC --
    /// moments can't be used because a Cauchy has none.
    #[test]
    fn cauchy_linear_exact_vs_monte_carlo() {
        let dev = <B as Backend>::Device::default();
        let (n, d_in, d_out, k) = (3usize, 4usize, 3usize, 40_000usize);

        let loc = Tensor::<B, 2>::random([n, d_in], Distribution::Normal(0.0, 1.0), &dev);
        let scale = Tensor::<B, 2>::full([n, d_in], 0.5, &dev);
        let w = Tensor::<B, 2>::random([d_in, d_out], Distribution::Normal(0.0, 1.0), &dev);
        let b = Tensor::<B, 1>::random([d_out], Distribution::Normal(0.0, 0.2), &dev);

        let out = propagate_linear_cauchy(
            &Cauchy::new(loc.clone(), scale.clone()),
            w.clone(),
            Some(b.clone()),
        );
        let p_loc = out.location.to_data().to_vec::<f32>().unwrap();
        let p_scale = out.scale.to_data().to_vec::<f32>().unwrap();

        let loc_v = loc.to_data().to_vec::<f32>().unwrap();
        let scale_v = scale.to_data().to_vec::<f32>().unwrap();
        let w_v = w.to_data().to_vec::<f32>().unwrap();
        let b_v = b.to_data().to_vec::<f32>().unwrap();

        let mut rng = 0x00C0_FFEE_u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
        };
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(k); n * d_out];
        for _ in 0..k {
            for i in 0..n {
                let x: Vec<f64> = (0..d_in)
                    .map(|c| {
                        loc_v[i * d_in + c] as f64
                            + scale_v[i * d_in + c] as f64 * (PI * (next() - 0.5)).tan()
                    })
                    .collect();
                for j in 0..d_out {
                    let y = b_v[j] as f64
                        + (0..d_in)
                            .map(|c| x[c] * w_v[c * d_out + j] as f64)
                            .sum::<f64>();
                    samples[i * d_out + j].push(y);
                }
            }
        }
        for idx in 0..n * d_out {
            let s = &mut samples[idx];
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[k / 2];
            let mc_scale = (s[3 * k / 4] - s[k / 4]) / 2.0;
            assert!(
                (p_loc[idx] as f64 - med).abs() < 0.08,
                "loc {idx}: {} vs median {med}",
                p_loc[idx]
            );
            let rel = (p_scale[idx] as f64 - mc_scale).abs() / mc_scale.max(1e-6);
            assert!(
                rel < 0.10,
                "scale {idx}: {} vs half-IQR {mc_scale}",
                p_scale[idx]
            );
        }
    }

    /// Leaky-ReLU moments must match Monte Carlo.
    #[test]
    fn leaky_relu_matches_monte_carlo() {
        let dev = <B as Backend>::Device::default();
        let (n, d, k) = (2usize, 4usize, 80_000usize);
        let (alpha, std) = (0.1f64, 0.7f64);

        let mean = Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, 0.5), &dev);
        let var = Tensor::<B, 2>::full([n, d], std * std, &dev);
        let out = propagate_leaky_relu(&Moments::new(mean.clone(), var.clone()), alpha);
        let sm = out.mean.to_data().to_vec::<f32>().unwrap();
        let sv = out.var.to_data().to_vec::<f32>().unwrap();

        let len = n * d;
        let mut samples = Vec::with_capacity(k);
        for _ in 0..k {
            let noise = Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, std), &dev);
            let x = mean.clone() + noise;
            let y = x.clone().clamp_min(0.0) + x.clamp_max(0.0).mul_scalar(alpha);
            samples.push(
                y.to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|v| *v as f64)
                    .collect(),
            );
        }
        let (mc_mean, mc_var) = mc_moments(&samples, len);
        for i in 0..len {
            assert!(
                (sm[i] as f64 - mc_mean[i]).abs() < 0.02,
                "mean {i}: {} vs {}",
                sm[i],
                mc_mean[i]
            );
            let rel = (sv[i] as f64 - mc_var[i]).abs() / mc_var[i].max(1e-9);
            assert!(rel < 0.08, "var {i}: {} vs {}", sv[i], mc_var[i]);
        }
    }

    /// Residual `y = x + branch(x)` with a SMALL branch: the independence
    /// approximation of `propagate_residual_add` should be close to Monte Carlo.
    #[test]
    fn residual_add_matches_monte_carlo_small_branch() {
        let dev = <B as Backend>::Device::default();
        let (n, d, h, k) = (3usize, 4usize, 8usize, 80_000usize);
        let std = 0.5f64;

        // Small branch weights so the skip dominates.
        let w1 = Tensor::<B, 2>::random([d, h], Distribution::Normal(0.0, 0.07), &dev);
        let b1 = Tensor::<B, 1>::random([h], Distribution::Normal(0.0, 0.1), &dev);
        let w2 = Tensor::<B, 2>::random([h, d], Distribution::Normal(0.0, 0.07), &dev);
        let b2 = Tensor::<B, 1>::random([d], Distribution::Normal(0.0, 0.1), &dev);
        let mean = Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, 1.0), &dev);
        let var = Tensor::<B, 2>::full([n, d], std * std, &dev);

        let skip = Moments::new(mean.clone(), var.clone());
        let branch = propagate_linear(
            &propagate_relu(&propagate_linear(&skip, w1.clone(), Some(b1.clone()))),
            w2.clone(),
            Some(b2.clone()),
        );
        let res = propagate_residual_add(&skip, &branch);
        let r_mean = res.mean.to_data().to_vec::<f32>().unwrap();
        let r_var = res.var.to_data().to_vec::<f32>().unwrap();

        let len = n * d;
        let mut samples = Vec::with_capacity(k);
        for _ in 0..k {
            let noise = Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, std), &dev);
            let x = mean.clone() + noise;
            let br = (x.clone().matmul(w1.clone()) + b1.clone().reshape([1, h]))
                .clamp_min(0.0)
                .matmul(w2.clone())
                + b2.clone().reshape([1, d]);
            let y = x + br;
            samples.push(
                y.to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|v| *v as f64)
                    .collect(),
            );
        }
        let (mc_mean, mc_var) = mc_moments(&samples, len);
        for i in 0..len {
            assert!(
                (r_mean[i] as f64 - mc_mean[i]).abs() < 0.03,
                "mean {i}: {} vs {}",
                r_mean[i],
                mc_mean[i]
            );
            let rel = (r_var[i] as f64 - mc_var[i]).abs() / mc_var[i].max(1e-9);
            assert!(
                rel < 0.15,
                "var {i}: {} vs {} (rel {rel})",
                r_var[i],
                mc_var[i]
            );
        }
    }

    /// Conv2d is linear, so `propagate_conv2d` variance is exact: it must match
    /// Monte Carlo tightly.
    #[test]
    fn conv2d_variance_matches_monte_carlo() {
        let dev = <B as Backend>::Device::default();
        let (n, cin, hw, cout, ksz, k) = (2usize, 3usize, 6usize, 4usize, 3usize, 20_000usize);
        let std = 0.3f64;
        let opts = burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1);

        let weight =
            Tensor::<B, 4>::random([cout, cin, ksz, ksz], Distribution::Normal(0.0, 0.4), &dev);
        let bias = Tensor::<B, 1>::random([cout], Distribution::Normal(0.0, 0.2), &dev);
        let mean = Tensor::<B, 4>::random([n, cin, hw, hw], Distribution::Normal(0.0, 1.0), &dev);
        let var = Tensor::<B, 4>::full([n, cin, hw, hw], std * std, &dev);

        let (_, var_out) = propagate_conv2d(
            mean.clone(),
            var,
            weight.clone(),
            Some(bias.clone()),
            opts.clone(),
        );
        let p_var = var_out.to_data().to_vec::<f32>().unwrap();
        let len = p_var.len();

        let mut samples = Vec::with_capacity(k);
        for _ in 0..k {
            let noise =
                Tensor::<B, 4>::random([n, cin, hw, hw], Distribution::Normal(0.0, std), &dev);
            let y = burn::tensor::module::conv2d(
                mean.clone() + noise,
                weight.clone(),
                Some(bias.clone()),
                opts.clone(),
            );
            samples.push(
                y.to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|v| *v as f64)
                    .collect(),
            );
        }
        let (_, mc_var) = mc_moments(&samples, len);
        for i in 0..len {
            let rel = (p_var[i] as f64 - mc_var[i]).abs() / mc_var[i].max(1e-9);
            assert!(rel < 0.10, "var {i}: {} vs {}", p_var[i], mc_var[i]);
        }
    }

    // --- Fast invariant / reduction / edge-case tests (no Monte Carlo) ---

    fn close(a: &Tensor<B, 2>, b: &Tensor<B, 2>, tol: f64) {
        let (av, bv) = (
            a.to_data().to_vec::<f32>().unwrap(),
            b.to_data().to_vec::<f32>().unwrap(),
        );
        for i in 0..av.len() {
            assert!(
                (av[i] as f64 - bv[i] as f64).abs() < tol,
                "elem {i}: {} vs {}",
                av[i],
                bv[i]
            );
        }
    }

    fn fixture() -> (Tensor<B, 2>, Moments<B>) {
        let dev = <B as Backend>::Device::default();
        let mean = Tensor::<B, 2>::random([4, 5], Distribution::Normal(0.0, 1.0), &dev);
        let var = Tensor::<B, 2>::full([4, 5], 0.4, &dev);
        (mean.clone(), Moments::new(mean, var))
    }

    /// Leaky ReLU at alpha = 0 is plain ReLU.
    #[test]
    fn leaky_reduces_to_relu() {
        let (_, m) = fixture();
        let r = propagate_relu(&m);
        let l = propagate_leaky_relu(&m, 0.0);
        close(&r.mean, &l.mean, 1e-5);
        close(&r.var, &l.var, 1e-5);
    }

    /// Weight-uncertainty propagation with zero weight variance is plain linear.
    #[test]
    fn bayes_reduces_to_linear_at_zero_weight_var() {
        let dev = <B as Backend>::Device::default();
        let (_, m) = fixture();
        let w = Tensor::<B, 2>::random([5, 3], Distribution::Normal(0.0, 1.0), &dev);
        let lin = propagate_linear(&m, w.clone(), None);
        let wvar = w.clone().zeros_like();
        let bayes = propagate_linear_bayes(&m, w, wvar, None);
        close(&lin.mean, &bayes.mean, 1e-5);
        close(&lin.var, &bayes.var, 1e-5);
    }

    /// The diagonal of full-covariance propagation equals diagonal propagation
    /// after a single linear layer.
    #[test]
    fn full_cov_diagonal_matches_diagonal_linear() {
        let dev = <B as Backend>::Device::default();
        let (mean, m) = fixture();
        let w = Tensor::<B, 2>::random([5, 3], Distribution::Normal(0.0, 1.0), &dev);
        let diag = propagate_linear(&m, w.clone(), None);
        let full = propagate_linear_full(&MomentsFull::from_diagonal(mean, m.var.clone()), w, None);
        close(&diag.var, &full.variance(), 1e-4);
    }

    /// `from_diagonal(..).variance()` round-trips the variance.
    #[test]
    fn from_diagonal_roundtrip() {
        let (mean, m) = fixture();
        let mf = MomentsFull::from_diagonal(mean, m.var.clone());
        close(&m.var, &mf.variance(), 1e-5);
    }

    /// Deterministic input (zero variance): ReLU output mean is `max(0, mean)`
    /// and output variance stays ~0.
    #[test]
    fn relu_deterministic_input() {
        let dev = <B as Backend>::Device::default();
        let mean =
            Tensor::<B, 2>::from_data(TensorData::new(vec![1.0f32, -1.0, 2.0, -0.5], [1, 4]), &dev);
        let var = Tensor::<B, 2>::full([1, 4], 0.0, &dev);
        let out = propagate_relu(&Moments::new(mean, var));
        let m = out.mean.to_data().to_vec::<f32>().unwrap();
        let v = out.var.to_data().to_vec::<f32>().unwrap();
        let expect = [1.0, 0.0, 2.0, 0.0];
        for i in 0..4 {
            assert!((m[i] - expect[i]).abs() < 1e-3, "mean {i}: {}", m[i]);
            assert!(v[i] < 1e-3, "var {i}: {}", v[i]);
        }
    }

    /// All propagated variances stay non-negative.
    #[test]
    fn variance_stays_nonnegative() {
        let dev = <B as Backend>::Device::default();
        let (mean, m) = fixture();
        let w = Tensor::<B, 2>::random([5, 5], Distribution::Normal(0.0, 2.0), &dev);
        let chain = propagate_relu(&propagate_linear(
            &propagate_leaky_relu(&propagate_linear(&m, w.clone(), None), 0.1),
            w,
            None,
        ));
        let v = chain.var.to_data().to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| *x >= 0.0), "negative variance present");
        let _ = mean;
    }

    /// Cauchy ReLU gates the scale to zero where the location is negative.
    #[test]
    fn cauchy_relu_gates_scale() {
        let dev = <B as Backend>::Device::default();
        let loc = Tensor::<B, 2>::from_data(TensorData::new(vec![2.0f32, -3.0, 0.5], [1, 3]), &dev);
        let scale = Tensor::<B, 2>::full([1, 3], 1.0, &dev);
        let out = propagate_relu_cauchy(&Cauchy::new(loc, scale));
        let s = out.scale.to_data().to_vec::<f32>().unwrap();
        assert!(s[0] > 0.5, "active scale kept");
        assert!(s[1] < 1e-6, "inactive scale gated to 0");
    }
}
