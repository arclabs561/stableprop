//! Distribution propagation on Burn tensors.
//!
//! Tracks a per-feature mean and variance for a batch of independent Gaussians
//! and pushes them through linear, fixed-matmul (e.g. a GCN adjacency), and ReLU
//! layers. Linear and matmul propagate variance exactly under the diagonal
//! assumption; for Gaussian inputs ReLU evaluates the closed-form univariate
//! moments from Frey & Hinton (1999) with numerical tail handling. Burn tensor
//! operations keep propagation differentiable and backend-independent.
//!
//! The default [`Moments`] path approximates covariance as diagonal:
//! cross-feature correlations introduced by a layer are dropped before the
//! next layer. [`MomentsFull`] retains it with a third-order covariance series,
//! while [`Cauchy`] applies a local-linear marginal approximation. These are
//! related to, but do not reproduce, the Jacobian propagation in Petersen et
//! al. (ICLR 2024).
//!
//! Callers must supply finite tensor values, nonnegative variances and scales,
//! and valid positive-semidefinite covariance matrices. Constructors check
//! shapes, but do not inspect tensor contents or synchronize devices to validate
//! values. Public fields carry the same requirements. ReLU formulas use linear
//! tail limits at eight standard deviations for numerical stability.

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, ElementLimits, Tensor, TensorData};
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
    /// Construct matching mean and variance tensors. See the module's value
    /// requirements; tensor contents are not checked.
    ///
    /// # Panics
    /// Panics if the tensor shapes differ.
    pub fn new(mean: Tensor<B, 2>, var: Tensor<B, 2>) -> Self {
        assert_eq!(
            mean.dims(),
            var.dims(),
            "mean and variance shapes must match"
        );
        Self { mean, var }
    }
}

/// Propagate through an affine map `y = x @ weight + bias`.
///
/// `weight` is `[d_in, d_out]` (Burn's `Linear` layout); `bias` is `[d_out]`.
/// Mean is exact; variance is `var @ weight^2`, which is the exact marginal
/// variance of each output when the input covariance is diagonal.
///
/// # Panics
/// Panics if the supplied bias width differs from the output width.
pub fn propagate_linear<B: Backend>(
    m: &Moments<B>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Moments<B> {
    let mut mean = m.mean.clone().matmul(weight.clone());
    if let Some(b) = bias {
        let d = b.dims()[0];
        assert_eq!(d, weight.dims()[1], "bias shape must match output width");
        mean = mean + b.reshape([1, d]);
    }
    let w2 = weight.clone() * weight;
    let var = m.var.clone().matmul(w2);
    Moments { mean, var }
}

/// Propagate through `y = x @ W + b` where both inputs and weights are uncertain
/// (mean-field: all elements independent). This is the linear step of
/// Probabilistic Backpropagation (Hernandez-Lobato 2015) / Deterministic
/// Variational Inference (Wu 2019). The supplied weight variances can describe
/// posterior uncertainty or a chosen parameter-noise model; this function does
/// not fit them. Reduces to [`propagate_linear`] when both `w_var` and any
/// supplied bias variance are zero.
///
/// `var_out = mean_x^2 @ w_var  +  var_x @ mean_W^2  +  var_x @ w_var  +  b_var`
/// (the first term is the parameter-variance contribution, the second is the
/// input-variance propagation, the third is the cross term).
/// Shared uncertain weights also induce covariance across batch rows. This
/// function retains only marginal variances; a following left matrix multiply
/// therefore uses an independence approximation for those rows.
///
/// Input moments are `[n, d_in]`; `w_mean` and `w_var` are `[d_in, d_out]`.
/// Both bias tensors, when supplied, are `[d_out]`. Variances must be finite
/// and nonnegative, as required by the module's input contract.
///
/// # Panics
/// Panics if the weight mean and variance shapes differ, or supplied bias mean
/// and variance shapes differ or do not match the output width.
pub fn propagate_linear_bayes<B: Backend>(
    m: &Moments<B>,
    w_mean: Tensor<B, 2>,
    w_var: Tensor<B, 2>,
    bias: Option<(Tensor<B, 1>, Tensor<B, 1>)>,
) -> Moments<B> {
    assert_eq!(
        w_mean.dims(),
        w_var.dims(),
        "weight mean and variance shapes must match"
    );
    let d_out = w_mean.dims()[1];
    let mut mean = m.mean.clone().matmul(w_mean.clone());
    let wm2 = w_mean.clone() * w_mean;
    let mx2 = m.mean.clone() * m.mean.clone();
    let mut var =
        mx2.matmul(w_var.clone()) + m.var.clone().matmul(wm2) + m.var.clone().matmul(w_var);
    if let Some((bm, bv)) = bias {
        assert_eq!(
            bm.dims(),
            bv.dims(),
            "bias mean and variance shapes must match"
        );
        let d = bm.dims()[0];
        assert_eq!(d, d_out, "bias shapes must match output width");
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
/// `a` has shape `[n_out, n_in]` and input moments have shape `[n_in, d]`;
/// both output tensors have shape `[n_out, d]`.
pub fn propagate_matmul_left<B: Backend>(a: Tensor<B, 2>, m: &Moments<B>) -> Moments<B> {
    let mean = a.clone().matmul(m.mean.clone());
    let a2 = a.clone() * a;
    let var = a2.matmul(m.var.clone());
    Moments { mean, var }
}

struct GaussianReluTerms<B: Backend> {
    mean: Tensor<B, 2>,
    var: Tensor<B, 2>,
    p: Tensor<B, 2>,
    alpha: Tensor<B, 2>,
    deterministic: Tensor<B, 2, burn::tensor::Bool>,
    active: Tensor<B, 2, burn::tensor::Bool>,
    inactive: Tensor<B, 2, burn::tensor::Bool>,
}

/// Central-region Gaussian ReLU terms, with zero variance made safe only for
/// intermediate division. Callers restore deterministic outputs with the masks.
fn gaussian_relu_terms<B: Backend>(
    mu: Tensor<B, 2>,
    input_var: Tensor<B, 2>,
) -> GaussianReluTerms<B> {
    let deterministic = input_var.clone().lower_equal_elem(0.0);
    let var = input_var.clone().mask_fill(deterministic.clone(), 1.0);
    let sigma = var.clone().sqrt();
    let alpha_raw = mu.clone() / sigma.clone();
    let alpha = alpha_raw.clone().clamp(-8.0, 8.0);
    let p = alpha
        .clone()
        .mul_scalar(FRAC_1_SQRT_2)
        .erf()
        .add_scalar(1.0)
        .mul_scalar(0.5);
    let phi = (alpha.clone() * alpha.clone())
        .mul_scalar(-0.5)
        .exp()
        .mul_scalar(1.0 / (2.0 * PI).sqrt());
    let mean = mu.clone() * p.clone() + sigma * phi.clone();
    let normalized_var =
        alpha.clone() * alpha.clone() * p.clone() * p.clone().mul_scalar(-1.0).add_scalar(1.0)
            + p.clone()
            + alpha.clone() * phi.clone() * p.clone().mul_scalar(-2.0).add_scalar(1.0)
            - phi.clone() * phi;
    GaussianReluTerms {
        mean,
        var: (var * normalized_var).clamp_min(0.0),
        p,
        alpha,
        deterministic,
        active: alpha_raw.clone().greater_equal_elem(8.0),
        inactive: alpha_raw.lower_equal_elem(-8.0),
    }
}

/// Propagate through an element-wise ReLU via Frey & Hinton (1999) moment
/// matching.
///
/// Per element with mean `mu`, std `sigma`, `alpha = mu / sigma`:
/// `mu'    = mu * Phi(alpha) + sigma * phi(alpha)`
/// `var'` is evaluated in an algebraically equivalent form that avoids
/// cancellation when ReLU is nearly linear. `Phi` / `phi` are the standard
/// normal CDF / PDF, with numerical tail limits described at module level.
pub fn propagate_relu<B: Backend>(m: &Moments<B>) -> Moments<B> {
    let mu = m.mean.clone();
    let terms = gaussian_relu_terms(mu.clone(), m.var.clone());

    Moments {
        mean: terms
            .mean
            .mask_where(terms.active.clone(), mu.clone())
            .mask_fill(terms.inactive.clone(), 0.0)
            .mask_where(terms.deterministic.clone(), mu.clamp_min(0.0)),
        var: terms
            .var
            .mask_where(terms.active, m.var.clone())
            .mask_fill(terms.inactive, 0.0)
            .mask_fill(terms.deterministic, 0.0),
    }
}

/// Propagate through leaky ReLU `x` for `x >= 0` and `alpha * x` otherwise via
/// closed-form Gaussian moments with numerical tail handling.
///
/// Uses `leaky(x) = alpha*x + (1-alpha)*relu(x)`, so the moments combine the raw
/// and rectified-Gaussian moments. Reduces to [`propagate_relu`] at `alpha = 0`.
///
/// # Panics
/// Panics if `alpha` is not finite or the moment coefficients are not finite
/// and representable in the backend's scalar element type. Large input values
/// can still overflow the resulting moments.
pub fn propagate_leaky_relu<B: Backend>(m: &Moments<B>, alpha: f64) -> Moments<B> {
    assert!(alpha.is_finite(), "leaky-ReLU slope must be finite");
    let one_minus_alpha = 1.0 - alpha;
    let alpha_sq = alpha * alpha;
    let complement_sq = one_minus_alpha * one_minus_alpha;
    let cross_coefficient = 2.0 * alpha * one_minus_alpha;
    let max_scalar = B::FloatElem::MAX.elem::<f64>();
    assert!(
        [
            alpha,
            one_minus_alpha,
            alpha_sq,
            complement_sq,
            cross_coefficient
        ]
        .iter()
        .all(|x| x.is_finite() && x.abs() <= max_scalar),
        "leaky-ReLU moment coefficients must be finite and representable by the backend"
    );
    let mu = m.mean.clone();
    let terms = gaussian_relu_terms(mu.clone(), m.var.clone());
    let mean = terms.mean.mul_scalar(one_minus_alpha) + mu.clone().mul_scalar(alpha);
    // Cov(X, ReLU(X)) = var * Phi(a). This avoids another unstable
    // second-moment subtraction in the leaky-ReLU variance.
    let central_var = m.var.clone().mul_scalar(alpha_sq)
        + terms.var.mul_scalar(complement_sq)
        + m.var.clone() * terms.p.mul_scalar(cross_coefficient);
    Moments {
        mean: mean
            .mask_where(terms.active.clone(), m.mean.clone())
            .mask_where(terms.inactive.clone(), m.mean.clone().mul_scalar(alpha))
            .mask_where(terms.deterministic.clone(), {
                let x = m.mean.clone();
                x.clone()
                    .clamp_min(0.0)
                    .add(x.clamp_max(0.0).mul_scalar(alpha))
            }),
        var: central_var
            .mask_where(terms.active, m.var.clone())
            .mask_where(terms.inactive, m.var.clone().mul_scalar(alpha_sq))
            .mask_fill(terms.deterministic, 0.0),
    }
}

/// Combine a residual skip and a branch `y = skip + branch` under the
/// independence approximation: `mean = skip.mean + branch.mean`,
/// `var = skip.var + branch.var`.
///
/// This ignores the skip-branch covariance (the branch is a function of the
/// skip's input, so they are correlated). Use
/// [`propagate_residual_add_correlated`] when that covariance is available.
///
/// # Panics
/// Panics if the residual shapes differ. Batch broadcasting is not supported.
pub fn propagate_residual_add<B: Backend>(skip: &Moments<B>, branch: &Moments<B>) -> Moments<B> {
    assert_eq!(
        skip.mean.dims(),
        branch.mean.dims(),
        "residual shapes must match"
    );
    Moments {
        mean: skip.mean.clone() + branch.mean.clone(),
        var: skip.var.clone() + branch.var.clone(),
    }
}

/// Combine `y = skip + branch` with the diagonal skip-branch covariance.
///
/// `skip_branch_cov[i]` is `Cov(skip[i], branch[i])`, giving the exact marginal
/// variance `Var(y[i]) = Var(skip[i]) + Var(branch[i]) + 2 Cov(skip[i], branch[i])`.
/// All tensors must have the same shape and describe a valid joint distribution;
/// in particular `abs(skip_branch_cov) <= sqrt(skip.var * branch.var)`.
/// Values are not checked; invalid joint moments can produce negative variance.
pub fn propagate_residual_add_correlated<B: Backend>(
    skip: &Moments<B>,
    branch: &Moments<B>,
    skip_branch_cov: Tensor<B, 2>,
) -> Moments<B> {
    assert_eq!(
        skip.mean.dims(),
        branch.mean.dims(),
        "residual shapes must match"
    );
    assert_eq!(
        skip.mean.dims(),
        skip_branch_cov.dims(),
        "cross-covariance shape must match residuals"
    );
    Moments {
        mean: skip.mean.clone() + branch.mean.clone(),
        var: skip.var.clone() + branch.var.clone() + skip_branch_cov.mul_scalar(2.0),
    }
}

/// Propagate independent Gaussian inputs through a 2-D convolution.
/// Output means and marginal variances are exact:
/// `mean_out = conv(mean, w) + b`, `var_out = conv(var, w^2)`.
/// Shared inputs induce correlations between output positions and channels;
/// these correlations are not represented by the returned variance tensor.
///
/// `mean` and `var` must have matching shape `[N, C_in, H, W]`;
/// `weight` is `[C_out, C_in / groups, kh, kw]` and optional `bias` is `[C_out]`.
/// The module's finite-value and nonnegative-variance requirements apply.
///
/// # Panics
/// Panics if the mean and variance shapes differ. The underlying convolution
/// also panics for invalid weight, bias, or option shapes.
pub fn propagate_conv2d<B: Backend>(
    mean: Tensor<B, 4>,
    var: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    options: burn::tensor::ops::ConvOptions<2>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    assert_eq!(
        mean.dims(),
        var.dims(),
        "convolution mean and variance shapes must match"
    );
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

/// Mean and full covariance of a batch of `n` independent Gaussians.
///
/// `mean` is `[n, d]`, `cov` is `[n, d, d]`. Unlike [`Moments`], this keeps the
/// cross-feature correlations that diagonal propagation drops. Cost is `O(n d^2)` memory and
/// `O(n d^3)` per linear layer, so it suits small-to-medium feature dimensions.
#[derive(Clone, Debug)]
pub struct MomentsFull<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub cov: Tensor<B, 3>,
}

impl<B: Backend> MomentsFull<B> {
    /// Construct means `[n, d]` and covariance matrices `[n, d, d]`.
    /// The module's value requirements apply; contents are not inspected.
    ///
    /// # Panics
    /// Panics if the covariance shape does not match the mean shape.
    pub fn new(mean: Tensor<B, 2>, cov: Tensor<B, 3>) -> Self {
        let [n, d] = mean.dims();
        assert_eq!(cov.dims(), [n, d, d], "covariance shape must be [n, d, d]");
        Self { mean, cov }
    }

    /// Build from a diagonal variance `[n, d]` (independent input features):
    /// `cov = diag(var)` per row.
    pub fn from_diagonal(mean: Tensor<B, 2>, var: Tensor<B, 2>) -> Self {
        assert_eq!(
            mean.dims(),
            var.dims(),
            "mean and variance shapes must match"
        );
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
///
/// Input mean and covariance are `[n, d_in]` and `[n, d_in, d_in]`;
/// `weight` is `[d_in, d_out]` and optional `bias` is `[d_out]`.
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

/// Full-covariance ReLU: closed-form Gaussian marginal moments on the diagonal
/// with numerical tail handling, and a third-order covariance series off-diagonal.
///
/// The leading series term is the smooth gate `Phi(alpha_i) Phi(alpha_j)`.
pub fn propagate_relu_full<B: Backend>(m: &MomentsFull<B>) -> MomentsFull<B> {
    let [n, d, _] = m.cov.dims();
    let dev = m.cov.device();
    let eye_d = eye::<B>(d, &dev);

    let input_var = (m.cov.clone() * eye_d.clone().unsqueeze::<3>())
        .sum_dim(2)
        .reshape([n, d]);
    let mu = m.mean.clone();
    let terms = gaussian_relu_terms(mu.clone(), input_var.clone());
    let var_out = terms
        .var
        .mask_where(terms.active.clone(), input_var.clone())
        .mask_fill(terms.inactive.clone(), 0.0)
        .mask_fill(terms.deterministic.clone(), 0.0);
    let mu_out = terms
        .mean
        .mask_where(terms.active.clone(), mu.clone())
        .mask_fill(terms.inactive.clone(), 0.0)
        .mask_where(terms.deterministic.clone(), mu.clone().clamp_min(0.0));

    // Off-diagonal: post-ReLU covariance via the Wright et al. (2024) series
    // Cov_ij = sum_k (Sigma_ij^k / k!) d_k(i) d_k(j), to 3rd order. The k=1 term
    // is the smooth gate Phi(a_i) Phi(a_j) Sigma_ij; the derivatives of E[relu]
    // in the input mean are d1 = Phi(a), d2 = phi(a)/sigma, d3 = -a phi(a)/sigma^2.
    // The diagonal is then overwritten with the univariate variance.
    let safe_var = input_var
        .clone()
        .mask_fill(terms.deterministic.clone(), 1.0);
    let sigma = safe_var.clone().sqrt();
    let sigma_outer = sigma.clone().unsqueeze_dim::<3>(2) * sigma.clone().unsqueeze_dim::<3>(1);
    let off_mask = eye_d
        .clone()
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .unsqueeze::<3>();
    // Normalize one axis at a time: dividing by sigma_i * sigma_j directly
    // makes its squared denominator underflow during f32 autodiff at tiny
    // variances. Exclude the diagonal, which is replaced below.
    let rho = ((m.cov.clone() * off_mask / sigma.clone().unsqueeze_dim::<3>(2))
        / sigma.unsqueeze_dim::<3>(1))
    .clamp(-1.0, 1.0);
    let phi = (terms.alpha.clone() * terms.alpha.clone())
        .mul_scalar(-0.5)
        .exp()
        .mul_scalar(1.0 / (2.0 * PI).sqrt());
    let outer = |t: Tensor<B, 2>| t.clone().unsqueeze_dim::<3>(2) * t.unsqueeze_dim::<3>(1);
    let rho2 = rho.clone() * rho.clone();
    let rho3 = rho2.clone() * rho.clone();
    // Apply the same linear tail limits to covariance as to marginal moments.
    // An inactive output cannot covary; an active output is the input itself.
    let tail = terms.active.clone().bool_or(terms.inactive.clone());
    let off_p = terms
        .p
        .mask_fill(terms.active, 1.0)
        .mask_fill(terms.inactive, 0.0);
    let off_phi = phi.mask_fill(tail.bool_or(terms.deterministic), 0.0);
    let off_alpha_phi = terms.alpha * off_phi.clone();
    let off = sigma_outer
        * (rho * outer(off_p)
            + rho2.mul_scalar(0.5) * outer(off_phi)
            + rho3.mul_scalar(1.0 / 6.0) * outer(off_alpha_phi));
    let diag = var_out.unsqueeze_dim::<3>(2).expand([n, d, d]) * eye_d.unsqueeze::<3>();
    MomentsFull {
        mean: mu_out,
        cov: off + diag,
    }
}

/// Transport `Cov(U, V)` through an affine map of the right variable,
/// `Y = V W + b`: `Cov(U, Y) = Cov(U, V) W`.
///
/// `cross_cov` is `[n, d_left, d_in]`, with the left variable on the first
/// feature axis; `weight` is `[d_in, d_out]`. The result is `[n, d_left, d_out]`.
/// Bias does not affect covariance. This identity is exact for any joint
/// distribution with finite second moments. Rows represent separate joint
/// distributions; covariance between rows is not represented.
///
/// # Panics
/// Panics if the right feature count does not match the weight's input size.
pub fn propagate_linear_cross_covariance<B: Backend>(
    cross_cov: Tensor<B, 3>,
    weight: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [n, _, d_in] = cross_cov.dims();
    let [w_in, d_out] = weight.dims();
    assert_eq!(
        d_in, w_in,
        "cross-covariance right features must match weight inputs"
    );
    cross_cov.matmul(weight.unsqueeze::<3>().expand([n, d_in, d_out]))
}

/// Transport `Cov(U, V)` through a ReLU of the right variable.
///
/// For jointly Gaussian `(U, V)`, Gaussian integration by parts gives
/// `Cov(U, ReLU(V)) = Cov(U, V) diag(Phi(mean_V / std_V))`.
/// Only the right marginal means and variances are needed; correlations among
/// right features may be tracked separately. `cross_cov` is `[n, d_left, d_right]`
/// and `right` contains those marginal moments as `[n, d_right]` tensors.
///
/// The cross-covariance and both margins must describe a valid joint Gaussian.
/// In particular a deterministic right feature has a zero covariance column.
/// Values and joint positive semidefiniteness are caller requirements, not
/// runtime checks. The same numerical tail limits as [`propagate_relu`] apply.
///
/// ReLU makes the joint distribution non-Gaussian. Further affine transport
/// remains exact; another use of this Gaussian ReLU identity is then an
/// approximation. Use [`propagate_residual_add_correlated`] with the diagonal
/// of the resulting skip-branch covariance to combine a residual's marginals.
///
/// # Panics
/// Panics if batch sizes, right feature counts, or marginal shapes differ.
pub fn propagate_relu_cross_covariance<B: Backend>(
    cross_cov: Tensor<B, 3>,
    right: &Moments<B>,
) -> Tensor<B, 3> {
    let [n, _, d_right] = cross_cov.dims();
    assert_eq!(
        right.mean.dims(),
        [n, d_right],
        "cross-covariance must match right moments"
    );
    assert_eq!(
        right.var.dims(),
        [n, d_right],
        "right mean and variance shapes must match"
    );
    // Only the CDF is needed here; avoid evaluating unused rectified moments.
    let deterministic = right.var.clone().lower_equal_elem(0.0);
    let sigma = right
        .var
        .clone()
        .mask_fill(deterministic.clone(), 1.0)
        .sqrt();
    let alpha = right.mean.clone() / sigma;
    let gate = alpha
        .clone()
        .clamp(-8.0, 8.0)
        .mul_scalar(FRAC_1_SQRT_2)
        .erf()
        .add_scalar(1.0)
        .mul_scalar(0.5)
        .mask_fill(alpha.clone().greater_equal_elem(8.0), 1.0)
        .mask_fill(alpha.lower_equal_elem(-8.0), 0.0)
        .mask_fill(deterministic, 0.0);
    cross_cov * gate.unsqueeze_dim::<3>(1)
}

/// Independent Cauchy distributions per feature: `location` and `scale`, `[n, d]`.
///
/// Cauchy has no finite mean or variance, so we propagate its location (median)
/// and scale rather than moments. An affine map of independent Cauchy inputs
/// has Cauchy output marginals, but mixing creates dependence between outputs
/// that this type does not represent.
#[derive(Clone, Debug)]
pub struct Cauchy<B: Backend> {
    pub location: Tensor<B, 2>,
    pub scale: Tensor<B, 2>,
}

impl<B: Backend> Cauchy<B> {
    /// Construct matching location and scale tensors. Scales must be finite and
    /// nonnegative; tensor contents are not checked.
    ///
    /// # Panics
    /// Panics if the tensor shapes differ.
    pub fn new(location: Tensor<B, 2>, scale: Tensor<B, 2>) -> Self {
        assert_eq!(
            location.dims(),
            scale.dims(),
            "location and scale shapes must match"
        );
        Self { location, scale }
    }

    /// Half-width of the symmetric central interval of probability mass `p`:
    /// `scale * tan(pi p / 2)` (e.g. p=0.9 -> scale * 6.31).
    pub fn interval_halfwidth(&self, p: f64) -> Tensor<B, 2> {
        assert!(
            p.is_finite() && (0.0..1.0).contains(&p),
            "probability mass must be finite and in [0, 1)"
        );
        self.scale.clone().mul_scalar((PI * p / 2.0).tan())
    }
}

/// Cauchy propagation through `y = x W + b`. Location maps linearly; scale adds
/// under absolute weights: `scale_out = scale @ |W|` (vs `var @ W^2` for
/// Gaussians -- the `|.|` and the lack of squaring are the heavy-tail signature).
/// Exact for independent Cauchy input marginals; dependencies between features
/// are not represented by this type.
/// Shared inputs create dependent output features. Treating the returned
/// marginals as independent in a subsequent affine layer is an approximation.
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
/// gate is 1 where the location is strictly positive (including 0 uses the
/// inactive branch); the location is rectified and the scale is gated.
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

    /// Correlated ReLU outputs are recombined by a following linear layer. A
    /// fixed covariance fixture keeps this ordering check deterministic.
    #[test]
    fn full_cov_beats_diagonal_on_correlated_relu_fixture() {
        let dev = <B as Backend>::Device::default();
        let mean = Tensor::<B, 2>::zeros([1, 2], &dev);
        let cov = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0f32, 0.5, 0.5, 1.0], [1, 2, 2]),
            &dev,
        );
        let weight = Tensor::<B, 2>::ones([2, 1], &dev);

        let full = propagate_linear_full(
            &propagate_relu_full(&MomentsFull::new(mean.clone(), cov)),
            weight.clone(),
            None,
        );
        let diagonal = propagate_linear(
            &propagate_relu(&Moments::new(mean, Tensor::<B, 2>::ones([1, 2], &dev))),
            weight,
            None,
        );
        let full_var = full.variance().to_data().to_vec::<f32>().unwrap()[0];
        let diagonal_var = diagonal.var.to_data().to_vec::<f32>().unwrap()[0];

        // Exact zero-mean ReLU covariance kernel for rho = 0.5.
        let rho = 0.5f32;
        let relu_var = 0.5 - 1.0 / (2.0 * core::f32::consts::PI);
        let relu_cov = ((1.0 - rho * rho).sqrt() + (core::f32::consts::PI - rho.acos()) * rho
            - 1.0)
            / (2.0 * core::f32::consts::PI);
        let expected = 2.0 * (relu_var + relu_cov);
        assert!(
            (full_var - expected).abs() < 0.01,
            "full variance: {full_var}"
        );
        assert!(
            full_var > diagonal_var + 0.2,
            "full ({full_var}) should retain the covariance diagonal ({diagonal_var}) drops"
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
        let (alpha, std) = (0.3f64, 0.7f64);

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

    #[test]
    fn activation_variance_is_stable_at_large_means_and_tiny_noise() {
        let dev = <B as Backend>::Device::default();
        let mean = Tensor::<B, 2>::from_data(
            TensorData::new(vec![10_000.0f32, -1.0, 0.0, 0.0], [1, 4]),
            &dev,
        );
        let var = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 1.0e-16, 0.0], [1, 4]),
            &dev,
        );
        let relu = propagate_relu(&Moments::new(mean.clone(), var.clone()));
        let relu_mean = relu.mean.to_data().to_vec::<f32>().unwrap();
        let relu_var = relu.var.to_data().to_vec::<f32>().unwrap();
        assert!((relu_mean[0] - 10_000.0).abs() < 1e-3);
        assert!(
            (relu_var[0] - 1.0).abs() < 1e-5,
            "large-mean ReLU variance: {}",
            relu_var[0]
        );
        assert_eq!(relu_mean[1], 0.0);
        assert_eq!(relu_var[1], 0.0);
        assert!(
            (relu_var[2] - (0.5 - 1.0 / (2.0 * core::f32::consts::PI)) * 1.0e-16).abs() < 1e-20,
            "tiny-noise ReLU variance: {}",
            relu_var[2]
        );
        assert_eq!(relu_var[3], 0.0);

        let leaky = propagate_leaky_relu(&Moments::new(mean, var), 0.3);
        let leaky_var = leaky.var.to_data().to_vec::<f32>().unwrap();
        assert!(
            (leaky_var[0] - 1.0).abs() < 1e-5,
            "large-mean leaky-ReLU variance: {}",
            leaky_var[0]
        );
        assert_eq!(leaky_var[1], 0.0);
        assert_eq!(leaky_var[3], 0.0);
    }

    #[test]
    fn full_relu_variance_is_stable_at_large_means() {
        let dev = <B as Backend>::Device::default();
        let mean = Tensor::<B, 2>::from_data(TensorData::new(vec![10_000.0f32], [1, 1]), &dev);
        let cov = Tensor::<B, 3>::from_data(TensorData::new(vec![1.0f32], [1, 1, 1]), &dev);
        let out = propagate_relu_full(&MomentsFull::new(mean, cov));
        let var = out.variance().to_data().to_vec::<f32>().unwrap();
        assert!(
            (var[0] - 1.0).abs() < 1e-5,
            "large-mean full-ReLU variance: {}",
            var[0]
        );
    }

    #[test]
    fn full_relu_handles_zero_and_tiny_covariances() {
        let dev = <B as Backend>::Device::default();
        let zero_mean =
            Tensor::<B, 2>::from_data(TensorData::new(vec![0.0f32, -2.0], [1, 2]), &dev);
        let zero_cov = Tensor::<B, 3>::zeros([1, 2, 2], &dev);
        let zero = propagate_relu_full(&MomentsFull::new(zero_mean, zero_cov));
        assert_eq!(zero.mean.to_data().to_vec::<f32>().unwrap(), vec![0.0, 0.0]);
        assert_eq!(zero.cov.to_data().to_vec::<f32>().unwrap(), vec![0.0; 4]);

        let tiny_mean =
            Tensor::<B, 2>::from_data(TensorData::new(vec![0.0f32, 1.0e-8], [1, 2]), &dev);
        let tiny_cov = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0e-24f32, 0.5e-24, 0.5e-24, 1.0e-24], [1, 2, 2]),
            &dev,
        );
        let tiny = propagate_relu_full(&MomentsFull::new(tiny_mean, tiny_cov));
        assert!(
            tiny.cov
                .to_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|x| x.is_finite()),
            "tiny covariance produced non-finite output"
        );
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

    /// Linear WITH bias: the bias must be added to the mean (and not the
    /// variance). Catches mutations that drop or misplace the bias.
    #[test]
    fn linear_bias_added_to_mean_only() {
        let dev = <B as Backend>::Device::default();
        let mean = Tensor::<B, 2>::zeros([2, 3], &dev);
        let var = Tensor::<B, 2>::full([2, 3], 0.5, &dev);
        let w = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], [3, 2]),
            &dev,
        );
        let bias = Tensor::<B, 1>::from_data(TensorData::new(vec![5.0f32, -2.0], [2]), &dev);
        let out = propagate_linear(&Moments::new(mean, var), w, Some(bias));
        let m = out.mean.to_data().to_vec::<f32>().unwrap();
        let v = out.var.to_data().to_vec::<f32>().unwrap();
        // mean = 0*W + bias = [5, -2] per row.
        assert!(
            (m[0] - 5.0).abs() < 1e-4 && (m[1] + 2.0).abs() < 1e-4,
            "bias not in mean"
        );
        // variance unaffected by bias: col 0 gets var(=0.5), col 1 gets 0.
        assert!((v[0] - 0.5).abs() < 1e-4, "bias leaked into variance");
    }

    /// GCN-adjacency propagation: `var_out = (a*a) @ var`. Validate vs MC.
    #[test]
    fn matmul_left_variance_matches_monte_carlo() {
        let dev = <B as Backend>::Device::default();
        let (n, d, k) = (4usize, 3usize, 60_000usize);
        let std = 0.4f64;
        let a = Tensor::<B, 2>::random([n, n], Distribution::Normal(0.0, 0.6), &dev);
        let mean = Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, 1.0), &dev);
        let var = Tensor::<B, 2>::full([n, d], std * std, &dev);
        let out = propagate_matmul_left(a.clone(), &Moments::new(mean.clone(), var));
        let pv = out.var.to_data().to_vec::<f32>().unwrap();

        let len = n * d;
        let mut samples = Vec::with_capacity(k);
        for _ in 0..k {
            let noise = Tensor::<B, 2>::random([n, d], Distribution::Normal(0.0, std), &dev);
            let y = a.clone().matmul(mean.clone() + noise);
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
            let rel = (pv[i] as f64 - mc_var[i]).abs() / mc_var[i].max(1e-9);
            assert!(rel < 0.10, "var {i}: {} vs {}", pv[i], mc_var[i]);
        }
    }

    /// Cauchy interval half-width is `scale * tan(pi p / 2)`.
    #[test]
    fn cauchy_interval_halfwidth_value() {
        let dev = <B as Backend>::Device::default();
        let scale = Tensor::<B, 2>::full([1, 2], 2.0, &dev);
        let c = Cauchy::new(Tensor::zeros([1, 2], &dev), scale);
        let hw = c.interval_halfwidth(0.9).to_data().to_vec::<f32>().unwrap();
        let expect = 2.0 * (PI * 0.9 / 2.0).tan();
        assert!(
            (hw[0] as f64 - expect).abs() < 0.01,
            "{} vs {expect}",
            hw[0]
        );
    }

    /// Weight-uncertainty propagation with non-zero weight and bias variance:
    /// the mean and variance must match Monte Carlo over inputs, weights, bias.
    #[test]
    fn bayes_weight_uncertainty_matches_mc() {
        let dev = <B as Backend>::Device::default();
        let (n, din, dout, k) = (3usize, 4usize, 3usize, 60_000usize);
        let (in_std, w_std, b_std) = (0.3f64, 0.2f64, 0.15f64);

        let w_mean = Tensor::<B, 2>::random([din, dout], Distribution::Normal(0.0, 1.0), &dev);
        let w_var = Tensor::<B, 2>::full([din, dout], w_std * w_std, &dev);
        let b_mean = Tensor::<B, 1>::random([dout], Distribution::Normal(0.0, 0.5), &dev);
        let b_var = Tensor::<B, 1>::full([dout], b_std * b_std, &dev);
        let mean = Tensor::<B, 2>::random([n, din], Distribution::Normal(0.0, 1.0), &dev);
        let var = Tensor::<B, 2>::full([n, din], in_std * in_std, &dev);

        let out = propagate_linear_bayes(
            &Moments::new(mean.clone(), var.clone()),
            w_mean.clone(),
            w_var,
            Some((b_mean.clone(), b_var)),
        );
        let p_mean = out.mean.to_data().to_vec::<f32>().unwrap();
        let p_var = out.var.to_data().to_vec::<f32>().unwrap();

        let len = n * dout;
        let mut samples = Vec::with_capacity(k);
        for _ in 0..k {
            let wk = w_mean.clone()
                + Tensor::<B, 2>::random([din, dout], Distribution::Normal(0.0, w_std), &dev);
            let bk = b_mean.clone()
                + Tensor::<B, 1>::random([dout], Distribution::Normal(0.0, b_std), &dev);
            let xk = mean.clone()
                + Tensor::<B, 2>::random([n, din], Distribution::Normal(0.0, in_std), &dev);
            let yk = xk.matmul(wk) + bk.reshape([1, dout]);
            samples.push(
                yk.to_data()
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
                (p_mean[i] as f64 - mc_mean[i]).abs() < 0.03,
                "mean {i}: {} vs {}",
                p_mean[i],
                mc_mean[i]
            );
            let rel = (p_var[i] as f64 - mc_var[i]).abs() / mc_var[i].max(1e-9);
            assert!(rel < 0.10, "var {i}: {} vs {}", p_var[i], mc_var[i]);
        }
    }

    /// On a diagonal input the full-covariance ReLU diagonal equals the plain
    /// diagonal ReLU (the off-diagonal gating contributes nothing).
    #[test]
    fn full_cov_relu_diagonal_matches_diagonal_relu() {
        let dev = <B as Backend>::Device::default();
        let mean = Tensor::<B, 2>::random([4, 5], Distribution::Normal(0.0, 1.0), &dev);
        let var = Tensor::<B, 2>::full([4, 5], 0.4, &dev);
        let diag = propagate_relu(&Moments::new(mean.clone(), var.clone()));
        let full = propagate_relu_full(&MomentsFull::from_diagonal(mean, var));
        close(&diag.var, &full.variance(), 1e-4);
        close(&diag.mean, &full.mean, 1e-4);
    }

    /// Residual-add is exactly the element-wise sum of the two moments.
    #[test]
    fn residual_add_is_exact_sum() {
        let dev = <B as Backend>::Device::default();
        let m1 = Tensor::<B, 2>::random([2, 3], Distribution::Normal(0.0, 1.0), &dev);
        let v1 = Tensor::<B, 2>::full([2, 3], 0.5, &dev);
        let m2 = Tensor::<B, 2>::random([2, 3], Distribution::Normal(0.0, 1.0), &dev);
        let v2 = Tensor::<B, 2>::full([2, 3], 0.3, &dev);
        let r = propagate_residual_add(
            &Moments::new(m1.clone(), v1.clone()),
            &Moments::new(m2.clone(), v2.clone()),
        );
        close(&r.mean, &(m1 + m2), 1e-5);
        close(&r.var, &(v1 + v2), 1e-5);
    }

    #[test]
    fn correlated_residual_add_includes_cross_covariance() {
        let dev = <B as Backend>::Device::default();
        let mean = Tensor::<B, 2>::from_data(TensorData::new(vec![1.0f32, -2.0], [1, 2]), &dev);
        let var = Tensor::<B, 2>::from_data(TensorData::new(vec![0.5f32, 2.0], [1, 2]), &dev);
        let skip = Moments::new(mean.clone(), var.clone());
        let branch = Moments::new(mean, var.clone());

        // branch == skip, so Cov(skip, branch) == Var(skip) and
        // Var(skip + branch) == 4 Var(skip).
        let out = propagate_residual_add_correlated(&skip, &branch, var);

        let out_mean = out.mean.to_data().to_vec::<f32>().unwrap();
        let out_var = out.var.to_data().to_vec::<f32>().unwrap();
        assert_eq!(out_mean, vec![2.0, -4.0]);
        assert_eq!(out_var, vec![2.0, 8.0]);
    }

    #[test]
    #[should_panic(expected = "probability mass must be finite and in [0, 1)")]
    fn cauchy_interval_rejects_invalid_probability_mass() {
        let dev = <B as Backend>::Device::default();
        let c = Cauchy::<B>::new(Tensor::zeros([1, 1], &dev), Tensor::ones([1, 1], &dev));
        let _ = c.interval_halfwidth(1.0);
    }

    /// Full post-ReLU covariance (diagonal and off-diagonal) must match Monte
    /// Carlo. This exercises the Wright-series off-diagonal terms; correlated
    /// pre-activations are produced by a linear layer from a diagonal input.
    #[test]
    fn relu_full_covariance_matches_monte_carlo() {
        let dev = <B as Backend>::Device::default();
        let (n, din, dh, k) = (2usize, 4usize, 5usize, 150_000usize);
        let std = 0.6f64;
        let weights = vec![
            0.8f32, -0.3, 0.6, 0.1, -0.7, 0.2, 0.9, -0.4, 0.5, 0.3, -0.5, 0.4, 0.7, -0.8, 0.2, 0.3,
            -0.6, 0.2, 0.4, 0.9,
        ];
        let means = vec![0.0f32, 0.0, 0.0, 0.0, 0.3, -0.2, 0.4, -0.1];
        let w = Tensor::<B, 2>::from_data(TensorData::new(weights.clone(), [din, dh]), &dev);
        let mean = Tensor::<B, 2>::from_data(TensorData::new(means.clone(), [n, din]), &dev);
        let var = Tensor::<B, 2>::full([n, din], std * std, &dev);

        let pre = propagate_linear_full(
            &MomentsFull::from_diagonal(mean.clone(), var),
            w.clone(),
            None,
        );
        let post = propagate_relu_full(&pre);
        let p_cov = post.cov.to_data().to_vec::<f32>().unwrap(); // [n, dh, dh]

        let mut sh = vec![0.0f64; n * dh];
        let mut shh = vec![0.0f64; n * dh * dh];
        // Local seeded noise is independent of Burn's shared RNG and test order.
        let mut state = 0x5eed_c0a1_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        };
        for _ in 0..k {
            let x: Vec<f64> = means
                .iter()
                .map(|&mu| {
                    let z = (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();
                    mu as f64 + std * z
                })
                .collect();
            let mut h = vec![0.0f64; n * dh];
            for i in 0..n {
                for a in 0..dh {
                    h[i * dh + a] = (0..din)
                        .map(|j| x[i * din + j] * weights[j * dh + a] as f64)
                        .sum::<f64>()
                        .max(0.0);
                }
            }
            for i in 0..n {
                for a in 0..dh {
                    let ha = h[i * dh + a];
                    sh[i * dh + a] += ha;
                    for b in 0..dh {
                        shh[i * dh * dh + a * dh + b] += ha * h[i * dh + b];
                    }
                }
            }
        }
        let kf = k as f64;
        for i in 0..n {
            for a in 0..dh {
                for b in 0..dh {
                    let mc = shh[i * dh * dh + a * dh + b] / kf
                        - (sh[i * dh + a] / kf) * (sh[i * dh + b] / kf);
                    let p = p_cov[i * dh * dh + a * dh + b] as f64;
                    assert!((p - mc).abs() < 0.03, "cov[{i}][{a}][{b}]: {p} vs MC {mc}");
                }
            }
        }
    }
}
