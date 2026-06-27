# stableprop

Propagate a distribution through a neural network analytically, to get output
uncertainty in one forward pass instead of Monte Carlo sampling.

Given a Gaussian (or Cauchy) over a network's inputs, `stableprop` pushes its
moments through linear, ReLU, leaky-ReLU, and GCN-adjacency layers and returns
the output mean and (co)variance. It targets the case where Monte Carlo or
ensembles are the only alternative: **regression / surrogate models with known
input uncertainty**.

## What it's good for (and not)

On an MLP regressor with known per-point input noise, the analytic error bars
match a 200-sample Monte Carlo estimate (`Pearson r = 0.81` on the per-point
std, magnitude ratio `0.96`, 90% interval coverage `0.90`) in **one** forward
pass instead of 200. There is no softmax baseline for regression, so this is a
real win over sampling.

It is **not** a classification uncertainty / OOD detector: for that, the model's
own softmax confidence is a strong free baseline that this does not beat. The
honest niche is propagating *known input uncertainty* through regressors.

## Usage

```toml
[dependencies]
stableprop = { version = "0.1", features = ["burn"] }
```

```rust
use stableprop::burn_sdp::{propagate_linear, propagate_relu, Moments};

// mean [n, d_in], input variance [n, d_in]
let m0 = Moments::new(mean, var);
let m1 = propagate_relu(&propagate_linear(&m0, w1, b1));
let m2 = propagate_linear(&m1, w2, b2);
// m2.mean, m2.var are the analytic output moments
```

See `examples/`:

- `regression_intervals`: sampling-free error bars vs Monte Carlo (the flagship).
- `conformal_intervals`: wrap the analytic std in split-conformal for a
  distribution-free coverage *guarantee* (the raw intervals are a heuristic scale;
  conformal makes them calibrated).
- `robust_training`: train *with* the differentiable propagated variance to
  reduce error under input noise (shared-init A/B vs plain MSE).
- `misclassification_risk`: full-covariance propagation of input noise into an
  analytic estimate of a classifier's error rate (tracks Monte Carlo closely;
  an estimate, not a guaranteed certificate).
- `cora_uncertainty`: honest evidence on classification, where the method is
  dominated by the softmax baseline.

## What it propagates

- Diagonal Gaussian moments (`Moments`): exact linear, Frey-Hinton ReLU,
  leaky-ReLU, GCN-adjacency, residual-add.
- Full covariance (`MomentsFull`): keeps the cross-feature correlations a layer
  introduces; more accurate than diagonal (validated against Monte Carlo). The
  ReLU uses exact diagonal moments with a smooth `Phi(alpha)` gate on the
  off-diagonal, which avoids the hard-gate decision-boundary brittleness of the
  local-linearization method it is based on.
- Weight uncertainty (`propagate_linear_bayes`): epistemic propagation in the
  style of Probabilistic Backpropagation / Deterministic Variational Inference.
- Cauchy (`Cauchy`): the heavy-tailed stable distribution (no moments; location
  and scale are propagated), for heavy-tailed robustness.

Every propagation rule has a Monte-Carlo cross-check in the test suite.

## Background

The method is moment / stable-distribution propagation; see Frey & Hinton (1999)
for the rectified-Gaussian ReLU moments, Hernandez-Lobato & Adams (2015) and
Wu et al. (2019) for weight-uncertainty propagation, and Petersen et al.
(ICLR 2024, "Uncertainty Quantification via Stable Distribution Propagation")
for the Gaussian/Cauchy stable-distribution framing.

## Roadmap

Convolutional and attention layers are not yet implemented. The residual-add is
the independence approximation (it ignores the skip-branch covariance). The
misclassification-risk estimate is an estimate, not a sound certificate; rigorous
certified bounds would need interval / Lipschitz methods.

## License

MIT OR Apache-2.0.
