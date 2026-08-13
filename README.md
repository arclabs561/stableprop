# stableprop

Propagate uncertainty through neural networks analytically.

Given a Gaussian (or Cauchy) over a network's inputs, `stableprop` pushes its
moments through linear, ReLU, leaky-ReLU, and GCN-adjacency layers and returns
the output mean and (co)variance. It targets the case where Monte Carlo or
ensembles are the only alternative: **regression / surrogate models with known
input uncertainty**.

## What it's good for (and not)

In the seeded synthetic MLP example, the analytic error bars agree with a
200-sample Monte Carlo estimate (`Pearson r = 0.80` on per-point standard
deviation, mean magnitude ratio `1.06`, empirical 95% interval coverage
`0.93`) using one propagated forward pass instead of 200 sampled passes. This
is one comparison, not a general accuracy or calibration result.

It is **not** a classification uncertainty / OOD detector: for that, the model's
own softmax confidence is a strong free baseline that this does not beat. The
honest niche is propagating *known input uncertainty* through regressors.

## Usage

```toml
[dependencies]
stableprop = { version = "0.3", features = ["burn"] }
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

Full gallery with commands and captured output: [`examples/README.md`](examples/README.md).

- `regression_intervals`: sampling-free error bars vs Monte Carlo.
- `conformal_intervals`: use the analytic standard deviation as a split-conformal
  scale. Under exchangeability, this targets finite-sample marginal coverage;
  realized coverage on a particular test split can differ.
- `robust_training`: train *with* the differentiable propagated variance to
  reduce error under input noise (shared-init A/B vs plain MSE).
- `misclassification_risk`: full-covariance propagation of input noise into an
  analytic estimate of a classifier's error rate (tracks Monte Carlo closely;
  an estimate, not a guaranteed certificate).
- `cora_uncertainty`: honest evidence on classification, where the method is
  dominated by the softmax baseline.

## What it propagates

- Diagonal Gaussian moments (`Moments`): exact linear, Frey-Hinton ReLU,
  leaky-ReLU, 2-D convolution, GCN-adjacency, residual-add.
- Full covariance (`MomentsFull`): keeps cross-feature correlations through
  affine and ReLU layers. The ReLU uses exact univariate moments on the diagonal
  and a truncated Wright-series calculation for off-diagonal covariance; tests
  compare both parts with Monte Carlo on correlated Gaussian inputs.
- Weight uncertainty (`propagate_linear_bayes`): epistemic propagation in the
  style of Probabilistic Backpropagation / Deterministic Variational Inference.
- Cauchy (`Cauchy`): the heavy-tailed stable distribution (no moments; location
  and scale are propagated), for heavy-tailed robustness.

Tests use closed-form identities, invariants, property checks, and Monte Carlo
oracles for the main Gaussian affine and activation paths. The examples provide
additional empirical comparisons for full networks.

## Background

The method is moment / stable-distribution propagation; see Frey & Hinton (1999)
for the rectified-Gaussian ReLU moments, Hernandez-Lobato & Adams (2015) and
Wu et al. (2019) for weight-uncertainty propagation, and Petersen et al.
(ICLR 2024, "Uncertainty Quantification via Stable Distribution Propagation")
for the Gaussian/Cauchy stable-distribution framing.

## Roadmap

Attention layers are not yet implemented (moments through softmax and uncertain
query-key products are a research problem, not a clean addition). The default
residual-add assumes independent branches; `propagate_residual_add_correlated`
accepts diagonal skip-branch covariance when it is available. The
misclassification-risk estimate is an estimate, not a sound certificate; rigorous
certified bounds would need interval / Lipschitz methods.

## License

MIT OR Apache-2.0.
