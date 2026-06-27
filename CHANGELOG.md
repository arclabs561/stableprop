# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-27

### Added

- `propagate_leaky_relu`: exact Gaussian moments through leaky ReLU.
- `propagate_residual_add`: residual skip + branch combination (independence
  approximation; exact when the branch is small relative to the skip).
- `robust_training` example: training with the differentiable propagated variance
  as a loss term, reducing error under input noise.
- `misclassification_risk` example: full-covariance propagation of input noise
  into an analytic estimate of a classifier's error rate (an estimate that tracks
  Monte Carlo, not a guaranteed certificate).

## [0.1.0] - 2026-06-27

### Added

- Diagonal Gaussian moment propagation: linear, ReLU (Frey-Hinton), GCN-adjacency.
- Full-covariance propagation (`MomentsFull`): exact linear, smooth-gated ReLU;
  more accurate than diagonal, validated against Monte Carlo.
- Weight-uncertainty (Bayesian) linear propagation (`propagate_linear_bayes`).
- Cauchy stable-distribution propagation (`Cauchy`).
- Examples: `regression_intervals`, `conformal_intervals`, `cora_uncertainty`,
  `gcn_uncertainty`.
