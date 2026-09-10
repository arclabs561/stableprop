# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-10

### Added

- A `metal` feature for Burn's fused WGPU Metal backend, a GPU training example,
  and CPU/Metal value and gradient comparisons.
- A justfile shared with CI, including explicit local Metal checks and timings.

### Changed

- Update the optional tensor API to Burn 0.21, requiring Rust 1.92. Burn 0.20
  tensors are incompatible; the default vector API still supports Rust 1.80.
- Update the GCN and contrastive examples to ricci 0.10 and tuplet 0.3.
  The GCN moment pass adds bias after adjacency aggregation, matching ricci.
- Replace unseeded tensor Monte Carlo checks for exact operations with
  deterministic analytical references. Retain seeded distributional checks.
- Add full-covariance and ReLU moment properties, and distinguish analytical
  gradients at positive variance from deterministic boundary conventions.

### Fixed

- Reject standard deviations whose squared variance underflows to zero.
- Reject broadcastable affine bias, Bayesian weight/bias, and convolution
  moment shapes that violate the documented tensor layout.
- Remove common-logit offsets from Cora input-noise and weight-uncertainty
  scores; reject citation edges whose endpoints are absent from the dataset.
- Correct the regression example guide's interval level and clarify Gaussian
  information-gain assumptions.

## [0.4.0] - 2026-09-10

### Added

- Affine and Gaussian ReLU cross-covariance transport for supported branches,
  with a `correlated_residual` example and Monte Carlo and autodiff tests.
- An `active_selection` experiment comparing entropy, random selection, and
  analytic and sampled centered-logit disagreement under shared controls.

### Fixed

- Preserve tiny positive Gaussian variances and avoid cancellation in ReLU
  variance at large positive means, including the Burn and full-covariance paths.
- Keep full-covariance gradients finite at tiny noise scales and apply ReLU
  tail limits consistently to marginal and cross-covariance terms.
- Reject malformed vector inputs and mismatched tensor moment shapes.
- Reject residual batch broadcasting and leaky-ReLU coefficients that overflow
  the backend's scalar type.
- Train the Cora novel-class baseline with known classes only, and compare
  abstention with the exact expected accuracy of random selection.
- Seed example training and sampling; evaluate contrastive embeddings on held-out
  points with shared noise. Accept an explicitly supplied Cora data path.

### Changed

- Explain moment matching versus distprop's local linearization, covariance
  approximations, research history, and application limits in the documentation.
- Refresh example interpretation and add `basic`, `uncertainty_sources`, and
  `pairwise_ranking_risk` examples.
- Check library compiler floors, tensor documentation, and example builds in CI.

## [0.3.1] - 2026-06-27

### Changed

- Full-covariance ReLU now uses the Wright et al. (2024) covariance series to
  3rd order for the off-diagonal terms, replacing the first-order gate.

### Added

- Property tests (proptest) on the reference propagation.
- `full_covariance` and `cauchy_tails` examples.
- CI workflow (fmt, clippy, tests, both feature sets).

## [0.3.0] - 2026-06-27

### Added

- `propagate_conv2d`: exact diagonal-Gaussian propagation through a 2-D
  convolution (`var_out = conv(var, w^2)`), validated against Monte Carlo.

## [0.2.0] - 2026-06-27

### Added

- `propagate_leaky_relu`: exact Gaussian moments through leaky ReLU.
- `propagate_residual_add`: residual skip + branch combination (independence
  approximation; skip-branch covariance is not represented by this API).
- `robust_training` example: training with the differentiable propagated variance
  as a loss term, with a shared-initialization comparison under input noise.
- `misclassification_risk` example: full-covariance propagation of input noise
  into an analytic estimate of a classifier's error rate, compared with
  Monte Carlo (not a guaranteed certificate).

## [0.1.0] - 2026-06-27

### Added

- Diagonal Gaussian moment propagation: linear, ReLU (Frey-Hinton), GCN-adjacency.
- Full-covariance propagation (`MomentsFull`): exact linear, smooth-gated ReLU;
  compared with diagonal propagation and Monte Carlo in tests.
- Weight-uncertainty (Bayesian) linear propagation (`propagate_linear_bayes`).
- Cauchy stable-distribution propagation (`Cauchy`).
- Examples: `regression_intervals`, `conformal_intervals`, `cora_uncertainty`,
  `gcn_uncertainty`.

[Unreleased]: https://github.com/arclabs561/stableprop/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/arclabs561/stableprop/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/arclabs561/stableprop/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/arclabs561/stableprop/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/arclabs561/stableprop/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/arclabs561/stableprop/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/arclabs561/stableprop/tree/v0.1.0
