# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Reject mismatched floating dtypes in Burn distribution constructors, before
  they reach backend arithmetic.
- Preserve unbounded conformal intervals when a calibration set is too small
  for the requested finite-sample rank.

### Added

- Candidate selection under Gaussian execution noise, comparing a trained
  surrogate's predicted losses with exact quadratic-simulator losses.
- Grouped real-data interval comparisons on Airfoil and Parkinsons Telemonitoring,
  with group-disjoint fitting and calibration, Monte Carlo model checks, and
  constant-width baselines.
- An exact-moment ReLU ranking control that isolates Gaussian margin-shape error.
- Conformal diagnostics separating covariance approximation, sampled model
  variation, and synthetic target noise and bias.
- Ranking evaluations across generated model/candidate fixtures, with paired
  deferral comparisons and explicit noise-regime weighting.
- Mean-agreement, risk-bin, covariance-depth, embedding-scale, and selected-class
  diagnostics in the examples, plus a quick Cora workflow.
- Opt-in reference-generator checks against frozen Gaussian-pair and marginal
  fixtures.
- Exact-input f32 marginal references and autodiff checks near formula switches
  and at extreme scales; generated directions in covariance PSD properties.

## [0.5.2] - 2026-09-11

### Fixed

- Expand both standard-deviation tensors before masked selection in
  full-covariance ReLU. This avoids partially evaluated broadcast outputs in
  Burn 0.21 GPU kernels and non-finite gradients on larger batches.

### Added

- Executable API recipes for Gaussian propagation, explicit tensor precision,
  full-covariance autodiff, and correlated residual addition.
- A reproducible Gaussian pair reference generator and independent checks of
  composed moment gradients with respect to means, covariance, and weights.
- Repeated conformal-interval and ranking-deferral studies with separate
  evaluation draws and uncertainty summaries across trials.
- A Metal full-covariance forward/backward workload with shared CPU/GPU inputs
  and gradient comparisons.

## [0.5.1] - 2026-09-11

### Fixed

- Validate example variances before computing intervals or risk, and handle
  deterministic margins and undefined comparison statistics explicitly.
- Materialize Burn example parameters before cloning baselines or changing RNG
  seeds, so training comparisons share their intended initial weights.
- Avoid cancellation in negative Gaussian ReLU tails, including Burn means,
  variances, derivatives, and cross-covariance gates. Replace the vector API's
  central CDF approximation with a convergent series.
- Preserve the input dtype in full-covariance helpers, and check leaky-ReLU
  coefficients against the actual tensor dtype under Burn 0.21.
- Bound ReLU standardization before division to keep tail gradients finite
  when the mean is large relative to a tiny variance.
- Order full-covariance normalization by feature scale to avoid intermediate
  gradient overflow for correlated inputs with very different variances.

### Changed

- Reuse Gaussian density and safe standard deviation tensors in full-covariance
  ReLU propagation. Add benchmarks for propagation followed by backpropagation.
- Accumulate the vector API's covariance product along contiguous rows.
- Benchmark central and negative-tail Burn ReLU workloads, and check tail
  values and gradients against high-precision references on CPU and Metal.
- Evaluate the full-covariance ReLU series in Horner form, using two fewer
  pairwise multiplications.
- Check feature permutation, batch partitioning, correlated residual identities,
  and the centered Gaussian ReLU covariance remainder with property tests.
- Compare covariance propagation across depths and seeds, with score-margin
  errors, an independent Monte Carlo repeat, and a scalar closure control.

### Added

- A derivation reference for Gaussian moments, cross-covariance, the Hermite
  series, its PSD property and error bound, and numerical design choices.
- Nonzero-mean Gaussian pair references, independent feature-scale properties,
  and covariance-gradient checks against the finite series and an independent
  exact-derivative error bound. `just property-test` runs
  the algebraic and Gaussian-reference properties with a larger sample count.

## [0.5.0] - 2026-09-10

### Added

- A `metal` feature for Burn's fused WGPU Metal backend, a GPU training example,
  and CPU/Metal value and gradient comparisons.
- A justfile shared with CI, including explicit local Metal checks and timings.
- Criterion benchmarks for vector and Burn affine propagation, with independent
  covariance references checked by CI.
- Burn properties for rectangular affine maps, cross-covariance composition,
  ReLU scaling and reflection, and covariance symmetry and signed quadratic forms.

### Changed

- Update the optional tensor API to Burn 0.21, requiring Rust 1.92. Burn 0.20
  tensors are incompatible; the default vector API still supports Rust 1.80.
- Update the GCN and contrastive examples to ricci 0.10 and tuplet 0.3.
  The GCN moment pass adds bias after adjacency aggregation, matching ricci.
- Replace unseeded tensor Monte Carlo checks for exact operations with
  deterministic analytical references. Retain seeded distributional checks.
- Add full-covariance and ReLU moment properties, and distinguish analytical
  gradients at positive variance from deterministic boundary conventions.
- Use contiguous weight rows in the vector API's covariance transport, avoiding
  a transpose allocation.
- Report output covariance error in the full-covariance example, alongside
  marginal standard-deviation errors.

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

[Unreleased]: https://github.com/arclabs561/stableprop/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/arclabs561/stableprop/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/arclabs561/stableprop/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/arclabs561/stableprop/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/arclabs561/stableprop/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/arclabs561/stableprop/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/arclabs561/stableprop/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/arclabs561/stableprop/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/arclabs561/stableprop/tree/v0.1.0
