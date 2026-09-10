# Working on stableprop

Read [README.md](README.md) for the API, [docs/methods.md](docs/methods.md)
for mathematical assumptions and provenance, and
[examples/README.md](examples/README.md) before changing examples.

## Implementation

- `src/lib.rs` is the dependency-free `f64` API. Its ReLU step discards
  off-diagonal covariance. Keep backend dependencies behind the `burn` feature.
- `src/burn_sdp.rs` contains differentiable Burn operations. Full covariance
  covers features within each batch row, not dependence between batch rows.
- Distinguish exact affine moments, Gaussian activation moments, covariance
  series truncation, and the repeated Gaussian approximation across layers.
  The Cauchy path carries locations and scales, not means and variances.
- Preserve positive small variances. Numerical fixes must retain finite
  gradients at zero variance, in the tails, and for correlated inputs.
- Covariance inputs must be symmetric positive semidefinite. Shape checks do
  not establish that condition. Document caller assumptions and avoid hidden
  tensor-to-host transfers for validation.

## Validation

[CI](.github/workflows/ci.yml) defines the baseline checks and compiler matrix.
Before committing implementation changes, run:

```sh
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo build --release --examples --all-features
```

Run changed examples using the commands in their guide. For numerical changes,
check the relevant mathematical invariant or independent reference, including
autodiff when changing Burn operations. Use local seeded sampling for Monte
Carlo regression tests; a shared backend RNG can couple parallel tests.

## Documentation and delivery

- Keep the README focused on use. Method history belongs in `docs/methods.md`;
  acquisition and learning connections belong in
  `docs/sensitivity-and-selection.md`.
- Use concise technical prose. State assumptions alongside claims. Distinguish
  sensitivity, parameter uncertainty, calibration, and acquisition value.
  Report example outcomes with their experimental conditions.
- Verify research claims against primary sources. Distinguish publication
  dates, revisions, and preprints. Visually check rendered GitHub math after
  changing equations, and check local links when moving documentation.
- Keep examples portable; document external data and feature requirements.
  Add behavioral changes to `CHANGELOG.md` under Unreleased.
- Use descriptive Conventional Commit subjects such as `fix: stabilize ReLU
  gradients` or `docs: explain covariance assumptions`. A repository-name
  prefix adds no useful scope here.
