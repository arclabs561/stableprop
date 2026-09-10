# Benchmarks

Run from the repository root:

```sh
just bench
just bench-burn
```

`bench` measures the `f64` vector API on square and rectangular affine maps,
plus ReLU as a separate workload. `bench-burn` measures diagonal and full
affine propagation using Burn's `f32` NdArray backend. Their precision and
input validation differ; these suites are not a direct backend comparison.

Fixtures and scalar reference calculations run outside the timed loop.
Measurements include the public propagation call, its output allocation and
destruction, and any tensor clones needed by that call. Full-covariance
fixtures include signed correlations; the diagonal oracle uses only their
marginal variances. `just bench-check` runs each fixture once without collecting
timing samples and is included in `just check`.

Criterion accepts filters and saved baselines through the recipes:

```sh
just bench reference_affine --save-baseline before
# After changing the implementation:
just bench reference_affine --baseline before
```

Repeat measurements on an otherwise idle machine. Retain adjacent shapes and
the ReLU workload as controls when optimizing affine propagation. Short runs
are useful for checking the harness; small timing differences need longer
measurements and repeat runs. See Criterion's
[timing-loop documentation](https://criterion-rs.github.io/book/user_guide/timing_loops.html).

On macOS, `just metal-test` separately checks CPU/Metal values and gradients
and prints synchronized workload timings. Each workload is warmed up first.
Measurements include tensor allocation; host result validation runs afterward.
These are local diagnostics, not Criterion comparisons.
