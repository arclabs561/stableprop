# Benchmarks

Run from the repository root:

```sh
just bench
just bench-burn
```

`bench` measures the `f64` vector API on square and rectangular affine maps,
plus ReLU as a separate workload. `bench-burn` measures diagonal and full
affine and ReLU propagation using Burn's `f32` NdArray backend. ReLU cases
separate central inputs from negative tails (`mean / std = -7`), where
accurate small moments need different numerical formulas. Their precision and
input validation differ; these suites are not a direct backend comparison.

Fixtures and affine scalar reference calculations run outside the timed loop.
Measurements include the public propagation call, its output allocation and
destruction, and any tensor clones needed by that call. Full-covariance
fixtures include signed correlations; the diagonal oracle uses only their
marginal variances. `just bench-check` runs each fixture once without collecting
timing samples and is included in `just check`.

`bench-burn` also includes `burn_relu_full_backward_f32`: central and
negative-tail full-covariance ReLU workloads at batch/width 8/16 and 64/64.
Each iteration creates fresh tracked leaves from prebuilt, untracked tensors,
propagates both moments, reduces them to a scalar loss, and runs `backward`.
Tensor allocation, graph construction, and result destruction are included.
Fixture creation, the analytic gradient check, and host readback are outside
the timed closure.

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

The full-covariance backward workload composes affine propagation and ReLU,
then differentiates the sum of output means and covariances. CPU and Metal
start from the same host values. Input upload is outside timing; each of three
synchronized repeats includes graph construction and intermediate allocation.
Mean, covariance, and weight gradients are compared after measurement.

For a matched deterministic affine–ReLU comparison, run `just metal-profile`.
This uses identical input means, independent variances, weights and biases for
diagonal and full propagation. Biases place the ReLU inputs at standardized
means of -1, 0 and 1, or -7 for the tail workload. Forward checks compare both
moments; backward checks compare input, weight and bias gradients of the sum of
output means and marginal variances. The full path still computes its dense
covariance. It alternates CPU-first and Metal-first execution over three warmed,
synchronized repeats at batch/width 8/16 and 64/64.

The ignored test accepts these environment filters; each defaults to `both`:

| Variable | Values besides `both` |
| --- | --- |
| `STABLEPROP_PROFILE_BACKEND` | `cpu`, `metal` |
| `STABLEPROP_PROFILE_SHAPE` | `8x16`, `64x64` |
| `STABLEPROP_PROFILE_REPRESENTATION` | `diagonal`, `full` |
| `STABLEPROP_PROFILE_PASS` | `forward`, `backward` |
| `STABLEPROP_PROFILE_REGIME` | `central`, `tail` |

For peak process memory, build first, then wrap the printed test executable in
`/usr/bin/time -l`, selecting one backend and workload per fresh process. Pass
`metal_matched_diagonal_full_forward_backward_timings --ignored --nocapture
--test-threads=1` to that executable. A CPU-only run never initializes Metal;
a diagonal-only run never allocates the full covariance fixture. Repeat fresh
processes and report the range. Peak RSS includes initialization, fixtures,
warmup and validation; it is not a measurement of GPU device memory or the
propagation buffers alone. Timing excludes host uploads and result readback,
but includes intermediate allocation and synchronization.
