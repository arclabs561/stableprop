# stableprop

Propagate uncertainty through neural networks analytically.

Propagate Gaussian moments through supported neural-network layers. The
optional Burn backend also propagates Cauchy locations and scales. Compose
the layer functions to match your model's forward pass.

Inspired by [distprop](https://github.com/Felix-Petersen/distprop) and
[Petersen et al. (ICLR 2024)](https://arxiv.org/abs/2402.08324). The Gaussian
implementation here uses moment matching; distprop uses local linearization.
The [method guide](docs/methods.md) explains that distinction, the research
history, and which applications each approach supports.
The [derivations](docs/derivations.md) give the moment formulas, covariance-series
proofs, and numerical assumptions.

## What uncertainty means here

You supply Gaussian means with variances or covariance, or locations and scales
for independent Cauchy inputs. You can also supply independent weight variances
from another model. stableprop estimates the resulting output moments or scales;
it does not infer those distributions from data.

| Related method | Its job | Where stableprop fits |
| --- | --- | --- |
| [Gaussian embeddings](docs/methods.md#uncertainty-sources-and-downstream-methods) | Learn a distribution for each representation | Propagate supplied embedding moments through supported layers |
| [Bayesian models](docs/methods.md#how-the-methods-developed) | Learn parameter uncertainty from observations | Propagate supplied moments; posterior fitting and updating are external |
| Contrastive learning | Train representations using pair relationships | Add a differentiable sensitivity penalty, as in [`tuplet_contrastive`](examples/tuplet_contrastive.rs) |
| [Conformal prediction](docs/methods.md#calibration-of-prediction-intervals) | Calibrate prediction sets using held-out observations | Supply an input-dependent scale for [`conformal_intervals`](examples/conformal_intervals.rs), with calibration ranks from [statskit](https://github.com/arclabs561/statskit) |

Input sensitivity and parameter uncertainty arise from different random
quantities. A stable score can still be poorly learned; a well-learned model
can still be sensitive to noisy measurements. See the
[method guide](docs/methods.md#uncertainty-sources-and-downstream-methods) for the distinction.
The [selection note](docs/sensitivity-and-selection.md) connects score covariance
to exploration value and augmentation disagreement to training-data selection.

## Start with a small network

The published default API uses `f64` vectors, has no runtime dependencies, and
supports Rust 1.80.

```toml
[dependencies]
stableprop = "0.5.2"
```

The input standard deviations describe independent Gaussian features. Affine
layers retain covariance; this API drops off-diagonal covariance at each ReLU.

```rust
use stableprop::{propagate_sequential, Layer};

let layers = [
    Layer::Linear {
        weight: vec![vec![1.0, -1.0]], // [output, input]
        bias: vec![0.0],
    },
    Layer::ReLU,
];
let input_mean = [0.0, 0.0];
let input_std = [0.3, 0.4];
let output = propagate_sequential(&layers, &input_mean, &input_std);
println!("mean = {:.4}, variance = {:.4}", output.mean[0], output.cov[0][0]);
```

```text
mean = 0.1995, variance = 0.0852
```

Run this example from the checkout:

```sh
cargo run --release --example basic
```

## Burn models

The Burn tensor integration is development API, not a published crate release.
It requires Rust 1.95 or newer. Use stableprop's `main` branch with the same
Burn revision when importing `burn::tensor::Tensor`:

```toml
[dependencies.stableprop]
git = "https://github.com/arclabs561/stableprop"
branch = "main"
features = ["burn"]

[dependencies.burn]
git = "https://github.com/tracel-ai/burn"
rev = "1414c8a14e5169ef5e5fc67f9b8ab01a25d6352d"
default-features = false
features = ["std", "flex", "autodiff"]
```

Burn weights use `[input, output]`, the transpose of the vector API's layout.
CPU examples use `Device::flex()`; call `.autodiff()` when differentiating.
On macOS, `Device::metal(DeviceKind::DefaultDevice)` selects Metal. Burn selects
tensor precision at creation. For `f64`, pass `(&device, DType::F64)` to tensor
constructors. Keep moments, weights and biases on the same runtime device
and use the same floating dtype. Propagation preserves that dtype.

| Representation | What it tracks | Main approximation |
| --- | --- | --- |
| [`burn_sdp::Moments`](src/burn_sdp.rs) | Mean and variance, each `[batch, features]` | Drops feature and row correlations |
| [`burn_sdp::MomentsFull`](src/burn_sdp.rs) | Mean `[batch, features]`; covariance `[batch, features, features]` | No cross-row covariance; Gaussian layer inputs; [third-order ReLU covariance series](docs/derivations.md#relu-coefficients-and-the-implemented-order) |
| [`burn_sdp::Cauchy`](src/burn_sdp.rs) | Location and scale, each `[batch, features]` | Drops dependence; local ReLU gate |

The tensor API also includes leaky ReLU, diagonal convolution, fixed left
matrix multiplication, residual addition, and affine propagation with supplied
weight variances. Cross-covariance helpers propagate supplied within-row
covariance through affine and Gaussian ReLU steps. Pass the resulting diagonal
to `propagate_residual_add_correlated` for the residual cross term. The
[source documentation](src/burn_sdp.rs) describes the development API.

On macOS, `features = ["metal"]` enables Burn's WGPU Metal backend with
operation fusion. Run `just metal-train` for the training example or
`just metal-test` for CPU/GPU value and gradient comparisons and synchronized
timings. These checks use `f32`; small workloads can be faster on CPU.

## Try an application

| I want to… | Start here |
| --- | --- |
| Separate input noise from supplied weight uncertainty | [uncertainty_sources](examples/uncertainty_sources.rs) |
| Estimate whether noisy query features change a ranking | [pairwise_ranking_risk](examples/pairwise_ranking_risk.rs) |
| Compare output uncertainty with sampled noisy inputs | [regression_intervals](examples/regression_intervals.rs) |
| Propagate an external state posterior through a learned surrogate | [kalman_sensor_intervals](examples/README.md#state-posteriors-and-derived-targets) |
| Calibrate prediction intervals against held-out labels | [conformal_intervals](examples/conformal_intervals.rs) |
| Evaluate intervals on grouped real measurements | [grouped_intervals](examples/README.md#grouped-measurements) |
| Choose settings under execution noise | [robust_selection](examples/README.md#candidate-choice-under-execution-noise) |
| Train with an output-variance penalty | [robust_training](examples/robust_training.rs) |
| Measure the effect of retaining covariance | [full_covariance](examples/full_covariance.rs) |
| Derive a residual branch's covariance with its input | [correlated_residual](examples/correlated_residual.rs) |
| Test sensitivity as a data-selection score | [active_selection](examples/active_selection.rs) |
| Regularize contrastive embeddings | [tuplet_contrastive](examples/tuplet_contrastive.rs), using [tuplet](https://github.com/arclabs561/tuplet)'s Burn loss |
| Propagate node-feature noise through a GCN | [gcn_uncertainty](examples/gcn_uncertainty.rs), using [ricci](https://github.com/arclabs561/ricci) |

The [example guide](examples/README.md) has commands, output interpretation,
and the remaining classification and heavy-tail comparisons.

## Limits

Affine moments are exact for the represented covariance. ReLU uses closed-form
Gaussian marginal moments with numerical tail handling, but the output
distribution is not Gaussian. Repeating moment matching through a network is an approximation.
Full covariance reduces information loss at quadratic memory cost; its ReLU
off-diagonal terms are truncated to third order.

Propagated input noise does not account for label noise, model bias, or an
unknown weight posterior. Supplying weight variances does not fit that
posterior. Neither a variance penalty nor a misclassification-risk estimate is
an adversarial robustness certificate. Validate interval coverage separately;
split conformal gives marginal coverage under exchangeability, not a guarantee
for every input or a shifted deployment distribution.

Cauchy distributions have no finite mean or variance. Layerwise Cauchy scales
discard dependence, and ReLU gating can collapse a marginal to zero. A Cauchy
output interval is an approximation after nonlinear propagation.

## Checks

Use Rust 1.95 or newer for repository development. The Rust 1.80 floor applies
to consumers of the default library; checking this checkout also resolves
Burn's Git workspace. CI checks the default API through an isolated consumer.
Install [just](https://github.com/casey/just#installation), then run:

```sh
just check
```

This runs formatting, lints, tests, strict documentation builds, example builds
and smoke runs, and benchmark correctness checks. Run `just` to list individual
recipes, or read the [justfile](justfile)
for the Cargo commands used by CI.

The API docs include executable recipes for propagation, tensor precision,
autodiff, and correlated residuals. The [reference generator](scripts/README.md)
uses independent numerical integration; `just reference-check` regenerates its
Gaussian pair values and marginal derivatives, then checks the frozen fixtures.

Use `just bench` or `just bench-burn` for repeatable CPU measurements; the
[benchmark guide](benches/README.md) explains fixtures and timing boundaries.

## License

MIT OR Apache-2.0.
