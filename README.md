# stableprop

Propagate uncertainty through neural networks analytically.

Supply input means and uncertainty, then propagate them through supported
layers using your model's weights. stableprop estimates the resulting output
means and variances without repeatedly sampling the noisy inputs. The Burn
API is differentiable, so these estimates can also be used in a training loss.

## What uncertainty means here

You choose the input distribution: for example, Gaussian measurement noise,
an upstream state estimate, or a perturbation scale for sensitivity analysis.
stableprop propagates that distribution; it does not learn it from data.
Weight uncertainty can also be supplied by an external model.

The output describes variation under those assumptions. Input sensitivity,
uncertainty about learned parameters, and confidence in an observed target
are different quantities. The [method guide](docs/methods.md#uncertainty-sources-and-downstream-methods)
explains how they relate.

## Start with a small network

For an application using Rust 1.80 or newer, add the published vector API to
`Cargo.toml`. It uses `f64` and has no runtime dependencies.

```toml
[dependencies]
stableprop = "0.5.2"
```

This computes `ReLU(x₁ − x₂)` for two independent, zero-mean Gaussian inputs
with standard deviations 0.3 and 0.4. Put the following inside `fn main()` in
your application and run `cargo run --release`:

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

The output mean is positive because ReLU clips negative values to zero. The
variance describes variation in the model output under the supplied input
noise; it does not establish prediction-interval coverage for observed targets.

From a repository checkout, use Rust 1.95 or newer and run the same example:

```sh
cargo run --release --example basic
```

## Burn models

The current tensor API uses an unreleased Burn revision and requires Rust 1.95
or newer. These Git revisions are tested together; update them together:

```toml
[dependencies.stableprop]
git = "https://github.com/arclabs561/stableprop"
rev = "9c64e3d3b97536022e195eff3dd66de4cf6f784f"
features = ["burn"]

[dependencies.burn]
git = "https://github.com/tracel-ai/burn"
rev = "1414c8a14e5169ef5e5fc67f9b8ab01a25d6352d"
default-features = false
features = ["std", "flex", "autodiff"]
```

Follow the [short integration recipe](examples/README.md#use-your-own-burn-model)
to reuse a Burn layer's weights. Propagation is a separate sequence of calls
matching the supported operations in your forward pass; it does not convert
an arbitrary model automatically.

| Representation | Supplied uncertainty | What it retains |
| --- | --- | --- |
| [`Moments`](src/burn_sdp.rs) | Gaussian means and **variances**, each `[batch, features]` | Marginal moments; discards feature covariance |
| [`MomentsFull`](src/burn_sdp.rs) | Means `[batch, features]` and covariance `[batch, features, features]` | Feature covariance, with a [third-order ReLU approximation](docs/derivations.md#relu-coefficients-and-the-implemented-order) |
| [`Cauchy`](src/burn_sdp.rs) | Locations and scales, each `[batch, features]` | Heavy-tailed marginals; discards dependence and uses local ReLU gating |

Each batch row represents a separate distribution; none of these types stores
cross-row covariance. Burn weights use `[input, output]`, the transpose of the
vector API's layout. Keep operands on the same device and in the same floating
dtype. For explicit `f64`, pass `(&device, DType::F64)` to tensor constructors.

CPU examples use `Device::flex()`; call `.autodiff()` for gradients. On macOS,
`features = ["metal"]` enables WGPU Metal with operation fusion. Use `just metal-test` for GPU value and gradient checks, or `just metal-train`
for a training example.
The [API source](src/burn_sdp.rs) documents supported operations and their assumptions.

## Try an application

These examples use `--features burn`. The linked guide sections give commands,
data requirements, and help interpreting the results.

| I want to… | Start here |
| --- | --- |
| Distinguish input noise from parameter uncertainty, or estimate ranking flips | [Uncertainty sources and ranking decisions](examples/README.md#uncertainty-sources-and-ranking-decisions) |
| Compare propagated moments with samples, calibrate intervals, or train with a variance penalty | [Regression and training](examples/README.md#regression-and-training) |
| Propagate an external state posterior through a learned model | [State posteriors and derived targets](examples/README.md#state-posteriors-and-derived-targets) |
| Evaluate calibrated intervals on grouped real measurements | [Grouped measurements](examples/README.md#grouped-measurements), using [statskit](https://github.com/arclabs561/statskit) for calibration ranks |
| Choose settings under execution noise | [Candidate selection](examples/README.md#candidate-choice-under-execution-noise) |
| Compare covariance representations or heavy-tailed noise | [Covariance and heavy tails](examples/README.md#covariance-and-heavy-tails) |
| Propagate node-feature noise through a graph model | [Classification and graphs](examples/README.md#classification-and-graphs), using [ricci](https://github.com/arclabs561/ricci) |
| Regularize contrastive embeddings | [Contrastive embeddings](examples/README.md#contrastive-embeddings), using [tuplet](https://github.com/arclabs561/tuplet) |
| Test whether sensitivity or diversity helps choose training data | [Active selection](examples/README.md#active-selection) |

## Limits

With fixed weights, affine moments are exact for the represented input
covariance. ReLU evaluates Gaussian marginal moments with numerical tail
handling, but its output is not Gaussian. Repeated Gaussian moment matching
across layers is an approximation. The vector API drops off-diagonal covariance
at each ReLU; Burn's `Moments` drops it at every layer. `MomentsFull` retains
approximate feature covariance at quadratic memory cost.

Supplied weight variances assume independent input and weight entries and
retain only marginal output variances. Cauchy distributions have no finite mean
or variance; their locations and scales must be interpreted separately.

Input-noise propagation does not account for label noise, model bias, or an
unknown weight posterior. Validate target coverage separately: the conformal
examples add calibration using held-out observations. A variance penalty or a
risk estimate is not an adversarial robustness certificate.

Inspired by [distprop](https://github.com/Felix-Petersen/distprop) and
[Petersen et al. (ICLR 2024)](https://arxiv.org/abs/2402.08324).
The Gaussian implementation here uses moment matching; distprop uses local
linearization. Read the [method guide](docs/methods.md) for that distinction and
the research history, the [derivations](docs/derivations.md) for proofs and
numerical assumptions, and the [selection note](docs/sensitivity-and-selection.md)
for the connection to learning and exploration.

## Checks

Repository development requires Rust 1.95 or newer because Cargo also resolves
Burn's Git workspace. The published default library supports Rust 1.80.
Install [just](https://github.com/casey/just#installation), then run:

```sh
just check
```

This runs formatting, lints, tests, documentation and example builds, smoke
examples, and benchmark correctness checks. The [justfile](justfile) also
provides extended property tests, independent numerical references, and Metal
value/gradient checks. See the [benchmark guide](benches/README.md) for timing
methods and the [reference guide](scripts/README.md) for numerical oracles.

## License

MIT OR Apache-2.0.
