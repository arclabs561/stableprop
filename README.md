# stableprop

Propagate uncertainty through neural networks analytically.

Given your model's weights and Gaussian input uncertainty, stableprop estimates
output means and variances through supported layers without repeatedly sampling
the noisy inputs. The Burn API is differentiable, so those estimates can also
be used in a training loss.

You supply the uncertainty: measurement noise, an upstream state estimate, or
a perturbation scale for sensitivity analysis. The library propagates that
distribution; it does not learn it from data. It also supports externally
supplied weight variances and a separate Cauchy location/scale representation.

## Start with a small network

For an application using Rust 1.80 or newer, add the published vector API to
`Cargo.toml`. It uses `f64` and has no runtime dependencies.

```toml
[dependencies]
stableprop = "0.5.2"
```

This computes `ReLU(x₁ − x₂)` for two independent, zero-mean Gaussian inputs
with standard deviations 0.3 and 0.4. Put this in `src/main.rs` and run
`cargo run --release`:

```rust
use stableprop::{propagate_sequential, Layer};

fn main() {
    let layers = [
        Layer::Linear {
            weight: vec![vec![1.0, -1.0]], // [output, input]
            bias: vec![0.0],
        },
        Layer::ReLU,
    ];
    let input_mean = [0.0, 0.0];
    let input_std = [0.3, 0.4]; // Standard deviations, not variances.
    let output = propagate_sequential(&layers, &input_mean, &input_std);
    println!("mean = {:.4}, variance = {:.4}", output.mean[0], output.cov[0][0]);
}
```

```text
mean = 0.1995, variance = 0.0852
```

Evaluating the network at the input mean gives zero. Its noisy output has a
positive mean because ReLU clips negative values to zero. The variance describes
variation under the supplied noise, not prediction-interval coverage for observed
targets.

From a repository checkout, use Rust 1.95 or newer and run the same example:

```sh
cargo run --release --example basic
```

## Burn models

The development tensor API below differs from the published crate's Burn API.
It requires Rust 1.95 or newer and an unreleased Burn revision. These Git
revisions are tested together; update them together:

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
| [`MomentsFull`](src/burn_sdp.rs) | Means `[batch, features]` and covariance `[batch, features, features]` | Feature covariance through affine/ReLU layers, with a [third-order ReLU approximation](docs/derivations.md#relu-coefficients-and-the-implemented-order) |
| [`Cauchy`](src/burn_sdp.rs) | Locations and scales, each `[batch, features]` | Heavy-tailed marginals; discards dependence and uses local ReLU gating |

Each batch row represents a separate distribution; none of these types stores
cross-row covariance. `MomentsFull` requires symmetric positive semidefinite
covariance matrices; shape checks do not establish that condition. Burn weights
use `[input, output]`, the transpose of the vector API's layout. Keep operands
on the same device and in the same floating dtype. For explicit `f64`, pass
`(&device, DType::F64)` to tensor constructors.

CPU examples use `Device::flex()`; call `.autodiff()` for gradients. On macOS,
`features = ["metal"]` enables WGPU Metal with operation fusion. Use
`just metal-test` for GPU value and gradient checks, or `just metal-train` for
a training example.
The [API source](src/burn_sdp.rs) documents supported operations and their assumptions.

## Try an application

For a self-contained decision example, compare ways to choose a setting under
Gaussian execution noise:

```sh
cargo run --release --features burn --example robust_selection -- --quick
```

It compares propagated expected squared losses with sampled model outputs and
an exact simulator oracle, separating propagation error from model error. For
a fixed target, expected squared loss depends only on output mean and variance;
ranking-flip probabilities generally need more than those two moments.

The guide covers further applications, their data requirements, and how to
interpret the results. These examples use `--features burn`.

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
risk estimate is not an adversarial robustness certificate. The
[method guide](docs/methods.md#uncertainty-sources-and-downstream-methods)
distinguishes input sensitivity, parameter uncertainty, and target coverage.

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
just property-test
```

`just check` runs formatting, lints, tests, documentation and example builds,
smoke examples, and benchmark correctness checks. `just property-test` extends
the propagation, calibration, and selection properties to 4,096 cases each.
The [justfile](https://github.com/arclabs561/stableprop/blob/main/justfile) also provides independent numerical references and
Metal value/gradient checks. See the [benchmark guide](benches/README.md) for
timing methods and the [reference guide](scripts/README.md) for numerical oracles.

## License

MIT OR Apache-2.0.
