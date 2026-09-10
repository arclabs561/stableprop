# stableprop

Propagate uncertainty through neural networks analytically.

Start with known input noise and estimate how it changes a network's outputs.
Gaussian paths carry means and variances or full covariance; the Cauchy path
carries locations and scales. Compose the layer functions with your model.

Inspired by [distprop](https://github.com/Felix-Petersen/distprop) and
[Petersen et al. (ICLR 2024)](https://arxiv.org/abs/2402.08324). The Gaussian
implementation here uses moment matching; distprop uses local linearization.
The [method guide](docs/methods.md) explains that distinction, the research
history, and which applications each approach supports.

## What uncertainty means here

You supply an input or embedding distribution, or independent weight variances
from another model. stableprop estimates the resulting output moments or scales.
It does not infer those distributions from data.

| Related method | Its job | Where stableprop fits |
| --- | --- | --- |
| Gaussian embeddings | Learn a distribution for each representation | Propagate supplied embedding moments through supported layers |
| Bayesian models | Learn parameter uncertainty from observations | Propagate supplied moments; posterior fitting and updating are external |
| Contrastive learning | Train representations using pair relationships | Add a differentiable sensitivity penalty, as in the tuplet example |
| Conformal prediction | Calibrate prediction sets using held-out observations | Supply an input-dependent scale for calibration |

Input sensitivity and parameter uncertainty arise from different random
quantities. A stable score can still be poorly learned; a well-learned model
can still be sensitive to noisy measurements. See the
[method guide](docs/methods.md#uncertainty-sources-and-downstream-methods) for the distinction.
The [selection note](docs/sensitivity-and-selection.md) connects score covariance
to exploration value and augmentation disagreement to training-data selection.

## Start with a small network

The default API uses `f64` vectors, has no runtime dependencies, and supports
Rust 1.80. The optional Burn backend requires Rust 1.89 or newer.

```toml
[dependencies]
stableprop = "0.3.1"
```

```rust
use stableprop::{propagate_sequential, Layer};

let layers = [
    Layer::Linear {
        weight: vec![vec![1.0, -1.0]], // [output, input]
        bias: vec![0.0],
    },
    Layer::ReLU,
];
let output = propagate_sequential(&layers, &[0.0, 0.0], &[0.3, 0.4]);
println!("mean = {:.4}, variance = {:.4}", output.mean[0], output.cov[0][0]);
```

```text
mean = 0.1995, variance = 0.0852
```

Run this example from the checkout:

```sh
cargo run --release --example basic
```

The input standard deviations describe independent features. Affine layers
retain covariance; this API drops off-diagonal covariance at each ReLU.

## Burn models

Enable `features = ["burn"]` for batched tensors and differentiable propagation.
Choose a compatible Burn 0.20 backend in your application. Burn weights use
`[input, output]`, the transpose of the vector API's layout.

| Representation | What it tracks | Main approximation |
| --- | --- | --- |
| `burn_sdp::Moments` | Mean and variance, `[batch, features]` | Drops feature and row correlations |
| `burn_sdp::MomentsFull` | Mean and covariance, `[batch, features, features]` | Gaussian layer inputs; third-order ReLU covariance series |
| `burn_sdp::Cauchy` | Location and scale, `[batch, features]` | Independent marginals; local ReLU gate |

The tensor API also includes leaky ReLU, diagonal convolution, fixed left
matrix multiplication, residual addition, and affine propagation with supplied
weight variances. See the [API documentation](https://docs.rs/stableprop/latest/stableprop/burn_sdp/).

## Try an application

| I want to… | Start here |
| --- | --- |
| Separate input noise from supplied weight uncertainty | [uncertainty_sources](examples/uncertainty_sources.rs) |
| Estimate whether noisy query features change a ranking | [pairwise_ranking_risk](examples/pairwise_ranking_risk.rs) |
| Compare output uncertainty with sampled noisy inputs | [regression_intervals](examples/regression_intervals.rs) |
| Calibrate prediction intervals against held-out labels | [conformal_intervals](examples/conformal_intervals.rs) |
| Train with an output-variance penalty | [robust_training](examples/robust_training.rs) |
| Measure the effect of retaining covariance | [full_covariance](examples/full_covariance.rs) |
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

Use current stable Rust for repository development. The Rust 1.80 floor applies
to the default library; tests and examples also resolve the Burn dependencies.

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## License

MIT OR Apache-2.0.
