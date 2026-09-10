# Examples

Run these commands from the repository root with current stable Rust. `basic`
uses the dependency-free vector API; the other examples use Burn's CPU NdArray
backend. Training and sampling use fixed seeds. Floating-point results may
still differ across backend or dependency versions.

| I want to… | Start with |
| --- | --- |
| Understand the input and output moments | [`basic`](basic.rs) |
| Separate input noise from parameter uncertainty | [`uncertainty_sources`](uncertainty_sources.rs) |
| Estimate ranking flips under noisy query features | [`pairwise_ranking_risk`](pairwise_ranking_risk.rs) |
| Compare propagated uncertainty with Monte Carlo | [`regression_intervals`](regression_intervals.rs) |
| Calibrate intervals against observed targets | [`conformal_intervals`](conformal_intervals.rs) |
| Add a differentiable variance penalty to training | [`robust_training`](robust_training.rs) |
| Compare diagonal and full covariance | [`full_covariance`](full_covariance.rs) |
| Explore heavy-tailed input noise | [`cauchy_tails`](cauchy_tails.rs) |
| Estimate classification error under input noise | [`misclassification_risk`](misclassification_risk.rs) |
| Compose with graph convolutions | [`gcn_uncertainty`](gcn_uncertainty.rs) |
| Evaluate uncertainty rankings on a citation graph | [`cora_uncertainty`](cora_uncertainty.rs) |
| Regularize contrastive embeddings | [`tuplet_contrastive`](tuplet_contrastive.rs) |

See the [method guide](../docs/methods.md) for the assumptions, research
history, and differences from distprop.

## Start small

```sh
cargo run --release --example basic
```

Two independent Gaussian inputs pass through an affine map and ReLU. The
preactivation has mean zero and standard deviation 0.5; rectification makes its
mean positive:

```text
mean = 0.1995, variance = 0.0852
```

This vector API retains covariance through affine layers and drops its
off-diagonal entries at ReLU. Try `full_covariance` for a tensor representation
that retains approximate nonlinear covariance.

## Uncertainty sources and ranking decisions

```sh
cargo run --release --features burn --example uncertainty_sources
cargo run --release --features burn --example pairwise_ranking_risk
```

`uncertainty_sources` compares a scalar prediction with uncertain inputs,
uncertain weights, or both. Independent Gaussian inputs and weights give an
exact variance decomposition with a product term; simply adding the two
single-source variances misses it. Seeded Monte Carlo checks the result.
The parameter distributions are supplied, not learned by this example.

`pairwise_ranking_risk` scores two fixed candidates for 128 query points.
It estimates how often Gaussian query-feature noise changes the winner,
then compares with 2,048 Monte Carlo draws. The score difference depends on
both score variances and their covariance. Its baseline drops only the final
score covariance; the hidden-layer calculation is shared.

An affine control isolates the exact Gaussian-margin calculation. The ReLU
network additionally approximates hidden moments and the final margin
distribution. Here the joint score distribution comes from input noise. A
fitted reward posterior can supply a joint score distribution too, using the
same covariance arithmetic to describe uncertain margins. Exploration adds an
observation model: how would feedback change competing scores, and would that
improve future choices? The [selection note](../docs/sensitivity-and-selection.md#from-ranking-uncertainty-to-exploration)
connects these calculations through Bayesian conditioning and decision value.

## Regression and training

```sh
cargo run --release --features burn --example regression_intervals
cargo run --release --features burn --example conformal_intervals
cargo run --release --features burn --example robust_training
```

`regression_intervals` trains an MLP and compares output standard deviations
with 200 Monte Carlo samples. It prints correlation, a scale ratio, and coverage
of sampled model outputs by Gaussian-reference intervals. High correlation can
coexist with incorrect scale. This coverage calculation does not test intervals
against observed targets or account for label noise.

`conformal_intervals` compares raw moment-based intervals with adaptive and
constant-width split-conformal intervals on separate calibration and test data.
Read coverage and average width together. Split conformal targets marginal
coverage under exchangeability; a finite test split need not hit 90% exactly.
The adaptive scale need not produce narrower intervals than the constant one.

`robust_training` compares plain MSE with MSE plus a propagated-variance penalty,
using shared initial weights and test noise. It prints RMSE with and without
input perturbations. Compare both metrics; the penalty can trade accuracy for lower sensitivity.

## Covariance and heavy tails

```sh
cargo run --release --features burn --example full_covariance
cargo run --release --features burn --example cauchy_tails
```

`full_covariance` compares diagonal propagation and the third-order ReLU
covariance approximation against 400 Monte Carlo samples through an MLP. Read
the mean absolute relative standard-deviation error alongside the mean ratio:
a ratio near one can hide errors that cancel. This seeded comparison does not establish a general ordering.

`cauchy_tails` measures interval coverage when the actual input perturbations
are Cauchy. Gaussian standard deviation and Cauchy scale are different
parameters; Cauchy has no finite variance. The affine Cauchy rule is exact for
independent inputs, but the example's nonlinear chain uses local ReLU gates
and discards dependence. Its reported coverage is empirical, not a guarantee.

## Classification and graphs

```sh
cargo run --release --features burn --example misclassification_risk
cargo run --release --features burn --example gcn_uncertainty
```

`misclassification_risk` estimates logit-margin error probabilities from
propagated covariance and compares them with 400 Monte Carlo samples. It uses
true labels for evaluation. Gaussian margin tails and their summed risk are
approximations after nonlinear propagation, not robustness certificates.

`gcn_uncertainty` composes stableprop with
[ricci](https://github.com/arclabs561/ricci)'s `GCNConv` on a synthetic 32-node
graph. It reports variance correlation and scale agreement against Monte
Carlo. Shared neighbors create node correlations that the diagonal path drops.
Its thresholding display demonstrates how to defer uncertain predictions;
there are no labels to establish that deferral improves accuracy.

### Real Cora data

Download and extract the [original Cora archive](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz).
The extracted directory must contain `cora.content` and `cora.cites` (the
LINQS tab-separated format, not Planetoid pickle files). Put that directory at
`data/cora`, or supply its location:

```sh
cargo run --release --features burn --example cora_uncertainty
```

For a different location, append `-- /path/to/cora` to the command or set
`STABLEPROP_CORA_DIR`. An absent default dataset prints a diagnostic and skips
the run; an explicitly supplied missing path is an error. The dataset is not
bundled, and this example performs dense graph operations, so it takes longer
than the synthetic examples.

`cora_uncertainty` trains a GCN and compares accuracy at retained coverage,
error-detection AUROC, and uncertainty rankings against Monte Carlo. Its
weight-uncertainty signal is a diagonal empirical-Fisher proxy; it is not a
calibrated posterior and omits shared-weight and cross-node covariance.

A second run withholds one class's labels from training and compares uncertainty
with a maximum-softmax baseline. The full graph, including held-out-class
features and edges, remains visible during training. This is transductive
novel-class scoring, not evaluation on an unseen deployment graph. The split
is seeded and class-stratified, not the canonical Planetoid split.

## Contrastive embeddings

```sh
cargo run --release --features burn --example tuplet_contrastive
```

This example combines [tuplet](https://github.com/arclabs561/tuplet)'s pairwise
contrastive loss with a stableprop embedding-variance penalty. Both encoders
start from the same weights. Evaluation uses held-out points, class centroids
from training embeddings, and ten shared noise draws per test point. The
printed table compares nearest-centroid accuracy with and without noise.
It demonstrates differentiable composition; the fixed synthetic pairs and
single seed do not establish a general improvement in representation learning.
