# Examples

Run these commands from the repository root with Rust 1.92 or newer; Cargo
also builds the Burn development dependencies for `basic`. That example
uses the dependency-free vector API; the other examples default to Burn's CPU
NdArray backend. `robust_training` also accepts `--metal` on macOS with the
`metal` feature. Training and sampling use fixed seeds. Floating-point results may
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
| Carry dependence through a residual branch | [`correlated_residual`](correlated_residual.rs) |
| Evaluate sensitivity as an acquisition score | [`active_selection`](active_selection.rs) |
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
affine difference has standard deviation `sqrt(0.3^2 + 0.4^2) = 0.5`; its mean
is zero, and rectification makes the output mean positive:

```text
mean = 0.1995, variance = 0.0852
```

This vector API retains covariance through affine layers and drops its
off-diagonal entries at ReLU. Try `full_covariance` for a tensor representation
that retains approximate nonlinear covariance.

## Use your own Burn model

Supply input means and **variances** to `Moments::new`, or a covariance matrix
to `MomentsFull::new`. These tensor constructors take variances, unlike the
standard deviations passed to the vector example above. Each batch row is a
separate input; dependence between rows is not represented.

Mirror your model's supported operations in their forward order, using the
same weights and biases. See `forward_with_var` in [robust_training](robust_training.rs)
or `embedding_var` in [tuplet_contrastive](tuplet_contrastive.rs).
Compare propagated moments with samples under the same noise model before
using them in a loss or decision. Coverage of noisy model outputs and coverage
of observed targets are different checks.

## Uncertainty sources and ranking decisions

```sh
cargo run --release --features burn --example uncertainty_sources
cargo run --release --features burn --example pairwise_ranking_risk
cargo run --release --features burn --example pairwise_ranking_risk -- --study
```

`uncertainty_sources` compares a scalar prediction with uncertain inputs,
uncertain weights, or both. Independent Gaussian inputs and weights give an
exact variance decomposition with a product term; simply adding the two
single-source variances misses it. Seeded Monte Carlo checks the result.
The parameter distributions are supplied, not learned by this example.

`pairwise_ranking_risk` compares deferral policies for two fixed candidates
under Gaussian query-feature noise. Full covariance, a baseline that drops
only final score covariance, a local Jacobian, sampled scores, and the point
score margin each defer the same number of queries. It reports the flip rate
among retained queries and the flips avoided per deferred query. Deferral
withholds a decision; the example does not model how a later measurement would
resolve it.

Probability estimates are compared with a separately sampled reference.
Brier scores use held-out binary flip outcomes; the point margin is a rank-only
baseline. The default runs eight sets of 96 query points on one fixed model,
with 2,048 reference draws per query. The `--study` mode uses 30 sets and
16,384 reference draws. Reported standard errors summarize variation across
query sets, conditional on this model and perturbation distribution.

To vary the model and candidates too:

```sh
cargo run --release --features burn --example pairwise_ranking_risk -- --generalize --quick
cargo run --release --features burn --example pairwise_ranking_risk -- --generalize
```

The full mode uses 30 independently generated, untrained networks and candidate
pairs, three query sets per model, and feature-noise standard deviations
0.15, 0.30, and 0.45. Each model contributes one equally weighted average over
the regimes and query sets. Paired policy contrasts use models as the independent
units, with separate results for each regime. The three-model quick mode is
descriptive. Neither mode establishes performance for trained ranking models.
The printed elapsed time covers the whole evaluation, including references;
it is not a comparison of method runtimes.

An affine control isolates the exact Gaussian-margin calculation; the Monte
Carlo comparison still has sampling error. The ReLU network additionally
approximates hidden moments and the final margin distribution. Here the joint
score distribution comes from input noise. A
fitted reward posterior can supply a joint score distribution too, using the
same covariance arithmetic to describe uncertain margins. Exploration adds an
observation model: how would feedback change competing scores, and would that
improve future choices? The [selection note](../docs/sensitivity-and-selection.md#from-ranking-uncertainty-to-exploration)
connects these calculations through Bayesian conditioning and decision value.

## Regression and training

```sh
cargo run --release --features burn --example regression_intervals
cargo run --release --features burn --example conformal_intervals
cargo run --release --features burn --example conformal_intervals -- --study
cargo run --release --features burn --example robust_training
```

`regression_intervals` trains an MLP and compares output moments
with 200 Monte Carlo samples. It reports mean error alongside the Monte Carlo
mean's sampling error and output scale, then standard-deviation correlation,
a scale ratio, and coverage
of sampled model outputs by 95% Gaussian intervals (`mean +/- 1.96 * std`).
High correlation can coexist with incorrect scale. This coverage calculation
does not test intervals against observed targets or account for label noise.

`conformal_intervals` compares raw moment-based intervals with adaptive and
constant-width split-conformal intervals on separate calibration and test data.
The default is a single-split walkthrough; `--study` runs the repeated
heteroscedastic experiment below. The 90% calibration target applies to the
conformal rows; the raw sensitivity intervals are uncalibrated.
Read coverage and average width together. Split conformal targets marginal
coverage under exchangeability; a finite test split need not hit 90% exactly.
The adaptive scale need not produce narrower intervals than the constant one.

This slower mode fits 30 models on separate clean-feature training sets.
Calibration and test targets use one draw of independent Gaussian noise in
each feature, with a known scale that varies by input, plus label noise. The
feature scale is supplied to propagation and never fitted from residuals.
Calibration and test data share the same distribution; training data need not
share it for the split-conformal coverage argument.

The output gives approximate 95% intervals across repeats and paired
scaled-minus-constant differences. Read coverage and width together: the
predeclared screens require coverage within two percentage points of 90% and
at least 2% lower mean width with a paired interval below zero. These are
pilot-study criteria, not coverage guarantees. Low/high noise-bin coverage is
descriptive. The score construction follows
[locally weighted split conformal prediction](https://arxiv.org/html/1604.04173#S5.SS2),
using a propagated sensitivity scale in place of a fitted residual scale.

To diagnose the scale rather than change the calibration protocol:

```sh
cargo run --release --features burn --example conformal_intervals -- --diagnose
cargo run --release --features burn --example conformal_intervals -- --diagnose-study
```

These modes compare diagonal and full third-order covariance propagation (K3)
with paired Monte Carlo outputs from the same fitted model. The model has one
hidden ReLU, so there is
no repeated nonlinear Gaussian closure. A remaining full-versus-sampled
discrepancy combines truncation, numerical, and sampling error.
Separate diagnostics show target-signal variance under input noise, independent
label-noise variance, and squared bias of the interval center. Their sum is the
target residual mean square, not propagated model variance.
The squared-bias estimate subtracts Monte Carlo mean-estimation variance;
it can be negative at finite sample size even though population squared bias
is nonnegative.

The quick diagnosis uses four fitted models and 12 fresh centers per model;
the full diagnosis uses 30 and 48. Both use four independent batches of 512
draws per center. Batch spreads describe Monte Carlo variability; the full
mode's approximate intervals use fitted-model/data repeats as independent units.
The known target/noise quantities are diagnostic oracles from this synthetic
problem, not uncertainty estimates learned by stableprop.

`robust_training` compares plain MSE with MSE plus a propagated-variance penalty,
using shared initial weights and test noise. It prints RMSE with and without
input perturbations while holding clean targets fixed. This models
label-preserving measurement noise; another perturbation may change the target.
Compare both metrics; the penalty can trade accuracy for lower sensitivity.

On macOS, `just metal-train` runs the same training code on Burn's Metal
backend. Its synchronized training time includes first-use kernel compilation
and autotuning. Backend RNG streams differ, so compare the two objectives
within each run. For warmed CPU/GPU timings on fixed inputs, use
`just metal-test`; those timings include tensor allocation and report the
batch size, width, and number of iterations.

## Covariance and heavy tails

```sh
cargo run --release --features burn --example full_covariance
cargo run --release --features burn --example correlated_residual
cargo run --release --features burn --example cauchy_tails
```

`full_covariance` compares diagonal and full propagation at one, two, and three
hidden ReLU layers across three seeds. Each seed shares network prefixes,
the output affine map, 128 input centers, and Gaussian perturbations across
depths. Each Monte Carlo estimate uses 2,048 draws per center; an independent
repeat shows sampling variability, not an error bound.
The final depth summaries give the mean and range across those three fixed
seeds. Compare each analytic method with the MC-repeat discrepancy at the same
depth; these are descriptive summaries, not confidence intervals.

The table reports normalized errors in output means, covariance matrices, and
the standard deviation of the score difference `output[0] - output[1]`.
Mean error is scaled by the square root of total output variance; covariance
and margin-standard-deviation errors use the corresponding reference norms.
The diagonal estimate is embedded in a full matrix for comparison. Read each
metric separately: accurate marginal variances need not give accurate margins.

At depth one, full propagation has covariance-series error but no repeated
Gaussian closure. Additional layers introduce both errors. A scalar control
isolates closure: repeated ReLU leaves every sample unchanged after its first
application, yet repeatedly replacing that output by a Gaussian changes the
propagated moments. This sweep does not establish a general ordering of methods.

`correlated_residual` propagates `Y = X + ReLU(X W + b) V`. It carries
`Cov(X, branch)` through affine and ReLU helpers, then supplies its diagonal to
the correlated-add helper. A scalar hidden state makes these output marginal
moments exact for Gaussian input, apart from numerical tail handling. The
example compares them with 100,000 Monte Carlo draws. For its fixed weights,
ignoring skip–branch dependence understates one variance and overstates the other.

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
propagated covariance and compares them with a 400-draw Monte Carlo estimate
per input. Fixed risk bins show counts, predicted risk, sampled error, and
estimated Monte Carlo standard error at fixed inputs alongside the overall mean
absolute error. This sampling error excludes variation across datasets and
models. True labels are used for evaluation. Summing competing margin events can overcount
their overlap; Gaussian margin tails and their summed risk are also
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
cargo run --release --features burn --example cora_uncertainty -- --quick
```

For a different location, set `STABLEPROP_CORA_DIR` or pass the directory:

```sh
cargo run --release --features burn --example cora_uncertainty -- --quick /path/to/cora
```

An absent default dataset prints a diagnostic and skips
the run; an explicitly supplied missing path is an error. The dataset is not
bundled, and this example performs dense graph operations, so it takes longer
than the synthetic examples.
The quick preset prints its reduced epoch and sampling budgets. Use it to
check the workflow; it does not reproduce the study and retains dense graph
memory use.

`cora_uncertainty` trains a GCN and compares accuracy at retained coverage,
error-detection AUROC, and uncertainty rankings against Monte Carlo. Its
scores sum centered-logit variances, so a random offset shared by all classes
does not count as classification uncertainty. The
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
It also reports propagated embedding variance at the training noise scale and
the held-out embedding RMS. Read these together: reduced variance can accompany
a change in embedding scale rather than better task performance.
It demonstrates differentiable composition; the fixed synthetic pairs and
single seed do not establish a general improvement in representation learning.

## Active selection

```sh
cargo run --release --features burn --example active_selection
```

Compare random selection, predictive entropy, analytic disagreement, and
Monte Carlo disagreement on a synthetic two-moons pool. Each of three seeds
uses the same 256 pool points, 512 held-out points, 16 initial labels, and
2–16–2 ReLU network for all policies. The designed initial set is class-balanced;
subsequent acquisition does not read labels. Each budget refits from the same
initial weights for 250 epochs, without a variance penalty.
Class counts are printed after acquisition, so the composition can help explain
the learning curves without supplying labels to the selection policy.

Entropy uses the model's softmax probabilities at the unperturbed input.
The disagreement policies use independent Gaussian feature noise with standard
deviation 0.12. They center logits to remove offsets shared by all classes.
The analytic score is `2 trace(P Cov(logits | x) P)`, where `P = I - 11^T / K`
and `1` is the length-`K` all-ones column vector. The sampled score estimates
the same quantity from 64 views. For two classes,
this is the variance of the logit margin.

Read the learning curves separately from score agreement. In the reference
CPU NdArray run, entropy reached the highest mean accuracy at 96 labels;
analytic disagreement fell below random selection. The analytic score also
took slightly longer than batched Monte Carlo on this small network. Reported
acquisition times exclude retraining and agreement diagnostics, so they are not
end-to-end training costs or a general backend comparison.

This experiment isolates selection under one noise model. It does not include
noisy labels, irrelevant pool points, or a diversity baseline. The
[selection note](../docs/sensitivity-and-selection.md) explains why accurate
sensitivity estimates can still select unhelpful labels.
