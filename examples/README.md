# stableprop examples

Each example answers one question and is runnable from the repo root. Output
excerpts below are real, captured from a run. All examples need the `burn`
feature. `cora_uncertainty` is data-gated: if the Cora files are absent, it
exits 0 and prints the fetch command for the sibling `propago` checkout.

## Regression uncertainty

### `regression_intervals`: can analytic error bars replace Monte Carlo?

Trains an MLP regressor, propagates known input variance through it, and compares
the analytic output standard deviation against 200-sample Monte Carlo.

```bash
cargo run --release --features burn --example regression_intervals
```
```text
training MLP regressor (3000 samples, 800 epochs)...
train RMSE: 0.2088

sampling-free error bars vs 200-sample Monte Carlo:
  std agreement (Pearson r) = 0.8023   (1.0 = identical error bars)
  std mean ratio (mp / MC)  = 1.063  (1.0 = unbiased magnitude)
  95% interval coverage     = 0.929   (target ~0.95 = calibrated)

cost: stableprop = 1 forward pass, Monte Carlo = 200 passes
```

### `conformal_intervals`: can the analytic scale get distribution-free coverage?

Wraps stableprop's per-point standard deviation in split conformal prediction
and compares raw, adaptive conformal, and constant-width conformal intervals.

```bash
cargo run --release --features burn --example conformal_intervals
```
```text
target coverage = 0.90

  method                             coverage  avg width
  raw stableprop (1.645*sigma)          0.505      0.334
  conformalized stableprop (adaptive)    0.877      0.792
  constant-width conformal              0.905      0.792

raw is miscalibrated; both conformal methods hit the target with a guarantee.
```

### `robust_training`: can propagated variance be part of the training loss?

Trains two regressors from the same initial weights: plain MSE and MSE plus a
penalty on propagated output variance under input noise.

```bash
cargo run --release --features burn --example robust_training
```
```text
RMSE (lower = better), shared init + shared test noise:
  net                             clean    noisy
  plain MSE                      0.1465   0.4220
  MSE + variance penalty         0.1879   0.4115

the penalized net trades clean accuracy for lower error under input noise.
```

## Propagation variants

### `full_covariance`: what does keeping covariance buy?

Compares diagonal propagation with full-covariance propagation against Monte
Carlo through a two-layer MLP.

```bash
cargo run --release --features burn --example full_covariance
```
```text
output std vs 400-sample Monte Carlo (mean ratio, 1.0 = unbiased):
  diagonal       1.120
  full covariance 1.003

full covariance keeps the cross-feature correlations the diagonal drops.
```

### `cauchy_tails`: what happens under heavy-tailed input noise?

Propagates the same net under Gaussian and Cauchy assumptions and checks interval
coverage when the actual perturbations are Cauchy-tailed.

```bash
cargo run --release --features burn --example cauchy_tails
```
```text
90% interval coverage under heavy-tailed (Cauchy) input noise:
  Gaussian propagation  0.571
  Cauchy propagation    0.958

Gaussian intervals are too narrow for heavy tails; Cauchy keeps them.
```

## Classification and graphs

### `misclassification_risk`: can propagated covariance estimate error risk?

Propagates full covariance into classifier logits and estimates the probability
that a competitor class beats the true class, then compares with Monte Carlo.

```bash
cargo run --release --features burn --example misclassification_risk
```
```text
analytic misclassification-risk estimate under input noise std 0.25:
  mean analytic estimate = 0.0893
  mean MC rate           = 0.0864  (400 samples)
  estimate within 0.0029 of MC on average; lands above the per-input rate 79.2% of the time (an estimate, not a guaranteed bound).
```

### `gcn_uncertainty`: does diagonal propagation track MC variance through a GCN?

Builds a two-layer `propago` GCN, puts Gaussian noise on node features, and
compares analytic output variance with Monte Carlo.

```bash
cargo run --release --features burn --example gcn_uncertainty
```
```text
2-layer GCN (GCNConv -> ReLU -> GCNConv), n=6 nodes, d_out=4
input noise std = 0.3, MC samples = 4000

SDP var vs MC var:
  Pearson r   = 0.9040   (1.0 = perfect agreement)
  mean ratio  = 0.716  (SDP / MC; 1.0 = unbiased)

abstention (SDP node std > mean 0.0211 => defer):
  node 0: std=0.0266  -> ABSTAIN
  node 5: std=0.0242  -> ABSTAIN
  node 1: std=0.0238  -> ABSTAIN
```

### `cora_uncertainty`: does propagated uncertainty help on real Cora nodes?

Trains a GCN on Cora, ranks test nodes by propagated uncertainty, compares with
Monte Carlo, and checks the softmax baseline for OOD detection.

```bash
cargo run --release --features burn --example cora_uncertainty
```
```text
dataset: cora  nodes: 2708  features: 1433  classes: 7  test: 1000
training 2-layer GCN (200 epochs)...
test accuracy (full coverage): 0.8010

misclassification detection (AUROC of uncertainty vs error):
  input-noise (aleatoric) AUROC = 0.5612
  weight-Laplace (epistemic) AUROC = 0.5451
  MC input-noise AUROC = 0.5569  (200 samples)

SDP vs MC per-node uncertainty (test nodes): Spearman rho = 0.9073

OOD-detection AUROC (1.0 = uncertainty perfectly separates novel-class nodes):
  input-noise (aleatoric)    = 0.6243
  weight-Laplace (epistemic) = 0.6045
  max-softmax-prop (baseline) = 0.8096
```

## Composition

### `tuplet_contrastive`: can stableprop regularize contrastive embeddings?

Trains a `tuplet` contrastive encoder with and without a stableprop variance
penalty, then compares nearest-centroid accuracy under input noise.

```bash
cargo run --release --features burn --example tuplet_contrastive
```
```text
nearest-centroid accuracy on noisy inputs (test std 0.5):
  plain contrastive              0.898
  contrastive + variance penalty 0.903

stableprop and tuplet compose in Burn end to end: the encoder trains under
tuplet's contrastive loss while stableprop supplies the analytic embedding variance.
```
