# Methods, history, and applications

Selected literature reviewed through September 10, 2026. This guide covers methods
relevant to stableprop. Recent results are preprints unless a publication venue
is named; they describe their own implementations and experiments, not this crate.

The [derivations](derivations.md) give proofs of the moment and cross-covariance
identities, the Hermite covariance series, its positive-semidefiniteness and
truncation bounds, and the connections to Bussgang, Price, and Gaussian kernels.

## What is being approximated?

Given a distribution over inputs, parameters, or both, the target is the
induced output distribution. For fixed weights and uncertain input `X`, this
is the distribution of `f(X)`. Three choices determine what propagation returns:

- Representation: independent marginals, full covariance, a structured
  covariance, or a distribution with no finite moments such as Cauchy.
- Nonlinear approximation: match moments, linearize around an input, or
  sample the transformed distribution.
- Uncertainty source: noisy inputs, uncertain weights, observation noise,
  or a combination. The propagation rule does not determine how these were fit.

These choices are separate. Keeping covariance does not make a network's
output Gaussian, and attaching a variance to a deterministic model does not
learn a Bayesian posterior.

For an affine layer, using column-vector notation:

$$
\mu_y = W\mu_x + b, \qquad \Sigma_y = W\Sigma_x W^\top.
$$

For a scalar Gaussian input to ReLU, with $a=\mu/\sigma$ and standard normal
CDF $\Phi$ and density $\phi$:

$$
\mathbb{E}[\max(0,X)] = \mu\Phi(a) + \sigma\phi(a).
$$

The nonlinear mean shift distinguishes moment matching from evaluating
`ReLU(mean)`. The implementations use equivalent variance formulas chosen to
avoid floating-point cancellation, with numerical approximations in the tails.

## How the methods developed

| Work | Change in method | Relationship to stableprop |
| --- | --- | --- |
| [Bussgang, 1952, *Crosscorrelation Functions of Amplitude-Distorted Gaussian Signals*](https://hdl.handle.net/1721.1/4847) | Relates Gaussian input–output cross-correlation through a memoryless nonlinearity to a scalar gain. | Predecessor of the expected-slope identity used by the ReLU cross-covariance helper. |
| [Price, 1958, *A Useful Theorem for Nonlinear Devices Having Gaussian Inputs*](https://doi.org/10.1109/TIT.1958.1057444) | Relates derivatives of Gaussian expectations with respect to covariance to derivatives of the nonlinear functions. | Mathematical background for covariance expansions. [Voigtlaender's general form](https://arxiv.org/abs/1710.03576) handles distributional derivatives and states the regularity conditions. |
| [Frey & Hinton, 1999, *Variational Learning in Nonlinear Gaussian Belief Networks*](https://www.cs.toronto.edu/~hinton/absps/nlgbn.pdf), especially the rectified-unit moments | Analytic expectations of nonlinear Gaussian units support variational inference in belief networks. | Source for Gaussian ReLU marginal moments; stableprop is not the original belief-network learner. |
| [Minka, UAI 2001, *Expectation Propagation*](https://tminka.github.io/papers/ep/minka-ep-uai.pdf), §§2–3 | Assumed-density filtering repeatedly projects a distribution into a tractable family. EP generalizes it by revisiting approximate factors. | Explains Gaussian closure and why discarded information matters later. Forward propagation here does not implement EP's iterative posterior updates. |
| [Cho & Saul, NeurIPS 2009, *Kernel Methods for Deep Learning*](https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning), §2 | Evaluates Gaussian rectified pair integrals as arc-cosine kernels. | Supplies the centered-input pair reference used to measure series truncation error, after subtracting the product of marginal means. |
| [Hernández-Lobato & Adams, ICML 2015, *Probabilistic Backpropagation*](https://proceedings.mlr.press/v37/hernandez-lobatoc15.html), §3 | Gaussian moment propagation is combined with approximate Bayesian updates to learn weight distributions. | `propagate_linear_bayes` implements an independent-weight moment rule, not the PBP learning algorithm. |
| [Gast & Roth, CVPR 2018, *Lightweight Probabilistic Deep Networks*](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gast_Lightweight_Probabilistic_Deep_CVPR_2018_paper.pdf), §§3–4 | Carries activation means and variances through CNNs and introduces probabilistic output layers. | Practical precedent for the diagonal tensor path. Output likelihoods and training objectives are separate parts of that system. |
| [Wu et al., ICLR 2019, *Deterministic Variational Inference*](https://arxiv.org/abs/1810.03958), §3 and Appendix A | Propagates activation covariance with uncertain weights, approximates nonlinear cross-moments, and optimizes a variational objective. | A broader Bayesian framework. The crate's supplied-weight-variance API covers only one propagation operation. |
| [Petersen et al., ICLR 2024, *Stable Distribution Propagation*](https://arxiv.org/abs/2402.08324), §§3.1–3.3 | Uses local linearization and stable distributions; computes Gaussian output covariance from the network Jacobian. Extends the framing to Cauchy and discusses other symmetric stable laws. | The project's inspiration. Gaussian moment matching here is a different approximation; Cauchy ReLU uses a local gate. |
| [Wright, Nakahira & Moura, AISTATS 2024, *An Analytic Solution to Covariance Propagation in Neural Networks*](https://proceedings.mlr.press/v238/wright24a.html), Theorem 1 and §3 | Gives an infinite covariance series in derivatives of activation expectations under Gaussian inputs. | `propagate_relu_full` retains its first three off-diagonal terms and uses univariate variances on the diagonal. |

The broad trajectory is from tractable marginal expectations to richer
dependence models and application-specific inference. Local linearization is a
parallel approach, not a later version of moment matching.

## Generalizations and distinctions

Full Gaussian covariance generalizes diagonal covariance as a representation.
A diagonal matrix is a special full covariance matrix. The first affine layer
has the same marginal variances in both paths when given independent inputs.
Subsequent affine layers can mix correlated features, so the paths then differ.
`MomentsFull` costs quadratic memory in feature width; the diagonal path is
cheaper for repeated sensitivity evaluations. Even a scalar final output can
depend on correlations between hidden features.

The third-order series extends the first-order smooth-gate approximation.
Its first term is
`Cov(X_i, X_j) * Phi(a_i) * Phi(a_j)`, where `a = mean / std`. The second and
third terms add nonlinear covariance contributions. Higher order includes more
of the series, but need not improve every individual covariance entry at each
order; it does not eliminate the Gaussian approximation
between layers. The infinite series is exact under its assumptions, while
stableprop uses a finite truncation.

$$
\mathrm{Cov}(g(X_i),g(X_j)) \approx
\sum_{k=1}^{3}\frac{\Sigma_{ij}^{k}}{k!}
\frac{\partial^{k}\mathbb{E}[g(X_i)]}{\partial\mu_i^{k}}
\frac{\partial^{k}\mathbb{E}[g(X_j)]}{\partial\mu_j^{k}}.
$$

SDP and moment matching optimize different approximations. The
[distprop implementation](https://github.com/Felix-Petersen/distprop/blob/e727da2057ef45f18df31cc8b58597505a0b8b03/distprop/sdp.py)
returns the deterministic output `f(mu)` and `J * J^T * s^2` for isotropic
Gaussian input with standard deviation `s`, where `J` is the network Jacobian
at `mu`. Petersen's ReLU argument minimizes a univariate total-variation
distance. Moment matching instead preserves Gaussian-input expectations. At a
zero-mean ReLU input, the true rectified mean is positive, while local
linearization returns zero. Neither method contains the other or wins under
every error metric.

Cauchy and Gaussian are members of a broader stable-distribution family.
Cauchy is not a Gaussian with a larger variance: its mean and variance are
undefined. An affine sum of independent Cauchy variables has scale
`sum(abs(weight) * scale)`. The crate retains only these marginal scales;
mixing layers creates dependence that this representation discards. Petersen's
network-Jacobian formulation can retain the original noise dependence through
composition. The crate does not implement that full Jacobian calculation.

Uncertain-weight propagation generalizes deterministic weights. Setting
weight and bias variances to zero recovers the diagonal deterministic affine
rule. PBP and DVI additionally learn distributions over weights. Likewise,
`propagate_residual_add_correlated` generalizes the independent-add helper when
the caller supplies the cross-covariance. The affine and ReLU cross-covariance
helpers derive it through supported branches, as in
[`correlated_residual`](../examples/correlated_residual.rs).

For jointly Gaussian vectors `U, V`, Gaussian integration by parts gives
`Cov(U, ReLU(V)) = Cov(U, V) diag(Phi(mean_V / std_V))`. Unlike covariance
between two rectified variables, this cross-covariance needs only univariate
Gaussian CDFs. The identity is exact at that Gaussian layer, apart from numerical
tail handling. ReLU makes the joint distribution non-Gaussian; subsequent
affine transport stays exact, but another Gaussian ReLU step is an approximation.

## Efficiency and accuracy

| Approach | Computation | Main accuracy limit |
| --- | --- | --- |
| Local linearization | Uses derivatives of the deterministic network | Can miss activation-boundary crossings and nonlinear mean shifts |
| Diagonal moment matching | Carries one mean and variance per feature; no feature-pair covariance state | Discards correlations that later layers can amplify or cancel |
| Full moment matching | Carries a dense feature covariance matrix per input row | Gaussian layer-input approximation; this implementation truncates the covariance series |
| Sigma-point quadrature | Evaluates weighted input points through the network; cost grows with input dimension | A finite quadrature rule can miss activation boundaries |
| Monte Carlo | Repeats network evaluations under the chosen noise distribution | Sampling error; rare events need many samples |

For a dense square layer of width `d`, diagonal affine propagation costs
`O(d^2)` work and `O(d)` moment storage; full covariance costs `O(d^3)` work
and `O(d^2)` storage, per input. ReLU's fixed third-order pairwise series costs
`O(d^2)`. These are operation counts, not measured speedups: device kernels,
batch size, covariance structure, and graph aggregation affect runtime.

Closed-form moments still need careful numerical evaluation. For a Gaussian
mean far below zero, `1 + erf(a / sqrt(2))` loses the small activation
probability; subtracting terms to obtain ReLU moments compounds that error.
The negative-tail implementation instead uses the
[Laplace continued fraction](https://dlmf.nist.gov/7.9) and
[ratios of repeated Gaussian tail integrals](https://dlmf.nist.gov/7.18#v).
For $t=-a>0$, write $r_n=n/(t+r_{n+1})$, $R=1/(t+r_1)$,
and $X_+=\max(0,X)$. Then:

$$
\Phi(-t)=\phi(t)R,\qquad
\frac{\mathbb{E}[X_+]}{\sigma}=\phi(t)Rr_1,\qquad
\frac{\mathbb{E}[X_+^2]}{\sigma^2}=\phi(t)Rr_1r_2.
$$

These forms avoid subtracting nearly equal tail terms. Both APIs retain
linear limits at `a >= 8` and zero limits at `a <= -8`. Within those limits,
both switch to continued fractions at `a < -2`, with 32 levels for Burn
`f32` and 96 for `f64`. In the remaining region, the vector API uses a
[convergent CDF series](https://www.jstatsoft.org/v11/i04/) and Burn uses its
backend's error function. Tensor
masks evaluate both branches, so this accuracy costs arithmetic even for
central inputs; the [Burn benchmarks](../benches/README.md) separate both regimes.
These numerical choices do not remove Gaussian closure or covariance-series
truncation error.

Differentiability also depends on the coordinates and boundary. At zero mean,
the Gaussian ReLU mean is `sqrt(v / (2 pi))`, whose derivative with respect to
variance `v` diverges as `v` approaches zero. The Burn path selects finite
gradients for exactly deterministic inputs. Those are computational conventions,
not limits of every positive-variance derivative. Tests separate these boundary
conventions from analytical gradient checks at positive variance.

Finite forward values do not ensure finite autodiff gradients: an overflowing
intermediate derivative can contaminate a masked branch. Burn bounds the mean
before dividing by the standard deviation and normalizes covariance by the
larger standard deviation first to avoid these intermediate overflows.

The [unscented transform (Julier & Uhlmann, 1997)](https://www.robots.ox.ac.uk/~cvrg/hilary2003/Julier1997_SPIE_KF.pdf)
and [cubature Kalman methods (Arasaratnam & Haykin, 2009)](https://doi.org/10.1109/TAC.2009.2019800)
approximate transformed moments using weighted input points. Applied to a whole
network, these avoid intermediate Gaussian approximations, but retain quadrature
error. They are useful comparison methods when input dimension is small and
hidden layers are wide. Their polynomial integration guarantees do not make a
ReLU network's output moments exact. stableprop does not implement these rules.

Dropping covariance can either raise or lower the final variance, depending on
weights and correlation signs. More covariance terms do not ensure better
network-level accuracy: Gaussian closure and series truncation are separate
errors. Exact bivariate Gaussian ReLU formulas can remove the latter while
retaining the former, at the cost of additional numerical primitives.

Strong correlation can make truncation matter to a decision. For two
zero-mean, unit-variance Gaussian inputs with correlation one, the rectified
covariance is about 0.34085; the current series gives 0.32958. Subtracting
those identical rectified outputs should give zero variance, but combining
the truncated cross-term with exact marginal variances gives about 0.02254.
The [closed-form reference test](../tests/relu_covariance_reference.rs) checks
this limitation across positive and negative correlations.
For decisions driven by cancellation between nearly identical ReLU outputs,
compare against Monte Carlo or exact bivariate moments.

Independent Monte Carlo mean estimates have standard error proportional to
`1 / sqrt(samples)` when variance is finite. Cauchy means do not satisfy that
condition; compare quantiles or coverage instead. For Gaussian paths, compare
means and per-output errors as well as average ratios: positive and negative
errors can cancel. Test calibration against the quantity the application
actually observes, not only agreement with the model's own noisy outputs.
For correlated outputs, also compare the covariance matrices: correct marginal
variances can hide incorrect uncertainty in output differences or sums. The
[full-covariance example](../examples/full_covariance.rs) reports both marginal
standard-deviation errors and normalized covariance error against Monte Carlo.

## Developments after 2024

- [Akgül et al., *Deterministic Uncertainty Propagation for Improved Model-Based
  Offline Reinforcement Learning*](https://arxiv.org/abs/2406.04088), revised
  January 2025, §4: MOMBO uses marginal Gaussian moments for uncertainty in
  value targets. Their application favors diagonal propagation's cost over
  full covariance. It also needs a learned transition model, an ensemble, and
  a pessimistic policy-learning objective; stableprop supplies none of that
  surrounding RL system.
- [Diamzon & Venturi, *Uncertainty propagation in feed-forward neural network
  models*](https://doi.org/10.1016/j.neunet.2025.108178), *Neural Networks* 194
  (February 2026; online October 2025), §§4–8 and Appendix B of the
  [author preprint](https://arxiv.org/abs/2503.21059): develops local Leaky-ReLU linearization, output-density
  approximations, and Gaussian-copula surrogates, with activation-pattern error
  analysis. It is an alternative when densities or dependence beyond marginal
  error bars matter; errors still depend on the network and perturbations.
- [Kuang & Lin, *Exact Gaussian Moment Matching for Residual Networks: a
  Second-Order Method*](https://arxiv.org/abs/2601.22307), revised May 2026,
  §§2, 4–6: derives Gaussian layer moments for several activations and joint
  residual terms without truncating Wright's series. This is a direct
  accuracy reference for the full-covariance path. Exactness is per Gaussian
  layer; the higher-order error theorem requires smoothness assumptions and
  is not a blanket ReLU-network guarantee. Softmax and attention are excluded.
- [Thompson & McCrory, *Uncertainty propagation through trained multi-layer
  perceptrons: Exact analytical results*](https://arxiv.org/abs/2601.16830),
  January 2026, §§3–5: gives exact Gaussian-input output moments for a
  single-hidden-layer ReLU regressor using univariate and bivariate Gaussian
  integrals. With just one nonlinear layer, no intermediate Gaussian
  approximation is needed. This isolates covariance-series error from the
  additional approximation required by deeper networks.
- [Bergna et al., *Activation-Space Uncertainty Quantification for Pretrained
  Networks*](https://arxiv.org/abs/2602.14934), revised February 2026, §2:
  GAPA adds Gaussian-process uncertainty to activations of a frozen network.
  It preserves deterministic point predictions and uses diagonal activation
  kernels with local conditioning on cached training activations. This models
  epistemic uncertainty in activation space; it is different from propagating
  supplied input noise through a fixed activation function.
- [Wieczorek et al., *Calibrated Sampling-Free Uncertainty Estimation in
  Bayesian Deep Learning*](https://arxiv.org/abs/2606.16214), June 2026,
  §§4–6: CVP combines diagonal Bayesian variance propagation, an
  expectation-based normalization approximation, and per-layer variance
  scales fit on held-out data. The CNN and transformer results rely on trained
  IVON weight posteriors. This is a broader calibrated inference system,
  not evidence that adding an activation function gives stableprop transformer
  support. Evaluation covers encoder-style classification and VQA heads.
- [Nie et al., *Two-Step MV-DeepONet*](https://arxiv.org/abs/2608.09071),
  August 2026, §2.4 and Appendix B: learns a basis in which coefficient
  uncertainty is diagonal, then reconstructs correlated output fields.
  It illustrates a middle ground between independent output coordinates and
  dense output covariance. The method changes the surrogate's representation
  and training; it is not a layerwise replacement for this crate.
- [Sharma & Precup, *Analytic Planning under Uncertainty with Moment
  Closure*](https://arxiv.org/abs/2608.02519), August 2026, §§3–5:
  evaluates Bellman expectations analytically by pairing Gaussian transition
  predictions with radial-basis value functions and a quadratic action-value
  model. The useful principle is to choose a downstream function whose
  expectation is tractable under the propagated distribution. The closed-form
  backup depends on that model structure, not on arbitrary neural-network
  moment propagation.
- [Adams & Venturi, *Uncertainty propagation in auto-regressive random neural
  network models*](https://arxiv.org/abs/2608.20483), August 2026, §§3–5:
  uses a joint input–parameter Jacobian and retains state–parameter covariance
  during recurrent prediction. The joint linearization is a first-order
  approximation. A fixed activation pattern makes the network affine in its
  input for fixed parameters; varying both leaves mixed terms. Their longer
  horizon treatment also uses particles and resampling. Repeated calls to an
  independent-weight layer rule do not retain these evolving joint statistics.

These results support several directions rather than a single replacement
method: more accurate Gaussian layer integrals, uncertainty models for frozen
networks, and propagation designed around a downstream inference or control
calculation. For stableprop, the immediate comparison is whether better pair
moments improve covariance-sensitive outputs enough to justify their cost.
Sequential prediction additionally needs the joint statistics retained by its
state and parameter model.

## What is useful in practice?

### Uncertainty sources and downstream methods

Propagation starts after a distribution has been specified. A learned Gaussian
embedding supplies representation moments; a Bayesian model supplies a
parameter posterior; a sensor model supplies measurement noise. Similar
Gaussian arithmetic does not make these uncertainty sources interchangeable.

A neural-linear bandit models reward with learned features `phi` and uncertain
linear weights `beta`. For a supplied posterior `beta ~ N(m, S)`, its latent
score has mean `phi^T m` and variance `phi^T S phi`. Affine moment propagation
can evaluate those quantities, but fitting and updating `m, S` requires reward
observations and a statistical model. Independent weight variances in
`propagate_linear_bayes` do not represent a dense posterior `S`.
The full-covariance affine API can instead map the supplied coefficient
posterior to joint candidate scores. The [selection note](sensitivity-and-selection.md#from-ranking-uncertainty-to-exploration)
shows how their covariance enters Bayesian updates and exploration value.
[Riquelme et al. (2018)](https://arxiv.org/abs/1802.09127) study neural-linear
posterior methods; [Su et al. (WSDM 2024)](https://arxiv.org/abs/2305.07764)
apply one to exploration after candidate generation.

Contrastive learning defines relationships between embeddings. The
`tuplet_contrastive` example adds a penalty on variance propagated from an
explicit input-noise model. This regularizes sensitivity. Using sensitivity
to select pairs needs a specified selection objective and controlled evaluation:
high sensitivity can indicate useful signal or unreliable measurements.
The [sensitivity and selection note](sensitivity-and-selection.md) develops
that distinction, its active-learning evidence, and possible extensions.

Calibration asks another question: do the reported intervals cover the target
at the intended rate? Propagation alone does not answer it. The conformal
example uses held-out labels to calibrate a propagated scale.

### Application choices

| Application | What stableprop provides now | What to measure or add |
| --- | --- | --- |
| Sensor-noise propagation through a regressor | Gaussian moment estimates with diagonal or full covariance | Compare output means, variance error, coverage, and runtime against Monte Carlo. Coverage of noisy model outputs is different from coverage of observed labels. |
| Calibrated regression intervals | A per-input scale for the `conformal_intervals` example | Held-out calibration and test splits; interval width and coverage. [Split conformal](https://arxiv.org/abs/2107.07511) assumes exchangeability and targets marginal coverage. |
| Embedding stability | Differentiable variance penalty alongside [tuplet](https://github.com/arclabs561/tuplet)'s contrastive loss | Shared initialization, held-out examples, shared perturbations, and downstream accuracy with and without noise. A penalty can also erase useful signal. |
| Learned dynamics and state estimation | Marginal and cross-covariance transport through affine/ReLU layers | A filtering or control system also needs joint-state bookkeeping, process and observation noise, and conditioning. [Kuang & Lin's filtering and smoothing study](https://arxiv.org/abs/2511.09016), revised May 2026, constructs those joint distributions and evaluates Lorenz/Wiener systems and feedback control. It argues for scoring the uncertainty as well as RMSE. |
| GCN or classifier uncertainty | Input-noise propagation and experimental risk/ranking examples | Node correlations, calibration, and suitable softmax/MC baselines. A synthetic graph or one Cora split does not establish general OOD performance. |

For a first use, run
[`regression_intervals`](../examples/regression_intervals.rs), then
[`conformal_intervals`](../examples/conformal_intervals.rs). For a new
application, learned-surrogate state estimation is a closer fit than
general-purpose classification confidence: there is an explicit uncertain input
and a downstream consumer of covariance. That is an application recommendation,
not a capability claim for an implemented Kalman filter.

Choose richer covariance only when the downstream decision benefits from it.
The exact Gaussian formulas are worth testing against the current series for
modest feature widths. Diagonal propagation remains useful for inexpensive
sensitivity estimates; learned low-rank structure may be preferable for large
output fields. Reverse that choice if measured decision quality or covariance
error justifies the added computation.
