# Sensitivity, uncertainty, and data selection

Propagated variance can estimate disagreement under a specified perturbation
model. That can be useful for robustness analysis and training-data selection,
but it does not by itself measure what a new label would teach the model.

## What the score measures

| Quantity | What varies | What it answers |
| --- | --- | --- |
| Input-noise variance | Inputs, with model parameters fixed | How much does this prediction change under the chosen perturbations? |
| Parameter-posterior variance | Parameters drawn from a fitted posterior | How uncertain is the model about this prediction under that posterior? |
| Predictive label entropy | Possible labels under the predictive model | How ambiguous is the predicted label? |
| Per-example parameter gradient | Parameters in a specified loss | Which update would this labeled or pseudo-labeled example induce? |
| Acquisition utility | Possible observations followed by an update | Which observation is expected to improve the chosen learning objective? |

For small perturbations, input covariance is approximated by
`J_x S_x J_x^T`; parameter covariance by `J_theta S_theta J_theta^T`.
The similar algebra hides different random quantities. The latter represents
posterior uncertainty only when `S_theta` comes from an appropriate posterior
model. stableprop's Gaussian paths instead propagate moments through supported
layers; they need not use this first-order approximation at each activation.
See the [method history](methods.md) for the relationship to distprop.

## The direct connection to augmentation consistency

Fix an original input `x`. Let `Y1` and `Y2` be independent model outputs under
the same perturbation distribution given `x`. With finite conditional second moments:

$$
\mathbb{E}\left[\lVert Y_1-Y_2\rVert^2\mid x\right]
= 2\,\mathrm{tr}\!\left(\mathrm{Cov}(Y\mid x)\right).
$$

This identity is exact. It gives propagated covariance a concrete use:
estimating a squared-disagreement score without repeatedly sampling outputs.
The approximation lies in the propagated covariance and the perturbation model,
not in the identity. It does not apply to Cauchy outputs with undefined variance,
or to two dependent views without their cross-covariance term.

Hong et al. use disagreement under cutout/cutmix for active selection and
consistency regularization. Their experiments support a conditional connection
to learning value, but selection alone is not consistently strongest; combining
selection and regularization accounts for much of the reported benefit.[^hong]
Their image transformations are not additive Gaussian noise. A Gaussian
feature-noise score would be a different implementation that needs comparison.

GATTA extends test-time augmentation to graph active learning. Its August 2026
preprint reports improvements for entropy and least-confidence scores, mixed results for
more elaborate selectors, and failures under unsuitable augmentations.[^gatta]
Averaging a score over transformed predictions differs from scoring their
average. Neither is generally determined by the first two logit moments.
This crate's fixed adjacency operation also does not represent edge dropout.

A useful first test is therefore narrow: compare analytic and Monte Carlo
squared disagreement under the *same Gaussian feature-noise distribution*.
Only then test whether selecting by that score improves learning. Agreement
with Monte Carlo establishes the uncertainty calculation, not the selector.

## When posterior variance ranks Gaussian information gain

For a scalar Gaussian observation with independent noise variance `v_noise`
and posterior variance `v_model` in its latent mean, expected information about
the parameters from one observation is:

$$
I = \tfrac{1}{2}\log\!\left(1+\frac{v_{\mathrm{model}}}{v_{\mathrm{noise}}}\right).
$$

This is the Gaussian information-gain calculation used in Bayesian experimental
design.[^mackay] With the same observation noise across candidates, ranking
`v_model` also ranks this information gain. Ranking total predictive variance
can fail when noise differs:

| Candidate | Model variance | Observation-noise variance | Total variance | Information, nats |
| --- | ---: | ---: | ---: | ---: |
| A | 0.04 | 9.00 | 9.04 | 0.0022 |
| B | 4.00 | 1.00 | 5.00 | 0.8047 |

A has more predictive variance; B offers much more information about the
parameters. These numbers follow the stated Gaussian model, not an empirical
result from stableprop. Supplying input noise to a fixed network does not
supply the parameter posterior, likelihood, or update needed for this objective.

For classification, BALD measures mutual information between the label and
model parameters: predictive entropy minus the average conditional entropy
of posterior models.[^bald] It distinguishes model disagreement from
shared label ambiguity within the specified model. An approximate or
misspecified posterior can still give poor acquisition scores.

Two further choices matter. BatchBALD considers joint label information, so
selecting redundant points is different from taking the largest individual
scores.[^batchbald] EPIG targets information about predictions on a specified
target population, rather than information about all parameters.[^epig]
Even parameter information can be irrelevant to the predictions an application
cares about.

## From ranking uncertainty to exploration

Both noisy-input scoring and Bayesian reward models can produce a joint
distribution over candidate scores. The covariance has the same mathematical
role: the margin between scores `i` and `j` has variance
`C_ii + C_jj - 2 C_ij`. Under a Gaussian score model, that margin gives a
probability that one candidate outranks the other. Its interpretation comes
from the source of randomness: repeated input perturbations or beliefs about
unknown rewards.

For a neural-linear model with supplied posterior `beta ~ N(m, S)`, stack fixed
candidate features into the rows of `Phi`. The joint scores have mean `Phi m`
and covariance `Phi S Phi^T`. stableprop can compute this with its existing
full-covariance affine operations: treat the coefficient posterior as the input
distribution and candidate scores as output features. This retains shared
parameter uncertainty across candidates. With Burn, use one `MomentsFull`
batch row for the coefficient mean `m` and covariance `S`, then pass
`weight = Phi^T` to `propagate_linear_full`. Candidates occupy output features;
separate batch rows would lose their cross-covariance. The diagonal-weight helper
`propagate_linear_bayes` is a different representation.

To value an observation, add a likelihood. Suppose the latent reward vector
has posterior `theta ~ N(mu, C)` and observing candidate `a` gives
`Y_a = theta_a + epsilon`, with independent Gaussian noise of variance
`lambda_a` and `C_aa + lambda_a > 0`. Gaussian conditioning gives:

$$
\mu^+ = \mu + \frac{C_{:a}}{C_{aa}+\lambda_a}(Y_a-\mu_a),
\qquad
C^+ = C-\frac{C_{:a}C_{a:}}{C_{aa}+\lambda_a}.
$$

Here `C_:a` is a column and `C_a:` its corresponding row. Observing one
candidate updates every candidate correlated with it. These are the same
conditioning equations used in Gaussian-process regression and Kalman
measurement updates; the decision objective determines how they are used.
They are exact for this Gaussian observation model. Binary clicks require a
different likelihood and generally approximate inference.

The knowledge gradient values a measurement by the expected improvement in
the best posterior-mean choice:[^kg]

$$
\mathrm{KG}(a)
= \mathbb{E}_{Y_a}\!\left[\max_i \mu_i^+(Y_a)\right]-\max_i\mu_i.
$$

This is expected value of sample information for a risk-neutral terminal
choice. Selecting the action with largest KG is optimal with one observation
remaining and equal sampling costs under this objective. It need not be optimal
over a longer decision horizon. The correlated-normal
method generalizes earlier independent-normal ranking-and-selection policies:
off-diagonal covariance allows an observation to inform other alternatives.

Consider two scores `theta_1 = 1 + Z` and `theta_2 = Z`. Both may have large
variance, but their margin is always one. Learning the shared offset `Z`
reveals information without improving the choice; its knowledge gradient is
zero. Differential uncertainty can change the winner, but a flip probability
still omits the size of the reward improvement.

Online recommendation also earns or loses reward while gathering information.
Information-directed sampling addresses that objective by choosing an action
distribution that trades squared expected immediate regret against information
about the optimal action.[^ids] Its target differs from BALD's parameter
information and from the knowledge gradient's one-step terminal value.

stableprop supplies moment calculations within these systems. Posterior
fitting, conditioning, observation noise, and the acquisition policy belong
to the surrounding statistical model. Input uncertainty can also support a
decision to acquire a better measurement, if that measurement's likelihood
specifies what it reveals. The useful extension is therefore an explicit
joint-distribution and observation calculation, not a generic rule to explore
high-variance items.

## Contrastive learning

Probabilistic embedding methods such as HIB and PCME learn representation
means and scales through matching objectives.[^hib][^pcme] MCInfoNCE studies
recovery of embedding uncertainty under an explicit generative model.[^mcinfonce]
Those learned scales are not automatically equal to uncertainty propagated
from a chosen input-noise distribution.

There is also task-specific precedent for uncertainty-guided contrastive
weighting. UACL uses uncertainty about pairwise clustering relationships,
while hard-negative methods weight similarity and account for false
negatives.[^uacl][^hardneg] This supports testing uncertainty-aware weighting;
it does not establish that input sensitivity estimates pair reliability.
The UACL publisher preview supports this broad mechanism; its full algorithm
and ablations were not available for verification.

The [tuplet example](../examples/tuplet_contrastive.rs) uses a different,
explicit objective: contrastive loss plus a propagated embedding-variance
penalty. It encourages stability under specified input noise. It neither
selects batches nor estimates whether a candidate negative is semantically
valid. Stronger invariance can remove useful distinctions if the perturbation
changes the meaning of an example.

## What an experiment should establish

A selection experiment needs an outcome beyond uncertainty agreement. Keep
pool, label budget, initialization, training recipe, and held-out target fixed.
Compare random selection, ordinary entropy or margin, analytic disagreement,
and Monte Carlo disagreement under identical perturbations. Add a diversity
baseline such as BADGE, whose embeddings are parameter gradients rather than
input Jacobians.[^badge]

Vary selection and variance regularization separately. Otherwise an improvement
from the training loss can be credited to the sampler. Report learning curves
against both labels and wall time, unperturbed and perturbed retrieval/classification
quality, and the composition of selected batches. Repeat across seeds.

Include noisy or irrelevant examples and perturbations that violate the task's
invariance assumptions. Active-learning experiments on heteroskedastic pools
show that uncertainty methods can over-select noise and underperform random
selection.[^noise] A consistency or relevance filter is another hypothesis to
ablate; agreement with the model's original prediction is not proof that a
transformation preserves the true label.

These checks would justify an experimental selector. Until then, propagated
variance is a diagnostic or regularizer with a defined noise model.

## Extensions worth testing

| Extension | New use | Required evidence |
| --- | --- | --- |
| Joint cross-covariance through affine/ReLU layers | Derive residual covariance; maintain state–measurement coupling for filtering | Joint-Gaussian Monte Carlo checks, residual example, finite gradients and tail handling |
| More accurate bivariate ReLU moments | Reduce the current covariance-series truncation error | A reference grid including degenerate correlations; a backend-compatible derivative implementation; decision-level benefit |
| Jointly uncertain dot-product moments | Score two uncertain embeddings | A consumer with both input distributions and their dependence modeled; comparison with fixed-candidate scoring |
| Analytic augmentation-consistency score | Reduce repeated feature-noise evaluations | Agreement with sampled disagreement, followed by a controlled acquisition study |
| Structured covariance | Reduce memory at larger feature widths | Explicit rank/projection policy and measured accuracy–memory tradeoff |

Cross-covariance is the smallest missing primitive for the residual and
filtering applications discussed in the [method guide](methods.md).
An exact Gaussian layer calculation still does not make a deep network's
pushforward Gaussian. Fixed-rank covariance also needs care: an affine map
turns a diagonal residual into a generally dense covariance. Keeping only its
diagonal is a projection, not an exact update.

## References

[^mackay]: David MacKay, 1992. [Information-Based Objective Functions for Active Data Selection](https://authors.library.caltech.edu/records/efefp-2j353/files/MACnc92c.pdf).
[^bald]: Neil Houlsby et al., 2011. [Bayesian Active Learning for Classification and Preference Learning](https://arxiv.org/abs/1112.5745).
[^batchbald]: Andreas Kirsch et al., NeurIPS 2019. [BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning](https://arxiv.org/abs/1906.08158).
[^epig]: Freddie Bickford Smith et al., AISTATS 2023. [Prediction-Oriented Bayesian Active Learning](https://proceedings.mlr.press/v206/bickfordsmith23a.html).
[^kg]: Peter Frazier, Warren Powell and Savas Dayanik, 2009. [The Knowledge-Gradient Policy for Correlated Normal Beliefs](https://people.orie.cornell.edu/pfrazier/pub/Correlated_main_paper.pdf).
[^ids]: Daniel Russo and Benjamin Van Roy, 2014 preprint; Operations Research, 2018. [Learning to Optimize Via Information-Directed Sampling](https://arxiv.org/abs/1403.5556).
[^hong]: SeulGi Hong et al., 2020 preprint. [Deep Active Learning with Augmentation-based Consistency Estimation](https://arxiv.org/abs/2011.02666).
[^gatta]: Zsombor Bánfi et al., August 2026 preprint. [GATTA: Graph Active Learning with Test-Time Augmentation](https://arxiv.org/abs/2608.15084).
[^hib]: Seong Joon Oh et al., ICLR 2019. [Modeling Uncertainty with Hedged Instance Embeddings](https://arxiv.org/abs/1810.00319).
[^pcme]: Sanghyuk Chun et al., CVPR 2021. [Probabilistic Embeddings for Cross-Modal Retrieval](https://arxiv.org/abs/2101.05068).
[^mcinfonce]: Michael Kirchhof et al., ICML 2023. [Probabilistic Contrastive Learning Recovers the Correct Aleatoric Uncertainty of Ambiguous Inputs](https://proceedings.mlr.press/v202/kirchhof23a.html).
[^uacl]: Luyao Chang, Leiting Chen and Chuan Zhou, 2025. [Uncertainty-Aware Contrastive Learning for deep clustering](https://www.sciencedirect.com/science/article/pii/S0925231225012408).
[^hardneg]: Joshua Robinson et al., ICLR 2021. [Contrastive Learning with Hard Negative Samples](https://arxiv.org/abs/2010.04592).
[^badge]: Jordan Ash et al., ICLR 2020. [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds](https://arxiv.org/abs/1906.03671).
[^noise]: Savya Khosla et al., 2023 revision. [Understanding and Improving Neural Active Learning on Heteroskedastic Distributions](https://arxiv.org/abs/2211.00928).
