# Gaussian moment propagation: derivations and design

This reference derives the identities behind the Gaussian APIs and the limits
of their approximations. The [method guide](methods.md) covers the literature
and application choices; the [selection note](sensitivity-and-selection.md)
covers conditioning, exploration, and learning objectives.

Throughout, vectors are columns, covariance is centered, and all stated moments
are finite. Write the mean and covariance of a vector as $\mu$ and $\Sigma$.
Gaussian identities use standard deviations $\sigma_{i}>0$, standard normal
density $\phi$, and CDF $\Phi$. A zero-variance coordinate is deterministic and
has zero covariance with every other coordinate. The notation
$\mathrm{diag}(a)$ makes a diagonal matrix from a vector $a$.

The vector API stores affine weights as `[output, input]`. Burn stores their
transpose, `[input, output]`. Each Burn batch row represents a separate
distribution; `MomentsFull` carries dependence between features within a row.

## Affine transport and discarded covariance

For a deterministic affine map, no Gaussian assumption is needed:

$$
Y=WX+b,\qquad
\mu_{Y}=W\mu_{X}+b,\qquad
\Sigma_{Y}=W\Sigma_XW^\top.
$$

**Proof.** Centering gives $Y-\mu_{Y}=W(X-\mu_{X})$. Take the expectation of its
outer product. Likewise, for $C=\mathrm{Cov}(U,X)$,

$$
\mathrm{Cov}(U,WX+b)=CW^\top.
$$

These identities justify composing affine layers without intermediate loss
when full covariance is retained. If a diagonal approximation discards the
off-diagonal matrix $E=\Sigma-\mathrm{diag}(\Sigma_{11},\ldots,\Sigma_{dd})$, the omitted variance of a
scalar output with weight vector $w$ is

$$
w^\top E w=2\sum_{i\lt j}w_iw_{j}\Sigma_{ij}.
$$

This can have either sign. A diagonal approximation can understate uncertainty
in a sum and overstate it in a difference. Even independent inputs can become
correlated after a dense affine map.

### Independent uncertain weights

Suppose the input coordinates, weight entries, and bias are mutually
independent. For one output, write input means and variances as $m_{i},v_{i}$,
weight means and variances as $a_{i},s_{i}$, and bias moments as $b,t$. Then

$$
\mathbb{E}[Y]=\sum_{i} a_{i} m_{i}+b,\qquad
\mathrm{Var}(Y)=\sum_{i}\left[a_{i}^2v_{i}+s_{i}(m_{i}^2+v_{i})\right]+t.
$$

**Proof.** Independence gives
$\mathbb{E}[W_{i}^2X_{i}^2]=(a_{i}^2+s_{i})(m_{i}^2+v_{i})$.
Subtract $a_{i}^2m_{i}^2$, then sum independent contributions.
This is the rule in `propagate_linear_bayes`. Setting $s_{i}=t=0$ recovers
deterministic diagonal affine propagation. Learning a posterior over weights,
and retaining correlations between its entries, require additional machinery.
Shared uncertain weights also couple batch rows, which this API does not retain.

A joint first-order expansion of $WX$ about the two means omits the product
of the input and weight perturbations. Under the independence assumptions
above, that omitted product contributes $s_{i}v_{i}$ to the variance of each
summand. Even with no activation-boundary crossing, a map that is affine in
its input for fixed weights need not be jointly affine in inputs and weights.

## Rectified Gaussian moments and derivatives

Let $X=\mu+\sigma Z$, with $Z$ standard normal, $v=\sigma^2$, and
$\alpha=\mu/\sigma$. For $X_+=\max(0,X)$, define its mean $m$, raw second
moment $q$, and variance $V$. Integration over $Z>-\alpha$ gives

$$
\begin{aligned}
m&=\mu\Phi(\alpha)+\sigma\phi(\alpha),\\
q&=(\mu^2+v)\Phi(\alpha)+\mu\sigma\phi(\alpha),\\
V&=q-m^2.
\end{aligned}
$$

**Proof.** Use $\phi'(z)=-z\phi(z)$ and integration by parts to obtain

$$
\int_{-\alpha}^{\infty}z\phi(z)\,dz=\phi(\alpha),\qquad
\int_{-\alpha}^{\infty}z^2\phi(z)\,dz
=\Phi(\alpha)-\alpha\phi(\alpha).
$$

Expand $(\mu+\sigma z)$ and its square under the integral. These are exact
Gaussian-input moments; the rectified distribution has an atom at zero and is
not Gaussian. See [Frey & Hinton (1999)](https://www.cs.toronto.edu/~hinton/absps/nlgbn.pdf)
for their use in nonlinear Gaussian belief networks.

At positive variance the derivatives simplify to

$$
\begin{aligned}
\frac{\partial m}{\partial\mu}&=\Phi(\alpha),&
\frac{\partial m}{\partial v}&=\frac{\phi(\alpha)}{2\sigma},\\
\frac{\partial V}{\partial\mu}&=2m[1-\Phi(\alpha)],&
\frac{\partial V}{\partial v}&=\Phi(\alpha)-\frac{m}{\sigma}\phi(\alpha).
\end{aligned}
$$

These provide gradient references independent of a tensor expression's
autodiff graph. At $\mu=0$, the mean's variance derivative diverges as
$v\downarrow0$. Finite gradients selected for exactly deterministic inputs
are implementation conventions, not that limit.

### Evaluating the negative tail

The preceding formulas subtract nearly equal quantities when $\alpha$ is
negative and large in magnitude. Put $t=-\alpha>0$ and define

$$
I_{n}(t)=\int_{0}^\infty u^n e^{-tu-u^2/2}\,du,\qquad
r_{n}=\frac{I_{n}}{I_{n-1}}\quad(n\geq1).
$$

Integration by parts gives $nI_{n-1}=tI_{n}+I_{n+1}$ and
$1=tI_{0}+I_{1}$. Consequently,

$$
r_{n}=\frac{n}{t+r_{n+1}},\qquad
I_{0}=\frac{1}{t+r_{1}},\qquad
\Phi(-t)=\phi(t)I_{0}.
$$

Substitution into the rectified moments gives

$$
\frac{m}{\sigma}=\phi(t)I_0r_{1},\qquad
\frac{q}{\sigma^2}=\phi(t)I_0r_1r_{2}.
$$

This is the connection between the
[Laplace continued fraction](https://dlmf.nist.gov/7.9) and
[repeated Gaussian tail integrals](https://dlmf.nist.gov/7.18#v).
The implementation evaluates a finite continued fraction below $\alpha=-2$.
Its recurrence uses tensor division because some CPU vector backends implement
reciprocal as a lower-precision estimate. This affects Burn 0.21's NdArray
backend; an [upstream fix](https://github.com/tracel-ai/burn/pull/5553) postdates
that release. The
[batched tail tests](../tests/relu_tail_accuracy.rs) check moments and derivatives
against independent references at sizes that exercise vector kernels.
The implementation selects zero and linear limits at $\alpha\leq-8$ and $\alpha\geq8$.
Thus the mathematical identities and their floating-point evaluation have
different exactness claims. The [method guide](methods.md#efficiency-and-accuracy)
describes precision and autodiff safeguards.

Burn's division backward can also use an approximate reciprocal. For ratios
with a positive standard deviation or tail denominator, the implementation
computes an untracked scale $s=1/b$, then evaluates $q=bs$ and $y=(as)/q$.
Holding the numerical scale fixed during differentiation gives

$$
\frac{\partial y}{\partial a}=\frac{s}{q}=\frac{1}{b},\qquad
\frac{\partial y}{\partial b}=-\frac{as^2}{q^2}=-\frac{a}{b^2}.
$$

The tracked inverse uses a denominator near one. This avoids both an
approximate reciprocal adjoint and an intermediate inverse square of a tiny
standard deviation. The final derivative is still subject to the chosen
floating-point range.

## One-sided nonlinear covariance and residuals

Let $U,V$ be jointly Gaussian scalars with $\mathrm{Var}(V)=\sigma_{V}^2>0$.
Suppose that $g$ is locally absolutely continuous and that both
$\mathbb{E}|g'(V)|$ and $\mathbb{E}|(V-\mu_{V})g(V)|$ are finite. Then

$$
\mathrm{Cov}(U,g(V))
=\mathrm{Cov}(U,V)\,\mathbb{E}[g'(V)].
$$

**Proof.** Gaussian conditioning gives
$\mathbb{E}[U-\mu_{U}\mid V]=\mathrm{Cov}(U,V)(V-\mu_{V})/\sigma_{V}^2$.
Gaussian integration by parts gives
$\mathbb{E}[(V-\mu_{V})g(V)]=\sigma_{V}^2\mathbb{E}[g'(V)]$.
Combining them proves the identity. ReLU is absolutely continuous and its
derivative is $1_{V>0}$ almost everywhere. Applying this identity to each coordinate with positive variance gives

$$
\mathrm{Cov}(U,\mathrm{ReLU}(V))=CB,\qquad
B=\mathrm{diag}\!\left(\Phi(\mu_{j}/\sigma_{j})\right),\qquad
C=\mathrm{Cov}(U,V).
$$

This is a centered form of the Gaussian cross-correlation identity associated
with [Bussgang's 1952 report](https://hdl.handle.net/1721.1/4847).
[Demir & Björnson (2020)](https://arxiv.org/html/2005.01597) derive its scalar
and vector forms and distinguish uncorrelated distortion from independent noise.

More generally, for a componentwise function $g$, let $B$ be the diagonal
matrix of expected slopes. The centered output then admits the decomposition

$$
g(V)-\mathbb{E}[g(V)]=B(V-\mu_{V})+\varepsilon,\qquad
\mathrm{Cov}(\varepsilon,V)=0.
$$

The coefficient is an **expected slope**, not the Jacobian evaluated at the
mean. For ReLU at zero mean these are respectively $1/2$ and the chosen
derivative at the kink. The residual $\varepsilon$ need not be Gaussian or
independent of $V$, and its coordinates need not be uncorrelated.

For a residual sum $Y=U+g(V)$ with matching feature widths, the two branches
contribute their own covariances and two cross terms:

$$
\Sigma_{Y}=\Sigma_{U}+\Sigma_{g(V)}+CB+B^\top C^\top.
$$

The cross-covariance helpers implement affine transport and the Gaussian ReLU
identity. `propagate_residual_add_correlated` uses the diagonal of the two
cross terms. Another nonlinear Gaussian step requires a new approximation
to the now non-Gaussian joint distribution.

## The covariance series

Let jointly standard normal coordinates have correlation matrix $R$.
Set $h_{i}(z)=g_{i}(\mu_{i}+\sigma_{i} z)$ and assume $\mathbb{E}[h_{i}(Z)^2]\lt \infty$.
Use the probabilists' Hermite polynomials, normalized by

$$
e^{tz-t^2/2}=\sum_{k=0}^{\infty}\mathrm{He}_{k}(z)\frac{t^k}{k!},\qquad
\mathbb{E}[\mathrm{He}_{k}(Z)\mathrm{He}_\ell(Z)]=k!\,1_{k=\ell}.
$$

Define coefficients $c_{ik}=\mathbb{E}[h_{i}(Z)\mathrm{He}_{k}(Z)]$.
The Hermite expansion then gives

$$
\mathrm{Cov}(g_{i}(X_{i}),g_{j}(X_{j}))
=\sum_{k=1}^{\infty}\frac{R_{ij}^k}{k!}c_{ik}c_{jk}.
$$

**Proof.** First take $|R_{ij}|\lt 1$. The expectation of the product of the two
generating functions is
$e^{R_{ij}st}$. Comparing coefficients yields

$$
\mathbb{E}[\mathrm{He}_{k}(Z_{i})\mathrm{He}_{\ell}(Z_{j})]
=1_{k=\ell}k!R_{ij}^k.
$$

Expand each centered transform in the orthogonal Hermite basis and take their
inner product. Parseval's identity gives
$\mathrm{Var}(h_{i}(Z))=\sum_{k\geq1}c_{ik}^2/k!$.
Cauchy–Schwarz bounds the sum of absolute covariance terms by the product of
standard deviations. This also proves convergence at $R_{ij}=\pm1$ without
assuming a nonsingular bivariate density.

For the Gaussian-smoothed mean $F_{i}(\mu)=\mathbb{E}[g_{i}(\mu+\sigma_iZ)]$,
differentiating the Gaussian density gives, for polynomially bounded activations
such as ReLU,

$$
c_{ik}=\sigma_{i}^k\frac{\partial^k F_{i}}{\partial\mu_{i}^k}.
$$

It is the smoothed mean that is differentiated; ordinary higher derivatives
of ReLU are not suitable substitutes. This is the covariance-series form in
[Wright, Nakahira & Moura (2024), Theorem 1](https://proceedings.mlr.press/v238/wright24a.html).
[Price's theorem](https://arxiv.org/html/1710.03576) relates derivatives with
respect to Gaussian covariance to expected derivatives of the transformed
function. Its distributional formulation handles nonsmooth functions, with
covariance derivatives taken in the positive-definite interior. The endpoint
value argument above instead uses the Hermite expansion in $L^2$.

### What a finite series preserves

Let $\Sigma_{g}=\mathrm{Cov}(g(X))$ denote the exact transformed covariance.
Write $D_{k}=\mathrm{diag}(c_{1k},\ldots,c_{dk})$ and let
$R^{\circ k}$ denote an entrywise power. Truncating after order $K$ gives

$$
S_{K}=\sum_{k=1}^{K}\frac{D_kR^{\circ k}D_{k}}{k!}.
$$

**Positive semidefiniteness, in exact arithmetic.** The Schur product theorem
makes every $R^{\circ k}$ positive semidefinite. Multiplication on both sides
by $D_{k}$ preserves that property, including signed coefficients. The sum is
therefore positive semidefinite. Replacing its diagonal with exact marginal
variances adds a nonnegative diagonal matrix:

$$
\widetilde S_{K}=S_{K}+\mathrm{diag}(\delta_{1K},\ldots,\delta_{dK}),\qquad
\delta_{iK}=\sum_{k>K}\frac{c_{ik}^2}{k!}\geq0.
$$

The omitted off-diagonal covariance obeys

$$
\left|\Sigma_{g,ij}-(S_{K})_{ij}\right|
\leq |R_{ij}|^{K+1}\sqrt{\delta_{iK}\delta_{jK}}.
$$

**Proof.** Bound each remaining power by $|R_{ij}|^{K+1}$ and apply
Cauchy–Schwarz to the coefficient tails. Thus small correlations suppress
truncation error. Near perfect correlation that factor gives little help.
The bound does not imply monotone improvement of every entry at each order,
or of every downstream decision.

These are exact-arithmetic properties using consistent coefficients and
exact marginal variances. Floating-point CDF evaluation, tail limits,
correlation clamping, and rounding can violate their premises. Tests of
quadratic forms check numerical behavior; they do not prove PSD for all inputs.

### ReLU coefficients and the implemented order

Differentiating the rectified mean gives

$$
c_{i1}=\sigma_{i}\Phi(\alpha_{i}),\qquad
c_{i2}=\sigma_{i}\phi(\alpha_{i}),\qquad
c_{i3}=-\sigma_{i}\alpha_{i}\phi(\alpha_{i}).
$$

For $k\geq2$, the general coefficient is

$$
c_{ik}=\sigma_{i}(-1)^k\mathrm{He}_{k-2}(\alpha_{i})\phi(\alpha_{i}).
$$

`propagate_relu_full` keeps the first three off-diagonal terms and evaluates
the univariate variances separately. The first term is $B\Sigma B^\top$ from
the Bussgang decomposition. Higher terms describe covariance of the nonlinear
residual. Retaining them generalizes the expected-slope approximation; it does
not make the output jointly Gaussian.

### Correlation-gradient error

Hold the marginal means and positive standard deviations fixed. Write
$C(\rho)$ for the exact ReLU pair covariance and $C_{K}(\rho)$ for its
truncation after order $K\geq1$. By
[Price's theorem, Corollary 2](https://arxiv.org/html/1710.03576#Thmthm2),

$$
C'(\rho)=\sigma_i\sigma_j\,\mathbb{P}(X_i>0,X_j>0),\qquad |\rho|\lt1.
$$

Thus the exact covariance increases with correlation. A finite series need
not preserve that sign. Its derivative error requires a separate bound; the
covariance error bound alone cannot be differentiated.

For ReLU, the weak derivative of $h_{i}(z)$ is
$\sigma_{i}1_{z>-\alpha_{i}}$. Define the omitted derivative energy

$$
\eta_{iK}
=\sum_{k>K}\frac{k\,c_{ik}^2}{k!}
=\sigma_i^2\Phi(\alpha_i)-\sum_{k=1}^{K}\frac{k\,c_{ik}^2}{k!}\geq0.
$$

**Proof.** Gaussian integration by parts gives

$$
\mathbb{E}[h_i'(Z)\mathrm{He}_{k-1}(Z)]=c_{ik}.
$$

Parseval applied to this square-integrable derivative gives
$\sum_{k\geq1}k\,c_{ik}^2/k!=\mathbb{E}[h_{i}'(Z)^2]$.
The derivative energies make the differentiated covariance series uniformly
absolutely convergent on $[-1,1]$. Bound each omitted power by
$|\rho|^K$ and apply Cauchy–Schwarz to obtain

$$
\left|C'(\rho)-C_K'(\rho)\right|
\leq |\rho|^K\sqrt{\eta_{iK}\eta_{jK}}.
$$

The endpoint statement uses one-sided derivatives. For the implemented order,

$$
\eta_{i3}=\sigma_i^2\left[
\Phi(\alpha_i)-\Phi(\alpha_i)^2
-\phi(\alpha_i)^2-\frac{\alpha_i^2\phi(\alpha_i)^2}{2}
\right].
$$

This controls an absolute derivative error at a Gaussian layer. It need not
give a useful relative error when the exact joint activation probability is
tiny. To differentiate with respect to input covariance instead of correlation,
divide both the derivative and its bound by $\sigma_{i}\sigma_{j}$. Bounds for
mean, variance, weight, or multilayer gradients require further analysis.

The energy subtraction can lose precision for nearly linear marginals;
$\eta_{iK}\leq\sigma_{i}^2\Phi(\alpha_{i})$ gives a looser upper bound without
that subtraction. Floating-point evaluation and clipped boundary gradients
remain separate concerns. The tests use interior correlations, moderate
standardized means, independent pair-probability references, and a scaled
roundoff allowance.

## Exact pairs, Gaussian-process kernels, and deeper networks

For zero-mean jointly Gaussian $X,Y$ with correlation $\rho$, polar integration
over the intersection of their positive half-planes gives

$$
\mathbb{E}[X_+Y_+]
=\frac{\sigma_{X}\sigma_{Y}}{2\pi}
\left[\sqrt{1-\rho^2}+(\pi-\arccos\rho)\rho\right].
$$

Subtract $\sigma_{X}\sigma_{Y}/(2\pi)$ to obtain centered covariance.
The same integral underlies the degree-one
[arc-cosine kernel of Cho & Saul (2009)](https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning).
In a neural-network Gaussian process it is typically a raw second moment
over random weights. Here it describes activation covariance under a supplied
input distribution for fixed weights. The integral is shared; the random
quantity and the centering convention differ.

For example, the pinned [Neural Tangents `ABRelu` implementation](https://github.com/google/neural-tangents/blob/c17e770bb74f1771da7be4a69fabfa68b6078960/neural_tangents/_src/stax/elementwise.py#L405-L499)
evaluates a zero-mean kernel using angle and square-root expressions. That
formula is not a replacement for nonzero-mean Gaussian activation covariance.
Nor does a neural tangent kernel by itself specify a posterior uncertainty model.

General means require bivariate Gaussian integrals. The 2026 preprints by
[Thompson & McCrory](https://arxiv.org/abs/2601.16830) and
[Kuang & Lin](https://arxiv.org/abs/2601.22307) give relevant exact Gaussian
moment formulas. For one hidden ReLU layer followed by an affine output,
exact pair moments suffice for exact output mean and covariance. Repeating
that calculation in a deeper network still replaces non-Gaussian layer inputs
by Gaussians. Removing covariance-series truncation removes one error source,
not this closure error.

An exact identity still needs numerical special functions. The
[Kuang–Lin implementation](https://github.com/simontheflutist/analytic-moments/blob/79186d364b41c93a4299eda52d260da7ee08a65a/neural_uncertainty_propagation/activation.py#L83-L122)
evaluates a bivariate Gaussian CDF contribution using 30-point Gaussian
quadrature. Replacing the series therefore requires checking integration error,
gradients near degenerate correlations, and runtime against the current path.

## Exact moments do not determine tail probabilities

Let $X\sim\mathcal{N}(0,1)$ and compare scores $s_{0}=0.1$ and
$s_{1}=X_{+}$. At the input mean, score zero wins. It loses after perturbation
exactly when $X>0.1$, so the flip probability is $1-\Phi(0.1)\approx0.4602$.
The Gaussian ReLU identities give exact moments for the margin $M=0.1-X_{+}$:

$$
\mathbb{E}[M]=0.1-\frac{1}{\sqrt{2\pi}},
\qquad
\text{Var}(M)=\frac12-\frac{1}{2\pi}.
$$

Replacing this margin by a Gaussian with those moments gives

$$
\Phi\!\left(-\frac{\mathbb{E}[M]}{\sqrt{\text{Var}(M)}}\right)
\approx0.6957.
$$

The true margin has an atom at 0.1 with probability one half. Its first two
moments do not specify that shape. This example has one hidden coordinate
and one ReLU: there is neither off-diagonal series truncation nor repeated
Gaussian closure. The [ranking control](../examples/pairwise_ranking_risk.rs)
checks the propagated moments and displays the probability gap. Improving
covariance alone cannot remove this error.

### When two moments suffice

For a scalar random output $Y$ with finite second moment and a fixed target
$t$, expand $Y-t=(Y-\mathbb{E}[Y])+(\mathbb{E}[Y]-t)$. The cross term has
zero expectation, giving

$$
\mathbb{E}[(Y-t)^2]=(\mathbb{E}[Y]-t)^2+\text{Var}(Y).
$$

This identity needs no Gaussian output assumption. Accurate moments therefore
suffice to evaluate expected squared deviation from a fixed setpoint, even
when they do not determine a threshold probability. For a neural surrogate
under execution noise, it gives a concrete objective for comparing candidate
settings; uncertainty about the surrogate itself remains a separate error.
The [`robust_selection` example](../examples/robust_selection.rs) compares these
choices against an independent quadratic simulator.

The mean in this identity is the expectation over perturbed inputs, which can
differ from the network at the input mean. The
[`robust_training` example](../examples/robust_training.rs) instead adds a
weighted variance penalty to point-prediction MSE. Even with unit weight, that
objective is not generally the expected noisy squared loss. If the target is
also random and correlated with the output, its variance and cross-covariance
must enter the expected-loss calculation too.

### An exact reference for candidate selection

Let the simulator be $h(x)=x^\top Qx+b^\top x+c$, with symmetric $Q$, and let
execution at nominal setting $u$ give $X=u+\epsilon$, where
$\epsilon\sim\mathcal{N}(0,\Sigma)$. Write $g=2Qu+b$. Expansion gives

$$
h(X)=h(u)+g^\top\epsilon+\epsilon^\top Q\epsilon.
$$

The linear term has zero mean. Gaussian third moments vanish, so it is
uncorrelated with the quadratic term. Using
$\mathbb{E}[\epsilon_{i}\epsilon_{j}\epsilon_{k}\epsilon_{l}]
=\Sigma_{ij}\Sigma_{kl}+\Sigma_{ik}\Sigma_{jl}+\Sigma_{il}\Sigma_{jk}$
for the quadratic term gives

$$
\begin{aligned}
\mathbb{E}[h(X)]&=h(u)+\text{tr}(Q\Sigma),\\
\text{Var}(h(X))&=g^\top\Sigma g+2\text{tr}(Q\Sigma Q\Sigma).
\end{aligned}
$$

Substitute these moments into the squared-loss identity above to obtain the
exact expected simulator loss $L(u)$ at each candidate. The example's
selectors only see a fitted ReLU surrogate; the simulator formula evaluates
their choices afterward. Regret is $L(\hat u)-\min_{u\in C}L(u)$ for the fixed
candidate set $C$. It is not regret against the best setting in a continuous
domain, and its reference does not have Monte Carlo selection bias.

### From score error to decision error

Suppose estimated losses $\hat L$ satisfy
$|\hat L(u)-L(u)|\leq\delta$ for every candidate. If $\hat u$ minimizes
$\hat L$ and $u^*$ minimizes $L$, then

$$
L(\hat u)\leq\hat L(\hat u)+\delta
\leq\hat L(u^*)+\delta\leq L(u^*)+2\delta.
$$

Thus uniform loss accuracy controls finite-candidate regret. An average
moment error does not establish this bound: a large error at one promising
candidate can change the choice.

There are several sources of loss error. For a fixed surrogate $f$, let
$L_f^a$ and $L_f^*$ denote its exact expected loss under the assumed and true
execution laws, and let $L_h^*$ be the true simulator loss. Then

$$
\hat L-L_h^*
=(\hat L-L_f^a)+(L_f^a-L_f^*)+(L_f^*-L_h^*).
$$

These terms separate propagation or quadrature error, noise-model error, and
surrogate error. The example compares each method with an independent
surrogate-Monte-Carlo reference under the assumed law, then compares reference
estimates across laws and against the analytic simulator. Those comparisons
still have Monte Carlo error; their absolute errors do not add to an exact
decomposition.

Surrogate error must be checked under the execution law. The triangle
inequality in $L^2$ gives

$$
\left|\sqrt{\mathbb{E}[(f(X)-t)^2]}
-\sqrt{\mathbb{E}[(h(X)-t)^2]}\right|
\leq\sqrt{\mathbb{E}[(f(X)-h(X))^2]}.
$$

A small fitting or clean-test RMSE under a different input distribution does
not supply this bound. Improving covariance propagation cannot correct a
surrogate or execution model that is wrong where candidates are evaluated.

## Design consequences and verification

The historical connections are distinct: Bussgang relates a nonlinear output
to its Gaussian input; Price differentiates Gaussian expectations with respect
to covariance; Hermite expansions resolve covariance by order; arc-cosine
kernels evaluate particular pair integrals. Modern moment-propagation methods
combine these tools with a choice of representation and inference objective.
The [history table](methods.md#how-the-methods-developed) places them alongside
Gaussian belief networks, PBP, DVI, and distprop's local linearization.

| Change | What it could improve | Required evidence |
| --- | --- | --- |
| [Reassociate or reuse tensor expressions](../benches/README.md) | Runtime and allocation without changing the moment approximation | Values and gradients across scales, CPU/Metal parity, repeated timings |
| [Exact bivariate ReLU evaluation](#exact-pairs-gaussian-process-kernels-and-deeper-networks) | Remove off-diagonal series truncation | Stable values and interior derivatives in tails and near perfect correlation; explicit zero-variance gradient conventions; matrix and decision errors |
| [Structured activation covariance](#the-covariance-series) | Reduce dense storage | Affine and nonlinear transport rules, recompression error, and a workload whose downstream result benefits |
| [Richer uncertainty inputs](../README.md#what-uncertainty-means-here) | Represent a different source of randomness | A model that supplies those distributions and evaluation against the intended observations |

A diagonal-plus-low-rank representation is not closed under these operations:
even $W\,\mathrm{diag}(v)W^\top$ is generally dense, and entrywise powers of
a low-rank correlation matrix can increase rank. Parameter-posterior methods
that use low-rank precision do not directly supply an activation-covariance
algorithm.

The validation layers correspond to different claims:

- [Tail references](../tests/relu_tail_accuracy.rs) compare moments and
  derivatives with independent high-precision Gaussian calculations.
- [Cross-covariance tests](../tests/cross_covariance.rs) exercise analytical
  identities and sampled joint distributions.
- [Property tests](../tests/burn_properties.rs) check transformations that
  should commute with propagation and numerical covariance constraints.
- [Gaussian pair references](../tests/relu_covariance_reference.rs) use the
  centered closed form and independent nonzero-mean quadrature fixtures to
  check covariance and correlation-gradient remainder bounds across feature
  scales. A separate test checks autodiff against the derivative of the finite
  series itself.
- [Extreme-scale gradients](../tests/burn_extreme_scales.rs) and
  [Metal comparisons](../tests/burn_metal.rs) test floating-point and backend
  behavior separately from the symbolic formulas.

Passing these checks is evidence about the implementation. It does not establish
that an application's noise model is correct, that its predictive intervals are
calibrated, or that selecting high-variance observations improves learning.
