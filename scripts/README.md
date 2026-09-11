# Gaussian ReLU references

`reference_relu.py` is an opt-in fixture generator. It does not import the
crate or evaluate its covariance series. Run it with its pinned dependency:

```sh
uv run --script scripts/reference_relu.py > references.json
uv run --script scripts/reference_relu.py --format rust
uv run --script scripts/reference_relu.py --check-fixtures
```

It embeds the current nonzero-mean inputs as decimal strings. For an interior
correlation, it conditions on one standard-normal coordinate and integrates
the conditional ReLU mean of the other. It uses the same construction for the
joint activation probability and one-dimensional formulas at rank-one
correlations.

The output includes marginal ReLU means, variances, and standard deviations;
the raw pair moment and covariance; the joint activation probability; and the
Price correlation derivative. It does not claim mean or standard-deviation
pair derivatives without a separate derivation and endpoint contract. See
[the covariance derivation](../docs/derivations.md) for the identities and
scope.

The 90-digit computation is cross-checked at 50-digit working precision and
for pair-swap symmetry.
The tool fails above `1e-40`. JSON is deterministic; Rust output matches the
current `NonzeroReluFixture` fields and is a review fragment, not a write to
tests.

`--check-fixtures` is an opt-in gate over the frozen
`tests/relu_covariance_reference.rs` array. Pass another path to check a copied
fixture source: `--check-fixtures path/to/relu_covariance_reference.rs`. It
rejects a missing, extra, reordered, or syntactically unrecognized fixture
instead of accepting a partial parse. Every input and generated Rust field is
compared to the independent 90-digit reference with a two-ULP `f64` allowance:
one rounding to binary64 and one decimal-literal rendering.

For exact binary32 marginal references, supply the input mean and positive
variance as hexadecimal bit words. The option is repeatable and emits JSON
with the ReLU `mean`, `variance`, and derivatives `d_mean_d_mean`,
`d_mean_d_variance`, `d_variance_d_mean`, and `d_variance_d_variance` evaluated
at those actual `f32` values:

```sh
uv run --script scripts/reference_relu.py --marginal-bits 0x3f000000 0x3f800000
```

The marginal path is separate from pair integration. It independently repeats
its computation at 50 and 90 decimal digits, rejecting a per-field relative
difference above `1e-40` (or a disagreeing zero). Its fixed precision supports
finite inputs with `|mean / sqrt(variance)| <= 8`, which covers the frozen
tail-boundary fixtures and prevents false zero variances in extreme tails.

Check the nine frozen exact-bit marginal literals without running pair
quadrature:

```sh
uv run --script scripts/reference_relu.py --check-marginal-fixtures
```

This strict gate rejects missing, extra, duplicate, or unrecognized
`F32MarginalReference` entries and compares all six values with a two-ULP
`f64` allowance. Marginal row order is immaterial because each row carries its
own exact input bits; pair rows remain ordered against their embedded parameter
list.
