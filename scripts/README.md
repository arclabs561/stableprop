# Gaussian ReLU pair references

`reference_relu.py` is an opt-in fixture generator. It does not import the
crate or evaluate its covariance series. Run it with its pinned dependency:

```sh
uv run --script scripts/reference_relu.py > references.json
uv run --script scripts/reference_relu.py --format rust
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
