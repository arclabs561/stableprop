#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Generate independently integrated Gaussian ReLU pair references.

This is an opt-in developer tool. It does not import stableprop or reproduce
its Hermite series. It conditions on one standard-normal coordinate and uses
mpmath quadrature for a one-dimensional Gaussian integral.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp

PINNED_MPMATH_VERSION = "1.4.1"
REFERENCE_DIGITS = 90
CROSS_CHECK_DIGITS = 50
MAXIMUM_ALLOWED_ERROR = "1e-40"
MAXIMUM_MARGINAL_ABSOLUTE_ALPHA = mp.mpf("8")
DEFAULT_FIXTURE_RELATIVE_PATH = Path("tests/relu_covariance_reference.rs")
DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent.parent / DEFAULT_FIXTURE_RELATIVE_PATH
DEFAULT_MARGINAL_FIXTURE_RELATIVE_PATH = Path("tests/relu_tail_accuracy.rs")
DEFAULT_MARGINAL_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / DEFAULT_MARGINAL_FIXTURE_RELATIVE_PATH
)
FIXTURES = (
    ("fixture_0", "-0.75", "0.8", "1.1", "1.7", "-0.8"),
    ("fixture_1", "1.4", "0.6", "-0.35", "1.3", "0.75"),
    ("fixture_2", "-1.25", "1.1", "-0.55", "0.7", "0.6"),
    ("fixture_3", "0.35", "1.7", "1.25", "0.5", "-0.65"),
    ("fixture_4", "-0.4", "1.2", "0.9", "0.8", "0.98"),
    ("fixture_5", "0.4", "1.2", "-0.9", "0.8", "-0.98"),
    ("fixture_6", "0.25", "0.9", "-0.45", "1.4", "1.0"),
    ("fixture_7", "0.25", "0.9", "-0.45", "1.4", "-1.0"),
)

# This deliberately recognizes only the frozen array's field order and field
# set. A permissive Rust parser would turn a changed test schema into a false
# green reference check.
RUST_NUMBER = r"[+-]?(?:\d(?:_?\d)*(?:\.\d(?:_?\d)*)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?"
RUST_FIXTURE = re.compile(
    rf"""NonzeroReluFixture\s*\{{\s*
    mean:\s*\[(?P<mean_left>{RUST_NUMBER})\s*,\s*(?P<mean_right>{RUST_NUMBER})\]\s*,\s*
    std:\s*\[(?P<std_left>{RUST_NUMBER})\s*,\s*(?P<std_right>{RUST_NUMBER})\]\s*,\s*
    cdf:\s*\[(?P<cdf_left>{RUST_NUMBER})\s*,\s*(?P<cdf_right>{RUST_NUMBER})\]\s*,\s*
    rho:\s*(?P<rho>{RUST_NUMBER})\s*,\s*
    covariance:\s*(?P<covariance>{RUST_NUMBER})\s*,\s*
    joint_activation_probability:\s*(?P<joint_activation_probability>{RUST_NUMBER})\s*,\s*
    \}}\s*,""",
    re.VERBOSE,
)
FIXTURE_ARRAY_MARKER = "const NONZERO_RELU_FIXTURES: [NonzeroReluFixture; 8] = ["
MARGINAL_FIXTURE_ARRAY_MARKER = (
    "const F32_MARGINAL_REFERENCES: [F32MarginalReference; 9] = ["
)
RUST_BITS = r"(?:0x[0-9a-fA-F](?:_?[0-9a-fA-F])*|\d(?:_?\d)*)"
RUST_TRIVIA = re.compile(r"(?:\s+|//[^\n]*(?:\n|$))*")
RUST_MARGINAL_FIXTURE = re.compile(
    rf"""F32MarginalReference\s*\{{\s*
    mean_bits:\s*(?P<mean_bits>{RUST_BITS})\s*,\s*
    variance_bits:\s*(?P<variance_bits>{RUST_BITS})\s*,\s*
    mean:\s*(?P<mean>{RUST_NUMBER})\s*,\s*
    variance:\s*(?P<variance>{RUST_NUMBER})\s*,\s*
    d_mean_d_mean:\s*(?P<d_mean_d_mean>{RUST_NUMBER})\s*,\s*
    d_mean_d_variance:\s*(?P<d_mean_d_variance>{RUST_NUMBER})\s*,\s*
    d_variance_d_mean:\s*(?P<d_variance_d_mean>{RUST_NUMBER})\s*,\s*
    d_variance_d_variance:\s*(?P<d_variance_d_variance>{RUST_NUMBER})\s*,\s*
    \}}\s*,""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Fixture:
    name: str
    mean_left: mp.mpf
    std_left: mp.mpf
    mean_right: mp.mpf
    std_right: mp.mpf
    rho: mp.mpf


def parse_fixture(row: tuple[str, str, str, str, str, str]) -> Fixture:
    return Fixture(row[0], *(mp.mpf(value) for value in row[1:]))


def phi(value: mp.mpf) -> mp.mpf:
    return mp.exp(-value * value / 2) / mp.sqrt(2 * mp.pi)


def normal_cdf(value: mp.mpf) -> mp.mpf:
    return mp.erfc(-value / mp.sqrt(2)) / 2


def normal_survival(value: mp.mpf) -> mp.mpf:
    """Return 1 - Phi(value) without subtracting a nearly unit CDF."""
    return mp.erfc(value / mp.sqrt(2)) / 2


def relu_mean(mean: mp.mpf, std: mp.mpf) -> mp.mpf:
    """Return the exact Gaussian ReLU mean for a strictly positive std."""
    alpha = mean / std
    probability = normal_cdf(alpha)
    density = phi(alpha)
    return mean * probability + std * density


def relu_moments(mean: mp.mpf, std: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Return exact Gaussian ReLU mean, variance, and its positive std."""
    alpha = mean / std
    probability = normal_cdf(alpha)
    density = phi(alpha)
    rectified_mean = mean * probability + std * density
    raw_second = (mean * mean + std * std) * probability + mean * std * density
    variance = raw_second - rectified_mean * rectified_mean
    return rectified_mean, variance, mp.sqrt(variance)


def relu_marginal_with_derivatives(mean: mp.mpf, variance: mp.mpf) -> dict[str, mp.mpf]:
    """Return exact Gaussian ReLU marginal moments and mean/variance derivatives."""
    std = mp.sqrt(variance)
    alpha = mean / std
    probability = normal_cdf(alpha)
    density = phi(alpha)
    rectified_mean, rectified_variance, _ = relu_moments(mean, std)
    values = {
        "mean": rectified_mean,
        "variance": rectified_variance,
        "d_mean_d_mean": probability,
        "d_mean_d_variance": density / (2 * std),
        "d_variance_d_mean": 2 * rectified_mean * normal_survival(alpha),
        "d_variance_d_variance": probability - rectified_mean * density / std,
    }
    if (
        not all(mp.isfinite(value) for value in values.values())
        or values["mean"] <= 0
        or values["variance"] <= 0
    ):
        raise ValueError("marginal reference contains a non-finite or false-zero moment")
    return values


def split_points(lower: mp.mpf, crossing: mp.mpf | None) -> list[mp.mpf]:
    """Supply ordered breakpoints for adaptive integration on [lower, infinity)."""
    points = [lower]
    if crossing is not None and crossing > lower:
        points.append(crossing)
    if 0 > lower:
        points.append(mp.mpf("0"))
    points.append(mp.inf)
    return sorted(set(points))


def joint_activation_probability(fixture: Fixture) -> mp.mpf:
    """P(X > 0, Y > 0) from a conditional one-dimensional integral."""
    alpha_left = fixture.mean_left / fixture.std_left
    alpha_right = fixture.mean_right / fixture.std_right
    rho = fixture.rho
    if rho == 1:
        return normal_cdf(min(alpha_left, alpha_right))
    if rho == -1:
        return max(normal_cdf(alpha_left) + normal_cdf(alpha_right) - 1, mp.mpf("0"))

    conditional_std = mp.sqrt(1 - rho * rho)
    lower = -alpha_left

    def integrand(z: mp.mpf) -> mp.mpf:
        conditional_alpha = (alpha_right + rho * z) / conditional_std
        return phi(z) * normal_cdf(conditional_alpha)

    crossing = -alpha_right / rho if rho else None
    return mp.quad(integrand, split_points(lower, crossing))


def raw_relu_pair_moment(fixture: Fixture) -> mp.mpf:
    """E[ReLU(X) ReLU(Y)] by conditioning on the first coordinate."""
    lower = -fixture.mean_left / fixture.std_left
    rho = fixture.rho
    if abs(rho) < 1:
        conditional_std = fixture.std_right * mp.sqrt(1 - rho * rho)

        def integrand(z: mp.mpf) -> mp.mpf:
            x = fixture.mean_left + fixture.std_left * z
            conditional_mean = fixture.mean_right + fixture.std_right * rho * z
            return x * phi(z) * relu_mean(conditional_mean, conditional_std)

        crossing = -fixture.mean_right / (fixture.std_right * rho) if rho else None
        return mp.quad(integrand, split_points(lower, crossing))

    sign = mp.sign(rho)

    def integrand(z: mp.mpf) -> mp.mpf:
        x = fixture.mean_left + fixture.std_left * z
        y = fixture.mean_right + sign * fixture.std_right * z
        return x * max(y, 0) * phi(z)

    if rho == 1:
        return mp.quad(
            integrand,
            [max(lower, -fixture.mean_right / fixture.std_right), mp.inf],
        )
    upper = fixture.mean_right / fixture.std_right
    return mp.mpf("0") if lower >= upper else mp.quad(integrand, [lower, upper])


def as_decimal(value: mp.mpf) -> str:
    return mp.nstr(value, 70)


def evaluate(fixture: Fixture) -> dict[str, mp.mpf]:
    mean_left, variance_left, std_left = relu_moments(
        fixture.mean_left, fixture.std_left
    )
    mean_right, variance_right, std_right = relu_moments(
        fixture.mean_right, fixture.std_right
    )
    raw_pair = raw_relu_pair_moment(fixture)
    covariance = raw_pair - mean_left * mean_right
    joint_probability = joint_activation_probability(fixture)
    return {
        "cdf_left": normal_cdf(fixture.mean_left / fixture.std_left),
        "cdf_right": normal_cdf(fixture.mean_right / fixture.std_right),
        "relu_mean_left": mean_left,
        "relu_variance_left": variance_left,
        "relu_std_left": std_left,
        "relu_mean_right": mean_right,
        "relu_variance_right": variance_right,
        "relu_std_right": std_right,
        "raw_relu_pair_moment": raw_pair,
        "relu_covariance": covariance,
        "joint_activation_probability": joint_probability,
        "d_relu_covariance_d_rho": fixture.std_left
        * fixture.std_right
        * joint_probability,
    }


def close_rows(
    low_precision: dict[str, mp.mpf], high_precision: dict[str, mp.mpf]
) -> dict[str, str]:
    return {
        key: as_decimal(abs(high_precision[key] - low_precision[key]))
        for key in high_precision
    }


def row_for_output(fixture: Fixture, values: dict[str, mp.mpf]) -> dict[str, object]:
    return {
        "name": fixture.name,
        "mean": [as_decimal(fixture.mean_left), as_decimal(fixture.mean_right)],
        "std": [as_decimal(fixture.std_left), as_decimal(fixture.std_right)],
        "rho": as_decimal(fixture.rho),
        "values": {key: as_decimal(value) for key, value in values.items()},
    }


def evaluate_all(digits: int) -> tuple[dict[str, dict[str, mp.mpf]], dict[str, mp.mpf]]:
    mp.mp.dps = digits
    rows = {fixture.name: evaluate(fixture) for fixture in map(parse_fixture, FIXTURES)}
    swap_errors: dict[str, mp.mpf] = {}
    for fixture in map(parse_fixture, FIXTURES):
        swapped = Fixture(
            fixture.name,
            fixture.mean_right,
            fixture.std_right,
            fixture.mean_left,
            fixture.std_left,
            fixture.rho,
        )
        swapped_values = evaluate(swapped)
        swap_errors[fixture.name] = max(
            abs(rows[fixture.name][key] - swapped_values[key])
            for key in (
                "raw_relu_pair_moment",
                "relu_covariance",
                "joint_activation_probability",
                "d_relu_covariance_d_rho",
            )
        )
    return rows, swap_errors


def rust_rows(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for row in rows:
        values = row["values"]
        assert isinstance(values, dict)
        lines.extend(
            [
                "NonzeroReluFixture {",
                f"    mean: [{row['mean'][0]}, {row['mean'][1]}],",
                f"    std: [{row['std'][0]}, {row['std'][1]}],",
                f"    cdf: [{values['cdf_left']}, {values['cdf_right']}],",
                f"    rho: {row['rho']},",
                f"    covariance: {values['relu_covariance']},",
                f"    joint_activation_probability: {values['joint_activation_probability']},",
                "},",
            ]
        )
    return "\n".join(lines)


def fixture_array_body(source: str, path: Path, marker: str, description: str) -> str:
    """Extract one frozen fixture array, rejecting ambiguous syntax."""
    occurrences = source.count(marker)
    if occurrences != 1:
        raise ValueError(
            f"{path}: expected exactly one {description} marker, found {occurrences}"
        )
    start = source.index(marker) + len(marker)
    depth = 1
    for index in range(start, len(source)):
        character = source[index]
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                if source[index + 1 :].lstrip().startswith(";"):
                    return source[start:index]
                break
    raise ValueError(f"{path}: {description} has no unambiguous closing ];")


def parsed_rust_fixtures(path: Path) -> list[dict[str, mp.mpf]]:
    """Parse every and only strictly recognized fixture literal in ``path``."""
    try:
        body = fixture_array_body(
            path.read_text(encoding="utf-8"),
            path,
            FIXTURE_ARRAY_MARKER,
            "frozen pair fixture array",
        )
    except OSError as error:
        raise ValueError(f"cannot read fixture source {path}: {error}") from error

    parsed: list[dict[str, mp.mpf]] = []
    position = 0
    while position < len(body):
        trivia = RUST_TRIVIA.match(body, position)
        assert trivia is not None
        position = trivia.end()
        if position == len(body):
            break
        fixture = RUST_FIXTURE.match(body, position)
        if fixture is None:
            raise ValueError(
                f"{path}: unrecognized fixture syntax at array offset {position}"
            )
        parsed.append(
            {
                key: mp.mpf(value.replace("_", ""))
                for key, value in fixture.groupdict().items()
            }
        )
        position = fixture.end()
    return parsed


def parsed_rust_marginal_fixtures(path: Path) -> list[dict[str, object]]:
    """Parse every and only strictly recognized f32 marginal literals in ``path``."""
    try:
        body = fixture_array_body(
            path.read_text(encoding="utf-8"),
            path,
            MARGINAL_FIXTURE_ARRAY_MARKER,
            "frozen marginal fixture array",
        )
    except OSError as error:
        raise ValueError(f"cannot read fixture source {path}: {error}") from error

    parsed: list[dict[str, object]] = []
    position = 0
    while position < len(body):
        trivia = RUST_TRIVIA.match(body, position)
        assert trivia is not None
        position = trivia.end()
        if position == len(body):
            break
        fixture = RUST_MARGINAL_FIXTURE.match(body, position)
        if fixture is None:
            raise ValueError(
                f"{path}: unrecognized marginal fixture syntax at array offset {position}"
            )
        groups = fixture.groupdict()
        parsed.append(
            {
                "mean_bits": int(groups["mean_bits"].replace("_", ""), 0),
                "variance_bits": int(groups["variance_bits"].replace("_", ""), 0),
                **{
                    key: mp.mpf(value.replace("_", ""))
                    for key, value in groups.items()
                    if key not in {"mean_bits", "variance_bits"}
                },
            }
        )
        position = fixture.end()
    return parsed


def rounded_f64_tolerance(expected: mp.mpf) -> mp.mpf:
    """Allow two ULPs: one f64 rounding and one decimal literal rendering."""
    return 2 * mp.mpf(math.ulp(float(expected)))


def f32_from_hex_bits(value: str, label: str) -> tuple[str, float]:
    """Parse one finite binary32 value supplied as an exact hexadecimal bit word."""
    try:
        bits = int(value, 0)
    except ValueError as error:
        raise ValueError(f"{label} must be a hexadecimal f32 bit word, got {value!r}") from error
    if not value.lower().startswith("0x") or not 0 <= bits <= 0xFFFF_FFFF:
        raise ValueError(f"{label} must be a hexadecimal f32 bit word, got {value!r}")
    result = struct.unpack(">f", bits.to_bytes(4, byteorder="big"))[0]
    if not math.isfinite(result):
        raise ValueError(f"{label} must encode a finite f32, got {value!r}")
    return f"0x{bits:08x}", result


def marginal_bit_inputs(values: list[list[str]]) -> list[tuple[str, str, mp.mpf, mp.mpf]]:
    """Decode validated positive-variance f32 bit pairs at their exact f32 values."""
    decoded: list[tuple[str, str, mp.mpf, mp.mpf]] = []
    for index, (mean_value, variance_value) in enumerate(values):
        mean_bits, mean = f32_from_hex_bits(mean_value, f"marginal pair {index} mean")
        variance_bits, variance = f32_from_hex_bits(
            variance_value, f"marginal pair {index} variance"
        )
        if variance <= 0:
            raise ValueError(
                f"marginal pair {index} variance must encode a positive f32, got {variance_bits}"
            )
        exact_mean = mp.mpf(mean)
        exact_variance = mp.mpf(variance)
        alpha = exact_mean / mp.sqrt(exact_variance)
        if abs(alpha) > MAXIMUM_MARGINAL_ABSOLUTE_ALPHA:
            raise ValueError(
                f"marginal pair {index} has |mean / sqrt(variance)|={alpha}; "
                f"the reference supports |alpha| <= {MAXIMUM_MARGINAL_ABSOLUTE_ALPHA}"
            )
        decoded.append((mean_bits, variance_bits, exact_mean, exact_variance))
    return decoded


def marginal_reference_rows(
    inputs: list[tuple[str, str, mp.mpf, mp.mpf]], digits: int
) -> list[dict[str, object]]:
    """Evaluate exact-f32 marginal inputs at a requested working precision."""
    mp.mp.dps = digits
    return [
        {
            "mean_bits": mean_bits,
            "variance_bits": variance_bits,
            "values": relu_marginal_with_derivatives(mean, variance),
        }
        for mean_bits, variance_bits, mean, variance in inputs
    ]


def validate_marginal_fixture_inputs(
    inputs: list[tuple[str, str, mp.mpf, mp.mpf]]
) -> None:
    """Reject a missing, extra, or duplicate exact-f32 marginal input pair."""
    if len(inputs) != 9:
        raise ValueError(
            f"expected 9 marginal fixtures, found {len(inputs)} "
            "(missing or extra fixtures are not accepted)"
        )
    bit_pairs = {(mean_bits, variance_bits) for mean_bits, variance_bits, _, _ in inputs}
    if len(bit_pairs) != len(inputs):
        raise ValueError("marginal fixtures contain a duplicate exact-f32 input pair")


def relative_marginal_error(
    low_precision: list[dict[str, object]], high_precision: list[dict[str, object]]
) -> mp.mpf:
    """Return the largest relative precision difference across marginal values."""
    errors = (
        (abs(high_value - low_value) / abs(high_value))
        if high_value != 0
        else (mp.mpf("0") if low_value == 0 else mp.inf)
        for low_row, high_row in zip(low_precision, high_precision)
        for key, high_value in high_row["values"].items()
        for low_value in [low_row["values"][key]]
    )
    return max(errors)


def marginal_rows_for_output(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Render marginal values without rounding their exact f32 inputs."""
    return [
        {
            "mean_bits": row["mean_bits"],
            "variance_bits": row["variance_bits"],
            "values": {
                key: as_decimal(value) for key, value in row["values"].items()
            },
        }
        for row in rows
    ]


def check_fixtures(path: Path, rows: list[dict[str, object]]) -> None:
    """Check frozen Rust literals against independently integrated references."""
    actual = parsed_rust_fixtures(path)
    if len(actual) != len(rows):
        raise ValueError(
            f"{path}: expected {len(rows)} fixtures, found {len(actual)} "
            "(missing or extra fixtures are not accepted)"
        )
    for index, (fixture, row, rust_values) in enumerate(
        zip(map(parse_fixture, FIXTURES), rows, actual)
    ):
        values = row["values"]
        assert isinstance(values, dict)
        expected = {
            "mean_left": fixture.mean_left,
            "mean_right": fixture.mean_right,
            "std_left": fixture.std_left,
            "std_right": fixture.std_right,
            "cdf_left": mp.mpf(values["cdf_left"]),
            "cdf_right": mp.mpf(values["cdf_right"]),
            "rho": fixture.rho,
            "covariance": mp.mpf(values["relu_covariance"]),
            "joint_activation_probability": mp.mpf(
                values["joint_activation_probability"]
            ),
        }
        for field, expected_value in expected.items():
            actual_value = rust_values[field]
            tolerance = rounded_f64_tolerance(expected_value)
            error = abs(actual_value - expected_value)
            if not mp.isfinite(actual_value) or error > tolerance:
                raise ValueError(
                    f"{path}: fixture {index} field {field} differs by {as_decimal(error)}; "
                    f"f64 rounding tolerance is {as_decimal(tolerance)}"
                )


def check_marginal_fixtures(path: Path) -> list[dict[str, object]]:
    """Check the frozen exact-f32 marginal references and return MP90 rows."""
    actual = parsed_rust_marginal_fixtures(path)
    inputs = marginal_bit_inputs(
        [
            [f"0x{row['mean_bits']:08x}", f"0x{row['variance_bits']:08x}"]
            for row in actual
        ]
    )
    validate_marginal_fixture_inputs(inputs)
    low_precision = marginal_reference_rows(inputs, CROSS_CHECK_DIGITS)
    high_precision = marginal_reference_rows(inputs, REFERENCE_DIGITS)
    observed_precision = relative_marginal_error(low_precision, high_precision)
    maximum_allowed_error = mp.mpf(MAXIMUM_ALLOWED_ERROR)
    if not mp.isfinite(observed_precision) or observed_precision > maximum_allowed_error:
        raise ValueError(
            "marginal reference did not converge: "
            f"relative cross-check={observed_precision}, limit={maximum_allowed_error}"
        )
    if len(actual) != len(high_precision):
        raise ValueError(
            f"{path}: parser ambiguity changed marginal fixture count"
        )
    for index, (rust_row, reference_row) in enumerate(zip(actual, high_precision)):
        if (
            rust_row["mean_bits"] != int(reference_row["mean_bits"], 0)
            or rust_row["variance_bits"]
            != int(reference_row["variance_bits"], 0)
        ):
            raise ValueError(f"{path}: marginal fixture {index} input bits changed")
        values = reference_row["values"]
        assert isinstance(values, dict)
        for field, expected_value in values.items():
            actual_value = rust_row[field]
            assert isinstance(actual_value, mp.mpf)
            tolerance = rounded_f64_tolerance(expected_value)
            error = abs(actual_value - expected_value)
            if not mp.isfinite(actual_value) or error > tolerance:
                raise ValueError(
                    f"{path}: marginal fixture {index} field {field} differs by "
                    f"{as_decimal(error)}; f64 rounding tolerance is "
                    f"{as_decimal(tolerance)}"
                )
    return high_precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "rust"), default="json")
    parser.add_argument(
        "--check-fixtures",
        nargs="?",
        const=DEFAULT_FIXTURE_PATH,
        type=Path,
        help=(
            "compare the frozen Rust fixture array with this independent reference; "
            f"defaults to repository {DEFAULT_FIXTURE_RELATIVE_PATH}"
        ),
    )
    parser.add_argument(
        "--marginal-bits",
        action="append",
        nargs=2,
        metavar=("MEAN_HEX", "VARIANCE_HEX"),
        default=[],
        help=(
            "emit exact Gaussian ReLU marginal moments and derivatives for a "
            "finite mean and positive variance supplied as f32 bit words; repeatable"
        ),
    )
    parser.add_argument(
        "--check-marginal-fixtures",
        nargs="?",
        const=DEFAULT_MARGINAL_FIXTURE_PATH,
        type=Path,
        help=(
            "compare the frozen exact-f32 marginal references with this independent "
            f"reference; defaults to repository {DEFAULT_MARGINAL_FIXTURE_RELATIVE_PATH}"
        ),
    )
    args = parser.parse_args()
    try:
        marginal_inputs = marginal_bit_inputs(args.marginal_bits)
    except ValueError as error:
        raise SystemExit(f"invalid --marginal-bits input: {error}") from error
    marginal_requested = bool(marginal_inputs) or args.check_marginal_fixtures is not None
    pair_requested = args.check_fixtures is not None or not marginal_requested
    if marginal_requested and args.format != "json":
        raise SystemExit("marginal reference modes require the default JSON format")
    if mp.__version__ != PINNED_MPMATH_VERSION:
        raise SystemExit(
            f"requires mpmath=={PINNED_MPMATH_VERSION}, found {mp.__version__}"
        )
    maximum_allowed_error = mp.mpf(MAXIMUM_ALLOWED_ERROR)
    marginal_output: list[dict[str, object]] = []
    marginal_precision_error: mp.mpf | None = None
    if marginal_inputs:
        marginal_low_precision = marginal_reference_rows(
            marginal_inputs, CROSS_CHECK_DIGITS
        )
        marginal_high_precision = marginal_reference_rows(
            marginal_inputs, REFERENCE_DIGITS
        )
        marginal_error = relative_marginal_error(
            marginal_low_precision, marginal_high_precision
        )
        if not mp.isfinite(marginal_error) or marginal_error > maximum_allowed_error:
            raise SystemExit(
                "marginal reference did not converge: "
                f"normalized cross-check={marginal_error}, limit={maximum_allowed_error}"
        )
        marginal_output = marginal_rows_for_output(marginal_high_precision)
        marginal_precision_error = marginal_error
    if args.check_marginal_fixtures is not None:
        try:
            checked_marginal_rows = check_marginal_fixtures(
                args.check_marginal_fixtures
            )
        except ValueError as error:
            raise SystemExit(f"marginal fixture check failed: {error}") from error
        marginal_output.extend(marginal_rows_for_output(checked_marginal_rows))
    payload = {
        "mpmath_version": PINNED_MPMATH_VERSION,
        "digits": {"reference": REFERENCE_DIGITS, "cross_check": CROSS_CHECK_DIGITS},
    }
    if marginal_output:
        payload["marginal_bits"] = marginal_output
    if marginal_precision_error is not None:
        payload["marginal_relative_cross_check_error"] = as_decimal(
            marginal_precision_error
        )
    if pair_requested:
        low_precision, _ = evaluate_all(CROSS_CHECK_DIGITS)
        high_precision, swap_errors = evaluate_all(REFERENCE_DIGITS)
        fixtures = [parse_fixture(row) for row in FIXTURES]
        rows = [
            row_for_output(fixture, high_precision[fixture.name]) for fixture in fixtures
        ]
        observed_cross_check = max(
            abs(high_precision[fixture.name][key] - low_precision[fixture.name][key])
            for fixture in fixtures
            for key in high_precision[fixture.name]
        )
        observed_swap = max(swap_errors.values())
        if not mp.isfinite(observed_cross_check) or not mp.isfinite(observed_swap):
            raise SystemExit("reference contains a non-finite precision or symmetry error")
        if (
            observed_cross_check > maximum_allowed_error
            or observed_swap > maximum_allowed_error
        ):
            raise SystemExit(
                f"reference did not converge: cross-check={observed_cross_check}, "
                f"pair-swap={observed_swap}, limit={maximum_allowed_error}"
            )
        if args.check_fixtures is not None:
            try:
                check_fixtures(args.check_fixtures, rows)
            except ValueError as error:
                raise SystemExit(f"fixture check failed: {error}") from error
        payload.update(
            {
                "method": (
                    "conditional one-dimensional Gaussian integration; "
                    "Price correlation derivative"
                ),
                "fixtures": rows,
                "absolute_cross_check_error": {
                    fixture.name: close_rows(
                        low_precision[fixture.name], high_precision[fixture.name]
                    )
                    for fixture in fixtures
                },
                "maximum_absolute_error": {
                    "accepted_limit": as_decimal(maximum_allowed_error),
                    "cross_check": as_decimal(observed_cross_check),
                    "pair_swap": as_decimal(observed_swap),
                },
                "pair_swap_error": {
                    name: as_decimal(error) for name, error in swap_errors.items()
                },
            }
        )
    output = (
        json.dumps(payload, indent=2, sort_keys=True)
        if args.format == "json"
        else rust_rows(rows)
    )
    print(output)


if __name__ == "__main__":
    main()
