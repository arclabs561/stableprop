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
from dataclasses import dataclass

import mpmath as mp

PINNED_MPMATH_VERSION = "1.4.1"
REFERENCE_DIGITS = 90
CROSS_CHECK_DIGITS = 50
MAXIMUM_ALLOWED_ERROR = "1e-40"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "rust"), default="json")
    args = parser.parse_args()
    if mp.__version__ != PINNED_MPMATH_VERSION:
        raise SystemExit(
            f"requires mpmath=={PINNED_MPMATH_VERSION}, found {mp.__version__}"
        )
    low_precision, _ = evaluate_all(CROSS_CHECK_DIGITS)
    high_precision, swap_errors = evaluate_all(REFERENCE_DIGITS)
    fixtures = [parse_fixture(row) for row in FIXTURES]
    rows = [
        row_for_output(fixture, high_precision[fixture.name]) for fixture in fixtures
    ]
    cross_check = {
        fixture.name: close_rows(
            low_precision[fixture.name], high_precision[fixture.name]
        )
        for fixture in fixtures
    }
    observed_cross_check = max(
        abs(high_precision[fixture.name][key] - low_precision[fixture.name][key])
        for fixture in fixtures
        for key in high_precision[fixture.name]
    )
    observed_swap = max(swap_errors.values())
    maximum_allowed_error = mp.mpf(MAXIMUM_ALLOWED_ERROR)
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
    payload = {
        "method": (
            "conditional one-dimensional Gaussian integration; "
            "Price correlation derivative"
        ),
        "mpmath_version": PINNED_MPMATH_VERSION,
        "digits": {"reference": REFERENCE_DIGITS, "cross_check": CROSS_CHECK_DIGITS},
        "fixtures": rows,
        "absolute_cross_check_error": cross_check,
        "maximum_absolute_error": {
            "accepted_limit": as_decimal(maximum_allowed_error),
            "cross_check": as_decimal(observed_cross_check),
            "pair_swap": as_decimal(observed_swap),
        },
        "pair_swap_error": {
            name: as_decimal(error) for name, error in swap_errors.items()
        },
    }
    output = (
        json.dumps(payload, indent=2, sort_keys=True)
        if args.format == "json"
        else rust_rows(rows)
    )
    print(output)


if __name__ == "__main__":
    main()
