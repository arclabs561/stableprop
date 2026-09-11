set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

# List available recipes.
default:
    @just --list

# Run the checks used by CI. GPU tests require an explicit Metal run.
check: fmt-check lint test docs examples smoke bench-check

# Format Rust source.
fmt:
    cargo fmt --all

# Check formatting without changing files.
fmt-check:
    cargo fmt --all --check

# Lint default and optional APIs, including examples and tests.
lint:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --all-targets --all-features -- -D warnings

# Test both APIs and the controls embedded in evaluation examples.
test:
    cargo test
    cargo test --all-features
    cargo test --all-features --examples

# Sample more propagation and calibration-rank cases; optionally set the count.
property-test cases="4096":
    PROPTEST_CASES="$1" cargo test --features burn --test properties --test burn_properties --test relu_covariance_reference --example grouped_intervals

# Regenerate independent values and compare them with frozen Rust fixtures.
# Requires uv; kept separate from the Python-free Rust checks.
reference-check:
    uv run --script scripts/reference_relu.py --check-fixtures
    uv run --script scripts/reference_relu.py --check-marginal-fixtures

# Build both sets of API docs with warnings treated as errors.
docs:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Compile every example in release mode.
examples:
    cargo build --release --examples --all-features

# Run the small examples exercised by CI.
smoke:
    cargo run --release --example basic
    cargo run --release --features burn --example uncertainty_sources
    cargo run --release --features burn --example pairwise_ranking_risk

# Run an example on CPU; forward remaining arguments to it.
example name *args:
    cargo run --release --features burn --example "$1" -- "${@:2}"

# Benchmark the f64 vector API; forward Criterion filters and options.
bench *args:
    cargo bench --bench reference -- "$@"

# Benchmark diagonal and full covariance on Burn's f32 CPU backend.
bench-burn *args:
    cargo bench --features burn --bench burn -- "$@"

# Execute benchmark fixtures and oracles once, without timing measurements.
bench-check:
    cargo bench --bench reference -- --test
    cargo bench --features burn --bench burn -- --test

# Compile the Metal tests and training example on macOS.
[macos]
metal-check:
    cargo check --test burn_metal --example robust_training --features metal

# Check GPU values and gradients, then print synchronized workload timings.
[macos]
metal-test:
    cargo test --release --features metal --test burn_metal -- --ignored --nocapture --test-threads=1

# Train the variance-penalty example on Metal.
[macos]
metal-train:
    cargo run --release --features metal --example robust_training -- --metal
