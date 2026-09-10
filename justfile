set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

# List available recipes.
default:
    @just --list

# Run the checks used by CI. GPU tests require an explicit Metal run.
check: fmt-check lint test docs examples smoke

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
    cargo test --all-features --example cora_uncertainty --example active_selection

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
