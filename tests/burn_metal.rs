//! Local Apple-GPU parity and timing checks.
//!
//! The suite needs the `metal` feature on macOS. The gate lives on the module
//! declaration below instead of a file-level `#![cfg(...)]`, so other targets
//! and feature sets still compile and run a real check rather than reporting
//! success having compared nothing.

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "burn_metal/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[test]
fn burn_metal_suite_is_not_silently_empty() {
    // Always-true invariant: linear moment propagation is exact, so a scalar
    // affine layer maps N(mu, var) to N(w*mu + b, w^2 * var) with no backend.
    let moments = stableprop::Moments {
        mean: vec![3.0],
        cov: vec![vec![4.0]],
    };
    let out = stableprop::propagate_linear(&moments, &[vec![2.0]], &[1.0]);
    assert_eq!(out.mean, vec![7.0], "mean must be w*mu + b");
    assert_eq!(out.cov, vec![vec![16.0]], "variance must be w^2 * var");

    eprintln!(
        "tests/burn_metal.rs: enable feature metal on macOS to run the gated GPU parity suite"
    );
}
