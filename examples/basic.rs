//! Propagate independent input noise through an affine layer and a ReLU.
//!
//! The difference of the two inputs has standard deviation 0.5. Rectifying
//! that zero-mean Gaussian gives mean sigma/sqrt(2*pi) and variance
//! sigma^2 * (1/2 - 1/(2*pi)). See full_covariance for a tensor comparison.
//!
//! Run: `cargo run --release --example basic`

use stableprop::{propagate_sequential, Layer};

fn main() {
    let layers = [
        Layer::Linear {
            weight: vec![vec![1.0, -1.0]],
            bias: vec![0.0],
        },
        Layer::ReLU,
    ];
    let output = propagate_sequential(&layers, &[0.0, 0.0], &[0.3, 0.4]);
    println!(
        "mean = {:.4}, variance = {:.4}",
        output.mean[0], output.cov[0][0]
    );
}
