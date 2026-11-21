//! Copy a zero-mean Gaussian into two channels, rectify both and subtract.

use stableprop::{propagate_sequential, Layer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let std_dev: f64 = args.next().as_deref().unwrap_or("1").parse()?;
    if args.next().is_some() || std_dev < 0.0 || !std_dev.powi(2).is_finite() {
        return Err("usage: duplicate_relu [standard deviation with finite variance]".into());
    }

    let layers = [
        Layer::Linear {
            weight: vec![vec![1.0], vec![1.0]],
            bias: vec![0.0, 0.0],
        },
        Layer::ReLU,
        Layer::Linear {
            weight: vec![vec![1.0, -1.0]],
            bias: vec![0.0],
        },
    ];
    let output = propagate_sequential(&layers, &[0.0], &[std_dev]);

    // Both channels carry the same value for every draw, so they cancel.
    println!("exact variance: 0");
    println!("propagated variance: {:.6}", output.cov[0][0]);
    Ok(())
}
