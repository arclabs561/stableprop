//! Cross-check momentprop's analytic moment propagation against Monte Carlo.
//!
//! propagate_linear is exact and propagate_relu is the Frey-Hinton moment match;
//! both should agree with the empirical moments of sampled Gaussians. MC is an
//! independent oracle that catches a wrong sign or missing term in the closed
//! forms, which the existing inequality/zero-mean unit tests do not exercise
//! (they check variance-reduction and one mean point, not the full mean+cov
//! against samples). Fixed-seed Box-Muller so the statistical check is stable.

use momentprop::{propagate_linear, propagate_relu, Moments};

struct Rng {
    s: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed | 1 }
    }
    fn u(&mut self) -> f64 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        ((self.s >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn linear_propagation_matches_monte_carlo() {
    // Independent input dims; a dense W induces output covariance cross-terms
    // (W·Σ·Wᵀ), so MC validates the off-diagonal math, not just the diagonal.
    let mean = [1.0_f64, -2.0, 0.5];
    let std = [0.7_f64, 1.3, 0.4];
    let w = [vec![1.0, -0.5, 0.3], vec![0.2, 1.0, -0.8]];
    let b = [0.4_f64, -0.1];

    let moments = Moments {
        mean: mean.to_vec(),
        cov: (0..3)
            .map(|i| {
                let mut row = vec![0.0; 3];
                row[i] = std[i] * std[i];
                row
            })
            .collect(),
    };
    let analytic = propagate_linear(&moments, &w, &b);

    let n = 400_000;
    let mut rng = Rng::new(0xD157_9407);
    let mut sum = [0.0; 2];
    let mut sum2 = [[0.0; 2]; 2];
    for _ in 0..n {
        let x: Vec<f64> = (0..3).map(|i| mean[i] + std[i] * rng.normal()).collect();
        let y: Vec<f64> = (0..2)
            .map(|o| b[o] + (0..3).map(|i| w[o][i] * x[i]).sum::<f64>())
            .collect();
        for o in 0..2 {
            sum[o] += y[o];
            for p in 0..2 {
                sum2[o][p] += y[o] * y[p];
            }
        }
    }
    let nf = n as f64;
    let emp_mean: Vec<f64> = sum.iter().map(|s| s / nf).collect();
    for o in 0..2 {
        assert!(
            (analytic.mean[o] - emp_mean[o]).abs() < 0.02,
            "mean[{o}] analytic {} vs MC {}",
            analytic.mean[o],
            emp_mean[o]
        );
        for p in 0..2 {
            let emp_cov = sum2[o][p] / nf - emp_mean[o] * emp_mean[p];
            assert!(
                (analytic.cov[o][p] - emp_cov).abs() < 0.03,
                "cov[{o}][{p}] analytic {} vs MC {}",
                analytic.cov[o][p],
                emp_cov
            );
        }
    }
}

#[test]
fn relu_moments_match_monte_carlo() {
    let n = 600_000;
    for &(mu, sigma) in &[
        (1.0_f64, 1.0_f64),
        (-1.0, 1.0),
        (0.0, 1.0),
        (2.0, 0.5),
        (-0.5, 2.0),
        (3.0, 0.3),
    ] {
        let moments = Moments {
            mean: vec![mu],
            cov: vec![vec![sigma * sigma]],
        };
        let analytic = propagate_relu(&moments);

        let mut rng = Rng::new(0x5EED ^ (mu.to_bits() ^ sigma.to_bits()));
        let (mut s, mut s2) = (0.0_f64, 0.0_f64);
        for _ in 0..n {
            let x = (mu + sigma * rng.normal()).max(0.0);
            s += x;
            s2 += x * x;
        }
        let nf = n as f64;
        let emp_mean = s / nf;
        let emp_var = s2 / nf - emp_mean * emp_mean;
        assert!(
            (analytic.mean[0] - emp_mean).abs() < 0.01,
            "ReLU mean (mu={mu}, σ={sigma}): analytic {} vs MC {}",
            analytic.mean[0],
            emp_mean
        );
        assert!(
            (analytic.cov[0][0] - emp_var).abs() < 0.02,
            "ReLU var (mu={mu}, σ={sigma}): analytic {} vs MC {}",
            analytic.cov[0][0],
            emp_var
        );
    }
}
