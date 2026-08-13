//! Cross-check stableprop's analytic moment propagation against Monte Carlo.
//!
//! propagate_linear is exact and propagate_relu is the Frey-Hinton moment match;
//! both should agree with the empirical moments of sampled Gaussians. MC is an
//! independent oracle that catches a wrong sign or missing term in the closed
//! forms, which the existing inequality/zero-mean unit tests do not exercise
//! (they check variance-reduction and one mean point, not the full mean+cov
//! against samples). Fixed-seed Box-Muller so the statistical check is stable.

use stableprop::{propagate_linear, propagate_relu, propagate_sequential, Layer, Moments};

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

#[test]
fn chained_linear_covariance_is_exact() {
    let input = Moments {
        mean: vec![0.5, -1.0],
        cov: vec![vec![0.8, 0.3], vec![0.3, 0.5]],
    };
    let w1 = vec![vec![1.0, 2.0], vec![-0.5, 1.0], vec![0.25, -1.5]];
    let b1 = vec![0.2, -0.1, 0.4];
    let w2 = vec![vec![2.0, -1.0, 0.5], vec![0.3, 0.7, -2.0]];
    let b2 = vec![-0.3, 0.8];

    let chained = propagate_linear(&propagate_linear(&input, &w1, &b1), &w2, &b2);
    let collapsed_w = vec![vec![2.625, 2.25], vec![-0.55, 4.3]];
    let collapsed_b = vec![0.4, -0.01];
    let collapsed = propagate_linear(&input, &collapsed_w, &collapsed_b);

    for i in 0..2 {
        assert!((chained.mean[i] - collapsed.mean[i]).abs() < 1e-12);
        for j in 0..2 {
            assert!((chained.cov[i][j] - collapsed.cov[i][j]).abs() < 1e-12);
        }
    }
}

#[test]
fn chained_nonlinear_moments_track_seeded_monte_carlo() {
    let layers = vec![
        Layer::Linear {
            weight: vec![vec![0.8, -0.2], vec![0.3, 0.7]],
            bias: vec![0.4, 0.2],
        },
        Layer::ReLU,
        Layer::Linear {
            weight: vec![vec![0.6, 0.4], vec![-0.2, 0.5]],
            bias: vec![0.1, 0.3],
        },
        Layer::ReLU,
        Layer::Linear {
            weight: vec![vec![1.0, -0.4]],
            bias: vec![0.2],
        },
    ];
    let input_mean = [0.5, -0.2];
    let input_std = [0.4, 0.3];
    let analytic = propagate_sequential(&layers, &input_mean, &input_std);

    let mut rng = Rng::new(0xC1A1_5EED);
    let samples = 500_000;
    let (mut sum, mut sumsq) = (0.0, 0.0);
    for _ in 0..samples {
        let x0 = input_mean[0] + input_std[0] * rng.normal();
        let x1 = input_mean[1] + input_std[1] * rng.normal();
        let h0 = (0.8 * x0 - 0.2 * x1 + 0.4).max(0.0);
        let h1 = (0.3 * x0 + 0.7 * x1 + 0.2).max(0.0);
        let z0 = (0.6 * h0 + 0.4 * h1 + 0.1).max(0.0);
        let z1 = (-0.2 * h0 + 0.5 * h1 + 0.3).max(0.0);
        let y = z0 - 0.4 * z1 + 0.2;
        sum += y;
        sumsq += y * y;
    }
    let n = samples as f64;
    let mc_mean = sum / n;
    let mc_var = sumsq / n - mc_mean * mc_mean;

    assert!((analytic.mean[0] - mc_mean).abs() < 0.02);
    assert!((analytic.cov[0][0] - mc_var).abs() < 0.02);
}
