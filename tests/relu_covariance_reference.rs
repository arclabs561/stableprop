#![cfg(feature = "burn")]

use burn::backend::Autodiff;
use burn::tensor::{DType, Tensor, TensorData};
use burn_ndarray::NdArray;
use proptest::prelude::*;
use stableprop::burn_sdp::{propagate_linear_full, propagate_relu_full, MomentsFull};

fn relu_moments_and_omitted_energy(mu: f64, sigma: f64, p: f64) -> (f64, f64, f64) {
    let alpha = mu / sigma;
    let phi = (-0.5 * alpha * alpha).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let mean = mu * p + sigma * phi;
    let raw_second = (mu * mu + sigma * sigma) * p + mu * sigma * phi;
    let variance = raw_second - mean * mean;
    let c1 = sigma * p;
    let c2 = sigma * phi;
    let c3 = -sigma * alpha * phi;
    let omitted_energy = variance - c1 * c1 - c2 * c2 / 2.0 - c3 * c3 / 6.0;
    (mean, variance, omitted_energy)
}

fn relu_derivative_tail_energy(sigma: f64, alpha: f64, p: f64) -> f64 {
    let phi_squared = (-alpha * alpha).exp() / (2.0 * std::f64::consts::PI);
    sigma * sigma * (p - p * p - phi_squared - alpha * alpha * phi_squared / 2.0)
}

fn relu_covariance_correlation_gradient(mean: [f64; 2], std: [f64; 2], rho: f64) -> f64 {
    type Ad = Autodiff<NdArray<f64>>;

    let device = Default::default();
    let sigma_product = std[0] * std[1];
    let mean = Tensor::<Ad, 2>::from_data(
        TensorData::new(mean.to_vec(), [1, 2]),
        (&device, DType::F64),
    );
    let covariance = Tensor::<Ad, 3>::from_data(
        TensorData::new(
            vec![
                std[0] * std[0],
                rho * sigma_product,
                rho * sigma_product,
                std[1] * std[1],
            ],
            [1, 2, 2],
        ),
        (&device, DType::F64),
    )
    .require_grad();
    let output = propagate_relu_full(&MomentsFull::new(mean, covariance.clone()));
    let gradients = output.cov.slice([0..1, 0..1, 1..2]).sum().backward();
    let gradient = covariance
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f64>()
        .unwrap();
    // A symmetric correlation changes both input off-diagonal entries. Sum
    // both partials before restoring the covariance unit dSigma_ij/d rho.
    sigma_product * (gradient[1] + gradient[2])
}

#[derive(Clone, Copy)]
struct ScalarReluOracle {
    mean: f64,
    variance: f64,
    mean_mu: f64,
    mean_var: f64,
    variance_mu: f64,
    variance_var: f64,
    coefficients: [f64; 3],
    coefficient_mu: [f64; 3],
    coefficient_var: [f64; 3],
}

fn scalar_relu_oracle(mu: f64, variance: f64, p: f64) -> ScalarReluOracle {
    let sigma = variance.sqrt();
    let alpha = mu / sigma;
    let phi = (-0.5 * alpha * alpha).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let mean = mu * p + sigma * phi;
    let raw_second = (mu * mu + variance) * p + mu * sigma * phi;
    let output_variance = raw_second - mean * mean;

    ScalarReluOracle {
        mean,
        variance: output_variance,
        mean_mu: p,
        mean_var: phi / (2.0 * sigma),
        variance_mu: 2.0 * mean * (1.0 - p),
        variance_var: p - mean * phi / sigma,
        coefficients: [sigma * p, sigma * phi, -sigma * alpha * phi],
        coefficient_mu: [phi, -alpha * phi, (alpha * alpha - 1.0) * phi],
        coefficient_var: [
            (p - alpha * phi) / (2.0 * sigma),
            (1.0 + alpha * alpha) * phi / (2.0 * sigma),
            -alpha.powi(3) * phi / (2.0 * sigma),
        ],
    }
}

#[derive(Clone, Copy)]
struct K3PairOracle {
    covariance: f64,
    covariance_mu: [f64; 2],
    covariance_var: [f64; 2],
    covariance_q: f64,
}

fn k3_pair_oracle(
    mu: [f64; 2],
    variance: [f64; 2],
    q: f64,
    p: [f64; 2],
) -> (ScalarReluOracle, ScalarReluOracle, K3PairOracle) {
    let left = scalar_relu_oracle(mu[0], variance[0], p[0]);
    let right = scalar_relu_oracle(mu[1], variance[1], p[1]);
    let sigma_product = (variance[0] * variance[1]).sqrt();
    let rho = q / sigma_product;
    let factorial = [1.0, 2.0, 6.0];
    let mut covariance = 0.0;
    let mut covariance_mu = [0.0; 2];
    let mut covariance_var = [0.0; 2];
    let mut covariance_rho = 0.0;

    for (k, denominator) in factorial.iter().enumerate() {
        let order = (k + 1) as i32;
        let factor = rho.powi(order) / denominator;
        let product = left.coefficients[k] * right.coefficients[k];
        covariance += factor * product;
        covariance_mu[0] += factor * left.coefficient_mu[k] * right.coefficients[k];
        covariance_mu[1] += factor * left.coefficients[k] * right.coefficient_mu[k];
        covariance_var[0] += factor * left.coefficient_var[k] * right.coefficients[k];
        covariance_var[1] += factor * left.coefficients[k] * right.coefficient_var[k];
        covariance_rho += order as f64 * rho.powi(order - 1) * product / denominator;
    }
    covariance_var[0] += covariance_rho * -rho / (2.0 * variance[0]);
    covariance_var[1] += covariance_rho * -rho / (2.0 * variance[1]);

    (
        left,
        right,
        K3PairOracle {
            covariance,
            covariance_mu,
            covariance_var,
            covariance_q: covariance_rho / sigma_product,
        },
    )
}

#[derive(Clone, Copy)]
struct NonzeroReluFixture {
    mean: [f64; 2],
    std: [f64; 2],
    cdf: [f64; 2],
    rho: f64,
    covariance: f64,
    joint_activation_probability: f64,
}

// mpmath at 90 decimal digits generated these values and agreed with an
// independent 50-digit run. For |rho| < 1, write X = mu_l + sigma_l Z;
// condition on Z and integrate X_+ E[Y_+ | Z] phi(Z), where Y | Z has mean
// mu_r + rho sigma_r Z and standard deviation sigma_r sqrt(1-rho^2). At
// rho = +/-1, integrate the resulting deterministic one-dimensional pair.
// The joint activation probabilities use the same conditioning, integrating
// phi(Z) Phi((alpha_r + rho Z) / sqrt(1-rho^2)) over Z > -alpha_l.
// The fixtures include positive and negative products alpha_l alpha_r, so the
// third Hermite coefficient has both product signs.
// Regenerate the Rust literals with: uv run --script scripts/reference_relu.py --format rust
const NONZERO_RELU_FIXTURES: [NonzeroReluFixture; 8] = [
    NonzeroReluFixture {
        mean: [-0.75, 1.1],
        std: [0.8, 1.7],
        cdf: [0.174_250_711_880_542_4, 0.7412030632713038],
        rho: -0.8,
        covariance: -0.09702746076403964,
        joint_activation_probability: 0.03918893688859086,
    },
    NonzeroReluFixture {
        mean: [1.4, -0.35],
        std: [0.6, 1.3],
        cdf: [0.9901846713713547, 0.39387605185723835],
        rho: 0.75,
        covariance: 0.22969598162322754,
        joint_activation_probability: 0.39387143501695365,
    },
    NonzeroReluFixture {
        mean: [-1.25, -0.55],
        std: [1.1, 0.7],
        cdf: [0.12790220398830822, 0.21601744600250372],
        rho: 0.6,
        covariance: 0.022873544351819026,
        joint_activation_probability: 0.07544406642249715,
    },
    NonzeroReluFixture {
        mean: [0.35, 1.25],
        std: [1.7, 0.5],
        cdf: [0.5815585947678864, 0.9937903346742238],
        rho: -0.65,
        covariance: -0.3183978902145745,
        joint_activation_probability: 0.5753761265388645,
    },
    NonzeroReluFixture {
        mean: [-0.4, 0.9],
        std: [1.2, 0.8],
        cdf: [0.3694413401817636, 0.869_705_482_863_191],
        rho: 0.98,
        covariance: 0.3316340030070209,
        joint_activation_probability: 0.3694413401817621,
    },
    NonzeroReluFixture {
        mean: [0.4, -0.9],
        std: [1.2, 0.8],
        cdf: [0.6305586598182364, 0.13029451713680886],
        rho: -0.98,
        covariance: -0.03683090939107107,
        joint_activation_probability: 5.263693173000107e-7,
    },
    NonzeroReluFixture {
        mean: [0.25, -0.45],
        std: [0.9, 1.4],
        cdf: [0.609408524566425, 0.3739428170267853],
        rho: 1.0,
        covariance: 0.3814289914100701,
        joint_activation_probability: 0.37394281702678533,
    },
    NonzeroReluFixture {
        mean: [0.25, -0.45],
        std: [0.9, 1.4],
        cdf: [0.609408524566425, 0.3739428170267853],
        rho: -1.0,
        covariance: -0.18027030895089536,
        joint_activation_probability: 0.0,
    },
];

fn assert_nonzero_relu_fixture(fixture: NonzeroReluFixture, scales: [f64; 2]) {
    let mean_left = fixture.mean[0] * scales[0];
    let mean_right = fixture.mean[1] * scales[1];
    let sigma_left = fixture.std[0] * scales[0];
    let sigma_right = fixture.std[1] * scales[1];
    let covariance_unit = sigma_left * sigma_right;
    let device = Default::default();
    let input = MomentsFull::new(
        Tensor::<NdArray<f64>, 2>::from_data(
            TensorData::new(vec![mean_left, mean_right], [1, 2]),
            (&device, DType::F64),
        ),
        Tensor::<NdArray<f64>, 3>::from_data(
            TensorData::new(
                vec![
                    sigma_left * sigma_left,
                    fixture.rho * covariance_unit,
                    fixture.rho * covariance_unit,
                    sigma_right * sigma_right,
                ],
                [1, 2, 2],
            ),
            (&device, DType::F64),
        ),
    );
    let output = propagate_relu_full(&input);
    let mean = output.mean.into_data().to_vec::<f64>().unwrap();
    let cov = output.cov.into_data().to_vec::<f64>().unwrap();
    let (expected_mean_left, expected_variance_left, delta_left) =
        relu_moments_and_omitted_energy(mean_left, sigma_left, fixture.cdf[0]);
    let (expected_mean_right, expected_variance_right, delta_right) =
        relu_moments_and_omitted_energy(mean_right, sigma_right, fixture.cdf[1]);
    assert!(delta_left >= 0.0 && delta_right >= 0.0);

    let mean_floor_left = 4096.0 * f64::EPSILON * (mean_left.abs() + sigma_left);
    let mean_floor_right = 4096.0 * f64::EPSILON * (mean_right.abs() + sigma_right);
    let variance_floor_left =
        4096.0 * f64::EPSILON * (mean_left * mean_left + sigma_left * sigma_left);
    let variance_floor_right =
        4096.0 * f64::EPSILON * (mean_right * mean_right + sigma_right * sigma_right);
    let covariance_floor = 4096.0 * f64::EPSILON * covariance_unit;
    let remainder_bound = fixture.rho.abs().powi(4) * (delta_left * delta_right).sqrt();
    let exact_covariance = fixture.covariance * scales[0] * scales[1];

    assert!((mean[0] - expected_mean_left).abs() <= mean_floor_left);
    assert!((mean[1] - expected_mean_right).abs() <= mean_floor_right);
    assert!((cov[0] - expected_variance_left).abs() <= variance_floor_left);
    assert!((cov[3] - expected_variance_right).abs() <= variance_floor_right);
    assert!(
        (cov[1] - exact_covariance).abs() <= remainder_bound + covariance_floor,
        "rho={}, scales={scales:?}, exact={exact_covariance}, implemented={}, bound={remainder_bound}, floor={covariance_floor}",
        fixture.rho,
        cov[1],
    );
    assert!((cov[1] - cov[2]).abs() <= covariance_floor);
}

#[test]
fn centered_relu_covariance_stays_within_series_remainder() {
    let device = Default::default();
    let pi = std::f64::consts::PI;
    // The exact centered bivariate Gaussian ReLU kernel has a closed form.
    // Its omitted even powers have nonnegative coefficients; their sum at
    // |rho| = 1 bounds the third-order remainder on [-1, 1].
    let max_remainder = (pi - 3.0) / (4.0 * pi);
    for rho in [-1.0f64, -0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 1.0] {
        let input = MomentsFull::new(
            Tensor::<NdArray<f32>, 2>::zeros([1, 2], &device),
            Tensor::from_data([[[1.0, rho as f32], [rho as f32, 1.0]]], &device),
        );
        let cov = propagate_relu_full(&input)
            .cov
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let exact = ((1.0 - rho * rho).sqrt() + (pi - rho.acos()) * rho - 1.0) / (2.0 * pi);
        let error = exact - cov[1] as f64;
        assert!(
            (-2e-6..=max_remainder + 2e-6).contains(&error),
            "rho={rho}: covariance={} reference={exact}",
            cov[1]
        );
        assert!((cov[0] as f64 - (0.5 - 1.0 / (2.0 * pi))).abs() < 2e-6);
        assert_eq!(cov[1], cov[2]);
        assert_eq!(cov[0], cov[3]);
        if rho == 1.0 {
            // Identical inputs have identical rectified outputs. Keeping exact
            // diagonals with truncated cross-covariance leaves a spurious
            // positive margin variance: a known approximation limit.
            let margin_var = cov[0] as f64 + cov[3] as f64 - 2.0 * cov[1] as f64;
            assert!((margin_var - 2.0 * max_remainder).abs() < 4e-6);
        }
    }
}

proptest! {
    // The exact pair kernel is independent from the implementation's Hermite
    // series.  At zero mean, c_3 vanishes; keeping three orders is therefore
    // the same as retaining the first two nonzero coefficients.  Parseval
    // gives the remaining marginal energy below.
    #[test]
    fn zero_mean_relu_k3_error_obeys_hermite_remainder(
        rho in prop_oneof![Just(-1.0f64), Just(1.0f64), -1.0f64..1.0f64],
        left_exponent in -6i32..=6,
        right_exponent in -6i32..=6,
    ) {
        let sigma_left = 10.0f64.powi(left_exponent);
        let sigma_right = 10.0f64.powi(right_exponent);
        let sigma_product = sigma_left * sigma_right;
        let device = Default::default();
        let input = MomentsFull::new(
            Tensor::<NdArray<f64>, 2>::zeros([1, 2], (&device, DType::F64)),
            Tensor::<NdArray<f64>, 3>::from_data(
                TensorData::new(
                    vec![
                        sigma_left * sigma_left,
                        rho * sigma_product,
                        rho * sigma_product,
                        sigma_right * sigma_right,
                    ],
                    [1, 2, 2],
                ),
                (&device, DType::F64),
            ),
        );
        let implemented = propagate_relu_full(&input)
            .cov
            .into_data()
            .to_vec::<f64>()
            .unwrap()[1];

        let pi = std::f64::consts::PI;
        let exact_centered = sigma_product
            * ((1.0 - rho * rho).sqrt() + (pi - rho.acos()) * rho - 1.0)
            / (2.0 * pi);
        let remaining_marginal_energy = 0.25 - 3.0 / (4.0 * pi);
        let remainder_bound = rho.abs().powi(4) * sigma_product * remaining_marginal_energy;
        // The pair oracle subtracts close terms near rho = 0, and the tensor
        // path obtains rho by two divisions. Scale the floor with the natural
        // covariance unit rather than imposing an absolute tolerance.
        let roundoff_floor = 512.0 * f64::EPSILON * sigma_product;

        prop_assert!(
            (exact_centered - implemented).abs() <= remainder_bound + roundoff_floor,
            "rho={rho}, sigmas=({sigma_left}, {sigma_right}), exact={exact_centered}, \
             implemented={implemented}, bound={remainder_bound}, floor={roundoff_floor}",
        );
    }
}

#[test]
fn nonzero_mean_covariance_gradient_matches_the_truncated_series() {
    let p = 0.158_655_253_931_457_05; // Phi(-1), independently tabulated.
    let phi_squared = (-1.0f64).exp() / (2.0 * std::f64::consts::PI);
    for rho in [-0.9, 0.0, 0.9] {
        let actual = relu_covariance_correlation_gradient([-1.0, -1.0], [1.0, 1.0], rho);
        // Both off-diagonal input entries change with rho. Differentiate the
        // cubic polynomial, independently of the tensor graph. At rho=-0.9,
        // this derivative is negative; the exact Gaussian covariance is
        // increasing by Price's identity. Autodiff correctness does not remove
        // the approximation error in the derivative.
        let expected = p * p + phi_squared * (rho + 0.5 * rho * rho);
        assert!(
            (actual - expected).abs() <= 4096.0 * f64::EPSILON,
            "rho={rho}, derivative={actual}, series derivative={expected}",
        );
        if rho == -0.9 {
            let exact = 1.452_984_385_414_640_2e-7;
            let derivative_tail_energy = relu_derivative_tail_energy(1.0, -1.0, p);
            let remainder_bound = rho.abs().powi(3) * derivative_tail_energy;
            assert!(
                (actual - exact).abs() <= remainder_bound + 4096.0 * f64::EPSILON,
                "rho={rho}, exact={exact}, derivative={actual}, bound={remainder_bound}",
            );
        }
    }
}

fn assert_composed_k3_moment_gradients(fixture: NonzeroReluFixture, scales: [f64; 2]) {
    // Interior, strictly positive-definite fixtures keep every perturbation
    // coordinate inside the Gaussian model. This validates derivatives of the
    // implemented K3 approximation, not derivatives of exact pair moments.
    type Ad = Autodiff<NdArray<f64>>;

    let mean_values = [fixture.mean[0] * scales[0], fixture.mean[1] * scales[1]];
    let std = [fixture.std[0] * scales[0], fixture.std[1] * scales[1]];
    let variance = [std[0] * std[0], std[1] * std[1]];
    let q = fixture.rho * std[0] * std[1];
    assert!(q * q < variance[0] * variance[1]);
    let weight_values = [0.7, -1.1];
    let variance_weight = 0.3;
    let (left, right, pair) = k3_pair_oracle(mean_values, variance, q, fixture.cdf);

    let expected_mean_gradient = [
        weight_values[0] * left.mean_mu
            + variance_weight
                * (weight_values[0] * weight_values[0] * left.variance_mu
                    + 2.0 * weight_values[0] * weight_values[1] * pair.covariance_mu[0]),
        weight_values[1] * right.mean_mu
            + variance_weight
                * (weight_values[1] * weight_values[1] * right.variance_mu
                    + 2.0 * weight_values[0] * weight_values[1] * pair.covariance_mu[1]),
    ];
    let expected_variance_gradient = [
        weight_values[0] * left.mean_var
            + variance_weight
                * (weight_values[0] * weight_values[0] * left.variance_var
                    + 2.0 * weight_values[0] * weight_values[1] * pair.covariance_var[0]),
        weight_values[1] * right.mean_var
            + variance_weight
                * (weight_values[1] * weight_values[1] * right.variance_var
                    + 2.0 * weight_values[0] * weight_values[1] * pair.covariance_var[1]),
    ];
    let expected_q_gradient =
        variance_weight * 2.0 * weight_values[0] * weight_values[1] * pair.covariance_q;
    let expected_weight_gradient = [
        left.mean
            + variance_weight
                * (2.0 * weight_values[0] * left.variance
                    + 2.0 * weight_values[1] * pair.covariance),
        right.mean
            + variance_weight
                * (2.0 * weight_values[1] * right.variance
                    + 2.0 * weight_values[0] * pair.covariance),
    ];

    let device = Default::default();
    let mean = Tensor::<Ad, 2>::from_data(
        TensorData::new(mean_values.to_vec(), [1, 2]),
        (&device, DType::F64),
    )
    .require_grad();
    let covariance = Tensor::<Ad, 3>::from_data(
        TensorData::new(vec![variance[0], q, q, variance[1]], [1, 2, 2]),
        (&device, DType::F64),
    )
    .require_grad();
    let weight = Tensor::<Ad, 2>::from_data(
        TensorData::new(weight_values.to_vec(), [2, 1]),
        (&device, DType::F64),
    )
    .require_grad();
    let output = propagate_linear_full(
        &propagate_relu_full(&MomentsFull::new(mean.clone(), covariance.clone())),
        weight.clone(),
        None,
    );
    let gradients = (output.mean.sum() + output.cov.sum().mul_scalar(variance_weight)).backward();
    let actual_mean = mean
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f64>()
        .unwrap();
    let actual_covariance = covariance
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f64>()
        .unwrap();
    let actual_weight = weight
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f64>()
        .unwrap();

    let tolerance = 8192.0 * f64::EPSILON;
    for i in 0..2 {
        assert!(
            (actual_mean[i] - expected_mean_gradient[i]).abs()
                <= tolerance * expected_mean_gradient[i].abs().max(1.0),
            "mean gradient {i}: {} vs {}",
            actual_mean[i],
            expected_mean_gradient[i],
        );
        let covariance_index = if i == 0 { 0 } else { 3 };
        assert!(
            (actual_covariance[covariance_index] - expected_variance_gradient[i]).abs()
                <= tolerance * expected_variance_gradient[i].abs().max(1.0),
            "variance gradient {i}: {} vs {}",
            actual_covariance[covariance_index],
            expected_variance_gradient[i],
        );
        assert!(
            (actual_weight[i] - expected_weight_gradient[i]).abs()
                <= tolerance * expected_weight_gradient[i].abs().max(1.0),
            "weight gradient {i}: {} vs {}",
            actual_weight[i],
            expected_weight_gradient[i],
        );
    }
    // The symmetric q coordinate changes both stored off-diagonal entries.
    let actual_q_gradient = actual_covariance[1] + actual_covariance[2];
    assert!(
        (actual_q_gradient - expected_q_gradient).abs()
            <= tolerance * expected_q_gradient.abs().max(1.0),
        "covariance gradient: {actual_q_gradient} vs {expected_q_gradient}",
    );
}

#[test]
fn composed_k3_moment_gradients_match_an_independent_oracle() {
    // Endpoint correlations are excluded: their mathematical derivative is
    // one-sided while the tensor clamp defines the implementation boundary.
    for fixture in NONZERO_RELU_FIXTURES.iter().take(6) {
        for scales in [[1.0, 1.0], [1e-6, 1e6], [1e6, 1e-6]] {
            assert_composed_k3_moment_gradients(*fixture, scales);
        }
    }
}

#[test]
fn nonzero_mean_covariance_gradient_obeys_hermite_remainder() {
    // Exact endpoint derivatives belong to the mathematical extension; the
    // tensor clamp selects its own boundary convention.
    for fixture in NONZERO_RELU_FIXTURES.iter().filter(|f| f.rho.abs() < 1.0) {
        for scales in [[1.0, 1.0], [1e-6, 1e6], [1e6, 1e-6]] {
            let mean = [fixture.mean[0] * scales[0], fixture.mean[1] * scales[1]];
            let std = [fixture.std[0] * scales[0], fixture.std[1] * scales[1]];
            let sigma_product = std[0] * std[1];
            let alpha_left = mean[0] / std[0];
            let alpha_right = mean[1] / std[1];
            let tail_left = relu_derivative_tail_energy(std[0], alpha_left, fixture.cdf[0]);
            let tail_right = relu_derivative_tail_energy(std[1], alpha_right, fixture.cdf[1]);
            assert!(tail_left >= 0.0 && tail_right >= 0.0);

            let actual = relu_covariance_correlation_gradient(mean, std, fixture.rho);
            let exact = sigma_product * fixture.joint_activation_probability;
            let remainder_bound = fixture.rho.abs().powi(3) * (tail_left * tail_right).sqrt();
            let roundoff_floor = 4096.0 * f64::EPSILON * sigma_product;
            assert!(
                (actual - exact).abs() <= remainder_bound + roundoff_floor,
                "rho={}, scales={scales:?}, exact={exact}, derivative={actual}, bound={remainder_bound}, floor={roundoff_floor}",
                fixture.rho,
            );
        }
    }
}

#[test]
fn nonzero_mean_relu_covariance_obeys_hermite_remainder() {
    for fixture in NONZERO_RELU_FIXTURES {
        assert_nonzero_relu_fixture(fixture, [1.0, 1.0]);
    }
}

proptest! {
    #[test]
    fn nonzero_mean_relu_covariance_preserves_hermite_bound_across_scales(
        fixture_index in 0usize..NONZERO_RELU_FIXTURES.len(),
        left_exponent in -6i32..=6,
        right_exponent in -6i32..=6,
    ) {
        assert_nonzero_relu_fixture(
            NONZERO_RELU_FIXTURES[fixture_index],
            [10.0f64.powi(left_exponent), 10.0f64.powi(right_exponent)],
        );
    }
}

proptest! {
    #[test]
    fn centered_relu_covariance_gradient_obeys_hermite_remainder(
        rho in -0.999f64..0.999,
        left_exponent in -6i32..=6,
        right_exponent in -6i32..=6,
    ) {
        let sigma_left = 10.0f64.powi(left_exponent);
        let sigma_right = 10.0f64.powi(right_exponent);
        let sigma_product = sigma_left * sigma_right;
        let actual = relu_covariance_correlation_gradient(
            [0.0, 0.0],
            [sigma_left, sigma_right],
            rho,
        );
        let pi = std::f64::consts::PI;
        let exact = sigma_product * (0.25 + rho.asin() / (2.0 * pi));
        let remainder_bound = rho.abs().powi(3) * sigma_product * (0.25 - 1.0 / (2.0 * pi));
        let roundoff_floor = 4096.0 * f64::EPSILON * sigma_product;

        prop_assert!(
            (actual - exact).abs() <= remainder_bound + roundoff_floor,
            "rho={rho}, sigmas=({sigma_left}, {sigma_right}), exact={exact}, \
             derivative={actual}, bound={remainder_bound}, floor={roundoff_floor}",
        );
    }
}
