#![cfg(feature = "burn")]

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_linear_cross_covariance, propagate_relu_cross_covariance, Moments,
};

type Nd = NdArray<f32>;
type Ad = Autodiff<Nd>;

#[test]
fn affine_transport_preserves_left_right_orientation() {
    let device = Default::default();
    let cross = Tensor::<Nd, 3>::from_data([[[1.0, 2.0], [3.0, 4.0]]], &device);
    let weight = Tensor::<Nd, 2>::from_data([[2.0, -1.0], [0.5, 1.0]], &device);
    let actual = propagate_linear_cross_covariance(cross, weight)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(actual, vec![3.0, 1.0, 8.0, 1.0]);
}

#[test]
fn relu_cross_covariance_has_half_gate_and_linear_tail_limits() {
    let device = Default::default();
    let right = Moments::new(
        Tensor::<Nd, 2>::from_data([[0.0, 9.0, -9.0, 2.0]], &device),
        Tensor::<Nd, 2>::from_data([[1.0, 1.0, 1.0, 0.0]], &device),
    );
    let cross = Tensor::<Nd, 3>::from_data([[[0.2, 0.3, -0.4, 0.0]]], &device);
    let actual = propagate_relu_cross_covariance(cross, &right)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (&value, expected) in actual.iter().zip([0.1, 0.3, 0.0, 0.0]) {
        assert!((value - expected).abs() < 1e-6, "{actual:?}");
    }
}

#[test]
fn cross_covariance_gradients_are_finite_at_zero_tiny_and_tail_variances() {
    let device = Default::default();
    let mean = Tensor::<Ad, 2>::from_data([[0.0, 1e-12, 9.0, -9.0]], &device).require_grad();
    let var = Tensor::<Ad, 2>::from_data([[0.0, 1e-24, 1.0, 1.0]], &device).require_grad();
    let cross = Tensor::<Ad, 3>::from_data([[[0.0, 0.5e-24, 0.1, 0.1]]], &device).require_grad();
    let out =
        propagate_relu_cross_covariance(cross.clone(), &Moments::new(mean.clone(), var.clone()));
    let gradients = out.sum().backward();
    for values in [
        mean.grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        var.grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        cross
            .grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    ] {
        assert!(values.iter().all(|x| x.is_finite()), "{values:?}");
    }
}

#[test]
fn cross_covariance_gradient_matches_gaussian_cdf_derivative() {
    let device = Default::default();
    let mean = Tensor::<Ad, 2>::from_data([[0.3]], &device).require_grad();
    let var = Tensor::<Ad, 2>::from_data([[0.49]], &device).require_grad();
    let cross = Tensor::<Ad, 3>::from_data([[[0.2]]], &device).require_grad();
    let gradients =
        propagate_relu_cross_covariance(cross, &Moments::new(mean.clone(), var.clone()))
            .sum()
            .backward();
    let sigma = 0.7f64;
    let a = 0.3 / sigma;
    let pdf = (-0.5 * a * a).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let expected_mean = 0.2 * pdf / sigma;
    let expected_var = -0.2 * 0.3 * pdf / (2.0 * sigma.powi(3));
    let actual_mean = mean
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0] as f64;
    let actual_var = var
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0] as f64;
    assert!((actual_mean - expected_mean).abs() < 1e-6);
    assert!((actual_var - expected_var).abs() < 1e-6);
}

#[test]
fn relu_transport_matches_joint_gaussian_monte_carlo() {
    let device = Default::default();
    let right = Moments::new(
        Tensor::<Nd, 2>::from_data([[0.2, -0.3]], &device),
        Tensor::<Nd, 2>::from_data([[0.65, 0.90]], &device),
    );
    let cross = Tensor::<Nd, 3>::from_data([[[0.58, 0.57], [-0.75, 0.75]]], &device);
    let analytic = propagate_relu_cross_covariance(cross, &right)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let mut state = 0xC205_5C0Fu64;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
    };
    let mut normal = || (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
    let samples = 150_000;
    let mut sum_u = [0.0; 2];
    let mut sum_v = [0.0; 2];
    let mut product = [[0.0; 2]; 2];
    for _ in 0..samples {
        let (z0, z1) = (normal(), normal());
        let u = [0.4 + z0 + 0.3 * z1, -0.2 - 0.5 * z0 + z1];
        let v = [
            (0.2 + 0.7 * z0 - 0.4 * z1).max(0.0),
            (-0.3 + 0.3 * z0 + 0.9 * z1).max(0.0),
        ];
        for i in 0..2 {
            sum_u[i] += u[i];
            sum_v[i] += v[i];
            for j in 0..2 {
                product[i][j] += u[i] * v[j];
            }
        }
    }
    let n = samples as f64;
    for i in 0..2 {
        for j in 0..2 {
            let empirical = product[i][j] / n - sum_u[i] * sum_v[j] / (n * n);
            assert!((analytic[2 * i + j] as f64 - empirical).abs() < 0.01);
        }
    }
}

#[test]
#[should_panic(expected = "right features must match weight inputs")]
fn affine_transport_rejects_transposed_weight_layout() {
    let device = Default::default();
    let cross = Tensor::<Nd, 3>::zeros([2, 3, 4], &device);
    let weight = Tensor::<Nd, 2>::zeros([5, 4], &device);
    let _ = propagate_linear_cross_covariance(cross, weight);
}
