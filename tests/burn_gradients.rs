#![cfg(feature = "burn")]

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_leaky_relu, propagate_relu, propagate_relu_full, Moments, MomentsFull,
};

type Ad = Autodiff<NdArray<f32>>;

#[test]
fn activation_gradients_remain_finite_at_zero_and_small_noise() {
    let device = Default::default();
    for full in [false, true] {
        let mean = Tensor::<Ad, 2>::from_data(
            TensorData::new(vec![0.0f32, 1e-12, 10_000.0, -1.0], [1, 4]),
            &device,
        )
        .require_grad();
        let var = Tensor::<Ad, 2>::from_data(
            TensorData::new(vec![0.0f32, 1e-24, 1.0, 0.0], [1, 4]),
            &device,
        )
        .require_grad();
        let loss = if full {
            let out = propagate_relu_full(&MomentsFull::from_diagonal(mean.clone(), var.clone()));
            out.variance().sum() + out.mean.sum()
        } else {
            let moments = Moments::new(mean.clone(), var.clone());
            let relu = propagate_relu(&moments);
            let leaky = propagate_leaky_relu(&moments, 0.3);
            relu.mean.sum() + relu.var.sum() + leaky.mean.sum() + leaky.var.sum()
        };
        let grads = loss.backward();
        for gradient in [mean.grad(&grads).unwrap(), var.grad(&grads).unwrap()] {
            let values = gradient.to_data().to_vec::<f32>().unwrap();
            assert!(
                values.iter().all(|x| x.is_finite()),
                "full={full}: {values:?}"
            );
        }
    }
}

#[test]
fn tiny_correlated_relu_preserves_scale_and_finite_gradients() {
    let device = Default::default();
    let mean = Tensor::<Ad, 2>::zeros([1, 2], &device).require_grad();
    let cov = Tensor::<Ad, 3>::from_data(
        TensorData::new(vec![1e-24f32, 0.5e-24, 0.5e-24, 1e-24], [1, 2, 2]),
        &device,
    )
    .require_grad();
    let out = propagate_relu_full(&MomentsFull::new(mean.clone(), cov.clone()));
    let values = out.cov.to_data().to_vec::<f32>().unwrap();
    // Independent zero-mean bivariate ReLU formula, scaled by input variance.
    let rho = 0.5f32;
    let pi = core::f32::consts::PI;
    let exact = ((1.0 - rho * rho).sqrt() + (pi - rho.acos()) * rho - 1.0) / (2.0 * pi);
    assert!((values[1] / 1e-24 - exact).abs() < 0.001);
    assert_eq!(values[1], values[2]);
    let grads = (out.mean.sum() + out.cov.sum()).backward();
    for values in [
        mean.grad(&grads)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
        cov.grad(&grads).unwrap().to_data().to_vec::<f32>().unwrap(),
    ] {
        assert!(values.iter().all(|x| x.is_finite()), "{values:?}");
    }
}

#[test]
fn inactive_relu_tail_has_zero_covariance_with_other_features() {
    let device = Default::default();
    let mean = Tensor::<Ad, 2>::from_data([[-9.0f32, 0.0]], &device).require_grad();
    let cov = Tensor::<Ad, 3>::from_data([[[1.0f32, 0.5], [0.5, 1.0]]], &device).require_grad();
    let out = propagate_relu_full(&MomentsFull::new(mean.clone(), cov.clone()));
    let values = out.cov.to_data().to_vec::<f32>().unwrap();
    // A deterministic zero output cannot covary with another feature.
    assert_eq!(&values[..3], &[0.0; 3]);
    let grads = (out.mean.sum() + out.cov.sum()).backward();
    assert!(mean
        .grad(&grads)
        .unwrap()
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|x| x.is_finite()));
    assert!(cov
        .grad(&grads)
        .unwrap()
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|x| x.is_finite()));
}
