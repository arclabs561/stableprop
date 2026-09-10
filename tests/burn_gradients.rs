#![cfg(feature = "burn")]

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_leaky_relu, propagate_relu, propagate_relu_full, Moments, MomentsFull,
};

type Ad = Autodiff<NdArray<f32>>;

#[test]
fn activation_boundary_conventions_keep_gradients_finite() {
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
fn positive_variance_relu_gradients_match_analytical_derivatives() {
    let device = Default::default();
    // Phi(-2), Phi(0), Phi(2), independently tabulated standard-normal CDFs.
    let cdf = [0.022_750_131_948_179_21, 0.5, 0.977_249_868_051_820_8];
    for variance in [1e-24f32, 0.25, 4.0] {
        let sigma = variance.sqrt();
        for mean_loss in [true, false] {
            let mean = Tensor::<Ad, 2>::from_data([[-2.0 * sigma, 0.0, 2.0 * sigma]], &device)
                .require_grad();
            let var = Tensor::<Ad, 2>::from_data([[variance; 3]], &device).require_grad();
            let out = propagate_relu(&Moments::new(mean.clone(), var.clone()));
            let gradients = if mean_loss {
                out.mean.sum()
            } else {
                out.var.sum()
            }
            .backward();
            let mean_gradient = mean
                .grad(&gradients)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let var_gradient = var
                .grad(&gradients)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for (i, a) in [-2.0f64, 0.0, 2.0].into_iter().enumerate() {
                let sigma = (variance as f64).sqrt();
                let pdf = (-0.5 * a * a).exp() / (2.0 * core::f64::consts::PI).sqrt();
                let rectified_mean = sigma * (pdf + a * cdf[i]);
                // dM/dmu = Phi; dM/dv = phi/(2 sigma).
                // dV/dmu = 2 M (1-Phi); dV/dv = Phi-M phi/sigma.
                let expected = if mean_loss {
                    [cdf[i], pdf / (2.0 * sigma)]
                } else {
                    [
                        2.0 * rectified_mean * (1.0 - cdf[i]),
                        cdf[i] - rectified_mean * pdf / sigma,
                    ]
                };
                for (actual, expected) in [mean_gradient[i], var_gradient[i]]
                    .into_iter()
                    .zip(expected)
                {
                    let tolerance = 1e-4 * expected.abs() + 1e-30;
                    assert!((actual as f64 - expected).abs() <= tolerance,
                        "mean_loss={mean_loss}, variance={variance:e}, a={a}: {actual:e} vs {expected:e}");
                }
            }
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
    // Closed-form zero-mean bivariate ReLU covariance, scaled by input variance.
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
