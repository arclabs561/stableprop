#![cfg(feature = "burn")]

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_leaky_relu, propagate_relu, propagate_relu_cross_covariance, propagate_relu_full,
    Moments, MomentsFull,
};

type Ad = Autodiff<NdArray<f32>>;

#[test]
fn finite_gradients_for_representable_subnormal_variances() {
    let device = Default::default();
    for variance in [f32::MIN_POSITIVE, 1e-40f32] {
        let sigma = variance.sqrt();
        for mode in ["relu", "leaky", "full", "cross"] {
            let mean =
                Tensor::<Ad, 2>::from_data([[0.0, sigma, 9.0 * sigma, -9.0 * sigma]], &device)
                    .require_grad();
            let var = Tensor::<Ad, 2>::from_data([[variance; 4]], &device).require_grad();
            let moments = Moments::new(mean.clone(), var.clone());
            let loss = match mode {
                "relu" => {
                    let out = propagate_relu(&moments);
                    out.mean.sum() + out.var.sum()
                }
                "leaky" => {
                    let out = propagate_leaky_relu(&moments, 0.3);
                    out.mean.sum() + out.var.sum()
                }
                "full" => {
                    let out =
                        propagate_relu_full(&MomentsFull::from_diagonal(mean.clone(), var.clone()));
                    out.mean.sum() + out.cov.sum()
                }
                "cross" => {
                    let cross = Tensor::<Ad, 3>::from_data([[[0.5 * variance; 4]]], &device);
                    propagate_relu_cross_covariance(cross, &moments).sum()
                }
                _ => unreachable!(),
            };
            let gradients = loss.backward();
            for gradient in [
                mean.grad(&gradients).unwrap(),
                var.grad(&gradients).unwrap(),
            ] {
                let values = gradient.into_data().to_vec::<f32>().unwrap();
                assert!(
                    values.iter().all(|x| x.is_finite()),
                    "{mode}, var={variance:e}: {values:?}"
                );
            }
        }
    }
}
