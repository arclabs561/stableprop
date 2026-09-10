#![cfg(feature = "burn")]

use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{propagate_leaky_relu, propagate_residual_add, Moments};

type Nd = NdArray<f32>;

#[test]
#[should_panic(expected = "residual shapes must match")]
fn residual_add_rejects_batch_broadcasting() {
    let device = Default::default();
    let skip = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], &device),
        Tensor::<Nd, 2>::ones([1, 2], &device),
    );
    let branch = Moments::new(
        Tensor::<Nd, 2>::zeros([3, 2], &device),
        Tensor::<Nd, 2>::ones([3, 2], &device),
    );
    let _ = propagate_residual_add(&skip, &branch);
}

#[test]
#[should_panic(expected = "representable")]
fn leaky_relu_rejects_unrepresentable_variance_coefficients() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 1], &device),
        Tensor::<Nd, 2>::ones([1, 1], &device),
    );
    // The slope fits in f32, but its square does not.
    let _ = propagate_leaky_relu(&input, 1e20);
}
