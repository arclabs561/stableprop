#![cfg(feature = "burn")]

use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_conv2d, propagate_leaky_relu, propagate_linear, propagate_linear_bayes,
    propagate_residual_add, Moments,
};

type Nd = NdArray<f32>;

#[test]
#[should_panic(expected = "bias shape must match output width")]
fn linear_rejects_broadcastable_scalar_bias() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], &device),
        Tensor::<Nd, 2>::ones([1, 2], &device),
    );
    let _ = propagate_linear(
        &input,
        Tensor::<Nd, 2>::ones([2, 3], &device),
        Some(Tensor::<Nd, 1>::zeros([1], &device)),
    );
}

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

#[test]
#[should_panic(expected = "weight mean and variance shapes must match")]
fn linear_bayes_rejects_broadcastable_weight_variance() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], &device),
        Tensor::<Nd, 2>::ones([1, 2], &device),
    );
    let _ = propagate_linear_bayes(
        &input,
        Tensor::<Nd, 2>::zeros([2, 3], &device),
        Tensor::<Nd, 2>::zeros([2, 1], &device),
        None,
    );
}

#[test]
#[should_panic(expected = "bias mean and variance shapes must match")]
fn linear_bayes_rejects_broadcastable_bias_variance() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], &device),
        Tensor::<Nd, 2>::ones([1, 2], &device),
    );
    let _ = propagate_linear_bayes(
        &input,
        Tensor::<Nd, 2>::zeros([2, 2], &device),
        Tensor::<Nd, 2>::zeros([2, 2], &device),
        Some((
            Tensor::<Nd, 1>::zeros([2], &device),
            Tensor::<Nd, 1>::zeros([1], &device),
        )),
    );
}

#[test]
#[should_panic(expected = "bias shapes must match output width")]
fn linear_bayes_rejects_bias_with_wrong_output_width() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], &device),
        Tensor::<Nd, 2>::ones([1, 2], &device),
    );
    let _ = propagate_linear_bayes(
        &input,
        Tensor::<Nd, 2>::zeros([2, 2], &device),
        Tensor::<Nd, 2>::zeros([2, 2], &device),
        Some((
            Tensor::<Nd, 1>::zeros([1], &device),
            Tensor::<Nd, 1>::zeros([1], &device),
        )),
    );
}

#[test]
#[should_panic(expected = "convolution mean and variance shapes must match")]
fn conv2d_rejects_batch_broadcasting() {
    let device = Default::default();
    let options = burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let _ = propagate_conv2d(
        Tensor::<Nd, 4>::zeros([2, 1, 3, 3], &device),
        Tensor::<Nd, 4>::ones([1, 1, 3, 3], &device),
        Tensor::<Nd, 4>::ones([1, 1, 1, 1], &device),
        None,
        options,
    );
}
