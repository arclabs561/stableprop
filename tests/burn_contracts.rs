#![cfg(feature = "burn")]

use burn::tensor::{DType, Tensor};
use burn_ndarray::NdArray;
use stableprop::burn_sdp::{
    propagate_conv2d, propagate_leaky_relu, propagate_linear, propagate_linear_bayes,
    propagate_linear_cauchy, propagate_linear_full, propagate_relu, propagate_relu_full,
    propagate_residual_add, Cauchy, Moments, MomentsFull,
};

type Nd = NdArray<f32>;

#[test]
#[should_panic(expected = "same dtype")]
fn moments_rejects_mixed_dynamic_float_dtypes() {
    let device = Default::default();
    let _ = Moments::<Nd>::new(
        Tensor::zeros([1, 1], (&device, DType::F64)),
        Tensor::ones([1, 1], (&device, DType::F32)),
    );
}

#[test]
#[should_panic(expected = "same dtype")]
fn full_moments_rejects_mixed_dynamic_float_dtypes() {
    let device = Default::default();
    let _ = MomentsFull::<Nd>::new(
        Tensor::zeros([1, 1], (&device, DType::F64)),
        Tensor::ones([1, 1, 1], (&device, DType::F32)),
    );
}

#[test]
#[should_panic(expected = "same dtype")]
fn full_moments_from_diagonal_rejects_mixed_dynamic_float_dtypes() {
    let device = Default::default();
    let _ = MomentsFull::<Nd>::from_diagonal(
        Tensor::zeros([1, 1], (&device, DType::F64)),
        Tensor::ones([1, 1], (&device, DType::F32)),
    );
}

#[test]
#[should_panic(expected = "same dtype")]
fn cauchy_rejects_mixed_dynamic_float_dtypes() {
    let device = Default::default();
    let _ = Cauchy::<Nd>::new(
        Tensor::zeros([1, 1], (&device, DType::F64)),
        Tensor::ones([1, 1], (&device, DType::F32)),
    );
}

#[test]
fn diagonal_gaussian_f64_affine_relu_preserves_dtype_and_moments() {
    let device = Default::default();
    let input = Moments::<Nd>::new(
        Tensor::zeros([1, 1], (&device, DType::F64)),
        Tensor::full([1, 1], 0.25, (&device, DType::F64)),
    );
    let output = propagate_relu(&propagate_linear(
        &input,
        Tensor::full([1, 1], 2.0, (&device, DType::F64)),
        None,
    ));
    let expected_mean = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let expected_var = 0.5 - 1.0 / (2.0 * std::f64::consts::PI);
    assert_eq!(output.mean.dims(), [1, 1]);
    assert_eq!(output.var.dims(), [1, 1]);
    assert_eq!(output.mean.dtype(), DType::F64);
    assert_eq!(output.var.dtype(), DType::F64);
    let mean = output.mean.into_data().to_vec::<f64>().unwrap()[0];
    let var = output.var.into_data().to_vec::<f64>().unwrap()[0];
    assert!(
        (mean - expected_mean).abs() < 1e-12,
        "{mean} != {expected_mean}"
    );
    assert!(
        (var - expected_var).abs() < 1e-12,
        "{var} != {expected_var}"
    );
}

#[test]
fn full_gaussian_f64_affine_relu_preserves_dtype_and_moments() {
    let device = Default::default();
    let input = MomentsFull::<Nd>::from_diagonal(
        Tensor::zeros([1, 1], (&device, DType::F64)),
        Tensor::full([1, 1], 0.25, (&device, DType::F64)),
    );
    let output = propagate_relu_full(&propagate_linear_full(
        &input,
        Tensor::full([1, 1], 2.0, (&device, DType::F64)),
        None,
    ));
    let expected_mean = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let expected_var = 0.5 - 1.0 / (2.0 * std::f64::consts::PI);
    assert_eq!(output.mean.dims(), [1, 1]);
    assert_eq!(output.cov.dims(), [1, 1, 1]);
    assert_eq!(output.mean.dtype(), DType::F64);
    assert_eq!(output.cov.dtype(), DType::F64);
    let mean = output.mean.into_data().to_vec::<f64>().unwrap()[0];
    let cov = output.cov.into_data().to_vec::<f64>().unwrap()[0];
    assert!(
        (mean - expected_mean).abs() < 1e-12,
        "{mean} != {expected_mean}"
    );
    assert!(
        (cov - expected_var).abs() < 1e-12,
        "{cov} != {expected_var}"
    );
}

#[test]
fn cauchy_f64_affine_preserves_dtype_parameters_and_zero_mass_width() {
    let device = Default::default();
    let output = propagate_linear_cauchy(
        &Cauchy::<Nd>::new(
            Tensor::full([1, 1], 1.0, (&device, DType::F64)),
            Tensor::full([1, 1], 0.25, (&device, DType::F64)),
        ),
        Tensor::full([1, 1], 2.0, (&device, DType::F64)),
        Some(Tensor::full([1], 0.5, (&device, DType::F64))),
    );
    assert_eq!(output.location.dims(), [1, 1]);
    assert_eq!(output.scale.dims(), [1, 1]);
    assert_eq!(output.location.dtype(), DType::F64);
    assert_eq!(output.scale.dtype(), DType::F64);
    let width = output
        .interval_halfwidth(0.0)
        .into_data()
        .to_vec::<f64>()
        .unwrap()[0];
    assert_eq!(width, 0.0);
    let location = output.location.into_data().to_vec::<f64>().unwrap()[0];
    let scale = output.scale.into_data().to_vec::<f64>().unwrap()[0];
    assert_eq!(location, 2.5);
    assert_eq!(scale, 0.5);
}

#[test]
#[should_panic(expected = "bias shape must match output width")]
fn linear_rejects_broadcastable_scalar_bias() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], (&device, DType::F32)),
        Tensor::<Nd, 2>::ones([1, 2], (&device, DType::F32)),
    );
    let _ = propagate_linear(
        &input,
        Tensor::<Nd, 2>::ones([2, 3], (&device, DType::F32)),
        Some(Tensor::<Nd, 1>::zeros([1], (&device, DType::F32))),
    );
}

#[test]
#[should_panic(expected = "residual shapes must match")]
fn residual_add_rejects_batch_broadcasting() {
    let device = Default::default();
    let skip = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], (&device, DType::F32)),
        Tensor::<Nd, 2>::ones([1, 2], (&device, DType::F32)),
    );
    let branch = Moments::new(
        Tensor::<Nd, 2>::zeros([3, 2], (&device, DType::F32)),
        Tensor::<Nd, 2>::ones([3, 2], (&device, DType::F32)),
    );
    let _ = propagate_residual_add(&skip, &branch);
}

#[test]
#[should_panic(expected = "representable")]
fn leaky_relu_rejects_unrepresentable_variance_coefficients() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 1], (&device, DType::F32)),
        Tensor::<Nd, 2>::ones([1, 1], (&device, DType::F32)),
    );
    // The slope fits in f32, but its square does not.
    let _ = propagate_leaky_relu(&input, 1e20);
}

#[test]
#[should_panic(expected = "representable")]
fn leaky_relu_checks_actual_tensor_dtype() {
    let device = Default::default();
    // Burn 0.21 selects dtype at creation, independently of the backend alias.
    let input = Moments::new(
        Tensor::<NdArray<f64>, 2>::zeros([1, 1], (&device, DType::F32)),
        Tensor::<NdArray<f64>, 2>::ones([1, 1], (&device, DType::F32)),
    );
    let _ = propagate_leaky_relu(&input, 1e20);
}

#[test]
#[should_panic(expected = "weight mean and variance shapes must match")]
fn linear_bayes_rejects_broadcastable_weight_variance() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], (&device, DType::F32)),
        Tensor::<Nd, 2>::ones([1, 2], (&device, DType::F32)),
    );
    let _ = propagate_linear_bayes(
        &input,
        Tensor::<Nd, 2>::zeros([2, 3], (&device, DType::F32)),
        Tensor::<Nd, 2>::zeros([2, 1], (&device, DType::F32)),
        None,
    );
}

#[test]
#[should_panic(expected = "bias mean and variance shapes must match")]
fn linear_bayes_rejects_broadcastable_bias_variance() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], (&device, DType::F32)),
        Tensor::<Nd, 2>::ones([1, 2], (&device, DType::F32)),
    );
    let _ = propagate_linear_bayes(
        &input,
        Tensor::<Nd, 2>::zeros([2, 2], (&device, DType::F32)),
        Tensor::<Nd, 2>::zeros([2, 2], (&device, DType::F32)),
        Some((
            Tensor::<Nd, 1>::zeros([2], (&device, DType::F32)),
            Tensor::<Nd, 1>::zeros([1], (&device, DType::F32)),
        )),
    );
}

#[test]
#[should_panic(expected = "bias shapes must match output width")]
fn linear_bayes_rejects_bias_with_wrong_output_width() {
    let device = Default::default();
    let input = Moments::new(
        Tensor::<Nd, 2>::zeros([1, 2], (&device, DType::F32)),
        Tensor::<Nd, 2>::ones([1, 2], (&device, DType::F32)),
    );
    let _ = propagate_linear_bayes(
        &input,
        Tensor::<Nd, 2>::zeros([2, 2], (&device, DType::F32)),
        Tensor::<Nd, 2>::zeros([2, 2], (&device, DType::F32)),
        Some((
            Tensor::<Nd, 1>::zeros([1], (&device, DType::F32)),
            Tensor::<Nd, 1>::zeros([1], (&device, DType::F32)),
        )),
    );
}

#[test]
#[should_panic(expected = "convolution mean and variance shapes must match")]
fn conv2d_rejects_batch_broadcasting() {
    let device = Default::default();
    let options = burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let _ = propagate_conv2d(
        Tensor::<Nd, 4>::zeros([2, 1, 3, 3], (&device, DType::F32)),
        Tensor::<Nd, 4>::ones([1, 1, 3, 3], (&device, DType::F32)),
        Tensor::<Nd, 4>::ones([1, 1, 1, 1], (&device, DType::F32)),
        None,
        options,
    );
}
