#![cfg(feature = "burn")]

//! Property tests for Burn's NdArray implementation.  The scalar expectations
//! are deliberately evaluated on the host so they do not repeat tensor shapes
//! or reductions from the implementation under test.

use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use proptest::prelude::*;
use stableprop::burn_sdp::{
    propagate_linear_cross_covariance, propagate_linear_full, propagate_relu_cross_covariance,
    propagate_relu_full, Moments, MomentsFull,
};

type Nd = NdArray<f32>;

const BATCH: usize = 2;

#[derive(Clone, Debug)]
struct AffineCase {
    d_in: usize,
    d_out: usize,
    mean: Vec<f32>,
    cov: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// PSD covariances are formed as `L L^T`.  The first two features share a
/// forced signed latent component, so the oracle always has a real cross term.
fn full_affine_case() -> impl Strategy<Value = AffineCase> {
    ((3usize..5, 1usize..5).prop_filter("affine map must be non-square", |(i, o)| i != o))
        .prop_flat_map(|(d_in, d_out)| {
            (
                prop::collection::vec(-2.0f32..2.0, BATCH * d_in),
                prop::collection::vec(-1.5f32..1.5, BATCH * d_in * 2),
                prop::collection::vec(0.3f32..1.5, BATCH),
                Just(vec![-1.0f32, 1.0f32]),
                prop::collection::vec(0.3f32..1.5, BATCH),
                prop::collection::vec(-2.0f32..2.0, d_in * d_out),
                prop::collection::vec(-2.0f32..2.0, d_out),
            )
                .prop_map(
                    move |(mean, raw_factor, first, sign, second, weight, bias)| {
                        let mut cov = vec![0.0; BATCH * d_in * d_in];
                        for batch in 0..BATCH {
                            let mut factor =
                                raw_factor[batch * d_in * 2..(batch + 1) * d_in * 2].to_vec();
                            factor[0] = first[batch];
                            factor[1] = 0.0;
                            factor[2] = sign[batch] * second[batch];
                            factor[3] = 0.0;
                            for i in 0..d_in {
                                for j in 0..d_in {
                                    cov[(batch * d_in + i) * d_in + j] =
                                        (0..2).map(|k| factor[i * 2 + k] * factor[j * 2 + k]).sum();
                                }
                            }
                        }
                        AffineCase {
                            d_in,
                            d_out,
                            mean,
                            cov,
                            weight,
                            bias,
                        }
                    },
                )
        })
}

#[derive(Clone, Debug)]
struct CrossCase {
    d_left: usize,
    d_in: usize,
    d_hidden: usize,
    d_out: usize,
    cross: Vec<f32>,
    first: Vec<f32>,
    second: Vec<f32>,
}

/// A valid cross covariance from two variables driven by the same two latent
/// standard normals. Its [0, 0] entry is signed and nonzero for every batch.
fn cross_case() -> impl Strategy<Value = CrossCase> {
    (2usize..5, 2usize..5, 1usize..5, 1usize..5).prop_flat_map(|(d_left, d_in, d_hidden, d_out)| {
        (
            prop::collection::vec(-1.5f32..1.5, BATCH * d_left * 2),
            prop::collection::vec(-1.5f32..1.5, BATCH * d_in * 2),
            prop::collection::vec(0.3f32..1.5, BATCH),
            Just(vec![-1.0f32, 1.0f32]),
            prop::collection::vec(0.3f32..1.5, BATCH),
            prop::collection::vec(-2.0f32..2.0, d_in * d_hidden),
            prop::collection::vec(-2.0f32..2.0, d_hidden * d_out),
        )
            .prop_map(move |(left_raw, right_raw, first, sign, second, w1, w2)| {
                let mut cross = vec![0.0; BATCH * d_left * d_in];
                for batch in 0..BATCH {
                    let mut left = left_raw[batch * d_left * 2..(batch + 1) * d_left * 2].to_vec();
                    let mut right = right_raw[batch * d_in * 2..(batch + 1) * d_in * 2].to_vec();
                    left[0] = first[batch];
                    left[1] = 0.0;
                    right[0] = sign[batch] * second[batch];
                    right[1] = 0.0;
                    for i in 0..d_left {
                        for j in 0..d_in {
                            cross[(batch * d_left + i) * d_in + j] =
                                left[i * 2] * right[j * 2] + left[i * 2 + 1] * right[j * 2 + 1];
                        }
                    }
                }
                CrossCase {
                    d_left,
                    d_in,
                    d_hidden,
                    d_out,
                    cross,
                    first: w1,
                    second: w2,
                }
            })
    })
}

#[derive(Clone, Debug)]
struct ReluCase {
    d_left: usize,
    cross: Vec<f32>,
    mean: Vec<f32>,
    var: Vec<f32>,
}

/// `cross = L R^T` and `var = diag(R R^T)` describe one valid joint Gaussian
/// per batch. The forced first latent component makes the generated cross
/// covariance contain both useful magnitude and signed terms.
fn relu_case() -> impl Strategy<Value = ReluCase> {
    (2usize..5).prop_flat_map(|d_left| {
        (
            prop::collection::vec(-1.5f32..1.5, BATCH * d_left * 2),
            prop::collection::vec(-1.5f32..1.5, BATCH * 4 * 2),
            prop::collection::vec(-3.0f32..3.0, BATCH * 4),
            prop::collection::vec(0.3f32..1.5, BATCH),
            Just(vec![-1.0f32, 1.0f32]),
            prop::collection::vec(0.3f32..1.5, BATCH),
        )
            .prop_map(move |(left_raw, right_raw, mean, first, sign, second)| {
                let mut cross = vec![0.0; BATCH * d_left * 4];
                let mut var = vec![0.0; BATCH * 4];
                for batch in 0..BATCH {
                    let mut left = left_raw[batch * d_left * 2..(batch + 1) * d_left * 2].to_vec();
                    let mut right = right_raw[batch * 8..(batch + 1) * 8].to_vec();
                    left[0] = first[batch];
                    left[1] = 0.0;
                    right[0] = sign[batch] * second[batch];
                    right[1] = 0.0;
                    for i in 0..d_left {
                        for j in 0..4 {
                            cross[(batch * d_left + i) * 4 + j] =
                                left[i * 2] * right[j * 2] + left[i * 2 + 1] * right[j * 2 + 1];
                        }
                    }
                    for j in 0..4 {
                        var[batch * 4 + j] = right[j * 2].powi(2) + right[j * 2 + 1].powi(2);
                    }
                }
                ReluCase {
                    d_left,
                    cross,
                    mean,
                    var,
                }
            })
    })
}

/// This covers several f32 tensor operations and the host-side dot-product
/// accumulation at the small (at most four-feature) dimensions below.
fn f32_roundoff_bound(summed_magnitude: f64) -> f64 {
    64.0 * f32::EPSILON as f64 * summed_magnitude
}

/// The scale-invariance check evaluates a CDF after independently rounded
/// mean, variance, and cross-covariance rescalings. Its output cannot exceed
/// the supplied cross covariance in magnitude, so that covariance supplies a
/// meaningful absolute scale even when a CDF tail rounds to zero.
fn relu_scale_equivariance_bound(expected: f64, scaled_cross_covariance: f64) -> f64 {
    128.0 * f32::EPSILON as f64 * (expected.abs() + scaled_cross_covariance.abs())
}

proptest! {
    /// The full Burn path is an exact affine covariance transport, including
    /// signed off-diagonal terms, for rectangular input/output widths.
    #[test]
    fn full_affine_matches_scalar_quadratic_oracle(case in full_affine_case()) {
        prop_assert!(case.d_in != case.d_out);
        for batch in 0..BATCH {
            prop_assert!(case.cov[batch * case.d_in * case.d_in + 1].abs() > 0.0);
        }
        let device = Default::default();
        let moments = MomentsFull::new(
            Tensor::<Nd, 2>::from_data(TensorData::new(case.mean.clone(), [BATCH, case.d_in]), &device),
            Tensor::<Nd, 3>::from_data(TensorData::new(case.cov.clone(), [BATCH, case.d_in, case.d_in]), &device),
        );
        let out = propagate_linear_full(
            &moments,
            Tensor::<Nd, 2>::from_data(TensorData::new(case.weight.clone(), [case.d_in, case.d_out]), &device),
            Some(Tensor::<Nd, 1>::from_data(TensorData::new(case.bias.clone(), [case.d_out]), &device)),
        );
        let actual_mean = out.mean.into_data().to_vec::<f32>().unwrap();
        let actual_cov = out.cov.into_data().to_vec::<f32>().unwrap();
        for batch in 0..BATCH {
            for out_i in 0..case.d_out {
                let mut expected_mean = case.bias[out_i] as f64;
                let mut mean_scale = expected_mean.abs();
                for i in 0..case.d_in {
                    let term = case.mean[batch * case.d_in + i] as f64
                        * case.weight[i * case.d_out + out_i] as f64;
                    expected_mean += term;
                    mean_scale += term.abs();
                }
                prop_assert!(
                    (actual_mean[batch * case.d_out + out_i] as f64 - expected_mean).abs()
                        <= f32_roundoff_bound(mean_scale),
                    "batch {batch}, mean [{out_i}]: {} vs {expected_mean}",
                    actual_mean[batch * case.d_out + out_i],
                );
                for out_j in 0..case.d_out {
                    let mut expected_cov = 0.0;
                    let mut scale = 0.0;
                    for i in 0..case.d_in {
                        for j in 0..case.d_in {
                            let term = case.weight[i * case.d_out + out_i] as f64
                                * case.cov[(batch * case.d_in + i) * case.d_in + j] as f64
                                * case.weight[j * case.d_out + out_j] as f64;
                            expected_cov += term;
                            scale += term.abs();
                        }
                    }
                    let index = (batch * case.d_out + out_i) * case.d_out + out_j;
                    prop_assert!(
                        (actual_cov[index] as f64 - expected_cov).abs()
                            <= f32_roundoff_bound(scale),
                        "batch {batch}, [{out_i}, {out_j}]: {} vs {expected_cov}",
                        actual_cov[index],
                    );
                }
            }
        }
    }

    /// Transporting a valid signed cross covariance through two affine maps
    /// composes as one map; the scalar three-index product is the oracle.
    #[test]
    fn affine_cross_covariance_composes(case in cross_case()) {
        for batch in 0..BATCH {
            prop_assert!(case.cross[batch * case.d_left * case.d_in].abs() > 0.0);
        }
        let device = Default::default();
        let cross = Tensor::<Nd, 3>::from_data(TensorData::new(case.cross.clone(), [BATCH, case.d_left, case.d_in]), &device);
        let w1 = Tensor::<Nd, 2>::from_data(TensorData::new(case.first.clone(), [case.d_in, case.d_hidden]), &device);
        let w2 = Tensor::<Nd, 2>::from_data(TensorData::new(case.second.clone(), [case.d_hidden, case.d_out]), &device);
        let first_hop = propagate_linear_cross_covariance(cross, w1);
        let first_values = first_hop.clone().into_data().to_vec::<f32>().unwrap();
        for batch in 0..BATCH {
            for left in 0..case.d_left {
                for hidden in 0..case.d_hidden {
                    let mut expected = 0.0;
                    let mut scale = 0.0;
                    for input in 0..case.d_in {
                        let term = case.cross[(batch * case.d_left + left) * case.d_in + input]
                            as f64
                            * case.first[input * case.d_hidden + hidden] as f64;
                        expected += term;
                        scale += term.abs();
                    }
                    let index = (batch * case.d_left + left) * case.d_hidden + hidden;
                    prop_assert!(
                        (first_values[index] as f64 - expected).abs() <= f32_roundoff_bound(scale),
                        "first hop, batch {batch}, [{left}, {hidden}]: {} vs {expected}",
                        first_values[index],
                    );
                }
            }
        }
        let sequential = propagate_linear_cross_covariance(first_hop, w2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for batch in 0..BATCH {
            for left in 0..case.d_left {
                for out in 0..case.d_out {
                    let mut expected = 0.0;
                    let mut scale = 0.0;
                    for input in 0..case.d_in {
                        for hidden in 0..case.d_hidden {
                            let term = case.cross[(batch * case.d_left + left) * case.d_in + input] as f64
                                * case.first[input * case.d_hidden + hidden] as f64
                                * case.second[hidden * case.d_out + out] as f64;
                            expected += term;
                            scale += term.abs();
                        }
                    }
                    let index = (batch * case.d_left + left) * case.d_out + out;
                    prop_assert!(
                        (sequential[index] as f64 - expected).abs() <= f32_roundoff_bound(scale),
                        "batch {batch}, [{left}, {out}]: {} vs {expected}",
                        sequential[index],
                    );
                }
            }
        }
    }

    /// The truncated Hermite covariance series is a PSD kernel; replacing its
    /// diagonal with exact rectified variances preserves that property.  These
    /// rank-two factors make every generated 3- or 4-feature input singular.
    #[test]
    fn full_relu_covariance_is_symmetric_and_psd_on_signed_directions(case in full_affine_case()) {
        let device = Default::default();
        let out = propagate_relu_full(&MomentsFull::new(
            Tensor::<Nd, 2>::from_data(TensorData::new(case.mean.clone(), [BATCH, case.d_in]), &device),
            Tensor::<Nd, 3>::from_data(TensorData::new(case.cov.clone(), [BATCH, case.d_in, case.d_in]), &device),
        ));
        let covariance = out.cov.into_data().to_vec::<f32>().unwrap();
        let mut contrast = vec![0.0; case.d_in];
        contrast[0] = 1.0;
        contrast[1] = -1.0;
        let directions = vec![
            vec![1.0; case.d_in],
            (0..case.d_in)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            contrast,
        ];
        for batch in 0..BATCH {
            for i in 0..case.d_in {
                for j in 0..case.d_in {
                    let a = covariance[(batch * case.d_in + i) * case.d_in + j];
                    let b = covariance[(batch * case.d_in + j) * case.d_in + i];
                    // Series terms can cancel near zero. Their natural scale
                    // is sigma_i * sigma_j, not the final covariance entry.
                    let var_i = case.cov[(batch * case.d_in + i) * case.d_in + i] as f64;
                    let var_j = case.cov[(batch * case.d_in + j) * case.d_in + j] as f64;
                    prop_assert!(
                        (a as f64 - b as f64).abs()
                            <= f32_roundoff_bound((var_i * var_j).sqrt()),
                        "batch {batch}, asymmetric [{i}, {j}]: {a} vs {b}",
                    );
                }
            }
            for direction in &directions {
                let mut quadratic = 0.0;
                let mut term_scale = 0.0;
                for i in 0..case.d_in {
                    for j in 0..case.d_in {
                        let term = direction[i]
                            * covariance[(batch * case.d_in + i) * case.d_in + j] as f64
                            * direction[j];
                        quadratic += term;
                        term_scale += term.abs();
                    }
                }
                prop_assert!(
                    quadratic >= -f32_roundoff_bound(term_scale),
                    "batch {batch}, direction {direction:?}: v^T Sigma v = {quadratic}",
                );
            }
        }
    }

    /// ReLU(y) - ReLU(-y) = y, so the corresponding Gaussian cross-covariance
    /// transports must differ by Cov(U, y). The reflected variable has both
    /// its mean and its cross covariance negated, while its variance is fixed.
    #[test]
    fn relu_cross_covariance_reflection_recovers_input_cross_covariance(case in relu_case()) {
        let d_right = 4;
        let device = Default::default();
        let right = Moments::new(
            Tensor::<Nd, 2>::from_data(TensorData::new(case.mean.clone(), [BATCH, d_right]), &device),
            Tensor::<Nd, 2>::from_data(TensorData::new(case.var.clone(), [BATCH, d_right]), &device),
        );
        let reflected_right = Moments::new(
            Tensor::<Nd, 2>::from_data(TensorData::new(case.mean.iter().map(|value| -value).collect(), [BATCH, d_right]), &device),
            Tensor::<Nd, 2>::from_data(TensorData::new(case.var.clone(), [BATCH, d_right]), &device),
        );
        let positive = propagate_relu_cross_covariance(
            Tensor::<Nd, 3>::from_data(TensorData::new(case.cross.clone(), [BATCH, case.d_left, d_right]), &device),
            &right,
        ).into_data().to_vec::<f32>().unwrap();
        let negative = propagate_relu_cross_covariance(
            Tensor::<Nd, 3>::from_data(TensorData::new(case.cross.iter().map(|value| -value).collect(), [BATCH, case.d_left, d_right]), &device),
            &reflected_right,
        ).into_data().to_vec::<f32>().unwrap();
        prop_assert_eq!(positive.len(), negative.len());
        prop_assert_eq!(positive.len(), case.cross.len());
        for (index, ((positive, negative), cross)) in positive.iter().zip(&negative).zip(&case.cross).enumerate() {
            let observed = *positive as f64 - *negative as f64;
            let scale = (*positive as f64).abs() + (*negative as f64).abs() + (*cross as f64).abs();
            prop_assert!(
                (observed - *cross as f64).abs() <= f32_roundoff_bound(scale),
                "index {index}: {observed} vs {cross}",
            );
        }
    }

    /// Scaling the right jointly Gaussian variable by a positive constant
    /// leaves the ReLU gate unchanged and scales its transported covariance.
    #[test]
    fn relu_cross_covariance_is_positive_scale_equivariant(
        case in relu_case(),
        scale in 0.2f32..3.0,
    ) {
        let d_right = 4;
        let device = Default::default();
        prop_assert!(case.cross.iter().any(|value| *value < 0.0));
        prop_assert!(case.cross.iter().any(|value| *value > 0.0));
        let right = Moments::new(
            Tensor::<Nd, 2>::from_data(TensorData::new(case.mean.clone(), [BATCH, d_right]), &device),
            Tensor::<Nd, 2>::from_data(TensorData::new(case.var.clone(), [BATCH, d_right]), &device),
        );
        let base = propagate_relu_cross_covariance(
            Tensor::<Nd, 3>::from_data(TensorData::new(case.cross.clone(), [BATCH, case.d_left, d_right]), &device),
            &right,
        ).into_data().to_vec::<f32>().unwrap();
        let scaled_right = Moments::new(
            Tensor::<Nd, 2>::from_data(TensorData::new(case.mean.iter().map(|value| value * scale).collect(), [BATCH, d_right]), &device),
            Tensor::<Nd, 2>::from_data(TensorData::new(case.var.iter().map(|value| value * scale * scale).collect(), [BATCH, d_right]), &device),
        );
        let scaled = propagate_relu_cross_covariance(
            Tensor::<Nd, 3>::from_data(TensorData::new(case.cross.iter().map(|value| value * scale).collect(), [BATCH, case.d_left, d_right]), &device),
            &scaled_right,
        ).into_data().to_vec::<f32>().unwrap();
        prop_assert_eq!(scaled.len(), base.len());
        prop_assert_eq!(scaled.len(), case.cross.len());
        for (index, (actual, expected)) in scaled.iter().zip(base.iter().map(|value| value * scale)).enumerate() {
            let scaled_cross = case.cross[index] as f64 * scale as f64;
            prop_assert!(
                (*actual as f64 - expected as f64).abs()
                    <= relu_scale_equivariance_bound(expected as f64, scaled_cross),
                "index {index}: {actual} vs {expected}",
            );
        }
    }
}
