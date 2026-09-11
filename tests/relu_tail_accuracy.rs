//! Negative Gaussian-ReLU tail regressions.
//!
//! Constants are reproduced by mpmath 1.3.0 at `mp.dps = 90` using
//! `p = erfc(-a / sqrt(2)) / 2`, `phi = exp(-a*a/2) / sqrt(2*pi)`,
//! `m = phi + a*p`, and
//! `v = (a*a + 1)*p + a*phi - m*m`. Reproduce with no runtime dependency:
//! `uv run --with mpmath==1.3.0 python`, then evaluate those expressions at the
//! alphas below. The table stores normalized mean and variance for sigma=1.
//! Derivative identities are DLMF §7.18: dM/dmu=P, dM/dv=phi/(2 sigma),
//! dV/dmu=2M(1-P), dV/dv=P-(M/sigma)phi.

use stableprop::{propagate_relu, Moments};

#[derive(Clone, Copy)]
struct Tail {
    alpha: f64,
    p: f64,
    phi: f64,
    mean: f64,
    var: f64,
}

/// Independent MP90 references evaluated at the exact f32 inputs identified
/// by their IEEE-754 bit patterns.  Regenerate with the pinned reference tool:
/// `uv run --script scripts/reference_relu.py --marginal-bits 0xMEAN_HEX 0xVARIANCE_HEX`.
/// The ordinary, small, large, and subnormal cases complement the normalized
/// tail table below; they deliberately do not re-test those nominal alphas.
#[cfg(feature = "burn")]
#[derive(Clone, Copy)]
struct F32MarginalReference {
    mean_bits: u32,
    variance_bits: u32,
    mean: f64,
    variance: f64,
    d_mean_d_mean: f64,
    d_mean_d_variance: f64,
    d_variance_d_mean: f64,
    d_variance_d_variance: f64,
}

#[cfg(feature = "burn")]
const F32_MARGINAL_REFERENCES: [F32MarginalReference; 9] = [
    // Adjacent representable interior values at the -8, -2, and +8 switches.
    F32MarginalReference {
        mean_bits: 0xc0ff_ffff,
        variance_bits: 0x3f80_0000,
        mean: 7.550_292_075_855_552e-17,
        variance: 1.807_513_647_651_338_6e-17,
        d_mean_d_mean: 6.220_984_665_423_594e-16,
        d_mean_d_variance: 2.526_145_178_228_883e-15,
        d_variance_d_mean: 1.510_058_415_171_109_5e-16,
        d_variance_d_variance: 6.220_984_665_423_59e-16,
    },
    F32MarginalReference {
        mean_bits: 0xc000_0001,
        variance_bits: 0x3f80_0000,
        mean: 0.008_490_697_192_777_04,
        variance: 0.005_696_630_727_019_286,
        d_mean_d_mean: 0.022_750_119_075_732_756,
        d_mean_d_variance: 0.026_995_470_384_146_81,
        d_variance_d_mean: 0.016_595_065_641_210_742,
        d_variance_d_variance: 0.022_291_698_346_516_014,
    },
    F32MarginalReference {
        mean_bits: 0xc000_0000,
        variance_bits: 0x3f80_0000,
        mean: 0.008_490_702_616_829_638,
        variance: 0.005_696_634_683_592_494,
        d_mean_d_mean: 0.022_750_131_948_179_21,
        d_mean_d_variance: 0.026_995_483_256_594_026,
        d_variance_d_mean: 0.016_595_076_023_928_025,
        d_variance_d_variance: 0.022_291_710_707_520_52,
    },
    F32MarginalReference {
        mean_bits: 0xbfff_ffff,
        variance_bits: 0x3f80_0000,
        mean: 0.008_490_705_328_857_088,
        variance: 0.005_696_636_661_880_027,
        d_mean_d_mean: 0.022_750_138_384_404_735,
        d_mean_d_variance: 0.026_995_489_692_819_36,
        d_variance_d_mean: 0.016_595_081_215_288_773,
        d_variance_d_variance: 0.022_291_716_888_024_88,
    },
    F32MarginalReference {
        mean_bits: 0x40ff_ffff,
        variance_bits: 0x3f80_0000,
        mean: 7.999_999_523_162_842,
        variance: 0.999_999_999_999_998_8,
        d_mean_d_mean: 0.999_999_999_999_999_3,
        d_mean_d_variance: 2.526_145_178_228_883e-15,
        d_variance_d_mean: 9.953_574_871_398_42e-15,
        d_variance_d_variance: 0.999_999_999_999_959,
    },
    // Ordinary non-power-of-two variance, then matched small and large scales.
    F32MarginalReference {
        mean_bits: 0xbf9e_064b,
        variance_bits: 0x3ebd_70a4,
        mean: 0.004_768_981_051_853_294,
        variance: 0.001_932_819_200_477_744,
        d_mean_d_mean: 0.021_197_808_459_668_247,
        d_mean_d_variance: 0.041_809_589_768_053_97,
        d_variance_d_mean: 0.009_335_778_209_936_64,
        d_variance_d_variance: 0.020_799_030_176_889_03,
    },
    F32MarginalReference {
        mean_bits: 0xa690_1d7d,
        variance_bits: 0x0da2_4260,
        mean: 8.331_547_039_581_924e-17,
        variance: 6.839_831_563_514_943e-32,
        d_mean_d_mean: 0.158_655_253_437_363_65,
        d_mean_d_variance: 1.209_853_618_206_980_3e14,
        d_variance_d_mean: 1.401_940_666_498_347_4e-16,
        d_variance_d_variance: 0.138_495_348_775_163_96,
    },
    F32MarginalReference {
        mean_bits: 0xd863_5fa9,
        variance_bits: 0x7149_f2ca,
        mean: 8.331_547_447_213_919e13,
        variance: 6.839_831_961_229_732e28,
        d_mean_d_mean: 0.158_655_258_899_752_2,
        d_mean_d_variance: 1.209_853_638_334_576_4e-16,
        d_variance_d_mean: 1.401_940_725_988_125e14,
        d_variance_d_variance: 0.138_495_352_915_814_38,
    },
    // Minimum positive f32 variance: output variance may round below one ULP.
    F32MarginalReference {
        mean_bits: 0,
        variance_bits: 1,
        mean: 1.493_397_393_008_226e-23,
        variance: 4.776_256_548_180_32e-46,
        d_mean_d_mean: 0.5,
        d_mean_d_variance: 5.328_619_958_660_216e21,
        d_variance_d_mean: 1.493_397_393_008_226e-23,
        d_variance_d_variance: 0.340_845_056_908_104_7,
    },
];

// Include both sides of the formula boundary at alpha = -2.
const TAILS: [Tail; 11] = [
    Tail {
        alpha: -1.999,
        p: 0.02280417693265889,
        phi: 0.054099029450649934,
        mean: 0.008513479762264811,
        var: 0.005713251550229037,
    },
    Tail {
        alpha: -2.0,
        p: 0.02275013194817921,
        phi: 0.05399096651318805,
        mean: 0.008490702616829637,
        var: 0.005696634683592494,
    },
    Tail {
        alpha: -2.001,
        p: 0.02269619494564155,
        phi: 0.05388306554860322,
        mean: 0.008467979462374474,
        var: 0.005680061365255032,
    },
    Tail {
        alpha: -2.999,
        p: 0.0013543365337271066,
        phi: 0.004445161697868555,
        mean: 0.0003835064332209618,
        var: 0.0002040536633131203,
    },
    Tail {
        alpha: -3.0,
        p: 1.3498980316300945e-3,
        phi: 4.431848411938007e-3,
        mean: 3.821543170477236e-4,
        var: 2.0328903856488553e-4,
    },
    Tail {
        alpha: -3.001,
        p: 0.001345472825084966,
        phi: 0.004418570580805835,
        mean: 0.0003808066327258519,
        var: 0.00020252710658315648,
    },
    Tail {
        alpha: -4.0,
        p: 3.167124183311992e-5,
        phi: 1.3383022576488535e-4,
        mean: 7.145258432405667e-6,
        var: 3.0901570487791884e-6,
    },
    Tail {
        alpha: -5.0,
        p: 2.866515718791939e-7,
        phi: 1.4867195147342977e-6,
        mean: 5.346165533832815e-8,
        var: 1.9343292329404572e-8,
    },
    Tail {
        alpha: -6.0,
        p: 9.865_876_450_376_98e-10,
        phi: 6.075_882_849_823_285e-9,
        mean: 1.5635697959709664e-10,
        var: 4.844576743067078e-11,
    },
    Tail {
        alpha: -7.0,
        p: 1.279812543885835e-12,
        phi: 9.134720408364593e-12,
        mean: 1.760326011637483e-13,
        var: 4.758433573956583e-14,
    },
    Tail {
        alpha: -7.5,
        p: 3.190891672910896e-14,
        phi: 2.43432053302901e-13,
        mean: 4.115177834583766e-15,
        var: 1.045082969730704e-15,
    },
];

fn assert_relative(actual: f64, expected: f64, relative: f64, label: &str) {
    assert!(expected != 0.0);
    let error = (actual - expected).abs() / expected.abs();
    assert!(
        error <= relative,
        "{label}: {actual:e} vs {expected:e}, relative error {error:e}"
    );
}

#[cfg(feature = "burn")]
fn assert_f32_reference(actual: f32, expected: f64, relative: f64, label: &str) {
    let absolute = (actual as f64 - expected).abs();
    let subnormal_allowance = 4.0 * f32::from_bits(1) as f64;
    let allowed = (relative * expected.abs()).max(subnormal_allowance);
    assert!(
        absolute <= allowed,
        "{label}: {actual:e} vs {expected:e}, absolute error {absolute:e}, allowed {allowed:e}"
    );
}

fn reference_output(mean: f64, var: f64) -> (f64, f64) {
    let out = propagate_relu(&Moments {
        mean: vec![mean],
        cov: vec![vec![var]],
    });
    (out.mean[0], out.cov[0][0])
}

fn five_point(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
}

#[test]
fn f64_negative_relu_tail_moments_and_gradients_match_erfc_references() {
    for tail in TAILS {
        for var in [1e-24f64, 1.0, 1e24] {
            let sigma = var.sqrt();
            let mean = tail.alpha * sigma;
            let (actual_mean, actual_var) = reference_output(mean, var);
            assert_relative(actual_mean, sigma * tail.mean, 1e-10, "f64 mean");
            assert_relative(actual_var, var * tail.var, 1e-10, "f64 variance");

            let h_mu = sigma * 1e-4;
            let h_var = var * 1e-4;
            let dmdmu = five_point(|mu| reference_output(mu, var).0, mean, h_mu);
            let dmdvar = five_point(|v| reference_output(mean, v).0, var, h_var);
            let dvdmu = five_point(|mu| reference_output(mu, var).1, mean, h_mu);
            let dvdvar = five_point(|v| reference_output(mean, v).1, var, h_var);
            assert_relative(dmdmu, tail.p, 2e-5, "f64 dM/dmu");
            assert_relative(dmdvar, tail.phi / (2.0 * sigma), 2e-5, "f64 dM/dv");
            assert_relative(
                dvdmu,
                2.0 * sigma * tail.mean * (1.0 - tail.p),
                3e-5,
                "f64 dV/dmu",
            );
            assert_relative(dvdvar, tail.p - tail.mean * tail.phi, 3e-5, "f64 dV/dv");
        }
    }
}

#[cfg(feature = "burn")]
mod burn {
    use super::{
        assert_f32_reference, assert_relative, F32MarginalReference, Tail, F32_MARGINAL_REFERENCES,
        TAILS,
    };
    use burn::backend::Autodiff;
    use burn::tensor::{DType, Tensor};
    use burn_ndarray::NdArray;
    use stableprop::burn_sdp::{
        propagate_leaky_relu, propagate_relu, propagate_relu_cross_covariance, propagate_relu_full,
        Moments, MomentsFull,
    };

    type Nd = NdArray<f32>;
    type Ad = Autodiff<Nd>;

    fn values(t: Tensor<Nd, 2>) -> Vec<f32> {
        t.into_data().to_vec::<f32>().unwrap()
    }

    fn expected(tail: Tail, variance: f32) -> (f64, f64, f64) {
        let sigma = (variance as f64).sqrt();
        (
            sigma * tail.mean,
            variance as f64 * tail.var,
            variance as f64 * tail.p,
        )
    }

    #[test]
    fn ndarray_f32_tail_values_match_references_for_diagonal_full_leaky_and_cross_gate() {
        let device = Default::default();
        for tail in TAILS {
            for variance in [1e-24f32, 1.0, 1e24] {
                let sigma = variance.sqrt();
                let mean = tail.alpha as f32 * sigma;
                let input = Moments::new(
                    Tensor::from_data([[mean]], &device),
                    Tensor::from_data([[variance]], &device),
                );
                let diagonal = propagate_relu(&input);
                let leaky = propagate_leaky_relu(&input, 0.0);
                let full = propagate_relu_full(&MomentsFull::from_diagonal(
                    input.mean.clone(),
                    input.var.clone(),
                ));
                let (expect_mean, expect_var, expect_gate) = expected(tail, variance);
                let full_var = values(full.variance())[0];
                let full_mean = values(full.mean)[0];
                for (name, mean_value, var_value) in [
                    (
                        "diagonal",
                        values(diagonal.mean)[0],
                        values(diagonal.var)[0],
                    ),
                    ("leaky0", values(leaky.mean)[0], values(leaky.var)[0]),
                    ("full", full_mean, full_var),
                ] {
                    assert_relative(mean_value as f64, expect_mean, 2e-4, name);
                    assert_relative(var_value as f64, expect_var, 2e-4, name);
                }
                let cross = Tensor::<Nd, 3>::from_data([[[variance]]], &device);
                let gated = propagate_relu_cross_covariance(cross, &input)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()[0];
                assert_relative(gated as f64, expect_gate, 2e-4, "cross gate");
            }
        }
    }

    fn gradients(mode: &str, mean: f32, variance: f32, mean_loss: bool) -> (f32, f32) {
        let device = Default::default();
        let mean = Tensor::<Ad, 2>::from_data([[mean]], &device).require_grad();
        let var = Tensor::<Ad, 2>::from_data([[variance]], &device).require_grad();
        let loss = match mode {
            "diagonal" => {
                let out = propagate_relu(&Moments::new(mean.clone(), var.clone()));
                if mean_loss {
                    out.mean.sum()
                } else {
                    out.var.sum()
                }
            }
            "full" => {
                let out =
                    propagate_relu_full(&MomentsFull::from_diagonal(mean.clone(), var.clone()));
                if mean_loss {
                    out.mean.sum()
                } else {
                    out.variance().sum()
                }
            }
            "leaky0" => {
                let out = propagate_leaky_relu(&Moments::new(mean.clone(), var.clone()), 0.0);
                if mean_loss {
                    out.mean.sum()
                } else {
                    out.var.sum()
                }
            }
            _ => unreachable!(),
        };
        let grads = loss.backward();
        (
            mean.grad(&grads)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0],
            var.grad(&grads)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0],
        )
    }

    fn f32_outputs(mode: &str, mean: f32, variance: f32) -> (f32, f32) {
        let device = Default::default();
        let input = Moments::new(
            Tensor::<Nd, 2>::from_data([[mean]], &device),
            Tensor::<Nd, 2>::from_data([[variance]], &device),
        );
        match mode {
            "diagonal" => {
                let out = propagate_relu(&input);
                (values(out.mean)[0], values(out.var)[0])
            }
            "full" => {
                let out = propagate_relu_full(&MomentsFull::from_diagonal(input.mean, input.var));
                let variance = out.variance();
                (values(out.mean)[0], values(variance)[0])
            }
            "leaky0" => {
                let out = propagate_leaky_relu(&input, 0.0);
                (values(out.mean)[0], values(out.var)[0])
            }
            _ => unreachable!(),
        }
    }

    fn scaled_variance_gradients(mode: &str, mean_value: f32, variance_value: f32) -> (f64, f64) {
        let device = Default::default();
        let mean = Tensor::<Ad, 2>::from_data([[mean_value]], &device).require_grad();
        let variance = Tensor::<Ad, 2>::from_data([[variance_value]], &device).require_grad();
        let loss_scale = 4_294_967_296.0f64;
        let loss = match mode {
            "diagonal" => propagate_relu(&Moments::new(mean.clone(), variance.clone()))
                .var
                .sum(),
            "full" => {
                propagate_relu_full(&MomentsFull::from_diagonal(mean.clone(), variance.clone()))
                    .variance()
                    .sum()
            }
            "leaky0" => propagate_leaky_relu(&Moments::new(mean.clone(), variance.clone()), 0.0)
                .var
                .sum(),
            _ => unreachable!(),
        }
        .mul_scalar(loss_scale);
        let gradients = loss.backward();
        let dvariance_dmean = mean
            .grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0] as f64
            / loss_scale;
        let dvariance_dvariance = variance
            .grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0] as f64
            / loss_scale;
        (dvariance_dmean, dvariance_dvariance)
    }

    fn assert_f32_marginal_reference(reference: F32MarginalReference) {
        let mean = f32::from_bits(reference.mean_bits);
        let variance = f32::from_bits(reference.variance_bits);
        let sigma = variance.sqrt() as f64;
        let cancellation_scale = mean.abs() as f64 + sigma;
        for mode in ["diagonal", "full", "leaky0"] {
            let (actual_mean, actual_variance) = f32_outputs(mode, mean, variance);
            assert_f32_reference(actual_mean, reference.mean, 3e-5, "f32 ReLU mean");
            assert_f32_reference(
                actual_variance,
                reference.variance,
                3e-5,
                "f32 ReLU variance",
            );

            let mean_derivatives = gradients(mode, mean, variance, true);
            assert_f32_reference(
                mean_derivatives.0,
                reference.d_mean_d_mean,
                4e-4,
                "f32 dmean/dmean",
            );
            assert_f32_reference(
                mean_derivatives.1,
                reference.d_mean_d_variance,
                4e-4,
                "f32 dmean/dvariance",
            );

            if variance.is_subnormal() {
                continue;
            }

            let variance_derivatives = gradients(mode, mean, variance, false);
            // In the positive saturated tail, the exact dV/dmu is tiny but
            // the f32 CDF may round to one before its cancellation.  Bound
            // that loss on the input scale, rather than relative to the tiny
            // derivative itself.
            let dvariance_dmean_error =
                (variance_derivatives.0 as f64 - reference.d_variance_d_mean).abs();
            let relative_allowance = 4e-4 * reference.d_variance_d_mean.abs();
            let positive_saturated = mean > 0.0 && mean as f64 / sigma >= 7.99;
            let dvariance_dmean_allowance = if positive_saturated {
                relative_allowance.max(128.0 * f32::EPSILON as f64 * cancellation_scale)
            } else {
                relative_allowance.max(4.0 * f32::from_bits(1) as f64)
            };
            assert!(
                dvariance_dmean_error <= dvariance_dmean_allowance,
                "f32 dvariance/dmean: {} vs {}, absolute error {}, allowed {}",
                variance_derivatives.0,
                reference.d_variance_d_mean,
                dvariance_dmean_error,
                dvariance_dmean_allowance,
            );
            assert_f32_reference(
                variance_derivatives.1,
                reference.d_variance_d_variance,
                4e-4,
                "f32 dvariance/dvariance",
            );
        }
    }

    #[test]
    fn ndarray_f32_tail_gradients_match_gaussian_identities() {
        for tail in TAILS {
            for variance in [1e-24f32, 1.0, 1e24] {
                let sigma = (variance as f64).sqrt();
                let mean = tail.alpha as f32 * variance.sqrt();
                let expected_mean = sigma * tail.mean;
                let expected = [
                    [tail.p, tail.phi / (2.0 * sigma)],
                    [
                        2.0 * expected_mean * (1.0 - tail.p),
                        tail.p - tail.mean * tail.phi,
                    ],
                ];
                for mode in ["diagonal", "full", "leaky0"] {
                    for (mean_loss, derivatives) in [true, false].into_iter().zip(expected) {
                        let (dmu, dvar) = gradients(mode, mean, variance, mean_loss);
                        assert_relative(dmu as f64, derivatives[0], 2e-4, "f32 d/dmu");
                        assert_relative(dvar as f64, derivatives[1], 2e-4, "f32 d/dvar");
                    }
                }
            }
        }
    }

    #[test]
    fn ndarray_f32_marginals_match_frozen_references_at_actual_input_bits() {
        for reference in F32_MARGINAL_REFERENCES {
            assert_f32_marginal_reference(reference);
        }
    }

    #[test]
    fn ndarray_f32_cutoffs_and_deterministic_inputs_follow_documented_conventions() {
        let cutoff_cases = [
            (0xc100_0001, 0.0, 0.0),
            (0xc100_0000, 0.0, 0.0),
            (0x4100_0000, 8.0, 1.0),
            (0x4100_0001, f32::from_bits(0x4100_0001), 1.0),
        ];
        for (mean_bits, expected_mean, expected_variance) in cutoff_cases {
            for mode in ["diagonal", "full", "leaky0"] {
                let (actual_mean, actual_variance) =
                    f32_outputs(mode, f32::from_bits(mean_bits), 1.0);
                assert_eq!(
                    actual_mean, expected_mean,
                    "{mode} mean bits {mean_bits:#010x}"
                );
                assert_eq!(
                    actual_variance, expected_variance,
                    "{mode} variance bits {mean_bits:#010x}"
                );
            }
        }

        for (mean, expected_mean) in [(-1.0, 0.0), (0.0, 0.0), (1.0, 1.0)] {
            for mode in ["diagonal", "full", "leaky0"] {
                let (actual_mean, actual_variance) = f32_outputs(mode, mean, 0.0);
                assert_eq!(
                    actual_mean, expected_mean,
                    "{mode} deterministic mean {mean}"
                );
                assert_eq!(actual_variance, 0.0, "{mode} deterministic variance {mean}");
            }
        }
    }

    #[test]
    fn ndarray_f32_scaled_subnormal_variance_adjoints_match_frozen_reference() {
        let reference = F32_MARGINAL_REFERENCES
            .into_iter()
            .find(|reference| f32::from_bits(reference.variance_bits).is_subnormal())
            .expect("the frozen references include a subnormal variance");
        // Mathematically, dV/dalpha is below f32 range here even though the
        // final dV/dmean is representable. Scale only this variance loss and
        // divide extracted gradients on the host; mean-loss gradients remain
        // deliberately unscaled because dM/dvariance is already very large.
        for mode in ["diagonal", "full", "leaky0"] {
            let (dvariance_dmean, dvariance_dvariance) = scaled_variance_gradients(
                mode,
                f32::from_bits(reference.mean_bits),
                f32::from_bits(reference.variance_bits),
            );
            assert_relative(
                dvariance_dmean,
                reference.d_variance_d_mean,
                4e-4,
                "scaled f32 dvariance/dmean",
            );
            assert_relative(
                dvariance_dvariance,
                reference.d_variance_d_variance,
                4e-4,
                "scaled f32 dvariance/dvariance",
            );
        }
    }

    #[test]
    fn ndarray_f64_tail_values_and_gradients_match_erfc_references() {
        type Ad64 = Autodiff<NdArray<f64>>;
        let device = Default::default();
        for tail in TAILS {
            for mean_loss in [true, false] {
                let mean = Tensor::<Ad64, 2>::from_data([[tail.alpha]], (&device, DType::F64))
                    .require_grad();
                let var =
                    Tensor::<Ad64, 2>::from_data([[1.0]], (&device, DType::F64)).require_grad();
                let full = MomentsFull::from_diagonal(mean.clone(), var.clone());
                assert_eq!(full.cov.dtype(), DType::F64);
                let full_out = propagate_relu_full(&full);
                let full_var = full_out.variance();
                let out = propagate_relu(&Moments::new(mean.clone(), var.clone()));
                let scalar = |t: Tensor<Ad64, 2>| t.into_data().to_vec::<f64>().unwrap()[0];
                assert_relative(
                    scalar(full_out.mean),
                    tail.mean,
                    1e-11,
                    "Burn full f64 mean",
                );
                assert_relative(scalar(full_var), tail.var, 1e-11, "Burn full f64 variance");
                assert_relative(scalar(out.mean.clone()), tail.mean, 1e-11, "Burn f64 mean");
                assert_relative(
                    scalar(out.var.clone()),
                    tail.var,
                    1e-11,
                    "Burn f64 variance",
                );
                let (loss, expected) = if mean_loss {
                    (out.mean.sum(), [tail.p, tail.phi / 2.0])
                } else {
                    (
                        out.var.sum(),
                        [
                            2.0 * tail.mean * (1.0 - tail.p),
                            tail.p - tail.mean * tail.phi,
                        ],
                    )
                };
                let gradients = loss.backward();
                let dmu = mean
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .to_vec::<f64>()
                    .unwrap()[0];
                let dvar = var
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .to_vec::<f64>()
                    .unwrap()[0];
                assert_relative(dmu, expected[0], 1e-10, "Burn f64 d/dmu");
                assert_relative(dvar, expected[1], 1e-10, "Burn f64 d/dvar");
            }
        }
    }
}
