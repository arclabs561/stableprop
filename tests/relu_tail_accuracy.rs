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
    use super::{assert_relative, Tail, TAILS};
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
