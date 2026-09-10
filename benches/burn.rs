#![cfg(feature = "burn")]

use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stableprop::burn_sdp::{
    propagate_linear, propagate_linear_full, propagate_relu, propagate_relu_full, Moments,
    MomentsFull,
};
use std::hint::black_box;

type Backend = NdArray<f32>;

fn covariance(width: usize) -> Vec<f32> {
    let mut factors = vec![vec![0.0f32; width]; width];
    for (i, row) in factors.iter_mut().enumerate() {
        row[i] = 0.6;
        row[0] += 0.08 * (i % 5) as f32 - 0.16;
        row[1] += if i % 2 == 0 { 0.06 } else { -0.05 };
        row[2] += 0.03 * (i % 3) as f32 - 0.03;
    }
    (0..width)
        .flat_map(|i| {
            let factors = &factors;
            (0..width).map(move |j| (0..width).map(|k| factors[i][k] * factors[j][k]).sum())
        })
        .collect()
}

fn relu_covariance(width: usize) -> Vec<f32> {
    let cov = covariance(width);
    (0..width)
        .flat_map(|i| {
            let cov = &cov;
            (0..width)
                .map(move |j| cov[i * width + j] / (cov[i * width + i] * cov[j * width + j]).sqrt())
        })
        .collect()
}

fn scalar_affine(
    mean: &[f32],
    cov: &[f32],
    batch: usize,
    width: usize,
    weight: &[f32],
    bias: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_mean = vec![0.0; batch * width];
    let mut out_cov = vec![0.0; batch * width * width];
    for row in 0..batch {
        for out in 0..width {
            out_mean[row * width + out] = bias[out]
                + (0..width)
                    .map(|input| mean[row * width + input] * weight[input * width + out])
                    .sum::<f32>();
        }
    }
    let mut one_cov = vec![0.0; width * width];
    for left in 0..width {
        for right in 0..width {
            one_cov[left * width + right] = (0..width)
                .map(|i| {
                    (0..width)
                        .map(|j| {
                            weight[i * width + left]
                                * cov[i * width + j]
                                * weight[j * width + right]
                        })
                        .sum::<f32>()
                })
                .sum();
        }
    }
    for row in 0..batch {
        out_cov[row * width * width..(row + 1) * width * width].copy_from_slice(&one_cov);
    }
    (out_mean, out_cov)
}

fn scalar_diagonal_variance(
    variance: &[f32],
    batch: usize,
    width: usize,
    weight: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0; batch * width];
    for row in 0..batch {
        for out in 0..width {
            output[row * width + out] = (0..width)
                .map(|input| {
                    let weight = weight[input * width + out];
                    variance[row * width + input] * weight * weight
                })
                .sum();
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 2e-5 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}

fn burn_benches(c: &mut Criterion) {
    let device = Default::default();
    let mut affine = c.benchmark_group("burn_affine_f32");

    for (batch, width) in [(8, 16), (64, 64)] {
        let mean: Vec<f32> = (0..batch * width)
            .map(|i| 0.1 * (i % 7) as f32 - 0.3)
            .collect();
        let cov_one = covariance(width);
        let cov: Vec<f32> = (0..batch).flat_map(|_| cov_one.iter().copied()).collect();
        let mut variance = Vec::with_capacity(batch * width);
        for row in 0..batch {
            for i in 0..width {
                variance.push(cov[(row * width + i) * width + i]);
            }
        }
        let weight: Vec<f32> = (0..width * width)
            .map(|i| 0.025 * ((i * 5 + i / width * 3) % 11) as f32 - 0.125)
            .collect();
        let bias: Vec<f32> = (0..width).map(|i| 0.04 * (i % 5) as f32 - 0.08).collect();
        let (expected_mean, expected_cov) =
            scalar_affine(&mean, &cov, batch, width, &weight, &bias);
        let expected_diagonal_var = scalar_diagonal_variance(&variance, batch, width, &weight);

        let mean_tensor =
            Tensor::<Backend, 2>::from_data(TensorData::new(mean.clone(), [batch, width]), &device);
        let variance_tensor =
            Tensor::<Backend, 2>::from_data(TensorData::new(variance, [batch, width]), &device);
        let cov_tensor =
            Tensor::<Backend, 3>::from_data(TensorData::new(cov, [batch, width, width]), &device);
        let weight_tensor =
            Tensor::<Backend, 2>::from_data(TensorData::new(weight, [width, width]), &device);
        let bias_tensor = Tensor::<Backend, 1>::from_data(TensorData::new(bias, [width]), &device);

        let diagonal_input = Moments::new(mean_tensor.clone(), variance_tensor);
        let full_input = MomentsFull::new(mean_tensor, cov_tensor);
        let diagonal_out = propagate_linear(
            &diagonal_input,
            weight_tensor.clone(),
            Some(bias_tensor.clone()),
        );
        assert_close(
            &diagonal_out.mean.into_data().to_vec::<f32>().unwrap(),
            &expected_mean,
        );
        let diagonal_var = diagonal_out.var.into_data().to_vec::<f32>().unwrap();
        assert_close(&diagonal_var, &expected_diagonal_var);
        let full_out = propagate_linear_full(
            &full_input,
            weight_tensor.clone(),
            Some(bias_tensor.clone()),
        );
        assert_close(
            &full_out.mean.into_data().to_vec::<f32>().unwrap(),
            &expected_mean,
        );
        assert_close(
            &full_out.cov.into_data().to_vec::<f32>().unwrap(),
            &expected_cov,
        );

        let id = format!("batch{batch}_width{width}");
        affine.bench_with_input(
            BenchmarkId::new("diagonal", &id),
            &(diagonal_input, weight_tensor.clone(), bias_tensor.clone()),
            |b, (input, weight, bias)| {
                b.iter(|| {
                    propagate_linear(
                        black_box(input),
                        black_box(weight.clone()),
                        Some(black_box(bias.clone())),
                    )
                })
            },
        );
        affine.bench_with_input(
            BenchmarkId::new("full", id),
            &(full_input, weight_tensor, bias_tensor),
            |b, (input, weight, bias)| {
                b.iter(|| {
                    propagate_linear_full(
                        black_box(input),
                        black_box(weight.clone()),
                        Some(black_box(bias.clone())),
                    )
                })
            },
        );
    }
    affine.finish();

    let mut relu = c.benchmark_group("burn_relu_f32");
    for (batch, width) in [(8, 16), (64, 64)] {
        let cov_one = relu_covariance(width);
        let cov: Vec<f32> = (0..batch).flat_map(|_| cov_one.iter().copied()).collect();
        let variance = vec![1.0; batch * width];
        for (case, mean) in [
            (
                "central",
                (0..batch * width)
                    .map(|i| [-1.0f32, 0.0, 1.0][i % 3])
                    .collect(),
            ),
            ("negative_tail", vec![-7.0; batch * width]),
        ] {
            let mean_tensor =
                Tensor::<Backend, 2>::from_data(TensorData::new(mean, [batch, width]), &device);
            let diagonal_input = Moments::new(
                mean_tensor.clone(),
                Tensor::<Backend, 2>::from_data(
                    TensorData::new(variance.clone(), [batch, width]),
                    &device,
                ),
            );
            let full_input = MomentsFull::new(
                mean_tensor,
                Tensor::<Backend, 3>::from_data(
                    TensorData::new(cov.clone(), [batch, width, width]),
                    &device,
                ),
            );
            let id = format!("{case}_b{batch}w{width}");
            relu.bench_with_input(
                BenchmarkId::new("diagonal", &id),
                &diagonal_input,
                |b, input| b.iter(|| propagate_relu(black_box(input))),
            );
            relu.bench_with_input(BenchmarkId::new("full", id), &full_input, |b, input| {
                b.iter(|| propagate_relu_full(black_box(input)))
            });
        }
    }
    relu.finish();
}

criterion_group!(benches, burn_benches);
criterion_main!(benches);
