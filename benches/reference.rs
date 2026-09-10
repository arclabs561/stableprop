use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stableprop::{propagate_linear, propagate_relu, Moments};
use std::hint::black_box;

fn covariance(width: usize) -> Vec<Vec<f64>> {
    let mut factors = vec![vec![0.0; width]; width];
    for (i, row) in factors.iter_mut().enumerate() {
        row[i] = 0.6;
        row[0] += 0.08 * (i % 5) as f64 - 0.16;
        row[1] += if i % 2 == 0 { 0.06 } else { -0.05 };
        row[2] += 0.03 * (i % 3) as f64 - 0.03;
    }
    (0..width)
        .map(|i| {
            (0..width)
                .map(|j| (0..width).map(|k| factors[i][k] * factors[j][k]).sum())
                .collect()
        })
        .collect()
}

fn affine_fixture(input: usize, output: usize) -> (Moments, Vec<Vec<f64>>, Vec<f64>) {
    let moments = Moments {
        mean: (0..input).map(|i| 0.1 * (i % 7) as f64 - 0.3).collect(),
        cov: covariance(input),
    };
    let weight = (0..output)
        .map(|row| {
            (0..input)
                .map(|column| 0.025 * ((row * 3 + column * 5) % 11) as f64 - 0.125)
                .collect()
        })
        .collect();
    let bias = (0..output).map(|i| 0.04 * (i % 5) as f64 - 0.08).collect();
    (moments, weight, bias)
}

fn scalar_affine(input: &Moments, weight: &[Vec<f64>], bias: &[f64]) -> Moments {
    let output = weight.len();
    let width = input.mean.len();
    let mean = (0..output)
        .map(|row| {
            bias[row]
                + (0..width)
                    .map(|column| weight[row][column] * input.mean[column])
                    .sum::<f64>()
        })
        .collect();
    let cov = (0..output)
        .map(|left| {
            (0..output)
                .map(|right| {
                    (0..width)
                        .map(|i| {
                            (0..width)
                                .map(|j| weight[left][i] * input.cov[i][j] * weight[right][j])
                                .sum::<f64>()
                        })
                        .sum()
                })
                .collect()
        })
        .collect();
    Moments { mean, cov }
}

fn assert_moments_close(actual: &Moments, expected: &Moments) {
    assert_eq!(actual.mean.len(), expected.mean.len());
    assert_eq!(actual.cov.len(), expected.cov.len());
    for (actual, expected) in actual.mean.iter().zip(&expected.mean) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "mean {actual} != {expected}"
        );
    }
    for (actual, expected) in actual.cov.iter().zip(&expected.cov) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "covariance {actual} != {expected}"
            );
        }
    }
}

fn reference_benches(c: &mut Criterion) {
    let mut affine = c.benchmark_group("reference_affine");
    for (input, output) in [(8, 8), (32, 8), (8, 32), (64, 64), (128, 128)] {
        let (moments, weight, bias) = affine_fixture(input, output);
        let expected = scalar_affine(&moments, &weight, &bias);
        assert_moments_close(&propagate_linear(&moments, &weight, &bias), &expected);
        affine.bench_with_input(
            BenchmarkId::new("features", format!("{input}to{output}")),
            &(moments, weight, bias),
            |b, (moments, weight, bias)| {
                b.iter(|| propagate_linear(black_box(moments), black_box(weight), black_box(bias)))
            },
        );
    }
    affine.finish();

    let mut relu = c.benchmark_group("reference_relu");
    for width in [8, 64, 128] {
        let moments = Moments {
            mean: (0..width).map(|i| 0.1 * (i % 7) as f64 - 0.3).collect(),
            cov: covariance(width),
        };
        relu.bench_with_input(
            BenchmarkId::new("features", width),
            &moments,
            |b, moments| b.iter(|| propagate_relu(black_box(moments))),
        );
    }
    relu.finish();
}

criterion_group!(benches, reference_benches);
criterion_main!(benches);
