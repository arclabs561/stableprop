//! Combines tuplet's pairwise contrastive loss with an embedding-variance penalty.
//!
//! tuplet supplies the loss; stableprop propagates input noise through the
//! Burn MLP to estimate embedding variance. The example compares encoders
//! from the same initialization on held-out clean and noisy inputs. This fixed
//! synthetic comparison does not establish general robustness.
//!
//! Run: `cargo run --release --example tuplet_contrastive --features burn`

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Device, Distribution, Int, Tensor, TensorData};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{propagate_linear, propagate_relu, Moments};
use tuplet::burn_losses::contrastive_loss;

type Ad = Autodiff<NdArray<f32>>;

const D_IN: usize = 6;
const HIDDEN: usize = 32;
const EMBED: usize = 4;
const N_CLASS: usize = 3;
const TRAIN_PER_CLASS: usize = 400;
const TEST_PER_CLASS: usize = 200;
const N_TRAIN: usize = N_CLASS * TRAIN_PER_CLASS;
const N_TEST: usize = N_CLASS * TEST_PER_CLASS;
const TRAIN_STD: f64 = 0.3;
const TEST_STD: f64 = 0.5;
const MARGIN: f32 = 1.0;

#[derive(Module, Debug)]
struct Encoder<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> Encoder<B> {
    fn init(device: &B::Device) -> Self {
        let model = Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, EMBED).init(device),
        };
        // Burn parameters initialize lazily; materialize before cloning a baseline.
        for layer in [&model.lin1, &model.lin2] {
            drop(layer.weight.val());
            if let Some(bias) = &layer.bias {
                drop(bias.val());
            }
        }
        model
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.lin2.forward(activation::relu(self.lin1.forward(x)))
    }
    /// Analytic embedding variance under input noise `std` (the stableprop part).
    fn embedding_var(&self, x: Tensor<B, 2>, std: f64) -> Tensor<B, 2> {
        let [n, d] = x.dims();
        let var0 = Tensor::<B, 2>::full([n, d], std * std, &x.device());
        let w1 = self.lin1.weight.val();
        let b1 = self.lin1.bias.as_ref().map(|p| p.val());
        let w2 = self.lin2.weight.val();
        let b2 = self.lin2.bias.as_ref().map(|p| p.val());
        let m1 = propagate_relu(&propagate_linear(&Moments::new(x, var0), w1, b1));
        propagate_linear(&m1, w2, b2).var
    }
}

fn main() {
    let dev = Device::<Ad>::default();

    // Class blobs: class c shifted along a per-class direction.
    let make_split = |per_class: usize, seed: u64| -> (Tensor<Ad, 2>, Vec<i64>) {
        let n = N_CLASS * per_class;
        <Ad as Backend>::seed(&dev, seed);
        let mut xv = Tensor::<Ad, 2>::random([n, D_IN], Distribution::Normal(0.0, 0.7), &dev)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let labels: Vec<i64> = (0..n).map(|i| (i / per_class) as i64).collect();
        for i in 0..n {
            let c = labels[i] as f32;
            xv[i * D_IN] += 1.8 * c;
            xv[i * D_IN + 1] -= 1.4 * c;
            xv[i * D_IN + 2] += 1.0 * c;
        }
        (
            Tensor::<Ad, 2>::from_data(TensorData::new(xv, [n, D_IN]), &dev),
            labels,
        )
    };
    let (x_train, train_labels) = make_split(TRAIN_PER_CLASS, 0x7A91_0001);
    let (x_test, test_labels) = make_split(TEST_PER_CLASS, 0x7A91_0002);

    // One deterministic pair per anchor; this exercises pairwise contrastive loss,
    // not a triplet or multi-negative objective.
    let perm: Vec<usize> = (0..N_TRAIN).map(|i| (i * 7 + 13) % N_TRAIN).collect();
    let same: Vec<i64> = (0..N_TRAIN)
        .map(|i| (train_labels[i] == train_labels[perm[i]]) as i64)
        .collect();
    let perm_t = Tensor::<Ad, 1, Int>::from_data(
        TensorData::new(
            perm.iter().map(|&p| p as i64).collect::<Vec<_>>(),
            [N_TRAIN],
        ),
        &dev,
    );
    let same_t = Tensor::<Ad, 1, Int>::from_data(TensorData::new(same, [N_TRAIN]), &dev);

    <Ad as Backend>::seed(&dev, 0x7A91_1000);
    let init = Encoder::<Ad>::init(&dev);
    let train = |mut model: Encoder<Ad>, lambda: f64| -> Encoder<Ad> {
        let mut optim = AdamConfig::new().init();
        for _ in 0..600 {
            let ea = model.forward(x_train.clone());
            let eb = model.forward(x_train.clone().select(0, perm_t.clone()));
            let mut loss = contrastive_loss(ea, eb, same_t.clone(), MARGIN);
            if lambda > 0.0 {
                loss = loss
                    + model
                        .embedding_var(x_train.clone(), TRAIN_STD)
                        .mean()
                        .mul_scalar(lambda);
            }
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-3, model, grads);
        }
        model
    };
    let plain = train(init.clone(), 0.0);
    let robust = train(init, 0.3);

    // Train-set centroids; evaluate only on the held-out split.
    let centroids = |model: &Encoder<Ad>| -> Vec<f32> {
        let e = model
            .forward(x_train.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut c = vec![0.0f32; N_CLASS * EMBED];
        for i in 0..N_TRAIN {
            for k in 0..EMBED {
                c[train_labels[i] as usize * EMBED + k] +=
                    e[i * EMBED + k] / TRAIN_PER_CLASS as f32;
            }
        }
        c
    };
    <Ad as Backend>::seed(&dev, 0x7A91_2000);
    let draws = 10;
    let noisy_test_draws: Vec<Tensor<Ad, 2>> = (0..draws)
        .map(|_| Tensor::<Ad, 2>::random([N_TEST, D_IN], Distribution::Normal(0.0, TEST_STD), &dev))
        .collect();
    let accuracy = |model: &Encoder<Ad>, cen: &[f32], noise_draws: &[Tensor<Ad, 2>]| -> f64 {
        let mut correct = 0;
        for noise in noise_draws {
            let e = model
                .forward(x_test.clone() + noise.clone())
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for i in 0..N_TEST {
                let mut best = (0usize, f32::MAX);
                for cl in 0..N_CLASS {
                    let dist: f32 = (0..EMBED)
                        .map(|k| (e[i * EMBED + k] - cen[cl * EMBED + k]).powi(2))
                        .sum();
                    if dist < best.1 {
                        best = (cl, dist);
                    }
                }
                if best.0 as i64 == test_labels[i] {
                    correct += 1;
                }
            }
        }
        correct as f64 / (N_TEST * noise_draws.len()) as f64
    };

    let clean_draw = vec![Tensor::<Ad, 2>::zeros([N_TEST, D_IN], &dev)];
    let p_clean = accuracy(&plain, &centroids(&plain), &clean_draw);
    let r_clean = accuracy(&robust, &centroids(&robust), &clean_draw);
    let p_noisy = accuracy(&plain, &centroids(&plain), &noisy_test_draws);
    let r_noisy = accuracy(&robust, &centroids(&robust), &noisy_test_draws);
    // This is the actual regularizer quantity, measured on the training
    // inputs at the perturbation scale used while optimizing.
    let propagated_variance = |model: &Encoder<Ad>| -> f64 {
        model
            .embedding_var(x_train.clone(), TRAIN_STD)
            .mean()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0] as f64
    };
    // A low variance can come from shrinking all embeddings. Report their RMS
    // scale on held-out clean inputs beside task quality to expose that collapse.
    let embedding_rms = |model: &Encoder<Ad>| -> f64 {
        let embedding = model
            .forward(x_test.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        (embedding
            .iter()
            .map(|&value| (value as f64).powi(2))
            .sum::<f64>()
            / embedding.len() as f64)
            .sqrt()
    };
    println!("nearest-centroid accuracy on held-out inputs:");
    println!(
        "  model                         clean    noisy (std {TEST_STD})  prop var (std {TRAIN_STD})  embed RMS"
    );
    println!(
        "  plain contrastive              {p_clean:.3}    {p_noisy:.3}             {:.5}      {:.4}",
        propagated_variance(&plain),
        embedding_rms(&plain),
    );
    println!(
        "  contrastive + variance penalty {r_clean:.3}    {r_noisy:.3}             {:.5}      {:.4}",
        propagated_variance(&robust),
        embedding_rms(&robust),
    );
    println!("\nstableprop and tuplet compose in Burn end to end: the encoder trains under");
    println!(
        "tuplet's pairwise contrastive loss while stableprop supplies the analytic embedding variance."
    );
    println!("This fixed-pair synthetic comparison is not a general performance guarantee.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_baseline_has_identical_forward_predictions() {
        let device = Device::<Ad>::default();
        <Ad as Backend>::seed(&device, 0x7A91_7E57);
        let baseline = Encoder::<Ad>::init(&device);
        let left = baseline.clone();
        let right = baseline.clone();
        let probe = Tensor::<Ad, 2>::from_data(
            TensorData::new(
                vec![
                    -1.0, -0.5, 0.25, 0.75, 1.25, 1.5, 0.4, -0.8, 1.2, -1.6, 2.0, -2.4,
                ],
                [2, D_IN],
            ),
            &device,
        );

        let left = left
            .forward(probe.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let right = right.forward(probe).into_data().to_vec::<f32>().unwrap();
        assert_eq!(
            left, right,
            "cloned baselines must share initialized weights"
        );
    }
}
