//! stableprop composed with tuplet: contrastive embeddings with analytic
//! embedding uncertainty, and a noise-robust training variant.
//!
//! tuplet supplies the contrastive loss; the encoder (a Burn MLP) is where
//! stableprop applies. Propagating input noise through the encoder gives the
//! analytic variance of the EMBEDDING. Penalizing it during training nudges the
//! embeddings to move less under input noise. This trains two encoders (plain
//! contrastive vs contrastive + the stableprop variance penalty) from the same
//! init and evaluates nearest-centroid accuracy on noisy inputs. The point is
//! that the two crates compose in Burn end to end; the robustness gain here is
//! modest.
//!
//! Run: `cargo run --release --example tuplet_contrastive --features burn`

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Distribution, Int, Tensor, TensorData};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{propagate_linear, propagate_relu, Moments};
use tuplet::burn_losses::contrastive_loss;

type Ad = Autodiff<NdArray<f32>>;

const D_IN: usize = 6;
const HIDDEN: usize = 32;
const EMBED: usize = 4;
const N_CLASS: usize = 3;
const PER_CLASS: usize = 400;
const N: usize = N_CLASS * PER_CLASS;
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
        Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, EMBED).init(device),
        }
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
    let dev = <Ad as Backend>::Device::default();

    // Class blobs: class c shifted along a per-class direction.
    let base = Tensor::<Ad, 2>::random([N, D_IN], Distribution::Normal(0.0, 0.7), &dev)
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let labels: Vec<i64> = (0..N).map(|i| (i / PER_CLASS) as i64).collect();
    let mut xv = base;
    for i in 0..N {
        let c = labels[i] as f32;
        xv[i * D_IN] += 1.8 * c;
        xv[i * D_IN + 1] -= 1.4 * c;
        xv[i * D_IN + 2] += 1.0 * c;
    }
    let x = Tensor::<Ad, 2>::from_data(TensorData::new(xv, [N, D_IN]), &dev);

    // Fixed pairing: each anchor paired with a shifted partner; label = same class.
    let perm: Vec<usize> = (0..N).map(|i| (i * 7 + 13) % N).collect();
    let same: Vec<i64> = (0..N)
        .map(|i| (labels[i] == labels[perm[i]]) as i64)
        .collect();
    let perm_t = Tensor::<Ad, 1, Int>::from_data(
        TensorData::new(perm.iter().map(|&p| p as i64).collect::<Vec<_>>(), [N]),
        &dev,
    );
    let same_t = Tensor::<Ad, 1, Int>::from_data(TensorData::new(same, [N]), &dev);

    let init = Encoder::<Ad>::init(&dev);
    let train = |mut model: Encoder<Ad>, lambda: f64| -> Encoder<Ad> {
        let mut optim = AdamConfig::new().init();
        for _ in 0..600 {
            let ea = model.forward(x.clone());
            let eb = model.forward(x.clone().select(0, perm_t.clone()));
            let mut loss = contrastive_loss(ea, eb, same_t.clone(), MARGIN);
            if lambda > 0.0 {
                loss = loss
                    + model
                        .embedding_var(x.clone(), TRAIN_STD)
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

    // Nearest-class-centroid accuracy on noisy inputs (same noise for both).
    let centroids = |model: &Encoder<Ad>| -> Vec<f32> {
        let e = model
            .forward(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut c = vec![0.0f32; N_CLASS * EMBED];
        for i in 0..N {
            for k in 0..EMBED {
                c[labels[i] as usize * EMBED + k] += e[i * EMBED + k] / PER_CLASS as f32;
            }
        }
        c
    };
    let acc = |model: &Encoder<Ad>, cen: &[f32]| -> f64 {
        let mut correct = 0;
        let draws = 10;
        for _ in 0..draws {
            let noise =
                Tensor::<Ad, 2>::random([N, D_IN], Distribution::Normal(0.0, TEST_STD), &dev);
            let e = model
                .forward(x.clone() + noise)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for i in 0..N {
                let mut best = (0usize, f32::MAX);
                for cl in 0..N_CLASS {
                    let dist: f32 = (0..EMBED)
                        .map(|k| (e[i * EMBED + k] - cen[cl * EMBED + k]).powi(2))
                        .sum();
                    if dist < best.1 {
                        best = (cl, dist);
                    }
                }
                if best.0 as i64 == labels[i] {
                    correct += 1;
                }
            }
        }
        correct as f64 / (N * draws) as f64
    };

    let p_acc = acc(&plain, &centroids(&plain));
    let r_acc = acc(&robust, &centroids(&robust));
    println!("nearest-centroid accuracy on noisy inputs (test std {TEST_STD}):");
    println!("  plain contrastive              {p_acc:.3}");
    println!("  contrastive + variance penalty {r_acc:.3}");
    println!("\nstableprop and tuplet compose in Burn end to end: the encoder trains under");
    println!(
        "tuplet's contrastive loss while stableprop supplies the analytic embedding variance."
    );
}
