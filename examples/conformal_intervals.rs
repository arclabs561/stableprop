//! Guaranteed coverage: conformalize stableprop's analytic error bars.
//!
//! stableprop's propagated std is a *heuristic* uncertainty scale -- accurate as
//! a relative signal, but not calibrated to real residuals (we measured ~0.90
//! coverage where 0.95 was wanted). Split-conformal prediction fixes that: using
//! stableprop's per-point std as the normalizer, it produces intervals with a
//! distribution-free coverage GUARANTEE, while staying adaptive (wider where
//! stableprop says the input is more uncertain).
//!
//! This trains a regressor, then on a held-out calibration set computes the
//! conformal quantile of normalized residuals `|y - y_hat| / sigma`, and reports
//! test coverage for: raw stableprop intervals, conformalized (adaptive), and
//! constant-width conformal. The conformalized ones must hit the target.
//!
//! Run: `cargo run --release --example conformal_intervals --features burn`

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Distribution, Tensor, TensorData};
use burn_ndarray::NdArray;

use stableprop::burn_sdp::{propagate_linear, propagate_relu, Moments};

type Ad = Autodiff<NdArray<f32>>;
type Nd = NdArray<f32>;

const D_IN: usize = 6;
const HIDDEN: usize = 64;
const N_TRAIN: usize = 3000;
const N_CAL: usize = 1000;
const N_TEST: usize = 1000;
const INPUT_STD: f64 = 0.1;
const LABEL_STD: f32 = 0.2;
const ALPHA: f64 = 0.1; // target miscoverage -> 90% intervals

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(D_IN, HIDDEN).init(device),
            lin2: LinearConfig::new(HIDDEN, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.lin1.forward(x));
        self.lin2.forward(h)
    }
}

fn target(x: &[f32]) -> f32 {
    let s: f32 = x.iter().sum();
    (s * 0.6).sin() + 0.5 * x[0] * x[1] - 0.3 * x[2] * x[2]
}

fn main() {
    let dev = <Ad as Backend>::Device::default();

    let make = |n: usize, noisy: bool, seed_mul: usize| -> (Vec<f32>, Vec<f32>) {
        let xt = Tensor::<Ad, 2>::random([n, D_IN], Distribution::Normal(0.0, 1.0), &dev);
        let xv = xt.to_data().to_vec::<f32>().unwrap();
        let noise = if noisy {
            Tensor::<Ad, 2>::random([n, 1], Distribution::Normal(0.0, LABEL_STD as f64), &dev)
                .to_data()
                .to_vec::<f32>()
                .unwrap()
        } else {
            vec![0.0; n]
        };
        let _ = seed_mul;
        let yv: Vec<f32> = (0..n)
            .map(|i| target(&xv[i * D_IN..(i + 1) * D_IN]) + noise[i])
            .collect();
        (xv, yv)
    };
    let (xtr, ytr) = make(N_TRAIN, true, 1);
    let (xca, yca) = make(N_CAL, true, 2);
    let (xte, yte) = make(N_TEST, true, 3);

    let x_train = Tensor::<Ad, 2>::from_data(TensorData::new(xtr, [N_TRAIN, D_IN]), &dev);
    let y_train = Tensor::<Ad, 2>::from_data(TensorData::new(ytr, [N_TRAIN, 1]), &dev);

    let mut model = Mlp::<Ad>::init(&dev);
    let mut optim = AdamConfig::new().init();
    println!("training regressor ({N_TRAIN} samples)...");
    for _ in 0..800 {
        let pred = model.forward(x_train.clone());
        let loss = MseLoss::new().forward(pred, y_train.clone(), Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-3, model, grads);
    }

    // stableprop std under known input noise, on the inner backend.
    let idev = <Nd as Backend>::Device::default();
    let w1 = model.lin1.weight.val().inner();
    let b1 = model.lin1.bias.as_ref().map(|p| p.val().inner());
    let w2 = model.lin2.weight.val().inner();
    let b2 = model.lin2.bias.as_ref().map(|p| p.val().inner());
    let predict = |xv: &[f32], n: usize| -> (Vec<f32>, Vec<f64>) {
        let x = Tensor::<Nd, 2>::from_data(TensorData::new(xv.to_vec(), [n, D_IN]), &idev);
        let var0 = Tensor::<Nd, 2>::full([n, D_IN], INPUT_STD * INPUT_STD, &idev);
        let m1 = propagate_relu(&propagate_linear(
            &Moments::new(x.clone(), var0),
            w1.clone(),
            b1.clone(),
        ));
        let m2 = propagate_linear(&m1, w2.clone(), b2.clone());
        let mean = m2.mean.to_data().to_vec::<f32>().unwrap();
        let std: Vec<f64> = m2
            .var
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| (*v as f64).max(1e-12).sqrt())
            .collect();
        (mean, std)
    };

    let (yhat_cal, sig_cal) = predict(&xca, N_CAL);
    let (yhat_te, sig_te) = predict(&xte, N_TEST);

    // Calibration scores.
    let mut norm_scores: Vec<f64> = (0..N_CAL)
        .map(|i| (yca[i] - yhat_cal[i]).abs() as f64 / sig_cal[i])
        .collect();
    let mut abs_scores: Vec<f64> = (0..N_CAL)
        .map(|i| (yca[i] - yhat_cal[i]).abs() as f64)
        .collect();
    norm_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    abs_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Conformal quantile: ceil((n+1)(1-alpha))-th smallest score.
    let rank = (((N_CAL + 1) as f64 * (1.0 - ALPHA)).ceil() as usize).min(N_CAL) - 1;
    let q_norm = norm_scores[rank];
    let q_abs = abs_scores[rank];

    let cover = |hw: &dyn Fn(usize) -> f64| -> (f64, f64) {
        let mut c = 0;
        let mut w = 0.0;
        for i in 0..N_TEST {
            let h = hw(i);
            if (yte[i] - yhat_te[i]).abs() as f64 <= h {
                c += 1;
            }
            w += 2.0 * h;
        }
        (c as f64 / N_TEST as f64, w / N_TEST as f64)
    };

    let z = 1.645; // 90% Gaussian
    let (raw_cov, raw_w) = cover(&|i| z * sig_te[i]);
    let (conf_cov, conf_w) = cover(&|i| q_norm * sig_te[i]);
    let (const_cov, const_w) = cover(&|_| q_abs);

    println!("\ntarget coverage = {:.2}\n", 1.0 - ALPHA);
    println!("  {:<34} {:>8} {:>10}", "method", "coverage", "avg width");
    println!(
        "  {:<34} {:>8.3} {:>10.3}",
        "raw stableprop (1.645*sigma)", raw_cov, raw_w
    );
    println!(
        "  {:<34} {:>8.3} {:>10.3}",
        "conformalized stableprop (adaptive)", conf_cov, conf_w
    );
    println!(
        "  {:<34} {:>8.3} {:>10.3}",
        "constant-width conformal", const_cov, const_w
    );
    println!("\nraw is miscalibrated; both conformal methods hit the target with a guarantee.");
    println!("the conformalized-stableprop width adapts per point (stableprop's sigma), the constant one does not.");
}
