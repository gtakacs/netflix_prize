//! Pure-Rust MLP regressor mirroring sklearn's `MLPRegressor` as used by
//! `blending/mlp.py:mlpr40_new` (`create_nn`): `[d,64,64,1]`, ReLU hidden layers,
//! SGD with Nesterov momentum, L2 regularization and an internal StandardScaler.
//! All math is f64 (inputs/outputs f32). Dense products go through `ndarray`'s
//! `.dot()`, which dispatches to OpenBLAS `dgemm` when the crate is built with
//! `--features blas` and falls back to the pure-Rust kernel otherwise.

use crate::blend::Blender;
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2, Axis};
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Uniform};

/// Hyperparameters for `MlpBlender`. `Default` mirrors the Python `create_nn`
/// (`MLPRegressor(activation='relu', solver='sgd', alpha=0.05,
/// learning_rate_init=0.0004, max_iter=64, hidden_layer_sizes=(64, 64))`) with
/// sklearn's default `momentum=0.9` (Nesterov), `batch_size='auto'` -> 200,
/// `tol=1e-4`, `n_iter_no_change=10`.
#[derive(Clone, Debug)]
pub struct MlpCfg {
    pub hidden: Vec<usize>,
    pub alpha: f64,
    pub lr: f64,
    pub max_iter: usize,
    pub batch_size: usize,
    pub momentum: f64,
    pub tol: f64,
    pub n_iter_no_change: usize,
    pub seed: u64,
}

impl Default for MlpCfg {
    fn default() -> Self {
        Self {
            hidden: vec![64, 64],
            alpha: 0.05,
            lr: 0.0004,
            max_iter: 64,
            batch_size: 200,
            momentum: 0.9,
            tol: 1e-4,
            n_iter_no_change: 10,
            seed: 1,
        }
    }
}

/// StandardScaler equivalent: per-feature centering and scaling by the
/// population standard deviation (`ddof=0`); zero-variance columns get scale 1.
struct Scaler {
    mean: Array1<f64>,
    scale: Array1<f64>,
}

impl Scaler {
    fn fit(x: &Array2<f64>) -> Self {
        let mean = x.mean_axis(Axis(0)).unwrap();
        let meanb = mean.broadcast(x.raw_dim()).unwrap();
        let var = (x - &meanb).mapv(|v| v * v).mean_axis(Axis(0)).unwrap();
        let scale = var.mapv(|v| {
            let s = v.sqrt();
            if s == 0.0 { 1.0 } else { s }
        });
        Self { mean, scale }
    }

    fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let meanb = self.mean.broadcast(x.raw_dim()).unwrap();
        let scaleb = self.scale.broadcast(x.raw_dim()).unwrap();
        (x - &meanb) / scaleb
    }
}

/// Forward pass; returns the activation of every layer (index 0 = input,
/// index `coefs.len()` = output). Hidden layers use ReLU, the output is linear.
fn forward(coefs: &[Array2<f64>], intercepts: &[Array1<f64>], x0: Array2<f64>) -> Vec<Array2<f64>> {
    let l = coefs.len();
    let mut acts = Vec::with_capacity(l + 1);
    acts.push(x0);
    for li in 0..l {
        let mut z = acts[li].dot(&coefs[li]);
        let b = intercepts[li].broadcast(z.raw_dim()).unwrap();
        z += &b;
        if li < l - 1 {
            z.mapv_inplace(|v| v.max(0.0));
        }
        acts.push(z);
    }
    acts
}

/// A trained MLP behind the `Blender` interface. Holds the fitted scaler and the
/// per-layer weight matrices / bias vectors.
pub struct MlpBlender {
    scaler: Scaler,
    coefs: Vec<Array2<f64>>,
    intercepts: Vec<Array1<f64>>,
}

impl Blender for MlpBlender {
    type Cfg = MlpCfg;

    fn fit(x: &[f32], y: &[f32], n_features: usize, cfg: &MlpCfg) -> Self {
        let d = n_features;
        let n = y.len();
        assert_eq!(x.len(), n * d, "x length {} != {n}*{d}", x.len());

        let x64 = Array2::from_shape_fn((n, d), |(i, j)| x[i * d + j] as f64);
        let scaler = Scaler::fit(&x64);
        let xs = scaler.transform(&x64);
        drop(x64); // free the raw copy before training

        // Layer sizes: input, hidden..., single output.
        let mut sizes = vec![d];
        sizes.extend(cfg.hidden.iter().copied());
        sizes.push(1);
        let l = sizes.len() - 1;

        // Glorot-uniform init: bound = sqrt(6 / (fan_in + fan_out)); per layer the
        // weight matrix is drawn (row-major) then the bias vector.
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut coefs: Vec<Array2<f64>> = Vec::with_capacity(l);
        let mut intercepts: Vec<Array1<f64>> = Vec::with_capacity(l);
        for li in 0..l {
            let (fan_in, fan_out) = (sizes[li], sizes[li + 1]);
            let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let u = Uniform::new(-bound, bound).unwrap();
            let w = Array2::from_shape_fn((fan_in, fan_out), |_| u.sample(&mut rng));
            let b = Array1::from_shape_fn(fan_out, |_| u.sample(&mut rng));
            coefs.push(w);
            intercepts.push(b);
        }

        // Nesterov-momentum velocities, one per parameter tensor.
        let mut v_coefs: Vec<Array2<f64>> = coefs.iter().map(|c| Array2::zeros(c.raw_dim())).collect();
        let mut v_inter: Vec<Array1<f64>> = intercepts.iter().map(|b| Array1::zeros(b.raw_dim())).collect();

        let mut idx: Vec<usize> = (0..n).collect();
        let mut best_loss = f64::INFINITY;
        let mut no_improve = 0usize;

        for _epoch in crate::progress!(0..cfg.max_iter) {
            idx.shuffle(&mut rng);
            let mut accumulated = 0.0f64;

            for batch in idx.chunks(cfg.batch_size) {
                let m = batch.len();
                let nb = m as f64;
                let xb = Array2::from_shape_fn((m, d), |(i, j)| xs[[batch[i], j]]);
                let yb = Array2::from_shape_fn((m, 1), |(i, _)| y[batch[i]] as f64);

                let acts = forward(&coefs, &intercepts, xb);

                // Squared loss (mean over elements / 2) + L2 term (alpha/2 * Σw² / m).
                let diff = &acts[l] - &yb;
                let sq = diff.mapv(|e| e * e).sum() / nb / 2.0;
                let l2: f64 = coefs.iter().map(|c| c.iter().map(|w| w * w).sum::<f64>()).sum();
                let batch_loss = sq + 0.5 * cfg.alpha * l2 / nb;
                accumulated += batch_loss * nb;

                // Backprop: per-layer weight/bias gradients.
                let mut coef_grads: Vec<Array2<f64>> = vec![Array2::zeros((0, 0)); l];
                let mut inter_grads: Vec<Array1<f64>> = vec![Array1::zeros(0); l];
                let mut delta = diff; // error at the output layer
                let mut layer = l - 1;
                loop {
                    let mut cg = acts[layer].t().dot(&delta);
                    cg.scaled_add(cfg.alpha, &coefs[layer]);
                    cg.mapv_inplace(|v| v / nb);
                    inter_grads[layer] = delta.mean_axis(Axis(0)).unwrap();
                    coef_grads[layer] = cg;

                    if layer == 0 {
                        break;
                    }
                    let mut prev = delta.dot(&coefs[layer].t());
                    // ReLU derivative: zero out where the layer's activation is 0.
                    prev.zip_mut_with(&acts[layer], |g, &a| {
                        if a == 0.0 {
                            *g = 0.0;
                        }
                    });
                    delta = prev;
                    layer -= 1;
                }

                // SGD update with Nesterov momentum:
                //   v      = momentum*v - lr*grad
                //   update = momentum*v - lr*grad
                //   param += update
                for li in 0..l {
                    v_coefs[li].mapv_inplace(|v| v * cfg.momentum);
                    v_coefs[li].scaled_add(-cfg.lr, &coef_grads[li]);
                    let mut upd = &v_coefs[li] * cfg.momentum;
                    upd.scaled_add(-cfg.lr, &coef_grads[li]);
                    coefs[li] += &upd;

                    v_inter[li].mapv_inplace(|v| v * cfg.momentum);
                    v_inter[li].scaled_add(-cfg.lr, &inter_grads[li]);
                    let mut updb = &v_inter[li] * cfg.momentum;
                    updb.scaled_add(-cfg.lr, &inter_grads[li]);
                    intercepts[li] += &updb;
                }
            }

            // Training-loss based early stopping (mirrors sklearn's no-improvement
            // logic with constant learning rate and no validation split).
            let loss = accumulated / n as f64;
            if loss > best_loss - cfg.tol {
                no_improve += 1;
            } else {
                no_improve = 0;
            }
            if loss < best_loss {
                best_loss = loss;
            }
            if no_improve > cfg.n_iter_no_change {
                break;
            }
        }

        Self { scaler, coefs, intercepts }
    }

    fn predict(&self, x: &[f32], n_features: usize) -> Vec<f32> {
        let d = n_features;
        let n = x.len() / d;
        let l = self.coefs.len();
        // Process in row blocks so the full f64 design matrix and activations are
        // never materialized at once (qual has ~2.8M rows).
        const CHUNK: usize = 100_000;
        let mut out = Vec::with_capacity(n);
        let mut start = 0;
        while start < n {
            let m = (n - start).min(CHUNK);
            let xb = Array2::from_shape_fn((m, d), |(i, j)| x[(start + i) * d + j] as f64);
            let xs = self.scaler.transform(&xb);
            let acts = forward(&self.coefs, &self.intercepts, xs);
            out.extend(acts[l].column(0).iter().map(|&v| v as f32));
            start += m;
        }
        out
    }
}
