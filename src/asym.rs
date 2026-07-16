// Asymmetric matrix factorization (NSVD1-style) with optional per-user scaling.
//
// User factors are not learned directly; they're built as the (optionally scaled)
// average of the user's rated-item NSVD1 vectors:
//     su[u] = (us[u] / √|N(u)|) · Σ_{j ∈ N(u)} yfeat[j]
// Prediction:
//     r̂(u, i) = gbias + ubias[u] + ibias[i] + su[u] · ifeat[i]
use crate::{Dataset, MaskedDataset, Regressor, calc_gbias, rand_array2, calc_user_offsets, get_users};
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use rand::{SeedableRng, rngs::StdRng};

#[derive(Clone, Copy, Debug)]
pub struct AsymConfig {
    pub n_feat: usize,        // latent factor dimension
    pub n_epochs: usize,      // SGD passes over all users
    pub seed: u64,            // RNG seed (init + per-epoch shuffle)
    pub shuffle_users: bool,  // randomize user order each epoch

    // Learning rates
    pub lr_ub: f32,           // for: ubias
    pub lr_i: f32,            // for: ifeat
    pub lr_ib: f32,           // for: ibias
    pub lr_y: f32,            // for: yfeat (NSVD1)
    pub lr_us: f32,           // for: user_scaling (set to 0 to disable)

    // Regularizations (L2)
    pub reg_i: f32,           // for: ifeat
    pub reg_y: f32,           // for: yfeat
    pub reg_us: f32,          // for: user_scaling (toward 1.0)

    // Init stddevs (Normal(0, σ))
    pub sigma_i: f32,         // for: ifeat
    pub sigma_y: f32,         // for: yfeat

    pub init_with_user_std: bool, // initialize user_scaling with per-user stddev (else 1.0)
    pub save_ifeat: bool,         // save ifeat as `.ifeat.<ds>.npy` artifact
}

pub struct AsymModel {
    cfg: AsymConfig,               // training config
    gbias: f32,                    // global mean residual (μ)
    ubias: Array1<f32>,            // per-user bias [n_users]
    ibias: Array1<f32>,            // per-item bias [n_items]
    ifeat: Array2<f32>,            // item factors qi [n_items, n_feat]
    yfeat: Array2<f32>,            // NSVD1 factors [n_items, n_feat]
    ycache: Array2<f32>,           // cached su[u] = us[u]/√|N(u)| · Σ yfeat[j] [n_users, n_feat]
    user_scaling: Array1<f32>,     // per-user scaling factor us[u] [n_users]
}

/// Global variance of residuals (used for shrinkage in `calc_user_stds`).
fn calc_gvar(tr: &Dataset, gmean: f32) -> f64 {
    let mut sum_diff2 = 0.0;
    for t in 0..tr.n_ratings {
        let diff = (tr.residuals[t] - gmean) as f64;
        sum_diff2 += diff * diff;
    }
    sum_diff2 / tr.n_ratings as f64
}

/// Per-user residual stddevs, shrunk toward the global variance with strength `alpha`.
fn calc_user_stds(tr: &Dataset, gmean: f32, alpha: f64) -> Array1<f32> {
    let gvar = calc_gvar(tr, gmean);

    let mut usum1 = vec![0_f64; tr.n_users];
    let mut usum2 = vec![0_f64; tr.n_users];
    for t in 0..tr.residuals.len() {
        let u = tr.user_idxs[t] as usize;
        let r = tr.residuals[t] as f64;
        usum1[u] += r;
        usum2[u] += r * r;
    }

    let mut user_stds = Array1::<f32>::ones(tr.n_users);
    for u in 0..tr.n_users {
        let n = tr.user_cnts[u] as f64;
        let uvar = if n == 0.0 {
            0.0
        } else {
            (usum2[u] / n - (usum1[u] / n).powi(2)).max(0.0)
        };
        // Shrink user variance toward global variance.
        let uvar_shrunk = (n * uvar + alpha * gvar) / (n + alpha);
        user_stds[u] = uvar_shrunk.sqrt() as f32;
    }
    user_stds
}

impl AsymModel {
    /// Aggregate NSVD1 factors over each user's rated items in tr ∪ pr,
    /// scaled by `user_scaling[u] / √|N(u)|`. Used for prediction.
    fn rebuild_ycache(&mut self, tr: &Dataset, pr: &MaskedDataset) {
        self.ycache.fill(0.0);

        for t in 0..tr.n_ratings {
            let u = tr.user_idxs[t] as usize;
            let i = tr.item_idxs[t] as usize;
            let mut su = self.ycache.row_mut(u);
            su += &self.yfeat.row(i);
        }
        for t in 0..pr.n_ratings {
            let u = pr.user_idxs[t] as usize;
            let i = pr.item_idxs[t] as usize;
            let mut su = self.ycache.row_mut(u);
            su += &self.yfeat.row(i);
        }

        for u in 0..tr.n_users {
            let mut su = self.ycache.row_mut(u);
            let cnt = tr.user_cnts[u] + pr.user_cnts[u];
            if cnt > 0 {
                su *= self.user_scaling[u] / (cnt as f32).sqrt();
            }
        }
    }
}

impl Regressor for AsymModel {
    type Config = AsymConfig;

    fn new(tr: &Dataset, _pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let gmean = calc_gbias(tr);
        let user_scaling = if cfg.init_with_user_std {
            calc_user_stds(tr, gmean, 40.0)
        } else {
            Array1::<f32>::ones(tr.n_users)
        };

        Self {
            cfg,
            gbias: gmean,
            ubias: Array1::<f32>::zeros(tr.n_users),
            ibias: Array1::<f32>::zeros(tr.n_items),
            ifeat: rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_i),
            yfeat: rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_y),
            ycache: Array2::<f32>::zeros((tr.n_users, cfg.n_feat)),
            user_scaling,
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn save_artifacts(&self, model_name: &str, tr_set: &str, preds_dir: &str) {
        if self.cfg.save_ifeat {
            let path = format!("{}/{}.ifeat.{}.npy", preds_dir, model_name, tr_set);
            write_npy(&path, &self.ifeat).unwrap();
            crate::teeln!("Saved item factors to {}", path);
        }
    }

    fn predict(&self, u: usize, i: usize, _day: i32) -> f32 {
        let su = &self.ycache.row(u);
        let qi = &self.ifeat.row(i);
        self.gbias + self.ubias[u] + self.ibias[i] + su.dot(qi)
    }

    fn fit_epoch(&mut self, tr: &Dataset, pr: &MaskedDataset, epoch: usize) {
        let cfg = self.cfg;

        let user_offsets = calc_user_offsets(tr);
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        for &u in progress!(users.iter()) {
            let cnt = tr.user_cnts[u] as usize;
            if cnt == 0 { continue; }
            let start = user_offsets[u];
            let end = user_offsets[u + 1];

            // User-aggregated NSVD1 vector (training set only here, for the SGD step).
            let mut sum_y = Array1::<f32>::zeros(cfg.n_feat);
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                sum_y += &self.yfeat.row(j);
            }
            let mut us = self.user_scaling[u];
            let norm = us / (cnt as f32).sqrt();
            let su = sum_y.clone() * norm;
            self.ycache.row_mut(u).assign(&su);

            let mut grad_us = 0.0;
            let mut sum_err_q = Array1::<f32>::zeros(cfg.n_feat);
            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                let r = tr.residuals[t];
                let err = self.predict(u, i, 0) - r;

                self.ubias[u] -= cfg.lr_ub * err;
                self.ibias[i] -= cfg.lr_ib * err;

                let qi = self.ifeat.row(i);
                grad_us += err * su.dot(&qi) / us;
                for k in 0..cfg.n_feat {
                    let qik = self.ifeat[[i, k]];
                    sum_err_q[k] += err * qik;
                    self.ifeat[[i, k]] -= cfg.lr_i * (err * su[k] + cfg.reg_i * qik);
                }
            }

            // User scaling: regularized toward 1.0, clamped at 0
            us -= cfg.lr_us * (grad_us + cfg.reg_us * (us - 1.0));
            self.user_scaling[u] = us.max(0.0);

            // NSVD1 factor updates
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                for k in 0..cfg.n_feat {
                    let yj = self.yfeat[[j, k]];
                    self.yfeat[[j, k]] -= cfg.lr_y * (sum_err_q[k] * norm + cfg.reg_y * yj);
                }
            }
        }

        self.rebuild_ycache(tr, pr);
    }
}
