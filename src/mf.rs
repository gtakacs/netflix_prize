// Plain matrix factorization model: predicts rating as
//     gbias + ubias[u] + ibias[i] + ufeat[u, :] · ifeat[i, :]
// Optionally:
//   - load `ifeat` from an external `.npy` file (e.g. an item-embedding source),
//   - replace the MSE loss with an ordinal-regression head.
use crate::{
    Dataset, MaskedDataset, OrdinalHeadConfig, OrdinalHead, Regressor,
    calc_gbias, calc_user_offsets, get_users, rand_array2,
};
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2, s};
use ndarray_npy::{read_npy, write_npy};
use rand::{SeedableRng, rngs::StdRng};

#[derive(Clone, Copy, Debug)]
pub struct MfConfig {
    pub n_feat: usize,        // latent factor dimension
    pub n_epochs: usize,      // SGD passes over all users
    pub seed: u64,            // RNG seed (init + per-epoch shuffle)
    pub shuffle_users: bool,  // randomize user order each epoch

    // Learning rates
    pub lr_u: f32,            // for: ufeat
    pub lr_i: f32,            // for: ifeat (set to 0.0 to keep items fixed)
    pub lr_ub: f32,           // for: ubias
    pub lr_ib: f32,           // for: ibias

    // Regularizations (L2)
    pub reg_u: f32,           // for: ufeat
    pub reg_i: f32,           // for: ifeat

    // Init stddevs (Normal(0, σ))
    pub sigma_u: f32,         // for: ufeat
    pub sigma_i: f32,         // for: ifeat (ignored if `item_feat_npy` is Some)

    pub reset_u_epoch: usize, // epoch at which ufeat is zeroed (0 = never)

    /// Optional item-factor source: load `ifeat` from `<path>.<train>.npy`
    /// (the `{train}` placeholder is replaced with `tr.name`).
    pub item_feat_npy: Option<&'static str>,

    pub ordinal_head: Option<OrdinalHeadConfig>, // optional ordinal head (None = MSE loss)
    pub save_ifeat: bool,     // save ifeat as `.ifeat.<ds>.npy` artifact
}

pub struct MfModel {
    cfg: MfConfig,                     // training config
    gbias: f32,                        // global mean residual (μ)
    ubias: Array1<f32>,                // per-user bias [n_users]
    ibias: Array1<f32>,                // per-item bias [n_items]
    ufeat: Array2<f32>,                // user factors pu [n_users, n_feat]
    ifeat: Array2<f32>,                // item factors qi [n_items, n_feat]
    ordinal_head: Option<OrdinalHead>, // optional ordinal regression head
}

impl Regressor for MfModel {
    type Config = MfConfig;

    fn new(tr: &Dataset, _pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        let ufeat = rand_array2(tr.n_users, cfg.n_feat, &mut rng, cfg.sigma_u);

        // Item factors: load from external file (and pick the last n_feat columns)
        // or initialize randomly.
        let ifeat = if let Some(npy_path) = cfg.item_feat_npy {
            let npy = npy_path.replace("{train}", &tr.name);
            let all_ifeat: Array2<f32> = read_npy(&npy).expect(&npy);
            assert_eq!(all_ifeat.nrows(), tr.n_items,
                "Item feature rows mismatch: expected {}, got {}", tr.n_items, all_ifeat.nrows());
            assert!(cfg.n_feat <= all_ifeat.ncols(),
                "n_feat ({}) exceeds available columns ({})", cfg.n_feat, all_ifeat.ncols());
            let start = all_ifeat.ncols() - cfg.n_feat;
            all_ifeat.slice(s![.., start..]).to_owned()
        } else {
            rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_i)
        };

        Self {
            cfg,
            gbias: calc_gbias(tr),
            ubias: Array1::<f32>::zeros(tr.n_users),
            ibias: Array1::<f32>::zeros(tr.n_items),
            ufeat,
            ifeat,
            ordinal_head: cfg.ordinal_head.map(OrdinalHead::new),
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
        let pu = self.ufeat.row(u);
        let qi = self.ifeat.row(i);
        let score = self.gbias + self.ubias[u] + self.ibias[i] + pu.dot(&qi);

        match &self.ordinal_head {
            Some(ordinal) => {
                let probs = ordinal.predict_probs(score);
                1.0 * probs[0] + 2.0 * probs[1] + 3.0 * probs[2] + 4.0 * probs[3] + 5.0 * probs[4]
            }
            None => score,
        }
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        let cfg = self.cfg;
        if epoch == cfg.reset_u_epoch { self.ufeat.fill(0.0); }

        let user_offsets = calc_user_offsets(tr);
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        for &u in progress!(users.iter()) {
            let start = user_offsets[u];
            let end = user_offsets[u + 1];

            for t in start..end {
                let u = tr.user_idxs[t] as usize;
                let i = tr.item_idxs[t] as usize;

                // Error / gradient w.r.t. score (depends on loss)
                let err = if let Some(ordinal_head) = &mut self.ordinal_head {
                    // Ordinal regression: gradient of negative log-likelihood
                    let y = tr.raw_ratings[t] as usize;
                    let pu = self.ufeat.row(u);
                    let qi = self.ifeat.row(i);
                    let score = self.gbias + self.ubias[u] + self.ibias[i] + pu.dot(&qi);
                    let (g_ds, g_t) = ordinal_head.grad(score, y);
                    ordinal_head.update_thresholds(g_t);
                    g_ds
                } else {
                    // MSE loss: prediction error
                    self.predict(u, i, 0) - tr.residuals[t]
                };

                self.ubias[u] -= cfg.lr_ub * err;
                self.ibias[i] -= cfg.lr_ib * err;

                for k in 0..cfg.n_feat {
                    let pk = self.ufeat[[u, k]];
                    let qk = self.ifeat[[i, k]];
                    self.ufeat[[u, k]] -= cfg.lr_u * (err * qk + cfg.reg_u * pk);
                    self.ifeat[[i, k]] -= cfg.lr_i * (err * pk + cfg.reg_i * qk);
                }
            }
        }

        // Maintain threshold ordering for the ordinal head
        if let Some(ordinal_head) = &mut self.ordinal_head {
            ordinal_head.enforce_sorted_with_gap();
        }
    }
}
