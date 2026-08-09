// Bias model with support-bucketed interaction terms: on top of mu + b_u + b_i it
// learns b_{i,bucket(u)} and b_{u,bucket(i)}, where bucket(s) = floor(log2 s) of the
// rating count. Cheap, and it blends well.
// Produced: fbias.  Frozen archive, see README.md; does not build.
use indicatif::ProgressIterator;
use gravity::{calc_gbias, calc_user_offsets, get_users, Dataset, Regressor};
use ndarray::{Array1, Array2};

/// Map support count to bucket index via log2
#[inline]
fn support_bucket(s: i32) -> usize {
    (s.max(1) as u32).ilog2() as usize
}

#[derive(Clone, Copy, Debug)]
struct BiasConfig {
    n_epochs: usize,
    seed: u64,
    shuffle_users: bool,

    lr_ub: f32,   // User bias learning rate
    lr_ib: f32,   // Item bias learning rate
    lr_isb: f32,  // Item×support bias learning rate
    lr_usb: f32,  // User×support bias learning rate
    reg_ub: f32,  // User bias regularization
    reg_ib: f32,  // Item bias regularization
    reg_isb: f32, // Item×support bias regularization
    reg_usb: f32, // User×support bias regularization
}

struct BiasModel {
    cfg: BiasConfig,
    gbias: f32,
    ubias: Array1<f32>,
    ibias: Array1<f32>,
    usupp_bucket: Array1<usize>, // precomputed support bucket per user
    isupp_bucket: Array1<usize>, // precomputed support bucket per item
    is_bias: Array2<f32>,        // item × support_bucket bias (n_items × N_SUPPORT_BUCKETS)
    us_bias: Array2<f32>,        // user × support_bucket bias (n_users × N_SUPPORT_BUCKETS)
    user_offsets: Array1<usize>, // cached user offsets
}

impl Regressor for BiasModel {
    type Config = BiasConfig;

    fn new(tr: &Dataset, pr: &Dataset, cfg: Self::Config) -> Self {
        // Compute user support = rating count across tr + pr, then bucket
        let mut usupp_bucket = Array1::<usize>::zeros(tr.n_users);
        let mut usb_max = 1;
        for u in 0..tr.n_users {
            let usb = support_bucket(tr.user_cnts[u] + pr.user_cnts[u]);
            usupp_bucket[u] = usb;
            usb_max = usb_max.max(usb);
        }

        // Compute item support = rating count across tr + pr, then bucket
        let mut isupp_bucket = Array1::<usize>::zeros(tr.n_items);
        let mut isb_max = 1;
        for i in 0..tr.n_items {
            let isb = support_bucket(tr.item_cnts[i] + pr.item_cnts[i]);
            isupp_bucket[i] = isb;
            isb_max = isb_max.max(isb);
        }

        Self {
            cfg,
            gbias: calc_gbias(tr),
            ubias: Array1::zeros(tr.n_users),
            ibias: Array1::zeros(tr.n_items),
            usupp_bucket,
            isupp_bucket,
            is_bias: Array2::zeros((tr.n_items, usb_max + 1)),
            us_bias: Array2::zeros((tr.n_users, isb_max + 1)),
            user_offsets: calc_user_offsets(tr),
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn predict(&self, u: usize, i: usize, _day: i32) -> f32 {
        let ub = self.usupp_bucket[u];
        let ib = self.isupp_bucket[i];
        self.gbias + self.ubias[u] + self.ibias[i] + self.is_bias[[i, ub]] + self.us_bias[[u, ib]]
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        for &u in users.iter().progress() {
            let start = self.user_offsets[u];
            let end = self.user_offsets[u + 1];
            if start == end { continue; }

            let ub = self.usupp_bucket[u];

            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                let r = tr.residuals[t];
                let ib = self.isupp_bucket[i];

                let err = self.predict(u, i, 0) - r;

                self.ubias[u] -= cfg.lr_ub * (err + cfg.reg_ub * self.ubias[u]);
                self.ibias[i] -= cfg.lr_ib * (err + cfg.reg_ib * self.ibias[i]);
                self.is_bias[[i, ub]] -= cfg.lr_isb * (err + cfg.reg_isb * self.is_bias[[i, ub]]);
                self.us_bias[[u, ib]] -= cfg.lr_usb * (err + cfg.reg_usb * self.us_bias[[u, ib]]);
            }
        }
    }
}

fn main() {
    let cfg = BiasConfig {
        n_epochs: 18,
        seed: 42,
        shuffle_users: true,
        lr_ub: 0.02,
        lr_ib: 0.003,
        lr_isb: 0.001,
        lr_usb: 0.001,
        reg_ub: 0.15,
        reg_ib: 0.015,
        reg_isb: 0.03,
        reg_usb: 0.03,
    };

    gravity::fit2::<BiasModel>(
        cfg,
        "rtg",     // target
        "fbias",   // model_name
        false,     // save_subscores
        true,      // save_train
        false,     // save_probe_each_epoch
    );
}
