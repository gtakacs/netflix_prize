// TimeSVD++ trained with SGD — baseline of the tsvd* family in this directory.
// Predicts mu + b_u(t) + b_i(t) + (p_u + s_u) . q_i, where s_u is the NSVD1
// implicit-feedback term over the user's rated items.
// Produced: tsvd-256, tsvd-2048.  Frozen archive — see README.md, does not build.
use gravity::{Dataset, Regressor, calc_gbias, rand_array2};
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2};
use rand::{SeedableRng, rngs::StdRng};

#[derive(Clone, Copy, Debug)]
struct TimeSvdConfig {
    n_feat: usize,   // Number of latent factors
    n_epochs: usize, // Training epochs
    seed: u64,       // Random number generator seed

    lr_u: f32,       // User learning rate
    lr_ub: f32,      // User bias learning rate
    lr_i: f32,       // Item learning rate
    lr_ib: f32,      // Item bias learning rate
    lr_y: f32,       // NSVD1 learning rate

    reg_u: f32,      // User regularization
    reg_i: f32,      // Item regularization
    reg_y: f32,      // NSVD1 regularization
    sigma_u: f32,    // Random init stddev (user)
    sigma_i: f32,    // Random init stddev (item)
    sigma_y: f32,    // Random init stddev (NSVD1)

    n_bins: usize,   // Number of time bins
    beta: f32,       // dev(u,t) exponent
    lr_t: f32,       // Time bias (alpha) learning rate

    reset_u_epoch: usize,
}

struct TimeSvdModel {
    cfg: TimeSvdConfig, // Model hyperparameters

    gbias: f32,         // Global bias
    ubias: Array1<f32>, // User biases
    ibias: Array1<f32>, // Item biases

    ufeat: Array2<f32>, // User feature vectors
    ifeat: Array2<f32>, // Item feature vectors
    yfeat: Array2<f32>, // NSVD1 feature vectors
    ycache: Array2<f32>,

    day_range: i32,
    tu_mean: Array1<f32>, // User mean dates
    alpha_u: Array1<f32>, // User drift scales
    but_bin: Array2<f32>, // User × time bin biases
    bit_bin: Array2<f32>, // Item × time bin biases
}

impl TimeSvdModel {
    #[inline]
    // Map day to [0..n_bins) using proportional scaling
    fn time_bin(&self, day: i32) -> usize {
        let num = (day as i64) * (self.cfg.n_bins as i64);
        let b = (num / self.day_range as i64) as usize;
        b.min(self.cfg.n_bins - 1)
    }

    #[inline]
    // User time deviation: dev_u(t) = sign(dt) * |dt|^beta, dt = t - t̄_u.
    fn dev(&self, u: usize, day: i32) -> f32 {
        let dt = (day as f32) - self.tu_mean[u];
        if dt == 0.0 {
            0.0
        } else {
            let s = if dt > 0.0 { 1.0 } else { -1.0 };
            s * dt.abs().powf(self.cfg.beta)
        }
    }

    fn rebuild_ycache(&mut self, tr: &Dataset, pr: &Dataset) {
        self.ycache.fill(0.0);

        let tr_iter = tr.user_idxs.iter().zip(tr.item_idxs.iter());
        let pr_iter = pr.user_idxs.iter().zip(pr.item_idxs.iter());
        for (&u, &i) in tr_iter.chain(pr_iter) {
            let mut su = self.ycache.row_mut(u as usize);
            su += &self.yfeat.row(i as usize);
        }

        for u in 0..tr.n_users {
            let mut su = self.ycache.row_mut(u);
            let cnt = tr.user_cnts[u] + pr.user_cnts[u];
            if cnt > 0 { su /= (cnt as f32).sqrt(); }
        }
    }
}

impl Regressor for TimeSvdModel {
    type Config = TimeSvdConfig;

    // Initialize model parameters
    fn new(tr: &Dataset, _pr: &Dataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        // Compute per-user mean rating date
        // and overall day_range for time binning
        let mut day_range = 0;
        let mut tu_mean = Array1::<f32>::zeros(tr.n_users);
        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            tu_mean[u] += tr.dates[idx] as f32;
            day_range = day_range.max(tr.dates[idx] as i32 + 1)
        }
        for u in 0..tr.n_users {
            let cnt = tr.user_cnts[u];
            if cnt > 0 { tu_mean[u] /= cnt as f32; }
        }

        Self {
            cfg,
            gbias: calc_gbias(tr),
            ubias: Array1::<f32>::zeros(tr.n_users),
            ibias: Array1::<f32>::zeros(tr.n_items),

            ufeat: rand_array2(tr.n_users, cfg.n_feat, &mut rng, cfg.sigma_u),
            ifeat: rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_i),
            yfeat: rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_y),
            ycache: Array2::<f32>::zeros((tr.n_users, cfg.n_feat)),

            day_range,
            tu_mean,
            alpha_u: Array1::<f32>::zeros(tr.n_users),
            but_bin: Array2::<f32>::zeros((tr.n_users, cfg.n_bins)),
            bit_bin: Array2::<f32>::zeros((tr.n_items, cfg.n_bins)),
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    // Predict rating for a given (user, item, day) triplet
    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let b = self.time_bin(day);
        let dev = self.dev(u, day);

        let bu_t = self.ubias[u] + self.but_bin[[u, b]] + self.alpha_u[u] * dev;
        let bi_t = self.ibias[i] + self.bit_bin[[i, b]];

        let pu = &self.ufeat.row(u);
        let su = &self.ycache.row(u);
        let qi = &self.ifeat.row(i);

        self.gbias + bu_t + bi_t + (pu + su).dot(qi)
    }

    // Train model & report RMSE
    fn fit_epoch(&mut self, tr: &Dataset, pr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        if epoch == cfg.reset_u_epoch { self.ufeat.fill(0.0); }

        let mut idx = 0;
        for u in (0..tr.n_users).progress() {
            let cnt = tr.user_cnts[u] as usize;
            if cnt == 0 { continue; }
            let start = idx;
            let end = idx + cnt;
            idx = end;

            // Compute ycache[u]
            let mut su = Array1::<f32>::zeros(cfg.n_feat);
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                su += &self.yfeat.row(j);
            }
            let norm = (cnt as f32).sqrt();
            su /= norm;
            self.ycache.row_mut(u).assign(&su);

            let mut sum_err_q = Array1::<f32>::zeros(cfg.n_feat);
            for t in start..end {
                let i = tr.item_idxs[t] as usize; // Item index
                let r = tr.residuals[t] as f32;   // Rating value
                let day = tr.dates[t] as i32;     // Rating date
                let b = self.time_bin(day);
                let dev = self.dev(u, day);
                let err = self.predict(u, i, day) - r;

                // Base bias updates
                self.ubias[u] -= cfg.lr_ub * err;
                self.ibias[i] -= cfg.lr_ib * err;

                // Time bias/drift updates
                self.but_bin[[u, b]] -= cfg.lr_t * err;
                self.bit_bin[[i, b]] -= cfg.lr_t * err;
                self.alpha_u[u] -= cfg.lr_t * err * dev;

                // Factor updates
                for k in 0..cfg.n_feat {
                    let qi = self.ifeat[[i, k]];
                    let pu = self.ufeat[[u, k]];

                    sum_err_q[k] += err * qi;
                    self.ufeat[[u, k]] -= cfg.lr_u * (err * qi + cfg.reg_u * pu);
                    self.ifeat[[i, k]] -= cfg.lr_i * (err * (pu + su[k]) + cfg.reg_i * qi);
                }
            }

            // NSVD1 factor updates
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                for k in 0..cfg.n_feat {
                    let yj = self.yfeat[[j, k]];
                    self.yfeat[[j, k]] -= cfg.lr_y * (sum_err_q[k] / norm + cfg.reg_y * yj);
                }
            }
        }
        self.rebuild_ycache(tr, pr);
    }
}

fn main() {
    let cfg = TimeSvdConfig {
        n_feat: 16,
        n_epochs: 6,
        seed: 42,
        lr_u: 0.06,
        lr_ub: 0.0,
        lr_i: 0.007,
        lr_ib: 0.0,
        lr_y: 0.0004,
        reg_u: 0.04,
        reg_i: 0.007,
        reg_y: 0.04,
        sigma_u: 0.004,
        sigma_i: 0.005,
        sigma_y: 0.005,
        n_bins: 30,
        beta: 0.3,
        lr_t: 5e-5,
        reset_u_epoch: 1024,
    };

    // Snapshot of one run — n_feat/model_name were edited by hand per experiment.
    gravity::fit2::<TimeSvdModel>(
        cfg,
        "rtg",   // target
        "tmp",   // model_name
        false,   // save_subscores
        false,   // save_train
        false,   // save_probe_each_epoch
    );
}
