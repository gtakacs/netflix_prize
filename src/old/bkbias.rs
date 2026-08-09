// BellKor baseline predictor — equation (10) from "The BellKor Solution to
// the Netflix Grand Prize" (Koren, 2009).
//
// b_ui = μ + b_u + α_u·dev_u(t) + b_{u,t} + (b_i + b_{i,Bin(t)})·(c_u + c_{u,t})
//
// Trained via SGD minimizing equation (12) with per-parameter learning rates
// and L2 regularization.  Target RMSE ≈ 0.9555 on Probe.
//
// Produced: bkbias.  Frozen archive — see README.md; the bias block lives on in
// src/bk1.rs / src/bk3.rs.

use indicatif::ProgressIterator;
use gravity::{calc_gbias, calc_user_offsets, get_users, Dataset, Regressor};
use ndarray::Array1;

// ---------------------------------------------------------------------------
// Sparse (user, day) index
// ---------------------------------------------------------------------------
struct SparseUD {
    starts: Vec<usize>,
    dates: Vec<i16>,
}

impl SparseUD {
    fn new(ds1: &Dataset, ds2: &Dataset) -> Self {
        let n_users = ds1.n_users;
        let mut per_user: Vec<Vec<i16>> = vec![Vec::new(); n_users];
        for t in 0..ds1.n_ratings {
            per_user[ds1.user_idxs[t] as usize].push(ds1.dates[t]);
        }
        for t in 0..ds2.n_ratings {
            per_user[ds2.user_idxs[t] as usize].push(ds2.dates[t]);
        }

        let mut starts = Vec::with_capacity(n_users + 1);
        let mut dates = Vec::new();
        for u in 0..n_users {
            starts.push(dates.len());
            per_user[u].sort_unstable();
            per_user[u].dedup();
            dates.extend_from_slice(&per_user[u]);
        }
        starts.push(dates.len());

        Self { starts, dates }
    }

    #[inline]
    fn n_total(&self) -> usize { self.dates.len() }

    #[inline]
    fn index(&self, u: usize, day: i16) -> Option<usize> {
        let start = self.starts[u];
        let end = self.starts[u + 1];
        self.dates[start..end]
            .binary_search(&day)
            .ok()
            .map(|i| start + i)
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct BkBiasConfig {
    n_epochs: usize,
    seed: u64,
    shuffle_users: bool,

    n_time_bins: usize,
    beta: f32,

    // Learning rates (per parameter group)
    lr_bu: f32,     // b_u
    lr_but: f32,    // b_{u,t}
    lr_alpha: f32,  // α_u
    lr_bi: f32,     // b_i
    lr_bit: f32,    // b_{i,Bin(t)}
    lr_cu: f32,     // c_u
    lr_cut: f32,    // c_{u,t}

    // Regularization (per parameter group)
    reg_bu: f32,
    reg_but: f32,
    reg_alpha: f32,
    reg_bi: f32,
    reg_bit: f32,
    reg_cu: f32,    // penalizes (c_u - 1)^2
    reg_cut: f32,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct BkBiasModel {
    cfg: BkBiasConfig,

    gbias: f32,

    // User parameters
    bu: Array1<f32>,        // [n_users] — user bias
    alpha_u: Array1<f32>,   // [n_users] — user time drift slope
    cu: Array1<f32>,        // [n_users] — user scaling factor (init 1.0)

    // Item parameters
    bi: Array1<f32>,        // [n_items] — item bias
    bit_bin: Vec<Vec<f32>>, // [n_items][n_time_bins] — item bias per time bin

    // Day-specific parameters (sparse, indexed by SparseUD)
    but: Vec<f32>,          // [n_ud] — user bias per day
    cut: Vec<f32>,          // [n_ud] — user scaling per day

    // Precomputed
    tu_mean: Array1<f32>,   // [n_users] — mean rating date per user
    day_range: i32,         // max date + 1
    ud: SparseUD,
    user_offsets: Array1<usize>,
}

impl BkBiasModel {
    #[inline]
    fn time_bin(&self, day: i32) -> usize {
        let num = (day as i64) * (self.cfg.n_time_bins as i64);
        let b = (num / self.day_range as i64) as usize;
        b.min(self.cfg.n_time_bins - 1)
    }

    #[inline]
    fn dev(&self, u: usize, day: i32) -> f32 {
        let dt = (day as f32) - self.tu_mean[u];
        if dt == 0.0 {
            0.0
        } else {
            let s = if dt > 0.0 { 1.0 } else { -1.0 };
            s * dt.abs().powf(self.cfg.beta)
        }
    }
}

impl Regressor for BkBiasModel {
    type Config = BkBiasConfig;

    fn new(tr: &Dataset, pr: &Dataset, cfg: Self::Config) -> Self {
        // Compute mean rating date per user (from training set only)
        let mut tu_mean = Array1::<f32>::zeros(tr.n_users);
        let mut day_range: i32 = 0;
        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            tu_mean[u] += tr.dates[idx] as f32;
            day_range = day_range.max(tr.dates[idx] as i32 + 1);
        }
        for u in 0..tr.n_users {
            let cnt = tr.user_cnts[u];
            if cnt > 0 { tu_mean[u] /= cnt as f32; }
        }

        let ud = SparseUD::new(tr, pr);
        let n_ud = ud.n_total();

        Self {
            cfg,
            gbias: calc_gbias(tr),
            bu: Array1::zeros(tr.n_users),
            alpha_u: Array1::zeros(tr.n_users),
            cu: Array1::ones(tr.n_users),  // c_u initialized to 1
            bi: Array1::zeros(tr.n_items),
            bit_bin: vec![vec![0.0; cfg.n_time_bins]; tr.n_items],
            but: vec![0.0; n_ud],
            cut: vec![0.0; n_ud],
            tu_mean,
            day_range,
            ud,
            user_offsets: calc_user_offsets(tr),
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let b = self.time_bin(day);
        let dev = self.dev(u, day);
        let day16 = day as i16;

        let but_val = self.ud.index(u, day16).map_or(0.0, |idx| self.but[idx]);
        let cut_val = self.ud.index(u, day16).map_or(0.0, |idx| self.cut[idx]);

        let bu_t = self.bu[u] + self.alpha_u[u] * dev + but_val;
        let bi_t = self.bi[i] + self.bit_bin[i][b];
        let cu_t = self.cu[u] + cut_val;

        self.gbias + bu_t + bi_t * cu_t
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        for &u in users.iter().progress() {
            let start = self.user_offsets[u];
            let end = self.user_offsets[u + 1];
            if start == end { continue; }

            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                let r = tr.residuals[t];
                let day = tr.dates[t] as i32;
                let day16 = tr.dates[t];
                let b = self.time_bin(day);
                let dev = self.dev(u, day);

                let ud_idx = self.ud.index(u, day16);
                let but_val = ud_idx.map_or(0.0, |idx| self.but[idx]);
                let cut_val = ud_idx.map_or(0.0, |idx| self.cut[idx]);

                let bu_t = self.bu[u] + self.alpha_u[u] * dev + but_val;
                let bi_t = self.bi[i] + self.bit_bin[i][b];
                let cu_t = self.cu[u] + cut_val;

                let pred = self.gbias + bu_t + bi_t * cu_t;
                let err = pred - r;

                // SGD updates — gradient of (12) w.r.t. each parameter
                // d(err^2)/d(param) = 2*err * d(pred)/d(param), absorbed into lr

                // b_u: d(pred)/d(b_u) = 1
                self.bu[u] -= cfg.lr_bu * (err + cfg.reg_bu * self.bu[u]);

                // α_u: d(pred)/d(α_u) = dev
                self.alpha_u[u] -= cfg.lr_alpha * (err * dev + cfg.reg_alpha * self.alpha_u[u]);

                // b_{u,t}: d(pred)/d(b_{u,t}) = 1
                if let Some(idx) = ud_idx {
                    self.but[idx] -= cfg.lr_but * (err + cfg.reg_but * self.but[idx]);
                }

                // b_i: d(pred)/d(b_i) = c_u(t)
                self.bi[i] -= cfg.lr_bi * (err * cu_t + cfg.reg_bi * self.bi[i]);

                // b_{i,Bin(t)}: d(pred)/d(b_{i,Bin(t)}) = c_u(t)
                self.bit_bin[i][b] -= cfg.lr_bit * (err * cu_t + cfg.reg_bit * self.bit_bin[i][b]);

                // c_u: d(pred)/d(c_u) = b_i(t), reg penalizes (c_u - 1)^2
                self.cu[u] -= cfg.lr_cu * (err * bi_t + cfg.reg_cu * (self.cu[u] - 1.0));

                // c_{u,t}: d(pred)/d(c_{u,t}) = b_i(t)
                if let Some(idx) = ud_idx {
                    self.cut[idx] -= cfg.lr_cut * (err * bi_t + cfg.reg_cut * self.cut[idx]);
                }
            }
        }
    }
}

fn main() {
    // Hyperparameters from Table in Section III-D of the BellKor paper
    // lrate ×10^3:  b_u=3, b_ut=2.5, α_u=0.01, b_i=2, b_{i,Bin}=0.05, c_u=8, c_ut=2
    // reg ×10^2:    b_u=3, b_ut=0.5, α_u=5000, b_i=3, b_{i,Bin}=10,   c_u=1, c_ut=0.5
    let cfg = BkBiasConfig {
        n_epochs: 30,
        seed: 42,
        shuffle_users: true,

        n_time_bins: 30,
        beta: 0.4,

        lr_bu:    3e-3,
        lr_but:   2.5e-3,
        lr_alpha: 1e-5,
        lr_bi:    2e-3,
        lr_bit:   5e-5,
        lr_cu:    8e-3,
        lr_cut:   2e-3,

        reg_bu:    3e-2,
        reg_but:   5e-3,
        reg_alpha: 50.0,
        reg_bi:    3e-2,
        reg_bit:   0.1,
        reg_cu:    1e-2,
        reg_cut:   5e-3,
    };

    gravity::fit2::<BkBiasModel>(
        cfg,
        "rtg",      // target
        "bkbias",   // model_name
        false,      // save_subscores
        true,       // save_train
        false,      // save_probe_each_epoch
    );
}
