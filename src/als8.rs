// Alternating Least Squares with 8 generalized bias features and optional
// latent MF factors. Per-user / per-item parameter blocks (bias_feats + MF
// factors) are solved jointly via normal equations + Cholesky.
//
// Idea adapted from Pragmatic Theory's Grand Prize report.

use crate::{Dataset, MaskedDataset, Regressor, calc_gbias, rand_array2};
use indicatif::ParallelProgressIterator;
use ndarray::Array1;
use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::collections::HashMap;

// Number of generalized bias features (constant for this model)
const N_BIAS_FEAT: usize = 8;

// Index types for efficient ALS updates
type UIndex = Vec<Vec<(usize, i16, f32)>>; // user -> [(item, day, residual)]
type IIndex = Vec<Vec<(usize, i16, f32)>>; // item -> [(user, day, residual)]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Compute cube root for non-negative integers (clamped at 0)
#[inline]
fn cbrt_pos_i32(x: i32) -> f32 {
    if x <= 0 { 0.0 } else { (x as f32).powf(1.0 / 3.0) }
}

// Compute shrunk mean with regularization: (sum + m*gbias) / (cnt + m)
#[inline]
fn shrunk_mean(sum: f64, cnt: i32, gbias: f32, m: f32) -> f32 {
    ((sum as f32) + m * gbias) / ((cnt as f32) + m)
}

// Solve linear system A*x = b using Cholesky decomposition
// a_flat: row-major flattened n×n matrix, b: right-hand side vector of size n
fn solve_linear_flat(a_flat: &[f64], b: &[f64], n: usize) -> Vec<f32> {
    use nalgebra::{DMatrix, DVector};
    let a = DMatrix::<f64>::from_row_slice(n, n, a_flat);
    let b = DVector::<f64>::from_row_slice(b);
    let x = a.cholesky().unwrap().solve(&b);
    x.iter().copied().map(|x| x as f32).collect()
}

// Update temporal caches from a dataset's (user_idxs, item_idxs, dates) slices.
// Used for both train and probe — only structural (user/item/day) info is read,
// no ratings. Slice signature lets us call this with both `Dataset` and
// `MaskedDataset` (the latter exposes the same arrays).
fn update_union_caches(
    n_ratings: usize,
    user_idxs: &Array1<i32>,
    item_idxs: &Array1<i32>,
    dates: &Array1<i16>,
    user_first_day: &mut [i16],
    item_first_day: &mut [i16],
    user_day_cnt: &mut [HashMap<i16, i32>],
    item_day_cnt: &mut [HashMap<i16, i32>],
    user_cnt_total: &mut [i32],
    item_cnt_total: &mut [i32],
) {
    for idx in 0..n_ratings {
        let u = user_idxs[idx] as usize;
        let i = item_idxs[idx] as usize;
        let d = dates[idx];

        if d < user_first_day[u] {
            user_first_day[u] = d;
        }
        if d < item_first_day[i] {
            item_first_day[i] = d;
        }

        *user_day_cnt[u].entry(d).or_insert(0) += 1;
        *item_day_cnt[i].entry(d).or_insert(0) += 1;

        user_cnt_total[u] += 1;
        item_cnt_total[i] += 1;
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Als8Config {
    pub n_epochs: usize,     // Number of training epochs
    pub seed: u64,           // Random number generator seed

    pub n_feat: usize,       // Latent dimension

    // Random init standard deviations
    pub sigma_u: f32,        // User factor
    pub sigma_i: f32,        // Item factor

    // Regularizations
    pub reg_ub: f32,         // User bias
    pub reg_ib: f32,         // Item bias
    pub reg_u: f32,          // User factor
    pub reg_i: f32,          // Item factor
    pub shrink_m: f32,       // Shrinkage strength for mean feature

    pub use_probe: bool,     // Use probe in temporal caches (not residuals)
    pub n_bias_used: usize,  // Number of bias features to actually use (<= N_BIAS_FEAT)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Als8Model {
    cfg: Als8Config,

    gbias: f32,         // Global bias

    // Generalized bias features: (n_users x N_BIAS_FEAT), (n_items x N_BIAS_FEAT)
    ubias: Array2<f32>, // User biases
    ibias: Array2<f32>, // Item biases

    ufeat: Array2<f32>, // User factors
    ifeat: Array2<f32>, // Item factors

    // Stats for bias features (computed from Train u Probe)
    user_first_day: Vec<i16>,             // First rating day per user
    item_first_day: Vec<i16>,             // First rating day per item
    user_day_cnt: Vec<HashMap<i16, i32>>, // Count of ratings per user per day
    item_day_cnt: Vec<HashMap<i16, i32>>, // Count of ratings per item per day
    user_cnt_total: Vec<i32>,             // Total ratings per user
    item_cnt_total: Vec<i32>,             // Total ratings per item

    // Mean features (computed from Train only)
    user_mean_shrunk: Vec<f32>,
    item_mean_shrunk: Vec<f32>,

    // Indices for efficient ALS updates
    by_user: UIndex, // Ratings indexed by user
    by_item: IIndex, // Ratings indexed by item
}

impl Als8Model {
    // Compute bias features for user parameter update
    #[inline]
    fn bias_feats_user(&self, u: usize, i: usize, day: i32) -> [f32; N_BIAS_FEAT] {
        let d = day as i16;

        let du = (d as i32) - (self.user_first_day[u] as i32);
        let di = (d as i32) - (self.item_first_day[i] as i32);

        let cud = *self.user_day_cnt[u].get(&d).unwrap_or(&0);
        let cid = *self.item_day_cnt[i].get(&d).unwrap_or(&0);

        let mut f = [
            1.0,                                         // Bias term (always 1.0)
            cbrt_pos_i32(du),                            // Cube root of user time-since-first
            cbrt_pos_i32(di),                            // Cube root of item time-since-first
            cbrt_pos_i32(cud),                           // Cube root of user ratings on day
            cbrt_pos_i32(cid),                           // Cube root of item ratings on day
            self.item_mean_shrunk[i],                    // Item shrunk mean
            cbrt_pos_i32(self.user_cnt_total[u] as i32), // Cube root of total user ratings
            cbrt_pos_i32(self.item_cnt_total[i] as i32), // Cube root of total item ratings
        ];
        for k in self.cfg.n_bias_used..N_BIAS_FEAT { f[k] = 0.0; }
        f
    }

    // Compute bias features for item parameter update
    #[inline]
    fn bias_feats_item(&self, u: usize, i: usize, day: i32) -> [f32; N_BIAS_FEAT] {
        let d = day as i16;

        let du = (d as i32) - (self.user_first_day[u] as i32);
        let di = (d as i32) - (self.item_first_day[i] as i32);

        let cud = *self.user_day_cnt[u].get(&d).unwrap_or(&0);
        let cid = *self.item_day_cnt[i].get(&d).unwrap_or(&0);

        let mut f = [
            1.0,                                         // Bias term (always 1.0)
            cbrt_pos_i32(du),                            // Cube root of user time-since-first
            cbrt_pos_i32(di),                            // Cube root of item time-since-first
            cbrt_pos_i32(cud),                           // Cube root of user ratings on day
            cbrt_pos_i32(cid),                           // Cube root of item ratings on day
            self.user_mean_shrunk[u],                    // User shrunk mean
            cbrt_pos_i32(self.user_cnt_total[u] as i32), // Cube root of total user ratings
            cbrt_pos_i32(self.item_cnt_total[i] as i32), // Cube root of total item ratings
        ];
        for k in self.cfg.n_bias_used..N_BIAS_FEAT { f[k] = 0.0; }
        f
    }

    // Dot product of bias weight row with feature vector
    #[inline]
    fn dot_bias_row(mat: &Array2<f32>, row: usize, x: &[f32; N_BIAS_FEAT]) -> f32 {
        let mut s = 0.0;
        for f in 0..N_BIAS_FEAT {
            s += mat[[row, f]] * x[f];
        }
        s
    }

    // Dot product of first k columns of two matrix rows
    #[inline]
    fn dot_feat_row(a: &Array2<f32>, ra: usize, b: &Array2<f32>, rb: usize, k: usize) -> f32 {
        let mut s = 0.0;
        for f in 0..k {
            s += a[[ra, f]] * b[[rb, f]];
        }
        s
    }

    // Build user-indexed and item-indexed rating lists from training data
    fn build_indices(tr: &Dataset) -> (UIndex, IIndex) {
        let mut by_user: UIndex = vec![Vec::new(); tr.n_users];
        let mut by_item: IIndex = vec![Vec::new(); tr.n_items];

        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;
            let d = tr.dates[idx];
            let r = tr.residuals[idx];
            by_user[u].push((i, d, r));
            by_item[i].push((u, d, r));
        }
        (by_user, by_item)
    }

    // ALS step: update user bias features and latent factors
    fn als_user_step(&mut self) {
        let cfg = self.cfg;
        let n_users = self.by_user.len();
        let k = cfg.n_feat;
        let n = N_BIAS_FEAT + k;

        let ibias = &self.ibias;
        let ifeat = &self.ifeat;

        let xs: Vec<(usize, Vec<f32>)> = crate::progress!((0..n_users).into_par_iter())
            .map(|u| {
                let obs = &self.by_user[u];
                if obs.is_empty() {
                    return (u, vec![0.0; n]);
                }

                let mut xtx = vec![0.0f64; n * n];
                let mut xty = vec![0.0f64; n];
                let mut z = vec![0.0f64; n];

                for &(i, d, r) in obs.iter() {
                    let day = d as i32;

                    let xu = self.bias_feats_user(u, i, day);
                    let xi = self.bias_feats_item(u, i, day);

                    let y = (r - self.gbias - Self::dot_bias_row(ibias, i, &xi)) as f64;

                    for f in 0..N_BIAS_FEAT {
                        z[f] = xu[f] as f64;
                    }
                    for f in 0..k {
                        z[N_BIAS_FEAT + f] = ifeat[[i, f]] as f64;
                    }

                    for row in 0..n {
                        let zr = z[row];
                        xty[row] += zr * y;
                        let base = row * n;
                        for col in 0..n {
                            xtx[base + col] += zr * z[col];
                        }
                    }
                }

                let lam_b = cfg.reg_ub.max(0.0) as f64;
                for f in 0..N_BIAS_FEAT {
                    xtx[f * n + f] += lam_b;
                }
                let lam_f = cfg.reg_u.max(0.0) as f64;
                for f in 0..k {
                    let idx = N_BIAS_FEAT + f;
                    xtx[idx * n + idx] += lam_f;
                }

                let w = solve_linear_flat(&xtx, &xty, n);
                (u, w)
            })
            .collect();

        for (u, w) in xs {
            for f in 0..N_BIAS_FEAT {
                self.ubias[[u, f]] = w[f];
            }
            for f in 0..k {
                self.ufeat[[u, f]] = w[N_BIAS_FEAT + f];
            }
        }
    }

    // ALS step: update item bias features and latent factors
    fn als_item_step(&mut self) {
        let cfg = self.cfg;
        let n_items = self.by_item.len();
        let k = cfg.n_feat;
        let n = N_BIAS_FEAT + k;

        let ubias = &self.ubias;
        let ufeat = &self.ufeat;

        let xs: Vec<(usize, Vec<f32>)> = crate::progress!((0..n_items).into_par_iter())
            .map(|i| {
                let obs = &self.by_item[i];
                if obs.is_empty() {
                    return (i, vec![0.0; n]);
                }

                let mut xtx = vec![0.0f64; n * n];
                let mut xty = vec![0.0f64; n];
                let mut z = vec![0.0f64; n];

                for &(u, d, r) in obs.iter() {
                    let day = d as i32;

                    let xu = self.bias_feats_user(u, i, day);
                    let xi = self.bias_feats_item(u, i, day);

                    let y = (r - self.gbias - Self::dot_bias_row(ubias, u, &xu)) as f64;

                    for f in 0..N_BIAS_FEAT {
                        z[f] = xi[f] as f64;
                    }
                    for f in 0..k {
                        z[N_BIAS_FEAT + f] = ufeat[[u, f]] as f64;
                    }

                    for row in 0..n {
                        let zr = z[row];
                        xty[row] += zr * y;
                        let base = row * n;
                        for col in 0..n {
                            xtx[base + col] += zr * z[col];
                        }
                    }
                }

                let lam_b = cfg.reg_ib.max(0.0) as f64;
                for f in 0..N_BIAS_FEAT {
                    xtx[f * n + f] += lam_b;
                }
                let lam_f = cfg.reg_i.max(0.0) as f64;
                for f in 0..k {
                    let idx = N_BIAS_FEAT + f;
                    xtx[idx * n + idx] += lam_f;
                }

                let w = solve_linear_flat(&xtx, &xty, n);
                (i, w)
            })
            .collect();

        for (i, w) in xs {
            for f in 0..N_BIAS_FEAT {
                self.ibias[[i, f]] = w[f];
            }
            for f in 0..k {
                self.ifeat[[i, f]] = w[N_BIAS_FEAT + f];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Regressor impl
// ---------------------------------------------------------------------------

impl Regressor for Als8Model {
    type Config = Als8Config;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let gbias = calc_gbias(tr);

        let (by_user, by_item) = Self::build_indices(tr);

        let mut user_first_day = vec![i16::MAX; tr.n_users];
        let mut item_first_day = vec![i16::MAX; tr.n_items];

        let mut user_day_cnt: Vec<HashMap<i16, i32>> =
            (0..tr.n_users).map(|_| HashMap::new()).collect();
        let mut item_day_cnt: Vec<HashMap<i16, i32>> =
            (0..tr.n_items).map(|_| HashMap::new()).collect();

        let mut user_cnt_total = vec![0i32; tr.n_users];
        let mut item_cnt_total = vec![0i32; tr.n_items];

        update_union_caches(
            tr.n_ratings,
            &tr.user_idxs,
            &tr.item_idxs,
            &tr.dates,
            &mut user_first_day,
            &mut item_first_day,
            &mut user_day_cnt,
            &mut item_day_cnt,
            &mut user_cnt_total,
            &mut item_cnt_total,
        );
        if cfg.use_probe {
            update_union_caches(
                pr.n_ratings,
                pr.user_idxs,
                pr.item_idxs,
                pr.dates,
                &mut user_first_day,
                &mut item_first_day,
                &mut user_day_cnt,
                &mut item_day_cnt,
                &mut user_cnt_total,
                &mut item_cnt_total,
            );
        }

        for u in 0..tr.n_users {
            if user_first_day[u] == i16::MAX {
                user_first_day[u] = 0;
            }
        }
        for i in 0..tr.n_items {
            if item_first_day[i] == i16::MAX {
                item_first_day[i] = 0;
            }
        }

        let mut user_sum_tr = vec![0.0f64; tr.n_users];
        let mut user_cnt_tr = vec![0i32; tr.n_users];
        let mut item_sum_tr = vec![0.0f64; tr.n_items];
        let mut item_cnt_tr = vec![0i32; tr.n_items];

        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;
            let r = tr.residuals[idx] as f64;
            user_sum_tr[u] += r;
            user_cnt_tr[u] += 1;
            item_sum_tr[i] += r;
            item_cnt_tr[i] += 1;
        }

        let mut user_mean_shrunk = vec![gbias; tr.n_users];
        let mut item_mean_shrunk = vec![gbias; tr.n_items];
        for u in 0..tr.n_users {
            user_mean_shrunk[u] = shrunk_mean(user_sum_tr[u], user_cnt_tr[u], gbias, cfg.shrink_m);
        }
        for i in 0..tr.n_items {
            item_mean_shrunk[i] = shrunk_mean(item_sum_tr[i], item_cnt_tr[i], gbias, cfg.shrink_m);
        }

        // init MF factors
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let ufeat = if cfg.n_feat == 0 {
            Array2::<f32>::zeros((tr.n_users, 0))
        } else {
            rand_array2(tr.n_users, cfg.n_feat, &mut rng, cfg.sigma_u)
        };
        let ifeat = if cfg.n_feat == 0 {
            Array2::<f32>::zeros((tr.n_items, 0))
        } else {
            rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_i)
        };

        Self {
            cfg,
            gbias,
            ubias: Array2::<f32>::zeros((tr.n_users, N_BIAS_FEAT)),
            ibias: Array2::<f32>::zeros((tr.n_items, N_BIAS_FEAT)),
            ufeat,
            ifeat,
            user_first_day,
            item_first_day,
            user_day_cnt,
            item_day_cnt,
            user_cnt_total,
            item_cnt_total,
            user_mean_shrunk,
            item_mean_shrunk,
            by_user,
            by_item,
        }
    }

    fn n_epochs(&self) -> usize {
        self.cfg.n_epochs
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let x_u = self.bias_feats_user(u, i, day);
        let x_i = self.bias_feats_item(u, i, day);

        let mut s = self.gbias
            + Self::dot_bias_row(&self.ubias, u, &x_u)
            + Self::dot_bias_row(&self.ibias, i, &x_i);

        if self.cfg.n_feat > 0 {
            s += Self::dot_feat_row(&self.ufeat, u, &self.ifeat, i, self.cfg.n_feat);
        }
        s
    }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, _epoch: usize) {
        self.als_user_step();
        self.als_item_step();
    }
}
