// tsvdx.rs + (item, time-bin) factors: the single item factor q_i is replaced by
// a per-bin factor q_{i,Bin(t)} (an Array3), so item taste itself drifts in time.
// There is no plain q_i left — hence no lr_i/reg_i/sigma_i in the config.
// Produced: tsvdx2-16.  Frozen archive — see README.md, does not build.
use gravity::{Dataset, Regressor, calc_gbias, rand_array2};
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2, Array3};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

// Sample a (dim0 x dim1 x dim2) tensor from Normal(0, sigma)
#[inline]
fn rand_array3(dim0: usize, dim1: usize, dim2: usize, rng: &mut StdRng, sigma: f32) -> Array3<f32> {
    let dist = Normal::<f32>::new(0.0, sigma).unwrap();
    Array3::from_shape_fn((dim0, dim1, dim2), |_| dist.sample(rng))
}

#[derive(Clone, Copy, Debug)]
struct TimeSvdConfig {
    n_feat: usize,   // Number of latent factors
    n_epochs: usize, // Training epochs
    seed: u64,       // Random number generator seed

    lr_u: f32,       // User learning rate
    lr_ub: f32,      // User bias learning rate
    lr_ib: f32,      // Item bias learning rate
    lr_y: f32,       // NSVD1 learning rate
    lr_yd: f32,      // Per-day NSVD1 learning rate

    reg_u: f32,      // User regularization
    reg_y: f32,      // NSVD1 regularization
    reg_yd: f32,     // Per-day NSVD1 regularization

    // Time-bin item factors — these replace the plain item factors entirely
    lr_ifb: f32,     // (item, time bin) factor learning rate
    reg_ifb: f32,    // (item, time bin) factor regularization

    sigma_u: f32,    // Random init stddev (user)
    sigma_ifb: f32,  // Random init stddev ((item, time bin) factors)
    sigma_y: f32,    // Random init stddev (NSVD1)
    sigma_yd: f32,   // Random init stddev (per-day NSVD1)

    n_bins: usize,   // Number of time bins
    beta: f32,       // dev(u,t) exponent
    lr_t: f32,       // Time bias (alpha) learning rate

    reset_u_epoch: usize,       // Epoch at which ufeat is zeroed (1024 = never)
    store_ycache_day_tr: bool,  // Keep per-day NSVD1 vectors for train (u, day) pairs too
}

struct TimeSvdModel {
    cfg: TimeSvdConfig,

    gbias: f32,
    ubias: Array1<f32>,
    ibias: Array1<f32>,

    ufeat: Array2<f32>,
    ifeat_bin: Array3<f32>, // only bin-specific item factors
    yfeat: Array2<f32>,
    ycache: Array2<f32>,
    yfeat_day: Array2<f32>,
    ycache_day: HashMap<(usize, i32), Array1<f32>>,

    day_range: i32,
    tu_mean: Array1<f32>,
    alpha_u: Array1<f32>,
    but_bin: Array2<f32>,
    bit_bin: Array2<f32>,
}

impl TimeSvdModel {
    #[inline]
    fn time_bin(&self, day: i32) -> usize {
        let num = (day as i64) * (self.cfg.n_bins as i64);
        let b = (num / self.day_range as i64) as usize;
        b.min(self.cfg.n_bins - 1)
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
            if cnt > 0 {
                su /= (cnt as f32).sqrt();
            }
        }
    }

    fn rebuild_ycache_day(&mut self, tr: &Dataset, pr: &Dataset) {
        let mut cnts: HashMap<(usize, i32), f32> = HashMap::new();
        self.ycache_day.clear();

        for idx in 0..pr.n_ratings {
            let u = pr.user_idxs[idx] as usize;
            let i = pr.item_idxs[idx] as usize;
            let day = pr.dates[idx] as i32;
            let cnt = cnts.entry((u, day)).or_insert(0.0);
            *cnt += 1.0;
            let su = self
                .ycache_day
                .entry((u, day))
                .or_insert(Array1::<f32>::zeros(self.cfg.n_feat));
            *su += &self.yfeat_day.row(i);
        }

        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;
            let day = tr.dates[idx] as i32;
            if self.cfg.store_ycache_day_tr {
                let cnt = cnts.entry((u, day)).or_insert(0.0);
                *cnt += 1.0;
                let su = self
                    .ycache_day
                    .entry((u, day))
                    .or_insert(Array1::<f32>::zeros(self.cfg.n_feat));
                *su += &self.yfeat_day.row(i);
            } else if cnts.contains_key(&(u, day)) {
                let cnt = cnts.get_mut(&(u, day)).unwrap();
                *cnt += 1.0;
                let su = self.ycache_day.get_mut(&(u, day)).unwrap();
                *su += &self.yfeat_day.row(i);
            }
        }

        for (ud, cnt) in &cnts {
            let su = self.ycache_day.get_mut(ud).unwrap();
            *su /= cnt.sqrt();
        }
    }
}

// Group consecutive equal values into "runs" in arr[start..end)
fn eq_ranges(arr: &Array1<i16>, start: usize, end: usize) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
    let mut run_starts = HashMap::new();
    let mut run_stops = HashMap::new();
    if start == end {
        return (run_starts, run_stops);
    }

    let mut run_start = start;
    let mut prev_val = arr[start];
    for idx in (start + 1)..end {
        let val = arr[idx];
        if val != prev_val {
            run_starts.insert(run_start, idx - 1);
            run_stops.insert(idx - 1, run_start);
            run_start = idx;
            prev_val = val;
        }
    }

    run_starts.insert(run_start, end - 1);
    run_stops.insert(end - 1, run_start);
    (run_starts, run_stops)
}

impl Regressor for TimeSvdModel {
    type Config = TimeSvdConfig;

    fn new(tr: &Dataset, _pr: &Dataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        let mut day_range = 0;
        let mut tu_mean = Array1::<f32>::zeros(tr.n_users);
        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            tu_mean[u] += tr.dates[idx] as f32;
            day_range = day_range.max(tr.dates[idx] as i32 + 1)
        }
        for u in 0..tr.n_users {
            let cnt = tr.user_cnts[u];
            if cnt > 0 {
                tu_mean[u] /= cnt as f32;
            }
        }

        Self {
            cfg,
            gbias: calc_gbias(tr),
            ubias: Array1::<f32>::zeros(tr.n_users),
            ibias: Array1::<f32>::zeros(tr.n_items),

            ufeat: rand_array2(tr.n_users, cfg.n_feat, &mut rng, cfg.sigma_u),
            ifeat_bin: rand_array3(tr.n_items, cfg.n_bins, cfg.n_feat, &mut rng, cfg.sigma_ifb),
            yfeat: rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_y),
            ycache: Array2::<f32>::zeros((tr.n_users, cfg.n_feat)),
            yfeat_day: rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_yd),
            ycache_day: HashMap::new(),

            day_range,
            tu_mean,
            alpha_u: Array1::<f32>::zeros(tr.n_users),
            but_bin: Array2::<f32>::zeros((tr.n_users, cfg.n_bins)),
            bit_bin: Array2::<f32>::zeros((tr.n_items, cfg.n_bins)),
        }
    }

    fn n_epochs(&self) -> usize {
        self.cfg.n_epochs
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let b = self.time_bin(day);
        let dev = self.dev(u, day);

        let bu_t = self.ubias[u] + self.but_bin[[u, b]] + self.alpha_u[u] * dev;
        let bi_t = self.ibias[i] + self.bit_bin[[i, b]];

        let pu = &self.ufeat.row(u);
        let su = &self.ycache.row(u);
        let su_day = self.ycache_day.get(&(u, day)).unwrap();

        let x = pu + su + su_day;
        let qbin = self.ifeat_bin.slice(ndarray::s![i, b, ..]); // scale is constant 1

        self.gbias + bu_t + bi_t + x.dot(&qbin)
    }

    fn fit_epoch(&mut self, tr: &Dataset, pr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        if epoch == cfg.reset_u_epoch {
            self.ufeat.fill(0.0);
        }

        let mut idx = 0;
        for u in (0..tr.n_users).progress() {
            let cnt = tr.user_cnts[u] as usize;
            if cnt == 0 {
                continue;
            }
            let start = idx;
            let end = idx + cnt;
            idx = end;

            let (dt_starts, dt_stops) = eq_ranges(&tr.dates, start, end);

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
            let mut sum_err_q_day = Array1::<f32>::zeros(cfg.n_feat);
            let mut su_day = Array1::<f32>::zeros(cfg.n_feat);
            let mut norm_day = 0.0;

            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                let r = tr.residuals[t];
                let day = tr.dates[t] as i32;
                let b = self.time_bin(day);
                let dev = self.dev(u, day);

                if let Some(dt_stop) = dt_starts.get(&t) {
                    sum_err_q_day.fill(0.0);

                    // Compute ycache_day[u, day]
                    su_day.fill(0.0);
                    for t_day in t..=(*dt_stop) {
                        let j = tr.item_idxs[t_day] as usize;
                        su_day += &self.yfeat_day.row(j);
                    }
                    let cnt_day = (*dt_stop - t + 1) as f32;
                    norm_day = cnt_day.sqrt();
                    su_day /= norm_day;
                    self.ycache_day.insert((u, day), su_day.clone());
                }

                let err = self.predict(u, i, day) - r;

                // Bias updates
                self.ubias[u] -= cfg.lr_ub * err;
                self.ibias[i] -= cfg.lr_ib * err;

                // Time bias/drift updates
                self.but_bin[[u, b]] -= cfg.lr_t * err;
                self.bit_bin[[i, b]] -= cfg.lr_t * err;
                self.alpha_u[u] -= cfg.lr_t * err * dev;

                // Factor updates (ONLY bin item factors)
                for k in 0..cfg.n_feat {
                    let q_bin = self.ifeat_bin[[i, b, k]];
                    let pu = self.ufeat[[u, k]];
                    let xk = pu + su[k] + su_day[k];

                    sum_err_q[k] += err * q_bin;
                    sum_err_q_day[k] += err * q_bin;

                    self.ufeat[[u, k]] -= cfg.lr_u * (err * q_bin + cfg.reg_u * pu);
                    self.ifeat_bin[[i, b, k]] -= cfg.lr_ifb * (err * xk + cfg.reg_ifb * q_bin);
                }

                if let Some(dt_start) = dt_stops.get(&t) {
                    // Per-day NSVD1 factor updates
                    for t_day in *dt_start..=t {
                        let j = tr.item_idxs[t_day] as usize;
                        for k in 0..cfg.n_feat {
                            let yj = self.yfeat_day[[j, k]];
                            self.yfeat_day[[j, k]] -= cfg.lr_yd * (sum_err_q_day[k] / norm_day + cfg.reg_yd * yj);
                        }
                    }
                    self.ycache_day.remove(&(u, day));
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
        self.rebuild_ycache_day(tr, pr);
    }
}

fn main() {
    let cfg = TimeSvdConfig {
        n_feat: 16,
        n_epochs: 12,
        seed: 42,
        lr_u: 0.004,
        lr_ub: 0.0031,
        lr_ib: 0.0036,
        lr_y: 0.0005,
        lr_yd: 0.0005,
        reg_u: 0.04,
        reg_y: 0.04,
        reg_yd: 0.04,
        lr_ifb: 0.0015,
        reg_ifb: 0.01,
        sigma_u: 0.005,
        sigma_ifb: 0.005,
        sigma_y: 0.005,
        sigma_yd: 0.005,
        n_bins: 30,
        beta: 0.3,
        lr_t: 3e-6,
        reset_u_epoch: 1024,
        store_ycache_day_tr: false,
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
