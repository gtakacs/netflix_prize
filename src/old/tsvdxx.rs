// tsvdx.rs, memory-efficient: the full train (user, day) -> vector map is never
// materialised. Day-runs are located by binary search over a compact copy of the
// train stream and held in a one-slot rolling Mutex<PredDayCache>; only the probe
// (u, day) vectors are cached outright.
// Produced: tsvdxx-2048.  Frozen archive — see README.md, does not build.
use indicatif::ProgressIterator;
use gravity::{Dataset, Regressor, calc_gbias, rand_array2};
use ndarray::{Array1, Array2};
use rand::{SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use std::sync::Mutex;

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
    lr_yd: f32,      // per day NSVD1 learning rate

    reg_u: f32,      // User regularization
    reg_i: f32,      // Item regularization
    reg_y: f32,      // NSVD1 regularization
    reg_yd: f32,     // per day NSVD1 regularization

    sigma_u: f32,    // Random init stddev (user)
    sigma_i: f32,    // Random init stddev (item)
    sigma_y: f32,    // Random init stddev (NSVD1)
    sigma_yd: f32,   // Random init stddev (per day NSVD1)

    n_bins: usize,   // Number of time bins
    beta: f32,       // dev(u,t) exponent
    lr_t: f32,       // Time bias (alpha) learning rate

    reset_u_epoch: usize, // Epoch at which ufeat is zeroed (1024 = never)
}

struct TimeSvdModel {
    cfg: TimeSvdConfig, // Model hyperparameters

    gbias: f32,              // Global bias
    ubias: Array1<f32>,      // User biases
    ibias: Array1<f32>,      // Item biases

    ufeat: Array2<f32>,      // User feature vectors
    ifeat: Array2<f32>,      // Item feature vectors
    yfeat: Array2<f32>,      // NSVD1 feature vectors
    ycache: Array2<f32>,
    yfeat_day: Array2<f32>,  // Per-day NSVD1 feature vectors

    // Normalized per-(user, day) NSVD1 cache used for probe predictions.
    // We intentionally DO NOT store the full train cache here to avoid huge memory usage.
    ycache_day: HashMap<(usize, i32), Array1<f32>>,

    // Probe-only raw (sum, cnt) for (user,day) pairs appearing in probe.
    // During prediction on train, we add this probe-only contribution on-the-fly.
    ycache_day_pr_only_raw: HashMap<(usize, i32), (Array1<f32>, f32)>,

    // Compact copies of the train rating stream, used to compute per-(user,day) runs on the fly.
    // Assumption (as per request): within each user slice, train ratings are sorted by day.
    tr_dates: Vec<i16>,
    tr_items: Vec<i32>,
    tr_user_starts: Vec<usize>,

    // Rolling cache for on-the-fly train (user,day) ycache_day computation.
    pred_day_cache: Mutex<PredDayCache>,

    day_range: i32,
    tu_mean: Array1<f32>,    // User mean dates
    alpha_u: Array1<f32>,    // User drift scales
    but_bin: Array2<f32>,    // User × time bin biases
    bit_bin: Array2<f32>,    // Item × time bin biases
}

#[derive(Debug)]
struct PredDayCache {
    valid: bool,
    u: usize,
    day: i32,
    // Current run bounds in the *train* stream [run_start, run_end)
    run_start: usize,
    run_end: usize,
    su_day: Array1<f32>,
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

    fn rebuild_ycache_day(&mut self, tr: &Dataset, pr: &Dataset) {
        // Build a small probe-only cache, but ensure each probe (u,day) vector
        // incorporates BOTH probe + train items for that day (to match the original behavior).
        // We keep the probe-only raw sums so that train predictions can add them on-the-fly,
        // without storing the huge full-train (u,day)->vector map.
        self.ycache_day.clear();
        self.ycache_day_pr_only_raw.clear();

        // 1) Probe-only raw accumulation
        for idx in 0..pr.n_ratings {
            let u = pr.user_idxs[idx] as usize;
            let i = pr.item_idxs[idx] as usize;
            let day = pr.dates[idx] as i32;
            let entry = self
                .ycache_day_pr_only_raw
                .entry((u, day))
                .or_insert((Array1::<f32>::zeros(self.cfg.n_feat), 0.0));
            entry.1 += 1.0;
            entry.0 += &self.yfeat_day.row(i);
        }

        if self.ycache_day_pr_only_raw.is_empty() {
            return;
        }

        // 2) Copy probe sums to a mutable total map, then add matching train items
        let mut sum_total: HashMap<(usize, i32), Array1<f32>> =
            HashMap::with_capacity(self.ycache_day_pr_only_raw.len());
        let mut cnt_total: HashMap<(usize, i32), f32> =
            HashMap::with_capacity(self.ycache_day_pr_only_raw.len());
        for (ud, (sum_pr, cnt_pr)) in self.ycache_day_pr_only_raw.iter() {
            sum_total.insert(*ud, sum_pr.clone());
            cnt_total.insert(*ud, *cnt_pr);
        }

        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;
            let day = tr.dates[idx] as i32;
            if let Some(sum) = sum_total.get_mut(&(u, day)) {
                *sum += &self.yfeat_day.row(i);
                *cnt_total.get_mut(&(u, day)).unwrap() += 1.0;
            }
        }

        // 3) Normalize and store probe cache
        for (ud, sum) in sum_total {
            let cnt = cnt_total[&ud];
            if cnt > 0.0 {
                let mut su = sum;
                su /= cnt.sqrt();
                self.ycache_day.insert(ud, su);
            }
        }
    }

    #[inline]
    fn tr_user_slice(&self, u: usize) -> (usize, usize) {
        let s = self.tr_user_starts[u];
        let e = self.tr_user_starts[u + 1];
        (s, e)
    }

    // Find the [run_start, run_end) bounds in the train stream for a given (user, day).
    // Returns None if the day doesn't appear for the user in train.
    fn find_tr_day_run_bounds(&self, u: usize, day: i32) -> Option<(usize, usize)> {
        let (s, e) = self.tr_user_slice(u);
        if s == e {
            return None;
        }
        let target = day as i16;

        // Lower_bound on dates[s..e]
        let mut lo = s;
        let mut hi = e;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.tr_dates[mid] < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= e || self.tr_dates[lo] != target {
            return None;
        }
        let mut run_end = lo;
        while run_end < e && self.tr_dates[run_end] == target {
            run_end += 1;
        }
        Some((lo, run_end))
    }

    // Ensure the rolling train (user,day) ycache_day is up-to-date, then return a reference to it.
    fn get_train_su_day_cached<'a>(&'a self, u: usize, day: i32) -> std::sync::MutexGuard<'a, PredDayCache> {
        let mut guard = self.pred_day_cache.lock().unwrap();
        if guard.valid && guard.u == u && guard.day == day {
            return guard;
        }

        // Fast path: if next day-run follows immediately in the train stream for the same user,
        // we can find it by scanning forward.
        let (user_s, user_e) = self.tr_user_slice(u);
        let target = day as i16;
        let mut run_bounds: Option<(usize, usize)> = None;
        if guard.valid && guard.u == u && guard.run_end < user_e {
            let next = guard.run_end;
            if next >= user_s && self.tr_dates[next] == target {
                let mut run_end = next;
                while run_end < user_e && self.tr_dates[run_end] == target {
                    run_end += 1;
                }
                run_bounds = Some((next, run_end));
            }
        }
        if run_bounds.is_none() {
            run_bounds = self.find_tr_day_run_bounds(u, day);
        }
        let (run_start, run_end) = run_bounds.unwrap_or((0, 0));

        // Compute sum over train run
        guard.su_day.fill(0.0);
        let mut cnt_total: f32 = 0.0;
        if run_end > run_start {
            for idx in run_start..run_end {
                let i = self.tr_items[idx] as usize;
                guard.su_day += &self.yfeat_day.row(i);
            }
            cnt_total += (run_end - run_start) as f32;
        }

        // Add probe-only contribution for this (u,day)
        if let Some((sum_pr, cnt_pr)) = self.ycache_day_pr_only_raw.get(&(u, day)) {
            guard.su_day += sum_pr;
            cnt_total += *cnt_pr;
        }

        if cnt_total > 0.0 {
            guard.su_day /= cnt_total.sqrt();
        }

        guard.valid = true;
        guard.u = u;
        guard.day = day;
        guard.run_start = run_start;
        guard.run_end = run_end;
        guard
    }
}

// Group consecutive equal values into "runs" in arr[start..end)
fn eq_ranges(arr: &Array1<i16>, start: usize, end: usize)
                 -> (HashMap<usize, usize>, HashMap<usize, usize>) {

    let mut run_starts = HashMap::new();
    let mut run_stops = HashMap::new();
    if start == end { return (run_starts, run_stops); }

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

        // Build compact copies of the train stream and per-user offsets.
        let mut tr_user_starts: Vec<usize> = Vec::with_capacity(tr.n_users + 1);
        tr_user_starts.push(0);
        let mut acc: usize = 0;
        for u in 0..tr.n_users {
            acc += tr.user_cnts[u] as usize;
            tr_user_starts.push(acc);
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
            yfeat_day: rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_yd),

            ycache_day: HashMap::new(),
            ycache_day_pr_only_raw: HashMap::new(),

            tr_dates: tr.dates.to_vec(),
            tr_items: tr.item_idxs.to_vec(),
            tr_user_starts,

            pred_day_cache: Mutex::new(PredDayCache {
                valid: false,
                u: 0,
                day: 0,
                run_start: 0,
                run_end: 0,
                su_day: Array1::<f32>::zeros(cfg.n_feat),
            }),

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

        // su_day:
        // - If (u,day) exists in the probe cache, use it.
        // - Otherwise compute on-the-fly from the train stream (optionally adding probe-only items).
        if let Some(su_day) = self.ycache_day.get(&(u, day)) {
            self.gbias + bu_t + bi_t + (pu + su + su_day).dot(qi)
        } else {
            let guard = self.get_train_su_day_cached(u, day);
            self.gbias + bu_t + bi_t + (pu + su + &guard.su_day).dot(qi)
        }
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

            let (dt_starts, dt_stops) =
                eq_ranges(&tr.dates, start, end);

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
                let i = tr.item_idxs[t] as usize; // Item index
                let r = tr.residuals[t];          // Rating value
                let day = tr.dates[t] as i32;     // Rating date
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
                    let cnt = (*dt_stop - t + 1) as f32;
                    norm_day = (cnt as f32).sqrt();
                    su_day /= norm_day;
                    self.ycache_day.insert((u, day), su_day.clone());
                }

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
                    sum_err_q_day[k] += err * qi;
                    self.ufeat[[u, k]] -= cfg.lr_u * (err * qi + cfg.reg_u * pu);
                    self.ifeat[[i, k]] -= cfg.lr_i * (err * (pu + su[k] + su_day[k]) + cfg.reg_i * qi);
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
        n_epochs: 7,
        seed: 42,
        lr_u: 0.0,
        lr_ub: 0.0031,
        lr_i: 0.0036,
        lr_ib: 0.0036,
        lr_y: 0.0005,
        lr_yd: 0.0005,
        reg_u: 0.04,
        reg_i: 0.007,
        reg_y: 0.04,
        reg_yd: 0.04,
        sigma_u: 0.0,
        sigma_i: 0.005,
        sigma_y: 0.005,
        sigma_yd: 0.005,
        n_bins: 30,
        beta: 0.3,
        lr_t: 1e-5,
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
