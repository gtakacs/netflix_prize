// Union Time-SVD++ model supporting tsvdx4, tsvdx5, and tsvdx6 variants.
//
// tx4 mode: sum_err_bug=true, w_bias=w_factor=w_nbr=1.0, lr_w=lr_c=0 → no neighborhood
// tx5 mode: sum_err_bug=false, global neighborhood (w, c), no same-day terms
// tx6 mode: sum_err_bug=false, global + same-day neighborhood (w_day, c_day)

use indicatif::ProgressIterator;
use crate::{
    calc_gbias, calc_user_offsets, get_users, rand_array2,
    Dataset, MaskedDataset, Regressor, OrdinalHeadConfig, OrdinalHead,
};
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// Sparse (user, day) index — with day_cnts for frequency bins
// ---------------------------------------------------------------------------
pub struct SparseUD {
    starts: Vec<usize>,     // [n_users + 1] — index into `dates`
    dates: Vec<i16>,        // sorted unique dates, concatenated per user
    pub day_cnts: Vec<u32>, // rating count per (user, day) pair
}

impl SparseUD {
    /// Build from two datasets (train + probe/qual).
    /// Note: ds2 (probe/qual) may be sorted by item, not user,
    /// so we must scan linearly instead of using calc_user_offsets.
    pub fn new(ds1: &Dataset, ds2: &MaskedDataset) -> Self {
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
        let mut day_cnts = Vec::new();
        for u in 0..n_users {
            starts.push(dates.len());
            per_user[u].sort_unstable();
            let sorted = &per_user[u];
            if !sorted.is_empty() {
                let mut prev = sorted[0];
                let mut cnt: u32 = 1;
                for &d in &sorted[1..] {
                    if d == prev {
                        cnt += 1;
                    } else {
                        dates.push(prev);
                        day_cnts.push(cnt);
                        prev = d;
                        cnt = 1;
                    }
                }
                dates.push(prev);
                day_cnts.push(cnt);
            }
        }
        starts.push(dates.len());

        Self { starts, dates, day_cnts }
    }

    #[inline]
    pub fn n_total(&self) -> usize { self.dates.len() }

    /// Binary search for (user, day) → flat index, or None.
    #[inline]
    pub fn index(&self, u: usize, day: i16) -> Option<usize> {
        let start = self.starts[u];
        let end = self.starts[u + 1];
        self.dates[start..end]
            .binary_search(&day)
            .ok()
            .map(|i| start + i)
    }
}

#[inline]
pub fn freq_bin(day_cnt: u32, n_freq_bins: usize) -> usize {
    if day_cnt <= 1 { return 0; }
    (day_cnt as f32).log2().floor().min((n_freq_bins - 1) as f32) as usize
}

/// For a user's ratings in dates[start..end), return Vec of (run_start, run_end_inclusive).
pub fn day_ranges(dates: &Array1<i16>, start: usize, end: usize) -> Vec<(usize, usize)> {
    if start == end { return Vec::new(); }
    let mut ranges = Vec::new();
    let mut run_start = start;
    for idx in (start + 1)..end {
        if dates[idx] != dates[idx - 1] {
            ranges.push((run_start, idx - 1));
            run_start = idx;
        }
    }
    ranges.push((run_start, end - 1));
    ranges
}

// ---------------------------------------------------------------------------
// Neighborhood data structures — always carry day for tx6 same-day support
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RatedItem {
    pub item: u16,
    pub day: i16,
    pub r_minus_btilde: f32,
}

#[derive(Clone)]
pub struct ImplicitItem {
    pub item: u16,
    pub day: i16,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct TxConfig {
    pub n_feat: usize,        // latent factor dimension
    pub n_epochs: usize,      // SGD passes over all users
    pub seed: u64,            // RNG seed (init + per-epoch shuffle)
    pub shuffle_users: bool,  // randomize user order each epoch

    pub n_time_bins: usize,   // # bins for but_bin / bit_bin
    pub beta: f32,            // exponent in Koren's |Δt|^β time deviation
    pub n_freq_bins: usize,   // # bins for ibias_freq / ifeat_freq

    // Learning rates
    pub lr_u: f32,            // for: ufeat
    pub lr_ud: f32,           // for: ufeat_day
    pub lr_u2: f32,           // for: ufeat2 (time-drift factor)
    pub lr_ub: f32,           // for: ubias
    pub lr_ubd: f32,          // for: ubias_day
    pub lr_i: f32,            // for: ifeat
    pub lr_ib: f32,           // for: ibias
    pub lr_y: f32,            // for: yfeat (NSVD1)
    pub lr_yb: f32,           // for: ybias (NSVD1)
    pub lr_yd: f32,           // for: yfeat_day
    pub lr_tu: f32,           // for: but_bin (per-user time bin)
    pub lr_ti: f32,           // for: bit_bin (per-item time bin)
    pub lr_ta: f32,           // for: alpha_u (per-user time-drift coeff)
    pub lr_ibf: f32,          // for: ibias_freq
    pub lr_iqf: f32,          // for: ifeat_freq
    pub lr_cu: f32,           // for: cu (per-user item-bias multiplier)

    // Regularizations (L2)
    pub reg_iqf: f32,         // for: ifeat_freq
    pub reg_cu: f32,          // for: cu (toward 1.0)
    pub reg_u: f32,           // for: ufeat
    pub reg_u2: f32,          // for: ufeat2
    pub reg_ud: f32,          // for: ufeat_day
    pub reg_i: f32,           // for: ifeat
    pub reg_y: f32,           // for: yfeat
    pub reg_yd: f32,          // for: yfeat_day

    // Init stddevs (Normal(0, σ))
    pub sigma_iqf: f32,       // for: ifeat_freq
    pub sigma_u: f32,         // for: ufeat
    pub sigma_i: f32,         // for: ifeat
    pub sigma_y: f32,         // for: yfeat
    pub sigma_yd: f32,        // for: yfeat_day

    pub reset_u_epoch: usize, // epoch at which ufeat/ufeat2 are zeroed (0 = never)

    // Global neighborhood — set lr_w=lr_c=0 to disable (tx4 mode)
    pub max_neighbors: usize, // top-k cap when sampling neighbors (0 = all)
    pub lr_w: f32,            // for: w (rated-item weights)
    pub lr_c: f32,            // for: c (implicit-feedback weights)
    pub reg_w: f32,           // for: w
    pub reg_c: f32,           // for: c

    // Same-day neighborhood — set lr_w_day=lr_c_day=0 to disable (tx4/tx5 mode)
    pub lr_w_day: f32,        // for: w_day (same-day rated weights)
    pub lr_c_day: f32,        // for: c_day (same-day implicit weights)
    pub reg_w_day: f32,       // for: w_day
    pub reg_c_day: f32,       // for: c_day

    // Sub-model lr multipliers — set all to 1.0 for tx4 mode
    pub w_bias: f32,          // multiplier for: bias lr's (lr_ub, lr_tu, lr_ib, lr_ti, …)
    pub w_factor: f32,        // multiplier for: factor lr's (lr_u, lr_i, lr_y, …)
    pub w_nbr: f32,           // multiplier for: neighborhood lr's (lr_w, lr_c, lr_w_day, lr_c_day)

    pub sum_err_bug: bool,    // tx4-only quirk: accumulate sum_err inside the k-loop

    // Bayesian baseline damping for b̃ (used only when lr_w > 0 || lr_c > 0)
    pub lambda1: f32,         // for: per-item baseline b̃_i
    pub lambda2: f32,         // for: per-user baseline b̃_u

    pub ordinal_head: Option<OrdinalHeadConfig>, // optional ordinal head (None = MSE loss)
    pub save_ifeat: bool,     // save ifeat as `.ifeat.<ds>.npy` artifact
    pub low_memory: bool,     // skip per-day NSVD1 buffers (yfeat_day / ufeat_day / ycache_day)
    pub full_su: bool,        // build NSVD1 su from train ∪ probe (else train only)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct TxModel {
    cfg: TxConfig,                  // immutable training config

    // Bias terms
    gbias: f32,                     // global mean residual
    ubias: Array1<f32>,             // user biases [n_users]
    ibias: Array1<f32>,             // item biases [n_items]

    // Latent factors
    ufeat: Array2<f32>,             // user factors [n_users, n_feat]
    ufeat2: Array2<f32>,            // user time-drift factors [n_users, n_feat]
    ifeat: Array2<f32>,             // item factors [n_items, n_feat]
    yfeat: Array2<f32>,             // NSVD1 implicit-feedback factor [n_items, n_feat]
    ycache: Array2<f32>,            // cached Σ yfeat[j] / √|N(u)| per user [n_users, n_feat]
    yfeat_day: Array2<f32>,         // per-day NSVD1 factors [n_items, n_feat]

    // Temporal-bias state
    day_range: i32,                 // max day index + 1 (used for time_bin)
    tu_mean: Array1<f32>,           // mean rating day per user [n_users]
    alpha_u: Array1<f32>,           // per-user time-drift coefficient [n_users]
    but_bin: Array2<f32>,           // per-(user, time_bin) bias [n_users, n_time_bins]
    bit_bin: Array2<f32>,           // per-(item, time_bin) bias [n_items, n_time_bins]

    // NSVD1 bias
    ybias: Array1<f32>,             // NSVD1 per-item bias [n_items]
    ycache_bias: Array1<f32>,       // cached Σ ybias[j] / √|N(u)| per user [n_users]

    // Per-(user, day) state
    ud: SparseUD,                   // sparse (user, day) → linear index
    ubias_day: Vec<f32>,            // per-(user, day) bias [n_ud]
    ufeat_day: Array2<f32>,         // per-(user, day) factor [n_ud, n_feat]
    ycache_day: Array2<f32>,        // cached per-(user, day) NSVD1 sum [n_ud, n_feat]

    // Frequency-bin state (per-(item, freq_bin))
    ibias_freq: Array2<f32>,        // per-(item, freq_bin) bias [n_items, n_freq_bins]
    ifeat_freq: Array2<f32>,        // per-(item, freq_bin) factor [n_items*n_freq_bins, n_feat]

    // Per-user item-bias multiplier (cu)
    cu: Array1<f32>,                // global cu[u] (around 1.0) [n_users]
    cut: Vec<f32>,                  // per-(user, day) offset to cu [n_ud]

    // optional ordinal regression head
    ordinal_head: Option<OrdinalHead>,

    // Neighborhood weights - empty Vec means disabled
    w: Vec<f32>,                    // rated-item weights [n_items × n_items], flat row-major
    c: Vec<f32>,                    // implicit-feedback weights [n_items × n_items]
    w_day: Vec<f32>,                // same-day rated-item weights [n_items × n_items]
    c_day: Vec<f32>,                // same-day implicit-feedback weights [n_items × n_items]

    // Per-user neighborhood lists - empty when no neighborhood
    user_rated: Vec<Vec<RatedItem>>,       // R(u): items u has rated, with residuals minus baseline
    user_implicit: Vec<Vec<ImplicitItem>>, // N(u) \ R(u): probe items rated implicitly only
    norm_nu: Array1<f32>,           // 1/√|N(u)| normalization [n_users]
    norm_ru: Array1<f32>,           // 1/√|R(u)| normalization [n_users]

    n_items: usize,                 // cached tr.n_items
    user_offsets: Array1<usize>,    // cumulative user_cnts [n_users + 1]

    // Probe items per user (when full_su = true; included in NSVD1 su)
    probe_items_by_user: Vec<Vec<u32>>, // probe items per user [n_users]
    probe_items_by_ud: Vec<Vec<u32>>,   // probe items per (user, day) [n_ud]
}

impl TxModel {
    #[inline]
    fn has_nbr(&self) -> bool { !self.w.is_empty() || !self.c.is_empty() }

    #[inline]
    fn has_w_day(&self) -> bool { !self.w_day.is_empty() }

    #[inline]
    fn has_c_day(&self) -> bool { !self.c_day.is_empty() }

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

    #[inline]
    fn factor_parts(&self, u: usize, i: usize, day: i32) -> (f32, f32) {
        let cfg = &self.cfg;
        let b = self.time_bin(day);
        let dev = self.dev(u, day);
        let day16 = day as i16;
        let n_freq_bins = cfg.n_freq_bins;

        let ud_idx = self.ud.index(u, day16);
        let f = ud_idx.map_or(0, |idx| freq_bin(self.ud.day_cnts[idx], n_freq_bins));

        let bu_day = ud_idx.map_or(0.0, |idx| self.ubias_day[idx]);
        let bu_t = self.ubias[u] + self.but_bin[[u, b]] + self.alpha_u[u] * dev
            + bu_day + self.ycache_bias[u];
        let bi_t = self.ibias[i] + self.bit_bin[[i, b]] + self.ibias_freq[[i, f]];
        let cu_t = self.cu[u] + ud_idx.map_or(0.0, |idx| self.cut[idx]);

        let pu = self.ufeat.row(u);
        let pu2 = self.ufeat2.row(u);
        let su = self.ycache.row(u);
        let qi = self.ifeat.row(i);
        let qf = self.ifeat_freq.row(i * n_freq_bins + f);

        let mut dot = 0.0;
        for k in 0..cfg.n_feat {
            let qi_eff = qi[k] + qf[k];
            let mut pu_eff = pu[k] + su[k] + dev * pu2[k];
            if !cfg.low_memory {
                if let Some(idx) = ud_idx {
                    pu_eff += self.ufeat_day[[idx, k]];
                    pu_eff += self.ycache_day[[idx, k]];
                }
            }
            dot += pu_eff * qi_eff;
        }

        (self.gbias + bu_t + bi_t * cu_t, dot)
    }

    #[inline]
    fn nbr_score(&self, u: usize, i: usize, day: i16) -> f32 {
        let ni = self.n_items;
        let nn = self.norm_nu[u];
        let nr = self.norm_ru[u];
        let has_wd = self.has_w_day();
        let has_cd = self.has_c_day();

        let mut c_sum = 0.0_f32;
        let mut w_sum = 0.0_f32;
        for ri in &self.user_rated[u] {
            let j = ri.item as usize;
            if j == i { continue; }
            c_sum += self.c[i * ni + j];
            w_sum += ri.r_minus_btilde * self.w[i * ni + j];
        }
        for ii in &self.user_implicit[u] {
            let j = ii.item as usize;
            if j == i { continue; }
            c_sum += self.c[i * ni + j];
        }

        let mut c_day_sum = 0.0_f32;
        let mut w_day_sum = 0.0_f32;
        let mut n_same_rated = 0_u32;
        let mut n_same_total = 0_u32;

        if has_wd || has_cd {
            for ri in &self.user_rated[u] {
                let j = ri.item as usize;
                if j == i || ri.day != day { continue; }
                n_same_rated += 1;
                n_same_total += 1;
                if has_cd { c_day_sum += self.c_day[i * ni + j]; }
                if has_wd { w_day_sum += ri.r_minus_btilde * self.w_day[i * ni + j]; }
            }
            for ii in &self.user_implicit[u] {
                let j = ii.item as usize;
                if j == i || ii.day != day { continue; }
                n_same_total += 1;
                if has_cd { c_day_sum += self.c_day[i * ni + j]; }
            }
        }

        let nn_day = if n_same_total > 0 { (n_same_total as f32).powf(-0.5) } else { 0.0 };
        let nr_day = if n_same_rated > 0 { (n_same_rated as f32).powf(-0.5) } else { 0.0 };

        nn * c_sum + nr * w_sum + nn_day * c_day_sum + nr_day * w_day_sum
    }

    fn rebuild_ycache(&mut self, tr: &Dataset, pr: &MaskedDataset) {
        self.ycache.fill(0.0);
        self.ycache_bias.fill(0.0);

        for t in 0..tr.n_ratings {
            let u = tr.user_idxs[t] as usize;
            let i = tr.item_idxs[t] as usize;
            let mut su = self.ycache.row_mut(u);
            su += &self.yfeat.row(i);
            self.ycache_bias[u] += self.ybias[i];
        }
        for t in 0..pr.n_ratings {
            let u = pr.user_idxs[t] as usize;
            let i = pr.item_idxs[t] as usize;
            let mut su = self.ycache.row_mut(u);
            su += &self.yfeat.row(i);
            self.ycache_bias[u] += self.ybias[i];
        }

        for u in 0..tr.n_users {
            let mut su = self.ycache.row_mut(u);
            let cnt = tr.user_cnts[u] + pr.user_cnts[u];
            if cnt > 0 {
                su /= (cnt as f32).sqrt();
                self.ycache_bias[u] /= (cnt as f32).sqrt();
            }
        }
    }

    fn rebuild_ycache_day(&mut self, tr: &Dataset, pr: &MaskedDataset) {
        self.ycache_day.fill(0.0);
        let n_ud = self.ud.n_total();
        let mut cnts = vec![0.0f32; n_ud];

        for idx in 0..pr.n_ratings {
            let u = pr.user_idxs[idx] as usize;
            let i = pr.item_idxs[idx] as usize;
            let day = pr.dates[idx];
            if let Some(ud_idx) = self.ud.index(u, day) {
                cnts[ud_idx] += 1.0;
                let mut row = self.ycache_day.row_mut(ud_idx);
                row += &self.yfeat_day.row(i);
            }
        }

        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;
            let day = tr.dates[idx];
            if let Some(ud_idx) = self.ud.index(u, day) {
                cnts[ud_idx] += 1.0;
                let mut row = self.ycache_day.row_mut(ud_idx);
                row += &self.yfeat_day.row(i);
            }
        }

        for ud_idx in 0..n_ud {
            if cnts[ud_idx] > 0.0 {
                let norm = cnts[ud_idx].sqrt();
                let mut row = self.ycache_day.row_mut(ud_idx);
                row /= norm;
            }
        }
    }
}

impl Regressor for TxModel {
    type Config = TxConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let n_users = tr.n_users;
        let n_items = tr.n_items;

        let mut day_range = 0;
        let mut tu_mean = Array1::<f32>::zeros(n_users);
        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            tu_mean[u] += tr.dates[idx] as f32;
            day_range = day_range.max(tr.dates[idx] as i32 + 1);
        }
        for u in 0..n_users {
            let cnt = tr.user_cnts[u];
            if cnt > 0 { tu_mean[u] /= cnt as f32; }
        }

        let ud = SparseUD::new(tr, pr);
        let n_ud = ud.n_total();
        let n_freq_bins = cfg.n_freq_bins;
        let low_mem = cfg.low_memory;
        let user_offsets = calc_user_offsets(tr);
        let gbias = calc_gbias(tr);

        let build_nbr = cfg.lr_w > 0.0 || cfg.lr_c > 0.0;
        let build_nbr_day = cfg.lr_w_day > 0.0 || cfg.lr_c_day > 0.0;

        let (user_rated, user_implicit, norm_nu, norm_ru) = if build_nbr {
            let mu = gbias;

            let mut item_sum = vec![0.0_f64; n_items];
            let mut item_cnt = vec![0.0_f64; n_items];
            for idx in 0..tr.n_ratings {
                let i = tr.item_idxs[idx] as usize;
                let r = tr.raw_ratings[idx] as f64;
                item_sum[i] += r - mu as f64;
                item_cnt[i] += 1.0;
            }
            let mut btilde_i = vec![0.0_f32; n_items];
            for i in 0..n_items {
                btilde_i[i] = (item_sum[i] / (item_cnt[i] + cfg.lambda1 as f64)) as f32;
            }

            let mut user_sum = vec![0.0_f64; n_users];
            let mut user_cnt = vec![0.0_f64; n_users];
            for idx in 0..tr.n_ratings {
                let u = tr.user_idxs[idx] as usize;
                let i = tr.item_idxs[idx] as usize;
                let r = tr.raw_ratings[idx] as f64;
                user_sum[u] += r - mu as f64 - btilde_i[i] as f64;
                user_cnt[u] += 1.0;
            }
            let mut btilde_u = vec![0.0_f32; n_users];
            for u in 0..n_users {
                btilde_u[u] = (user_sum[u] / (user_cnt[u] + cfg.lambda2 as f64)) as f32;
            }

            // Build R(u)
            let mut user_rated: Vec<Vec<RatedItem>> = vec![Vec::new(); n_users];
            for u in 0..n_users {
                let start = user_offsets[u];
                let end = user_offsets[u + 1];
                let mut v = Vec::with_capacity(end - start);
                for idx in start..end {
                    let i = tr.item_idxs[idx] as usize;
                    let r = tr.raw_ratings[idx] as f32;
                    let btilde = mu + btilde_i[i] + btilde_u[u];
                    v.push(RatedItem {
                        item: i as u16,
                        day: tr.dates[idx],
                        r_minus_btilde: r - btilde,
                    });
                }
                user_rated[u] = v;
            }

            // Build N(u)\R(u)
            let mut user_implicit: Vec<Vec<ImplicitItem>> = vec![Vec::new(); n_users];
            let mut user_rated_set: Vec<Vec<u16>> = vec![Vec::new(); n_users];
            for u in 0..n_users {
                let mut items: Vec<u16> = user_rated[u].iter().map(|ri| ri.item).collect();
                items.sort_unstable();
                user_rated_set[u] = items;
            }
            for idx in 0..pr.n_ratings {
                let u = pr.user_idxs[idx] as usize;
                let i = pr.item_idxs[idx] as usize;
                let day = pr.dates[idx];
                if user_rated_set[u].binary_search(&(i as u16)).is_err() {
                    user_implicit[u].push(ImplicitItem { item: i as u16, day });
                }
            }
            for u in 0..n_users {
                if build_nbr_day {
                    user_implicit[u].sort_unstable_by_key(|ii| (ii.item, ii.day));
                    user_implicit[u].dedup_by_key(|ii| (ii.item, ii.day));
                } else {
                    user_implicit[u].sort_unstable_by_key(|ii| ii.item);
                    user_implicit[u].dedup_by_key(|ii| ii.item);
                }
            }

            let mut norm_nu = Array1::<f32>::zeros(n_users);
            let mut norm_ru = Array1::<f32>::zeros(n_users);
            for u in 0..n_users {
                let n_r = user_rated[u].len();
                let n_n = n_r + user_implicit[u].len();
                norm_ru[u] = if n_r > 0 { (n_r as f32).powf(-0.5) } else { 0.0 };
                norm_nu[u] = if n_n > 0 { (n_n as f32).powf(-0.5) } else { 0.0 };
            }

            (user_rated, user_implicit, norm_nu, norm_ru)
        } else {
            (
                vec![Vec::new(); n_users],
                vec![Vec::new(); n_users],
                Array1::zeros(n_users),
                Array1::zeros(n_users),
            )
        };

        let n2 = n_items * n_items;
        let w     = if build_nbr     { vec![0.0_f32; n2] } else { Vec::new() };
        let c     = if build_nbr     { vec![0.0_f32; n2] } else { Vec::new() };
        let w_day = if build_nbr_day { vec![0.0_f32; n2] } else { Vec::new() };
        let c_day = if build_nbr_day { vec![0.0_f32; n2] } else { Vec::new() };

        let mut probe_items_by_user: Vec<Vec<u32>> = vec![Vec::new(); n_users];
        let mut probe_items_by_ud: Vec<Vec<u32>> = vec![Vec::new(); n_ud];
        if cfg.full_su {
            for t in 0..pr.n_ratings {
                let u = pr.user_idxs[t] as usize;
                let item = pr.item_idxs[t] as u32;
                probe_items_by_user[u].push(item);
                if let Some(ud_idx) = ud.index(u, pr.dates[t]) {
                    probe_items_by_ud[ud_idx].push(item);
                }
            }
        }

        Self {
            cfg,
            gbias,
            ubias: Array1::zeros(n_users),
            ibias: Array1::zeros(n_items),

            ufeat: rand_array2(n_users, cfg.n_feat, &mut rng, cfg.sigma_u),
            ufeat2: Array2::zeros((n_users, cfg.n_feat)),
            ifeat: rand_array2(n_items, cfg.n_feat, &mut rng, cfg.sigma_i),
            yfeat: rand_array2(n_items, cfg.n_feat, &mut rng, cfg.sigma_y),
            ycache: Array2::zeros((n_users, cfg.n_feat)),
            yfeat_day: if low_mem { Array2::zeros((0, cfg.n_feat)) }
                       else { rand_array2(n_items, cfg.n_feat, &mut rng, cfg.sigma_yd) },

            day_range,
            tu_mean,
            alpha_u: Array1::zeros(n_users),
            but_bin: Array2::zeros((n_users, cfg.n_time_bins)),
            bit_bin: Array2::zeros((n_items, cfg.n_time_bins)),
            ybias: Array1::zeros(n_items),
            ycache_bias: Array1::zeros(n_users),

            ubias_day: vec![0.0; n_ud],
            ufeat_day: if low_mem { Array2::zeros((0, cfg.n_feat)) }
                       else { Array2::zeros((n_ud, cfg.n_feat)) },
            ycache_day: if low_mem { Array2::zeros((0, cfg.n_feat)) }
                        else { Array2::zeros((n_ud, cfg.n_feat)) },
            ud,

            ibias_freq: Array2::zeros((n_items, n_freq_bins)),
            ifeat_freq: rand_array2(n_items * n_freq_bins, cfg.n_feat, &mut rng, cfg.sigma_iqf),

            cu: Array1::ones(n_users),
            cut: vec![0.0; n_ud],

            ordinal_head: cfg.ordinal_head.map(OrdinalHead::new),

            w, c, w_day, c_day,
            user_rated,
            user_implicit,
            norm_nu,
            norm_ru,

            n_items,
            user_offsets,

            probe_items_by_user,
            probe_items_by_ud,
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn n_subscores(&self) -> usize { if self.has_nbr() { 4 } else { 3 } }

    fn subscore_names(&self) -> Vec<String> {
        if self.has_nbr() {
            ["bias", "mf", "nsvd1", "nbr"].iter().map(|s| s.to_string()).collect()
        } else {
            ["bias", "mf", "nsvd1"].iter().map(|s| s.to_string()).collect()
        }
    }

    fn predict_subscores(&self, u: usize, i: usize, day: i32) -> Array1<f32> {
        let cfg = &self.cfg;
        let b = self.time_bin(day);
        let dev = self.dev(u, day);
        let day16 = day as i16;
        let n_freq_bins = cfg.n_freq_bins;

        let ud_idx = self.ud.index(u, day16);
        let f = ud_idx.map_or(0, |idx| freq_bin(self.ud.day_cnts[idx], n_freq_bins));

        let bu_day = ud_idx.map_or(0.0, |idx| self.ubias_day[idx]);
        let bi_t = self.ibias[i] + self.bit_bin[[i, b]] + self.ibias_freq[[i, f]];
        let cu_t = self.cu[u] + ud_idx.map_or(0.0, |idx| self.cut[idx]);

        let bias = self.gbias + self.ubias[u] + self.ibias[i] + self.ibias_freq[[i, f]]
            + self.but_bin[[u, b]] + self.alpha_u[u] * dev + bu_day
            + self.bit_bin[[i, b]] + bi_t * (cu_t - 1.0)
            + self.ycache_bias[u];

        let pu = self.ufeat.row(u);
        let pu2 = self.ufeat2.row(u);
        let su = self.ycache.row(u);
        let qi = self.ifeat.row(i);
        let qf = self.ifeat_freq.row(i * n_freq_bins + f);

        let mut mf = 0.0_f32;
        let mut nsvd1 = 0.0_f32;

        for k in 0..cfg.n_feat {
            let qi_eff = qi[k] + qf[k];
            let mut pu_eff = pu[k] + dev * pu2[k];
            if !cfg.low_memory {
                if let Some(idx) = ud_idx { pu_eff += self.ufeat_day[[idx, k]]; }
            }
            mf += pu_eff * qi_eff;
            let mut su_eff = su[k];
            if !cfg.low_memory {
                if let Some(idx) = ud_idx { su_eff += self.ycache_day[[idx, k]]; }
            }
            nsvd1 += su_eff * qi_eff;
        }

        if self.has_nbr() {
            let nbr = self.nbr_score(u, i, day16);
            Array1::from_vec(vec![bias, mf, nsvd1, nbr])
        } else {
            Array1::from_vec(vec![bias, mf, nsvd1])
        }
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let (bias_part, dot_part) = self.factor_parts(u, i, day);
        let nbr_part = if self.has_nbr() { self.nbr_score(u, i, day as i16) } else { 0.0 };
        let score = bias_part + dot_part + nbr_part;
        match &self.ordinal_head {
            Some(ordinal) => {
                let probs = ordinal.predict_probs(score);
                1.0 * probs[0] + 2.0 * probs[1] + 3.0 * probs[2] + 4.0 * probs[3] + 5.0 * probs[4]
            }
            None => score,
        }
    }

    fn save_artifacts(&self, model_name: &str, tr_set: &str, preds_dir: &str) {
        if self.cfg.save_ifeat {
            let path = format!("{}/{}.ifeat.{}.npy", preds_dir, model_name, tr_set);
            write_npy(&path, &self.ifeat).unwrap();
            crate::teeln!("Saved item factors to {}", path);
        }
    }

    fn fit_epoch(&mut self, tr: &Dataset, pr: &MaskedDataset, epoch: usize) {
        let cfg = self.cfg;
        let ni = self.n_items;
        let n_freq_bins = cfg.n_freq_bins;
        let max_nbr = cfg.max_neighbors;
        let use_probe = cfg.full_su;
        let low_mem = cfg.low_memory;
        let has_nbr = self.has_nbr();
        let has_wd = self.has_w_day();
        let has_cd = self.has_c_day();

        if epoch == cfg.reset_u_epoch {
            self.ufeat.fill(0.0);
            self.ufeat2.fill(0.0);
        }

        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        let mut sampled_rated_idx: Vec<usize> = Vec::new();
        let mut sampled_impl_idx: Vec<usize> = Vec::new();
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(epoch as u64 * 1_000_003));

        for &u in progress!(users.iter()) {
            let start = self.user_offsets[u];
            let end = self.user_offsets[u + 1];
            let cnt = end - start;
            if cnt == 0 { continue; }

            // ----- Sample global neighbors (tx5/tx6 mode only) -----
            let (iter_rated, iter_impl, rated_scale, impl_scale, use_sampling) = if has_nbr {
                let nr_full = self.user_rated[u].len();
                let ni_full = self.user_implicit[u].len();
                let n_full = nr_full + ni_full;

                if max_nbr > 0 && n_full > max_nbr {
                    let rated_budget = ((max_nbr as f64 * nr_full as f64 / n_full as f64).round() as usize)
                        .max(1).min(nr_full);
                    let impl_budget = (max_nbr - rated_budget).min(ni_full);

                    sampled_rated_idx.clear();
                    sampled_rated_idx.extend(0..nr_full);
                    sampled_rated_idx.shuffle(&mut rng);
                    sampled_rated_idx.truncate(rated_budget);
                    sampled_rated_idx.sort_unstable();

                    sampled_impl_idx.clear();
                    sampled_impl_idx.extend(0..ni_full);
                    sampled_impl_idx.shuffle(&mut rng);
                    sampled_impl_idx.truncate(impl_budget);
                    sampled_impl_idx.sort_unstable();

                    let rs = nr_full as f32 / rated_budget as f32;
                    let is = if impl_budget > 0 { ni_full as f32 / impl_budget as f32 } else { 1.0 };
                    (rated_budget, impl_budget, rs, is, true)
                } else {
                    (nr_full, ni_full, 1.0_f32, 1.0_f32, false)
                }
            } else {
                (0, 0, 1.0_f32, 1.0_f32, false)
            };

            let nr = if has_nbr { self.norm_ru[u] } else { 0.0 };
            let nn = if has_nbr { self.norm_nu[u] } else { 0.0 };

            let day_rngs = if low_mem { Vec::new() } else { day_ranges(&tr.dates, start, end) };

            // NSVD1 user-level aggregation
            let mut su = Array1::<f32>::zeros(cfg.n_feat);
            let mut su_bias = 0.0;
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                su += &self.yfeat.row(j);
                su_bias += self.ybias[j];
            }
            let total_cnt = if use_probe {
                let probe_items = &self.probe_items_by_user[u];
                for &j in probe_items {
                    su += &self.yfeat.row(j as usize);
                    su_bias += self.ybias[j as usize];
                }
                cnt + probe_items.len()
            } else {
                cnt
            };
            let norm = (total_cnt as f32).sqrt();
            su /= norm;
            self.ycache.row_mut(u).assign(&su);
            su_bias /= norm;
            self.ycache_bias[u] = su_bias;

            // Aggregated errors for NSVD1 updates
            let mut sum_err_q = Array1::<f32>::zeros(cfg.n_feat);
            let mut sum_err_q_day = if low_mem { Array1::zeros(0) } else { Array1::zeros(cfg.n_feat) };
            let mut sum_err = 0.0_f32;

            // Per-day NSVD1 cache
            let mut su_day = if low_mem { Array1::zeros(0) } else { Array1::zeros(cfg.n_feat) };
            let mut su_freq = Array1::<f32>::zeros(cfg.n_feat);
            let mut norm_day = 0.0_f32;
            let mut dr_idx = 0usize;
            let mut cur_freq_bin = 0usize;

            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                let r = tr.residuals[t];
                let day = tr.dates[t] as i32;
                let day16 = tr.dates[t];
                let b = self.time_bin(day);
                let dev = self.dev(u, day);

                let (day_start, day_stop) = if low_mem { (0, 0) } else { day_rngs[dr_idx] };

                if !low_mem && t == day_start {
                    sum_err_q_day.fill(0.0);
                    su_day.fill(0.0);
                    su_freq.fill(0.0);

                    let ud_idx_day = self.ud.index(u, day16);
                    let f = ud_idx_day.map_or(0, |idx| freq_bin(self.ud.day_cnts[idx], n_freq_bins));
                    cur_freq_bin = f;

                    for t_day in day_start..=day_stop {
                        let j = tr.item_idxs[t_day] as usize;
                        su_day += &self.yfeat_day.row(j);
                    }
                    let mut day_cnt = (day_stop - day_start + 1) as f32;
                    if use_probe {
                        if let Some(ud_idx) = ud_idx_day {
                            for &j in &self.probe_items_by_ud[ud_idx] {
                                su_day += &self.yfeat_day.row(j as usize);
                                day_cnt += 1.0;
                            }
                        }
                    }
                    norm_day = day_cnt.sqrt();
                    su_day /= norm_day;
                    su_freq /= norm_day;
                }

                let f = cur_freq_bin;

                // =====================================================
                // Forward pass: factor score
                // =====================================================
                let ud_idx = self.ud.index(u, day16);
                let bu_day = ud_idx.map_or(0.0, |idx| self.ubias_day[idx]);
                let bu_t = self.ubias[u] + self.but_bin[[u, b]] + self.alpha_u[u] * dev
                    + bu_day + self.ycache_bias[u];
                let bi_t = self.ibias[i] + self.bit_bin[[i, b]] + self.ibias_freq[[i, f]];
                let cu_t = self.cu[u] + ud_idx.map_or(0.0, |idx| self.cut[idx]);

                let pu = self.ufeat.row(u);
                let pu2 = self.ufeat2.row(u);
                let su_row = self.ycache.row(u);
                let qi = self.ifeat.row(i);
                let qf = self.ifeat_freq.row(i * n_freq_bins + f);

                let mut dot = 0.0;
                for k in 0..cfg.n_feat {
                    let qi_eff = qi[k] + qf[k];
                    let mut pu_eff = pu[k] + su_row[k] + dev * pu2[k];
                    if !low_mem {
                        if let Some(idx) = ud_idx { pu_eff += self.ufeat_day[[idx, k]]; }
                        pu_eff += su_day[k] + su_freq[k];
                    }
                    dot += pu_eff * qi_eff;
                }
                let bias_part = self.gbias + bu_t + bi_t * cu_t;

                // =====================================================
                // Forward pass: same-day neighborhood (no sampling)
                // =====================================================
                let (c_day_sum, w_day_sum, nn_day, nr_day) = if has_wd || has_cd {
                    let mut c_day_sum = 0.0_f32;
                    let mut w_day_sum = 0.0_f32;
                    let mut n_same_rated = 0_u32;
                    let mut n_same_total = 0_u32;
                    for ri in &self.user_rated[u] {
                        let j = ri.item as usize;
                        if j == i || ri.day != day16 { continue; }
                        n_same_rated += 1;
                        n_same_total += 1;
                        if has_cd { c_day_sum += self.c_day[i * ni + j]; }
                        if has_wd { w_day_sum += ri.r_minus_btilde * self.w_day[i * ni + j]; }
                    }
                    for ii in &self.user_implicit[u] {
                        let j = ii.item as usize;
                        if j == i || ii.day != day16 { continue; }
                        n_same_total += 1;
                        if has_cd { c_day_sum += self.c_day[i * ni + j]; }
                    }
                    let nnd = if n_same_total > 0 { (n_same_total as f32).powf(-0.5) } else { 0.0 };
                    let nrd = if n_same_rated > 0 { (n_same_rated as f32).powf(-0.5) } else { 0.0 };
                    (c_day_sum, w_day_sum, nnd, nrd)
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

                // =====================================================
                // Forward pass: global neighborhood (sampled)
                // =====================================================
                let (c_sum, w_sum) = if has_nbr {
                    let mut c_sum = 0.0_f32;
                    let mut w_sum = 0.0_f32;
                    for k in 0..iter_rated {
                        let ri_idx = if use_sampling { sampled_rated_idx[k] } else { k };
                        let ri = &self.user_rated[u][ri_idx];
                        let j = ri.item as usize;
                        if j == i { continue; }
                        c_sum += self.c[i * ni + j] * rated_scale;
                        w_sum += ri.r_minus_btilde * self.w[i * ni + j] * rated_scale;
                    }
                    for k in 0..iter_impl {
                        let ii_idx = if use_sampling { sampled_impl_idx[k] } else { k };
                        let ii = &self.user_implicit[u][ii_idx];
                        let j = ii.item as usize;
                        if j == i { continue; }
                        c_sum += self.c[i * ni + j] * impl_scale;
                    }
                    (c_sum, w_sum)
                } else {
                    (0.0, 0.0)
                };

                let nbr_part = nn * c_sum + nr * w_sum + nn_day * c_day_sum + nr_day * w_day_sum;

                // =====================================================
                // Error
                // =====================================================
                let score = bias_part + dot + nbr_part;

                let err = if let Some(ordinal_head) = &mut self.ordinal_head {
                    let y = tr.raw_ratings[t] as usize;
                    let (g_ds, g_t) = ordinal_head.grad(score, y);
                    ordinal_head.update_thresholds(g_t);
                    g_ds
                } else {
                    score - r
                };

                let err_b = err * cfg.w_bias;
                let err_d = err * cfg.w_factor;
                let err_n = err * cfg.w_nbr;

                // =====================================================
                // Bias SGD updates
                // =====================================================
                self.ubias[u] -= cfg.lr_ub * err_b;
                self.but_bin[[u, b]] -= cfg.lr_tu * err_b;
                self.alpha_u[u] -= cfg.lr_ta * err_b * dev;
                if let Some(idx) = ud_idx {
                    self.ubias_day[idx] -= cfg.lr_ubd * err_b;
                }
                self.ibias[i] -= cfg.lr_ib * (err_b * cu_t);
                self.bit_bin[[i, b]] -= cfg.lr_ti * (err_b * cu_t);
                self.ibias_freq[[i, f]] -= cfg.lr_ibf * (err_b * cu_t);
                self.cu[u] -= cfg.lr_cu * (err_b * bi_t + cfg.reg_cu * (self.cu[u] - 1.0));

                // =====================================================
                // Factor SGD updates
                // =====================================================
                let has_ud = !low_mem && cfg.lr_ud != 0.0;
                if !cfg.sum_err_bug { sum_err += err_b; }
                for k in 0..cfg.n_feat {
                    let qi_k = self.ifeat[[i, k]];
                    let qf_k = self.ifeat_freq[[i * n_freq_bins + f, k]];
                    let qi_eff_k = qi_k + qf_k;
                    let pu_k = self.ufeat[[u, k]];
                    let pu2_k = self.ufeat2[[u, k]];

                    sum_err_q[k] += err_d * qi_eff_k;
                    if !low_mem { sum_err_q_day[k] += err_d * qi_eff_k; }
                    if cfg.sum_err_bug { sum_err += err_b; }

                    self.ufeat[[u, k]] -= cfg.lr_u * (err_d * qi_eff_k + cfg.reg_u * pu_k);
                    self.ufeat2[[u, k]] -= cfg.lr_u2 * (err_d * qi_eff_k * dev + cfg.reg_u2 * pu2_k);

                    let pud_k = if has_ud {
                        if let Some(idx) = ud_idx {
                            let pud = self.ufeat_day[[idx, k]];
                            self.ufeat_day[[idx, k]] -= cfg.lr_ud * (err_d * qi_eff_k + cfg.reg_ud * pud);
                            pud
                        } else { 0.0 }
                    } else { 0.0 };

                    let pu_eff = pu_k + dev * pu2_k + pud_k + su[k]
                        + if low_mem { 0.0 } else { su_day[k] + su_freq[k] };
                    self.ifeat[[i, k]] -= cfg.lr_i * (err_d * pu_eff + cfg.reg_i * qi_k);
                    self.ifeat_freq[[i * n_freq_bins + f, k]] -= cfg.lr_iqf * (err_d * pu_eff + cfg.reg_iqf * qf_k);
                }

                // End of day: NSVD1-day factor updates
                if !low_mem && t == day_stop {
                    for t_day in day_start..=day_stop {
                        let j = tr.item_idxs[t_day] as usize;
                        for k in 0..cfg.n_feat {
                            let yj = self.yfeat_day[[j, k]];
                            self.yfeat_day[[j, k]] -=
                                cfg.lr_yd * (sum_err_q_day[k] / norm_day + cfg.reg_yd * yj);
                        }
                    }
                    dr_idx += 1;
                }

                // =====================================================
                // Global neighborhood SGD updates (sampled)
                // =====================================================
                if has_nbr {
                    for k in 0..iter_rated {
                        let ri_idx = if use_sampling { sampled_rated_idx[k] } else { k };
                        let ri = &self.user_rated[u][ri_idx];
                        let j = ri.item as usize;
                        if j == i { continue; }
                        let rmb = ri.r_minus_btilde;
                        let idx = i * ni + j;

                        let w_ij = self.w[idx];
                        self.w[idx] -= cfg.lr_w * (err_n * nr * rmb * rated_scale + cfg.reg_w * w_ij);

                        let c_ij = self.c[idx];
                        self.c[idx] -= cfg.lr_c * (err_n * nn * rated_scale + cfg.reg_c * c_ij);
                    }

                    for k in 0..iter_impl {
                        let ii_idx = if use_sampling { sampled_impl_idx[k] } else { k };
                        let ii = &self.user_implicit[u][ii_idx];
                        let j = ii.item as usize;
                        if j == i { continue; }
                        let idx = i * ni + j;

                        let c_ij = self.c[idx];
                        self.c[idx] -= cfg.lr_c * (err_n * nn * impl_scale + cfg.reg_c * c_ij);
                    }
                }

                // =====================================================
                // Same-day neighborhood SGD updates (no sampling)
                // =====================================================
                if has_wd || has_cd {
                    for ri in &self.user_rated[u] {
                        let j = ri.item as usize;
                        if j == i || ri.day != day16 { continue; }
                        let idx = i * ni + j;
                        if has_wd {
                            let wd = self.w_day[idx];
                            self.w_day[idx] -= cfg.lr_w_day * (err_n * nr_day * ri.r_minus_btilde + cfg.reg_w_day * wd);
                        }
                        if has_cd {
                            let cd = self.c_day[idx];
                            self.c_day[idx] -= cfg.lr_c_day * (err_n * nn_day + cfg.reg_c_day * cd);
                        }
                    }
                    for ii in &self.user_implicit[u] {
                        let j = ii.item as usize;
                        if j == i || ii.day != day16 { continue; }
                        let idx = i * ni + j;
                        if has_cd {
                            let cd = self.c_day[idx];
                            self.c_day[idx] -= cfg.lr_c_day * (err_n * nn_day + cfg.reg_c_day * cd);
                        }
                    }
                }
            }

            // NSVD1 factor & bias updates (end of user)
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                for k in 0..cfg.n_feat {
                    let yj = self.yfeat[[j, k]];
                    self.yfeat[[j, k]] -= cfg.lr_y * (sum_err_q[k] / norm + cfg.reg_y * yj);
                }
                self.ybias[j] -= cfg.lr_yb * (sum_err / norm);
            }
        }

        self.rebuild_ycache(tr, pr);
        if !low_mem { self.rebuild_ycache_day(tr, pr); }

        if let Some(ordinal_head) = &mut self.ordinal_head {
            ordinal_head.enforce_sorted_with_gap();
        }
    }
}

// ---------------------------------------------------------------------------
// Temporary pub(crate) accessors for the parallel tsvdx4p variant.
// TODO: remove when tsvdx4p is refactored away from direct field access.
// ---------------------------------------------------------------------------

macro_rules! txm_accessors {
    ($( $get:ident , $set:ident : $ty:ty );* $(;)?) => {
        #[allow(dead_code)]
        impl TxModel {
            $(
                #[inline] pub(crate) fn $get(&self) -> &$ty { &self.$get }
                #[inline] pub(crate) fn $set(&mut self) -> &mut $ty { &mut self.$get }
            )*
        }
    };
}

txm_accessors! {
    cfg,         cfg_mut:         TxConfig;
    gbias,       gbias_mut:       f32;
    ubias,       ubias_mut:       Array1<f32>;
    ibias,       ibias_mut:       Array1<f32>;
    ufeat,       ufeat_mut:       Array2<f32>;
    ufeat2,      ufeat2_mut:      Array2<f32>;
    ifeat,       ifeat_mut:       Array2<f32>;
    yfeat,       yfeat_mut:       Array2<f32>;
    ycache,      ycache_mut:      Array2<f32>;
    yfeat_day,   yfeat_day_mut:   Array2<f32>;
    day_range,   day_range_mut:   i32;
    tu_mean,     tu_mean_mut:     Array1<f32>;
    alpha_u,     alpha_u_mut:     Array1<f32>;
    but_bin,     but_bin_mut:     Array2<f32>;
    bit_bin,     bit_bin_mut:     Array2<f32>;
    ybias,       ybias_mut:       Array1<f32>;
    ycache_bias, ycache_bias_mut: Array1<f32>;
    ud,          ud_mut:          SparseUD;
    ubias_day,   ubias_day_mut:   Vec<f32>;
    ufeat_day,   ufeat_day_mut:   Array2<f32>;
    ycache_day,  ycache_day_mut:  Array2<f32>;
    ibias_freq,  ibias_freq_mut:  Array2<f32>;
    ifeat_freq,  ifeat_freq_mut:  Array2<f32>;
    cu,          cu_mut:          Array1<f32>;
    cut,         cut_mut:         Vec<f32>;
    probe_items_by_user, probe_items_by_user_mut: Vec<Vec<u32>>;
    probe_items_by_ud,   probe_items_by_ud_mut:   Vec<Vec<u32>>;
    user_offsets, user_offsets_mut: Array1<usize>;
}
