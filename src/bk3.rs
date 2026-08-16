// BK3 integrated model.
// Ref: Piotte & Chabbert 2009, "The Pragmatic Theory solution to the Netflix Grand Prize"
// (Section 3.2, equations 18–20):
//
// dev̂(u,t)  = k1·sign(t−t̄_u)·|t−t̄_u|^β − dev̄_u
// z(u,m,t)  = μ
//           + bi[m] + bit_bin[m,t₃₀] + bmf[m,f₃₀]
//           + bu[u] + bu1[u]·dev̂ + k2·bu2[u,t]
//           + Σf [q[m,f]+qt[m,t₈,f]+qf[m,f₈,f]]
//              ·[p[u,f]+p1[u,f]·dev̂+h[u,t]+su[u,f]]
//           + (1/√|Rᵏ(m;u)|)·Σ_{j∈Rᵏ} (r−bl₁)·w[m,rank]
//           + (1/√|Nᵏ(m;u)|)·Σ_{j∈Nᵏ} c[m,rank]
//
// BK3 adds over BK1:
//   bmf[m, f₃₀]        — frequency-dependent item bias (30 quantile bins)
//   qt[m, t₈, f]        — time-dependent item factor correction (8 uniform bins)
//   qf[m, f₈, f]        — frequency-dependent item factor correction (8 quantile bins)
//
// Frequency f(u,t) = number of ratings user u gave on the same day as t.
// Bin boundaries are computed via quantile binning over per-rating frequency values.
//
// LR decays by factor (1 − Δγ) at the start of each epoch.

use crate::{Dataset, MaskedDataset, Regressor, calc_gbias, calc_user_offsets, get_users, rand_array2};
use crate::bk1::precompute_neighbors;
use crate::tx::SparseUD;
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2, Array3};
use rand::{SeedableRng, rngs::StdRng};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SparseUFreq
// ---------------------------------------------------------------------------
//
// Stores per-(user, day) frequency bins.
// f(u,t) = number of ratings by user u on the same day as rating t (in tr).
// Bin boundaries are quantile-based so each bin covers ~equal training samples.

struct SparseUFreq {
    starts:    Vec<usize>,
    days:      Vec<i16>,
    fbin_bias: Vec<u8>,  // freq bin ∈ [0, n_freq_bins_bias)  — for bmf
    fbin_fact: Vec<u8>,  // freq bin ∈ [0, n_freq_bins_fact)  — for qf
}

#[inline]
fn freq_bin(freq: u32, boundaries: &[u32]) -> usize {
    // Returns k such that boundaries[k-1] < freq <= boundaries[k]  (0-indexed)
    boundaries.partition_point(|&b| b < freq)
}

impl SparseUFreq {
    fn new(tr: &Dataset, n_bins_bias: usize, n_bins_fact: usize) -> Self {
        let n_users = tr.n_users;

        // Count ratings per (user, day)
        let mut ud_count: HashMap<(u32, i16), u32> = HashMap::new();
        for t in 0..tr.n_ratings {
            *ud_count
                .entry((tr.user_idxs[t] as u32, tr.dates[t]))
                .or_insert(0) += 1;
        }

        // Collect per-rating frequency for quantile boundary computation.
        // "Each bin contains roughly the same number of samples" (paper).
        let mut freq_per_rating: Vec<u32> = Vec::with_capacity(tr.n_ratings);
        for t in 0..tr.n_ratings {
            freq_per_rating.push(
                *ud_count.get(&(tr.user_idxs[t] as u32, tr.dates[t])).unwrap(),
            );
        }
        freq_per_rating.sort_unstable();

        // (n_bins - 1) quantile cut points → n_bins bins
        let len = freq_per_rating.len();
        let bnd_bias: Vec<u32> = (1..n_bins_bias)
            .map(|i| freq_per_rating[(i * len) / n_bins_bias])
            .collect();
        let bnd_fact: Vec<u32> = (1..n_bins_fact)
            .map(|i| freq_per_rating[(i * len) / n_bins_fact])
            .collect();

        // Build per-user sorted (day, fbin_bias, fbin_fact) lists
        let mut per_user: Vec<Vec<(i16, u8, u8)>> = vec![Vec::new(); n_users];
        for (&(u, day), &cnt) in &ud_count {
            let fb_bias = freq_bin(cnt, &bnd_bias) as u8;
            let fb_fact = freq_bin(cnt, &bnd_fact) as u8;
            per_user[u as usize].push((day, fb_bias, fb_fact));
        }

        let mut starts    = Vec::with_capacity(n_users + 1);
        let mut days      = Vec::new();
        let mut fbin_bias = Vec::new();
        let mut fbin_fact = Vec::new();

        for u in 0..n_users {
            starts.push(days.len());
            per_user[u].sort_unstable_by_key(|&(d, _, _)| d);
            for (d, fb_b, fb_f) in &per_user[u] {
                days.push(*d);
                fbin_bias.push(*fb_b);
                fbin_fact.push(*fb_f);
            }
        }
        starts.push(days.len());

        Self { starts, days, fbin_bias, fbin_fact }
    }

    /// Returns (fbin_bias, fbin_fact) for (user, day), or (0, 0) if not in training data.
    #[inline]
    fn get(&self, u: usize, day: i16) -> (u8, u8) {
        let (s, e) = (self.starts[u], self.starts[u + 1]);
        match self.days[s..e].binary_search(&day) {
            Ok(i) => (self.fbin_bias[s + i], self.fbin_fact[s + i]),
            Err(_) => (0, 0),
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Bk3Config {
    pub n_feat:        usize,
    pub n_epochs:      usize,
    pub seed:          u64,
    pub shuffle_users: bool,

    // Temporal / frequency bins
    pub n_time_bins:      usize,  // t₃₀: bins for item time-bias
    pub n_time_bins_fact: usize,  // t₈:  bins for item factor time-correction qt
    pub n_freq_bins_bias: usize,  // f₃₀: bins for item frequency-bias bmf
    pub n_freq_bins_fact: usize,  // f₈:  bins for item factor freq-correction qf

    // Temporal dynamics
    pub beta:     f32,    // k4: dev exponent (0.4)
    pub k1:       f32,    // dev scaling (~0.036–0.043)
    pub k2:       f32,    // k_s: per-day user bias scale (~0.95–1.0)
    pub dev_mean: bool,   // subtract per-user mean of dev̂

    // k-NN
    pub k_neighbors: usize,
    pub alpha_rho:   f32,

    // Baseline₁ shrinkage (for r − bl₁ in w-term)
    pub lambda1: f32,
    pub lambda2: f32,

    // Learning rates
    pub lr_bias:  f32,   // γ1:  bu, bu1, bu2, bi, bit_bin, bmf
    pub lr_fact:  f32,   // γ2:  p, p1, q, y
    pub lr_qt:    f32,   // γ11: qt (time-dep item factor correction)
    pub lr_qf:    f32,   // γ12: qf (freq-dep item factor correction)
    pub lr_nbr:   f32,   // γ3:  w, c
    pub lr_h:     f32,   // γ9:  h(u,t) per-day user correction
    pub lr_decay: f32,   // Δγ

    // Regularisation
    pub reg_bias: f32,   // λ6
    pub reg_fact: f32,   // λ7
    pub reg_qt:   f32,   // λ11
    pub reg_qf:   f32,   // λ12
    pub reg_nbr:  f32,   // λ8
    pub reg_h:    f32,   // λ9
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Bk3Model {
    cfg:   Bk3Config,
    gbias: f32,

    // Biases
    bu:      Array1<f32>,    // [n_users]
    bu1:     Array1<f32>,    // [n_users]
    bu2:     Vec<f32>,       // [n_ud]
    bi:      Array1<f32>,    // [n_items]
    bit_bin: Array2<f32>,    // [n_items, n_time_bins]
    bmf:     Array2<f32>,    // [n_items, n_freq_bins_bias]   ← BK3 new

    // Latent factors
    p:      Array2<f32>,     // [n_users, n_feat]
    p1:     Array2<f32>,     // [n_users, n_feat]
    h:      Vec<f32>,        // [n_ud]
    q:      Array2<f32>,     // [n_items, n_feat]
    qt:     Array3<f32>,     // [n_items, n_time_bins_fact, n_feat]   ← BK3 new
    qf:     Array3<f32>,     // [n_items, n_freq_bins_fact, n_feat]   ← BK3 new
    y:      Array2<f32>,     // [n_items, n_feat]
    ycache: Array2<f32>,     // [n_users, n_feat]  — (1/√|N(u)|)·Σ y[j]

    // k-NN
    neighbors: Vec<Vec<u16>>,
    w:  Vec<f32>,            // [n_items * k_neighbors]
    c:  Vec<f32>,            // [n_items * k_neighbors]

    // Per-user lookup tables
    user_rated_items:    Vec<Vec<u16>>,
    user_rated_rmbli:    Vec<Vec<f32>>,
    user_nu:             Vec<Vec<u16>>,
    probe_items_by_user: Vec<Vec<u16>>,

    // Temporal / frequency state
    tu_mean:      Array1<f32>,
    dev_mean_u:   Array1<f32>,
    day_range:    i32,
    ud:           SparseUD,
    ufreq:        SparseUFreq,
    user_offsets: Array1<usize>,

    lr_scale: f32,
}

impl Bk3Model {
    #[inline]
    fn dev(&self, u: usize, day: i32) -> f32 {
        let dt = day as f32 - self.tu_mean[u];
        let raw = if dt == 0.0 { 0.0 } else { dt.signum() * dt.abs().powf(self.cfg.beta) };
        self.cfg.k1 * raw - self.dev_mean_u[u]
    }

    /// Uniform time bin for item time-bias bit_bin (t₃₀).
    #[inline]
    fn time_bin(&self, day: i32) -> usize {
        let b = (day as i64 * self.cfg.n_time_bins as i64 / self.day_range as i64) as usize;
        b.min(self.cfg.n_time_bins - 1)
    }

    /// Uniform time bin for item factor correction qt (t₈).
    #[inline]
    fn time_bin_fact(&self, day: i32) -> usize {
        let b = (day as i64 * self.cfg.n_time_bins_fact as i64 / self.day_range as i64) as usize;
        b.min(self.cfg.n_time_bins_fact - 1)
    }

    /// Rebuild ycache: ycache[u] = (1/√|N(u)|) · Σ_{j∈N(u)} y[j]
    fn rebuild_ycache(&mut self, tr: &Dataset, pr: &MaskedDataset) {
        self.ycache.fill(0.0);
        for t in 0..tr.n_ratings {
            let u = tr.user_idxs[t] as usize;
            let j = tr.item_idxs[t] as usize;
            let mut row = self.ycache.row_mut(u);
            row += &self.y.row(j);
        }
        for t in 0..pr.n_ratings {
            let u = pr.user_idxs[t] as usize;
            let j = pr.item_idxs[t] as usize;
            let mut row = self.ycache.row_mut(u);
            row += &self.y.row(j);
        }
        for u in 0..tr.n_users {
            let cnt = (tr.user_cnts[u] + pr.user_cnts[u]) as f32;
            if cnt > 0.0 {
                let mut row = self.ycache.row_mut(u);
                row /= cnt.sqrt();
            }
        }
    }

    #[inline]
    fn nbr_score(&self, u: usize, i: usize) -> f32 {
        let nbrs = &self.neighbors[i];
        if nbrs.is_empty() { return 0.0; }
        let k = nbrs.len();
        let mut w_sum = 0.0f32;
        let mut c_sum = 0.0f32;
        let mut n_rk  = 0usize;
        let mut n_nk  = 0usize;

        for (rank, &j) in nbrs.iter().enumerate() {
            if self.user_nu[u].binary_search(&j).is_ok() {
                c_sum += self.c[i * k + rank];
                n_nk += 1;
                if let Ok(pos) = self.user_rated_items[u].binary_search(&j) {
                    w_sum += self.user_rated_rmbli[u][pos] * self.w[i * k + rank];
                    n_rk += 1;
                }
            }
        }
        let nr = if n_rk > 0 { (n_rk as f32).powf(-0.5) } else { 0.0 };
        let nn = if n_nk > 0 { (n_nk as f32).powf(-0.5) } else { 0.0 };
        nr * w_sum + nn * c_sum
    }
}

// ---------------------------------------------------------------------------
// Regressor impl
// ---------------------------------------------------------------------------

impl Regressor for Bk3Model {
    type Config = Bk3Config;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let n_users = tr.n_users;
        let n_items = tr.n_items;
        let nf      = cfg.n_feat;

        // ── Mean rating date per user + day range ────────────────────────────
        let mut tu_mean   = Array1::<f32>::zeros(n_users);
        let mut day_range = 0i32;
        for t in 0..tr.n_ratings {
            let u = tr.user_idxs[t] as usize;
            tu_mean[u] += tr.dates[t] as f32;
            day_range = day_range.max(tr.dates[t] as i32 + 1);
        }
        for u in 0..n_users {
            let cnt = tr.user_cnts[u];
            if cnt > 0 { tu_mean[u] /= cnt as f32; }
        }

        let ud           = SparseUD::new(tr, pr);
        let n_ud         = ud.n_total();
        let gbias        = calc_gbias(tr);
        let user_offsets = calc_user_offsets(tr);

        // ── Per-user mean of dev̂ (if dev_mean=true) ─────────────────────────
        let dev_mean_u = if cfg.dev_mean {
            let (k1, beta) = (cfg.k1, cfg.beta);
            let mut sums = vec![0.0f64; n_users];
            let mut cnts = vec![0u32; n_users];
            for t in 0..tr.n_ratings {
                let u  = tr.user_idxs[t] as usize;
                let dt = tr.dates[t] as f32 - tu_mean[u];
                let raw = if dt == 0.0 { 0.0 } else { dt.signum() * dt.abs().powf(beta) };
                sums[u] += (k1 * raw) as f64;
                cnts[u] += 1;
            }
            let mut dm = Array1::<f32>::zeros(n_users);
            for u in 0..n_users {
                if cnts[u] > 0 { dm[u] = (sums[u] / cnts[u] as f64) as f32; }
            }
            dm
        } else {
            Array1::zeros(n_users)
        };

        // ── Baseline₁ (3-iter ALS) for r − bl₁ in neighborhood w-term ─────
        let mu = gbias;
        let mut btilde_m = vec![0.0f32; n_items];
        let mut btilde_u = vec![0.0f32; n_users];
        for _iter in 0..3 {
            let mut sum_m = vec![0.0f64; n_items];
            let mut cnt_m = vec![0u32; n_items];
            for t in 0..tr.n_ratings {
                let i = tr.item_idxs[t] as usize;
                let u = tr.user_idxs[t] as usize;
                sum_m[i] += tr.raw_ratings[t] as f64 - mu as f64 - btilde_u[u] as f64;
                cnt_m[i] += 1;
            }
            for i in 0..n_items {
                btilde_m[i] = (sum_m[i] / (cnt_m[i] as f64 + cfg.lambda1 as f64)) as f32;
            }
            let mut sum_u = vec![0.0f64; n_users];
            let mut cnt_u = vec![0u32; n_users];
            for t in 0..tr.n_ratings {
                let i = tr.item_idxs[t] as usize;
                let u = tr.user_idxs[t] as usize;
                sum_u[u] += tr.raw_ratings[t] as f64 - mu as f64 - btilde_m[i] as f64;
                cnt_u[u] += 1;
            }
            for u in 0..n_users {
                btilde_u[u] = (sum_u[u] / (cnt_u[u] as f64 + cfg.lambda2 as f64)) as f32;
            }
        }

        // ── Per-user R(u): sorted by item, with r − bl₁ ─────────────────────
        let mut user_rated_items: Vec<Vec<u16>> = vec![Vec::new(); n_users];
        let mut user_rated_rmbli: Vec<Vec<f32>> = vec![Vec::new(); n_users];
        for u in 0..n_users {
            let s = user_offsets[u];
            let e = user_offsets[u + 1];
            let mut pairs: Vec<(u16, f32)> = (s..e).map(|t| {
                let i  = tr.item_idxs[t] as u16;
                let r  = tr.raw_ratings[t] as f32;
                let bl = mu + btilde_m[i as usize] + btilde_u[u];
                (i, r - bl)
            }).collect();
            pairs.sort_unstable_by_key(|&(i, _)| i);
            user_rated_items[u] = pairs.iter().map(|&(i, _)| i).collect();
            user_rated_rmbli[u] = pairs.iter().map(|&(_, v)| v).collect();
        }

        // ── Implicit items from pr (probe, item-sorted — linear scan) ────────
        let mut implicit_per_user: Vec<Vec<u16>> = vec![Vec::new(); n_users];
        for t in 0..pr.n_ratings {
            let u = pr.user_idxs[t] as usize;
            let i = pr.item_idxs[t] as u16;
            if user_rated_items[u].binary_search(&i).is_err() {
                implicit_per_user[u].push(i);
            }
        }
        let mut probe_items_by_user: Vec<Vec<u16>> = vec![Vec::new(); n_users];
        for u in 0..n_users {
            implicit_per_user[u].sort_unstable();
            implicit_per_user[u].dedup();
            probe_items_by_user[u] = implicit_per_user[u].clone();
        }

        // ── N(u) = R(u) ∪ implicit, sorted ──────────────────────────────────
        let mut user_nu: Vec<Vec<u16>> = vec![Vec::new(); n_users];
        for u in 0..n_users {
            let mut nu = user_rated_items[u].clone();
            nu.extend_from_slice(&implicit_per_user[u]);
            nu.sort_unstable();
            user_nu[u] = nu;
        }

        // ── k-NN neighbor precomputation ────────────────────────────────────
        let k = cfg.k_neighbors;
        let neighbors = if k > 0 {
            precompute_neighbors(&tr.name, k, cfg.alpha_rho, n_items)
        } else {
            vec![Vec::new(); n_items]
        };
        let w = vec![0.0f32; n_items * k];
        let c = vec![0.0f32; n_items * k];

        // ── Frequency bin structure ──────────────────────────────────────────
        let ufreq = SparseUFreq::new(tr, cfg.n_freq_bins_bias, cfg.n_freq_bins_fact);

        // ── Latent factor initialisation ─────────────────────────────────────
        let sigma = 0.01f32;
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let p  = rand_array2(n_users, nf, &mut rng, sigma);
        let p1 = Array2::zeros((n_users, nf));
        let q  = rand_array2(n_items, nf, &mut rng, sigma);
        let y  = Array2::zeros((n_items, nf));

        let mut model = Self {
            cfg,
            gbias,
            bu:      Array1::zeros(n_users),
            bu1:     Array1::zeros(n_users),
            bu2:     vec![0.0; n_ud],
            bi:      Array1::zeros(n_items),
            bit_bin: Array2::zeros((n_items, cfg.n_time_bins)),
            bmf:     Array2::zeros((n_items, cfg.n_freq_bins_bias)),
            p, p1,
            h: vec![0.0; n_ud],
            q,
            qt: Array3::zeros((n_items, cfg.n_time_bins_fact, nf)),
            qf: Array3::zeros((n_items, cfg.n_freq_bins_fact, nf)),
            y,
            ycache: Array2::zeros((n_users, nf)),
            neighbors, w, c,
            user_rated_items,
            user_rated_rmbli,
            user_nu,
            probe_items_by_user,
            tu_mean,
            dev_mean_u,
            day_range,
            ud,
            ufreq,
            user_offsets,
            lr_scale: 1.0,
        };
        model.rebuild_ycache(tr, pr);
        model
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let dev    = self.dev(u, day);
        let day16  = day as i16;
        let tb30   = self.time_bin(day);
        let tb8    = self.time_bin_fact(day);
        let ud_idx = self.ud.index(u, day16);
        let (fb30, fb8) = self.ufreq.get(u, day16);

        let bu2_val = ud_idx.map_or(0.0, |idx| self.bu2[idx]);
        let h_val   = ud_idx.map_or(0.0, |idx| self.h[idx]);

        let mut score = self.gbias
            + self.bu[u] + self.bu1[u] * dev + self.cfg.k2 * bu2_val
            + self.bi[i] + self.bit_bin[[i, tb30]] + self.bmf[[i, fb30 as usize]];

        let su = self.ycache.row(u);
        for f in 0..self.cfg.n_feat {
            let q_eff = self.q[[i, f]] + self.qt[[i, tb8, f]] + self.qf[[i, fb8 as usize, f]];
            let pu    = self.p[[u, f]] + self.p1[[u, f]] * dev + h_val + su[f];
            score += q_eff * pu;
        }

        score + self.nbr_score(u, i)
    }

    fn fit_epoch(&mut self, tr: &Dataset, pr: &MaskedDataset, epoch: usize) {
        if epoch > 0 { self.lr_scale *= 1.0 - self.cfg.lr_decay; }

        let s      = self.lr_scale;
        let cfg    = self.cfg;
        let nf     = cfg.n_feat;
        let k      = cfg.k_neighbors;
        let lr_b   = cfg.lr_bias * s;
        let lr_f   = cfg.lr_fact * s;
        let lr_qt  = cfg.lr_qt   * s;
        let lr_qf  = cfg.lr_qf   * s;
        let lr_n   = cfg.lr_nbr  * s;
        let lr_h   = cfg.lr_h    * s;

        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        let mut nbr_rmb:  Vec<Option<f32>> = vec![None; k];
        let mut nbr_in_n: Vec<bool>        = vec![false; k];

        for &u in crate::progress!(users.iter()) {
            let start = self.user_offsets[u];
            let end   = self.user_offsets[u + 1];
            if start == end { continue; }

            // ── Compute su (NSVD1 contribution) from scratch ─────────────────
            let cnt_r = (end - start) as f32;
            let cnt_p = self.probe_items_by_user[u].len() as f32;
            let norm  = (cnt_r + cnt_p).sqrt();

            let mut su = Array1::<f32>::zeros(nf);
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                su += &self.y.row(j);
            }
            for &j in &self.probe_items_by_user[u] {
                su += &self.y.row(j as usize);
            }
            if norm > 0.0 { su /= norm; }
            self.ycache.row_mut(u).assign(&su);

            // Accumulate per-user gradient for y-update (uses q_eff, not just q)
            let mut sum_err_q_eff = Array1::<f32>::zeros(nf);

            for t in start..end {
                let i     = tr.item_idxs[t] as usize;
                let r     = tr.residuals[t];
                let day   = tr.dates[t] as i32;
                let day16 = tr.dates[t];
                let dev   = self.dev(u, day);
                let tb30  = self.time_bin(day);
                let tb8   = self.time_bin_fact(day);
                let ud_idx = self.ud.index(u, day16);
                let (fb30, fb8) = self.ufreq.get(u, day16);

                let bu2_val = ud_idx.map_or(0.0, |idx| self.bu2[idx]);
                let h_val   = ud_idx.map_or(0.0, |idx| self.h[idx]);

                // ── Forward: bias ─────────────────────────────────────────────
                let mut score = self.gbias
                    + self.bu[u] + self.bu1[u] * dev + cfg.k2 * bu2_val
                    + self.bi[i] + self.bit_bin[[i, tb30]] + self.bmf[[i, fb30 as usize]];

                // ── Forward: latent factors ───────────────────────────────────
                // Also accumulate sum_q_eff for the h gradient: ∂z/∂h = Σ_f q_eff[f]
                let mut sum_q_eff = 0.0f32;
                for f in 0..nf {
                    let q_eff = self.q[[i, f]] + self.qt[[i, tb8, f]] + self.qf[[i, fb8 as usize, f]];
                    let pu    = self.p[[u, f]] + self.p1[[u, f]] * dev + h_val + su[f];
                    score += q_eff * pu;
                    sum_q_eff += q_eff;
                }

                // ── Forward: k-NN neighborhood ───────────────────────────────
                let mut w_sum = 0.0f32;
                let mut c_sum = 0.0f32;
                let mut n_rk  = 0usize;
                let mut n_nk  = 0usize;

                if k > 0 {
                    let nbrs = &self.neighbors[i];
                    for (rank, &j) in nbrs.iter().enumerate() {
                        nbr_in_n[rank] = false;
                        nbr_rmb[rank]  = None;
                        if self.user_nu[u].binary_search(&j).is_ok() {
                            nbr_in_n[rank] = true;
                            n_nk += 1;
                            c_sum += self.c[i * k + rank];
                            if let Ok(pos) = self.user_rated_items[u].binary_search(&j) {
                                let rmb = self.user_rated_rmbli[u][pos];
                                nbr_rmb[rank] = Some(rmb);
                                n_rk += 1;
                                w_sum += rmb * self.w[i * k + rank];
                            }
                        }
                    }
                }
                let nr = if n_rk > 0 { (n_rk as f32).powf(-0.5) } else { 0.0 };
                let nn = if n_nk > 0 { (n_nk as f32).powf(-0.5) } else { 0.0 };
                score += nr * w_sum + nn * c_sum;

                let err = score - r;

                // ── Backward: biases ──────────────────────────────────────────
                self.bu[u]                   -= lr_b * (err + cfg.reg_bias * self.bu[u]);
                self.bu1[u]                  -= lr_b * (err * dev + cfg.reg_bias * self.bu1[u]);
                self.bi[i]                   -= lr_b * (err + cfg.reg_bias * self.bi[i]);
                self.bit_bin[[i, tb30]]      -= lr_b * (err + cfg.reg_bias * self.bit_bin[[i, tb30]]);
                self.bmf[[i, fb30 as usize]] -= lr_b * (err + cfg.reg_bias * self.bmf[[i, fb30 as usize]]);
                if let Some(idx) = ud_idx {
                    self.bu2[idx] -= lr_b * (err * cfg.k2 + cfg.reg_bias * self.bu2[idx]);
                }

                // ── Backward: h(u,t) ─────────────────────────────────────────
                if lr_h != 0.0 {
                    if let Some(idx) = ud_idx {
                        self.h[idx] -= lr_h * (err * sum_q_eff + cfg.reg_h * self.h[idx]);
                    }
                }

                // ── Backward: latent factors ──────────────────────────────────
                for f in 0..nf {
                    let q_f   = self.q[[i, f]];
                    let qt_f  = self.qt[[i, tb8, f]];
                    let qf_f  = self.qf[[i, fb8 as usize, f]];
                    let q_eff = q_f + qt_f + qf_f;
                    let p_f   = self.p[[u, f]];
                    let p1_f  = self.p1[[u, f]];
                    let pu_eff = p_f + p1_f * dev + h_val + su[f];

                    // Accumulated for y-update: gradient uses q_eff, not just q
                    sum_err_q_eff[f] += err * q_eff;

                    self.p[[u, f]]              -= lr_f  * (err * q_eff       + cfg.reg_fact * p_f);
                    self.p1[[u, f]]             -= lr_f  * (err * q_eff * dev + cfg.reg_fact * p1_f);
                    self.q[[i, f]]              -= lr_f  * (err * pu_eff      + cfg.reg_fact * q_f);
                    self.qt[[i, tb8, f]]        -= lr_qt * (err * pu_eff      + cfg.reg_qt   * qt_f);
                    self.qf[[i, fb8 as usize, f]] -= lr_qf * (err * pu_eff   + cfg.reg_qf   * qf_f);
                }

                // ── Backward: k-NN weights ────────────────────────────────────
                if k > 0 {
                    let nbrs = &self.neighbors[i];
                    for (rank, _) in nbrs.iter().enumerate() {
                        if nbr_in_n[rank] {
                            let nb_idx = i * k + rank;
                            self.c[nb_idx] -= lr_n * (err * nn + cfg.reg_nbr * self.c[nb_idx]);
                            if let Some(rmb) = nbr_rmb[rank] {
                                self.w[nb_idx] -= lr_n * (err * nr * rmb + cfg.reg_nbr * self.w[nb_idx]);
                            }
                        }
                    }
                }
            } // end per-rating loop

            // ── y-factor update (accumulated gradient) ────────────────────────
            // ∂L/∂y[j,f] = sum_err_q_eff[f] / norm  (for each j ∈ R(u))
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                for f in 0..nf {
                    let yj = self.y[[j, f]];
                    self.y[[j, f]] -= lr_f * (sum_err_q_eff[f] / norm + cfg.reg_fact * yj);
                }
            }
        } // end user loop

        self.rebuild_ycache(tr, pr);
    }
}
