// BellKor neighborhood model with learned weights (eq 18) and per-user time decay.
// Ref: Koren 2010, "Factorization Meets the Neighborhood".
//
// r̂_ui = b_ui + |N(u)|^{-1/2} Σ_{j∈N(u)} e^{-β_u·|t_ui-t_uj|} · c_ij
//       + |R(u)|^{-1/2} Σ_{j∈R(u)} e^{-β_u·|t_ui-t_uj|} · (r_uj - b̃_uj) · w_ij
//
// Baseline b_ui follows eq 10: μ + (bu + α_u·dev + but) + (bi + bit_bin)·(cu + cut).
// b̃_uj are simple damped baselines: μ + b̃_i + b̃_u (eqs 2–3).
// Three training modes: Sequential, Hogwild (item-parallel + atomic user params),
// or TwoPass (item-parallel for w/c, then user-sequential for baseline).

use crate::{Dataset, MaskedDataset, Regressor, calc_gbias, calc_user_offsets, get_users};
use crate::tx::{SparseUD, RatedItem, ImplicitItem};
use indicatif::{ProgressIterator, ProgressStyle};
use ndarray::Array1;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// Parallel mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum ParallelMode {
    /// Original user-sequential SGD
    Sequential,
    /// Item-parallel with Hogwild-style atomic updates on user params
    Hogwild,
    /// Item-parallel for neighborhood/item params, then user-sequential for baseline params
    TwoPass,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct BknbrxConfig {
    pub n_epochs: usize,
    pub seed: u64,
    pub shuffle_users: bool,
    pub parallel_mode: ParallelMode,
    /// Thread count for Hogwild/TwoPass parallel modes (0 = use rayon default).
    pub n_threads: usize,

    // Baseline parameters (eq 10)
    pub n_time_bins: usize,
    pub beta: f32,

    pub lr_bu: f32,
    pub lr_but: f32,
    pub lr_alpha: f32,
    pub lr_bi: f32,
    pub lr_bit: f32,
    pub lr_cu: f32,
    pub lr_cut: f32,

    pub reg_bu: f32,
    pub reg_but: f32,
    pub reg_alpha: f32,
    pub reg_bi: f32,
    pub reg_bit: f32,
    pub reg_cu: f32,
    pub reg_cut: f32,

    // Neighborhood parameters
    pub max_neighbors: usize, // 0 = use all neighbors (no sampling)
    pub lr_w: f32,
    pub lr_c: f32,
    pub lr_beta_u: f32,
    pub reg_w: f32,
    pub reg_c: f32,
    pub reg_beta_u: f32,

    // Same-day neighborhood parameters (0 lr = disabled, no allocation)
    pub lr_w_day: f32,
    pub lr_c_day: f32,
    pub reg_w_day: f32,
    pub reg_c_day: f32,

    // Precomputed baseline damping
    pub lambda1: f32, // item damping (25)
    pub lambda2: f32, // user damping (10)
}

// ---------------------------------------------------------------------------
// Atomic f32 helpers (for Hogwild)
// ---------------------------------------------------------------------------

#[inline]
fn atomic_sub_f32(atom: &AtomicU32, delta: f32) {
    let mut old_bits = atom.load(Ordering::Relaxed);
    loop {
        let old_val = f32::from_bits(old_bits);
        let new_val = old_val - delta;
        match atom.compare_exchange_weak(
            old_bits, new_val.to_bits(),
            Ordering::Relaxed, Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => old_bits = x,
        }
    }
}

#[inline]
fn atomic_load_f32(atom: &AtomicU32) -> f32 {
    f32::from_bits(atom.load(Ordering::Relaxed))
}

#[inline]
fn atomic_clamp_nonneg(atom: &AtomicU32) {
    let mut old_bits = atom.load(Ordering::Relaxed);
    loop {
        let val = f32::from_bits(old_bits);
        if val >= 0.0 { break; }
        match atom.compare_exchange_weak(
            old_bits, 0.0_f32.to_bits(),
            Ordering::Relaxed, Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => old_bits = x,
        }
    }
}

fn vec_to_atomic(v: &[f32]) -> Vec<AtomicU32> {
    v.iter().map(|&x| AtomicU32::new(x.to_bits())).collect()
}

fn atomic_to_vec(a: &[AtomicU32]) -> Vec<f32> {
    a.iter().map(|x| f32::from_bits(x.load(Ordering::Relaxed))).collect()
}

fn array1_to_atomic(a: &Array1<f32>) -> Vec<AtomicU32> {
    a.iter().map(|&x| AtomicU32::new(x.to_bits())).collect()
}

fn atomic_to_array1(a: &[AtomicU32]) -> Array1<f32> {
    Array1::from_vec(a.iter().map(|x| f32::from_bits(x.load(Ordering::Relaxed))).collect())
}

// ---------------------------------------------------------------------------
// Thread pool dispatch
// ---------------------------------------------------------------------------

/// Run `f` inside a custom rayon pool of `n_threads` workers.
/// `n_threads == 0` falls back to rayon's global pool.
fn with_pool<F: FnOnce() -> R + Send, R: Send>(n_threads: usize, f: F) -> R {
    if n_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("rayon thread pool");
        pool.install(f)
    } else {
        f()
    }
}

// ---------------------------------------------------------------------------
// Precomputed per-user neighbor samples (for parallel modes)
// ---------------------------------------------------------------------------

struct UserSamples {
    /// Sampled rated-neighbor indices per user; empty = use all
    rated: Vec<Vec<u16>>,
    /// Sampled implicit-neighbor indices per user; empty = use all
    implicit: Vec<Vec<u16>>,
    /// Scale factor for rated neighbors (compensates for sampling)
    rated_scale: Vec<f32>,
    /// Scale factor for implicit neighbors
    impl_scale: Vec<f32>,
}

impl UserSamples {
    fn precompute(
        n_users: usize,
        user_rated: &[Vec<RatedItem>],
        user_implicit: &[Vec<ImplicitItem>],
        max_neighbors: usize,
        seed: u64,
        epoch: usize,
    ) -> Self {
        let mut rated = vec![Vec::new(); n_users];
        let mut implicit = vec![Vec::new(); n_users];
        let mut rated_scale = vec![1.0_f32; n_users];
        let mut impl_scale = vec![1.0_f32; n_users];

        for u in 0..n_users {
            let n_rated = user_rated[u].len();
            let n_impl = user_implicit[u].len();
            let n_full = n_rated + n_impl;

            if max_neighbors > 0 && n_full > max_neighbors {
                let mut rng = StdRng::seed_from_u64(
                    seed.wrapping_add(epoch as u64 * 1_000_003).wrapping_add(u as u64 * 7)
                );

                let rated_budget = ((max_neighbors as f64 * n_rated as f64 / n_full as f64)
                    .round() as usize).max(1).min(n_rated);
                let impl_budget = (max_neighbors - rated_budget).min(n_impl);

                let mut ridx: Vec<u16> = (0..n_rated as u16).collect();
                ridx.shuffle(&mut rng);
                ridx.truncate(rated_budget);
                ridx.sort_unstable();
                rated_scale[u] = n_rated as f32 / rated_budget as f32;
                rated[u] = ridx;

                if n_impl > 0 {
                    let mut iidx: Vec<u16> = (0..n_impl as u16).collect();
                    iidx.shuffle(&mut rng);
                    iidx.truncate(impl_budget);
                    iidx.sort_unstable();
                    impl_scale[u] = if impl_budget > 0 { n_impl as f32 / impl_budget as f32 } else { 1.0 };
                    implicit[u] = iidx;
                }
            }
            // else: empty vecs = use all, scale = 1.0
        }

        Self { rated, implicit, rated_scale, impl_scale }
    }

    #[inline]
    fn rated_info(&self, u: usize, n_rated_full: usize) -> (usize, f32, bool) {
        if self.rated[u].is_empty() {
            (n_rated_full, 1.0, false)
        } else {
            (self.rated[u].len(), self.rated_scale[u], true)
        }
    }

    #[inline]
    fn impl_info(&self, u: usize, n_impl_full: usize) -> (usize, f32, bool) {
        if self.implicit[u].is_empty() {
            (n_impl_full, 1.0, false)
        } else {
            (self.implicit[u].len(), self.impl_scale[u], true)
        }
    }

    #[inline]
    fn rated_idx(&self, u: usize, k: usize, use_sampling: bool) -> usize {
        if use_sampling { self.rated[u][k] as usize } else { k }
    }

    #[inline]
    fn impl_idx(&self, u: usize, k: usize, use_sampling: bool) -> usize {
        if use_sampling { self.implicit[u][k] as usize } else { k }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct BknbrxModel {
    cfg: BknbrxConfig,
    gbias: f32,

    // Baseline parameters (eq 10)
    bu: Array1<f32>,
    alpha_u: Array1<f32>,
    cu: Array1<f32>,
    bi: Array1<f32>,
    bit_bin: Vec<Vec<f32>>,
    but: Vec<f32>,
    cut: Vec<f32>,

    // Precomputed for baseline
    tu_mean: Array1<f32>,
    day_range: i32,
    ud: SparseUD,
    user_offsets: Array1<usize>,

    // Neighborhood parameters
    w: Vec<f32>,          // [n_items * n_items] — weights for explicit term (empty if lr_w==0)
    c: Vec<f32>,          // [n_items * n_items] — weights for implicit term (empty if lr_c==0)
    w_day: Vec<f32>,      // [n_items * n_items] — same-day explicit weights (empty if lr_w_day==0)
    c_day: Vec<f32>,      // [n_items * n_items] — same-day implicit weights (empty if lr_c_day==0)
    beta_u: Array1<f32>,  // [n_users] — per-user temporal decay rate

    // Per-user neighborhood data
    user_rated: Vec<Vec<RatedItem>>,
    user_implicit: Vec<Vec<ImplicitItem>>,
    norm_nu: Array1<f32>, // |N(u)|^{-1/2}
    norm_ru: Array1<f32>, // |R(u)|^{-1/2}

    // Item-indexed training data (for parallel modes)
    by_item: Vec<Vec<u32>>, // [n_items] -> list of rating indices

    n_items: usize,
}

impl BknbrxModel {
    #[inline] fn has_w(&self) -> bool { !self.w.is_empty() }
    #[inline] fn has_c(&self) -> bool { !self.c.is_empty() }
    #[inline] fn has_w_day(&self) -> bool { !self.w_day.is_empty() }
    #[inline] fn has_c_day(&self) -> bool { !self.c_day.is_empty() }

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

    /// Baseline prediction b_ui (eq 10)
    #[inline]
    fn baseline(&self, u: usize, i: usize, day: i32) -> f32 {
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

    /// Full prediction: baseline + neighborhood (eq 18) + same-day terms
    #[inline]
    fn predict_full(&self, u: usize, i: usize, day: i32) -> f32 {
        let bui = self.baseline(u, i, day);
        let beta = self.beta_u[u].max(0.0);
        let day16 = day as i16;
        let ni = self.n_items;
        let has_w = self.has_w();
        let has_c = self.has_c();
        let has_w_day = self.has_w_day();
        let has_c_day = self.has_c_day();

        let mut c_sum = 0.0_f32;
        let mut w_sum = 0.0_f32;
        let mut c_day_sum = 0.0_f32;
        let mut w_day_sum = 0.0_f32;
        let mut n_same_rated = 0_u32;
        let mut n_same_total = 0_u32;

        for ri in &self.user_rated[u] {
            let j = ri.item as usize;
            if j == i { continue; }
            let idx = i * ni + j;
            let dt = ((day16 as i32) - (ri.day as i32)).unsigned_abs() as f32;
            let decay = (-beta * dt).exp();
            if has_c { c_sum += decay * self.c[idx]; }
            if has_w { w_sum += decay * ri.r_minus_btilde * self.w[idx]; }
            if ri.day == day16 {
                n_same_rated += 1;
                n_same_total += 1;
                if has_c_day { c_day_sum += self.c_day[idx]; }
                if has_w_day { w_day_sum += ri.r_minus_btilde * self.w_day[idx]; }
            }
        }
        for ii in &self.user_implicit[u] {
            let j = ii.item as usize;
            if j == i { continue; }
            let idx = i * ni + j;
            let dt = ((day16 as i32) - (ii.day as i32)).unsigned_abs() as f32;
            let decay = (-beta * dt).exp();
            if has_c { c_sum += decay * self.c[idx]; }
            if ii.day == day16 {
                n_same_total += 1;
                if has_c_day { c_day_sum += self.c_day[idx]; }
            }
        }

        let nn_day = if n_same_total > 0 { (n_same_total as f32).powf(-0.5) } else { 0.0 };
        let nr_day = if n_same_rated > 0 { (n_same_rated as f32).powf(-0.5) } else { 0.0 };

        bui + self.norm_nu[u] * c_sum + self.norm_ru[u] * w_sum
            + nn_day * c_day_sum + nr_day * w_day_sum
    }

    // -----------------------------------------------------------------------
    // Sequential training
    // -----------------------------------------------------------------------
    fn fit_epoch_sequential(&mut self, tr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        let ni = self.n_items;
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);
        let max_nbr = cfg.max_neighbors;
        let has_w = self.has_w();
        let has_c = self.has_c();
        let has_w_day = self.has_w_day();
        let has_c_day = self.has_c_day();

        let mut sampled_rated_idx: Vec<usize> = Vec::new();
        let mut sampled_impl_idx: Vec<usize> = Vec::new();
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(epoch as u64 * 1_000_003));

        let mut decay_rated: Vec<f32> = Vec::new();
        let mut decay_implicit: Vec<f32> = Vec::new();

        for &u in crate::progress!(users.iter()) {
            let start = self.user_offsets[u];
            let end = self.user_offsets[u + 1];
            if start == end { continue; }

            let beta = self.beta_u[u].max(0.0);
            let nr = self.norm_ru[u];
            let nn = self.norm_nu[u];

            let n_rated_full = self.user_rated[u].len();
            let n_impl_full = self.user_implicit[u].len();
            let n_full = n_rated_full + n_impl_full;

            let use_sampling = if max_nbr > 0 && n_full > max_nbr {
                let rated_budget = ((max_nbr as f64 * n_rated_full as f64 / n_full as f64).round() as usize).max(1).min(n_rated_full);
                let impl_budget = (max_nbr - rated_budget).min(n_impl_full);

                sampled_rated_idx.clear();
                sampled_rated_idx.extend(0..n_rated_full);
                sampled_rated_idx.shuffle(&mut rng);
                sampled_rated_idx.truncate(rated_budget);
                sampled_rated_idx.sort_unstable();

                sampled_impl_idx.clear();
                sampled_impl_idx.extend(0..n_impl_full);
                sampled_impl_idx.shuffle(&mut rng);
                sampled_impl_idx.truncate(impl_budget);
                sampled_impl_idx.sort_unstable();

                true
            } else {
                false
            };

            let (iter_rated, iter_impl) = if use_sampling {
                (sampled_rated_idx.len(), sampled_impl_idx.len())
            } else {
                (n_rated_full, n_impl_full)
            };
            let rated_scale = if use_sampling { n_rated_full as f32 / iter_rated as f32 } else { 1.0 };
            let impl_scale = if use_sampling && iter_impl > 0 { n_impl_full as f32 / iter_impl as f32 } else { 1.0 };

            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                let r = tr.residuals[t];
                let day = tr.dates[t] as i32;
                let day16 = tr.dates[t];

                // --- Same-day forward pass (no sampling, iterate all neighbors) ---
                let mut c_day_sum = 0.0_f32;
                let mut w_day_sum = 0.0_f32;
                let mut n_same_rated = 0_u32;
                let mut n_same_total = 0_u32;
                if has_w_day || has_c_day {
                    for ri in &self.user_rated[u] {
                        let j = ri.item as usize;
                        if j == i || ri.day != day16 { continue; }
                        let idx = i * ni + j;
                        n_same_rated += 1;
                        n_same_total += 1;
                        if has_c_day { c_day_sum += self.c_day[idx]; }
                        if has_w_day { w_day_sum += ri.r_minus_btilde * self.w_day[idx]; }
                    }
                    for ii in &self.user_implicit[u] {
                        let j = ii.item as usize;
                        if j == i || ii.day != day16 { continue; }
                        n_same_total += 1;
                        if has_c_day { c_day_sum += self.c_day[i * ni + j]; }
                    }
                }
                let nn_day = if n_same_total > 0 { (n_same_total as f32).powf(-0.5) } else { 0.0 };
                let nr_day = if n_same_rated > 0 { (n_same_rated as f32).powf(-0.5) } else { 0.0 };

                // --- Global forward pass (sampled neighbors) ---
                decay_rated.clear();
                decay_rated.reserve(iter_rated);
                decay_implicit.clear();
                decay_implicit.reserve(iter_impl);

                let mut c_sum = 0.0_f32;
                let mut w_sum = 0.0_f32;
                let mut d_beta = 0.0_f32;

                for k in 0..iter_rated {
                    let ri_idx = if use_sampling { sampled_rated_idx[k] } else { k };
                    let ri = &self.user_rated[u][ri_idx];
                    let j = ri.item as usize;
                    if j == i { decay_rated.push(0.0); continue; }
                    let dt = ((day16 as i32) - (ri.day as i32)).unsigned_abs() as f32;
                    let decay = (-beta * dt).exp();
                    decay_rated.push(decay);

                    let c_ij = if has_c { self.c[i * ni + j] } else { 0.0 };
                    let w_ij = if has_w { self.w[i * ni + j] } else { 0.0 };
                    let rmb = ri.r_minus_btilde;

                    c_sum += decay * c_ij * rated_scale;
                    w_sum += decay * rmb * w_ij * rated_scale;
                    d_beta += -dt * decay * (nn * c_ij * rated_scale + nr * rmb * w_ij * rated_scale);
                }

                for k in 0..iter_impl {
                    let ii_idx = if use_sampling { sampled_impl_idx[k] } else { k };
                    let ii = &self.user_implicit[u][ii_idx];
                    let j = ii.item as usize;
                    let dt = ((day16 as i32) - (ii.day as i32)).unsigned_abs() as f32;
                    let decay = (-beta * dt).exp();
                    decay_implicit.push(decay);

                    let c_ij = if has_c { self.c[i * ni + j] } else { 0.0 };
                    c_sum += decay * c_ij * impl_scale;
                    d_beta += -dt * decay * nn * c_ij * impl_scale;
                }

                let bui = self.baseline(u, i, day);
                let pred = bui + nn * c_sum + nr * w_sum
                    + nn_day * c_day_sum + nr_day * w_day_sum;
                let err = pred - r;

                // --- Baseline updates ---
                let b = self.time_bin(day);
                let dev = self.dev(u, day);
                let ud_idx = self.ud.index(u, day16);
                let cut_val = ud_idx.map_or(0.0, |idx| self.cut[idx]);
                let bi_t = self.bi[i] + self.bit_bin[i][b];
                let cu_t = self.cu[u] + cut_val;

                self.bu[u] -= cfg.lr_bu * (err + cfg.reg_bu * self.bu[u]);
                self.alpha_u[u] -= cfg.lr_alpha * (err * dev + cfg.reg_alpha * self.alpha_u[u]);
                if let Some(idx) = ud_idx {
                    self.but[idx] -= cfg.lr_but * (err + cfg.reg_but * self.but[idx]);
                }
                self.bi[i] -= cfg.lr_bi * (err * cu_t + cfg.reg_bi * self.bi[i]);
                self.bit_bin[i][b] -= cfg.lr_bit * (err * cu_t + cfg.reg_bit * self.bit_bin[i][b]);
                self.cu[u] -= cfg.lr_cu * (err * bi_t + cfg.reg_cu * (self.cu[u] - 1.0));
                if let Some(idx) = ud_idx {
                    self.cut[idx] -= cfg.lr_cut * (err * bi_t + cfg.reg_cut * self.cut[idx]);
                }

                // --- Global neighborhood backward pass (sampled) ---
                for k in 0..iter_rated {
                    let ri_idx = if use_sampling { sampled_rated_idx[k] } else { k };
                    let ri = &self.user_rated[u][ri_idx];
                    let j = ri.item as usize;
                    if j == i { continue; }
                    let decay = decay_rated[k];
                    let rmb = ri.r_minus_btilde;
                    let idx = i * ni + j;

                    if has_w {
                        let w_ij = self.w[idx];
                        self.w[idx] -= cfg.lr_w * (err * nr * decay * rmb * rated_scale + cfg.reg_w * w_ij);
                    }
                    if has_c {
                        let c_ij = self.c[idx];
                        self.c[idx] -= cfg.lr_c * (err * nn * decay * rated_scale + cfg.reg_c * c_ij);
                    }
                }

                for k in 0..iter_impl {
                    let ii_idx = if use_sampling { sampled_impl_idx[k] } else { k };
                    let ii = &self.user_implicit[u][ii_idx];
                    let j = ii.item as usize;
                    let decay = decay_implicit[k];
                    let idx = i * ni + j;

                    if has_c {
                        let c_ij = self.c[idx];
                        self.c[idx] -= cfg.lr_c * (err * nn * decay * impl_scale + cfg.reg_c * c_ij);
                    }
                }

                // --- Same-day backward pass (no sampling, iterate all) ---
                if has_w_day || has_c_day {
                    for ri in &self.user_rated[u] {
                        let j = ri.item as usize;
                        if j == i || ri.day != day16 { continue; }
                        let idx = i * ni + j;
                        if has_w_day {
                            let wd = self.w_day[idx];
                            self.w_day[idx] -= cfg.lr_w_day * (err * nr_day * ri.r_minus_btilde + cfg.reg_w_day * wd);
                        }
                        if has_c_day {
                            let cd = self.c_day[idx];
                            self.c_day[idx] -= cfg.lr_c_day * (err * nn_day + cfg.reg_c_day * cd);
                        }
                    }
                    for ii in &self.user_implicit[u] {
                        let j = ii.item as usize;
                        if j == i || ii.day != day16 { continue; }
                        let idx = i * ni + j;
                        if has_c_day {
                            let cd = self.c_day[idx];
                            self.c_day[idx] -= cfg.lr_c_day * (err * nn_day + cfg.reg_c_day * cd);
                        }
                    }
                }

                self.beta_u[u] -= cfg.lr_beta_u * (err * d_beta + cfg.reg_beta_u * self.beta_u[u]);
                if self.beta_u[u] < 0.0 { self.beta_u[u] = 0.0; }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Hogwild: item-parallel, atomic updates on user params
    // -----------------------------------------------------------------------
    fn fit_epoch_hogwild(&mut self, tr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        let ni = self.n_items;
        let n_users = tr.n_users;
        let has_w = self.has_w();
        let has_c = self.has_c();
        let has_w_day = self.has_w_day();
        let has_c_day = self.has_c_day();

        // Precompute neighbor samples
        let samples = UserSamples::precompute(
            n_users, &self.user_rated, &self.user_implicit,
            cfg.max_neighbors, cfg.seed, epoch,
        );

        // Create atomic copies of user params
        let bu_a = array1_to_atomic(&self.bu);
        let alpha_u_a = array1_to_atomic(&self.alpha_u);
        let cu_a = array1_to_atomic(&self.cu);
        let beta_u_a = array1_to_atomic(&self.beta_u);
        let but_a = vec_to_atomic(&self.but);
        let cut_a = vec_to_atomic(&self.cut);

        // Each thread owns item i exclusively for w[i*ni..], c[i*ni..], bi[i], bit_bin[i].
        let w_base = if has_w { self.w.as_mut_ptr() as usize } else { 0 };
        let c_base = if has_c { self.c.as_mut_ptr() as usize } else { 0 };
        let wd_base = if has_w_day { self.w_day.as_mut_ptr() as usize } else { 0 };
        let cd_base = if has_c_day { self.c_day.as_mut_ptr() as usize } else { 0 };
        let bi_base = self.bi.as_mut_ptr() as usize;
        let bit_bin_base = self.bit_bin.as_mut_ptr() as usize;

        let pb = crate::make_pb(ni as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}]").unwrap());

        with_pool(cfg.n_threads, || {
        (0..ni).into_par_iter().for_each(|i| {
            let w_row = if has_w { unsafe { std::slice::from_raw_parts_mut((w_base as *mut f32).add(i * ni), ni) } } else { &mut [] as &mut [f32] };
            let c_row = if has_c { unsafe { std::slice::from_raw_parts_mut((c_base as *mut f32).add(i * ni), ni) } } else { &mut [] as &mut [f32] };
            let wd_row = if has_w_day { unsafe { std::slice::from_raw_parts_mut((wd_base as *mut f32).add(i * ni), ni) } } else { &mut [] as &mut [f32] };
            let cd_row = if has_c_day { unsafe { std::slice::from_raw_parts_mut((cd_base as *mut f32).add(i * ni), ni) } } else { &mut [] as &mut [f32] };
            let bi_ref = unsafe { &mut *(bi_base as *mut f32).add(i) };
            let bit_bin_ref = unsafe { &mut *(bit_bin_base as *mut Vec<f32>).add(i) };

            let mut decay_rated: Vec<f32> = Vec::new();
            let mut decay_implicit: Vec<f32> = Vec::new();

            for &t32 in &self.by_item[i] {
                let t = t32 as usize;
                let u = tr.user_idxs[t] as usize;
                let r = tr.residuals[t];
                let day = tr.dates[t] as i32;
                let day16 = tr.dates[t];

                let beta = atomic_load_f32(&beta_u_a[u]).max(0.0);
                let nr = self.norm_ru[u];
                let nn = self.norm_nu[u];

                let n_rated_full = self.user_rated[u].len();
                let n_impl_full = self.user_implicit[u].len();
                let (iter_rated, rated_scale, samp_r) = samples.rated_info(u, n_rated_full);
                let (iter_impl, impl_scale, samp_i) = samples.impl_info(u, n_impl_full);

                // --- Same-day forward pass (no sampling, iterate all) ---
                let mut c_day_sum = 0.0_f32;
                let mut w_day_sum = 0.0_f32;
                let mut n_same_rated = 0_u32;
                let mut n_same_total = 0_u32;
                if has_w_day || has_c_day {
                    for ri in &self.user_rated[u] {
                        let j = ri.item as usize;
                        if j == i || ri.day != day16 { continue; }
                        n_same_rated += 1;
                        n_same_total += 1;
                        if has_c_day { c_day_sum += cd_row[j]; }
                        if has_w_day { w_day_sum += ri.r_minus_btilde * wd_row[j]; }
                    }
                    for ii in &self.user_implicit[u] {
                        let j = ii.item as usize;
                        if j == i || ii.day != day16 { continue; }
                        n_same_total += 1;
                        if has_c_day { c_day_sum += cd_row[j]; }
                    }
                }
                let nn_day = if n_same_total > 0 { (n_same_total as f32).powf(-0.5) } else { 0.0 };
                let nr_day = if n_same_rated > 0 { (n_same_rated as f32).powf(-0.5) } else { 0.0 };

                // --- Global forward pass (sampled) ---
                decay_rated.clear();
                decay_implicit.clear();

                let mut c_sum = 0.0_f32;
                let mut w_sum = 0.0_f32;
                let mut d_beta = 0.0_f32;

                for k in 0..iter_rated {
                    let ri_idx = samples.rated_idx(u, k, samp_r);
                    let ri = &self.user_rated[u][ri_idx];
                    let j = ri.item as usize;
                    if j == i { decay_rated.push(0.0); continue; }
                    let dt = ((day16 as i32) - (ri.day as i32)).unsigned_abs() as f32;
                    let decay = (-beta * dt).exp();
                    decay_rated.push(decay);

                    let c_ij = if has_c { c_row[j] } else { 0.0 };
                    let w_ij = if has_w { w_row[j] } else { 0.0 };
                    let rmb = ri.r_minus_btilde;

                    c_sum += decay * c_ij * rated_scale;
                    w_sum += decay * rmb * w_ij * rated_scale;
                    d_beta += -dt * decay * (nn * c_ij * rated_scale + nr * rmb * w_ij * rated_scale);
                }

                for k in 0..iter_impl {
                    let ii_idx = samples.impl_idx(u, k, samp_i);
                    let ii = &self.user_implicit[u][ii_idx];
                    let j = ii.item as usize;
                    let dt = ((day16 as i32) - (ii.day as i32)).unsigned_abs() as f32;
                    let decay = (-beta * dt).exp();
                    decay_implicit.push(decay);

                    let c_ij = if has_c { c_row[j] } else { 0.0 };
                    c_sum += decay * c_ij * impl_scale;
                    d_beta += -dt * decay * nn * c_ij * impl_scale;
                }

                // Baseline (read user params atomically)
                let bu_val = atomic_load_f32(&bu_a[u]);
                let alpha_val = atomic_load_f32(&alpha_u_a[u]);
                let cu_val = atomic_load_f32(&cu_a[u]);
                let dev = self.dev(u, day);
                let ud_idx = self.ud.index(u, day16);
                let but_val = ud_idx.map_or(0.0, |idx| atomic_load_f32(&but_a[idx]));
                let cut_val = ud_idx.map_or(0.0, |idx| atomic_load_f32(&cut_a[idx]));

                let bu_t = bu_val + alpha_val * dev + but_val;
                let bi_t = *bi_ref + bit_bin_ref[self.time_bin(day)];
                let cu_t = cu_val + cut_val;
                let bui = self.gbias + bu_t + bi_t * cu_t;

                let pred = bui + nn * c_sum + nr * w_sum
                    + nn_day * c_day_sum + nr_day * w_day_sum;
                let err = pred - r;

                // Baseline updates — user params via atomics
                atomic_sub_f32(&bu_a[u], cfg.lr_bu * (err + cfg.reg_bu * bu_val));
                atomic_sub_f32(&alpha_u_a[u], cfg.lr_alpha * (err * dev + cfg.reg_alpha * alpha_val));
                if let Some(idx) = ud_idx {
                    let bv = atomic_load_f32(&but_a[idx]);
                    atomic_sub_f32(&but_a[idx], cfg.lr_but * (err + cfg.reg_but * bv));
                }
                // Item params — exclusive access, direct write
                let b = self.time_bin(day);
                *bi_ref -= cfg.lr_bi * (err * cu_t + cfg.reg_bi * *bi_ref);
                bit_bin_ref[b] -= cfg.lr_bit * (err * cu_t + cfg.reg_bit * bit_bin_ref[b]);
                // User params via atomics
                atomic_sub_f32(&cu_a[u], cfg.lr_cu * (err * bi_t + cfg.reg_cu * (cu_val - 1.0)));
                if let Some(idx) = ud_idx {
                    let cv = atomic_load_f32(&cut_a[idx]);
                    atomic_sub_f32(&cut_a[idx], cfg.lr_cut * (err * bi_t + cfg.reg_cut * cv));
                }

                // --- Global neighborhood backward pass (sampled) ---
                for k in 0..iter_rated {
                    let ri_idx = samples.rated_idx(u, k, samp_r);
                    let ri = &self.user_rated[u][ri_idx];
                    let j = ri.item as usize;
                    if j == i { continue; }
                    let decay = decay_rated[k];
                    let rmb = ri.r_minus_btilde;

                    if has_w {
                        let w_ij = w_row[j];
                        w_row[j] -= cfg.lr_w * (err * nr * decay * rmb * rated_scale + cfg.reg_w * w_ij);
                    }
                    if has_c {
                        let c_ij = c_row[j];
                        c_row[j] -= cfg.lr_c * (err * nn * decay * rated_scale + cfg.reg_c * c_ij);
                    }
                }

                for k in 0..iter_impl {
                    let ii_idx = samples.impl_idx(u, k, samp_i);
                    let ii = &self.user_implicit[u][ii_idx];
                    let j = ii.item as usize;
                    let decay = decay_implicit[k];

                    if has_c {
                        let c_ij = c_row[j];
                        c_row[j] -= cfg.lr_c * (err * nn * decay * impl_scale + cfg.reg_c * c_ij);
                    }
                }

                // --- Same-day backward pass (no sampling) ---
                if has_w_day || has_c_day {
                    for ri in &self.user_rated[u] {
                        let j = ri.item as usize;
                        if j == i || ri.day != day16 { continue; }
                        if has_w_day {
                            let wd = wd_row[j];
                            wd_row[j] -= cfg.lr_w_day * (err * nr_day * ri.r_minus_btilde + cfg.reg_w_day * wd);
                        }
                        if has_c_day {
                            let cd = cd_row[j];
                            cd_row[j] -= cfg.lr_c_day * (err * nn_day + cfg.reg_c_day * cd);
                        }
                    }
                    for ii in &self.user_implicit[u] {
                        let j = ii.item as usize;
                        if j == i || ii.day != day16 { continue; }
                        if has_c_day {
                            let cd = cd_row[j];
                            cd_row[j] -= cfg.lr_c_day * (err * nn_day + cfg.reg_c_day * cd);
                        }
                    }
                }

                // beta_u via atomic
                atomic_sub_f32(&beta_u_a[u], cfg.lr_beta_u * (err * d_beta + cfg.reg_beta_u * beta));
                atomic_clamp_nonneg(&beta_u_a[u]);
            }

            pb.inc(1);
        });
        });

        pb.finish_and_clear();

        // Copy atomics back
        self.bu = atomic_to_array1(&bu_a);
        self.alpha_u = atomic_to_array1(&alpha_u_a);
        self.cu = atomic_to_array1(&cu_a);
        self.beta_u = atomic_to_array1(&beta_u_a);
        self.but = atomic_to_vec(&but_a);
        self.cut = atomic_to_vec(&cut_a);
    }

    // -----------------------------------------------------------------------
    // TwoPass: item-parallel for w/c/bi/bit_bin, then user-sequential baseline
    // -----------------------------------------------------------------------
    fn fit_epoch_twopass(&mut self, tr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        let ni = self.n_items;
        let n_users = tr.n_users;
        let n_ratings = tr.n_ratings;
        let has_w = self.has_w();
        let has_c = self.has_c();
        let has_w_day = self.has_w_day();
        let has_c_day = self.has_c_day();

        // Precompute neighbor samples
        let samples = UserSamples::precompute(
            n_users, &self.user_rated, &self.user_implicit,
            cfg.max_neighbors, cfg.seed, epoch,
        );

        // Per-rating error buffer (filled in pass 1, consumed in pass 2)
        let errors: Vec<AtomicU32> = (0..n_ratings).map(|_| AtomicU32::new(0.0_f32.to_bits())).collect();

        // --- Pass 1: item-parallel, update w/c/w_day/c_day/bi/bit_bin, store errors ---
        let w_base = if has_w { self.w.as_mut_ptr() as usize } else { 0 };
        let c_base = if has_c { self.c.as_mut_ptr() as usize } else { 0 };
        let wd_base = if has_w_day { self.w_day.as_mut_ptr() as usize } else { 0 };
        let cd_base = if has_c_day { self.c_day.as_mut_ptr() as usize } else { 0 };
        let bi_base = self.bi.as_mut_ptr() as usize;
        let bit_bin_base = self.bit_bin.as_mut_ptr() as usize;

        let pb = crate::make_pb(ni as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} P1 [{elapsed_precise}]").unwrap());

        with_pool(cfg.n_threads, || {
        (0..ni).into_par_iter().for_each(|i| {
            let w_row = if has_w { unsafe { std::slice::from_raw_parts_mut((w_base as *mut f32).add(i * ni), ni) } } else { &mut [] as &mut [f32] };
            let c_row = if has_c { unsafe { std::slice::from_raw_parts_mut((c_base as *mut f32).add(i * ni), ni) } } else { &mut [] as &mut [f32] };
            let wd_row = if has_w_day { unsafe { std::slice::from_raw_parts_mut((wd_base as *mut f32).add(i * ni), ni) } } else { &mut [] as &mut [f32] };
            let cd_row = if has_c_day { unsafe { std::slice::from_raw_parts_mut((cd_base as *mut f32).add(i * ni), ni) } } else { &mut [] as &mut [f32] };
            let bi_ref = unsafe { &mut *(bi_base as *mut f32).add(i) };
            let bit_bin_ref = unsafe { &mut *(bit_bin_base as *mut Vec<f32>).add(i) };

            let mut decay_rated: Vec<f32> = Vec::new();
            let mut decay_implicit: Vec<f32> = Vec::new();

            for &t32 in &self.by_item[i] {
                let t = t32 as usize;
                let u = tr.user_idxs[t] as usize;
                let r = tr.residuals[t];
                let day = tr.dates[t] as i32;
                let day16 = tr.dates[t];

                let beta = self.beta_u[u].max(0.0);
                let nr = self.norm_ru[u];
                let nn = self.norm_nu[u];

                let n_rated_full = self.user_rated[u].len();
                let n_impl_full = self.user_implicit[u].len();
                let (iter_rated, rated_scale, samp_r) = samples.rated_info(u, n_rated_full);
                let (iter_impl, impl_scale, samp_i) = samples.impl_info(u, n_impl_full);

                // --- Same-day forward pass (no sampling) ---
                let mut c_day_sum = 0.0_f32;
                let mut w_day_sum = 0.0_f32;
                let mut n_same_rated = 0_u32;
                let mut n_same_total = 0_u32;
                if has_w_day || has_c_day {
                    for ri in &self.user_rated[u] {
                        let j = ri.item as usize;
                        if j == i || ri.day != day16 { continue; }
                        n_same_rated += 1;
                        n_same_total += 1;
                        if has_c_day { c_day_sum += cd_row[j]; }
                        if has_w_day { w_day_sum += ri.r_minus_btilde * wd_row[j]; }
                    }
                    for ii in &self.user_implicit[u] {
                        let j = ii.item as usize;
                        if j == i || ii.day != day16 { continue; }
                        n_same_total += 1;
                        if has_c_day { c_day_sum += cd_row[j]; }
                    }
                }
                let nn_day = if n_same_total > 0 { (n_same_total as f32).powf(-0.5) } else { 0.0 };
                let nr_day = if n_same_rated > 0 { (n_same_rated as f32).powf(-0.5) } else { 0.0 };

                // --- Global forward pass (sampled) ---
                decay_rated.clear();
                decay_implicit.clear();

                let mut c_sum = 0.0_f32;
                let mut w_sum = 0.0_f32;

                for k in 0..iter_rated {
                    let ri_idx = samples.rated_idx(u, k, samp_r);
                    let ri = &self.user_rated[u][ri_idx];
                    let j = ri.item as usize;
                    if j == i { decay_rated.push(0.0); continue; }
                    let dt = ((day16 as i32) - (ri.day as i32)).unsigned_abs() as f32;
                    let decay = (-beta * dt).exp();
                    decay_rated.push(decay);

                    let c_ij = if has_c { c_row[j] } else { 0.0 };
                    let w_ij = if has_w { w_row[j] } else { 0.0 };
                    let rmb = ri.r_minus_btilde;

                    c_sum += decay * c_ij * rated_scale;
                    w_sum += decay * rmb * w_ij * rated_scale;
                }

                for k in 0..iter_impl {
                    let ii_idx = samples.impl_idx(u, k, samp_i);
                    let ii = &self.user_implicit[u][ii_idx];
                    let j = ii.item as usize;
                    let dt = ((day16 as i32) - (ii.day as i32)).unsigned_abs() as f32;
                    let decay = (-beta * dt).exp();
                    decay_implicit.push(decay);

                    let c_ij = if has_c { c_row[j] } else { 0.0 };
                    c_sum += decay * c_ij * impl_scale;
                }

                // Baseline (user params frozen in pass 1)
                let bui = self.baseline(u, i, day);
                let pred = bui + nn * c_sum + nr * w_sum
                    + nn_day * c_day_sum + nr_day * w_day_sum;
                let err = pred - r;

                // Store error for pass 2
                errors[t].store(err.to_bits(), Ordering::Relaxed);

                // Item param updates (exclusive per item i)
                let b = self.time_bin(day);
                let cut_val = self.ud.index(u, day16).map_or(0.0, |idx| self.cut[idx]);
                let cu_t = self.cu[u] + cut_val;

                *bi_ref -= cfg.lr_bi * (err * cu_t + cfg.reg_bi * *bi_ref);
                bit_bin_ref[b] -= cfg.lr_bit * (err * cu_t + cfg.reg_bit * bit_bin_ref[b]);

                // --- Global neighborhood backward (sampled, exclusive per item i) ---
                for k in 0..iter_rated {
                    let ri_idx = samples.rated_idx(u, k, samp_r);
                    let ri = &self.user_rated[u][ri_idx];
                    let j = ri.item as usize;
                    if j == i { continue; }
                    let decay = decay_rated[k];
                    let rmb = ri.r_minus_btilde;

                    if has_w {
                        let w_ij = w_row[j];
                        w_row[j] -= cfg.lr_w * (err * nr * decay * rmb * rated_scale + cfg.reg_w * w_ij);
                    }
                    if has_c {
                        let c_ij = c_row[j];
                        c_row[j] -= cfg.lr_c * (err * nn * decay * rated_scale + cfg.reg_c * c_ij);
                    }
                }

                for k in 0..iter_impl {
                    let ii_idx = samples.impl_idx(u, k, samp_i);
                    let ii = &self.user_implicit[u][ii_idx];
                    let j = ii.item as usize;
                    let decay = decay_implicit[k];

                    if has_c {
                        let c_ij = c_row[j];
                        c_row[j] -= cfg.lr_c * (err * nn * decay * impl_scale + cfg.reg_c * c_ij);
                    }
                }

                // --- Same-day backward (no sampling, exclusive per item i) ---
                if has_w_day || has_c_day {
                    for ri in &self.user_rated[u] {
                        let j = ri.item as usize;
                        if j == i || ri.day != day16 { continue; }
                        if has_w_day {
                            let wd = wd_row[j];
                            wd_row[j] -= cfg.lr_w_day * (err * nr_day * ri.r_minus_btilde + cfg.reg_w_day * wd);
                        }
                        if has_c_day {
                            let cd = cd_row[j];
                            cd_row[j] -= cfg.lr_c_day * (err * nn_day + cfg.reg_c_day * cd);
                        }
                    }
                    for ii in &self.user_implicit[u] {
                        let j = ii.item as usize;
                        if j == i || ii.day != day16 { continue; }
                        if has_c_day {
                            let cd = cd_row[j];
                            cd_row[j] -= cfg.lr_c_day * (err * nn_day + cfg.reg_c_day * cd);
                        }
                    }
                }
            }

            pb.inc(1);
        });
        });

        pb.finish_and_clear();

        // --- Pass 2: user-sequential, update baseline user params + beta_u ---
        let users = get_users(n_users, cfg.shuffle_users, cfg.seed, epoch);
        for &u in crate::progress!(users.iter()) {
            let start = self.user_offsets[u];
            let end = self.user_offsets[u + 1];
            if start == end { continue; }

            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                let day = tr.dates[t] as i32;
                let day16 = tr.dates[t];
                let err = f32::from_bits(errors[t].load(Ordering::Relaxed));

                let b = self.time_bin(day);
                let dev = self.dev(u, day);
                let ud_idx = self.ud.index(u, day16);
                let bi_t = self.bi[i] + self.bit_bin[i][b];

                self.bu[u] -= cfg.lr_bu * (err + cfg.reg_bu * self.bu[u]);
                self.alpha_u[u] -= cfg.lr_alpha * (err * dev + cfg.reg_alpha * self.alpha_u[u]);
                if let Some(idx) = ud_idx {
                    self.but[idx] -= cfg.lr_but * (err + cfg.reg_but * self.but[idx]);
                }
                self.cu[u] -= cfg.lr_cu * (err * bi_t + cfg.reg_cu * (self.cu[u] - 1.0));
                if let Some(idx) = ud_idx {
                    self.cut[idx] -= cfg.lr_cut * (err * bi_t + cfg.reg_cut * self.cut[idx]);
                }

                // beta_u: need d_beta — recompute from current (updated) w/c
                let beta = self.beta_u[u].max(0.0);
                let nr = self.norm_ru[u];
                let nn = self.norm_nu[u];

                let n_rated_full = self.user_rated[u].len();
                let n_impl_full = self.user_implicit[u].len();
                let (iter_rated, rated_scale, samp_r) = samples.rated_info(u, n_rated_full);
                let (iter_impl, impl_scale, samp_i) = samples.impl_info(u, n_impl_full);

                let mut d_beta = 0.0_f32;
                for k in 0..iter_rated {
                    let ri_idx = samples.rated_idx(u, k, samp_r);
                    let ri = &self.user_rated[u][ri_idx];
                    let j = ri.item as usize;
                    if j == i { continue; }
                    let dt = ((day16 as i32) - (ri.day as i32)).unsigned_abs() as f32;
                    let decay = (-beta * dt).exp();
                    let c_ij = if has_c { self.c[i * ni + j] } else { 0.0 };
                    let w_ij = if has_w { self.w[i * ni + j] } else { 0.0 };
                    let rmb = ri.r_minus_btilde;
                    d_beta += -dt * decay * (nn * c_ij * rated_scale + nr * rmb * w_ij * rated_scale);
                }
                for k in 0..iter_impl {
                    let ii_idx = samples.impl_idx(u, k, samp_i);
                    let ii = &self.user_implicit[u][ii_idx];
                    let j = ii.item as usize;
                    let dt = ((day16 as i32) - (ii.day as i32)).unsigned_abs() as f32;
                    let decay = (-beta * dt).exp();
                    let c_ij = if has_c { self.c[i * ni + j] } else { 0.0 };
                    d_beta += -dt * decay * nn * c_ij * impl_scale;
                }

                self.beta_u[u] -= cfg.lr_beta_u * (err * d_beta + cfg.reg_beta_u * self.beta_u[u]);
                if self.beta_u[u] < 0.0 { self.beta_u[u] = 0.0; }
            }
        }
    }
}

impl Regressor for BknbrxModel {
    type Config = BknbrxConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let n_users = tr.n_users;
        let n_items = tr.n_items;

        // Compute mean rating date per user (from training set only)
        let mut tu_mean = Array1::<f32>::zeros(n_users);
        let mut day_range: i32 = 0;
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
        let user_offsets = calc_user_offsets(tr);
        let gbias = calc_gbias(tr);

        // ---------------------------------------------------------------
        // Precomputed simple baselines b̃ (eqs 2–3): μ + b̃_i + b̃_u
        // ---------------------------------------------------------------
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

        // ---------------------------------------------------------------
        // Build per-user R(u) from training data (user-sorted)
        // ---------------------------------------------------------------
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

        // ---------------------------------------------------------------
        // Build per-user N(u)\R(u) from probe/qual (item-sorted, linear scan)
        // ---------------------------------------------------------------
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
            if user_rated_set[u].binary_search(&(i as u16)).is_err() {
                user_implicit[u].push(ImplicitItem {
                    item: i as u16,
                    day: pr.dates[idx],
                });
            }
        }

        for u in 0..n_users {
            user_implicit[u].sort_unstable_by_key(|ii| ii.item);
            user_implicit[u].dedup_by_key(|ii| ii.item);
        }

        // ---------------------------------------------------------------
        // Normalization factors
        // ---------------------------------------------------------------
        let mut norm_nu = Array1::<f32>::zeros(n_users);
        let mut norm_ru = Array1::<f32>::zeros(n_users);
        for u in 0..n_users {
            let n_r = user_rated[u].len();
            let n_n = n_r + user_implicit[u].len();
            norm_ru[u] = if n_r > 0 { (n_r as f32).powf(-0.5) } else { 0.0 };
            norm_nu[u] = if n_n > 0 { (n_n as f32).powf(-0.5) } else { 0.0 };
        }

        // ---------------------------------------------------------------
        // Build by_item index (for parallel modes)
        // ---------------------------------------------------------------
        let mut by_item: Vec<Vec<u32>> = vec![Vec::new(); n_items];
        for t in 0..tr.n_ratings {
            by_item[tr.item_idxs[t] as usize].push(t as u32);
        }

        // ---------------------------------------------------------------
        // Allocate w, c, w_day, c_day matrices (conditional on lr > 0)
        // ---------------------------------------------------------------
        let n2 = n_items * n_items;
        let w = if cfg.lr_w > 0.0 { vec![0.0_f32; n2] } else { Vec::new() };
        let c = if cfg.lr_c > 0.0 { vec![0.0_f32; n2] } else { Vec::new() };
        let w_day = if cfg.lr_w_day > 0.0 { vec![0.0_f32; n2] } else { Vec::new() };
        let c_day = if cfg.lr_c_day > 0.0 { vec![0.0_f32; n2] } else { Vec::new() };
        let beta_u = Array1::<f32>::zeros(n_users);

        Self {
            cfg,
            gbias,
            bu: Array1::zeros(n_users),
            alpha_u: Array1::zeros(n_users),
            cu: Array1::ones(n_users),
            bi: Array1::zeros(n_items),
            bit_bin: vec![vec![0.0; cfg.n_time_bins]; n_items],
            but: vec![0.0; n_ud],
            cut: vec![0.0; n_ud],
            tu_mean,
            day_range,
            ud,
            user_offsets,
            w,
            c,
            w_day,
            c_day,
            beta_u,
            user_rated,
            user_implicit,
            norm_nu,
            norm_ru,
            by_item,
            n_items,
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        self.predict_full(u, i, day)
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        match self.cfg.parallel_mode {
            ParallelMode::Sequential => self.fit_epoch_sequential(tr, epoch),
            ParallelMode::Hogwild => self.fit_epoch_hogwild(tr, epoch),
            ParallelMode::TwoPass => self.fit_epoch_twopass(tr, epoch),
        }
    }
}
