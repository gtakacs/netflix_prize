// Restricted Boltzmann Machine for collaborative filtering (conditional RBM).
// Clean variant — without mini MF and without lr_bif_bug compatibility.
// Helpers are `pub` to be reused by `rbmx2b.rs` (legacy variant with MF + bug flag).
use crate::{Dataset, MaskedDataset, Regressor, get_users, rand_array2, rand_array3, sigmoid};
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::write_npy;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};


pub const K: usize = 5; // rating categories 1..=5

#[inline]
pub fn freq_bin(day_cnt: u32, n_freq_bins: usize) -> usize {
    if n_freq_bins == 0 || day_cnt <= 1 { return 0; }
    (day_cnt as f32).log2().floor().min((n_freq_bins - 1) as f32) as usize
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum HiddenType {
    Bernoulli,
    Bipolar,
    NReLU,
    TruncExp(f32, f32),
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum VisibleType {
    Softmax,
    TruncExp(f32, f32),
}

#[derive(Clone, Copy, Debug)]
pub struct RxConfig {
    pub hidden_type: HiddenType,
    pub visible_type: VisibleType,

    // Temperature for Boltzmann distribution (1.0 = standard)
    pub temperature: f32,

    pub n_hidden: usize,
    pub n_epochs: usize,
    pub seed: u64,
    pub shuffle_users: bool,
    pub init_sigma: f32,

    pub batch_size: usize,
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,

    // Per-user visible bias
    pub lr_bu: f32,
    pub wd_bu: f32,

    // Per-user-day visible bias
    pub lr_but: f32,
    pub wd_but: f32,

    // CD-k schedule
    pub cd_start: usize,
    pub cd_inc_every: usize,
    pub cd_inc_by: usize,
    pub cd_max: usize,

    // Conditional RBM: include rated/unrated vector r
    pub use_conditional: bool,
    // If true, include all pr-set pairs in r (train + pr). If false, only pr is_test pairs.
    pub r_include_pr_all: bool,

    // Speed-ups (toggleable)
    pub save_w: bool,

    // Factored RBM: decompose W = P @ Q (and Wc = Pc @ Q for TruncExp)
    pub n_factors: Option<usize>,

    // Item-user day frequency bin bias (0 = disabled)
    pub n_freq_bins: usize,
    pub lr_bif: f32,
    pub wd_bif: f32,
}

pub struct RxModel {
    cfg: RxConfig,
    n_users: usize,
    n_items: usize,

    // Parameters (Softmax visible)
    w: Array3<f32>,  // [item, rating, hidden]
    bv: Array2<f32>, // [item, rating]
    bh: Array1<f32>, // [hidden]
    d: Option<Array2<f32>>, // [item, hidden]
    bu: Array2<f32>, // [user, K]
    but: Vec<Vec<[f32; K]>>,  // [user][day_idx][k]

    // Parameters (TruncExp visible)
    wc: Array2<f32>,  // [item, hidden]
    bvc: Array1<f32>, // [item]
    buc: Array1<f32>, // [user]
    butc: Vec<Vec<f32>>,  // [user][day_idx]

    // Momentum buffers (Softmax)
    mw: Array3<f32>,
    mbv: Array2<f32>,
    mbh: Array1<f32>,
    md: Option<Array2<f32>>,
    mbu: Array2<f32>, // [user, K]
    mbut: Vec<Vec<[f32; K]>>, // [user][day_idx][k]

    // Momentum buffers (TruncExp)
    mwc: Array2<f32>,
    mbvc: Array1<f32>,
    mbuc: Array1<f32>,
    mbutc: Vec<Vec<f32>>,

    // Factored Softmax visible: w ≈ p @ q
    p: Option<Array3<f32>>,   // [n_items, K, n_factors]
    mp: Option<Array3<f32>>,  // momentum P

    // Factored TruncExp visible: wc ≈ pc @ q
    pc: Option<Array2<f32>>,  // [n_items, n_factors]
    mpc: Option<Array2<f32>>, // momentum Pc

    // Shared factor→hidden mapping
    q: Option<Array2<f32>>,   // [n_factors, n_hidden]
    mq: Option<Array2<f32>>,  // momentum Q

    // Item-user day frequency bin bias (Softmax): bif[item, freq_bin, K]
    bif: Array3<f32>,
    mbif: Array3<f32>,
    // Item-user day frequency bin bias (TruncExp): bifc[item, freq_bin]
    bifc: Array2<f32>,
    mbifc: Array2<f32>,
    // Day counts per user per day_idx (for freq_bin lookup)
    user_day_cnts: Vec<Vec<u32>>,

    // Training data by user: (item, rating, day_idx)
    user_ratings: Vec<Vec<(usize, u8, usize)>>,
    // Continuous target values per user per rating (from residuals, for TruncExp visible)
    user_targets: Vec<Vec<f32>>,
    // Sorted distinct days per user (for day_idx lookup)
    user_days: Vec<Vec<i16>>,
    // Rated/unrated items (train + pr test pairs)
    user_r_items: Option<Vec<Vec<usize>>>,
    // Cached hidden expected values for prediction (populated after each epoch)
    pred_cache: Vec<Vec<f32>>,

    rng: StdRng,
}

struct ThreadGrads {
    w: Array3<f32>,
    bv: Array2<f32>,
    bh: Array1<f32>,
    d: Option<Array2<f32>>,
    // TruncExp visible
    wc: Array2<f32>,
    bvc: Array1<f32>,
    // Factored
    p: Option<Array3<f32>>,   // [n_items, K, n_factors] — Softmax
    pc: Option<Array2<f32>>,  // [n_items, n_factors] — TruncExp
    q: Option<Array2<f32>>,   // [n_factors, n_hidden] — shared
}

impl ThreadGrads {
    fn new(n_items: usize, n_hidden: usize, use_d: bool, vis_cont: bool, n_factors: Option<usize>) -> Self {
        let factored = n_factors.is_some();
        let nf = n_factors.unwrap_or(0);
        Self {
            w: if vis_cont || factored { Array3::zeros((0, 0, 0)) } else { Array3::zeros((n_items, K, n_hidden)) },
            bv: if vis_cont { Array2::zeros((0, 0)) } else { Array2::zeros((n_items, K)) },
            bh: Array1::zeros(n_hidden),
            d: if use_d { Some(Array2::zeros((n_items, n_hidden))) } else { None },
            wc: if vis_cont && !factored { Array2::zeros((n_items, n_hidden)) } else { Array2::zeros((0, 0)) },
            bvc: if vis_cont { Array1::zeros(n_items) } else { Array1::zeros(0) },
            p: if factored && !vis_cont { Some(Array3::zeros((n_items, K, nf))) } else { None },
            pc: if factored && vis_cont { Some(Array2::zeros((n_items, nf))) } else { None },
            q: if factored { Some(Array2::zeros((nf, n_hidden))) } else { None },
        }
    }

    fn zero(&mut self) {
        self.w.fill(0.0);
        self.bv.fill(0.0);
        self.bh.fill(0.0);
        if let Some(d) = self.d.as_mut() { d.fill(0.0); }
        self.wc.fill(0.0);
        self.bvc.fill(0.0);
        if let Some(p) = self.p.as_mut() { p.fill(0.0); }
        if let Some(pc) = self.pc.as_mut() { pc.fill(0.0); }
        if let Some(q) = self.q.as_mut() { q.fill(0.0); }
    }
}

impl RxModel {
    fn cd_steps_for_epoch(&self, epoch: usize) -> usize {
        if self.cfg.cd_inc_every == 0 {
            return self.cfg.cd_start.max(1);
        }
        let incs = (epoch.saturating_sub(1)) / self.cfg.cd_inc_every;
        let steps = self.cfg.cd_start + incs * self.cfg.cd_inc_by;
        steps.min(self.cfg.cd_max).max(1)
    }

    fn apply_grads(
        &mut self,
        batch_count: f32,
        grad: &ThreadGrads,
    ) {
        let lr = self.cfg.lr / batch_count.max(1.0);
        let mom = self.cfg.momentum;
        let wd = self.cfg.weight_decay;

        match self.cfg.visible_type {
            VisibleType::Softmax => {
                // W with weight decay
                for ((w, mw), gw) in self.w.iter_mut().zip(self.mw.iter_mut()).zip(grad.w.iter()) {
                    *mw = mom * *mw + lr * (*gw - wd * *w);
                    *w += *mw;
                }
                // Visible bias
                for ((b, mb), gb) in self.bv.iter_mut().zip(self.mbv.iter_mut()).zip(grad.bv.iter()) {
                    *mb = mom * *mb + lr * *gb;
                    *b += *mb;
                }
            }
            VisibleType::TruncExp(..) => {
                // Wc with weight decay
                for ((w, mw), gw) in self.wc.iter_mut().zip(self.mwc.iter_mut()).zip(grad.wc.iter()) {
                    *mw = mom * *mw + lr * (*gw - wd * *w);
                    *w += *mw;
                }
                // Visible bias (continuous)
                for ((b, mb), gb) in self.bvc.iter_mut().zip(self.mbvc.iter_mut()).zip(grad.bvc.iter()) {
                    *mb = mom * *mb + lr * *gb;
                    *b += *mb;
                }
            }
        }
        // Hidden bias
        for ((b, mb), gb) in self.bh.iter_mut().zip(self.mbh.iter_mut()).zip(grad.bh.iter()) {
            *mb = mom * *mb + lr * *gb;
            *b += *mb;
        }
        // Conditional D with weight decay
        if let (Some(d), Some(md), Some(gd)) = (self.d.as_mut(), self.md.as_mut(), grad.d.as_ref()) {
            for ((w, mw), gw) in d.iter_mut().zip(md.iter_mut()).zip(gd.iter()) {
                *mw = mom * *mw + lr * (*gw - wd * *w);
                *w += *mw;
            }
        }
        // Factored P (Softmax)
        if let (Some(p), Some(mp), Some(gp)) = (self.p.as_mut(), self.mp.as_mut(), grad.p.as_ref()) {
            for ((w, mw), gw) in p.iter_mut().zip(mp.iter_mut()).zip(gp.iter()) {
                *mw = mom * *mw + lr * (*gw - wd * *w);
                *w += *mw;
            }
        }
        // Factored Pc (TruncExp)
        if let (Some(pc), Some(mpc), Some(gpc)) = (self.pc.as_mut(), self.mpc.as_mut(), grad.pc.as_ref()) {
            for ((w, mw), gw) in pc.iter_mut().zip(mpc.iter_mut()).zip(gpc.iter()) {
                *mw = mom * *mw + lr * (*gw - wd * *w);
                *w += *mw;
            }
        }
        // Factored Q (shared)
        if let (Some(q), Some(mq), Some(gq)) = (self.q.as_mut(), self.mq.as_mut(), grad.q.as_ref()) {
            for ((w, mw), gw) in q.iter_mut().zip(mq.iter_mut()).zip(gq.iter()) {
                *mw = mom * *mw + lr * (*gw - wd * *w);
                *w += *mw;
            }
        }
    }

    fn apply_bu_grad(&mut self, u: usize, grad_bu: &[f32; K]) {
        let mom = self.cfg.momentum;
        let lr_bu = self.cfg.lr_bu;
        let wd_bu = self.cfg.wd_bu;
        for k in 0..K {
            let m = mom * self.mbu[[u, k]] + lr_bu * (grad_bu[k] - wd_bu * self.bu[[u, k]]);
            self.mbu[[u, k]] = m;
            self.bu[[u, k]] += m;
        }
    }

    fn apply_but_grad(&mut self, u: usize, grad_but: &[[f32; K]]) {
        let mom = self.cfg.momentum;
        let lr_but = self.cfg.lr_but;
        let wd_but = self.cfg.wd_but;
        for (di, g) in grad_but.iter().enumerate() {
            for k in 0..K {
                let m = mom * self.mbut[u][di][k] + lr_but * (g[k] - wd_but * self.but[u][di][k]);
                self.mbut[u][di][k] = m;
                self.but[u][di][k] += m;
            }
        }
    }

    fn apply_buc_grad(&mut self, u: usize, grad_buc: f32) {
        let mom = self.cfg.momentum;
        let lr_bu = self.cfg.lr_bu;
        let wd_bu = self.cfg.wd_bu;
        let m = mom * self.mbuc[u] + lr_bu * (grad_buc - wd_bu * self.buc[u]);
        self.mbuc[u] = m;
        self.buc[u] += m;
    }

    fn apply_butc_grad(&mut self, u: usize, grad_butc: &[f32]) {
        let mom = self.cfg.momentum;
        let lr_but = self.cfg.lr_but;
        let wd_but = self.cfg.wd_but;
        for (di, &g) in grad_butc.iter().enumerate() {
            let m = mom * self.mbutc[u][di] + lr_but * (g - wd_but * self.butc[u][di]);
            self.mbutc[u][di] = m;
            self.butc[u][di] += m;
        }
    }

    fn freq_bin_for_day(&self, u: usize, day: i16) -> usize {
        if self.cfg.n_freq_bins == 0 { return 0; }
        if let Ok(di) = self.user_days[u].binary_search(&day) {
            freq_bin(self.user_day_cnts[u][di], self.cfg.n_freq_bins)
        } else {
            0
        }
    }

    /// Apply per-example bif gradient (Softmax) using CD indicators
    fn apply_bif_grad(&mut self, u: usize, v_state: &[usize]) {
        let n_freq_bins = self.cfg.n_freq_bins;
        if n_freq_bins == 0 { return; }
        let mom = self.cfg.momentum;
        let lr = self.cfg.lr_bif / self.cfg.batch_size as f32;
        let wd = self.cfg.wd_bif;
        for (idx, &(item, rating, day_idx)) in self.user_ratings[u].iter().enumerate() {
            let f = freq_bin(self.user_day_cnts[u][day_idx], n_freq_bins);
            let k_data = rating as usize;
            let k_model = v_state[idx];
            // Positive phase
            let m = mom * self.mbif[[item, f, k_data]] + lr * (1.0 - wd * self.bif[[item, f, k_data]]);
            self.mbif[[item, f, k_data]] = m;
            self.bif[[item, f, k_data]] += m;
            // Negative phase
            if k_model != k_data {
                let m = mom * self.mbif[[item, f, k_model]] + lr * (-1.0 - wd * self.bif[[item, f, k_model]]);
                self.mbif[[item, f, k_model]] = m;
                self.bif[[item, f, k_model]] += m;
            }
        }
    }

    /// Apply per-example bifc gradient (TruncExp) using CD indicators
    fn apply_bifc_grad(&mut self, u: usize, v_model: &[f32]) {
        let n_freq_bins = self.cfg.n_freq_bins;
        if n_freq_bins == 0 { return; }
        let mom = self.cfg.momentum;
        let lr = self.cfg.lr_bif / self.cfg.batch_size as f32;
        let wd = self.cfg.wd_bif;
        for (idx, &(item, _, day_idx)) in self.user_ratings[u].iter().enumerate() {
            let f = freq_bin(self.user_day_cnts[u][day_idx], n_freq_bins);
            let dv = self.user_targets[u][idx] - v_model[idx];
            let m = mom * self.mbifc[[item, f]] + lr * (dv - wd * self.bifc[[item, f]]);
            self.mbifc[[item, f]] = m;
            self.bifc[[item, f]] += m;
        }
    }

    fn fill_pred_cache(&mut self) {
        for u in 0..self.n_users {
            if self.user_ratings[u].is_empty() {
                continue;
            }
            let acts = self.compute_hidden_acts(u);
            self.pred_cache[u] = hidden_expected(&acts, self.cfg.hidden_type);
        }
    }

    /// Compute effective bias bu[u,k] + but[u,day_idx,k] + pu[u,:] · qi[i,k,:] for each of this user's distinct days
    /// NOTE: the MF part is item-dependent so we DON'T include it here — it's added per-item in logit computation
    fn effective_bu(&self, u: usize) -> Vec<[f32; K]> {
        let n_days = self.user_days[u].len();
        let mut eff = Vec::with_capacity(n_days);
        for di in 0..n_days {
            let mut eb = [0.0f32; K];
            for k in 0..K {
                eb[k] = self.bu[[u, k]] + self.but[u][di][k];
            }
            eff.push(eb);
        }
        eff
    }

    /// Compute effective bias for a single day (for prediction)
    fn effective_bu_day(&self, u: usize, day: i16) -> [f32; K] {
        let mut eb = [0.0f32; K];
        for k in 0..K {
            eb[k] = self.bu[[u, k]];
        }
        if let Ok(di) = self.user_days[u].binary_search(&day) {
            for k in 0..K {
                eb[k] += self.but[u][di][k];
            }
        }
        eb
    }

    /// Compute effective continuous user bias buc[u] + butc[u][day_idx] per day
    fn effective_buc(&self, u: usize) -> Vec<f32> {
        let n_days = self.user_days[u].len();
        let mut eff = Vec::with_capacity(n_days);
        for di in 0..n_days {
            eff.push(self.buc[u] + self.butc[u][di]);
        }
        eff
    }

    /// Compute effective continuous user bias for a single day (for prediction)
    fn effective_buc_day(&self, u: usize, day: i16) -> f32 {
        let mut eb = self.buc[u];
        if let Ok(di) = self.user_days[u].binary_search(&day) {
            eb += self.butc[u][di];
        }
        eb
    }

    fn fit_epoch_sequential(&mut self, epoch: usize) {
        let users = get_users(self.n_users, self.cfg.shuffle_users, self.cfg.seed, epoch);
        let cd_steps = self.cd_steps_for_epoch(epoch);
        let vis_cont = matches!(self.cfg.visible_type, VisibleType::TruncExp(..));

        let mut grad = ThreadGrads::new(self.n_items, self.cfg.n_hidden, self.d.is_some(), vis_cont, self.cfg.n_factors);

        let mut batch_count = 0usize;

        for &u in progress!(users.iter()) {
            let ratings = &self.user_ratings[u];
            if ratings.is_empty() { continue; }

            let r_items = self.user_r_items.as_ref().map(|v| v[u].as_slice());
            let r_contrib = compute_r_contrib(r_items, self.d.as_ref(), self.cfg.n_hidden);
            let n_days = self.user_days[u].len();
            let n_freq_bins = self.cfg.n_freq_bins;

            match self.cfg.visible_type {
                VisibleType::Softmax => {
                    let combined_biases: Vec<[f32; K]> = if n_freq_bins > 0 {
                        ratings.iter().map(|(item, _, day_idx)| {
                            let mut bias = [0.0f32; K];
                            let f = freq_bin(self.user_day_cnts[u][*day_idx], n_freq_bins);
                            for k in 0..K { bias[k] += self.bif[[*item, f, k]]; }
                            bias
                        }).collect()
                    } else {
                        Vec::new()
                    };
                    let eff_bu = self.effective_bu(u);
                    let (grad_bu, grad_but, v_state) = accumulate_user_grad(
                        ratings, r_items, cd_steps, self.cfg.n_hidden, self.cfg.hidden_type,
                        &self.w, &self.bv, &self.bh, self.d.as_ref(), &eff_bu, n_days,
                        &mut self.rng, r_contrib.as_deref(),
                        self.cfg.temperature,
                        &mut grad.w, &mut grad.bv, &mut grad.bh, &mut grad.d,
                        self.p.as_ref(), self.q.as_ref(), &mut grad.p, &mut grad.q,
                        if n_freq_bins > 0 { Some(&combined_biases) } else { None },
                    );
                    self.apply_bu_grad(u, &grad_bu);
                    self.apply_but_grad(u, &grad_but);
                    self.apply_bif_grad(u, &v_state);
                }
                VisibleType::TruncExp(v_min, v_max) => {
                    let combined_biases_c: Vec<f32> = if n_freq_bins > 0 {
                        ratings.iter().map(|(item, _, day_idx)| {
                            let f = freq_bin(self.user_day_cnts[u][*day_idx], n_freq_bins);
                            self.bifc[[*item, f]]
                        }).collect()
                    } else {
                        Vec::new()
                    };
                    let eff_buc = self.effective_buc(u);
                    let targets = &self.user_targets[u];
                    let (grad_buc, grad_butc, v_model) = accumulate_user_grad_cont(
                        ratings, targets, r_items, cd_steps, self.cfg.n_hidden, self.cfg.hidden_type,
                        &self.wc, &self.bvc, &self.bh, self.d.as_ref(), &eff_buc, n_days,
                        &mut self.rng, r_contrib.as_deref(),
                        self.cfg.temperature, v_min, v_max,
                        &mut grad.wc, &mut grad.bvc, &mut grad.bh, &mut grad.d,
                        self.pc.as_ref(), self.q.as_ref(), &mut grad.pc, &mut grad.q,
                        if n_freq_bins > 0 { Some(&combined_biases_c) } else { None },
                    );
                    self.apply_buc_grad(u, grad_buc);
                    self.apply_butc_grad(u, &grad_butc);
                    self.apply_bifc_grad(u, &v_model);
                }
            }

            batch_count += 1;
            if batch_count >= self.cfg.batch_size {
                self.apply_grads(batch_count as f32, &grad);
                grad.zero();
                batch_count = 0;
            }
        }

        if batch_count > 0 {
            self.apply_grads(batch_count as f32, &grad);
        }
    }

    fn compute_hidden_acts(&self, u: usize) -> Vec<f32> {
        let ratings = &self.user_ratings[u];
        let r_items = self.user_r_items.as_ref().map(|v| v[u].as_slice());
        let r_contrib = compute_r_contrib(r_items, self.d.as_ref(), self.cfg.n_hidden);
        let mut acts = match self.cfg.visible_type {
            VisibleType::Softmax => {
                let v_state: Vec<usize> = ratings.iter().map(|(_, r, _)| *r as usize).collect();
                hidden_acts_from_state(
                    &self.bh, &self.w, self.d.as_ref(), ratings, &v_state,
                    r_items, r_contrib.as_deref(), self.cfg.n_hidden,
                    self.p.as_ref(), self.q.as_ref(),
                )
            }
            VisibleType::TruncExp(..) => {
                let targets = &self.user_targets[u];
                hidden_acts_from_cont_state(
                    &self.bh, &self.wc, self.d.as_ref(), ratings, targets,
                    r_items, r_contrib.as_deref(), self.cfg.n_hidden,
                    self.pc.as_ref(), self.q.as_ref(),
                )
            }
        };
        acts.iter_mut().for_each(|a| *a /= self.cfg.temperature);
        acts
    }

    fn get_hidden_expected(&self, u: usize) -> Vec<f32> {
        if !self.pred_cache[u].is_empty() {
            return self.pred_cache[u].clone();
        }
        let acts = self.compute_hidden_acts(u);
        hidden_expected(&acts, self.cfg.hidden_type)
    }

    fn predict_probs(&self, u: usize, i: usize, day: i16) -> Array1<f32> {
        let h_exp = self.get_hidden_expected(u);
        let mut eff_bu = self.effective_bu_day(u, day);
        if self.cfg.n_freq_bins > 0 {
            let f = self.freq_bin_for_day(u, day);
            for k in 0..K { eff_bu[k] += self.bif[[i, f, k]]; }
        }
        probs_with_ph(&self.bv, &self.w, i, &h_exp, self.cfg.n_hidden, &eff_bu, self.cfg.temperature,
                      self.p.as_ref(), self.q.as_ref())
    }
}

/// Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi)
pub fn norm_pdf(x: f32) -> f32 {
    const INV_SQRT_2PI: f32 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF using Abramowitz & Stegun 7.1.26 erf approximation
pub fn norm_cdf(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let erf = 1.0 - (0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3
        - 1.453152027 * t4 + 1.061405429 * t5) * (-a * a).exp();
    let cdf = 0.5 * (1.0 + erf);
    if x >= 0.0 { cdf } else { 1.0 - cdf }
}

/// Expected value of max(0, x + N(0, sigma)) where sigma = sqrt(sigmoid(x))
pub fn nrelu_mean_scalar(x: f32) -> f32 {
    let sig = sigmoid(x);
    let sigma = sig.sqrt();
    if sigma < 1e-10 {
        return x.max(0.0);
    }
    let z = x / sigma;
    x * norm_cdf(z) + sigma * norm_pdf(z)
}

/// Expected value of Continuous Bernoulli distribution on [0,1] with natural parameter a
pub fn cont_bernoulli_mean(a: f32) -> f32 {
    if a.abs() < 1e-3 {
        // Taylor: 0.5 + a/12 - a³/720 + a⁵/30240
        let a2 = a * a;
        0.5 + a / 12.0 - a * a2 / 720.0 + a * a2 * a2 / 30240.0
    } else {
        1.0 / (1.0 - (-a).exp()) - 1.0 / a
    }
}

/// Expected value of Continuous Bernoulli on [h_min, h_max] (truncated exponential)
pub fn cont_bernoulli_mean_range(a: f32, h_min: f32, h_max: f32) -> f32 {
    let l = h_max - h_min;
    h_min + l * cont_bernoulli_mean(a * l)
}

/// Sample from Continuous Bernoulli on [h_min, h_max] (truncated exponential, inverse CDF)
pub fn cont_bernoulli_sample_range(rng: &mut StdRng, a: f32, h_min: f32, h_max: f32) -> f32 {
    let l = h_max - h_min;
    h_min + l * cont_bernoulli_sample(rng, a * l)
}

/// Sample from Continuous Bernoulli distribution on [0,1] with natural parameter a (inverse CDF)
pub fn cont_bernoulli_sample(rng: &mut StdRng, a: f32) -> f32 {
    let u: f32 = rng.random();
    if a.abs() < 1e-3 {
        (u + u * (1.0 - u) * a * 0.5).clamp(0.0, 1.0)
    } else if a > 20.0 {
        (1.0 + u.max(1e-30).ln() / a).clamp(0.0, 1.0)
    } else if a < -20.0 {
        (-(1.0 - u).max(1e-30).ln() / (-a)).clamp(0.0, 1.0)
    } else {
        (1.0 + u * (a.exp() - 1.0)).ln() / a
    }
}

/// Compute raw pre-activations for hidden units given visible state (no activation function)
pub fn hidden_acts_from_state(
    bh: &Array1<f32>,
    w: &Array3<f32>,
    d: Option<&Array2<f32>>,
    items: &[(usize, u8, usize)],
    v_state: &[usize],
    r_items: Option<&[usize]>,
    r_contrib: Option<&[f32]>,
    n_hidden: usize,
    p: Option<&Array3<f32>>,
    q: Option<&Array2<f32>>,
) -> Vec<f32> {
    let mut act = bh.to_vec();

    if let (Some(p), Some(q)) = (p, q) {
        // Factored: accumulate in factor space, then project to hidden
        let n_factors = q.dim().0;
        let mut z = vec![0.0f32; n_factors];
        for (idx, (item, _, _)) in items.iter().enumerate() {
            let k = v_state[idx];
            for f in 0..n_factors {
                z[f] += p[[*item, k, f]];
            }
        }
        for j in 0..n_hidden {
            let mut s = 0.0f32;
            for f in 0..n_factors {
                s += z[f] * q[[f, j]];
            }
            act[j] += s;
        }
    } else {
        for (idx, (item, _, _)) in items.iter().enumerate() {
            let k = v_state[idx];
            for j in 0..n_hidden {
                act[j] += w[[*item, k, j]];
            }
        }
    }

    if let Some(rc) = r_contrib {
        for j in 0..n_hidden {
            act[j] += rc[j];
        }
    } else if let (Some(r_items), Some(d)) = (r_items, d) {
        for &item in r_items {
            for j in 0..n_hidden {
                act[j] += d[[item, j]];
            }
        }
    }

    act
}

/// Compute raw pre-activations for hidden units given CONTINUOUS visible state
pub fn hidden_acts_from_cont_state(
    bh: &Array1<f32>,
    wc: &Array2<f32>,
    d: Option<&Array2<f32>>,
    items: &[(usize, u8, usize)],
    v_cont: &[f32],
    r_items: Option<&[usize]>,
    r_contrib: Option<&[f32]>,
    n_hidden: usize,
    pc: Option<&Array2<f32>>,
    q: Option<&Array2<f32>>,
) -> Vec<f32> {
    let mut act = bh.to_vec();

    if let (Some(pc), Some(q)) = (pc, q) {
        // Factored: accumulate in factor space, then project to hidden
        let n_factors = q.dim().0;
        let mut z = vec![0.0f32; n_factors];
        for (idx, (item, _, _)) in items.iter().enumerate() {
            let v = v_cont[idx];
            for f in 0..n_factors {
                z[f] += v * pc[[*item, f]];
            }
        }
        for j in 0..n_hidden {
            let mut s = 0.0f32;
            for f in 0..n_factors {
                s += z[f] * q[[f, j]];
            }
            act[j] += s;
        }
    } else {
        for (idx, (item, _, _)) in items.iter().enumerate() {
            let v = v_cont[idx];
            for j in 0..n_hidden {
                act[j] += wc[[*item, j]] * v;
            }
        }
    }

    if let Some(rc) = r_contrib {
        for j in 0..n_hidden {
            act[j] += rc[j];
        }
    } else if let (Some(r_items), Some(d)) = (r_items, d) {
        for &item in r_items {
            for j in 0..n_hidden {
                act[j] += d[[item, j]];
            }
        }
    }

    act
}

/// Convert raw pre-activations to expected hidden values
pub fn hidden_expected(acts: &[f32], hidden_type: HiddenType) -> Vec<f32> {
    match hidden_type {
        HiddenType::Bernoulli => acts.iter().map(|&x| sigmoid(x)).collect(),
        HiddenType::Bipolar => acts.iter().map(|&x| 2.0 * sigmoid(x) - 1.0).collect(),
        HiddenType::NReLU => acts.iter().map(|&x| nrelu_mean_scalar(x)).collect(),
        HiddenType::TruncExp(h_min, h_max) => acts.iter().map(|&x| cont_bernoulli_mean_range(x, h_min, h_max)).collect(),
    }
}

/// Sample hidden units from raw pre-activations
pub fn sample_hidden_units(rng: &mut StdRng, acts: &[f32], hidden_type: HiddenType) -> Vec<f32> {
    match hidden_type {
        HiddenType::Bernoulli => {
            let mut h = Vec::with_capacity(acts.len());
            for &a in acts {
                let p = sigmoid(a);
                let r: f32 = rng.random();
                h.push(if r < p { 1.0 } else { 0.0 });
            }
            h
        }
        HiddenType::Bipolar => {
            let mut h = Vec::with_capacity(acts.len());
            for &a in acts {
                let p = sigmoid(a);
                let r: f32 = rng.random();
                h.push(if r < p { 1.0 } else { -1.0 });
            }
            h
        }
        HiddenType::NReLU => {
            let std_normal = Normal::<f32>::new(0.0, 1.0).unwrap();
            let mut h = Vec::with_capacity(acts.len());
            for &a in acts {
                let sigma = sigmoid(a).sqrt();
                let noise = std_normal.sample(rng) * sigma;
                h.push((a + noise).max(0.0));
            }
            h
        }
        HiddenType::TruncExp(h_min, h_max) => {
            let mut h = Vec::with_capacity(acts.len());
            for &a in acts {
                h.push(cont_bernoulli_sample_range(rng, a, h_min, h_max));
            }
            h
        }
    }
}

pub fn sample_softmax(rng: &mut StdRng, logits: [f32; K]) -> usize {
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v > max { max = v; }
    }
    let mut exps = [0.0f32; K];
    let mut sum = 0.0f32;
    for k in 0..K {
        let e = (logits[k] - max).exp();
        exps[k] = e;
        sum += e;
    }
    let mut r = rng.random::<f32>() * sum;
    for k in 0..K {
        if r <= exps[k] {
            return k;
        }
        r -= exps[k];
    }
    K - 1
}

/// Compute logits for an item, optionally adding a per-rating extra bias vector
/// (used by callers to inject bif and/or MF contributions).
pub fn item_logits(bv: &Array2<f32>, w: &Array3<f32>, item: usize, h: &[f32], n_hidden: usize, bu_eff: &[f32],
                   p: Option<&Array3<f32>>, z_h: Option<&[f32]>,
                   extra_bias: Option<&[f32; K]>) -> [f32; K] {
    let mut logits = [0.0f32; K];
    if let (Some(p), Some(z_h)) = (p, z_h) {
        let n_factors = p.dim().2;
        for k in 0..K {
            let mut s = bv[[item, k]] + bu_eff[k];
            for f in 0..n_factors {
                s += p[[item, k, f]] * z_h[f];
            }
            logits[k] = s;
        }
    } else {
        for k in 0..K {
            let mut s = bv[[item, k]] + bu_eff[k];
            for j in 0..n_hidden {
                s += h[j] * w[[item, k, j]];
            }
            logits[k] = s;
        }
    }
    if let Some(extra) = extra_bias {
        for k in 0..K {
            logits[k] += extra[k];
        }
    }
    logits
}

pub fn compute_r_contrib(
    r_items: Option<&[usize]>,
    d: Option<&Array2<f32>>,
    n_hidden: usize,
) -> Option<Vec<f32>> {
    let r_items = r_items?;
    let d = d?;
    let mut rc = vec![0.0f32; n_hidden];
    for &item in r_items {
        for j in 0..n_hidden {
            rc[j] += d[[item, j]];
        }
    }
    Some(rc)
}

/// Returns (grad_bu [K], grad_but [n_days][K], v_state)
#[allow(clippy::too_many_arguments)]
pub fn accumulate_user_grad(
    ratings: &[(usize, u8, usize)],
    r_items: Option<&[usize]>,
    cd_steps: usize,
    n_hidden: usize,
    hidden_type: HiddenType,
    w: &Array3<f32>,
    bv: &Array2<f32>,
    bh: &Array1<f32>,
    d: Option<&Array2<f32>>,
    eff_bu: &[[f32; K]],  // [day_idx] -> effective bu+but per k
    n_days: usize,
    rng: &mut StdRng,
    r_contrib: Option<&[f32]>,
    temperature: f32,
    grad_w: &mut Array3<f32>,
    grad_bv: &mut Array2<f32>,
    grad_bh: &mut Array1<f32>,
    grad_d: &mut Option<Array2<f32>>,
    p: Option<&Array3<f32>>,
    q: Option<&Array2<f32>>,
    grad_p: &mut Option<Array3<f32>>,
    grad_q: &mut Option<Array2<f32>>,
    extra_biases: Option<&Vec<[f32; K]>>,  // optional precomputed per-rating bias (e.g. bif)
) -> ([f32; K], Vec<[f32; K]>, Vec<usize>) {
    let mut grad_bu = [0.0f32; K];
    let mut grad_but = vec![[0.0f32; K]; n_days];
    if ratings.is_empty() { return (grad_bu, grad_but, Vec::new()); }

    let mut v_state: Vec<usize> = ratings.iter().map(|(_, r, _)| *r as usize).collect();

    // Positive phase
    let mut acts_pos = hidden_acts_from_state(bh, w, d, ratings, &v_state, r_items, r_contrib, n_hidden, p, q);
    acts_pos.iter_mut().for_each(|a| *a /= temperature);
    let expected_pos = hidden_expected(&acts_pos, hidden_type);

    // CD loop
    let mut acts = acts_pos.clone();
    let factored = p.is_some() && q.is_some();
    let n_factors = if let Some(q) = q { q.dim().0 } else { 0 };
    for _ in 0..cd_steps {
        let h_sample = sample_hidden_units(rng, &acts, hidden_type);
        let z_h = if factored {
            let q = q.unwrap();
            let mut z = vec![0.0f32; n_factors];
            for f in 0..n_factors {
                let mut s = 0.0f32;
                for j in 0..n_hidden {
                    s += q[[f, j]] * h_sample[j];
                }
                z[f] = s;
            }
            Some(z)
        } else {
            None
        };
        for (idx, (item, _, day_idx)) in ratings.iter().enumerate() {
            let extra_ref = extra_biases.map(|mb| &mb[idx]);
            let mut logits = item_logits(bv, w, *item, &h_sample, n_hidden, &eff_bu[*day_idx],
                                         p, z_h.as_deref(), extra_ref);
            for k in 0..K { logits[k] /= temperature; }
            let k = sample_softmax(rng, logits);
            v_state[idx] = k;
        }
        acts = hidden_acts_from_state(bh, w, d, ratings, &v_state, r_items, r_contrib, n_hidden, p, q);
        acts.iter_mut().for_each(|a| *a /= temperature);
    }
    let expected_neg = hidden_expected(&acts, hidden_type);

    if let (Some(_p), Some(q), Some(gp), Some(gq)) = (p, q, grad_p.as_mut(), grad_q.as_mut()) {
        // Factored gradient accumulation
        let mut qep = vec![0.0f32; n_factors];
        let mut qen = vec![0.0f32; n_factors];
        for f in 0..n_factors {
            let mut sp = 0.0f32;
            let mut sn = 0.0f32;
            for j in 0..n_hidden {
                sp += q[[f, j]] * expected_pos[j];
                sn += q[[f, j]] * expected_neg[j];
            }
            qep[f] = sp;
            qen[f] = sn;
        }
        for (idx, (item, rating, day_idx)) in ratings.iter().enumerate() {
            let k_data = *rating as usize;
            let k_model = v_state[idx];
            for f in 0..n_factors {
                gp[[*item, k_data, f]] += qep[f];
                gp[[*item, k_model, f]] -= qen[f];
            }
            for j in 0..n_hidden {
                for f in 0..n_factors {
                    gq[[f, j]] += p.unwrap()[[*item, k_data, f]] * expected_pos[j]
                                - p.unwrap()[[*item, k_model, f]] * expected_neg[j];
                }
            }
            grad_bv[[*item, k_data]] += 1.0;
            grad_bv[[*item, k_model]] -= 1.0;
            grad_bu[k_data] += 1.0;
            grad_bu[k_model] -= 1.0;
            grad_but[*day_idx][k_data] += 1.0;
            grad_but[*day_idx][k_model] -= 1.0;
        }
    } else {
        // Standard (non-factored) gradient accumulation
        for (idx, (item, rating, day_idx)) in ratings.iter().enumerate() {
            let k_data = *rating as usize;
            let k_model = v_state[idx];
            for j in 0..n_hidden {
                grad_w[[*item, k_data, j]] += expected_pos[j];
                grad_w[[*item, k_model, j]] -= expected_neg[j];
            }
            grad_bv[[*item, k_data]] += 1.0;
            grad_bv[[*item, k_model]] -= 1.0;
            grad_bu[k_data] += 1.0;
            grad_bu[k_model] -= 1.0;
            grad_but[*day_idx][k_data] += 1.0;
            grad_but[*day_idx][k_model] -= 1.0;
        }
    }

    for j in 0..n_hidden {
        grad_bh[j] += expected_pos[j] - expected_neg[j];
    }

    if let (Some(r_items), Some(gd)) = (r_items, grad_d.as_mut()) {
        for &item in r_items {
            for j in 0..n_hidden {
                gd[[item, j]] += expected_pos[j] - expected_neg[j];
            }
        }
    }

    (grad_bu, grad_but, v_state)
}

/// Accumulate gradients for one user — TruncExp visible units (continuous).
/// Returns (grad_buc: f32, grad_butc: Vec<f32> [n_days], v_model)
#[allow(clippy::too_many_arguments)]
pub fn accumulate_user_grad_cont(
    ratings: &[(usize, u8, usize)],
    targets: &[f32],
    r_items: Option<&[usize]>,
    cd_steps: usize,
    n_hidden: usize,
    hidden_type: HiddenType,
    wc: &Array2<f32>,
    bvc: &Array1<f32>,
    bh: &Array1<f32>,
    d: Option<&Array2<f32>>,
    eff_buc: &[f32],  // [day_idx] -> effective buc+butc
    n_days: usize,
    rng: &mut StdRng,
    r_contrib: Option<&[f32]>,
    temperature: f32,
    v_min: f32,
    v_max: f32,
    grad_wc: &mut Array2<f32>,
    grad_bvc: &mut Array1<f32>,
    grad_bh: &mut Array1<f32>,
    grad_d: &mut Option<Array2<f32>>,
    pc: Option<&Array2<f32>>,
    q: Option<&Array2<f32>>,
    grad_pc: &mut Option<Array2<f32>>,
    grad_q: &mut Option<Array2<f32>>,
    extra_biases: Option<&Vec<f32>>,  // optional precomputed per-rating bias (e.g. bifc)
) -> (f32, Vec<f32>, Vec<f32>) {
    let mut grad_buc = 0.0f32;
    let mut grad_butc = vec![0.0f32; n_days];
    if ratings.is_empty() { return (grad_buc, grad_butc, Vec::new()); }

    // Data visible values (continuous targets from residuals or raw ratings)
    let v_data = targets;
    let factored = pc.is_some() && q.is_some();
    let n_factors = if let Some(q) = q { q.dim().0 } else { 0 };

    // Positive phase
    let mut acts_pos = hidden_acts_from_cont_state(bh, wc, d, ratings, &v_data, r_items, r_contrib, n_hidden, pc, q);
    acts_pos.iter_mut().for_each(|a| *a /= temperature);
    let expected_pos = hidden_expected(&acts_pos, hidden_type);

    // CD loop
    let mut v_model = v_data.to_vec();
    let mut acts = acts_pos.clone();
    for _ in 0..cd_steps {
        let h_sample = sample_hidden_units(rng, &acts, hidden_type);
        // Sample visible from TruncExp
        if factored {
            let pc = pc.unwrap();
            let q = q.unwrap();
            let mut z_h = vec![0.0f32; n_factors];
            for f in 0..n_factors {
                let mut s = 0.0f32;
                for j in 0..n_hidden {
                    s += q[[f, j]] * h_sample[j];
                }
                z_h[f] = s;
            }
            for (idx, (item, _, day_idx)) in ratings.iter().enumerate() {
                let mut lambda = bvc[*item] + eff_buc[*day_idx];
                for f in 0..n_factors {
                    lambda += pc[[*item, f]] * z_h[f];
                }
                if let Some(mb) = extra_biases {
                    lambda += mb[idx];
                }
                lambda /= temperature;
                v_model[idx] = cont_bernoulli_sample_range(rng, lambda, v_min, v_max);
            }
        } else {
            for (idx, (item, _, day_idx)) in ratings.iter().enumerate() {
                let mut lambda = bvc[*item] + eff_buc[*day_idx];
                for j in 0..n_hidden {
                    lambda += h_sample[j] * wc[[*item, j]];
                }
                if let Some(mb) = extra_biases {
                    lambda += mb[idx];
                }
                lambda /= temperature;
                v_model[idx] = cont_bernoulli_sample_range(rng, lambda, v_min, v_max);
            }
        }
        acts = hidden_acts_from_cont_state(bh, wc, d, ratings, &v_model, r_items, r_contrib, n_hidden, pc, q);
        acts.iter_mut().for_each(|a| *a /= temperature);
    }
    let expected_neg = hidden_expected(&acts, hidden_type);

    // Accumulate gradients: positive - negative phase
    if let (Some(pc), Some(q), Some(gpc), Some(gq)) = (pc, q, grad_pc.as_mut(), grad_q.as_mut()) {
        // Factored gradient accumulation
        let mut qep = vec![0.0f32; n_factors];
        let mut qen = vec![0.0f32; n_factors];
        for f in 0..n_factors {
            let mut sp = 0.0f32;
            let mut sn = 0.0f32;
            for j in 0..n_hidden {
                sp += q[[f, j]] * expected_pos[j];
                sn += q[[f, j]] * expected_neg[j];
            }
            qep[f] = sp;
            qen[f] = sn;
        }
        for (idx, (item, _, day_idx)) in ratings.iter().enumerate() {
            let vd = v_data[idx];
            let vm = v_model[idx];
            let dv = vd - vm;
            for f in 0..n_factors {
                gpc[[*item, f]] += vd * qep[f] - vm * qen[f];
            }
            for j in 0..n_hidden {
                for f in 0..n_factors {
                    gq[[f, j]] += vd * pc[[*item, f]] * expected_pos[j]
                                - vm * pc[[*item, f]] * expected_neg[j];
                }
            }
            grad_bvc[*item] += dv;
            grad_buc += dv;
            grad_butc[*day_idx] += dv;
        }
    } else {
        for (idx, (item, _, day_idx)) in ratings.iter().enumerate() {
            let vd = v_data[idx];
            let vm = v_model[idx];
            let dv = vd - vm;
            for j in 0..n_hidden {
                grad_wc[[*item, j]] += vd * expected_pos[j] - vm * expected_neg[j];
            }
            grad_bvc[*item] += dv;
            grad_buc += dv;
            grad_butc[*day_idx] += dv;
        }
    }

    for j in 0..n_hidden {
        grad_bh[j] += expected_pos[j] - expected_neg[j];
    }

    if let (Some(r_items), Some(gd)) = (r_items, grad_d.as_mut()) {
        for &item in r_items {
            for j in 0..n_hidden {
                gd[[item, j]] += expected_pos[j] - expected_neg[j];
            }
        }
    }

    (grad_buc, grad_butc, v_model)
}

/// Predict rating for continuous visible (TruncExp): E[v_i | h_expected].
pub fn predict_cont(
    wc: &Array2<f32>,
    bvc: &Array1<f32>,
    i: usize,
    p_h: &[f32],
    n_hidden: usize,
    buc_eff: f32,
    v_min: f32,
    v_max: f32,
    pc: Option<&Array2<f32>>,
    q: Option<&Array2<f32>>,
) -> f32 {
    let mut lambda = bvc[i] + buc_eff;
    if let (Some(pc), Some(q)) = (pc, q) {
        let n_factors = q.dim().0;
        let mut z = vec![0.0f32; n_factors];
        for f in 0..n_factors {
            let mut s = 0.0f32;
            for j in 0..n_hidden {
                s += q[[f, j]] * p_h[j];
            }
            z[f] = s;
        }
        for f in 0..n_factors {
            lambda += pc[[i, f]] * z[f];
        }
    } else {
        for j in 0..n_hidden {
            lambda += p_h[j] * wc[[i, j]];
        }
    }
    cont_bernoulli_mean_range(lambda, v_min, v_max)
}

pub fn init_visible_biases_cont(tr: &Dataset, v_min: f32, v_max: f32) -> Array1<f32> {
    let l = v_max - v_min;
    let center = (v_min + v_max) * 0.5;
    let mut sums = vec![0.0f64; tr.n_items];
    let mut counts = vec![0u32; tr.n_items];

    for idx in 0..tr.n_ratings {
        if tr.is_test[idx] != 0 { continue; }
        sums[tr.item_idxs[idx] as usize] += tr.residuals[idx] as f64;
        counts[tr.item_idxs[idx] as usize] += 1;
    }

    let global_mean = sums.iter().sum::<f64>() / counts.iter().sum::<u32>().max(1) as f64;
    let mut bvc = Array1::<f32>::zeros(tr.n_items);
    for i in 0..tr.n_items {
        let mean = if counts[i] > 0 { sums[i] / counts[i] as f64 } else { global_mean };
        bvc[i] = (mean as f32 - center) * 12.0 / (l * l);
    }
    bvc
}

pub fn build_user_ratings(ds: &Dataset) -> (Vec<Vec<(usize, u8, usize)>>, Vec<Vec<f32>>, Vec<Vec<i16>>, Vec<Vec<u32>>) {
    let mut user_day_set: Vec<Vec<i16>> = vec![Vec::new(); ds.n_users];
    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 { continue; }
        let u = ds.user_idxs[idx] as usize;
        user_day_set[u].push(ds.dates[idx]);
    }
    for days in user_day_set.iter_mut() {
        days.sort_unstable();
    }
    // Count ratings per (user, day) before dedup
    let user_day_cnts: Vec<Vec<u32>> = user_day_set.iter().map(|days| {
        let mut cnts = Vec::new();
        let mut i = 0;
        while i < days.len() {
            let mut j = i + 1;
            while j < days.len() && days[j] == days[i] { j += 1; }
            cnts.push((j - i) as u32);
            i = j;
        }
        cnts
    }).collect();
    for days in user_day_set.iter_mut() {
        days.dedup();
    }

    let mut ratings: Vec<Vec<(usize, u8, usize)>> = (0..ds.n_users)
        .map(|u| Vec::with_capacity(ds.user_cnts[u] as usize))
        .collect();
    let mut targets: Vec<Vec<f32>> = (0..ds.n_users)
        .map(|u| Vec::with_capacity(ds.user_cnts[u] as usize))
        .collect();
    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 { continue; }
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let mut r = ds.raw_ratings[idx] as u8;
        r = r.saturating_sub(1);
        let day = ds.dates[idx];
        let di = user_day_set[u].binary_search(&day).unwrap();
        ratings[u].push((i, r, di));
        targets[u].push(ds.residuals[idx]);
    }

    (ratings, targets, user_day_set, user_day_cnts)
}

pub fn build_r_items(tr: &Dataset, pr: &MaskedDataset, include_pr_all: bool) -> Vec<Vec<usize>> {
    let mut items: Vec<Vec<usize>> = vec![Vec::new(); tr.n_users];

    for idx in 0..tr.n_ratings {
        let u = tr.user_idxs[idx] as usize;
        let i = tr.item_idxs[idx] as usize;
        items[u].push(i);
    }

    for idx in 0..pr.n_ratings {
        if !include_pr_all && pr.is_test[idx] == 0 { continue; }
        let u = pr.user_idxs[idx] as usize;
        let i = pr.item_idxs[idx] as usize;
        items[u].push(i);
    }

    for u in 0..items.len() {
        items[u].sort_unstable();
        items[u].dedup();
    }

    items
}

pub fn init_visible_biases(tr: &Dataset) -> Array2<f32> {
    let mut counts = vec![[0u32; K]; tr.n_items];
    let mut totals = vec![0u32; tr.n_items];
    let mut global = [0u64; K];
    let mut global_total = 0u64;

    for idx in 0..tr.n_ratings {
        if tr.is_test[idx] != 0 { continue; }
        let i = tr.item_idxs[idx] as usize;
        let r = tr.raw_ratings[idx] as usize;
        let k = r.saturating_sub(1).min(K - 1);
        counts[i][k] += 1;
        totals[i] += 1;
        global[k] += 1;
        global_total += 1;
    }

    let eps = 1e-6f32;
    let mut bv = Array2::<f32>::zeros((tr.n_items, K));
    for i in 0..tr.n_items {
        let tot_i = totals[i] as f32;
        let (use_counts, tot) = if tot_i > 0.0 {
            (counts[i], tot_i)
        } else {
            let mut tmp = [0u32; K];
            for k in 0..K { tmp[k] = global[k] as u32; }
            (tmp, global_total as f32)
        };
        for k in 0..K {
            let c = use_counts[k] as f32;
            let p = (c + eps) / (tot + (K as f32) * eps);
            bv[[i, k]] = p.ln();
        }
    }
    bv
}

impl Regressor for RxModel {
    type Config = RxConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let vis_cont = matches!(cfg.visible_type, VisibleType::TruncExp(..));
        let factored = cfg.n_factors.is_some();

        // Softmax visible parameters
        let w = if vis_cont || factored { Array3::zeros((0, 0, 0)) } else { rand_array3(tr.n_items, K, cfg.n_hidden, &mut rng, cfg.init_sigma) };
        let bv = if vis_cont { Array2::zeros((0, 0)) } else { init_visible_biases(tr) };
        let bh = Array1::<f32>::zeros(cfg.n_hidden);

        let d = if cfg.use_conditional {
            Some(rand_array2(tr.n_items, cfg.n_hidden, &mut rng, cfg.init_sigma))
        } else {
            None
        };

        let mw = if vis_cont || factored { Array3::zeros((0, 0, 0)) } else { Array3::<f32>::zeros((tr.n_items, K, cfg.n_hidden)) };
        let mbv = if vis_cont { Array2::zeros((0, 0)) } else { Array2::<f32>::zeros((tr.n_items, K)) };
        let mbh = Array1::<f32>::zeros(cfg.n_hidden);
        let md = d.as_ref().map(|_| Array2::<f32>::zeros((tr.n_items, cfg.n_hidden)));

        let bu = if vis_cont { Array2::zeros((0, 0)) } else { Array2::<f32>::zeros((tr.n_users, K)) };
        let mbu = if vis_cont { Array2::zeros((0, 0)) } else { Array2::<f32>::zeros((tr.n_users, K)) };

        // TruncExp visible parameters
        let wc = if vis_cont && !factored { rand_array2(tr.n_items, cfg.n_hidden, &mut rng, cfg.init_sigma) } else { Array2::zeros((0, 0)) };
        let bvc = if let VisibleType::TruncExp(v_min, v_max) = cfg.visible_type {
            init_visible_biases_cont(tr, v_min, v_max)
        } else { Array1::zeros(0) };
        let buc = if vis_cont { Array1::<f32>::zeros(tr.n_users) } else { Array1::zeros(0) };
        let mwc = if vis_cont && !factored { Array2::<f32>::zeros((tr.n_items, cfg.n_hidden)) } else { Array2::zeros((0, 0)) };
        let mbvc = if vis_cont { Array1::<f32>::zeros(tr.n_items) } else { Array1::zeros(0) };
        let mbuc = if vis_cont { Array1::<f32>::zeros(tr.n_users) } else { Array1::zeros(0) };

        // Factored parameters
        let (p, mp, pc, mpc, q, mq) = if let Some(nf) = cfg.n_factors {
            let q_sigma = cfg.init_sigma / (nf as f32).sqrt();
            let q_mat = Some(rand_array2(nf, cfg.n_hidden, &mut rng, q_sigma));
            let mq_mat = Some(Array2::<f32>::zeros((nf, cfg.n_hidden)));
            if vis_cont {
                let pc_mat = Some(rand_array2(tr.n_items, nf, &mut rng, cfg.init_sigma));
                let mpc_mat = Some(Array2::<f32>::zeros((tr.n_items, nf)));
                (None, None, pc_mat, mpc_mat, q_mat, mq_mat)
            } else {
                let p_mat = Some(rand_array3(tr.n_items, K, nf, &mut rng, cfg.init_sigma));
                let mp_mat = Some(Array3::<f32>::zeros((tr.n_items, K, nf)));
                (p_mat, mp_mat, None, None, q_mat, mq_mat)
            }
        } else {
            (None, None, None, None, None, None)
        };

        let (user_ratings, user_targets, user_days, user_day_cnts) = build_user_ratings(tr);

        // Item-user day frequency bin bias
        let n_freq_bins = cfg.n_freq_bins;
        let (bif, mbif) = if n_freq_bins > 0 && !vis_cont {
            (Array3::<f32>::zeros((tr.n_items, n_freq_bins, K)),
             Array3::<f32>::zeros((tr.n_items, n_freq_bins, K)))
        } else {
            (Array3::zeros((0, 0, 0)), Array3::zeros((0, 0, 0)))
        };
        let (bifc, mbifc) = if n_freq_bins > 0 && vis_cont {
            (Array2::<f32>::zeros((tr.n_items, n_freq_bins)),
             Array2::<f32>::zeros((tr.n_items, n_freq_bins)))
        } else {
            (Array2::zeros((0, 0)), Array2::zeros((0, 0)))
        };

        let but: Vec<Vec<[f32; K]>> = if vis_cont {
            Vec::new()
        } else {
            user_days.iter().map(|days| vec![[0.0f32; K]; days.len()]).collect()
        };
        let mbut: Vec<Vec<[f32; K]>> = if vis_cont {
            Vec::new()
        } else {
            user_days.iter().map(|days| vec![[0.0f32; K]; days.len()]).collect()
        };

        let butc: Vec<Vec<f32>> = if vis_cont {
            user_days.iter().map(|days| vec![0.0f32; days.len()]).collect()
        } else {
            Vec::new()
        };
        let mbutc: Vec<Vec<f32>> = if vis_cont {
            user_days.iter().map(|days| vec![0.0f32; days.len()]).collect()
        } else {
            Vec::new()
        };

        let user_r_items = if cfg.use_conditional {
            Some(build_r_items(tr, pr, cfg.r_include_pr_all))
        } else {
            None
        };
        let pred_cache = vec![Vec::new(); tr.n_users];

        Self {
            cfg,
            n_users: tr.n_users,
            n_items: tr.n_items,
            w, bv, bh, d, bu, but,
            wc, bvc, buc, butc,
            mw, mbv, mbh, md, mbu, mbut,
            mwc, mbvc, mbuc, mbutc,
            p, mp, pc, mpc, q, mq,
            bif, mbif, bifc, mbifc,
            user_day_cnts,
            user_ratings, user_targets, user_days,
            user_r_items,
            pred_cache,
            rng,
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        self.fit_epoch_sequential(epoch);
        self.fill_pred_cache();
    }

    fn save_artifacts(&self, model_name: &str, tr_set: &str, preds_dir: &str) {
        if self.cfg.save_w {
            let path = format!("{}/{}.ifeat.{}.npy", preds_dir, model_name, tr_set);
            match self.cfg.visible_type {
                VisibleType::Softmax => {
                    if let Some(p) = &self.p {
                        let (ni, nk, nf) = p.dim();
                        let p2d = p.view().into_shape_with_order((ni, nk * nf)).unwrap();
                        write_npy(&path, &p2d).unwrap();
                    } else {
                        let (ni, nk, nh) = self.w.dim();
                        let w2d = self.w.view().into_shape_with_order((ni, nk * nh)).unwrap();
                        write_npy(&path, &w2d).unwrap();
                    }
                }
                VisibleType::TruncExp(..) => {
                    if let Some(pc) = &self.pc {
                        write_npy(&path, pc).unwrap();
                    } else {
                        write_npy(&path, &self.wc).unwrap();
                    }
                }
            }
            crate::teeln!("Saved W matrix to {}", path);
        }
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let h_exp = self.get_hidden_expected(u);
        match self.cfg.visible_type {
            VisibleType::Softmax => {
                let mut eff_bu = self.effective_bu_day(u, day as i16);
                if self.cfg.n_freq_bins > 0 {
                    let f = self.freq_bin_for_day(u, day as i16);
                    for k in 0..K { eff_bu[k] += self.bif[[i, f, k]]; }
                }
                predict_with_ph(&self.bv, &self.w, i, &h_exp, self.cfg.n_hidden, &eff_bu, self.cfg.temperature,
                               self.p.as_ref(), self.q.as_ref())
            }
            VisibleType::TruncExp(v_min, v_max) => {
                let mut eff_buc = self.effective_buc_day(u, day as i16);
                if self.cfg.n_freq_bins > 0 {
                    let f = self.freq_bin_for_day(u, day as i16);
                    eff_buc += self.bifc[[i, f]];
                }
                predict_cont(&self.wc, &self.bvc, i, &h_exp, self.cfg.n_hidden, eff_buc, v_min, v_max,
                            self.pc.as_ref(), self.q.as_ref())
            }
        }
    }

    fn n_subscores(&self) -> usize {
        match self.cfg.visible_type {
            VisibleType::Softmax => K,
            VisibleType::TruncExp(..) => 0,
        }
    }

    fn subscore_names(&self) -> Vec<String> {
        (1..=K).map(|k| format!("p{k}")).collect()
    }

    fn predict_subscores(&self, u: usize, i: usize, day: i32) -> Array1<f32> {
        self.predict_probs(u, i, day as i16)
    }
}

pub fn predict_with_ph(bv: &Array2<f32>, w: &Array3<f32>, i: usize, p_h: &[f32], n_hidden: usize, bu_eff: &[f32], temperature: f32,
                   p: Option<&Array3<f32>>, q: Option<&Array2<f32>>) -> f32 {
    let logits = compute_logits(bv, w, i, p_h, n_hidden, bu_eff, temperature, p, q);
    let probs = softmax(&logits);
    let mut exp_rating = 0.0f32;
    for k in 0..K {
        exp_rating += (k as f32 + 1.0) * probs[k];
    }
    exp_rating
}

pub fn probs_with_ph(bv: &Array2<f32>, w: &Array3<f32>, i: usize, p_h: &[f32], n_hidden: usize, bu_eff: &[f32], temperature: f32,
                 p: Option<&Array3<f32>>, q: Option<&Array2<f32>>) -> Array1<f32> {
    let logits = compute_logits(bv, w, i, p_h, n_hidden, bu_eff, temperature, p, q);
    let probs = softmax(&logits);
    Array1::from_vec(probs.to_vec())
}

pub fn compute_logits(bv: &Array2<f32>, w: &Array3<f32>, i: usize, p_h: &[f32], n_hidden: usize, bu_eff: &[f32], temperature: f32,
                  p: Option<&Array3<f32>>, q: Option<&Array2<f32>>) -> [f32; K] {
    let mut logits = [0.0f32; K];
    if let (Some(p), Some(q)) = (p, q) {
        let n_factors = q.dim().0;
        let mut z = vec![0.0f32; n_factors];
        for f in 0..n_factors {
            let mut s = 0.0f32;
            for j in 0..n_hidden {
                s += q[[f, j]] * p_h[j];
            }
            z[f] = s;
        }
        for k in 0..K {
            let mut s = bv[[i, k]] + bu_eff[k];
            for f in 0..n_factors {
                s += p[[i, k, f]] * z[f];
            }
            logits[k] = s / temperature;
        }
    } else {
        for k in 0..K {
            let mut s = bv[[i, k]] + bu_eff[k];
            for j in 0..n_hidden {
                s += p_h[j] * w[[i, k, j]];
            }
            logits[k] = s / temperature;
        }
    }
    logits
}

pub fn softmax(logits: &[f32; K]) -> [f32; K] {
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter() { if v > max { max = v; } }
    let mut exps = [0.0f32; K];
    let mut sum = 0.0f32;
    for k in 0..K {
        let e = (logits[k] - max).exp();
        exps[k] = e;
        sum += e;
    }
    for k in 0..K {
        exps[k] /= sum;
    }
    exps
}
