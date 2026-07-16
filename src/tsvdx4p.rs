use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use crate::{calc_user_offsets, get_users, make_pb as base_pb, Dataset, MaskedDataset, Regressor};
use crate::tx::{TxConfig, TxModel, freq_bin, day_ranges};

fn make_pb(total: u64, prefix: &str) -> ProgressBar {
    let pb = base_pb(total);
    let _ = pb.set_style(
        ProgressStyle::with_template("{prefix} [{elapsed_precise}] {bar:40} {pos}/{len} ({eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb.set_prefix(prefix.to_string());
    pb
}

// ---------------------------------------------------------------------------
// By-item index for item epochs
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ByItemEntry {
    user: u32,
    rating_idx: u32,
}

struct ByItemIndex {
    starts: Vec<usize>,
    entries: Vec<ByItemEntry>,
}

impl ByItemIndex {
    fn build(tr: &Dataset) -> Self {
        let n_items = tr.n_items;
        let mut counts = vec![0u32; n_items];
        for t in 0..tr.n_ratings {
            counts[tr.item_idxs[t] as usize] += 1;
        }
        let mut starts = vec![0usize; n_items + 1];
        for i in 0..n_items {
            starts[i + 1] = starts[i] + counts[i] as usize;
        }
        let mut entries = vec![ByItemEntry { user: 0, rating_idx: 0 }; tr.n_ratings];
        let mut pos = starts[..n_items].to_vec();
        for t in 0..tr.n_ratings {
            let i = tr.item_idxs[t] as usize;
            entries[pos[i]] = ByItemEntry {
                user: tr.user_idxs[t] as u32,
                rating_idx: t as u32,
            };
            pos[i] += 1;
        }
        Self { starts, entries }
    }
}

// ---------------------------------------------------------------------------
// Thread-local NSVD1 delta buffer
// ---------------------------------------------------------------------------

struct NsvdDelta {
    d_yfeat: Array2<f32>,
    d_ybias: Array1<f32>,
    d_yfeat_day: Option<Array2<f32>>,
    /// Number of users that contributed to each item's gradient
    n_contrib: Array1<u32>,
}

impl NsvdDelta {
    fn new(n_items: usize, n_feat: usize, has_day: bool) -> Self {
        Self {
            d_yfeat: Array2::zeros((n_items, n_feat)),
            d_ybias: Array1::zeros(n_items),
            d_yfeat_day: if has_day { Some(Array2::zeros((n_items, n_feat))) } else { None },
            n_contrib: Array1::zeros(n_items),
        }
    }
}

// ---------------------------------------------------------------------------
// Unsafe Send pointer wrapper for disjoint parallel mutation
// ---------------------------------------------------------------------------

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    unsafe fn get(&self, offset: usize) -> f32 {
        unsafe { *self.0.add(offset) }
    }
    #[inline(always)]
    unsafe fn set(&self, offset: usize, val: f32) {
        unsafe { *self.0.add(offset) = val; }
    }
    #[inline(always)]
    unsafe fn sub(&self, offset: usize, delta: f32) {
        unsafe { *self.0.add(offset) -= delta; }
    }
}

// ---------------------------------------------------------------------------
// Free helper functions for use in closures
// ---------------------------------------------------------------------------

#[inline]
fn time_bin_static(day: i32, day_range: i32, n_time_bins: usize) -> usize {
    let num = (day as i64) * (n_time_bins as i64);
    let b = (num / day_range as i64) as usize;
    b.min(n_time_bins - 1)
}

#[inline]
fn dev_static(day: i32, tu_mean: f32, beta: f32) -> f32 {
    let dt = (day as f32) - tu_mean;
    if dt == 0.0 { 0.0 }
    else { dt.signum() * dt.abs().powf(beta) }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Tsvdx4pConfig {
    pub n_feat: usize,
    pub n_epochs: usize,
    pub seed: u64,
    pub shuffle_users: bool,

    pub n_time_bins: usize,
    pub beta: f32,
    pub n_freq_bins: usize,

    pub lr_u: f32,
    pub lr_ud: f32,
    pub lr_u2: f32,
    pub lr_ub: f32,
    pub lr_ubd: f32,
    pub lr_i: f32,
    pub lr_ib: f32,
    pub lr_y: f32,
    pub lr_yb: f32,
    pub lr_yd: f32,
    pub lr_tu: f32,
    pub lr_ti: f32,
    pub lr_ta: f32,

    pub lr_ibf: f32,
    pub lr_iqf: f32,
    pub reg_iqf: f32,
    pub sigma_iqf: f32,

    pub lr_cu: f32,
    pub reg_cu: f32,

    pub reg_u: f32,
    pub reg_u2: f32,
    pub reg_ud: f32,
    pub reg_i: f32,
    pub reg_y: f32,
    pub reg_yd: f32,

    pub sigma_u: f32,
    pub sigma_i: f32,
    pub sigma_y: f32,
    pub sigma_yd: f32,

    pub reset_u_epoch: usize,
    pub save_ifeat: bool,
    pub low_memory: bool,
    pub full_su: bool,

    /// Exponent for per-item NSVD1 gradient normalization.
    /// Accumulated gradient is divided by n_contributors^nsvd_norm_exp.
    /// 0.0 = no normalization (sum), 0.5 = sqrt, 1.0 = average.
    pub nsvd_norm_exp: f32,

    /// Number of threads for parallel epochs (0 = use rayon default).
    pub n_threads: usize,

    /// Sequential epoch indices (1-based). 0 = unused slot.
    pub seq_epochs: [usize; 5],
}

pub fn into_tx_config(cfg: Tsvdx4pConfig) -> TxConfig {
    TxConfig {
        n_feat:        cfg.n_feat,
        n_epochs:      cfg.n_epochs,
        seed:          cfg.seed,
        shuffle_users: cfg.shuffle_users,

        n_time_bins:   cfg.n_time_bins,
        beta:          cfg.beta,
        n_freq_bins:   cfg.n_freq_bins,

        lr_u:          cfg.lr_u,
        lr_ud:         cfg.lr_ud,
        lr_u2:         cfg.lr_u2,
        lr_ub:         cfg.lr_ub,
        lr_ubd:        cfg.lr_ubd,
        lr_i:          cfg.lr_i,
        lr_ib:         cfg.lr_ib,
        lr_y:          cfg.lr_y,
        lr_yb:         cfg.lr_yb,
        lr_yd:         cfg.lr_yd,
        lr_tu:         cfg.lr_tu,
        lr_ti:         cfg.lr_ti,
        lr_ta:         cfg.lr_ta,

        lr_ibf:        cfg.lr_ibf,
        lr_iqf:        cfg.lr_iqf,
        reg_iqf:       cfg.reg_iqf,
        sigma_iqf:     cfg.sigma_iqf,

        lr_cu:         cfg.lr_cu,
        reg_cu:        cfg.reg_cu,

        reg_u:         cfg.reg_u,
        reg_u2:        cfg.reg_u2,
        reg_ud:        cfg.reg_ud,
        reg_i:         cfg.reg_i,
        reg_y:         cfg.reg_y,
        reg_yd:        cfg.reg_yd,

        sigma_u:       cfg.sigma_u,
        sigma_i:       cfg.sigma_i,
        sigma_y:       cfg.sigma_y,
        sigma_yd:      cfg.sigma_yd,

        reset_u_epoch: cfg.reset_u_epoch,

        // Neighborhood disabled
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,

        sum_err_bug:   false,
        lambda1:       0.0,
        lambda2:       0.0,

        ordinal_head:  None,
        save_ifeat:    cfg.save_ifeat,
        low_memory:    cfg.low_memory,
        full_su:       cfg.full_su,
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Tsvdx4pModel {
    cfg: Tsvdx4pConfig,
    inner: TxModel,
    by_item: ByItemIndex,
}

impl Tsvdx4pModel {
    fn is_seq_epoch(&self, epoch: usize) -> bool {
        self.cfg.seq_epochs.contains(&epoch)
    }

    fn rebuild_ycache(&mut self, tr: &Dataset, pr: &MaskedDataset) {
        self.inner.ycache_mut().fill(0.0);
        self.inner.ycache_bias_mut().fill(0.0);
        for t in 0..tr.n_ratings {
            let u = tr.user_idxs[t] as usize;
            let i = tr.item_idxs[t] as usize;
            let yfeat_row = self.inner.yfeat().row(i).to_owned();
            let ybias_i = self.inner.ybias()[i];
            let mut su = self.inner.ycache_mut().row_mut(u);
            su += &yfeat_row;
            drop(su);
            self.inner.ycache_bias_mut()[u] += ybias_i;
        }
        for t in 0..pr.n_ratings {
            let u = pr.user_idxs[t] as usize;
            let i = pr.item_idxs[t] as usize;
            let yfeat_row = self.inner.yfeat().row(i).to_owned();
            let ybias_i = self.inner.ybias()[i];
            let mut su = self.inner.ycache_mut().row_mut(u);
            su += &yfeat_row;
            drop(su);
            self.inner.ycache_bias_mut()[u] += ybias_i;
        }
        for u in 0..tr.n_users {
            let cnt = tr.user_cnts[u] + pr.user_cnts[u];
            if cnt > 0 {
                let norm = (cnt as f32).sqrt();
                let mut su = self.inner.ycache_mut().row_mut(u);
                su /= norm;
                drop(su);
                self.inner.ycache_bias_mut()[u] /= norm;
            }
        }
    }

    fn rebuild_ycache_day(&mut self, tr: &Dataset, pr: &MaskedDataset) {
        self.inner.ycache_day_mut().fill(0.0);
        let n_ud = self.inner.ud().n_total();
        let mut cnts = vec![0.0f32; n_ud];
        for idx in 0..pr.n_ratings {
            let u = pr.user_idxs[idx] as usize;
            let i = pr.item_idxs[idx] as usize;
            let day = pr.dates[idx];
            if let Some(ud_idx) = self.inner.ud().index(u, day) {
                cnts[ud_idx] += 1.0;
                let yfeat_day_row = self.inner.yfeat_day().row(i).to_owned();
                let mut row = self.inner.ycache_day_mut().row_mut(ud_idx);
                row += &yfeat_day_row;
            }
        }
        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;
            let day = tr.dates[idx];
            if let Some(ud_idx) = self.inner.ud().index(u, day) {
                cnts[ud_idx] += 1.0;
                let yfeat_day_row = self.inner.yfeat_day().row(i).to_owned();
                let mut row = self.inner.ycache_day_mut().row_mut(ud_idx);
                row += &yfeat_day_row;
            }
        }
        for ud_idx in 0..n_ud {
            if cnts[ud_idx] > 0.0 {
                let norm = cnts[ud_idx].sqrt();
                let mut row = self.inner.ycache_day_mut().row_mut(ud_idx);
                row /= norm;
            }
        }
    }

    // ------------------------------------------------------------------
    // Apply accumulated NSVD1 deltas with regularization
    // ------------------------------------------------------------------
    fn apply_nsvd_deltas(&mut self, deltas: &[NsvdDelta]) {
        let cfg = self.cfg;
        let n_items = self.inner.ifeat().dim().0;
        let n_feat = cfg.n_feat;
        let exp = cfg.nsvd_norm_exp;

        // Sum yfeat deltas and contributor counts across threads
        let mut total_dy = Array2::<f32>::zeros((n_items, n_feat));
        let mut total_dyb = Array1::<f32>::zeros(n_items);
        let mut total_nc = Array1::<u32>::zeros(n_items);
        for d in deltas {
            total_dy += &d.d_yfeat;
            total_dyb += &d.d_ybias;
            total_nc += &d.n_contrib;
        }
        for j in 0..n_items {
            let nc = total_nc[j];
            if nc == 0 { continue; }
            let scale = if exp == 0.0 { 1.0 } else { (nc as f32).powf(exp) };
            for k in 0..n_feat {
                let yj = self.inner.yfeat()[[j, k]];
                self.inner.yfeat_mut()[[j, k]] -= cfg.lr_y * (total_dy[[j, k]] / scale + cfg.reg_y * yj);
            }
            self.inner.ybias_mut()[j] -= cfg.lr_yb * (total_dyb[j] / scale);
        }

        // Sum yfeat_day deltas if applicable
        if !cfg.low_memory {
            let mut total_dyd = Array2::<f32>::zeros((n_items, n_feat));
            for d in deltas {
                if let Some(ref dyd) = d.d_yfeat_day {
                    total_dyd += dyd;
                }
            }
            for j in 0..n_items {
                let nc = total_nc[j];
                if nc == 0 { continue; }
                let scale = if exp == 0.0 { 1.0 } else { (nc as f32).powf(exp) };
                for k in 0..n_feat {
                    let yj = self.inner.yfeat_day()[[j, k]];
                    self.inner.yfeat_day_mut()[[j, k]] -= cfg.lr_yd * (total_dyd[[j, k]] / scale + cfg.reg_yd * yj);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Parallel user epoch
    // ------------------------------------------------------------------
    fn fit_epoch_user(&mut self, tr: &Dataset, pr: &MaskedDataset, epoch: usize) {
        let cfg = self.cfg;
        let n_feat = cfg.n_feat;
        let n_time_bins = cfg.n_time_bins;
        let n_freq_bins = cfg.n_freq_bins;
        let low_mem = cfg.low_memory;
        let use_probe = cfg.full_su;
        let n_items = tr.n_items;

        if epoch == cfg.reset_u_epoch {
            self.inner.ufeat_mut().fill(0.0);
            self.inner.ufeat2_mut().fill(0.0);
        }

        let user_offsets = calc_user_offsets(tr);
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        // Raw pointers for disjoint user-param mutation
        // Each line's mutable borrow of self.inner ends at the `;` since raw ptrs don't carry lifetimes.
        let ubias_ptr = SendPtr(self.inner.ubias_mut().as_slice_mut().unwrap().as_mut_ptr());
        let ufeat_ptr = SendPtr(self.inner.ufeat_mut().as_slice_mut().unwrap().as_mut_ptr());
        let ufeat2_ptr = SendPtr(self.inner.ufeat2_mut().as_slice_mut().unwrap().as_mut_ptr());
        let but_bin_ptr = SendPtr(self.inner.but_bin_mut().as_slice_mut().unwrap().as_mut_ptr());
        let alpha_u_ptr = SendPtr(self.inner.alpha_u_mut().as_slice_mut().unwrap().as_mut_ptr());
        let cu_ptr = SendPtr(self.inner.cu_mut().as_slice_mut().unwrap().as_mut_ptr());
        let ycache_ptr = SendPtr(self.inner.ycache_mut().as_slice_mut().unwrap().as_mut_ptr());
        let ycache_bias_ptr = SendPtr(self.inner.ycache_bias_mut().as_slice_mut().unwrap().as_mut_ptr());
        let ubias_day_ptr = SendPtr(self.inner.ubias_day_mut().as_mut_ptr());
        let ufeat_day_ptr = if !low_mem && cfg.lr_ud != 0.0 {
            SendPtr(self.inner.ufeat_day_mut().as_slice_mut().unwrap().as_mut_ptr())
        } else {
            SendPtr(std::ptr::null_mut())
        };

        // Shared refs for read-only params
        let ifeat_s = self.inner.ifeat().as_slice().unwrap();
        let ifeat_freq_s = self.inner.ifeat_freq().as_slice().unwrap();
        let ibias_s = self.inner.ibias().as_slice().unwrap();
        let bit_bin_s = self.inner.bit_bin().as_slice().unwrap();
        let ibias_freq_s = self.inner.ibias_freq().as_slice().unwrap();
        let yfeat_s = self.inner.yfeat().as_slice().unwrap();
        let ybias_s = self.inner.ybias().as_slice().unwrap();
        let yfeat_day_s = if !low_mem { self.inner.yfeat_day().as_slice().unwrap() } else { &[] };
        let gbias = *self.inner.gbias();
        let day_range = *self.inner.day_range();
        let beta = cfg.beta;
        let tu_mean_s = self.inner.tu_mean().as_slice().unwrap();
        let ud = self.inner.ud();
        let cut_s = self.inner.cut();
        let probe_items_by_user = self.inner.probe_items_by_user();
        let probe_items_by_ud = self.inner.probe_items_by_ud();
        let user_offsets_s = user_offsets.as_slice().unwrap();

        let users_slice = users.as_slice().unwrap();
        let n_threads = if cfg.n_threads > 0 { cfg.n_threads } else { rayon::current_num_threads() }.max(1);
        let chunk_size = (users_slice.len() + n_threads - 1) / n_threads;

        let pb = make_pb(tr.n_users as u64, "U");
        let counter = AtomicUsize::new(0);

        let deltas: Vec<NsvdDelta> = users_slice
            .par_chunks(chunk_size.max(1))
            .map(|user_chunk| {
                let mut delta = NsvdDelta::new(n_items, n_feat, !low_mem);

                for &u in user_chunk {
                    let start = user_offsets_s[u];
                    let end = user_offsets_s[u + 1];
                    let cnt = end - start;
                    if cnt == 0 { continue; }

                    let day_rngs = if low_mem { Vec::new() } else { day_ranges(&tr.dates, start, end) };

                    // Compute su from frozen yfeat
                    let mut su = vec![0.0f32; n_feat];
                    let mut su_bias = 0.0f32;
                    for t in start..end {
                        let j = tr.item_idxs[t] as usize;
                        for k in 0..n_feat { su[k] += yfeat_s[j * n_feat + k]; }
                        su_bias += ybias_s[j];
                    }
                    let total_cnt = if use_probe {
                        let pi = &probe_items_by_user[u];
                        for &j in pi {
                            let j = j as usize;
                            for k in 0..n_feat { su[k] += yfeat_s[j * n_feat + k]; }
                            su_bias += ybias_s[j];
                        }
                        cnt + pi.len()
                    } else { cnt };
                    let norm = (total_cnt as f32).sqrt();
                    for k in 0..n_feat { su[k] /= norm; }
                    su_bias /= norm;

                    // Store ycache (disjoint by user)
                    for k in 0..n_feat {
                        unsafe { ycache_ptr.set(u * n_feat + k, su[k]); }
                    }
                    unsafe { ycache_bias_ptr.set(u, su_bias); }

                    let mut sum_err_q = vec![0.0f32; n_feat];
                    let mut sum_err_q_day = vec![0.0f32; n_feat];
                    let mut sum_err = 0.0f32;

                    let mut su_day = if low_mem { Vec::new() } else { vec![0.0f32; n_feat] };
                    let mut norm_day = 0.0f32;
                    let mut dr_idx = 0usize;
                    let mut cur_freq_bin = 0usize;
                    let has_ud = !low_mem && cfg.lr_ud != 0.0;

                    for t in start..end {
                        let i = tr.item_idxs[t] as usize;
                        let r = tr.residuals[t];
                        let day = tr.dates[t] as i32;
                        let day16 = tr.dates[t];
                        let b = time_bin_static(day, day_range, n_time_bins);
                        let dev = dev_static(day, tu_mean_s[u], beta);

                        let (day_start, day_stop) = if low_mem { (0, 0) } else { day_rngs[dr_idx] };

                        // Start of new day
                        if !low_mem && t == day_start {
                            for k in 0..n_feat { sum_err_q_day[k] = 0.0; su_day[k] = 0.0; }
                            let ud_idx_day = ud.index(u, day16);
                            cur_freq_bin = ud_idx_day.map_or(0, |idx| freq_bin(ud.day_cnts[idx], n_freq_bins));
                            for t_day in day_start..=day_stop {
                                let j = tr.item_idxs[t_day] as usize;
                                for k in 0..n_feat { su_day[k] += yfeat_day_s[j * n_feat + k]; }
                            }
                            let mut day_cnt = (day_stop - day_start + 1) as f32;
                            if use_probe {
                                if let Some(ud_idx) = ud_idx_day {
                                    for &j in &probe_items_by_ud[ud_idx] {
                                        let j = j as usize;
                                        for k in 0..n_feat { su_day[k] += yfeat_day_s[j * n_feat + k]; }
                                        day_cnt += 1.0;
                                    }
                                }
                            }
                            norm_day = day_cnt.sqrt();
                            for k in 0..n_feat { su_day[k] /= norm_day; }
                        }

                        let f = cur_freq_bin;
                        let ud_idx = ud.index(u, day16);

                        // Read user params
                        let ubias_u = unsafe { ubias_ptr.get(u) };
                        let but_bin_ub = unsafe { but_bin_ptr.get(u * n_time_bins + b) };
                        let alpha_u_u = unsafe { alpha_u_ptr.get(u) };
                        let bu_day = ud_idx.map_or(0.0, |idx| unsafe { ubias_day_ptr.get(idx) });
                        let cu_u = unsafe { cu_ptr.get(u) };
                        let cu_t = cu_u + ud_idx.map_or(0.0, |idx| cut_s[idx]);

                        let bu_t = ubias_u + but_bin_ub + alpha_u_u * dev + bu_day + su_bias;
                        let bi_t = ibias_s[i] + bit_bin_s[i * n_time_bins + b] + ibias_freq_s[i * n_freq_bins + f];

                        // Dot product
                        let mut dot = 0.0f32;
                        for k in 0..n_feat {
                            let qi_eff = ifeat_s[i * n_feat + k] + ifeat_freq_s[(i * n_freq_bins + f) * n_feat + k];
                            let pu_k = unsafe { ufeat_ptr.get(u * n_feat + k) };
                            let pu2_k = unsafe { ufeat2_ptr.get(u * n_feat + k) };
                            let mut pu_eff = pu_k + su[k] + dev * pu2_k;
                            if !low_mem {
                                if has_ud {
                                    if let Some(idx) = ud_idx {
                                        pu_eff += unsafe { ufeat_day_ptr.get(idx * n_feat + k) };
                                    }
                                }
                                pu_eff += su_day[k];
                            }
                            dot += pu_eff * qi_eff;
                        }
                        let score = gbias + bu_t + bi_t * cu_t + dot;
                        let err = score - r;

                        // User bias updates (disjoint by user)
                        unsafe {
                            ubias_ptr.sub(u, cfg.lr_ub * err);
                            but_bin_ptr.sub(u * n_time_bins + b, cfg.lr_tu * err);
                            alpha_u_ptr.sub(u, cfg.lr_ta * err * dev);
                            if let Some(idx) = ud_idx {
                                ubias_day_ptr.sub(idx, cfg.lr_ubd * err);
                            }
                            cu_ptr.sub(u, cfg.lr_cu * (err * bi_t + cfg.reg_cu * (cu_u - 1.0)));
                        }

                        // User factor updates + NSVD1 error accumulation
                        sum_err += err;
                        for k in 0..n_feat {
                            let qi_eff = ifeat_s[i * n_feat + k] + ifeat_freq_s[(i * n_freq_bins + f) * n_feat + k];
                            let pu_k = unsafe { ufeat_ptr.get(u * n_feat + k) };
                            let pu2_k = unsafe { ufeat2_ptr.get(u * n_feat + k) };

                            sum_err_q[k] += err * qi_eff;
                            if !low_mem { sum_err_q_day[k] += err * qi_eff; }

                            unsafe {
                                ufeat_ptr.sub(u * n_feat + k, cfg.lr_u * (err * qi_eff + cfg.reg_u * pu_k));
                                ufeat2_ptr.sub(u * n_feat + k, cfg.lr_u2 * (err * qi_eff * dev + cfg.reg_u2 * pu2_k));
                            }

                            if has_ud {
                                if let Some(idx) = ud_idx {
                                    let pud = unsafe { ufeat_day_ptr.get(idx * n_feat + k) };
                                    unsafe {
                                        ufeat_day_ptr.sub(idx * n_feat + k, cfg.lr_ud * (err * qi_eff + cfg.reg_ud * pud));
                                    }
                                }
                            }
                        }

                        // End of day: accumulate yfeat_day delta
                        if !low_mem && t == day_stop {
                            for t_day in day_start..=day_stop {
                                let j = tr.item_idxs[t_day] as usize;
                                for k in 0..n_feat {
                                    delta.d_yfeat_day.as_mut().unwrap()[[j, k]] += sum_err_q_day[k] / norm_day;
                                }
                            }
                            dr_idx += 1;
                        }
                    }

                    // Accumulate yfeat/ybias deltas for this user
                    for t in start..end {
                        let j = tr.item_idxs[t] as usize;
                        for k in 0..n_feat {
                            delta.d_yfeat[[j, k]] += sum_err_q[k] / norm;
                        }
                        delta.d_ybias[j] += sum_err / norm;
                        delta.n_contrib[j] += 1;
                    }

                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c % 4096 == 0 { pb.set_position(c as u64); }
                }
                delta
            })
            .collect();
        pb.finish_and_clear();

        // Apply NSVD1 deltas with regularization
        self.apply_nsvd_deltas(&deltas);

        // Rebuild caches (yfeat changed)
        self.rebuild_ycache(tr, pr);
        if !low_mem { self.rebuild_ycache_day(tr, pr); }
    }

    // ------------------------------------------------------------------
    // Parallel item epoch
    // ------------------------------------------------------------------
    fn fit_epoch_item(&mut self, tr: &Dataset, _pr: &MaskedDataset, _epoch: usize) {
        let cfg = self.cfg;
        let n_feat = cfg.n_feat;
        let n_time_bins = cfg.n_time_bins;
        let n_freq_bins = cfg.n_freq_bins;
        let low_mem = cfg.low_memory;
        let n_items = tr.n_items;

        // Raw pointers for disjoint item-param mutation (borrows released after `;`)
        let ibias_ptr = SendPtr(self.inner.ibias_mut().as_slice_mut().unwrap().as_mut_ptr());
        let bit_bin_ptr = SendPtr(self.inner.bit_bin_mut().as_slice_mut().unwrap().as_mut_ptr());
        let ibias_freq_ptr = SendPtr(self.inner.ibias_freq_mut().as_slice_mut().unwrap().as_mut_ptr());
        let ifeat_ptr = SendPtr(self.inner.ifeat_mut().as_slice_mut().unwrap().as_mut_ptr());
        let ifeat_freq_ptr = SendPtr(self.inner.ifeat_freq_mut().as_slice_mut().unwrap().as_mut_ptr());

        // Shared refs for read-only user params and caches
        let ubias_s = self.inner.ubias().as_slice().unwrap();
        let ufeat_s = self.inner.ufeat().as_slice().unwrap();
        let ufeat2_s = self.inner.ufeat2().as_slice().unwrap();
        let but_bin_s = self.inner.but_bin().as_slice().unwrap();
        let alpha_u_s = self.inner.alpha_u().as_slice().unwrap();
        let cu_s = self.inner.cu().as_slice().unwrap();
        let cut_s = self.inner.cut();
        let ycache_s = self.inner.ycache().as_slice().unwrap();
        let ycache_bias_s = self.inner.ycache_bias().as_slice().unwrap();
        let ubias_day_s = self.inner.ubias_day();
        let ufeat_day_s = if !low_mem { self.inner.ufeat_day().as_slice().unwrap() } else { &[] };
        let ycache_day_s = if !low_mem { self.inner.ycache_day().as_slice().unwrap() } else { &[] };

        let gbias = *self.inner.gbias();
        let day_range = *self.inner.day_range();
        let beta = cfg.beta;
        let tu_mean_s = self.inner.tu_mean().as_slice().unwrap();
        let ud = self.inner.ud();

        let by_item_starts = &self.by_item.starts;
        let by_item_entries = &self.by_item.entries;

        let n_threads = if cfg.n_threads > 0 { cfg.n_threads } else { rayon::current_num_threads() }.max(1);
        let chunk_size = (n_items + n_threads - 1) / n_threads;

        let pb = make_pb(n_items as u64, "I");
        let counter = AtomicUsize::new(0);

        let items: Vec<usize> = (0..n_items).collect();
        items.par_chunks(chunk_size.max(1)).for_each(|item_chunk| {
            // Thread-local scratch buffer reused across all ratings in this chunk;
            // allocated once per thread, no per-rating allocation.
            let mut pu_effs = vec![0.0f32; n_feat];
            for &i in item_chunk {
                let entry_start = by_item_starts[i];
                let entry_end = by_item_starts[i + 1];

                for e_idx in entry_start..entry_end {
                    let entry = &by_item_entries[e_idx];
                    let u = entry.user as usize;
                    let t = entry.rating_idx as usize;
                    let r = tr.residuals[t];
                    let day = tr.dates[t] as i32;
                    let day16 = tr.dates[t];
                    let b = time_bin_static(day, day_range, n_time_bins);
                    let dev = dev_static(day, tu_mean_s[u], beta);

                    let ud_idx = ud.index(u, day16);
                    let f = ud_idx.map_or(0, |idx| freq_bin(ud.day_cnts[idx], n_freq_bins));

                    let bu_day = ud_idx.map_or(0.0, |idx| ubias_day_s[idx]);
                    let bu_t = ubias_s[u] + but_bin_s[u * n_time_bins + b] + alpha_u_s[u] * dev
                        + bu_day + ycache_bias_s[u];

                    // Read item params via pointers
                    let ibias_i = unsafe { ibias_ptr.get(i) };
                    let bit_bin_ib = unsafe { bit_bin_ptr.get(i * n_time_bins + b) };
                    let ibias_freq_if = unsafe { ibias_freq_ptr.get(i * n_freq_bins + f) };
                    let bi_t = ibias_i + bit_bin_ib + ibias_freq_if;

                    let cu_t = cu_s[u] + ud_idx.map_or(0.0, |idx| cut_s[idx]);

                    // Dot product
                    let mut dot = 0.0f32;
                    for k in 0..n_feat {
                        let qi_k = unsafe { ifeat_ptr.get(i * n_feat + k) };
                        let qf_k = unsafe { ifeat_freq_ptr.get((i * n_freq_bins + f) * n_feat + k) };
                        let qi_eff = qi_k + qf_k;
                        let mut pu_eff = ufeat_s[u * n_feat + k] + ycache_s[u * n_feat + k]
                            + dev * ufeat2_s[u * n_feat + k];
                        if !low_mem {
                            if let Some(idx) = ud_idx {
                                pu_eff += ufeat_day_s[idx * n_feat + k];
                                pu_eff += ycache_day_s[idx * n_feat + k];
                            }
                        }
                        pu_effs[k] = pu_eff;
                        dot += pu_eff * qi_eff;
                    }
                    let score = gbias + bu_t + bi_t * cu_t + dot;
                    let err = score - r;

                    // Item param updates (disjoint by item)
                    unsafe {
                        ibias_ptr.sub(i, cfg.lr_ib * (err * cu_t));
                        bit_bin_ptr.sub(i * n_time_bins + b, cfg.lr_ti * (err * cu_t));
                        ibias_freq_ptr.sub(i * n_freq_bins + f, cfg.lr_ibf * (err * cu_t));
                    }
                    for k in 0..n_feat {
                        let qi_k = unsafe { ifeat_ptr.get(i * n_feat + k) };
                        let qf_k = unsafe { ifeat_freq_ptr.get((i * n_freq_bins + f) * n_feat + k) };
                        unsafe {
                            ifeat_ptr.sub(i * n_feat + k, cfg.lr_i * (err * pu_effs[k] + cfg.reg_i * qi_k));
                            ifeat_freq_ptr.sub((i * n_freq_bins + f) * n_feat + k, cfg.lr_iqf * (err * pu_effs[k] + cfg.reg_iqf * qf_k));
                        }
                    }
                }
                let c = counter.fetch_add(1, Ordering::Relaxed);
                if c % 256 == 0 { pb.set_position(c as u64); }
            }
        });
        pb.finish_and_clear();
    }
}

// ---------------------------------------------------------------------------
// Regressor trait implementation
// ---------------------------------------------------------------------------

impl Regressor for Tsvdx4pModel {
    type Config = Tsvdx4pConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        Self {
            inner: TxModel::new(tr, pr, into_tx_config(cfg)),
            by_item: ByItemIndex::build(tr),
            cfg,
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn n_subscores(&self) -> usize { self.inner.n_subscores() }

    fn subscore_names(&self) -> Vec<String> { self.inner.subscore_names() }

    fn predict_subscores(&self, u: usize, i: usize, day: i32) -> Array1<f32> {
        self.inner.predict_subscores(u, i, day)
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        self.inner.predict(u, i, day)
    }

    fn save_artifacts(&self, model_name: &str, tr_set: &str, preds_dir: &str) {
        self.inner.save_artifacts(model_name, tr_set, preds_dir)
    }

    fn fit_epoch(&mut self, tr: &Dataset, pr: &MaskedDataset, epoch: usize) {
        if self.is_seq_epoch(epoch) {
            self.inner.fit_epoch(tr, pr, epoch);
        } else {
            self.fit_epoch_user(tr, pr, epoch);
            self.fit_epoch_item(tr, pr, epoch);
        }
    }
}
