// An integrated model with an attentive neighborhood term.
//
// A rating is a BellKor-style temporal baseline, an optional wide bilinear term,
// and a correction assembled from the *user's own other ratings*:
//
//     s(u,i,t) = Σ_j α_ij · c_j · (r_uj − base_uj)
//     α_ij     = softmax_j( q_i·q_j / √d − β·ln(1 + |t − t_j|) )
//
// so the neighbors compete for a fixed budget of attention, and how much each
// one counts depends both on how similar the model thinks the two films are and
// on how far apart in time the two ratings were. The ensemble already has plain
// item-item kNN and BellKor's learned neighborhood weights; what it does not have
// is a *normalised* similarity that is renormalised per target.
//
// Every parameter is per-user or per-item, so there is no dense weight matrix to
// share between threads: blocks of users run Hogwild, and `n_threads = 1` makes
// the whole fit deterministic. The correction is squashed to `a·tanh(s/a)` and
// the item embeddings live in a max-norm ball, which is what keeps the softmax
// logits, and with them the training, from running away.

use crate::cfnade::fmath::{exp, ln, tanh};
use crate::tx::SparseUD;
use crate::{
    Dataset, MaskedDataset, Regressor, bias_time_bin, calc_user_offsets, get_users, make_pb,
    user_time_dev,
};
use ndarray::Array1;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Project a parameter vector back into the max-norm ball.
#[inline]
fn cap_rms(v: &mut [f32], cap: f32) {
    let mut s = 0.0f32;
    for &x in v.iter() { s += x * x; }
    let rms = (s / v.len() as f32).sqrt();
    if rms > cap {
        let f = cap / rms;
        for x in v.iter_mut() { *x *= f; }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct AttnConfig {
    // Architecture
    pub n_feat: usize,        // similarity embedding width
    pub n_pool: usize,        // neighbors kept per user
    pub n_mf: usize,          // width of the plain bilinear term (0 = off)
    pub n_bins: usize,        // item-bias time bins
    pub out_scale: f32,       // half-range of the bounded correction
    pub beta: f32,            // time-distance penalty in the logits. Fixed, not
                              // learned: one global scalar fed by every rating
                              // diverges, and an infinite logit makes the
                              // softmax's `x - max` produce NaN.
    pub dev_beta: f32,        // exponent in BellKor's |dt|^beta user drift term.
                              // Unrelated to `beta` above.

    // Schedule
    pub n_epochs: usize,
    pub seed: u64,
    pub lr_decay: f32,

    // Learning rates
    pub lr_q: f32,            // similarity embeddings
    pub lr_c: f32,            // per-item neighbor weights
    pub lr_mf: f32,
    pub lr_bias: f32,
    pub lr_alpha: f32,

    // Regularisation
    pub reg_q: f32,
    pub reg_c: f32,
    pub reg_mf: f32,
    pub reg_bias: f32,
    pub reg_bu_day: f32,
    pub reg_alpha: f32,
    pub sigma: f32,           // embedding init std dev

    // Numerical guards
    pub huber: f32,           // gradient-side error clamp: below this the loss is
                              // squared, above it linear. 0 = plain squared loss.
                              // Changes *which* ratings the fit chases, which is
                              // what decorrelates a model from its neighbors.
    pub grad_clip: f32,
    pub emb_cap: f32,

    // Execution
    pub n_threads: usize,     // 1 = sequential, and then the fit is deterministic
    pub block_users: usize,
    pub train_frac: f32,      // below 1 only for quick tuning probes
}

impl Default for AttnConfig {
    fn default() -> Self {
        Self {
            n_feat: 32,
            n_pool: 32,
            n_mf: 0,
            n_bins: 30,
            out_scale: 2.5,
            beta: 0.3,
            dev_beta: 0.4,

            n_epochs: 10,
            seed: 42,
            lr_decay: 0.9,

            lr_q: 5e-3,
            lr_c: 5e-3,
            lr_mf: 8e-3,
            lr_bias: 5e-3,
            lr_alpha: 2e-6,

            reg_q: 0.02,
            reg_c: 0.02,
            reg_mf: 0.015,
            reg_bias: 0.03,
            reg_bu_day: 0.03,
            reg_alpha: 50.0,
            sigma: 0.05,

            huber: 0.0,
            grad_clip: 1.0,
            emb_cap: 1.0,

            n_threads: 0,
            block_users: 32,
            train_frac: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared-pointer helper
// ---------------------------------------------------------------------------

/// Raw pointer into a parameter array, shared across worker threads. Every
/// parameter here is per-user or per-item, so collisions are rare (Hogwild).
#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

impl Ptr {
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn sl(&self, off: usize, len: usize) -> &'static mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut ia = a.chunks_exact(8);
    let mut ib = b.chunks_exact(8);
    for (ca, cb) in ia.by_ref().zip(ib.by_ref()) {
        for l in 0..8 { acc[l] += ca[l] * cb[l]; }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ia.remainder().iter().zip(ib.remainder()) { s += x * y; }
    s
}

#[inline]
fn axpy(out: &mut [f32], src: &[f32], scale: f32) {
    for (o, &v) in out.iter_mut().zip(src) { *o += scale * v; }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct AttnModel {
    cfg: AttnConfig,
    mu: f32,

    // Temporal baseline
    bu: Vec<f32>,
    bi: Vec<f32>,
    alpha_u: Vec<f32>,
    bu_day: Vec<f32>,
    bit_bin: Vec<f32>,
    u_day_mean: Vec<f32>,
    ud: SparseUD,

    // Attention
    qi: Vec<f32>,   // [n_items × n_feat] similarity embedding
    ci: Vec<f32>,   // [n_items] how much this neighbor's residual counts

    // Optional wide bilinear term
    pmf: Vec<f32>,
    qmf: Vec<f32>,

    // Neighbor pool: up to n_pool of each user's training ratings, strided
    // evenly across their history so the sample spans their whole timeline.
    pool_off: Vec<u32>,
    pool_item: Vec<i32>,
    pool_day: Vec<i16>,
    pool_rtg: Vec<f32>,

    uoff: Vec<usize>,
}

impl AttnModel {
    #[inline]
    fn baseline(&self, u: usize, i: usize, day: i32) -> f32 {
        let bin = bias_time_bin(day, self.cfg.n_bins);
        let bud = self.ud.index(u, day as i16).map_or(0.0, |ix| self.bu_day[ix]);
        self.mu
            + self.bu[u]
            + self.alpha_u[u] * user_time_dev(day, self.u_day_mean[u], self.cfg.dev_beta)
            + bud
            + self.bi[i]
            + self.bit_bin[i * self.cfg.n_bins + bin]
    }

    /// Residual each pooled neighbor carries: what the baseline failed to
    /// explain about that rating.
    fn pool_residuals(&self, u: usize, out: &mut Vec<f32>) {
        let (lo, hi) = (self.pool_off[u] as usize, self.pool_off[u + 1] as usize);
        out.clear();
        for p in lo..hi {
            let (j, t) = (self.pool_item[p] as usize, self.pool_day[p] as i32);
            out.push(self.pool_rtg[p] - self.baseline(u, j, t));
        }
    }
}

/// Attention over one user's pool for a single target. Fills `w` with the
/// softmax weights and returns the assembled correction *before* squashing.
#[allow(clippy::too_many_arguments)]
#[inline]
fn attend(
    qi: &[f32], ci: &[f32], beta: f32, d: usize,
    target: usize, target_day: i32,
    pool_item: &[i32], pool_day: &[i16], pool_res: &[f32],
    w: &mut Vec<f32>,
) -> f32 {
    let inv = 1.0 / (d as f32).sqrt();
    let qt = &qi[target * d..target * d + d];
    w.clear();
    let mut max = f32::NEG_INFINITY;
    for (p, &j) in pool_item.iter().enumerate() {
        let j = j as usize;
        if j == target {
            w.push(f32::NEG_INFINITY);
            continue;
        }
        let dt = (target_day - pool_day[p] as i32).abs() as f32;
        let logit = dot(qt, &qi[j * d..j * d + d]) * inv - beta * ln(1.0 + dt);
        if logit > max { max = logit; }
        w.push(logit);
    }
    if max == f32::NEG_INFINITY { return 0.0; }
    let mut sum = 0.0;
    for x in w.iter_mut() {
        *x = if *x == f32::NEG_INFINITY { 0.0 } else { exp(*x - max) };
        sum += *x;
    }
    if sum <= 0.0 { return 0.0; }
    let inv_sum = 1.0 / sum;
    let mut s = 0.0;
    for (p, x) in w.iter_mut().enumerate() {
        *x *= inv_sum;
        s += *x * ci[pool_item[p] as usize] * pool_res[p];
    }
    s
}

// ---------------------------------------------------------------------------
// Regressor impl
// ---------------------------------------------------------------------------

impl Regressor for AttnModel {
    type Config = AttnConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let (n_users, n_items, d) = (tr.n_users, tr.n_items, cfg.n_feat);

        let mu = (tr.residuals.iter().map(|&x| x as f64).sum::<f64>() / tr.n_ratings as f64) as f32;

        let mut i_sum = vec![0.0f64; n_items];
        let mut i_cnt = vec![0u32; n_items];
        let mut u_cnt = vec![0u32; n_users];
        let mut u_day_sum = vec![0.0f64; n_users];
        for t in 0..tr.n_ratings {
            let (u, i) = (tr.user_idxs[t] as usize, tr.item_idxs[t] as usize);
            i_sum[i] += tr.residuals[t] as f64;
            i_cnt[i] += 1;
            u_cnt[u] += 1;
            u_day_sum[u] += tr.dates[t] as f64;
        }
        let u_day_mean: Vec<f32> = (0..n_users)
            .map(|u| if u_cnt[u] > 0 { (u_day_sum[u] / u_cnt[u] as f64) as f32 } else { 0.0 })
            .collect();

        let bi: Vec<f32> = (0..n_items)
            .map(|i| ((i_sum[i] - mu as f64 * i_cnt[i] as f64) / (i_cnt[i] as f64 + 25.0)) as f32)
            .collect();
        let mut bu_num = vec![0.0f64; n_users];
        for t in 0..tr.n_ratings {
            let (u, i) = (tr.user_idxs[t] as usize, tr.item_idxs[t] as usize);
            bu_num[u] += tr.residuals[t] as f64 - mu as f64 - bi[i] as f64;
        }
        let bu: Vec<f32> = (0..n_users)
            .map(|u| (bu_num[u] / (u_cnt[u] as f64 + 10.0)) as f32)
            .collect();

        // Neighbor pool, strided evenly over each user's ratings so a heavy
        // rater's sample still spans their whole history rather than its start.
        let uoff: Vec<usize> = calc_user_offsets(tr).to_vec();
        let mut pool_off = vec![0u32; n_users + 1];
        let mut pool_item = Vec::new();
        let mut pool_day = Vec::new();
        let mut pool_rtg = Vec::new();
        for u in 0..n_users {
            let (lo, hi) = (uoff[u], uoff[u + 1]);
            let n = hi - lo;
            let take = n.min(cfg.n_pool);
            if take > 0 {
                for s in 0..take {
                    let t = lo + s * n / take;
                    pool_item.push(tr.item_idxs[t]);
                    pool_day.push(tr.dates[t]);
                    pool_rtg.push(tr.residuals[t]);
                }
            }
            pool_off[u + 1] = pool_off[u] + take as u32;
        }

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let gauss = |n: usize, s: f32, rng: &mut StdRng| -> Vec<f32> {
            let dist = Normal::<f32>::new(0.0, s).unwrap();
            (0..n).map(|_| dist.sample(rng)).collect()
        };
        let pmf = gauss(n_users * cfg.n_mf, cfg.sigma, &mut rng);
        let qmf = gauss(n_items * cfg.n_mf, cfg.sigma, &mut rng);
        let qi = gauss(n_items * d, cfg.sigma, &mut rng);

        Self {
            cfg, mu, bu, bi,
            alpha_u: vec![0.0; n_users],
            bu_day: vec![0.0; SparseUD::new(tr, pr).n_total()],
            bit_bin: vec![0.0; n_items * cfg.n_bins],
            u_day_mean,
            ud: SparseUD::new(tr, pr),
            qi,
            ci: vec![0.0; n_items],
            pmf, qmf,
            pool_off, pool_item, pool_day, pool_rtg,
            uoff,
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn n_subscores(&self) -> usize { 2 }

    fn subscore_names(&self) -> Vec<String> {
        vec!["base".to_owned(), "attn".to_owned()]
    }

    fn predict_subscores(&self, u: usize, i: usize, day: i32) -> Array1<f32> {
        let base = self.baseline(u, i, day);
        Array1::from(vec![base, self.predict(u, i, day) - base])
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let (d, nmf) = (self.cfg.n_feat, self.cfg.n_mf);
        let (lo, hi) = (self.pool_off[u] as usize, self.pool_off[u + 1] as usize);
        let mut res = Vec::with_capacity(hi - lo);
        self.pool_residuals(u, &mut res);
        let mut w = Vec::with_capacity(hi - lo);
        let s = attend(
            &self.qi, &self.ci, self.cfg.beta, d, i, day,
            &self.pool_item[lo..hi], &self.pool_day[lo..hi], &res, &mut w,
        );
        let a = self.cfg.out_scale;
        let mf = if nmf > 0 {
            dot(&self.pmf[u * nmf..u * nmf + nmf], &self.qmf[i * nmf..i * nmf + nmf])
        } else { 0.0 };
        self.baseline(u, i, day) + mf + a * tanh(s / a)
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        let cfg = self.cfg;
        let (d, nmf, n_bins, dev_beta) = (cfg.n_feat, cfg.n_mf, cfg.n_bins, cfg.dev_beta);
        let decay = cfg.lr_decay.powi(epoch as i32 - 1);
        let (lr_q, lr_c) = (cfg.lr_q * decay, cfg.lr_c * decay);
        let (lr_mf, lr_bias, lr_alpha) =
            (cfg.lr_mf * decay, cfg.lr_bias * decay, cfg.lr_alpha * decay);
        let (mu, clip, a) = (self.mu, cfg.grad_clip, cfg.out_scale);
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();

        let (p_bu, p_bi) = (Ptr(self.bu.as_mut_ptr()), Ptr(self.bi.as_mut_ptr()));
        let p_al = Ptr(self.alpha_u.as_mut_ptr());
        let p_bd = Ptr(self.bu_day.as_mut_ptr());
        let p_bt = Ptr(self.bit_bin.as_mut_ptr());
        let (p_qi, p_ci) = (Ptr(self.qi.as_mut_ptr()), Ptr(self.ci.as_mut_ptr()));
        let (p_pm, p_qm) = (Ptr(self.pmf.as_mut_ptr()), Ptr(self.qmf.as_mut_ptr()));
        let (n_users, n_items, n_ud) = (tr.n_users, tr.n_items, self.bu_day.len());

        let (uoff, ud, u_day_mean) = (&self.uoff, &self.ud, &self.u_day_mean);
        let (pool_off, pool_item) = (&self.pool_off, &self.pool_item);
        let (pool_day, pool_rtg) = (&self.pool_day, &self.pool_rtg);

        let users = get_users(n_users, true, cfg.seed, epoch);
        let all: Vec<&[usize]> = users.as_slice().unwrap().chunks(cfg.block_users).collect();
        let n_use = ((all.len() as f32 * cfg.train_frac).ceil() as usize).clamp(1, all.len());
        let blocks = &all[..n_use];
        let n_par = if cfg.n_threads > 0 { cfg.n_threads } else { rayon::current_num_threads() };
        let pb = make_pb(blocks.len() as u64);

        let work = |block: &&[usize]| {
            let (bu, bi) = unsafe { (p_bu.sl(0, n_users), p_bi.sl(0, n_items)) };
            let al = unsafe { p_al.sl(0, n_users) };
            let bd = unsafe { p_bd.sl(0, n_ud) };
            let bt = unsafe { p_bt.sl(0, n_items * n_bins) };
            let qi = unsafe { p_qi.sl(0, n_items * d) };
            let ci = unsafe { p_ci.sl(0, n_items) };
            let mut w: Vec<f32> = Vec::new();
            let mut res: Vec<f32> = Vec::new();
            let mut dq: Vec<f32> = vec![0.0; d];
            let mut qt: Vec<f32> = vec![0.0; d]; // snapshot of q_i for the neighbor loop

            for &u in block.iter() {
                let (start, end) = (uoff[u], uoff[u + 1]);
                let (plo, phi) = (pool_off[u] as usize, pool_off[u + 1] as usize);
                if start == end || phi == plo { continue; }

                // Baseline residual of each pooled neighbor, refreshed once per
                // visit, since the baseline moves while we train.
                res.clear();
                for p in plo..phi {
                    let (j, t) = (pool_item[p] as usize, pool_day[p] as i32);
                    let bin = bias_time_bin(t, n_bins);
                    let base_j = mu + bu[u] + al[u] * user_time_dev(t, u_day_mean[u], dev_beta)
                        + ud.index(u, pool_day[p]).map_or(0.0, |ix| bd[ix])
                        + bi[j] + bt[j * n_bins + bin];
                    res.push(pool_rtg[p] - base_j);
                }

                for t in start..end {
                    let i = tr.item_idxs[t] as usize;
                    let day = tr.dates[t] as i32;
                    let bin = i * n_bins + bias_time_bin(day, n_bins);
                    let ud_ix = ud.index(u, tr.dates[t]);
                    let devu = user_time_dev(day, u_day_mean[u], dev_beta);

                    let s = attend(
                        qi, ci, cfg.beta, d, i, day,
                        &pool_item[plo..phi], &pool_day[plo..phi], &res, &mut w,
                    );
                    let tval = tanh(s / a);
                    let o = a * tval;

                    let qm = unsafe { p_qm.sl(i * nmf, nmf) };
                    let pm = unsafe { p_pm.sl(u * nmf, nmf) };
                    let mf = if nmf > 0 { dot(pm, qm) } else { 0.0 };

                    let base = mu + bu[u] + al[u] * devu + bi[i] + bt[bin]
                        + ud_ix.map_or(0.0, |ix| bd[ix]);
                    let mut err = (base + mf + o - tr.residuals[t]).clamp(-4.0, 4.0);
                    if cfg.huber > 0.0 { err = err.clamp(-cfg.huber, cfg.huber); }
                    let eo = err * (1.0 - tval * tval); // error through the tanh

                    // --- attention ---
                    // dL/dlogit_p = eo · α_p · (v_p − s), the softmax-mixture form.
                    // q_i is snapshotted first: every neighbor's gradient must
                    // see the same q_i, and q_i itself is updated after the loop.
                    dq.fill(0.0);
                    qt.copy_from_slice(&qi[i * d..i * d + d]);
                    for (p, &alpha) in w.iter().enumerate() {
                        if alpha == 0.0 { continue; }
                        // `w` indexes the user's slice of the pool, not the
                        // whole array.
                        let j = pool_item[plo + p] as usize;
                        if j == i { continue; }
                        let v = ci[j] * res[p];
                        ci[j] -= lr_c * (eo * alpha * res[p] + cfg.reg_c * ci[j]);
                        let dlogit = (eo * alpha * (v - s)).clamp(-clip, clip);
                        let qj = &mut qi[j * d..j * d + d];
                        axpy(&mut dq, qj, dlogit * inv_sqrt_d);
                        // q_j moves toward / away from q_i by the same amount
                        let g = dlogit * inv_sqrt_d;
                        for k in 0..d {
                            qj[k] -= lr_q * (g * qt[k] + cfg.reg_q * qj[k]);
                        }
                    }
                    {
                        let qrow = &mut qi[i * d..i * d + d];
                        for k in 0..d {
                            qrow[k] -= lr_q * (dq[k] + cfg.reg_q * qrow[k]);
                        }
                        cap_rms(qrow, cfg.emb_cap);
                    }

                    // --- bilinear term ---
                    for k in 0..nmf {
                        let (pk, qk) = (pm[k], qm[k]);
                        pm[k] -= lr_mf * (err * qk + cfg.reg_mf * pk);
                        qm[k] -= lr_mf * (err * pk + cfg.reg_mf * qk);
                    }

                    // --- temporal baseline ---
                    bu[u] -= lr_bias * (err + cfg.reg_bias * bu[u]);
                    bi[i] -= lr_bias * (err + cfg.reg_bias * bi[i]);
                    bt[bin] -= lr_bias * (err + cfg.reg_bias * bt[bin]);
                    al[u] -= lr_alpha * (err * devu + cfg.reg_alpha * al[u]);
                    if let Some(ix) = ud_ix {
                        bd[ix] -= lr_bias * (err + cfg.reg_bu_day * bd[ix]);
                    }
                }
            }
        };

        for round in blocks.chunks(n_par) {
            if n_par == 1 {
                round.iter().for_each(&work);
            } else {
                round.par_iter().for_each(&work);
            }
            pb.inc(round.len() as u64);
        }
        pb.finish_and_clear();
    }
}
