// Deep neural rating predictor. Three parts add up to a rating:
//
//   1. a BellKor-style temporal baseline: b_u, alpha_u·dev_u(t), a per-(user,day)
//      bias and item-bias time bins;
//   2. a plain bilinear term p·q of width `n_mf`, which costs O(n_mf) per rating
//      rather than the network's O(n_mf · h1) and so can be much wider than the
//      embeddings the MLP consumes;
//   3. a bounded correction a·tanh(raw/a) from a ReLU MLP whose input is the
//      user and item embeddings, their product, and N_CTX context features.
//
// The network is trained with Adam on user-block minibatches. Threads accumulate
// gradients and the update happens once per block, single-threaded, while the
// embeddings and biases use plain SGD. Blocks within a round run in parallel by
// default, so runs are not bit-exact; `n_threads = 1` makes the whole fit
// reproducible.

use crate::cfnade::fmath::{ln, powf, tanh};
use crate::tx::SparseUD;
use crate::{
    Dataset, MaskedDataset, Regressor, bias_time_bin, calc_user_offsets, get_users, make_pb,
    user_time_dev,
};
use ndarray::Array1;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Context features appended to the MLP input: log rating counts of the user
/// and the item, the day, the user's mean and spread, the item's mean, how long
/// the user has been active, and how many ratings they wrote that day.
pub const N_CTX: usize = 8;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct DnnConfig {
    // Architecture
    pub n_feat: usize,        // embedding dimension the MLP consumes
    pub h1: usize,            // first hidden layer width
    pub h2: usize,            // second hidden layer width
    pub n_mf: usize,          // width of the plain bilinear term beside the net (0 = off)
    pub n_bins: usize,        // item-bias time bins
    pub out_scale: f32,       // half-range of the network's bounded correction
    /// Scale that maps a rating date into roughly [-1, 1] for the context
    /// vector (`day / day_scale - 1`). Not a property of the data: 1128 is half
    /// of 2256, which is neither the true span (2243) nor the mean rating date
    /// (1790). Any nearby value serves as well, since the first layer rescales
    /// whatever it is handed, but moving it refits the model.
    pub day_scale: f32,
    pub dev_beta: f32,        // exponent in BellKor's |dt|^beta user drift term

    // Schedule
    pub n_epochs: usize,
    pub seed: u64,
    pub lr_decay: f32,        // multiplicative per-epoch decay of every rate

    // Learning rates
    pub lr_mlp: f32,          // Adam step size for the MLP weights
    pub lr_emb: f32,          // SGD rate for the p/q embeddings
    pub lr_mf: f32,           // SGD rate for the bilinear factors
    pub lr_bias: f32,         // SGD rate for the user/item biases
    pub lr_alpha: f32,        // SGD rate for the linear user drift alpha_u

    // Regularization
    pub reg_emb: f32,         // L2 on the embeddings (applied per update)
    pub reg_mf: f32,          // L2 on the bilinear factors (applied per update)
    pub reg_bias: f32,        // L2 on the biases (applied per update)
    pub reg_bu_day: f32,      // L2 on the per-(user, day) bias (own knob: it sees
                              // only a couple of updates per epoch, but shrinking
                              // it harder measurably hurts, so the term is signal)
    pub reg_alpha: f32,       // L2 on alpha_u (large: dev_u(t) has a wide range)
    pub reg_mlp: f32,         // L2 on the MLP weights (applied once per epoch)
    pub sigma: f32,           // embedding init std dev

    // Numerical guards
    pub grad_clip: f32,       // element-wise clamp on the backpropagated gradients
    pub emb_cap: f32,         // max-norm ball (per-coordinate RMS) for p and q

    // Execution
    pub n_threads: usize,     // blocks run concurrently; 1 = sequential, and then
                              // the whole fit is deterministic (0 = all cores)
    pub block_users: usize,   // users per minibatch
    pub train_frac: f32,      // fraction of the user blocks visited per epoch
                              // (below 1 only for quick tuning probes)
}

impl Default for DnnConfig {
    fn default() -> Self {
        Self {
            n_feat: 16,
            h1: 64,
            h2: 32,
            n_mf: 0,
            n_bins: 30,
            out_scale: 3.5,
            day_scale: 1128.0,
            dev_beta: 0.4,

            n_epochs: 14,
            seed: 42,
            lr_decay: 0.9,

            lr_mlp: 3e-5,
            lr_emb: 2e-2,
            lr_mf: 8e-3,
            lr_bias: 5e-3,
            lr_alpha: 2e-6,

            reg_emb: 0.02,
            reg_mf: 0.015,
            reg_bias: 0.03,
            reg_bu_day: 0.03,
            reg_alpha: 50.0,
            reg_mlp: 1e-4,
            sigma: 0.05,

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

/// Raw pointer into an embedding array, shared across worker threads. Only the
/// sparse per-user / per-item parameters are written this way; collisions
/// between blocks are rare and harmless (Hogwild).
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

// ---------------------------------------------------------------------------
// MLP parameter buffer
// ---------------------------------------------------------------------------

/// One buffer shaped like the MLP parameters, used for the weights themselves,
/// for a block's accumulated gradient, and for the two Adam moment estimates.
#[derive(Clone)]
struct MlpBuf {
    w1: Vec<f32>, b1: Vec<f32>,
    w2: Vec<f32>, b2: Vec<f32>,
    w3: Vec<f32>, b3: f32,
    rows: usize, // number of ratings accumulated (gradient buffers only)
}

impl MlpBuf {
    fn zeros(din: usize, n1: usize, n2: usize) -> Self {
        Self {
            w1: vec![0.0; din * n1], b1: vec![0.0; n1],
            w2: vec![0.0; n1 * n2], b2: vec![0.0; n2],
            w3: vec![0.0; n2], b3: 0.0,
            rows: 0,
        }
    }
}

/// One Adam update of `w` from the mean gradient `g / scale_div`.
fn adam_step(w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, t: f32, inv_rows: f32) {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    let c1 = 1.0 - powf(B1, t);
    let c2 = 1.0 - powf(B2, t);
    for k in 0..w.len() {
        let gk = g[k] * inv_rows;
        m[k] = B1 * m[k] + (1.0 - B1) * gk;
        v[k] = B2 * v[k] + (1.0 - B2) * gk * gk;
        w[k] -= lr * (m[k] / c1) / ((v[k] / c2).sqrt() + 1e-8);
    }
}

/// Project a parameter vector back into the max-norm ball. Without this the
/// shared implicit embeddings drift coherently, every j in N(u) taking the same
/// step, until the network's inputs, and with them its curvature, blow up.
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
// Context features
// ---------------------------------------------------------------------------

/// Read-only per-user / per-item statistics feeding the context features.
struct Ctx<'a> {
    u_logcnt: &'a [f32],
    u_mean: &'a [f32],
    u_std: &'a [f32],
    u_first: &'a [i16],
    i_logcnt: &'a [f32],
    i_mean: &'a [f32],
    ud: &'a SparseUD,
    /// `1 / cfg.day_scale`, precomputed: this runs once per rating.
    inv_day_scale: f32,
}

impl Ctx<'_> {
    /// Fill the `N_CTX` context features for one (user, item, day) triple.
    /// All values are roughly centred and scaled to unit order of magnitude.
    #[inline]
    fn fill(&self, u: usize, i: usize, day: i32, out: &mut [f32]) {
        let freq = self
            .ud
            .index(u, day as i16)
            .map_or(1.0, |ix| self.ud.day_cnts[ix] as f32);
        let recency = ((day - self.u_first[u] as i32).max(0) + 1) as f32;
        out[0] = self.u_logcnt[u];
        out[1] = self.i_logcnt[i];
        out[2] = day as f32 * self.inv_day_scale - 1.0;
        out[3] = self.u_mean[u];
        out[4] = self.i_mean[i];
        out[5] = self.u_std[u];
        out[6] = ln(recency) * 0.25 - 1.0;
        out[7] = ln(freq) * 0.5 - 0.3;
    }
}

/// Dot product with eight independent accumulators. A plain
/// `acc += a[k] * b[k]` reduction stays scalar, because LLVM may not reassociate
/// floating-point additions, and that costs roughly an order of magnitude here.
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

/// `out += scale * src`, the transposed counterpart of `dot`.
#[inline]
fn axpy(out: &mut [f32], src: &[f32], scale: f32) {
    for (o, &v) in out.iter_mut().zip(src) { *o += scale * v; }
}

// ---------------------------------------------------------------------------
// MLP forward pass
// ---------------------------------------------------------------------------

/// `x → relu(W1) → relu(W2) → a·tanh(raw/a)`. Writes the hidden activations
/// into `h1`/`h2` for the backward pass and returns the bounded correction
/// together with the tanh derivative, which damps the gradient on saturation.
#[inline]
fn forward(x: &[f32], p: &MlpBuf, a: f32, h1: &mut [f32], h2: &mut [f32]) -> (f32, f32) {
    let n1 = h1.len();
    let n2 = h2.len();

    h1.copy_from_slice(&p.b1);
    for (j, &xj) in x.iter().enumerate() {
        if xj == 0.0 { continue; }
        axpy(h1, &p.w1[j * n1..j * n1 + n1], xj);
    }
    for v in h1.iter_mut() { if *v < 0.0 { *v = 0.0; } }

    h2.copy_from_slice(&p.b2);
    for j in 0..n1 {
        let hj = h1[j];
        if hj == 0.0 { continue; }
        axpy(h2, &p.w2[j * n2..j * n2 + n2], hj);
    }
    for v in h2.iter_mut() { if *v < 0.0 { *v = 0.0; } }

    let o = p.b3 + dot(&p.w3, h2);
    let t = tanh(o / a);
    (a * t, 1.0 - t * t)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct DnnModel {
    cfg: DnnConfig,
    din: usize,           // 3 · n_feat + N_CTX
    mu: f32,              // global mean

    bu: Vec<f32>,         // [n_users]
    bi: Vec<f32>,         // [n_items]
    alpha_u: Vec<f32>,    // [n_users] linear drift slope
    bu_day: Vec<f32>,     // [ud.n_total()] per-(user, day) bias
    bit_bin: Vec<f32>,    // [n_items × n_bins] item bias per time bin
    u_day_mean: Vec<f32>, // [n_users] mean rating date
    pu: Vec<f32>,         // [n_users × d] free user embedding
    pmf: Vec<f32>,        // [n_users × n_mf] bilinear user factors
    qmf: Vec<f32>,        // [n_items × n_mf] bilinear item factors
    qi: Vec<f32>,         // [n_items × d] item embedding

    mlp: MlpBuf,
    mom: MlpBuf,          // Adam first moment
    vel: MlpBuf,          // Adam second moment
    adam_t: f32,

    u_logcnt: Vec<f32>, u_mean: Vec<f32>, u_std: Vec<f32>, u_first: Vec<i16>,
    i_logcnt: Vec<f32>, i_mean: Vec<f32>,

    uoff: Vec<usize>,     // [n_users + 1] train-set user offsets
    ud: SparseUD,
}

impl DnnModel {
    /// Temporal baseline: global mean, user bias with a linear drift and a
    /// per-day term, item bias with a time bin. This is what the network's
    /// correction is added on top of.
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

    #[inline]
    fn ctx(&self) -> Ctx<'_> {
        Ctx {
            u_logcnt: &self.u_logcnt,
            u_mean: &self.u_mean,
            u_std: &self.u_std,
            u_first: &self.u_first,
            i_logcnt: &self.i_logcnt,
            i_mean: &self.i_mean,
            ud: &self.ud,
            inv_day_scale: 1.0 / self.cfg.day_scale,
        }
    }

    /// Apply one block's accumulated gradient to the MLP weights.
    fn mlp_update(&mut self, g: &MlpBuf, lr: f32) {
        if g.rows == 0 { return; }
        self.adam_t += 1.0;
        let (t, inv) = (self.adam_t, 1.0 / g.rows as f32);
        adam_step(&mut self.mlp.w1, &g.w1, &mut self.mom.w1, &mut self.vel.w1, lr, t, inv);
        adam_step(&mut self.mlp.b1, &g.b1, &mut self.mom.b1, &mut self.vel.b1, lr, t, inv);
        adam_step(&mut self.mlp.w2, &g.w2, &mut self.mom.w2, &mut self.vel.w2, lr, t, inv);
        adam_step(&mut self.mlp.b2, &g.b2, &mut self.mom.b2, &mut self.vel.b2, lr, t, inv);
        adam_step(&mut self.mlp.w3, &g.w3, &mut self.mom.w3, &mut self.vel.w3, lr, t, inv);
        let mut b3 = [self.mlp.b3];
        let (mut m3, mut v3) = ([self.mom.b3], [self.vel.b3]);
        adam_step(&mut b3, &[g.b3], &mut m3, &mut v3, lr, t, inv);
        self.mlp.b3 = b3[0];
        self.mom.b3 = m3[0];
        self.vel.b3 = v3[0];
    }
}

// ---------------------------------------------------------------------------
// Regressor impl
// ---------------------------------------------------------------------------

impl Regressor for DnnModel {
    type Config = DnnConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let (n_users, n_items, d) = (tr.n_users, tr.n_items, cfg.n_feat);
        let din = 3 * d + N_CTX;

        // --- global / per-item / per-user rating statistics -----------------
        let mu = (tr.residuals.iter().map(|&x| x as f64).sum::<f64>() / tr.n_ratings as f64) as f32;

        let mut i_sum = vec![0.0f64; n_items];
        let mut i_cnt = vec![0u32; n_items];
        let mut u_sum = vec![0.0f64; n_users];
        let mut u_sq = vec![0.0f64; n_users];
        let mut u_cnt = vec![0u32; n_users];
        let mut u_first = vec![i16::MAX; n_users];

        let mut u_day_sum = vec![0.0f64; n_users];
        for t in 0..tr.n_ratings {
            let (u, i) = (tr.user_idxs[t] as usize, tr.item_idxs[t] as usize);
            let r = tr.residuals[t] as f64;
            i_sum[i] += r;
            i_cnt[i] += 1;
            u_sum[u] += r;
            u_sq[u] += r * r;
            u_cnt[u] += 1;
            u_day_sum[u] += tr.dates[t] as f64;
            if tr.dates[t] < u_first[u] { u_first[u] = tr.dates[t]; }
        }
        for f in u_first.iter_mut() {
            if *f == i16::MAX { *f = 0; }
        }
        let u_day_mean: Vec<f32> = (0..n_users)
            .map(|u| if u_cnt[u] > 0 { (u_day_sum[u] / u_cnt[u] as f64) as f32 } else { 0.0 })
            .collect();

        // Shrunk baseline biases: a good starting point so the net only has to
        // learn the interaction on top of it.
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

        let i_logcnt: Vec<f32> = (0..n_items)
            .map(|i| (ln(1.0 + i_cnt[i] as f32) - 7.5) * 0.5)
            .collect();
        let i_mean: Vec<f32> = (0..n_items)
            .map(|i| if i_cnt[i] > 0 { (i_sum[i] / i_cnt[i] as f64) as f32 - mu } else { 0.0 })
            .collect();
        let u_logcnt: Vec<f32> = (0..n_users)
            .map(|u| (ln(1.0 + u_cnt[u] as f32) - 4.5) * 0.5)
            .collect();
        let u_mean: Vec<f32> = (0..n_users)
            .map(|u| if u_cnt[u] > 0 { (u_sum[u] / u_cnt[u] as f64) as f32 - mu } else { 0.0 })
            .collect();
        let u_std: Vec<f32> = (0..n_users)
            .map(|u| {
                if u_cnt[u] < 2 { return 0.0; }
                let m = u_sum[u] / u_cnt[u] as f64;
                ((u_sq[u] / u_cnt[u] as f64 - m * m).max(0.0).sqrt() as f32) - 1.05
            })
            .collect();

        let uoff: Vec<usize> = calc_user_offsets(tr).to_vec();

        // --- parameters -----------------------------------------------------
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let gauss = |n: usize, s: f32, rng: &mut StdRng| -> Vec<f32> {
            let dist = Normal::<f32>::new(0.0, s).unwrap();
            (0..n).map(|_| dist.sample(rng)).collect()
        };
        // A plain matrix-factorisation term sitting beside the network. Its
        // cost is O(n_mf) per rating rather than the network's O(n_mf · h1), so
        // it can be far wider than the embeddings the MLP consumes, which is
        // where the accuracy of the ensemble's own MF models comes from.
        let pmf = gauss(n_users * cfg.n_mf, cfg.sigma, &mut rng);
        let qmf = gauss(n_items * cfg.n_mf, cfg.sigma, &mut rng);
        let pu = gauss(n_users * d, cfg.sigma, &mut rng);
        let qi = gauss(n_items * d, cfg.sigma, &mut rng);

        // He init for the hidden layers; a small output layer keeps the first
        // predictions close to the baseline while still passing gradient back.
        let mlp = MlpBuf {
            w1: gauss(din * cfg.h1, (2.0 / din as f32).sqrt(), &mut rng),
            b1: vec![0.0; cfg.h1],
            w2: gauss(cfg.h1 * cfg.h2, (2.0 / cfg.h1 as f32).sqrt(), &mut rng),
            b2: vec![0.0; cfg.h2],
            w3: gauss(cfg.h2, 0.05, &mut rng),
            b3: 0.0,
            rows: 0,
        };

        let ud = SparseUD::new(tr, pr);
        let model = Self {
            cfg, din, mu, bu, bi, pu, pmf, qmf, qi,
            alpha_u: vec![0.0; n_users],
            bu_day: vec![0.0; ud.n_total()],
            bit_bin: vec![0.0; n_items * cfg.n_bins],
            u_day_mean,
            mom: MlpBuf::zeros(din, cfg.h1, cfg.h2),
            vel: MlpBuf::zeros(din, cfg.h1, cfg.h2),
            mlp,
            adam_t: 0.0,
            u_logcnt, u_mean, u_std, u_first, i_logcnt, i_mean,
            uoff, ud,
        };
        model
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn n_subscores(&self) -> usize { 2 }

    fn subscore_names(&self) -> Vec<String> {
        vec!["base".to_owned(), "net".to_owned()]
    }

    /// Split the prediction into the bias baseline and the network's
    /// correction, so a blender can weight the (largely redundant) baseline
    /// and the nonlinear part separately.
    fn predict_subscores(&self, u: usize, i: usize, day: i32) -> Array1<f32> {
        let base = self.baseline(u, i, day);
        Array1::from(vec![base, self.predict(u, i, day) - base])
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let (d, n1, n2) = (self.cfg.n_feat, self.cfg.h1, self.cfg.h2);
        let mut x = vec![0.0f32; self.din];
        let mut h1 = vec![0.0f32; n1];
        let mut h2 = vec![0.0f32; n2];
        self.ctx().fill(u, i, day, &mut x[3 * d..]);
        for k in 0..d {
            let p = self.pu[u * d + k];
            let q = self.qi[i * d + k];
            x[k] = p * q;
            x[d + k] = p;
            x[2 * d + k] = q;
        }
        let (o, _) = forward(&x, &self.mlp, self.cfg.out_scale, &mut h1, &mut h2);
        let nmf = self.cfg.n_mf;
        let mf = if nmf > 0 {
            dot(&self.pmf[u * nmf..u * nmf + nmf], &self.qmf[i * nmf..i * nmf + nmf])
        } else { 0.0 };
        self.baseline(u, i, day) + mf + o
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        let cfg = self.cfg;
        let (d, n1, n2, din) = (cfg.n_feat, cfg.h1, cfg.h2, self.din);
        let decay = cfg.lr_decay.powi(epoch as i32 - 1);
        let (lr_mlp, lr_emb, lr_bias) =
            (cfg.lr_mlp * decay, cfg.lr_emb * decay, cfg.lr_bias * decay);
        let (mu, clip) = (self.mu, cfg.grad_clip);
        let (nmf, lr_mf) = (cfg.n_mf, cfg.lr_mf * decay);
        let (p_pm, p_qm) = (Ptr(self.pmf.as_mut_ptr()), Ptr(self.qmf.as_mut_ptr()));
        let (out_scale, lr_alpha) = (cfg.out_scale, cfg.lr_alpha * decay);

        // Raw pointers to the sparse parameters. These don't hold a borrow, so
        // the read-only slices taken from `self` below can coexist with them.
        let (p_bu, p_bi) = (Ptr(self.bu.as_mut_ptr()), Ptr(self.bi.as_mut_ptr()));
        let p_al = Ptr(self.alpha_u.as_mut_ptr());
        let p_bd = Ptr(self.bu_day.as_mut_ptr());
        let p_bt = Ptr(self.bit_bin.as_mut_ptr());
        let (n_bins, n_ud, dev_beta) = (cfg.n_bins, self.bu_day.len(), cfg.dev_beta);
        let (p_pu, p_qi) = (Ptr(self.pu.as_mut_ptr()), Ptr(self.qi.as_mut_ptr()));
        let (n_users, n_items) = (tr.n_users, tr.n_items);

        let users = get_users(n_users, true, cfg.seed, epoch);
        let all: Vec<&[usize]> = users.as_slice().unwrap().chunks(cfg.block_users).collect();
        let n_use = ((all.len() as f32 * cfg.train_frac).ceil() as usize).clamp(1, all.len());
        let blocks = &all[..n_use];
        let n_par = if cfg.n_threads > 0 { cfg.n_threads } else { rayon::current_num_threads() };
        let pb = make_pb(blocks.len() as u64);

        for round in blocks.chunks(n_par) {
            // Threads only read the MLP weights and accumulate gradients; the
            // weights themselves are updated after the round, single-threaded.
            let grads: Vec<MlpBuf> = {
                let ctx = self.ctx();
                let mlp = &self.mlp;
                let uoff = &self.uoff;
                let (ud, u_day_mean) = (&self.ud, &self.u_day_mean);

                let work = |block: &&[usize]| -> MlpBuf {
                    let mut g = MlpBuf::zeros(din, n1, n2);
                    let mut x = vec![0.0f32; din];
                    let mut h1 = vec![0.0f32; n1];
                    let mut h2 = vec![0.0f32; n2];
                    let mut dh1 = vec![0.0f32; n1];
                    let mut dh2 = vec![0.0f32; n2];
                    let mut dx = vec![0.0f32; 3 * d];
                    let mut puu = vec![0.0f32; d];
                    let (bu, bi) = unsafe { (p_bu.sl(0, n_users), p_bi.sl(0, n_items)) };
                    let al = unsafe { p_al.sl(0, n_users) };
                    let bd = unsafe { p_bd.sl(0, n_ud) };
                    let bt = unsafe { p_bt.sl(0, n_items * n_bins) };

                    for &u in block.iter() {
                        let (start, end) = (uoff[u], uoff[u + 1]);
                        if start == end { continue; }
                        let pm = unsafe { p_pm.sl(u * nmf, nmf) };
                        puu.copy_from_slice(unsafe { p_pu.sl(u * d, d) });

                        for t in start..end {
                            let i = tr.item_idxs[t] as usize;
                            let qi = unsafe { p_qi.sl(i * d, d) };
                            let day = tr.dates[t] as i32;
                            let bin = i * n_bins + bias_time_bin(day, n_bins);
                            let ud_ix = ud.index(u, tr.dates[t]);
                            let devu = user_time_dev(day, u_day_mean[u], dev_beta);
                            let qm = unsafe { p_qm.sl(i * nmf, nmf) };
                            let mf = if nmf > 0 { dot(pm, qm) } else { 0.0 };
                            let base = mu + bu[u] + al[u] * devu + bi[i] + bt[bin]
                                + ud_ix.map_or(0.0, |ix| bd[ix])
                                + mf;

                            ctx.fill(u, i, day, &mut x[3 * d..]);
                            for k in 0..d {
                                let q = qi[k];
                                x[k] = puu[k] * q;
                                x[d + k] = puu[k];
                                x[2 * d + k] = q;
                            }


                            let (o, dfac) = forward(&x, mlp, out_scale, &mut h1, &mut h2);
                            let err = (base + o - tr.residuals[t]).clamp(-4.0, 4.0);
                            let eo = err * dfac; // error as seen through the tanh
                            g.rows += 1;

                            // --- output layer ---
                            for k in 0..n2 {
                                dh2[k] = if h2[k] > 0.0 {
                                    (eo * mlp.w3[k]).clamp(-clip, clip)
                                } else { 0.0 };
                                g.w3[k] += eo * h2[k];
                            }
                            g.b3 += eo;

                            // --- second hidden layer ---
                            for j in 0..n1 {
                                let hj = h1[j];
                                if hj <= 0.0 { dh1[j] = 0.0; continue; }
                                let acc = dot(&mlp.w2[j * n2..j * n2 + n2], &dh2);
                                axpy(&mut g.w2[j * n2..j * n2 + n2], &dh2, hj);
                                dh1[j] = acc.clamp(-clip, clip);
                            }
                            for k in 0..n2 { g.b2[k] += dh2[k]; }

                            // --- first hidden layer (dx only for the embeddings) ---
                            for j in 0..din {
                                if j < 3 * d {
                                    let acc = dot(&mlp.w1[j * n1..j * n1 + n1], &dh1);
                                    dx[j] = acc.clamp(-clip, clip);
                                }
                                let xj = x[j];
                                if xj != 0.0 {
                                    axpy(&mut g.w1[j * n1..j * n1 + n1], &dh1, xj);
                                }
                            }
                            for k in 0..n1 { g.b1[k] += dh1[k]; }

                            // --- embeddings and biases (sparse SGD) ---
                            for k in 0..d {
                                let q = qi[k];
                                let du = dx[k] * q + dx[d + k];
                                let dq = dx[k] * puu[k] + dx[2 * d + k];
                                qi[k] = q - lr_emb * (dq + cfg.reg_emb * q);
                                puu[k] -= lr_emb * (du + cfg.reg_emb * puu[k]);
                            }
                            cap_rms(qi, cfg.emb_cap);
                            cap_rms(&mut puu, cfg.emb_cap);
                            // Plain MF gradient, exactly as a standalone SVD would do it.
                            for k in 0..nmf {
                                let (pk, qk) = (pm[k], qm[k]);
                                pm[k] -= lr_mf * (err * qk + cfg.reg_mf * pk);
                                qm[k] -= lr_mf * (err * pk + cfg.reg_mf * qk);
                            }
                            bu[u] -= lr_bias * (err + cfg.reg_bias * bu[u]);
                            bi[i] -= lr_bias * (err + cfg.reg_bias * bi[i]);
                            bt[bin] -= lr_bias * (err + cfg.reg_bias * bt[bin]);
                            al[u] -= lr_alpha * (err * devu + cfg.reg_alpha * al[u]);
                            if let Some(ix) = ud_ix {
                                bd[ix] -= lr_bias * (err + cfg.reg_bu_day * bd[ix]);
                            }
                        }

                        unsafe { p_pu.sl(u * d, d) }.copy_from_slice(&puu);

                    }
                    g
                };
                // Sequential blocks race on nothing, so the run is reproducible.
                if n_par == 1 {
                    round.iter().map(&work).collect()
                } else {
                    round.par_iter().map(&work).collect()
                }
            };

            for g in &grads {
                self.mlp_update(g, lr_mlp);
            }
            pb.inc(round.len() as u64);
        }
        pb.finish_and_clear();

        // Epoch-level L2 on the MLP weights, then a consistent scache.
        let keep = 1.0 - cfg.reg_mlp;
        for w in self.mlp.w1.iter_mut().chain(self.mlp.w2.iter_mut()).chain(self.mlp.w3.iter_mut()) {
            *w *= keep;
        }

        // Debug probes only run in the reduced-data tuning mode.
        if cfg.train_frac < 1.0 {
            self.report_parts(tr);
        }
    }
}

impl DnnModel {
    /// Split the training error into its baseline and network parts on a
    /// strided sample, tells apart "the biases drifted" from "the net blew up".
    fn report_parts(&self, tr: &Dataset) {
        let step = (tr.n_ratings / 100_000).max(1);
        let (mut so, mut sb, mut sp, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut omax = 0.0f32;
        for t in (0..tr.n_ratings).step_by(step) {
            let (u, i) = (tr.user_idxs[t] as usize, tr.item_idxs[t] as usize);
            let pred = self.predict(u, i, tr.dates[t] as i32);
            let base = self.baseline(u, i, tr.dates[t] as i32);
            let o = pred - base;
            so += (o * o) as f64;
            sb += ((base - tr.residuals[t]) as f64).powi(2);
            sp += ((pred - tr.residuals[t]) as f64).powi(2);
            omax = omax.max(o.abs());
            n += 1.0;
        }
        let rms = |v: &[f32]| -> f32 {
            if v.is_empty() { return 0.0; }
            (v.iter().map(|&x| (x * x) as f64).sum::<f64>() / v.len() as f64).sqrt() as f32
        };
        crate::teeln!(
            "         rms(o) {:.3} max|o| {:.2} | base {:.4} full {:.4} \
             | p {:.3} q {:.3} w1 {:.3} w3 {:.3} b3 {:.3}",
            (so / n).sqrt(), omax, (sb / n).sqrt(), (sp / n).sqrt(),
            rms(&self.pu), rms(&self.qi),
            rms(&self.mlp.w1), rms(&self.mlp.w3), self.mlp.b3
        );
    }
}
