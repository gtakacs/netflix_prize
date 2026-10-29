// Koren's integrated model trained by ALS, built up term by term (split_3; run
// by `s3 integ`, which reads the knobs named below from the environment).
// TERMS=y SORT=0 is the NSVD1 model that used to be lab-s3-nsvd (its REG_W is
// REG_Y here).
//
// Model, with the terms switched on by TERMS (a '+'-list of p, y, nb):
//
//   r_ui ~ mu + b_u + b_i + q_i . v_u + nb_ui,   v_u = p_u + z_u,   z_u = c_u sum_{j in N(u)} y_j
//
//   p   free user factors p_u (TERMS=p: biased MF)
//   y   implicit factors y_j, c_u = |N(u)|^-ALPHA (TERMS=y: NSVD1, TERMS=p+y: SVD++)
//   nb  Koren's neighborhood term over the NB_K items S(i) most similar to i:
//         nb_ui = |R(i;u)|^-1/2 sum_{j in R(i;u)} w_ij (r_uj - bb_uj) + |N(i;u)|^-1/2 sum_{j in N(i;u)} c_ij
//       R(i;u) = R(u) & S(i), N(i;u) = N(u) & S(i); bb_uj = mu + bb_u + bb_j is a
//       fixed baseline (biases alone, fitted once), so the term is linear in w, c.
//       S(i) from a precomputed sim: NB_SIM=mse (NB_STAT0 supp, NB_STAT2 diff2,
//       as knn-m) or cos (NB_STAT2 a *_prod), shrinkage NB_SHR.
//   ud  user-day bias b_ut: one parameter per (user, day) with training ratings;
//       0 for a (user, day) without any (prediction on an unseen day)
//   dr  user drift alpha_u dev_u(t), dev_u(t) = sign(t - tbar_u) |t - tbar_u|^DR_BETA,
//       tbar_u the mean day of u's training ratings
//   fq  frequency item bias b_{i,f}: f = floor(log_FQ_BASE F_ut), F_ut = number of
//       ratings u gave on day t, training and prediction set together (Koren 2009)
//
// N(u) = the items u rated in the training set (IMPL=1: plus u's items in the
// prediction set, rating unknown).
//
// Loss, one objective for every step:
//
//   L = sum_(u,i) (r_ui - mu - b_u - alpha_u dev_u(t) - b_ut - b_i - b_{i,f} - q_i . v_u - nb_ui)^2
//       + REG_BU |b_u|^2 + REG_DR |alpha|^2 + REG_UD |b_ut|^2 + REG_BI |b_i|^2 + REG_FQ |b_if|^2
//       + REG_P |P|^2 + REG_Q |Q|^2 + REG_Y |Y|^2 + REG_NW |W|^2 + REG_NC |C|^2
//
// Each epoch minimizes L exactly over one block at a time:
//   1. (p_u, alpha_u, b_u)  per-user ridge on x_i = (q_i, dev_u(t), 1), target the
//                   residual without these (without p / dr: the smaller system)
//   1b. b_ut        closed form per (user, day): mean residual shrunk by REG_UD
//   1c. b_{i,f}     closed form per (item, frequency bin), shrunk by REG_FQ
//   2. (q_i, b_i, w_i., c_i.)  per-item ridge on x_u = (v_u, 1, nb features of (u, i)),
//                   target r - mu - b_u
//   3. Y            jointly: L is quadratic in Y (z_u is linear in Y), so Y solves
//                   (H + REG_Y I) Y = B; block-Jacobi preconditioned conjugate
//                   gradient from the current Y, H never formed: the matrix-vector
//                   product is two passes over the ratings, the preconditioner holds
//                   one F x F block per item
// Exact block minimization cannot raise L, so the run prints L after every block
// and flags any rise (LOSSCHK=1). Everything is f64.

use nalgebra::{DMatrix, DVector};
use crate::{Dataset, MaskedDataset, Regressor, calc_gbias};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct IntegConfig {
    pub use_p: bool,
    pub use_y: bool,
    pub use_nb: bool,
    pub use_ud: bool,
    pub use_dr: bool,
    pub use_fq: bool,
    pub n_feat: usize,
    pub n_epochs: usize,
    pub alpha: f64,
    pub reg_bu: f64,
    pub reg_bi: f64,
    pub reg_p: f64,
    pub reg_q: f64,
    pub reg_y: f64,
    pub reg_nw: f64,
    pub reg_nc: f64,
    pub reg_ud: f64,
    pub reg_dr: f64,
    pub dr_beta: f64,
    pub reg_fq: f64,
    pub fq_base: f64,
    pub nb_k: usize,
    pub nb_start: usize,
    pub nb_freeze: bool,
    pub nb_mse: bool,
    pub nb_stat0: &'static str,
    pub nb_stat2: &'static str,
    pub nb_shr: f64,
    pub sim_dir: &'static str,
    pub sigma: f64,
    pub cg_iters: usize,
    pub cg_tol: f64,
    pub precond: bool,
    pub implicit: bool,
    pub seed: u64,
    pub loss_check: bool,
    pub sort: bool,
}

/// Compressed rows: row r holds idx[off[r]..off[r + 1]] (and val, if any).
struct Csr {
    off: Vec<usize>,
    idx: Vec<u32>,
    val: Vec<f32>,
}

impl Csr {
    fn row(&self, r: usize) -> std::ops::Range<usize> { self.off[r]..self.off[r + 1] }
}

/// Rows of (col, val) pairs grouped by `rows[t]`, sorted by col within a row
/// (`sort`) or in input order.
fn group(n_rows: usize, rows: &[i32], cols: &[i32], vals: Option<&[f32]>, sort: bool) -> Csr {
    let mut off = vec![0usize; n_rows + 1];
    for &r in rows { off[r as usize + 1] += 1; }
    for r in 0..n_rows { off[r + 1] += off[r]; }
    let mut pos = off.clone();
    let mut idx = vec![0u32; rows.len()];
    let mut val = vec![0f32; if vals.is_some() { rows.len() } else { 0 }];
    for t in 0..rows.len() {
        let r = rows[t] as usize;
        idx[pos[r]] = cols[t] as u32;
        if let Some(v) = vals { val[pos[r]] = v[t]; }
        pos[r] += 1;
    }
    if !sort { return Csr { off, idx, val }; }
    let has_val = vals.is_some();
    let mut rows: Vec<(&mut [u32], &mut [f32])> = Vec::with_capacity(n_rows);
    let (mut ri, mut rv) = (&mut idx[..], &mut val[..]);
    for r in 0..n_rows {
        let len = off[r + 1] - off[r];
        let (a, b) = std::mem::take(&mut ri).split_at_mut(len);
        let (c, d) = std::mem::take(&mut rv).split_at_mut(if has_val { len } else { 0 });
        rows.push((a, c));
        ri = b;
        rv = d;
    }
    rows.into_par_iter().for_each(|(ix, vs)| {
        if vs.is_empty() { ix.sort_unstable(); return; }
        let mut pairs: Vec<(u32, f32)> = ix.iter().copied().zip(vs.iter().copied()).collect();
        pairs.sort_unstable_by_key(|x| x.0);
        for (k, (a, b)) in pairs.into_iter().enumerate() { ix[k] = a; vs[k] = b; }
    });
    Csr { off, idx, val }
}

/// Item-step solution of one item: (q_i, b_i, w_i., c_i.) and the item's new nb_ui
/// values by training-rating position.
type ItemSol = (Vec<f64>, Vec<(usize, f64)>);

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

/// Solve the symmetric system whose lower triangle (row-major n x n) is in `a`.
fn ridge_solve(a: &mut [f64], b: &[f64], n: usize) -> Vec<f64> {
    if n == 1 { return vec![b[0] / a[0]]; }
    for r in 0..n { for c in 0..r { a[c * n + r] = a[r * n + c]; } }
    DMatrix::from_row_slice(n, n, a).cholesky().expect("ridge system not positive definite")
        .solve(&DVector::from_row_slice(b)).iter().copied().collect()
}

pub struct IntegModel {
    cfg: IntegConfig,
    n_items: usize,
    mu: f64,
    bu: Vec<f64>,
    bi: Vec<f64>,
    p: Vec<f64>, // n_users x F (zero without the p term)
    q: Vec<f64>, // n_items x F
    y: Vec<f64>, // n_items x F (zero without the y term)
    z: Vec<f64>, // n_users x F, c_u sum_{j in N(u)} y_j
    cu: Vec<f64>,
    rated: Csr,   // user -> (item, rating), training ratings
    raters: Csr,  // item -> (user, rating)
    present: Csr, // user -> items of N(u)
    // nb term
    nbr: Vec<u32>,       // n_items x NB_K: S(i), most similar first
    bbu: Vec<f64>,       // fixed baseline biases
    bbi: Vec<f64>,
    nw: Vec<f64>,        // n_items x NB_K, w_ij by slot of j in S(i)
    nc: Vec<f64>,        // n_items x NB_K, c_ij
    nbv: Vec<f64>,       // nb_ui per training rating, in `rated` order
    nb_on: bool,         // nb term fitted this epoch (from epoch NB_START)
    frozen: bool,        // p, q, y fixed this epoch (NB_FREEZE, once nb is on)
    rat2rated: Vec<u32>, // `raters` position -> `rated` position
    // ud term
    ud_off: Vec<usize>,  // user -> slots ud_off[u]..ud_off[u + 1]
    ud_day: Vec<i16>,    // day of each slot, ascending within a user
    rslot: Vec<u32>,     // slot of each training rating, in `rated` order
    bud: Vec<f64>,       // b_ut per slot
    // dr term
    tbar: Vec<f64>,      // mean training day per user
    rdev: Vec<f64>,      // dev_u(t) of each training rating, `rated` order
    alpha: Vec<f64>,     // alpha_u
    // fq term
    dc_off: Vec<usize>,  // user -> days dc_off[u]..dc_off[u + 1] (training + prediction set)
    dc_day: Vec<i16>,    // ascending within a user
    dc_cnt: Vec<u32>,    // ratings of u on that day
    rfbin: Vec<u8>,      // frequency bin of each training rating, `rated` order
    n_fbins: usize,
    bif: Vec<f64>,       // n_items x n_fbins
    last_loss: f64,
}

/// Sparse nb features of one (u, i): (slot, value) for the w part and the c part.
#[derive(Default)]
struct NbFeat {
    w: Vec<(usize, f64)>,
    c: Vec<(usize, f64)>,
}

impl IntegModel {
    fn f(&self) -> usize { self.cfg.n_feat }
    fn n_users(&self) -> usize { self.bu.len() }

    fn update_z(&mut self) {
        let f = self.f();
        let (y, cu, present) = (&self.y, &self.cu, &self.present);
        self.z.par_chunks_mut(f).enumerate().for_each(|(u, zu)| {
            zu.fill(0.0);
            for t in present.row(u) {
                let j = present.idx[t] as usize;
                for k in 0..f { zu[k] += y[j * f + k]; }
            }
            for z in zu.iter_mut() { *z *= cu[u]; }
        });
    }

    /// v_u = p_u + z_u into `out`.
    fn v(&self, u: usize, out: &mut [f64]) {
        let f = self.f();
        let (pu, zu) = (&self.p[u * f..(u + 1) * f], &self.z[u * f..(u + 1) * f]);
        for ((o, &p), &z) in out.iter_mut().zip(pu).zip(zu) { *o = p + z; }
    }

    /// Position of item j in row u of `csr` (rows sorted by item).
    fn find(csr: &Csr, u: usize, j: u32) -> Option<usize> {
        let r = csr.row(u);
        csr.idx[r.clone()].binary_search(&j).ok().map(|k| r.start + k)
    }

    /// The nb features of (u, i) into `out`.
    fn nb_feat(&self, u: usize, i: usize, out: &mut NbFeat) {
        out.w.clear();
        out.c.clear();
        let k = self.cfg.nb_k;
        for s in 0..k {
            let j = self.nbr[i * k + s];
            if j == u32::MAX { break; }
            if let Some(t) = Self::find(&self.rated, u, j) {
                let jj = j as usize;
                out.w.push((s, self.rated.val[t] as f64 - self.mu - self.bbu[u] - self.bbi[jj]));
                if !self.cfg.implicit { out.c.push((s, 1.0)); }
            }
            if self.cfg.implicit && Self::find(&self.present, u, j).is_some() { out.c.push((s, 1.0)); }
        }
        for part in [&mut out.w, &mut out.c] {
            if !part.is_empty() {
                let nrm = (part.len() as f64).powf(-0.5);
                part.iter_mut().for_each(|x| x.1 *= nrm);
            }
        }
    }

    fn nb_value(&self, i: usize, x: &NbFeat) -> f64 {
        let k = self.cfg.nb_k;
        x.w.iter().map(|&(s, v)| self.nw[i * k + s] * v).sum::<f64>()
            + x.c.iter().map(|&(s, v)| self.nc[i * k + s] * v).sum::<f64>()
    }

    /// nb_ui of training rating `t` (rated order); 0 without the nb term.
    fn nbt(&self, t: usize) -> f64 { if self.cfg.use_nb { self.nbv[t] } else { 0.0 } }

    /// b_ut of training rating `t` (rated order); 0 without the ud term.
    fn udt(&self, t: usize) -> f64 { if self.cfg.use_ud { self.bud[self.rslot[t] as usize] } else { 0.0 } }

    /// alpha_u dev_u(t) of training rating `t` of user `u`; 0 without the dr term.
    fn drt(&self, u: usize, t: usize) -> f64 { if self.cfg.use_dr { self.alpha[u] * self.rdev[t] } else { 0.0 } }

    /// b_{i,f} of training rating `t`; 0 without the fq term.
    fn fqt(&self, t: usize) -> f64 {
        if self.cfg.use_fq { self.bif[self.rated.idx[t] as usize * self.n_fbins + self.rfbin[t] as usize] } else { 0.0 }
    }

    /// The time terms of training rating `t` of user `u` (b_ut, b_{i,f}, drift).
    fn tterms(&self, u: usize, t: usize) -> f64 { self.udt(t) + self.fqt(t) + self.drt(u, t) }

    fn dev(beta: f64, day: f64, tbar: f64) -> f64 { let d = day - tbar; d.signum() * d.abs().powf(beta) }

    fn fbin(base: f64, n_fbins: usize, cnt: u32) -> u8 {
        (((cnt.max(1) as f64).ln() / base.ln()).floor() as usize).min(n_fbins - 1) as u8
    }

    /// b_{i,f} for any (user, item, day); a day unknown for u counts as F_ut = 1.
    fn fq_at(&self, u: usize, i: usize, day: i32) -> f64 {
        if !self.cfg.use_fq { return 0.0; }
        let (a, b) = (self.dc_off[u], self.dc_off[u + 1]);
        let cnt = self.dc_day[a..b].binary_search(&(day as i16)).map_or(1, |k| self.dc_cnt[a + k]);
        self.bif[i * self.n_fbins + Self::fbin(self.cfg.fq_base, self.n_fbins, cnt) as usize]
    }

    /// b_{i,f} per (item, frequency bin): shrunk mean of the residual without it.
    fn step_fq(&mut self) {
        let f = self.f();
        let nf = self.n_fbins;
        let sol: Vec<Vec<f64>> = (0..self.n_items).into_par_iter().map(|i| {
            let qi = &self.q[i * f..(i + 1) * f];
            let mut sum = vec![0.0f64; nf];
            let mut cnt = vec![0.0f64; nf];
            let mut vu = vec![0.0f64; f];
            for t in self.raters.row(i) {
                let u = self.raters.idx[t] as usize;
                let r = self.rat2rated[t] as usize;
                self.v(u, &mut vu);
                let e = self.raters.val[t] as f64 - self.mu - self.bu[u] - self.bi[i] - dot(qi, &vu)
                    - self.nbt(r) - self.udt(r) - self.drt(u, r);
                let k = self.rfbin[r] as usize;
                sum[k] += e;
                cnt[k] += 1.0;
            }
            sum.iter().zip(&cnt).map(|(s, c)| s / (c + self.cfg.reg_fq)).collect()
        }).collect();
        for (i, v) in sol.into_iter().enumerate() { self.bif[i * nf..(i + 1) * nf].copy_from_slice(&v); }
    }

    /// b_ut for any (user, day): 0 if u has no training rating that day.
    fn ud_at(&self, u: usize, day: i32) -> f64 {
        if !self.cfg.use_ud { return 0.0; }
        let (a, b) = (self.ud_off[u], self.ud_off[u + 1]);
        self.ud_day[a..b].binary_search(&(day as i16)).map_or(0.0, |k| self.bud[a + k])
    }

    /// b_ut per (user, day) slot: shrunk mean of the residual without b_ut.
    fn step_ud(&mut self) {
        let f = self.f();
        let sol: Vec<Vec<f64>> = (0..self.n_users()).into_par_iter().map(|u| {
            let (a, b) = (self.ud_off[u], self.ud_off[u + 1]);
            let mut sum = vec![0.0f64; b - a];
            let mut cnt = vec![0.0f64; b - a];
            let mut vu = vec![0.0f64; f];
            self.v(u, &mut vu);
            for t in self.rated.row(u) {
                let i = self.rated.idx[t] as usize;
                let e = self.rated.val[t] as f64 - self.mu - self.bu[u] - self.bi[i]
                    - dot(&self.q[i * f..(i + 1) * f], &vu) - self.nbt(t) - self.fqt(t) - self.drt(u, t);
                let k = self.rslot[t] as usize - a;
                sum[k] += e;
                cnt[k] += 1.0;
            }
            sum.iter().zip(&cnt).map(|(s, c)| s / (c + self.cfg.reg_ud)).collect()
        }).collect();
        for (u, v) in sol.into_iter().enumerate() {
            self.bud[self.ud_off[u]..self.ud_off[u + 1]].copy_from_slice(&v);
        }
    }

    fn loss(&self) -> f64 {
        let f = self.f();
        let sse: f64 = (0..self.n_users()).into_par_iter().map(|u| {
            let mut vu = vec![0.0; f];
            self.v(u, &mut vu);
            let mut s = 0.0;
            for t in self.rated.row(u) {
                let i = self.rated.idx[t] as usize;
                let e = self.rated.val[t] as f64 - self.mu - self.bu[u] - self.bi[i]
                    - dot(&self.q[i * f..(i + 1) * f], &vu) - self.nbt(t) - self.tterms(u, t);
                s += e * e;
            }
            s
        }).sum();
        let sq = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        let c = &self.cfg;
        sse + c.reg_bu * sq(&self.bu) + c.reg_bi * sq(&self.bi) + c.reg_p * sq(&self.p)
            + c.reg_q * sq(&self.q) + c.reg_y * sq(&self.y)
            + c.reg_nw * sq(&self.nw) + c.reg_nc * sq(&self.nc) + c.reg_ud * sq(&self.bud)
            + c.reg_dr * sq(&self.alpha) + c.reg_fq * sq(&self.bif)
    }

    /// Print L after a block; a rise beyond rounding means the step is broken.
    fn check(&mut self, what: &str) {
        if !self.cfg.loss_check { return; }
        let l = self.loss();
        let n = self.rated.idx.len() as f64;
        let flag = if l > self.last_loss * (1.0 + 1e-9) { "  <-- LOSS ROSE" } else { "" };
        crate::teeln!("    {what:<6} L {l:.6e}  (train rms of L {:.5}){flag}", (l / n).sqrt());
        self.last_loss = l;
    }

    /// (p_u, alpha_u, b_u) per user, layout [p | alpha | b_u]; p is left out when
    /// there is no p term or the factors are frozen (then q_i . p_u is part of the
    /// fixed offset), alpha without the dr term.
    fn step_users(&mut self) {
        let f = self.f();
        let up = self.cfg.use_p && !self.frozen;
        let g = if up { f } else { 0 };
        let dr = self.cfg.use_dr as usize;
        let n = g + dr + 1;
        let sol: Vec<Vec<f64>> = (0..self.n_users()).into_par_iter().map(|u| {
            let mut vu = vec![0.0f64; f];
            if up { vu.copy_from_slice(&self.z[u * f..(u + 1) * f]); } else { self.v(u, &mut vu); }
            let mut a = vec![0.0f64; n * n];
            let mut b = vec![0.0f64; n];
            for t in self.rated.row(u) {
                let i = self.rated.idx[t] as usize;
                let qi = &self.q[i * f..(i + 1) * f];
                let y = self.rated.val[t] as f64 - self.mu - self.bi[i] - dot(qi, &vu) - self.nbt(t)
                    - self.udt(t) - self.fqt(t);
                // x = (q_i [g], dev [dr], 1); lower triangle
                let dv = if dr == 1 { self.rdev[t] } else { 0.0 };
                for r in 0..g {
                    b[r] += qi[r] * y;
                    for c in 0..=r { a[r * n + c] += qi[r] * qi[c]; }
                }
                if dr == 1 {
                    b[g] += dv * y;
                    for c in 0..g { a[g * n + c] += dv * qi[c]; }
                    a[g * n + g] += dv * dv;
                }
                let l = n - 1;
                for c in 0..g { a[l * n + c] += qi[c]; }
                if dr == 1 { a[l * n + g] += dv; }
                b[l] += y;
                a[l * n + l] += 1.0;
            }
            for r in 0..g { a[r * n + r] += self.cfg.reg_p; }
            if dr == 1 { a[g * n + g] += self.cfg.reg_dr; }
            a[n * n - 1] += self.cfg.reg_bu;
            ridge_solve(&mut a, &b, n)
        }).collect();
        for (u, s) in sol.into_iter().enumerate() {
            if up { self.p[u * f..(u + 1) * f].copy_from_slice(&s[..f]); }
            if dr == 1 { self.alpha[u] = s[g]; }
            self.bu[u] = s[n - 1];
        }
    }

    /// (q_i, b_i, w_i., c_i.) per item on x_u = (v_u, 1, nb features); the nb
    /// part (slots g+1.. for w, g+1+NB_K.. for c) is sparse. g = F, or 0 when
    /// the factors are frozen (then q_i . v_u is part of the fixed offset).
    fn step_items(&mut self) {
        let f = self.f();
        let g = if self.frozen { 0 } else { f };
        let k = if self.nb_on { self.cfg.nb_k } else { 0 };
        let n = g + 1 + 2 * k;
        let sol: Vec<ItemSol> = (0..self.n_items).into_par_iter().map(|i| {
            let qi = &self.q[i * f..(i + 1) * f];
            let mut a = vec![0.0f64; n * n];
            let mut b = vec![0.0f64; n];
            let mut vu = vec![0.0f64; f];
            let mut x = NbFeat::default();
            let mut sp: Vec<(usize, f64)> = Vec::new();
            let mut feats: Vec<NbFeat> = Vec::new();
            for t in self.raters.row(i) {
                let u = self.raters.idx[t] as usize;
                self.v(u, &mut vu);
                let mut y = self.raters.val[t] as f64 - self.mu - self.bu[u]
                    - if self.rat2rated.is_empty() { 0.0 } else { self.tterms(u, self.rat2rated[t] as usize) };
                if self.frozen { y -= dot(qi, &vu); }
                for r in 0..g {
                    b[r] += vu[r] * y;
                    for c in 0..=r { a[r * n + c] += vu[r] * vu[c]; }
                    a[g * n + r] += vu[r];
                }
                b[g] += y;
                a[g * n + g] += 1.0;
                if k > 0 {
                    self.nb_feat(u, i, &mut x);
                    sp.clear();
                    sp.extend(x.w.iter().map(|&(s, v)| (g + 1 + s, v)));
                    sp.extend(x.c.iter().map(|&(s, v)| (g + 1 + k + s, v)));
                    for (m, &(r, xr)) in sp.iter().enumerate() {
                        b[r] += xr * y;
                        for c in 0..g { a[r * n + c] += xr * vu[c]; }
                        a[r * n + g] += xr;
                        for &(c, xc) in &sp[..=m] { a[r.max(c) * n + r.min(c)] += xr * xc; }
                    }
                    feats.push(std::mem::take(&mut x));
                }
            }
            for r in 0..g { a[r * n + r] += self.cfg.reg_q; }
            a[g * n + g] += self.cfg.reg_bi;
            for s in 0..k {
                a[(g + 1 + s) * n + g + 1 + s] += self.cfg.reg_nw;
                a[(g + 1 + k + s) * n + g + 1 + k + s] += self.cfg.reg_nc;
            }
            let sol = ridge_solve(&mut a, &b, n);
            // nb_ui of the item's ratings under the new w_i., c_i.
            let nbs = self.raters.row(i).zip(&feats).map(|(t, x)| {
                let v = x.w.iter().map(|&(s, v)| sol[g + 1 + s] * v).sum::<f64>()
                    + x.c.iter().map(|&(s, v)| sol[g + 1 + k + s] * v).sum::<f64>();
                (self.rat2rated[t] as usize, v)
            }).collect();
            (sol, nbs)
        }).collect();
        for (i, (s, nbs)) in sol.into_iter().enumerate() {
            self.q[i * f..(i + 1) * f][..g].copy_from_slice(&s[..g]);
            self.bi[i] = s[g];
            if k > 0 {
                self.nw[i * k..(i + 1) * k].copy_from_slice(&s[g + 1..g + 1 + k]);
                self.nc[i * k..(i + 1) * k].copy_from_slice(&s[g + 1 + k..]);
                for (t, v) in nbs { self.nbv[t] = v; }
            }
        }
    }

    /// Per-user pass shared by B and H . D: `None` gives
    /// B_j = sum_{u: j in N(u)} c_u sum_{i in R(u)} e_ui q_i with e_ui the residual
    /// without the y term; `Some(D)` gives H . D.
    fn user_pass(&self, src: Option<&[f64]>) -> Vec<f64> {
        let f = self.f();
        let n_users = self.n_users();
        let n_chunks = (rayon::current_num_threads() * 4).max(1);
        let size = n_users.div_ceil(n_chunks);
        (0..n_chunks).into_par_iter().map(|c| {
            let mut acc = vec![0.0f64; self.n_items * f];
            let mut d = vec![0.0f64; f];
            let mut v = vec![0.0f64; f];
            for u in c * size..((c + 1) * size).min(n_users) {
                let pres = self.present.row(u);
                if pres.is_empty() { continue; }
                v.fill(0.0);
                match src {
                    None => {
                        let pu = &self.p[u * f..(u + 1) * f];
                        for t in self.rated.row(u) {
                            let i = self.rated.idx[t] as usize;
                            let qi = &self.q[i * f..(i + 1) * f];
                            let e = self.rated.val[t] as f64 - self.mu - self.bu[u] - self.bi[i] - dot(qi, pu)
                                - self.nbt(t) - self.tterms(u, t);
                            for k in 0..f { v[k] += e * qi[k]; }
                        }
                    }
                    Some(dm) => {
                        d.fill(0.0);
                        for t in pres.clone() {
                            let j = self.present.idx[t] as usize;
                            for k in 0..f { d[k] += dm[j * f + k]; }
                        }
                        for v in d.iter_mut() { *v *= self.cu[u]; }
                        for t in self.rated.row(u) {
                            let qi = &self.q[self.rated.idx[t] as usize * f..][..f];
                            let s = dot(qi, &d);
                            for k in 0..f { v[k] += s * qi[k]; }
                        }
                    }
                }
                let cu = self.cu[u];
                for t in pres {
                    let j = self.present.idx[t] as usize;
                    for k in 0..f { acc[j * f + k] += cu * v[k]; }
                }
            }
            acc
        }).reduce(|| vec![0.0f64; self.n_items * f], |mut a, b| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        })
    }

    /// (H + REG_Y I) . D
    fn apply(&self, d: &[f64]) -> Vec<f64> {
        let mut h = self.user_pass(Some(d));
        h.iter_mut().zip(d).for_each(|(x, y)| *x += self.cfg.reg_y * y);
        h
    }

    /// Block-Jacobi preconditioner: per item j the F x F block
    /// sum_{u: j in N(u)} c_u^2 A_u + REG_Y I, A_u = sum_{i in R(u)} q_i q_i^T.
    fn precond(&self) -> Vec<nalgebra::Cholesky<f64, nalgebra::Dyn>> {
        let f = self.f();
        let ff = f * f;
        let n_users = self.n_users();
        let n_chunks = rayon::current_num_threads().max(1);
        let size = n_users.div_ceil(n_chunks);
        let blocks = (0..n_chunks).into_par_iter().map(|c| {
            let mut acc = vec![0.0f64; self.n_items * ff];
            let mut au = vec![0.0f64; ff];
            for u in c * size..((c + 1) * size).min(n_users) {
                let pres = self.present.row(u);
                if pres.is_empty() { continue; }
                au.fill(0.0);
                for t in self.rated.row(u) {
                    let qi = &self.q[self.rated.idx[t] as usize * f..][..f];
                    for r in 0..f { for k in 0..=r { au[r * f + k] += qi[r] * qi[k]; } }
                }
                let c2 = self.cu[u] * self.cu[u];
                for t in pres {
                    let blk = &mut acc[self.present.idx[t] as usize * ff..][..ff];
                    for r in 0..f { for k in 0..=r { blk[r * f + k] += c2 * au[r * f + k]; } }
                }
            }
            acc
        }).reduce(|| vec![0.0f64; self.n_items * ff], |mut a, b| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        });
        blocks.par_chunks(ff).map(|b| {
            let mut m = DMatrix::<f64>::zeros(f, f);
            for r in 0..f { for k in 0..=r { m[(r, k)] = b[r * f + k]; m[(k, r)] = b[r * f + k]; } }
            for r in 0..f { m[(r, r)] += self.cfg.reg_y; }
            m.cholesky().expect("preconditioner block not positive definite")
        }).collect()
    }

    fn step_y(&mut self) {
        let f = self.f();
        let b = self.user_pass(None);
        let bnorm = dot(&b, &b).sqrt();
        let pc = self.cfg.precond.then(|| self.precond());
        let solve = |r: &[f64]| -> Vec<f64> {
            match &pc {
                None => r.to_vec(),
                Some(pc) => {
                    let mut z = vec![0.0f64; r.len()];
                    z.par_chunks_mut(f).zip(r.par_chunks(f)).zip(pc.par_iter())
                        .for_each(|((z, r), c)| z.copy_from_slice(c.solve(&DVector::from_column_slice(r)).as_slice()));
                    z
                }
            }
        };
        let mut x = self.y.clone();
        let ax = self.apply(&x);
        let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
        let mut zr = solve(&r);
        let mut d = zr.clone();
        let mut rz = dot(&r, &zr);
        let r0 = dot(&r, &r).sqrt();
        let mut rn = r0;
        let mut it = 0;
        while it < self.cfg.cg_iters && rn > self.cfg.cg_tol * bnorm {
            let ad = self.apply(&d);
            let dad = dot(&d, &ad);
            if dad <= 0.0 { crate::teeln!("    cg: non-positive curvature {dad:.3e}, stop"); break; }
            let a = rz / dad;
            x.iter_mut().zip(&d).for_each(|(x, d)| *x += a * d);
            r.iter_mut().zip(&ad).for_each(|(r, ad)| *r -= a * ad);
            rn = dot(&r, &r).sqrt();
            zr = solve(&r);
            let rz_new = dot(&r, &zr);
            let beta = rz_new / rz;
            d.iter_mut().zip(&zr).for_each(|(d, z)| *d = z + beta * *d);
            rz = rz_new;
            it += 1;
        }
        crate::teeln!("    cg: {it} iters, residual {:.3e} -> {:.3e} (rel. to |B| {:.3e})",
               r0 / bnorm, rn / bnorm, bnorm);
        self.y = x;
        self.update_z();
    }
}

/// S(i): the NB_K items most similar to i (sim > 0), most similar first;
/// u32::MAX pads short lists. The sim matrices are dropped afterwards.
fn neighbors(cfg: &IntegConfig, ds: &str, n_items: usize) -> Vec<u32> {
    let load = |stem: &str| -> ndarray::Array2<f32> {
        ndarray_npy::read_npy(format!("{}/{}.{}.npy", cfg.sim_dir, stem, ds)).unwrap()
    };
    let s0 = load(cfg.nb_stat0);
    let s2 = load(cfg.nb_stat2);
    let k = cfg.nb_k;
    // knn-m's mse sim needs the global mean squared difference
    let (mut tot2, mut totn) = (0.0f64, 0.0f64);
    for i in 0..n_items { for j in 0..n_items { if i != j { tot2 += s2[[i, j]] as f64; totn += s0[[i, j]] as f64; } } }
    let gmse = if totn > 0.0 { tot2 / totn } else { 1.0 };
    let shr = cfg.nb_shr;
    let nbr: Vec<Vec<u32>> = (0..n_items).into_par_iter().map(|i| {
        let mut c: Vec<(f64, u32)> = (0..n_items).filter(|&j| j != i).filter_map(|j| {
            let n = s0[[i, j]] as f64;
            if n < 2.0 { return None; }
            let sim = if cfg.nb_mse {
                (n + shr) / (s2[[i, j]] as f64 + shr * gmse)
            } else {
                let den = (s2[[i, i]] as f64 * s2[[j, j]] as f64).sqrt();
                if den <= 0.0 { return None; }
                (s2[[i, j]] as f64 / den).abs() * n / (n + shr)
            };
            (sim > 0.0).then_some((sim, j as u32))
        }).collect();
        let take = k.min(c.len());
        if c.len() > take { c.select_nth_unstable_by(take, |a, b| b.0.total_cmp(&a.0)); c.truncate(take); }
        c.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut v: Vec<u32> = c.into_iter().map(|x| x.1).collect();
        v.resize(k, u32::MAX);
        v
    }).collect();
    let filled = nbr.iter().map(|v| v.iter().filter(|&&j| j != u32::MAX).count()).sum::<usize>();
    crate::teeln!("nb: S(i) from {} / {} ({}), k {k}, avg filled {:.1}", cfg.nb_stat0, cfg.nb_stat2,
           if cfg.nb_mse { "mse" } else { "cos" }, filled as f64 / n_items as f64);
    nbr.concat()
}

/// The fixed baseline of the nb term: biases alone (shrinkage 10 / 25), by
/// alternating closed-form updates.
fn baseline(rated: &Csr, raters: &Csr, mu: f64) -> (Vec<f64>, Vec<f64>) {
    let (n_users, n_items) = (rated.off.len() - 1, raters.off.len() - 1);
    let mut bu = vec![0.0f64; n_users];
    let mut bi = vec![0.0f64; n_items];
    for _ in 0..5 {
        bi = (0..n_items).into_par_iter().map(|i| {
            let r = raters.row(i);
            r.clone().map(|t| raters.val[t] as f64 - mu - bu[raters.idx[t] as usize]).sum::<f64>()
                / (r.len() as f64 + 25.0)
        }).collect();
        bu = (0..n_users).into_par_iter().map(|u| {
            let r = rated.row(u);
            r.clone().map(|t| rated.val[t] as f64 - mu - bi[rated.idx[t] as usize]).sum::<f64>()
                / (r.len() as f64 + 10.0)
        }).collect();
    }
    (bu, bi)
}

impl Regressor for IntegModel {
    type Config = IntegConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let f = cfg.n_feat;
        let (n_users, n_items) = (tr.n_users, tr.n_items);
        let us = tr.user_idxs.as_slice().unwrap();
        let is = tr.item_idxs.as_slice().unwrap();
        let rs = tr.residuals.as_slice().unwrap();
        let rated = group(n_users, us, is, Some(rs), cfg.sort);
        let raters = group(n_items, is, us, Some(rs), cfg.sort);
        let present = if cfg.implicit {
            let pu = pr.user_idxs.as_slice().unwrap();
            let pi = pr.item_idxs.as_slice().unwrap();
            let users: Vec<i32> = us.iter().chain(pu).copied().collect();
            let items: Vec<i32> = is.iter().chain(pi).copied().collect();
            group(n_users, &users, &items, None, cfg.sort)
        } else {
            Csr { off: rated.off.clone(), idx: rated.idx.clone(), val: Vec::new() }
        };
        let cu = (0..n_users).map(|u| {
            let n = present.row(u).len();
            if n > 0 { (n as f64).powf(-cfg.alpha) } else { 0.0 }
        }).collect();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let normal = Normal::new(0.0, cfg.sigma).unwrap();
        let mut init = |n: usize, on: bool| -> Vec<f64> {
            if on { (0..n).map(|_| normal.sample(&mut rng)).collect() } else { vec![0.0; n] }
        };
        // without p, q starts at 0 and is fitted on the random y (as NSVD1 did)
        let (q, p, y) = if cfg.use_p {
            let q = init(n_items * f, true);
            let p = init(n_users * f, true);
            (q, p, init(n_items * f, cfg.use_y))
        } else {
            let y = init(n_items * f, cfg.use_y);
            (vec![0.0; n_items * f], vec![0.0; n_users * f], y)
        };
        let on = [("p", cfg.use_p), ("y", cfg.use_y), ("nb", cfg.use_nb), ("ud", cfg.use_ud), ("dr", cfg.use_dr), ("fq", cfg.use_fq)];
        let terms: Vec<&str> = on.iter().filter(|x| x.1).map(|x| x.0).collect();
        crate::teeln!("integ: {} users, {} items, {} ratings, |N(u)| total {}, F {f}, terms {}",
               n_users, n_items, rated.idx.len(), present.idx.len(), terms.join("+"));
        let mu = calc_gbias(tr) as f64;
        let k = if cfg.use_nb { cfg.nb_k } else { 0 };
        let (nbr, bbu, bbi) = if cfg.use_nb {
            let nbr = neighbors(&cfg, tr.name.as_str(), n_items);
            let (bbu, bbi) = baseline(&rated, &raters, mu);
            (nbr, bbu, bbi)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let rat2rated = if cfg.use_nb || cfg.use_ud || cfg.use_dr || cfg.use_fq {
            // raters position -> rated position: both are sorted by (row, col)
            let mut rat2rated = vec![0u32; raters.idx.len()];
            let mut pos = rated.off.clone();
            for i in 0..n_items {
                for t in raters.row(i) {
                    let u = raters.idx[t] as usize;
                    rat2rated[t] = pos[u] as u32;
                    pos[u] += 1;
                }
            }
            assert!((0..n_items).into_par_iter().all(|i| raters.row(i).all(|t| {
                let r = rat2rated[t] as usize;
                rated.idx[r] as usize == i && rated.val[r] == raters.val[t]
            })), "raters -> rated map is inconsistent");
            rat2rated
        } else {
            Vec::new()
        };
        // days of the training ratings in `rated` order (same sort as `rated`)
        let days = if cfg.use_ud || cfg.use_dr || cfg.use_fq {
            let ds: Vec<f32> = tr.dates.iter().map(|&d| d as f32).collect();
            let days = group(n_users, us, is, Some(&ds), cfg.sort);
            assert!(days.idx == rated.idx, "date rows do not line up with rating rows");
            days
        } else {
            Csr { off: Vec::new(), idx: Vec::new(), val: Vec::new() }
        };
        let (ud_off, ud_day, rslot) = if cfg.use_ud {
            let mut ud_off = vec![0usize; n_users + 1];
            let mut ud_day: Vec<i16> = Vec::new();
            let mut rslot = vec![0u32; rated.idx.len()];
            for u in 0..n_users {
                let r = days.row(u);
                let mut dd: Vec<i16> = days.val[r.clone()].iter().map(|&d| d as i16).collect();
                dd.sort_unstable();
                dd.dedup();
                let base = ud_day.len();
                for t in r { rslot[t] = (base + dd.binary_search(&(days.val[t] as i16)).unwrap()) as u32; }
                ud_day.extend(dd);
                ud_off[u + 1] = ud_day.len();
            }
            crate::teeln!("ud: {} (user, day) slots", ud_day.len());
            (ud_off, ud_day, rslot)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let (tbar, rdev) = if cfg.use_dr {
            let tbar: Vec<f64> = (0..n_users).map(|u| {
                let r = days.row(u);
                if r.is_empty() { 0.0 } else { days.val[r.clone()].iter().map(|&d| d as f64).sum::<f64>() / r.len() as f64 }
            }).collect();
            let mut rdev = vec![0.0f64; rated.idx.len()];
            for (u, &tb) in tbar.iter().enumerate() {
                for t in days.row(u) { rdev[t] = Self::dev(cfg.dr_beta, days.val[t] as f64, tb); }
            }
            (tbar, rdev)
        } else {
            (Vec::new(), Vec::new())
        };
        let (dc_off, dc_day, dc_cnt, rfbin, n_fbins) = if cfg.use_fq {
            // ratings per (user, day), training and prediction set together
            let pu = pr.user_idxs.as_slice().unwrap();
            let pd = pr.dates.as_slice().unwrap();
            let mut per: Vec<Vec<i16>> = vec![Vec::new(); n_users];
            for (u, pe) in per.iter_mut().enumerate() { pe.extend(days.val[days.row(u)].iter().map(|&d| d as i16)); }
            for t in 0..pu.len() { per[pu[t] as usize].push(pd[t]); }
            let mut dc_off = vec![0usize; n_users + 1];
            let (mut dc_day, mut dc_cnt) = (Vec::new(), Vec::new());
            for u in 0..n_users {
                per[u].sort_unstable();
                for &d in &per[u] {
                    if dc_day.len() > dc_off[u] && *dc_day.last().unwrap() == d { *dc_cnt.last_mut().unwrap() += 1; }
                    else { dc_day.push(d); dc_cnt.push(1u32); }
                }
                dc_off[u + 1] = dc_day.len();
            }
            let maxc = dc_cnt.iter().copied().max().unwrap_or(1);
            let n_fbins = ((maxc as f64).ln() / cfg.fq_base.ln()).floor() as usize + 1;
            let mut rfbin = vec![0u8; rated.idx.len()];
            for u in 0..n_users {
                let (a, b) = (dc_off[u], dc_off[u + 1]);
                for t in days.row(u) {
                    let k = dc_day[a..b].binary_search(&(days.val[t] as i16)).unwrap();
                    rfbin[t] = Self::fbin(cfg.fq_base, n_fbins, dc_cnt[a + k]);
                }
            }
            let mut hist = vec![0usize; n_fbins];
            for &b in &rfbin { hist[b as usize] += 1; }
            crate::teeln!("fq: max F_ut {maxc}, {n_fbins} bins (base {}), training ratings per bin {hist:?}", cfg.fq_base);
            (dc_off, dc_day, dc_cnt, rfbin, n_fbins)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 1)
        };
        let n_ratings = rated.idx.len();
        let mut m = Self {
            cfg, n_items, mu,
            bu: vec![0.0; n_users], bi: vec![0.0; n_items],
            p, q, y, z: vec![0.0; n_users * f], cu,
            rated, raters, present,
            nbr, bbu, bbi, nw: vec![0.0; n_items * k], nc: vec![0.0; n_items * k],
            nbv: vec![0.0; if k > 0 { n_ratings } else { 0 }], nb_on: false, frozen: false, rat2rated,
            bud: vec![0.0; ud_day.len()], ud_off, ud_day, rslot,
            alpha: vec![0.0; if cfg.use_dr { n_users } else { 0 }], tbar, rdev,
            bif: vec![0.0; if cfg.use_fq { n_items * n_fbins } else { 0 }],
            dc_off, dc_day, dc_cnt, rfbin, n_fbins,
            last_loss: f64::INFINITY,
        };
        m.update_z();
        m
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        crate::teeln!("  epoch {epoch}:");
        // before NB_START, w and c stay 0 and the item step leaves them out
        self.nb_on = self.cfg.use_nb && epoch >= self.cfg.nb_start;
        self.frozen = self.nb_on && self.cfg.nb_freeze;
        self.step_users();
        self.check(if self.cfg.use_p && !self.frozen { "p,b_u" } else { "b_u" });
        if self.cfg.use_ud {
            self.step_ud();
            self.check("b_ut");
        }
        if self.cfg.use_fq {
            self.step_fq();
            self.check("b_if");
        }
        self.step_items();
        self.check("q,b_i");
        if self.cfg.use_y && !self.frozen {
            self.step_y();
            self.check("Y");
        }
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let f = self.f();
        let mut vu = vec![0.0; f];
        self.v(u, &mut vu);
        let mut s = self.mu + self.bu[u] + self.ud_at(u, day) + self.bi[i] + self.fq_at(u, i, day)
            + dot(&self.q[i * f..(i + 1) * f], &vu);
        if self.cfg.use_dr { s += self.alpha[u] * Self::dev(self.cfg.dr_beta, day as f64, self.tbar[u]); }
        if self.cfg.use_nb {
            let mut x = NbFeat::default();
            self.nb_feat(u, i, &mut x);
            s += self.nb_value(i, &x);
        }
        s as f32
    }
}
