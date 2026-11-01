// BK4 integrated model of Pragmatic Theory.
// Ref: Piotte & Chabbert 2009, "The Pragmatic Theory solution to the Netflix Grand
// Prize", Section 3.3, equations 21–32:
//
// z(u,m,t) = μ + bu(u,t) + bm(m,t)·s(u,t)
//          + Σ_k q_k(m,t)·( p_k(u,t) + s(u,t)·|N(u)|^-½ Σ_{j∈N(u)} y_k(j,t_j) )
//          + Σ_{j∈Rᵏ(m;u)} ww(m,j)·(r(u,j) − bl₂(u,j))·w[m,j]
//          + Σ_{j∈Nᵏ(m;u)} wc(m,j)·s(u,t)·c[m,j]
//
// bu(u,t) = bu0(u) + Σ_i bu1_i(u)·[buf_i(f36) + but_i(t_u)] + bu2(u,day)    (nbu terms)
// bm(m,t) = bm0(m) + Σ_i bm1_i(m)·[bmf_i(f36) + bmt_i(t_m)]                 (nbm terms)
// s(u,t)  = 1 + s0(u) + Σ_i s1_i(u)·[sf_i(f36) + st_i(t_u)] + s2(u,day)     (nsu terms)
// q(m,t)  = q0(m) + q1(m)·[σ_q(f) − σ̄_q(m) + qt(m,t36) + qf(m,f36)]
// p(u,t)  = p0(u) + p1(u)·[σ_p(t_u) − σ̄_p(u) + p2(u,day)]
// y(j,t)  = y0(j) + y1(j)·[σ_y(f) − σ̄_y(j) + yt(j,t36) + yf(j,f36)]   (t, f of the rating of j)
// ww(m,j) = d(m,j) / √Σ_{l∈Rᵏ} d(m,l)²,  d = 1/(1 + βw·|Δt|)   (wc likewise over Nᵏ)
//
// f = ratings of the user on that day (train ∪ probe/qual), f36 = 36 log bins over
// 1..5446, t36 = 36 uniform date bins, t_u / t_m = days from the user's / movie's
// median rating date (train ∪ probe/qual dates). σ̄ = mean of the sigmoid over the
// user's / movie's training ratings. The garbled PDF leaves the sigmoid argument
// ambiguous: SIGF=0 σ(α·x + β), 1 σ(α·(x + β)), 2 σ(α·(x − β)).
// bl₂ is the paper's baseline 2 (bm·(1+s) + bu, 3 ALS sweeps) with plain shrinkage
// (BL_M, BL_U, BL_S) in place of its regulariser, which the PDF does not render.
// Learning rates γ1..γ27 and weight decays λ1..λ27 follow the paper's table order,
// defaults bk4-d50's; LR_SCALE scales all rates, DECAY is Δγ per epoch.
// ENV=1 wraps z in the σ4-type envelope with its inflection at bl₂ (paper eq 5).
//
// Knobs (env, see `config_from_env`): F, EPOCHS, SEED, NBU, NBM, NSU, NBR (k),
// SORTED, DECAY, TIME, LR_<group> / REG_<group> (bu0 .. yf), LR_SCALE, REG_SCALE,
// AQ/BQ, AP/BP, AY/BY, SIGF, ALPHA_W, BETA_W, BL_M/BL_U/BL_S, ENV, SIGMA.
// Training is single-threaded SGD.
//
//   F=200 NBR=445 SIGF=2 TIME=1 LR_SCALE=1.5 EPOCHS=28 cargo run --release --bin s3 -- misc s3-bk4-f200x -p pipeline-3.toml

use crate::{
    Dataset, MaskedDataset, Regressor, calc_gbias, calc_user_offsets, get_users,
    lab::ev,
    rand_array2,
    tx::SparseUD,
};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use rand::{SeedableRng, rngs::StdRng};

const NB: usize = 36;      // f36 / t36 bins
const FMAX: f32 = 5446.0;  // largest per-day rating count in the Netflix data

// γ / λ index of each parameter group (the paper's numbering).
const BU0: usize = 1; const BU1: usize = 2; const BUT: usize = 3; const BUF: usize = 4;
const BU2: usize = 5; const BM0: usize = 6; const BM1: usize = 7; const BMT: usize = 8;
const BMF: usize = 9; const S0: usize = 10; const S1: usize = 11; const ST: usize = 12;
const SF: usize = 13; const S2: usize = 14; const Q0: usize = 15; const P0: usize = 16;
const Y0: usize = 17; const Q1: usize = 18; const P1: usize = 19; const Y1: usize = 20;
const W: usize = 21; const C: usize = 22; const P2: usize = 23; const QT: usize = 24;
const QF: usize = 25; const YT: usize = 26; const YF: usize = 27;

const NAMES: [&str; 28] = ["", "bu0", "bu1", "but", "buf", "bu2", "bm0", "bm1", "bmt", "bmf",
    "s0", "s1", "st", "sf", "s2", "q0", "p0", "y0", "q1", "p1", "y1", "w", "c", "p2", "qt", "qf",
    "yt", "yf"];

// bk4-d50 (γ1..γ27, λ1..λ27); the f200z4 rates for p2, qt, qf, yt, yf, which d50 leaves at 0.
const LR_D50: [f32; 28] = [0.0,
    0.000977427, 0.00246524, 0.00247907, 0.00024348, 0.00255845, 0.000249488, 0.000801197,
    0.000488952, 0.000283411, 0.000756619, 0.00215928, 0.000134043, 0.00311735, 0.00111064,
    0.00059987, 0.0122602, 0.0013921, 0.00062067, 0.00984111, 0.00216959, 8.09761e-5,
    0.000112904, 0.0, 0.0, 0.0, 0.0, 0.0];
const REG_D50: [f32; 28] = [0.0,
    0.00529725, 0.000131135, 0.00383613, 0.00791377, 0.000685716, 0.00398966, 0.00236947,
    0.0020634, 0.0016331, 0.000496923, 0.000184172, 0.000802785, 0.00111961, 0.000696543,
    0.0345866, 0.0254905, 0.0053402, 0.0034172, 0.092418, 0.00560977, 0.00864227, 0.00819148,
    0.0, 0.0, 0.0, 0.0, 0.0];
const LR_F200Z4_TIME: [f32; 5] = [0.01319, 1.48766e-5, 3.86391e-6, 0.0011758, 0.000688992];
const REG_F200Z4_TIME: [f32; 5] = [9.79141e-6, 0.0067143, 0.00867617, 0.000814275, 0.000874146];

#[derive(Clone, Copy, Debug)]
pub struct Bk4Config {
    pub n_feat: usize,
    pub n_epochs: usize,
    pub seed: u64,
    pub nbu: usize,
    pub nbm: usize,
    pub nsu: usize,
    pub k_nbr: usize,
    pub sorted: bool,
    pub lr_decay: f32,
    pub lr: [f32; 28],
    pub reg: [f32; 28],
    // sigmoid constants (α, β) for q (frequency), p (t_u), y (frequency)
    pub sq: (f32, f32),
    pub sp: (f32, f32),
    pub sy: (f32, f32),
    pub sig_form: u8,
    pub alpha_w: f32,   // shrinkage of the neighbour selection score
    pub beta_w: f32,    // time decay of the neighbour weights
    pub bl_m: f32, pub bl_u: f32, pub bl_s: f32,
    pub envelope: bool,
    pub sigma_init: f32,
}

#[inline]
fn sigm(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[inline]
fn sig_arg(form: u8, (a, b): (f32, f32), x: f32) -> f32 {
    match form { 0 => a * x + b, 1 => a * (x + b), _ => a * (x - b) }
}

#[inline]
fn fbin(f: u32) -> usize {
    let b = ((f.max(1) as f32).ln() / FMAX.ln() * NB as f32) as usize;
    b.min(NB - 1)
}

/// Shifted-inflection envelope of the paper's eq 5: maps z into (1, 5) with
/// value z0, slope 1 and curvature 0 at z0. Returns (value, derivative).
#[inline]
fn envelope(z: f32, z0: f32) -> (f32, f32) {
    let z0 = z0.clamp(1.2, 4.8);
    // below z0: 1 + 2a/(1+e), above: 2·z0 − 5 + 2a/(1+e), e = exp(−2(z − z0)/a)
    let (a, c) = if z < z0 { (z0 - 1.0, 1.0) } else { (5.0 - z0, 2.0 * z0 - 5.0) };
    let e = (-2.0 * (z - z0) / a).exp();
    (c + 2.0 * a / (1.0 + e), 4.0 * e / ((1.0 + e) * (1.0 + e)))
}

/// One entry of N(u): an item the user rated (train) or is known to have rated
/// (probe/qual, rating unknown), with what y(j,t) and the neighbour term need.
#[derive(Clone, Copy)]
struct NEnt {
    item: u16,
    day: i16,
    f36: u8,
    t36: u8,
    wy: f32,    // σ_y(f) − σ̄_y(j)
    rmb: f32,   // r − bl₂ (NaN if the rating is unknown)
}

pub struct Bk4Model {
    cfg: Bk4Config,
    mu: f32,
    n_feat: usize,
    // biases
    bu0: Vec<f32>, bu1: Vec<f32>, but: Vec<f32>, buf: Vec<f32>, bu2: Vec<f32>,
    bm0: Vec<f32>, bm1: Vec<f32>, bmt: Vec<f32>, bmf: Vec<f32>,
    s0: Vec<f32>, s1: Vec<f32>, st: Vec<f32>, sf: Vec<f32>, s2: Vec<f32>,
    // factors
    q0: Array2<f32>, q1: Array2<f32>, p0: Array2<f32>, p1: Array2<f32>,
    y0: Array2<f32>, y1: Array2<f32>,
    p2: Vec<f32>, qt: Vec<f32>, qf: Vec<f32>, yt: Vec<f32>, yf: Vec<f32>,
    ycache: Array2<f32>,
    // neighbourhood
    nbrs: Vec<Vec<u16>>,
    w: Vec<f32>, c: Vec<f32>,
    // structure
    nu: Vec<Vec<NEnt>>,          // N(u), sorted by item
    ud: SparseUD,
    med_u: Vec<i32>, med_m: Vec<i32>,
    sbar_q: Vec<f32>, sbar_p: Vec<f32>,
    day0: i32, day1: i32, toff: i32,
    bl_bu: Vec<f32>, bl_bm: Vec<f32>, bl_s: Vec<f32>,
    user_offsets: Array1<usize>,
    order: Vec<u32>,             // training rating order per user (date-sorted if cfg.sorted)
    lr_scale: f32,
}

/// Per-rating time features.
struct Tf { ud: usize, freq: u32, f36: usize, t36: usize, tu: usize, tm: usize, du: f32 }

impl Bk4Model {
    #[inline]
    fn tfeat(&self, u: usize, m: usize, day: i16) -> Tf {
        let ud = self.ud.index(u, day).expect("(user, day) missing from SparseUD");
        let freq = self.ud.day_cnts[ud];
        let d = day as i32;
        let t36 = (((d - self.day0) as i64 * NB as i64) / ((self.day1 - self.day0 + 1) as i64))
            .clamp(0, NB as i64 - 1) as usize;
        let du = d - self.med_u[u];
        let dm = d - self.med_m[m];
        Tf { ud, freq, f36: fbin(freq), t36,
             tu: (du + self.toff) as usize, tm: (dm + self.toff) as usize, du: du as f32 }
    }

    #[inline]
    fn bl2(&self, u: usize, m: usize) -> f32 {
        self.mu + self.bl_bm[m] * (1.0 + self.bl_s[u]) + self.bl_bu[u]
    }

    fn rebuild_ycache(&mut self) {
        let nf = self.n_feat;
        for u in 0..self.nu.len() {
            let mut acc = vec![0.0f32; nf];
            let n = self.nu[u].len();
            if n == 0 { continue; }
            for e in &self.nu[u] {
                let j = e.item as usize;
                let wy = e.wy + self.yt[j * NB + e.t36 as usize] + self.yf[j * NB + e.f36 as usize];
                let (a, b) = (self.y0.row(j), self.y1.row(j));
                for k in 0..nf { acc[k] += a[k] + b[k] * wy; }
            }
            let norm = 1.0 / (n as f32).sqrt();
            let mut row = self.ycache.row_mut(u);
            for k in 0..nf { row[k] = acc[k] * norm; }
        }
    }

    /// Neighbour term for prediction: (Σ ww·rmb·w, Σ wc·c).
    fn nbr_terms(&self, u: usize, m: usize, day: i16) -> (f32, f32) {
        let k = self.cfg.k_nbr;
        if k == 0 { return (0.0, 0.0); }
        let nu = &self.nu[u];
        let (mut sw, mut sw2, mut sc, mut sc2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (rank, &j) in self.nbrs[m].iter().enumerate() {
            if let Ok(pos) = nu.binary_search_by_key(&j, |e| e.item) {
                let e = &nu[pos];
                let d = 1.0 / (1.0 + self.cfg.beta_w * (day as f32 - e.day as f32).abs());
                sc += d * self.c[m * k + rank]; sc2 += d * d;
                if !e.rmb.is_nan() { sw += d * e.rmb * self.w[m * k + rank]; sw2 += d * d; }
            }
        }
        let nw = if sw2 > 0.0 { sw / sw2.sqrt() } else { 0.0 };
        let nc = if sc2 > 0.0 { sc / sc2.sqrt() } else { 0.0 };
        (nw, nc)
    }
}

fn median(v: &mut [i16]) -> i32 {
    if v.is_empty() { return 0; }
    let mid = v.len() / 2;
    *v.select_nth_unstable(mid).1 as i32
}

fn top_neighbors(dataset: &str, k: usize, alpha: f32, n_items: usize) -> Vec<Vec<u16>> {
    crate::teeln!("bk4: top-{k} neighbours from sim/rtg_{{prod,supp}}.{dataset}.npy (shrink {alpha})");
    let prod: Array2<f32> = read_npy(format!("sim/rtg_prod.{dataset}.npy")).unwrap();
    let supp: Array2<f32> = read_npy(format!("sim/rtg_supp.{dataset}.npy")).unwrap();
    let norms: Vec<f32> = (0..n_items).map(|m| prod[[m, m]].max(0.0).sqrt()).collect();
    let mut out = vec![Vec::new(); n_items];
    let mut buf: Vec<(f32, u16)> = Vec::with_capacity(n_items);
    for m in 0..n_items {
        if norms[m] == 0.0 { continue; }
        buf.clear();
        for j in 0..n_items {
            let n = supp[[m, j]];
            if j == m || norms[j] == 0.0 || n <= 0.0 { continue; }
            let s = prod[[m, j]] / (norms[m] * norms[j]) * n / (n + alpha);
            if s > 0.0 { buf.push((s, j as u16)); }
        }
        let take = k.min(buf.len());
        if buf.len() > take { buf.select_nth_unstable_by(take, |a, b| b.0.total_cmp(&a.0)); }
        buf[..take].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        out[m] = buf[..take].iter().map(|&(_, j)| j).collect();
    }
    out
}

impl Regressor for Bk4Model {
    type Config = Bk4Config;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Bk4Config) -> Self {
        let (nu_n, ni) = (tr.n_users, tr.n_items);
        let nf = cfg.n_feat;
        let mu = calc_gbias(tr);
        let user_offsets = calc_user_offsets(tr);
        let ud = SparseUD::new(tr, pr);

        // date range and medians over train ∪ pr
        let (mut day0, mut day1) = (i32::MAX, i32::MIN);
        let mut du: Vec<Vec<i16>> = vec![Vec::new(); nu_n];
        let mut dm: Vec<Vec<i16>> = vec![Vec::new(); ni];
        for t in 0..tr.n_ratings {
            let d = tr.dates[t];
            day0 = day0.min(d as i32); day1 = day1.max(d as i32);
            du[tr.user_idxs[t] as usize].push(d);
            dm[tr.item_idxs[t] as usize].push(d);
        }
        for t in 0..pr.n_ratings {
            let d = pr.dates[t];
            du[pr.user_idxs[t] as usize].push(d);
            dm[pr.item_idxs[t] as usize].push(d);
        }
        let med_u: Vec<i32> = du.iter_mut().map(|v| median(v)).collect();
        let med_m: Vec<i32> = dm.iter_mut().map(|v| median(v)).collect();
        drop(du); drop(dm);
        let toff = 3000;   // t_u, t_m ∈ (−3000, 3000) days
        let tsz = 2 * toff as usize + 1;

        // baseline 2: bl = μ + bm·(1 + s) + bu, 3 ALS sweeps starting with the movies
        let (mut bl_bu, mut bl_bm, mut bl_s) = (vec![0.0f32; nu_n], vec![0.0f32; ni], vec![0.0f32; nu_n]);
        for _ in 0..3 {
            let (mut num, mut den) = (vec![0.0f64; ni], vec![0.0f64; ni]);
            for t in 0..tr.n_ratings {
                let (u, m) = (tr.user_idxs[t] as usize, tr.item_idxs[t] as usize);
                let sc = 1.0 + bl_s[u] as f64;
                num[m] += (tr.residuals[t] - mu - bl_bu[u]) as f64 * sc;
                den[m] += sc * sc;
            }
            for m in 0..ni { bl_bm[m] = (num[m] / (den[m] + cfg.bl_m as f64)) as f32; }
            for u in 0..nu_n {
                let (s, e) = (user_offsets[u], user_offsets[u + 1]);
                // [n + Pb, Σb; Σb, Σb² + Ps]·[bu; s] = [Σr'; Σr'b], r' = r − μ − bm
                let (mut n, mut sb, mut sbb, mut sr, mut srb) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
                for t in s..e {
                    let b = bl_bm[tr.item_idxs[t] as usize] as f64;
                    let r = (tr.residuals[t] - mu) as f64 - b;
                    n += 1.0; sb += b; sbb += b * b; sr += r; srb += r * b;
                }
                let (a11, a12, a22) = (n + cfg.bl_u as f64, sb, sbb + cfg.bl_s as f64);
                let det = a11 * a22 - a12 * a12;
                if det.abs() > 1e-12 {
                    bl_bu[u] = ((sr * a22 - a12 * srb) / det) as f32;
                    bl_s[u] = ((a11 * srb - a12 * sr) / det) as f32;
                }
            }
        }

        // sigmoid means
        let sig_q = |f: u32| sigm(sig_arg(cfg.sig_form, cfg.sq, f as f32));
        let sig_y = |f: u32| sigm(sig_arg(cfg.sig_form, cfg.sy, f as f32));
        let sig_p = |d: f32| sigm(sig_arg(cfg.sig_form, cfg.sp, d));
        let (mut sq_sum, mut sy_sum, mut cnt_m) = (vec![0.0f64; ni], vec![0.0f64; ni], vec![0u32; ni]);
        let (mut sp_sum, mut cnt_u) = (vec![0.0f64; nu_n], vec![0u32; nu_n]);
        for t in 0..tr.n_ratings {
            let (u, m) = (tr.user_idxs[t] as usize, tr.item_idxs[t] as usize);
            let f = ud.day_cnts[ud.index(u, tr.dates[t]).unwrap()];
            sq_sum[m] += sig_q(f) as f64; sy_sum[m] += sig_y(f) as f64; cnt_m[m] += 1;
            sp_sum[u] += sig_p((tr.dates[t] as i32 - med_u[u]) as f32) as f64; cnt_u[u] += 1;
        }
        let mean = |s: &[f64], c: &[u32]| -> Vec<f32> {
            s.iter().zip(c).map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 }).collect()
        };
        let sbar_q = mean(&sq_sum, &cnt_m);
        let sbar_y = mean(&sy_sum, &cnt_m);
        let sbar_p = mean(&sp_sum, &cnt_u);

        // N(u)
        let t36 = |d: i16| ((((d as i32 - day0) as i64) * NB as i64 / ((day1 - day0 + 1) as i64))
            .clamp(0, NB as i64 - 1)) as u8;
        let mut nu: Vec<Vec<NEnt>> = vec![Vec::new(); nu_n];
        for t in 0..tr.n_ratings {
            let (u, m, d) = (tr.user_idxs[t] as usize, tr.item_idxs[t] as usize, tr.dates[t]);
            let f = ud.day_cnts[ud.index(u, d).unwrap()];
            let bl = mu + bl_bm[m] * (1.0 + bl_s[u]) + bl_bu[u];
            nu[u].push(NEnt { item: m as u16, day: d, f36: fbin(f) as u8, t36: t36(d),
                              wy: sig_y(f) - sbar_y[m], rmb: tr.residuals[t] - bl });
        }
        for t in 0..pr.n_ratings {
            let (u, m, d) = (pr.user_idxs[t] as usize, pr.item_idxs[t] as usize, pr.dates[t]);
            let f = ud.day_cnts[ud.index(u, d).unwrap()];
            nu[u].push(NEnt { item: m as u16, day: d, f36: fbin(f) as u8, t36: t36(d),
                              wy: sig_y(f) - sbar_y[m], rmb: f32::NAN });
        }
        for v in nu.iter_mut() {
            v.sort_unstable_by_key(|e| (e.item, e.rmb.is_nan()));
            v.dedup_by_key(|e| e.item);
        }

        // training order within each user
        let mut order: Vec<u32> = (0..tr.n_ratings as u32).collect();
        if cfg.sorted {
            for u in 0..nu_n {
                let (s, e) = (user_offsets[u], user_offsets[u + 1]);
                order[s..e].sort_by_key(|&t| (tr.dates[t as usize], t));
            }
        }

        let k = cfg.k_nbr;
        let nbrs = if k > 0 { top_neighbors(&tr.name, k, cfg.alpha_w, ni) } else { vec![Vec::new(); ni] };

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let p0 = rand_array2(nu_n, nf, &mut rng, cfg.sigma_init);
        let q0 = rand_array2(ni, nf, &mut rng, cfg.sigma_init);
        let n_ud = ud.n_total();
        crate::teeln!("bk4: {} users, {} items, {} user-days, days {day0}..{day1}, baseline2 done", nu_n, ni, n_ud);

        let mut model = Self {
            cfg, mu, n_feat: nf,
            bu0: vec![0.0; nu_n], bu1: vec![0.0; nu_n * cfg.nbu], but: vec![0.0; tsz * cfg.nbu],
            buf: vec![0.0; NB * cfg.nbu], bu2: vec![0.0; n_ud],
            bm0: vec![0.0; ni], bm1: vec![0.0; ni * cfg.nbm], bmt: vec![0.0; tsz * cfg.nbm],
            bmf: vec![0.0; NB * cfg.nbm],
            s0: vec![0.0; nu_n], s1: vec![0.0; nu_n * cfg.nsu], st: vec![0.0; tsz * cfg.nsu],
            sf: vec![0.0; NB * cfg.nsu], s2: vec![0.0; n_ud],
            q0, q1: Array2::zeros((ni, nf)), p0, p1: Array2::zeros((nu_n, nf)),
            y0: Array2::zeros((ni, nf)), y1: Array2::zeros((ni, nf)),
            p2: vec![0.0; n_ud], qt: vec![0.0; ni * NB], qf: vec![0.0; ni * NB],
            yt: vec![0.0; ni * NB], yf: vec![0.0; ni * NB],
            ycache: Array2::zeros((nu_n, nf)),
            w: vec![0.0; ni * k], c: vec![0.0; ni * k], nbrs,
            nu, ud, med_u, med_m, sbar_q, sbar_p, day0, day1, toff,
            bl_bu, bl_bm, bl_s, user_offsets, order, lr_scale: 1.0,
        };
        model.rebuild_ycache();
        model
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn predict(&self, u: usize, m: usize, day: i32) -> f32 {
        let cfg = &self.cfg;
        let tf = self.tfeat(u, m, day as i16);
        let mut bu = self.bu0[u] + self.bu2[tf.ud];
        for i in 0..cfg.nbu {
            bu += self.bu1[u * cfg.nbu + i] * (self.buf[tf.f36 * cfg.nbu + i] + self.but[tf.tu * cfg.nbu + i]);
        }
        let mut bm = self.bm0[m];
        for i in 0..cfg.nbm {
            bm += self.bm1[m * cfg.nbm + i] * (self.bmf[tf.f36 * cfg.nbm + i] + self.bmt[tf.tm * cfg.nbm + i]);
        }
        let mut s = 1.0 + self.s0[u] + self.s2[tf.ud];
        for i in 0..cfg.nsu {
            s += self.s1[u * cfg.nsu + i] * (self.sf[tf.f36 * cfg.nsu + i] + self.st[tf.tu * cfg.nsu + i]);
        }
        let wq = sigm(sig_arg(cfg.sig_form, cfg.sq, tf.freq as f32)) - self.sbar_q[m]
            + self.qt[m * NB + tf.t36] + self.qf[m * NB + tf.f36];
        let wp = sigm(sig_arg(cfg.sig_form, cfg.sp, tf.du)) - self.sbar_p[u] + self.p2[tf.ud];
        let (q0, q1, p0, p1, y) = (self.q0.row(m), self.q1.row(m), self.p0.row(u), self.p1.row(u), self.ycache.row(u));
        let mut dot = 0.0f32;
        for k in 0..self.n_feat {
            dot += (q0[k] + q1[k] * wq) * (p0[k] + p1[k] * wp + s * y[k]);
        }
        let (nw, nc) = self.nbr_terms(u, m, day as i16);
        let z = self.mu + bu + bm * s + dot + nw + s * nc;
        if cfg.envelope { envelope(z, self.bl2(u, m)).0 } else { z }
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        if epoch > 1 { self.lr_scale *= 1.0 - self.cfg.lr_decay; }
        let cfg = self.cfg;
        let g: Vec<f32> = cfg.lr.iter().map(|&x| x * self.lr_scale).collect();
        let l = cfg.reg;
        let (nf, k, nbu, nbm, nsu) = (self.n_feat, cfg.k_nbr, cfg.nbu, cfg.nbm, cfg.nsu);
        let users = get_users(tr.n_users, true, cfg.seed, epoch);

        // N(u) as dense per-item arrays (filled per user), so the neighbour loop is branch-free
        // [present, day, known, r − bl₂]
        let mut dense = vec![[0.0f32; 4]; tr.n_items];
        // time decay 1/(1 + βw·|Δt|) by |Δt|
        let decay: Vec<f32> = (0..4500).map(|x| 1.0 / (1.0 + cfg.beta_w * x as f32)).collect();
        let mut nb_p: Vec<f32> = vec![0.0; k];
        let mut yacc = vec![0.0f32; nf];
        let mut qv = vec![0.0f32; nf];
        let mut zv = vec![0.0f32; nf];
        let mut nb_d: Vec<f32> = vec![0.0; k];
        let mut nb_dk: Vec<f32> = vec![0.0; k];
        let mut nb_r: Vec<f32> = vec![0.0; k];
        let mut tmp_bu = vec![0.0f32; nbu];
        let mut tmp_bm = vec![0.0f32; nbm];
        let mut tmp_s = vec![0.0f32; nsu];

        for &u in users.iter() {
            let (start, end) = (self.user_offsets[u], self.user_offsets[u + 1]);
            if start == end { continue; }
            let nun = self.nu[u].len();
            if k > 0 {
                for e in &self.nu[u] {
                    let kn = !e.rmb.is_nan();
                    dense[e.item as usize] = [1.0, e.day as f32, kn as u8 as f32, if kn { e.rmb } else { 0.0 }];
                }
            }
            // Y(u) from scratch
            let ynorm = 1.0 / (nun as f32).sqrt();
            {
                let mut acc = vec![0.0f32; nf];
                for e in &self.nu[u] {
                    let j = e.item as usize;
                    let wy = e.wy + self.yt[j * NB + e.t36 as usize] + self.yf[j * NB + e.f36 as usize];
                    let (a, b) = (self.y0.row(j), self.y1.row(j));
                    for kk in 0..nf { acc[kk] += a[kk] + b[kk] * wy; }
                }
                let mut row = self.ycache.row_mut(u);
                for kk in 0..nf { row[kk] = acc[kk] * ynorm; }
            }
            yacc.iter_mut().for_each(|x| *x = 0.0);

            for oi in start..end {
                let t = self.order[oi] as usize;
                let m = tr.item_idxs[t] as usize;
                let r = tr.residuals[t];
                let day = tr.dates[t];
                let tf = self.tfeat(u, m, day);

                // forward
                let mut bu = self.bu0[u] + self.bu2[tf.ud];
                for i in 0..nbu {
                    tmp_bu[i] = self.buf[tf.f36 * nbu + i] + self.but[tf.tu * nbu + i];
                    bu += self.bu1[u * nbu + i] * tmp_bu[i];
                }
                let mut bm = self.bm0[m];
                for i in 0..nbm {
                    tmp_bm[i] = self.bmf[tf.f36 * nbm + i] + self.bmt[tf.tm * nbm + i];
                    bm += self.bm1[m * nbm + i] * tmp_bm[i];
                }
                let mut s = 1.0 + self.s0[u] + self.s2[tf.ud];
                for i in 0..nsu {
                    tmp_s[i] = self.sf[tf.f36 * nsu + i] + self.st[tf.tu * nsu + i];
                    s += self.s1[u * nsu + i] * tmp_s[i];
                }
                let wq = sigm(sig_arg(cfg.sig_form, cfg.sq, tf.freq as f32)) - self.sbar_q[m]
                    + self.qt[m * NB + tf.t36] + self.qf[m * NB + tf.f36];
                let wp = sigm(sig_arg(cfg.sig_form, cfg.sp, tf.du)) - self.sbar_p[u] + self.p2[tf.ud];
                let (mut dot, mut qy) = (0.0f32, 0.0f32);
                {
                    let (q0, q1) = (self.q0.row(m).to_slice().unwrap(), self.q1.row(m).to_slice().unwrap());
                    let (p0, p1) = (self.p0.row(u).to_slice().unwrap(), self.p1.row(u).to_slice().unwrap());
                    let y = self.ycache.row(u).to_slice().unwrap();
                    for kk in 0..nf {
                        qv[kk] = q0[kk] + q1[kk] * wq;
                        zv[kk] = p0[kk] + p1[kk] * wp + s * y[kk];
                        dot += qv[kk] * zv[kk];
                        qy += qv[kk] * y[kk];
                    }
                }
                let (mut nw, mut nc, mut sw2, mut sc2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                if k > 0 {
                    let nb = &self.nbrs[m];
                    let (cm, wm) = (&self.c[m * k..m * k + nb.len()], &self.w[m * k..m * k + nb.len()]);
                    let dayf = day as f32;
                    for (rank, &j) in nb.iter().enumerate() {
                        let [pr, dj, kn, rmb] = dense[j as usize];
                        let d = pr * decay[((dayf - dj).abs() as usize).min(4499)];
                        let dk = d * kn;
                        nb_p[rank] = pr; nb_d[rank] = d; nb_dk[rank] = dk; nb_r[rank] = rmb;
                        nc += d * cm[rank]; sc2 += d * d;
                        nw += dk * rmb * wm[rank]; sw2 += dk * dk;
                    }
                }
                let iw = if sw2 > 0.0 { 1.0 / sw2.sqrt() } else { 0.0 };
                let ic = if sc2 > 0.0 { 1.0 / sc2.sqrt() } else { 0.0 };
                let nc_n = nc * ic;
                let z = self.mu + bu + bm * s + dot + nw * iw + s * nc_n;
                let mut e = if cfg.envelope {
                    let (v, dv) = envelope(z, self.bl2(u, m));
                    (v - r) * dv
                } else { z - r };
                if !e.is_finite() { e = 0.0; }

                // backward: biases
                let es = e * s;
                let gs = e * (bm + qy + nc_n);
                self.bu0[u] -= g[BU0] * (e + l[BU0] * self.bu0[u]);
                self.bu2[tf.ud] -= g[BU2] * (e + l[BU2] * self.bu2[tf.ud]);
                for i in 0..nbu {
                    let b1 = self.bu1[u * nbu + i];
                    self.bu1[u * nbu + i] -= g[BU1] * (e * tmp_bu[i] + l[BU1] * b1);
                    let x = &mut self.buf[tf.f36 * nbu + i]; *x -= g[BUF] * (e * b1 + l[BUF] * *x);
                    let x = &mut self.but[tf.tu * nbu + i]; *x -= g[BUT] * (e * b1 + l[BUT] * *x);
                }
                self.bm0[m] -= g[BM0] * (es + l[BM0] * self.bm0[m]);
                for i in 0..nbm {
                    let b1 = self.bm1[m * nbm + i];
                    self.bm1[m * nbm + i] -= g[BM1] * (es * tmp_bm[i] + l[BM1] * b1);
                    let x = &mut self.bmf[tf.f36 * nbm + i]; *x -= g[BMF] * (es * b1 + l[BMF] * *x);
                    let x = &mut self.bmt[tf.tm * nbm + i]; *x -= g[BMT] * (es * b1 + l[BMT] * *x);
                }
                self.s0[u] -= g[S0] * (gs + l[S0] * self.s0[u]);
                self.s2[tf.ud] -= g[S2] * (gs + l[S2] * self.s2[tf.ud]);
                for i in 0..nsu {
                    let b1 = self.s1[u * nsu + i];
                    self.s1[u * nsu + i] -= g[S1] * (gs * tmp_s[i] + l[S1] * b1);
                    let x = &mut self.sf[tf.f36 * nsu + i]; *x -= g[SF] * (gs * b1 + l[SF] * *x);
                    let x = &mut self.st[tf.tu * nsu + i]; *x -= g[ST] * (gs * b1 + l[ST] * *x);
                }

                // backward: factors
                let (mut dq1, mut dp1) = (0.0f32, 0.0f32);
                {
                    let q0 = self.q0.row_mut(m).into_slice().unwrap();
                    let q1 = self.q1.row_mut(m).into_slice().unwrap();
                    for kk in 0..nf {
                        let gq = e * zv[kk];
                        dq1 += gq * q1[kk];
                        q0[kk] -= g[Q0] * (gq + l[Q0] * q0[kk]);
                        q1[kk] -= g[Q1] * (gq * wq + l[Q1] * q1[kk]);
                    }
                }
                {
                    let p0 = self.p0.row_mut(u).into_slice().unwrap();
                    let p1 = self.p1.row_mut(u).into_slice().unwrap();
                    for kk in 0..nf {
                        let gp = e * qv[kk];
                        dp1 += gp * p1[kk];
                        p0[kk] -= g[P0] * (gp + l[P0] * p0[kk]);
                        p1[kk] -= g[P1] * (gp * wp + l[P1] * p1[kk]);
                        yacc[kk] += es * qv[kk];
                    }
                }
                if g[QT] != 0.0 { let x = &mut self.qt[m * NB + tf.t36]; *x -= g[QT] * (dq1 + l[QT] * *x); }
                if g[QF] != 0.0 { let x = &mut self.qf[m * NB + tf.f36]; *x -= g[QF] * (dq1 + l[QF] * *x); }
                if g[P2] != 0.0 { let x = &mut self.p2[tf.ud]; *x -= g[P2] * (dp1 + l[P2] * *x); }

                // backward: neighbourhood
                if k > 0 {
                    let len = self.nbrs[m].len();
                    let (cm, wm) = (&mut self.c[m * k..m * k + len], &mut self.w[m * k..m * k + len]);
                    let (gc, gw, lc, lw) = (g[C], g[W], l[C], l[W]);
                    let (ec, ew) = (es * ic, e * iw);
                    for rank in 0..len {
                        // regularise only the neighbours in N(u) (resp. R(u)): branch-free masks
                        cm[rank] -= gc * (ec * nb_d[rank] + lc * nb_p[rank] * cm[rank]);
                        let kn = (nb_dk[rank] != 0.0) as u8 as f32;
                        wm[rank] -= gw * (ew * nb_dk[rank] * nb_r[rank] + lw * kn * wm[rank]);
                    }
                }
            }

            // y update with the accumulated gradient
            for x in 0..nun {
                let ent = self.nu[u][x];
                let j = ent.item as usize;
                let (it, ifq) = (j * NB + ent.t36 as usize, j * NB + ent.f36 as usize);
                let wy = ent.wy + self.yt[it] + self.yf[ifq];
                let mut dy1 = 0.0f32;
                let mut y0 = self.y0.row_mut(j);
                for kk in 0..nf {
                    let gy = yacc[kk] * ynorm;
                    y0[kk] -= g[Y0] * (gy + l[Y0] * y0[kk]);
                }
                let mut y1 = self.y1.row_mut(j);
                for kk in 0..nf {
                    let gy = yacc[kk] * ynorm;
                    dy1 += gy * y1[kk];
                    y1[kk] -= g[Y1] * (gy * wy + l[Y1] * y1[kk]);
                }
                if g[YT] != 0.0 { let v = &mut self.yt[it]; *v -= g[YT] * (dy1 + l[YT] * *v); }
                if g[YF] != 0.0 { let v = &mut self.yf[ifq]; *v -= g[YF] * (dy1 + l[YF] * *v); }
            }
            if k > 0 { for e in &self.nu[u] { dense[e.item as usize] = [0.0; 4]; } }
        }
        self.rebuild_ycache();
    }
}

/// The configuration from the environment knobs (defaults bk4-d50's).
pub fn config_from_env() -> Bk4Config {
    let mut lr = LR_D50;
    let mut reg = REG_D50;
    // TIME=1: the f200z4 rates for the time adjustments p2, qt, qf, yt, yf
    if ev("TIME", 0) != 0 {
        for (x, idx) in [P2, QT, QF, YT, YF].into_iter().enumerate() {
            lr[idx] = LR_F200Z4_TIME[x];
            reg[idx] = REG_F200Z4_TIME[x];
        }
    }
    // per-group overrides: LR_q0=..., REG_p1=...
    for idx in 1..28 {
        lr[idx] = ev(&format!("LR_{}", NAMES[idx]), lr[idx]);
        reg[idx] = ev(&format!("REG_{}", NAMES[idx]), reg[idx]);
    }
    let ls: f32 = ev("LR_SCALE", 1.0);
    let rs: f32 = ev("REG_SCALE", 1.0);
    for idx in 1..28 { lr[idx] *= ls; reg[idx] *= rs; }

    Bk4Config {
        n_feat: ev("F", 50),
        n_epochs: ev("EPOCHS", 30),
        seed: ev("SEED", 42),
        nbu: ev("NBU", 4),
        nbm: ev("NBM", 20),
        nsu: ev("NSU", 4),
        k_nbr: ev("NBR", 0),
        sorted: ev("SORTED", 1) != 0,
        lr_decay: ev("DECAY", 0.0),
        lr, reg,
        sq: (ev("AQ", 0.0539445), ev("BQ", -5.65647)),
        sp: (ev("AP", 0.00597875), ev("BP", 228.851)),
        sy: (ev("AY", 0.131885), ev("BY", 7.91665)),
        sig_form: ev("SIGF", 0),
        alpha_w: ev("ALPHA_W", 556.041),
        beta_w: ev("BETA_W", 0.00147446),
        bl_m: ev("BL_M", 25.0), bl_u: ev("BL_U", 10.0), bl_s: ev("BL_S", 20.0),
        envelope: ev("ENV", 0) != 0,
        sigma_init: ev("SIGMA", 0.01),
    }
}
