// BK1 integrated model.
// Ref: Piotte & Chabbert 2009, "The Pragmatic Theory solution to the Netflix Grand Prize"
// (Section 3.1, equations 14–16):
//
// dev̂(u,t)  = k1·sign(t−t̄_u)·|t−t̄_u|^β − dev̄_u
// z(u,m,t)  = μ
//           + bi[m] + bit_bin[m,t₃₀]
//           + bu[u] + bu1[u]·dev̂ + k2·bu2[u,t]
//           + Σₖ q[m,k]·(p[u,k] + p1[u,k]·dev̂ + h[u,t] + su[u,k])
//           + (1/√|Rᵏ(m;u)|)·Σ_{j∈Rᵏ} (r_uj−bl₁_uj)·w[m,rank_j]
//           + (1/√|Nᵏ(m;u)|)·Σ_{j∈Nᵏ} c[m,rank_j]
//
// LR decays by factor (1 − Δγ) at the start of each epoch.

use crate::{Dataset, MaskedDataset, Regressor, calc_gbias, calc_user_offsets, get_users, rand_array2};
use crate::tx::SparseUD;
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use rand::{SeedableRng, rngs::StdRng};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Bk1Config {
    pub n_feat:        usize,
    pub n_epochs:      usize,
    pub seed:          u64,
    pub shuffle_users: bool,

    // Temporal
    pub n_time_bins: usize,  // bins for item time-bias (t₃₀ = 30)
    pub beta:        f32,    // k4: dev exponent (0.4)
    pub k1:          f32,    // dev scaling     (0.0363636)
    pub k2:          f32,    // per-day user bias scale (0.909091)
    pub dev_mean:    bool,   // subtract per-user mean of dev̂

    // k-NN
    pub k_neighbors: usize,  // 0 = no neighborhood term
    pub alpha_rho:   f32,    // shrinkage for neighbor selection (100)

    // Baseline₁ shrinkage (for r − bl₁ in the w-term)
    pub lambda1: f32,        // item damping (25)
    pub lambda2: f32,        // user damping (10)

    // Learning rates — grouped as in the paper
    pub lr_bias:  f32,   // γ1: bu, bu1, bu2, bi, bit_bin
    pub lr_fact:  f32,   // γ2: p, p1, q, y
    pub lr_nbr:   f32,   // γ3: w, c
    pub lr_h:     f32,   // γ9: h(u,t) per-day correction
    pub lr_decay: f32,   // Δγ: lr_scale *= (1 − Δγ) at start of each epoch

    // Regularisation — grouped as in the paper
    pub reg_bias: f32,   // λ6
    pub reg_fact: f32,   // λ7
    pub reg_nbr:  f32,   // λ8
    pub reg_h:    f32,   // λ9
}

// ---------------------------------------------------------------------------
// k-NN neighbor precomputation from sim matrices
// ---------------------------------------------------------------------------
//
// Score(m, j) = cos(m,j) · n_mj / (n_mj + α_ρ)
// cos(m,j)    = rtg_prod[m,j] / sqrt(rtg_prod[m,m] · rtg_prod[j,j])
//
// Only positive scores are kept. Result is sorted descending.

pub(crate) fn precompute_neighbors(
    dataset: &str,
    k: usize,
    alpha: f32,
    n_items: usize,
) -> Vec<Vec<u16>> {
    println!("Loading sim matrices for neighbor precomputation…");
    let prod: Array2<f32> = read_npy(format!("sim/rtg_prod.{}.npy", dataset)).unwrap();
    let supp: Array2<f32> = read_npy(format!("sim/rtg_supp.{}.npy", dataset)).unwrap();

    // Per-item self-norm: sqrt(prod[m,m])
    let norms: Vec<f32> = (0..n_items).map(|m| prod[[m, m]].max(0.0).sqrt()).collect();

    let mut neighbors: Vec<Vec<u16>> = vec![Vec::new(); n_items];
    // Reuse a single buffer per item row to avoid repeated allocations.
    let mut buf: Vec<(f32, u16)> = vec![(0.0, 0); n_items];

    println!("Computing top-{k} neighbors per item…");
    for m in 0..n_items {
        let nm = norms[m];
        if nm == 0.0 { continue; }

        let mut cnt = 0usize;
        for j in 0..n_items {
            if j == m { continue; }
            let nj = norms[j];
            if nj == 0.0 { continue; }
            let n_mj = supp[[m, j]];
            if n_mj <= 0.0 { continue; }
            let cos = prod[[m, j]] / (nm * nj);
            let score = cos * n_mj / (n_mj + alpha);
            if score > 0.0 {
                buf[cnt] = (score, j as u16);
                cnt += 1;
            }
        }

        let take = k.min(cnt);
        if cnt > take {
            // Partial sort: bring the top-`take` elements to buf[0..take]
            buf[..cnt].select_nth_unstable_by(take, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        buf[..take].sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        neighbors[m] = buf[..take].iter().map(|&(_, j)| j).collect();
    }
    neighbors
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Bk1Model {
    cfg:   Bk1Config,
    gbias: f32,

    // Biases
    bu:      Array1<f32>,    // [n_users]        — user static bias
    bu1:     Array1<f32>,    // [n_users]        — user time-varying coeff
    bu2:     Vec<f32>,       // [n_ud]           — per-day user bias (×k2)
    bi:      Array1<f32>,    // [n_items]        — item static bias
    bit_bin: Array2<f32>,    // [n_items, n_time_bins] — item bin bias

    // Latent factors
    p:      Array2<f32>,     // [n_users, n_feat] — static user factors
    p1:     Array2<f32>,     // [n_users, n_feat] — time-varying user factors
    h:      Vec<f32>,        // [n_ud]            — per-(user,day) scalar correction
    q:      Array2<f32>,     // [n_items, n_feat] — item factors
    y:      Array2<f32>,     // [n_items, n_feat] — implicit-feedback factors
    ycache: Array2<f32>,     // [n_users, n_feat] — Σ_j y[j] / √|N(u)|

    // k-NN  (sizes n_items × k_neighbors, zero-init)
    neighbors: Vec<Vec<u16>>,  // [n_items][≤k] — top-k neighbor indices
    w:  Vec<f32>,              // [n_items * k] — explicit weights
    c:  Vec<f32>,              // [n_items * k] — implicit weights

    // Per-user lookup tables
    user_rated_items: Vec<Vec<u16>>,  // R(u) item indices (sorted)
    user_rated_rmbli: Vec<Vec<f32>>,  // r − baseline₁ at same positions
    user_nu:          Vec<Vec<u16>>,  // N(u) = R(u) ∪ implicit (sorted)

    // Items in pr per user (for ycache rebuild and su computation)
    probe_items_by_user: Vec<Vec<u16>>,

    // Temporal state
    tu_mean:      Array1<f32>,  // [n_users] — mean rating date
    dev_mean_u:   Array1<f32>,  // [n_users] — per-user mean dev̂ (0 if !dev_mean)
    day_range:    i32,
    ud:           SparseUD,
    user_offsets: Array1<usize>,

    // Learning-rate scale (decays each epoch)
    lr_scale: f32,
}

impl Bk1Model {
    #[inline]
    fn dev(&self, u: usize, day: i32) -> f32 {
        let dt = day as f32 - self.tu_mean[u];
        let raw = if dt == 0.0 { 0.0 } else { dt.signum() * dt.abs().powf(self.cfg.beta) };
        self.cfg.k1 * raw - self.dev_mean_u[u]
    }

    #[inline]
    fn time_bin(&self, day: i32) -> usize {
        let b = (day as i64 * self.cfg.n_time_bins as i64 / self.day_range as i64) as usize;
        b.min(self.cfg.n_time_bins - 1)
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

    /// Neighborhood score for prediction (no heap alloc).
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

impl Regressor for Bk1Model {
    type Config = Bk1Config;

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
        // bl₁(u,m) = μ + btilde_m[m] + btilde_u[u]
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

        // ── Per-user implicit items from pr (linear scan, pr is item-sorted) ─
        let mut implicit_per_user: Vec<Vec<u16>> = vec![Vec::new(); n_users];
        for t in 0..pr.n_ratings {
            let u = pr.user_idxs[t] as usize;
            let i = pr.item_idxs[t] as u16;
            // Only items not already in R(u)
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
            nu.sort_unstable(); // no duplicates since implicit is disjoint from rated
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
            p, p1,
            h: vec![0.0; n_ud],
            q, y,
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
            user_offsets,
            lr_scale: 1.0,
        };
        model.rebuild_ycache(tr, pr);
        model
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let dev   = self.dev(u, day);
        let day16 = day as i16;
        let b     = self.time_bin(day);
        let ud_idx = self.ud.index(u, day16);

        let bu2_val = ud_idx.map_or(0.0, |idx| self.bu2[idx]);
        let h_val   = ud_idx.map_or(0.0, |idx| self.h[idx]);

        let mut score = self.gbias
            + self.bu[u] + self.bu1[u] * dev + self.cfg.k2 * bu2_val
            + self.bi[i] + self.bit_bin[[i, b]];

        let su = self.ycache.row(u);
        for f in 0..self.cfg.n_feat {
            let pu = self.p[[u, f]] + self.p1[[u, f]] * dev + h_val + su[f];
            score += self.q[[i, f]] * pu;
        }

        score + self.nbr_score(u, i)
    }

    fn fit_epoch(&mut self, tr: &Dataset, pr: &MaskedDataset, epoch: usize) {
        // Apply LR decay at the start of each epoch (skip epoch 0 so the
        // first epoch runs at the full configured rate).
        if epoch > 0 { self.lr_scale *= 1.0 - self.cfg.lr_decay; }

        let s    = self.lr_scale;
        let cfg  = self.cfg;
        let nf   = cfg.n_feat;
        let k    = cfg.k_neighbors;
        let lr_b = cfg.lr_bias * s;
        let lr_f = cfg.lr_fact * s;
        let lr_n = cfg.lr_nbr  * s;
        let lr_h = cfg.lr_h    * s;

        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        // Scratch buffers for k-NN lookup — reused across ratings of the same user.
        // `nbr_rmb[rank]` = Some(r−bl₁) if neighbor rank ∈ R(u), else None.
        // `nbr_in_n[rank]` = true if neighbor rank ∈ N(u).
        let mut nbr_rmb: Vec<Option<f32>> = vec![None; k];
        let mut nbr_in_n: Vec<bool>       = vec![false; k];

        for &u in crate::progress!(users.iter()) {
            let start = self.user_offsets[u];
            let end   = self.user_offsets[u + 1];
            if start == end { continue; }

            // ── Compute su (NSVD1 contribution) from scratch ─────────────────
            let cnt_r  = (end - start) as f32;
            let cnt_p  = self.probe_items_by_user[u].len() as f32;
            let norm   = (cnt_r + cnt_p).sqrt();

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

            // Accumulate per-epoch gradient for y-update
            let mut sum_err_q = Array1::<f32>::zeros(nf);

            for t in start..end {
                let i     = tr.item_idxs[t] as usize;
                let r     = tr.residuals[t];
                let day   = tr.dates[t] as i32;
                let day16 = tr.dates[t];
                let dev   = self.dev(u, day);
                let b     = self.time_bin(day);
                let ud_idx = self.ud.index(u, day16);

                let bu2_val = ud_idx.map_or(0.0, |idx| self.bu2[idx]);
                let h_val   = ud_idx.map_or(0.0, |idx| self.h[idx]);

                // ── Forward: bias ─────────────────────────────────────────────
                let mut score = self.gbias
                    + self.bu[u] + self.bu1[u] * dev + cfg.k2 * bu2_val
                    + self.bi[i] + self.bit_bin[[i, b]];

                // ── Forward: latent factors ───────────────────────────────────
                for f in 0..nf {
                    let pu = self.p[[u, f]] + self.p1[[u, f]] * dev + h_val + su[f];
                    score += self.q[[i, f]] * pu;
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
                self.bu[u]          -= lr_b * (err + cfg.reg_bias * self.bu[u]);
                self.bu1[u]         -= lr_b * (err * dev + cfg.reg_bias * self.bu1[u]);
                self.bi[i]          -= lr_b * (err + cfg.reg_bias * self.bi[i]);
                self.bit_bin[[i,b]] -= lr_b * (err + cfg.reg_bias * self.bit_bin[[i, b]]);
                if let Some(idx) = ud_idx {
                    self.bu2[idx] -= lr_b * (err * cfg.k2 + cfg.reg_bias * self.bu2[idx]);
                }

                // ── Backward: h(u,t) ─────────────────────────────────────────
                // ∂z/∂h = Σ_f q[i,f]  (h shifts all features uniformly)
                if lr_h != 0.0 {
                    if let Some(idx) = ud_idx {
                        let sum_q: f32 = self.q.row(i).sum();
                        self.h[idx] -= lr_h * (err * sum_q + cfg.reg_h * self.h[idx]);
                    }
                }

                // ── Backward: latent factors (per-rating) ─────────────────────
                for f in 0..nf {
                    let q_f  = self.q[[i, f]];
                    let p_f  = self.p[[u, f]];
                    let p1_f = self.p1[[u, f]];
                    let pu_eff = p_f + p1_f * dev + h_val + su[f];

                    sum_err_q[f] += err * q_f;

                    self.p[[u, f]]  -= lr_f * (err * q_f + cfg.reg_fact * p_f);
                    self.p1[[u, f]] -= lr_f * (err * q_f * dev + cfg.reg_fact * p1_f);
                    self.q[[i, f]]  -= lr_f * (err * pu_eff + cfg.reg_fact * q_f);
                }

                // ── Backward: k-NN weights ────────────────────────────────────
                if k > 0 {
                    let nbrs = &self.neighbors[i];
                    for (rank, _) in nbrs.iter().enumerate() {
                        if nbr_in_n[rank] {
                            let c_idx = i * k + rank;
                            self.c[c_idx] -= lr_n * (err * nn + cfg.reg_nbr * self.c[c_idx]);
                            if let Some(rmb) = nbr_rmb[rank] {
                                let w_idx = i * k + rank;
                                self.w[w_idx] -= lr_n * (err * nr * rmb + cfg.reg_nbr * self.w[w_idx]);
                            }
                        }
                    }
                }
            } // end per-rating loop

            // ── y-factor update (accumulated) ────────────────────────────────
            // Gradient: ∂L/∂y[j,f] = (sum_err_q[f] / norm) per training item j∈R(u)
            for t in start..end {
                let j = tr.item_idxs[t] as usize;
                for f in 0..nf {
                    let yj = self.y[[j, f]];
                    self.y[[j, f]] -= lr_f * (sum_err_q[f] / norm + cfg.reg_fact * yj);
                }
            }
        } // end user loop

        self.rebuild_ycache(tr, pr);
    }
}
