use crate::{Dataset, N_USERS, MaskedDataset, Regressor, calc_user_offsets};
use indicatif::ParallelProgressIterator;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Knn3Config {
    pub threshold: f32, // proximity threshold for neighbor selection
    pub k_min: usize,   // minimum neighbors
    pub k_max: usize,   // maximum neighbors
    pub shrinkage: f32, // support proximity shrinkage
    pub reg: f32,       // weight computation regularization
    pub x: f32,         // mixing parameter (error vs bias-corrected)
    pub bl_reg_m: f32,  // baseline item regularization
    pub bl_reg_u: f32,  // baseline user regularization
}

impl Default for Knn3Config {
    fn default() -> Self {
        Self {
            threshold: 0.25,
            k_min: 10,
            k_max: 60,
            shrinkage: 20000.0,
            reg: 0.01,
            x: 0.8,
            bl_reg_m: 25.0,
            bl_reg_u: 10.0,
        }
    }
}

/// Per-user sorted error lookup for KNN3 prediction.
///
/// CSR-style layout: `items[starts[u]..starts[u+1]]` holds user u's rated items
/// in ascending order, with `errors` aligned. Sorted to enable O(log n) lookup
/// during predict.
struct UserErrors {
    starts: Vec<usize>,
    items: Vec<i32>,
    errors: Vec<f32>,
}

impl UserErrors {
    fn build(tr: &Dataset, user_offsets: &[usize], errors: &[f32]) -> Self {
        let mut starts = vec![0; tr.n_users + 1];
        let mut items = Vec::with_capacity(tr.n_ratings);
        let mut errs = Vec::with_capacity(tr.n_ratings);

        for u in 0..tr.n_users {
            starts[u] = items.len();
            let s = user_offsets[u];
            let e = user_offsets[u + 1];
            // Dataset is user-sorted but not item-sorted within a user;
            // sort here so lookup() can use binary search.
            let mut pairs: Vec<(i32, f32)> = (s..e)
                .filter(|&idx| tr.is_test[idx] == 0)
                .map(|idx| (tr.item_idxs[idx], errors[idx]))
                .collect();
            pairs.sort_unstable_by_key(|p| p.0);
            for (m, err) in pairs {
                items.push(m);
                errs.push(err);
            }
        }
        starts[tr.n_users] = items.len();
        Self { starts, items, errors: errs }
    }

    fn lookup(&self, u: usize, item: i32) -> Option<f32> {
        let s = self.starts[u];
        let e = self.starts[u + 1];
        match self.items[s..e].binary_search(&item) {
            Ok(pos) => Some(self.errors[s + pos]),
            Err(_) => None,
        }
    }
}

/// Solve mat @ w = rhs subject to w >= 0 (active set NNLS).
///
/// Iteratively: solve the unconstrained least-squares restricted to currently
/// active variables; if any solution component is negative, deactivate the most
/// negative one (force its weight to 0) and repeat. Terminates when all active
/// solution components are non-negative (KKT conditions met).
fn solve_nnls(mat: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
    let mut active = vec![true; n];
    for _ in 0..n {
        let idx: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        let k = idx.len();
        if k == 0 { return vec![0.0; n]; }

        // Extract the k×k submatrix and k-vector for the currently active set.
        let mut a = DMatrix::<f64>::zeros(k, k);
        let mut b = DVector::<f64>::zeros(k);
        for (ri, &i) in idx.iter().enumerate() {
            b[ri] = rhs[i];
            for (rj, &j) in idx.iter().enumerate() {
                a[(ri, rj)] = mat[i * n + j];
            }
        }

        // Cholesky fails if not positive-definite (e.g., redundant constraints) — bail to zeros.
        let Some(chol) = a.cholesky() else { return vec![0.0; n]; };
        let sol = chol.solve(&b);

        // Find the most violated non-negativity constraint.
        let mut worst_ri = 0;
        let mut worst_val = 0.0;
        for ri in 0..k {
            if sol[ri] < worst_val { worst_ri = ri; worst_val = sol[ri]; }
        }

        if worst_val >= 0.0 {
            // All weights non-negative — done. Scatter back to full n-vector.
            let mut result = vec![0.0; n];
            for (ri, &i) in idx.iter().enumerate() { result[i] = sol[ri]; }
            return result;
        }
        active[idx[worst_ri]] = false;
    }
    vec![0.0; n]
}

/// Compute regularized baseline: mu + bm[i] + bu[u], using non-test ratings only.
fn compute_baseline(tr: &Dataset, user_offsets: &[usize], reg_m: f32, reg_u: f32)
                    -> (f32, Vec<f32>, Vec<f32>) {
    let n_items = tr.n_items;
    let n_users = tr.n_users;
    let n_ratings = tr.n_ratings;

    let mut mu_sum = 0.0;
    let mut mu_cnt = 0;
    for idx in 0..n_ratings {
        if tr.is_test[idx] != 0 { continue; }
        mu_sum += tr.raw_ratings[idx] as f64;
        mu_cnt += 1;
    }
    let mu = (mu_sum / mu_cnt as f64) as f32;

    let mut bm_sum = vec![0.0; n_items];
    let mut bm_cnt = vec![0; n_items];
    for idx in 0..n_ratings {
        if tr.is_test[idx] != 0 { continue; }
        let m = tr.item_idxs[idx] as usize;
        bm_sum[m] += (tr.raw_ratings[idx] as f32 - mu) as f64;
        bm_cnt[m] += 1;
    }
    let bm: Vec<f32> = (0..n_items)
        .map(|m| (bm_sum[m] / (bm_cnt[m] as f64 + reg_m as f64)) as f32)
        .collect();

    let mut bu = vec![0.0; n_users];
    for u in 0..n_users {
        let (s, e) = (user_offsets[u], user_offsets[u + 1]);
        let mut sum = 0.0;
        let mut cnt = 0;
        for idx in s..e {
            if tr.is_test[idx] != 0 { continue; }
            sum += (tr.raw_ratings[idx] as f32 - mu - bm[tr.item_idxs[idx] as usize]) as f64;
            cnt += 1;
        }
        if cnt > 0 { bu[u] = (sum / (cnt as f64 + reg_u as f64)) as f32; }
    }

    (mu, bm, bu)
}

/// Shared read-only context for per-item neighbor computation.
struct BuildCtx<'a> {
    user_offsets: &'a [usize],
    item_idxs: &'a [i32],
    is_test: &'a [i8],
    errors: &'a [f32],
    r_prime: &'a [f32],
    item_cnts: &'a [f32],
    n_items: usize,
    cfg: &'a Knn3Config,
}

/// Compute per-item neighbor weights for a single item.
fn compute_item_neighbors(ctx: &BuildCtx, m: usize, item_users_m: &[u32])
                           -> Vec<(usize, f32)> {
    let n_m = ctx.item_cnts[m];
    if n_m < 1.0 { return Vec::new(); }

    // Count co-ratings with all other items
    let mut counts = vec![0; ctx.n_items];
    for &u in item_users_m {
        let u = u as usize;
        for idx in ctx.user_offsets[u]..ctx.user_offsets[u + 1] {
            if ctx.is_test[idx] != 0 { continue; }
            counts[ctx.item_idxs[idx] as usize] += 1;
        }
    }
    counts[m] = 0;

    // Compute proximity and select candidates.
    // phi  = lift over independence: P(co-rated) / [P(rated m) * P(rated j)],
    //        with N=N_USERS (Netflix Prize user count) plugged in.
    // prox = phi shrunk toward 0 for low-support pairs, à la Bell-Koren.
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for j in 0..ctx.n_items {
        if counts[j] == 0 { continue; }
        let n_mj = counts[j] as f32;
        let n_j = ctx.item_cnts[j];
        if n_j < 1.0 { continue; }
        let phi = n_mj * N_USERS as f32 / (n_m * n_j);
        let prox = phi * n_mj / (n_mj + ctx.cfg.shrinkage);
        candidates.push((j, prox));
    }
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take all neighbors above threshold, but clamp count to [k_min, k_max].
    let above = candidates.iter().filter(|(_, p)| *p >= ctx.cfg.threshold).count();
    let k = above.max(ctx.cfg.k_min).min(ctx.cfg.k_max).min(candidates.len());
    candidates.truncate(k);

    if candidates.is_empty() { return Vec::new(); }

    let big_k = candidates.len();
    let neigh_map: HashMap<i32, usize> = candidates.iter().enumerate()
        .map(|(ni, (j, _))| (*j as i32, ni)).collect();

    // Build covariance C and cross-correlation d for two signals:
    //   _err:  raw residuals (errors[]) — what the regression directly fits
    //   _bias: bias-corrected ratings (r_prime[]) — used as a stabilizing prior
    // Accumulating across all users who rated m: C[a,b] = Σ_u x_a(u) * x_b(u),
    // d[a] = Σ_u x_a(u) * x_m(u), where x_a(u) is u's value on neighbor a.
    let mut c_err  = vec![0.0; big_k * big_k];
    let mut c_bias = vec![0.0; big_k * big_k];
    let mut d_err  = vec![0.0; big_k];
    let mut d_bias = vec![0.0; big_k];

    for &u_idx in item_users_m {
        let u = u_idx as usize;
        let (s, e) = (ctx.user_offsets[u], ctx.user_offsets[u + 1]);
        // For this user, collect (neighbor_index, err, r_prime) for each of u's
        // rated items that's also a neighbor of m. Also pick up u's own value on m.
        let mut user_neigh: Vec<(usize, f32, f32)> = Vec::new();
        let mut m_err = 0.0;
        let mut m_rp = 0.0;
        let mut m_found = false;

        for idx in s..e {
            if ctx.is_test[idx] != 0 { continue; }
            let j = ctx.item_idxs[idx];
            if j as usize == m {
                m_err = ctx.errors[idx];
                m_rp = ctx.r_prime[idx];
                m_found = true;
            }
            if let Some(&ni) = neigh_map.get(&j) {
                user_neigh.push((ni, ctx.errors[idx], ctx.r_prime[idx]));
            }
        }
        if !m_found { continue; }

        // Cross-correlation with target m.
        for &(ni, e_val, rp_val) in &user_neigh {
            d_err[ni] += e_val as f64 * m_err as f64;
            d_bias[ni] += rp_val as f64 * m_rp as f64;
        }
        // Symmetric outer-product accumulation: only iterate b >= a, mirror across diagonal.
        for a in 0..user_neigh.len() {
            let (ni_a, ea, rpa) = user_neigh[a];
            for b in a..user_neigh.len() {
                let (ni_b, eb, rpb) = user_neigh[b];
                let ce = ea as f64 * eb as f64;
                let cb = rpa as f64 * rpb as f64;
                c_err[ni_a * big_k + ni_b] += ce;
                c_bias[ni_a * big_k + ni_b] += cb;
                if ni_a != ni_b {
                    c_err[ni_b * big_k + ni_a] += ce;
                    c_bias[ni_b * big_k + ni_a] += cb;
                }
            }
        }
    }

    // Mix: M = x*C + (1-x)*C' + reg*I, rhs = x*d + (1-x)*d'.
    // x interpolates between the two formulations; reg*I is Tikhonov regularization.
    // Solving M w = rhs s.t. w >= 0 (NNLS) gives the neighbor weights.
    let x = ctx.cfg.x as f64;
    let reg = ctx.cfg.reg as f64;
    let mut sys = vec![0.0; big_k * big_k];
    let mut sys_rhs = vec![0.0; big_k];
    for i in 0..big_k {
        sys_rhs[i] = x * d_err[i] + (1.0 - x) * d_bias[i];
        for j in 0..big_k {
            sys[i * big_k + j] = x * c_err[i * big_k + j] + (1.0 - x) * c_bias[i * big_k + j];
        }
        sys[i * big_k + i] += reg;
    }

    let weights = solve_nnls(&sys, &sys_rhs, big_k);
    let mut item_neigh: Vec<(usize, f32)> = Vec::new();
    for (ni, &w) in weights.iter().enumerate() {
        if w > 0.0 { item_neigh.push((candidates[ni].0, w as f32)); }
    }
    item_neigh
}

/// Build KNN3 neighbor weights for all items (parallel).
fn build_knn3(tr: &Dataset, user_offsets: &[usize], errors: &[f32], cfg: &Knn3Config)
              -> Vec<Vec<(usize, f32)>> {
    let n_items = tr.n_items;
    let n_users = tr.n_users;
    let n_ratings = tr.n_ratings;

    crate::teeln!("  KNN3: computing baseline...");
    let (mu, bm, bu) = compute_baseline(tr, user_offsets, cfg.bl_reg_m, cfg.bl_reg_u);

    // r_prime = bias-removed ratings; serves as the "stabilizing prior" signal
    // (vs `errors`, which are the residuals we actually want to fit).
    let mut r_prime = vec![0.0; n_ratings];
    for idx in 0..n_ratings {
        let u = tr.user_idxs[idx] as usize;
        let m = tr.item_idxs[idx] as usize;
        r_prime[idx] = tr.raw_ratings[idx] as f32 - mu - bm[m] - bu[u];
    }

    // Build item->users inverted index (non-test only)
    crate::teeln!("  KNN3: building inverted index...");
    let mut item_users: Vec<Vec<u32>> = vec![Vec::new(); n_items];
    for u in 0..n_users {
        for idx in user_offsets[u]..user_offsets[u + 1] {
            if tr.is_test[idx] != 0 { continue; }
            item_users[tr.item_idxs[idx] as usize].push(u as u32);
        }
    }

    // Item counts (non-test only)
    let item_cnts: Vec<f32> = item_users.iter().map(|v| v.len() as f32).collect();

    // Per-item: find neighbors and compute regression weights (PARALLEL)
    crate::teeln!("  KNN3: computing per-item weights ({} items, parallel)...", n_items);

    let ctx = BuildCtx {
        user_offsets,
        item_idxs: tr.item_idxs.as_slice().unwrap(),
        is_test: tr.is_test.as_slice().unwrap(),
        errors,
        r_prime: &r_prime,
        item_cnts: &item_cnts,
        n_items,
        cfg,
    };

    let neighbors: Vec<Vec<(usize, f32)>> = progress_count!((0..n_items).into_par_iter(), n_items as u64)
        .map(|m| compute_item_neighbors(&ctx, m, &item_users[m]))
        .collect();

    let total_neigh: usize = neighbors.iter().map(|v| v.len()).sum();
    let nonzero: usize = neighbors.iter().filter(|v| !v.is_empty()).count();
    crate::teeln!("  KNN3: {} items with neighbors, {} total weights", nonzero, total_neigh);

    neighbors
}

// ==== Regressor wrapper ====

pub struct Knn3Model {
    neighbors: Vec<Vec<(usize, f32)>>,
    uerr: UserErrors,
}

impl Regressor for Knn3Model {
    type Config = Knn3Config;

    fn new(tr: &Dataset, _pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let user_offsets = calc_user_offsets(tr);
        let user_offsets_sl = user_offsets.as_slice().unwrap();
        let errors = tr.residuals.as_slice().unwrap();

        let neighbors = build_knn3(tr, user_offsets_sl, errors, &cfg);
        let uerr = UserErrors::build(tr, user_offsets_sl, errors);

        Self { neighbors, uerr }
    }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, _epoch: usize) {}

    fn predict(&self, u: usize, i: usize, _day: i32) -> f32 {
        let neigh = &self.neighbors[i];
        if neigh.is_empty() { return 0.0; }
        let mut pred = 0.0;
        for &(j, w) in neigh {
            if let Some(err) = self.uerr.lookup(u, j as i32) {
                pred += w * err;
            }
        }
        pred
    }
}
