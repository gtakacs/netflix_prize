//! Non-Linear Post-Processing: a cubic-polynomial correction over a base model's
//! predictions, fitted by ALS on per-user and per-item polynomial coefficients.
//! For each rating, the correction is `Σ a_u[k] · φ_u[k] + Σ b_i[k] · φ_i[k]`
//! where `φ_u = [1, d, d², d³]` with `d = x − μ_u` (and similarly for items).
//! Optimal regularization can be tuned via Nelder-Mead — see `fit_with_nm`.

use crate::{Dataset, MaskedDataset, Regressor};
use ndarray::Array1;
use ndarray_npy::read_npy;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct NlppConfig {
    pub base_model: &'static str,
    pub preds_dir: &'static str,
    pub n_als_iters: usize,
    /// Per-degree (alpha, lambda) regularization for user coefficients a[k].
    pub reg_a: [(f32, f32); 4],
    /// Per-degree (alpha, lambda) regularization for item coefficients b[k].
    pub reg_b: [(f32, f32); 4],
    /// Bayesian shrinkage for per-user mean (toward global mean).
    pub shrinkage_u: f32,
    /// Bayesian shrinkage for per-item mean (toward global mean).
    pub shrinkage_i: f32,
    /// Optional path to a 16-element f32 .npy file with reg params.
    /// If `Some`, overrides `reg_a` / `reg_b` at construction.
    /// Layout: `[α_a0, λ_a0, α_a1, λ_a1, α_a2, λ_a2, α_a3, λ_a3,
    ///          α_b0, λ_b0, α_b1, λ_b1, α_b2, λ_b2, α_b3, λ_b3]`.
    pub regs_path: Option<&'static str>,
}

pub struct NlppModel {
    cfg: NlppConfig,
    a: Vec<[f32; 4]>,         // per-user coefficients
    b: Vec<[f32; 4]>,         // per-item coefficients
    mu_u: Vec<f32>,           // shrunk per-user mean rating
    mu_m: Vec<f32>,           // shrunk per-item mean rating
    x_tr: Array1<f32>,        // base predictions (train set)
    x_pr: Array1<f32>,        // base predictions (probe/qual set)
    by_user: Vec<Vec<usize>>, // user → [rating idx in train]
    by_item: Vec<Vec<usize>>, // item → [rating idx in train]

    // Lookup for predict(): per-user, sorted by item_idx
    x_lookup_items: Vec<i32>,
    x_lookup_vals: Vec<f32>,
    x_lookup_offsets: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Nelder-Mead (general-purpose pub fn)
// ---------------------------------------------------------------------------

/// Minimize `eval` starting from `x0` using the Nelder-Mead simplex method.
///
/// - `x0`: initial point (n-dim)
/// - `init_scale`: per-axis perturbation used to build the initial simplex
///   (vertex j is `x0` with `x0[j] += init_scale`)
/// - `n_steps`: number of NM iterations
/// - `eval`: objective function (lower is better)
/// - `on_step`: per-step callback `(step, best_x, best_f)` — pass `|_, _, _| {}` to silence
///
/// Returns `(best_x, best_f)` — the simplex vertex with the lowest f after `n_steps`.
pub fn nelder_mead<F, L>(
    x0: &[f64],
    init_scale: f64,
    n_steps: usize,
    mut eval: F,
    mut on_step: L,
) -> (Vec<f64>, f64)
where
    F: FnMut(&[f64]) -> f64,
    L: FnMut(usize, &[f64], f64),
{
    let n = x0.len();

    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    let f0 = eval(x0);
    simplex.push((x0.to_vec(), f0));
    for j in 0..n {
        let mut v = x0.to_vec();
        v[j] += init_scale;
        let fv = eval(&v);
        simplex.push((v, fv));
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for step in 0..n_steps {
        let worst = n;
        let second_worst = n - 1;
        let best = 0;

        let centroid: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| simplex[i].0[j]).sum::<f64>() / n as f64)
            .collect();

        // Reflection
        let xr: Vec<f64> = (0..n)
            .map(|j| 2.0 * centroid[j] - simplex[worst].0[j])
            .collect();
        let fr = eval(&xr);

        if fr < simplex[second_worst].1 && fr >= simplex[best].1 {
            simplex[worst] = (xr, fr);
        } else if fr < simplex[best].1 {
            // Expansion
            let xe: Vec<f64> = (0..n)
                .map(|j| 3.0 * centroid[j] - 2.0 * simplex[worst].0[j])
                .collect();
            let fe = eval(&xe);
            simplex[worst] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else {
            // Contraction
            let xc: Vec<f64> = if fr < simplex[worst].1 {
                (0..n).map(|j| 0.5 * (centroid[j] + xr[j])).collect()
            } else {
                (0..n).map(|j| 0.5 * (centroid[j] + simplex[worst].0[j])).collect()
            };
            let fc = eval(&xc);
            if fc < simplex[worst].1 {
                simplex[worst] = (xc, fc);
            } else {
                // Shrink toward best
                let xbest = simplex[best].0.clone();
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i].0[j] = 0.5 * (simplex[i].0[j] + xbest[j]);
                    }
                    simplex[i].1 = eval(&simplex[i].0);
                }
            }
        }

        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        on_step(step, &simplex[0].0, simplex[0].1);
    }

    let (best_x, best_f) = simplex.into_iter().next().unwrap();
    (best_x, best_f)
}

// ---------------------------------------------------------------------------
// Linear solve helpers
// ---------------------------------------------------------------------------

fn solve4(xtx: &[f64; 16], xty: &[f64; 4]) -> [f32; 4] {
    let l00 = xtx[0].sqrt();
    let l10 = xtx[4] / l00;
    let l11 = (xtx[5] - l10 * l10).sqrt();
    let l20 = xtx[8] / l00;
    let l21 = (xtx[9] - l20 * l10) / l11;
    let l22 = (xtx[10] - l20 * l20 - l21 * l21).sqrt();
    let l30 = xtx[12] / l00;
    let l31 = (xtx[13] - l30 * l10) / l11;
    let l32 = (xtx[14] - l30 * l20 - l31 * l21) / l22;
    let l33 = (xtx[15] - l30 * l30 - l31 * l31 - l32 * l32).sqrt();

    let y0 = xty[0] / l00;
    let y1 = (xty[1] - l10 * y0) / l11;
    let y2 = (xty[2] - l20 * y0 - l21 * y1) / l22;
    let y3 = (xty[3] - l30 * y0 - l31 * y1 - l32 * y2) / l33;

    let x3 = y3 / l33;
    let x2 = (y2 - l32 * x3) / l22;
    let x1 = (y1 - l21 * x2 - l31 * x3) / l11;
    let x0 = (y0 - l10 * x1 - l20 * x2 - l30 * x3) / l00;

    [x0 as f32, x1 as f32, x2 as f32, x3 as f32]
}

// ---------------------------------------------------------------------------
// NlppModel
// ---------------------------------------------------------------------------

impl NlppModel {
    /// Construct from raw `Dataset`s. Used by hyperparameter-search binaries
    /// (paropt) that need full access to `pr.residuals` for evaluating fitness.
    pub fn new_with_dataset(tr: &Dataset, pr: &Dataset, cfg: NlppConfig) -> Self {
        Self::new_inner(tr, &pr.name, &pr.user_idxs, &pr.item_idxs, pr.n_ratings, cfg)
    }

    fn new_inner(
        tr: &Dataset,
        pr_name: &str,
        pr_user_idxs: &Array1<i32>,
        pr_item_idxs: &Array1<i32>,
        pr_n_ratings: usize,
        mut cfg: NlppConfig,
    ) -> Self {
        // If regs_path is set, override reg_a/reg_b with file contents (16 f32s).
        if let Some(path) = cfg.regs_path {
            let regs: Array1<f32> = read_npy(path).expect(path);
            assert_eq!(regs.len(), 16, "regs file '{}' must contain 16 floats", path);
            for k in 0..4 {
                cfg.reg_a[k] = (regs[2 * k], regs[2 * k + 1]);
                cfg.reg_b[k] = (regs[8 + 2 * k], regs[8 + 2 * k + 1]);
            }
        }

        let x_tr: Array1<f32> = read_npy(
            format!("{}/{}.{}.npy", cfg.preds_dir, cfg.base_model, tr.name)
        ).unwrap();
        let x_pr: Array1<f32> = read_npy(
            format!("{}/{}.{}.npy", cfg.preds_dir, cfg.base_model, pr_name)
        ).unwrap();

        let n_users = tr.n_users;
        let n_items = tr.n_items;

        // Compute shrunk per-user / per-item means from training raw ratings.
        let mut sum_u = vec![0.0f64; n_users];
        let mut cnt_u = vec![0u32; n_users];
        let mut sum_m = vec![0.0f64; n_items];
        let mut cnt_m = vec![0u32; n_items];
        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;
            let r = tr.raw_ratings[idx] as f64;
            sum_u[u] += r;
            cnt_u[u] += 1;
            sum_m[i] += r;
            cnt_m[i] += 1;
        }
        let mut mu_u = vec![0.0f32; n_users];
        let mut mu_m = vec![0.0f32; n_items];
        let global_mean = sum_u.iter().sum::<f64>() / cnt_u.iter().map(|&c| c as f64).sum::<f64>();
        let sh_u = cfg.shrinkage_u as f64;
        let sh_i = cfg.shrinkage_i as f64;
        for u in 0..n_users {
            let n = cnt_u[u] as f64;
            mu_u[u] = ((sum_u[u] + sh_u * global_mean) / (n + sh_u)) as f32;
        }
        for i in 0..n_items {
            let n = cnt_m[i] as f64;
            mu_m[i] = ((sum_m[i] + sh_i * global_mean) / (n + sh_i)) as f32;
        }

        // Build by_user / by_item indices (one usize per training rating).
        let mut by_user = vec![Vec::new(); n_users];
        let mut by_item = vec![Vec::new(); n_items];
        for idx in 0..tr.n_ratings {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;
            by_user[u].push(idx);
            by_item[i].push(idx);
        }

        // Build x_lookup: per-user list of (item, base-pred), sorted by item.
        let total = tr.n_ratings + pr_n_ratings;
        let mut entries: Vec<(u32, i32, f32)> = Vec::with_capacity(total);
        for idx in 0..tr.n_ratings {
            entries.push((tr.user_idxs[idx] as u32, tr.item_idxs[idx], x_tr[idx]));
        }
        for idx in 0..pr_n_ratings {
            entries.push((pr_user_idxs[idx] as u32, pr_item_idxs[idx], x_pr[idx]));
        }
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut x_lookup_items: Vec<i32> = Vec::with_capacity(total);
        let mut x_lookup_vals: Vec<f32> = Vec::with_capacity(total);
        let mut x_lookup_offsets = vec![0usize; n_users + 1];
        for &(u, item, val) in &entries {
            x_lookup_items.push(item);
            x_lookup_vals.push(val);
            x_lookup_offsets[u as usize + 1] += 1;
        }
        for u in 0..n_users {
            x_lookup_offsets[u + 1] += x_lookup_offsets[u];
        }
        drop(entries);

        Self {
            cfg,
            a: vec![[0.0; 4]; n_users],
            b: vec![[0.0; 4]; n_items],
            mu_u,
            mu_m,
            x_tr,
            x_pr,
            by_user,
            by_item,
            x_lookup_items,
            x_lookup_vals,
            x_lookup_offsets,
        }
    }

    /// Polynomial features `[1, d, d², d³]` with `d = x − μ`.
    #[inline]
    fn phi(x: f32, mu: f32) -> [f32; 4] {
        let d = x - mu;
        let d2 = d * d;
        [1.0, d, d2, d2 * d]
    }

    /// Correction added to the base prediction `x` for the given (user, item).
    #[inline]
    fn correction(&self, u: usize, i: usize, x: f32) -> f32 {
        let a = &self.a[u];
        let b = &self.b[i];
        let phi_u = Self::phi(x, self.mu_u[u]);
        let phi_m = Self::phi(x, self.mu_m[i]);
        let mut corr = 0.0;
        for k in 0..4 {
            corr += a[k] * phi_u[k] + b[k] * phi_m[k];
        }
        corr
    }

    /// Run `n_als_iters` ALS sweeps with the given regularization. Resets `a`/`b`
    /// to zero at the start. Does not compute or report any RMSE.
    pub fn als_fit(
        &mut self,
        tr: &Dataset,
        reg_a: [(f32, f32); 4],
        reg_b: [(f32, f32); 4],
    ) {
        let n_users = tr.n_users;
        let n_items = tr.n_items;
        for u in 0..n_users { self.a[u] = [0.0; 4]; }
        for i in 0..n_items { self.b[i] = [0.0; 4]; }

        for _iter in 0..self.cfg.n_als_iters {
            // Item step: solve for b[i] given fixed a[u]
            {
                let a = &self.a;
                let by_item = &self.by_item;
                let tr_user_idxs = &tr.user_idxs;
                let tr_raw_ratings = &tr.raw_ratings;
                let x_tr = &self.x_tr;
                let mu_u = &self.mu_u;
                let mu_m = &self.mu_m;

                self.b.par_iter_mut().enumerate().for_each(|(i, b_i)| {
                    let obs = &by_item[i];
                    if obs.is_empty() {
                        *b_i = [0.0; 4];
                        return;
                    }
                    let n_m = obs.len() as f32;
                    let mu_m_i = mu_m[i];
                    let mut xtx = [0.0f64; 16];
                    let mut xty = [0.0f64; 4];

                    for &idx in obs {
                        let u = tr_user_idxs[idx] as usize;
                        let r = tr_raw_ratings[idx] as f32;
                        let x = x_tr[idx];
                        let a_u = &a[u];
                        let phi_u = NlppModel::phi(x, mu_u[u]);
                        let mut user_corr = 0.0f32;
                        for k in 0..4 { user_corr += a_u[k] * phi_u[k]; }
                        let y = (r - x - user_corr) as f64;

                        let phi = NlppModel::phi(x, mu_m_i);
                        for row in 0..4 {
                            let pr = phi[row] as f64;
                            xty[row] += pr * y;
                            for col in 0..4 {
                                xtx[row * 4 + col] += pr * phi[col] as f64;
                            }
                        }
                    }
                    for k in 0..4 {
                        xtx[k * 4 + k] += (reg_b[k].0 + reg_b[k].1 * n_m) as f64;
                    }
                    *b_i = solve4(&xtx, &xty);
                });
            }

            // User step: solve for a[u] given fixed b[i]
            {
                let b = &self.b;
                let by_user = &self.by_user;
                let tr_item_idxs = &tr.item_idxs;
                let tr_raw_ratings = &tr.raw_ratings;
                let x_tr = &self.x_tr;
                let mu_u = &self.mu_u;
                let mu_m = &self.mu_m;

                self.a.par_iter_mut().enumerate().for_each(|(u, a_u)| {
                    let obs = &by_user[u];
                    if obs.is_empty() {
                        *a_u = [0.0; 4];
                        return;
                    }
                    let n_u = obs.len() as f32;
                    let mu_u_val = mu_u[u];
                    let mut xtx = [0.0f64; 16];
                    let mut xty = [0.0f64; 4];

                    for &idx in obs {
                        let i = tr_item_idxs[idx] as usize;
                        let r = tr_raw_ratings[idx] as f32;
                        let x = x_tr[idx];
                        let b_i = &b[i];
                        let phi_m = NlppModel::phi(x, mu_m[i]);
                        let mut item_corr = 0.0f32;
                        for k in 0..4 { item_corr += b_i[k] * phi_m[k]; }
                        let y = (r - x - item_corr) as f64;

                        let phi = NlppModel::phi(x, mu_u_val);
                        for row in 0..4 {
                            let pr = phi[row] as f64;
                            xty[row] += pr * y;
                            for col in 0..4 {
                                xtx[row * 4 + col] += pr * phi[col] as f64;
                            }
                        }
                    }
                    for k in 0..4 {
                        xtx[k * 4 + k] += (reg_a[k].0 + reg_a[k].1 * n_u) as f64;
                    }
                    *a_u = solve4(&xtx, &xty);
                });
            }
        }
    }

    /// RMSE on the probe set's non-test entries. Requires full `Dataset` for residuals.
    /// Used by hyperparameter-search code (paropt); the standard fit2 flow gets RMSE
    /// reporting via the framework's `report_rmse`.
    pub fn probe_rmse(&self, pr: &Dataset) -> f64 {
        let mut sse = 0.0f64;
        let mut cnt = 0.0f64;
        for idx in 0..pr.n_ratings {
            if pr.is_test[idx] != 0 { continue; }
            let u = pr.user_idxs[idx] as usize;
            let i = pr.item_idxs[idx] as usize;
            let r = pr.residuals[idx];
            let x = self.x_pr[idx];
            let pred = x + self.correction(u, i, x);
            let err = (pred - r) as f64;
            sse += err * err;
            cnt += 1.0;
        }
        (sse / cnt).sqrt()
    }

    /// Run Nelder-Mead optimization in log-space over `(reg_a, reg_b)`. Each NM
    /// vertex evaluates one ALS fit + probe RMSE. Returns the optimal `(reg_a, reg_b)`.
    pub fn fit_with_nm(
        &mut self,
        tr: &Dataset,
        pr: &Dataset,
        n_steps: usize,
        init_scale: f64,
    ) -> ([(f32, f32); 4], [(f32, f32); 4]) {
        let mut x0 = [0.0f64; 16];
        for k in 0..4 {
            x0[2 * k]         = (self.cfg.reg_a[k].0 as f64).ln();
            x0[2 * k + 1]     = (self.cfg.reg_a[k].1 as f64).ln();
            x0[8 + 2 * k]     = (self.cfg.reg_b[k].0 as f64).ln();
            x0[8 + 2 * k + 1] = (self.cfg.reg_b[k].1 as f64).ln();
        }
        let decode = |v: &[f64]| -> ([(f32, f32); 4], [(f32, f32); 4]) {
            let mut ra = [(0.0f32, 0.0f32); 4];
            let mut rb = [(0.0f32, 0.0f32); 4];
            for k in 0..4 {
                ra[k] = (v[2 * k].exp() as f32, v[2 * k + 1].exp() as f32);
                rb[k] = (v[8 + 2 * k].exp() as f32, v[8 + 2 * k + 1].exp() as f32);
            }
            (ra, rb)
        };

        let n_init = x0.len() + 1;   // initial simplex has n+1 vertices
        let mut eval_count = 0usize;
        let (best_x, _) = nelder_mead(
            &x0, init_scale, n_steps,
            |v| {
                eval_count += 1;
                let t0 = std::time::Instant::now();
                let (ra, rb) = decode(v);
                self.als_fit(tr, ra, rb);
                let rmse = self.probe_rmse(pr);
                if eval_count <= n_init {
                    crate::teeln!(
                        "NM init eval {:2}/{}: RMSE {:.6}  ({:.1}s)",
                        eval_count, n_init, rmse, t0.elapsed().as_secs_f64(),
                    );
                }
                rmse
            },
            |step, x, f| {
                let (ra, rb) = decode(x);
                crate::teeln!(
                    "NM step {:3}: RMSE {:.6}  reg_a={:?}  reg_b={:?}",
                    step + 1, f, ra, rb,
                );
            },
        );

        let (best_ra, best_rb) = decode(&best_x);
        crate::teeln!("NM final reg_a={:?}", best_ra);
        crate::teeln!("NM final reg_b={:?}", best_rb);
        (best_ra, best_rb)
    }
}

impl Regressor for NlppModel {
    type Config = NlppConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        Self::new_inner(tr, pr.name, pr.user_idxs, pr.item_idxs, pr.n_ratings, cfg)
    }

    fn n_epochs(&self) -> usize { 1 }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, _epoch: usize) {
        let reg_a = self.cfg.reg_a;
        let reg_b = self.cfg.reg_b;
        self.als_fit(tr, reg_a, reg_b);
    }

    fn predict(&self, u: usize, i: usize, _day: i32) -> f32 {
        let start = self.x_lookup_offsets[u];
        let end = self.x_lookup_offsets[u + 1];
        let items = &self.x_lookup_items[start..end];
        let i32_i = i as i32;
        let pos = items.binary_search(&i32_i)
            .unwrap_or_else(|_| panic!("base prediction not found for u={} i={}", u, i));
        let x = self.x_lookup_vals[start + pos];
        x + self.correction(u, i, x)
    }
}
