// BellKor "kNN on the residual" solve with the interaction matrices precomputed
// offline: reads sim/{target}_prod and sim/{baseline}_prod and mixes them as
// x*err + (1-x)*bias. Adaptive neighborhood size, then either a hand-written
// row-major Cholesky or an active-set NNLS for non-negative weights.
// Produced: every `<base>__knn3x` predictor in preds_old/.
// Frozen archive — see README.md; src/knn3.rs builds the same system on the fly.
use gravity::{Dataset, Regressor, calc_user_offsets};
use ndarray::Array2;
use ndarray_npy::read_npy;

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum SimType { Support, Cosine }

#[derive(Clone, Copy, Debug)]
struct Knn3xConfig {
    sim_type: SimType,
    sim_stat: &'static str,      // support matrix name, e.g. "bin_supp"
    threshold: f32,               // proximity threshold
    k_min: usize,
    k_max: usize,
    shrinkage: f32,               // neighbor selection shrinkage
    //tau: f32,                     // temporal decay: sim /= (1 + tau * |day_m - day_j|)

    target_prefix: &'static str,  // -> sim/{target_prefix}_prod.{ds}.npy
    baseline_prefix: &'static str,// -> sim/{baseline_prefix}_prod.{ds}.npy

    reg: f32,                     // system regularization
    x: f32,                       // mixing: x*err + (1-x)*bias
    nonneg: bool,                 // true: NNLS (w >= 0), false: Cholesky solve
}

/// Per-user sorted error+date lookup
struct UserErrors {
    starts: Vec<usize>,
    movies: Vec<i32>,
    errors: Vec<f32>,
    dates: Vec<i16>,
}

impl UserErrors {
    fn build(tr: &Dataset, user_offsets: &[usize], errors: &[f32]) -> Self {
        let n_u = tr.n_users;
        let mut starts = vec![0usize; n_u + 1];
        let mut movies = Vec::with_capacity(tr.n_ratings);
        let mut errs = Vec::with_capacity(tr.n_ratings);
        let mut dates = Vec::with_capacity(tr.n_ratings);

        for u in 0..n_u {
            starts[u] = movies.len();
            let s = user_offsets[u];
            let e = user_offsets[u + 1];
            let mut pairs: Vec<(i32, f32, i16)> = (s..e)
                .filter(|&idx| tr.is_test[idx] == 0)
                .map(|idx| (tr.item_idxs[idx], errors[idx], tr.dates[idx]))
                .collect();
            pairs.sort_unstable_by_key(|p| p.0);
            for (m, err, d) in pairs {
                movies.push(m);
                errs.push(err);
                dates.push(d);
            }
        }
        starts[n_u] = movies.len();
        Self { starts, movies, errors: errs, dates }
    }

}

/// Cholesky decomposition and solve (row-major, in-place)
fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        for k in 0..j { a[j * n + j] -= a[j * n + k] * a[j * n + k]; }
        if a[j * n + j] <= 1e-15 { return false; }
        a[j * n + j] = a[j * n + j].sqrt();
        let ajj = a[j * n + j];
        for i in (j + 1)..n {
            for k in 0..j { a[i * n + j] -= a[i * n + k] * a[j * n + k]; }
            a[i * n + j] /= ajj;
        }
    }
    for i in 0..n {
        for k in 0..i { b[i] -= a[i * n + k] * b[k]; }
        b[i] /= a[i * n + i];
    }
    for i in (0..n).rev() {
        for k in (i + 1)..n { b[i] -= a[k * n + i] * b[k]; }
        b[i] /= a[i * n + i];
    }
    true
}

/// Solve M w = rhs subject to w >= 0 (active set NNLS)
fn solve_nnls(mat: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
    let mut active = vec![true; n];
    for _ in 0..n {
        let idx: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        let k = idx.len();
        if k == 0 { return vec![0.0; n]; }

        let mut a = vec![0.0f64; k * k];
        let mut b = vec![0.0f64; k];
        for (ri, &i) in idx.iter().enumerate() {
            b[ri] = rhs[i];
            for (rj, &j) in idx.iter().enumerate() {
                a[ri * k + rj] = mat[i * n + j];
            }
        }

        if !cholesky_solve(&mut a, &mut b, k) { return vec![0.0; n]; }

        let mut worst_ri = 0usize;
        let mut worst_val = 0.0f64;
        for (ri, &val) in b.iter().enumerate() {
            if val < worst_val { worst_ri = ri; worst_val = val; }
        }

        if worst_val >= 0.0 {
            let mut result = vec![0.0; n];
            for (ri, &i) in idx.iter().enumerate() { result[i] = b[ri]; }
            return result;
        }
        active[idx[worst_ri]] = false;
    }
    vec![0.0; n]
}

struct Knn3xModel {
    stat_supp: Array2<f32>,
    stat_err: Array2<f32>,
    stat_bias: Array2<f32>,
    uerr: UserErrors,
    cfg: Knn3xConfig,
}

impl Regressor for Knn3xModel {
    type Config = Knn3xConfig;

    fn new(tr: &Dataset, _pr: &Dataset, cfg: Self::Config) -> Self {
        let ds = &tr.name;
        println!("  KNN3X: loading stat matrices...");
        let stat_supp: Array2<f32> = read_npy(format!("sim/{}.{}.npy", cfg.sim_stat, ds)).unwrap();
        let stat_err: Array2<f32> = read_npy(format!("sim/{}_prod.{}.npy", cfg.target_prefix, ds)).unwrap();
        let stat_bias: Array2<f32> = read_npy(format!("sim/{}_prod.{}.npy", cfg.baseline_prefix, ds)).unwrap();
        println!("  KNN3X: matrices loaded.");

        let user_offsets = calc_user_offsets(tr);
        let user_offsets_sl: Vec<usize> = user_offsets.to_vec();
        let errors: Vec<f32> = tr.residuals.to_vec();
        let uerr = UserErrors::build(tr, &user_offsets_sl, &errors);

        Self { stat_supp, stat_err, stat_bias, uerr, cfg }
    }

    fn n_epochs(&self) -> usize { 0 }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &Dataset, _epoch: usize) {}

    fn predict(&self, u: usize, i: usize, _day: i32) -> f32 {
        let m = i;
        let n_m = self.stat_supp[[m, m]];
        if n_m < 1.0 { return 0.0; }

        // Only consider movies the user has rated
        let us = self.uerr.starts[u];
        let ue = self.uerr.starts[u + 1];
        if us == ue { return 0.0; }

        let mut cands: Vec<(usize, f32, f32, i16)> = Vec::new(); // (j, sim, error, date)
        for idx in us..ue {
            let j = self.uerr.movies[idx] as usize;
            if j == m { continue; }
            let n_mj = self.stat_supp[[m, j]];
            if n_mj < 1.0 { continue; }

            let sim = match self.cfg.sim_type {
                SimType::Support => {
                    let n_j = self.stat_supp[[j, j]];
                    if n_j < 1.0 { continue; }
                    let phi = n_mj * 480189.0 / (n_m * n_j);
                    phi * n_mj / (n_mj + self.cfg.shrinkage)
                }
                SimType::Cosine => {
                    let den = (self.stat_err[[m, m]] * self.stat_err[[j, j]]).sqrt();
                    if den <= 0.0 { continue; }
                    let cos = self.stat_err[[m, j]] / den;
                    cos.abs() * n_mj / (n_mj + self.cfg.shrinkage)
                }
            };

            if sim > 0.0 {
                cands.push((j, sim, self.uerr.errors[idx], self.uerr.dates[idx]));
            }
        }
        cands.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let above = cands.iter().filter(|(_, p, _, _)| *p >= self.cfg.threshold).count();
        let k = above.max(self.cfg.k_min).min(self.cfg.k_max).min(cands.len());
        cands.truncate(k);

        if cands.is_empty() { return 0.0; }

        let big_k = cands.len();
        let x = self.cfg.x as f64;
        let reg = self.cfg.reg as f64;

        // Build system from precomputed matrices
        let mut sys = vec![0.0f64; big_k * big_k];
        let mut sys_rhs = vec![0.0f64; big_k];

        for a in 0..big_k {
            let ja = cands[a].0;
            let d_err = self.stat_err[[ja, m]] as f64;
            let d_bias = self.stat_bias[[ja, m]] as f64;
            sys_rhs[a] = x * d_err + (1.0 - x) * d_bias;

            for b in 0..big_k {
                let jb = cands[b].0;
                let c_err = self.stat_err[[ja, jb]] as f64;
                let c_bias = self.stat_bias[[ja, jb]] as f64;
                sys[a * big_k + b] = x * c_err + (1.0 - x) * c_bias;
            }
            sys[a * big_k + a] += reg;
        }

        let weights = if self.cfg.nonneg {
            solve_nnls(&sys, &sys_rhs, big_k)
        } else {
            if cholesky_solve(&mut sys, &mut sys_rhs, big_k) {
                sys_rhs
            } else {
                vec![0.0; big_k]
            }
        };

        // Apply weights with temporal decay
        let mut pred = 0.0f32;
        for (ni, &(_, _, err, _d)) in cands.iter().enumerate() {
            let w = weights[ni] as f32;
            //if w.abs() == 0.0 { continue; }
            pred += w * err;
        }
        pred
    }
}

fn main() {
    let target = "tsvdx4-180f__nlpp";
    let baseline = "ge14-2";
    let cfg = Knn3xConfig {
        sim_type: SimType::Support,
        sim_stat: "bin_supp",
        threshold: 0.25,
        k_min: 10,
        k_max: 40,
        shrinkage: 20000.0,
        target_prefix: target,
        baseline_prefix: baseline,
        reg: 0.01,
        x: 0.8,
        nonneg: true,
    };
    // Snapshot of one run — `target`/`baseline` were edited by hand per experiment.
    gravity::fit2::<Knn3xModel>(
        cfg,
        format!("{target}").leak(),          // target (residual of `target`)
        format!("{target}__knn3x").leak(),   // model_name
        false,  // save_subscores
        false,  // save_train
        false,  // save_probe_each_epoch
    );
}
