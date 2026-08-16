// Item-item kNN over precomputed sim/*.npy stat matrices, with seven similarity
// types behind one SimType enum (support/cosine/MSE, plus four over externally
// supplied factor vectors). Optional temporal decay, similarity threshold, power
// scaling, and a ridge-regularised Cholesky regression over the k neighbors that
// falls back to a weighted average.
// Produced: knn-25w and every `<base>__knn-*` predictor in preds_old/ (30 in all,
// listed in README.md).  Frozen archive; the current version is src/knn.rs.
use gravity::{Dataset, Regressor};
use ndarray::{Array2};
use ndarray_npy::read_npy;
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Copy, Debug)]
pub enum SimType {
    Support,
    Cosine,
    Mse,
    FactorCosine,
    FactorPearson,
    FactorMse,
    FactorDot,
}

#[derive(Clone, Copy, Debug)]
pub struct ItemItemConfig {
    pub stat0: Option<&'static str>,
    pub stat1: Option<&'static str>,
    pub stat2: Option<&'static str>,
    pub factors: Option<&'static str>,
    pub sim_type: SimType,

    pub k: usize,
    pub sim_threshold: Option<f32>,
    pub shrinkage: f32,
    pub scaling: f32,
    pub tau: f32,
    pub use_stat1: bool,
    pub regression: bool,
    pub regression_lambda: f32,
    pub preds_dir: &'static str,
}

pub struct ItemItemModel {
    cfg: ItemItemConfig,

    // similarity stats (precomputed)
    stat0: Option<Array2<f32>>,
    stat1: Option<Array2<f32>>,
    stat2: Option<Array2<f32>>,

    // per-user residuals from training set: [(item, rating, date)]
    ur: Vec<Vec<(usize, f32, f32)>>,

    // means squared error averaged over all items pairs
    mse: f32,

    // factor-based similarity
    factors: Option<Array2<f32>>,
    factor_norms: Vec<f32>,
    factor_means: Vec<f32>,
    factor_cnorms: Vec<f32>,
}

impl ItemItemModel {
    // stat0: bin_[sum|wsum]
    fn support_sim(&self, i: usize, j: usize) -> f32 {
        let stat0 = self.stat0.as_ref().unwrap();
        if i == j { return 1.0; }
        let n = stat0[[i, j]];
        let den = stat0[[i, i]] * stat0[[j, j]];
        let phi = if den > 0.0 { n * 480189.0 / den } else { 0.0 };
        phi * n / (n + self.cfg.shrinkage)
    }

    // stat0: rtg_supp, stat2: *_prod
    fn cosine_sim(&self, i: usize, j: usize) -> f32 {
        let stat0 = self.stat0.as_ref().unwrap();
        let stat2 = self.stat2.as_ref().unwrap();
        if i == j { return 1.0; }
        let n = stat0[[i, j]];
        if n < 2.0 { return 0.0; }
        let den = (stat2[[i, i]] * stat2[[j, j]]).sqrt();
        let phi = if den > 0.0 { stat2[[i, j]] / den } else { 0.0 };
        phi.abs() * n / (n + self.cfg.shrinkage) // abs()!
    }

    // stat0: rtg_supp, stat2: *_diff2
    fn mse_sim(&self, i: usize, j: usize) -> f32 {
        let stat0 = self.stat0.as_ref().unwrap();
        let stat2 = self.stat2.as_ref().unwrap();
        let n = stat0[[i, j]];
        if n < 2.0 { return 0.0; }
        let mse = stat2[[i, j]] / n;
        let shr = self.cfg.shrinkage;
        (n + shr) / (n * mse + shr * self.mse)
    }

    fn factor_cosine_sim(&self, i: usize, j: usize) -> f32 {
        let ni = self.factor_norms[i];
        let nj = self.factor_norms[j];
        if ni == 0.0 || nj == 0.0 { return 0.0; }
        let factors = self.factors.as_ref().unwrap();
        let phi = factors.row(i).dot(&factors.row(j)) / (ni * nj);
        phi.max(0.0)
    }

    fn factor_pearson_sim(&self, i: usize, j: usize) -> f32 {
        let ci = self.factor_cnorms[i];
        let cj = self.factor_cnorms[j];
        if ci == 0.0 || cj == 0.0 { return 0.0; }
        let factors = self.factors.as_ref().unwrap();
        let d = factors.ncols() as f32;
        let dot = factors.row(i).dot(&factors.row(j));
        let phi = (dot - d * self.factor_means[i] * self.factor_means[j]) / (ci * cj);
        phi.max(0.0)
    }

    fn factor_mse_sim(&self, i: usize, j: usize) -> f32 {
        let factors = self.factors.as_ref().unwrap();
        let d = factors.ncols() as f32;
        let diff = &factors.row(i) - &factors.row(j);
        let mse = diff.dot(&diff) / d;
        1.0 / (1.0 + mse)
    }

    fn factor_dot_sim(&self, i: usize, j: usize) -> f32 {
        let factors = self.factors.as_ref().unwrap();
        let phi = factors.row(i).dot(&factors.row(j));
        phi.max(0.0)
    }

    fn sim(&self, i: usize, j: usize) -> f32 {
        match self.cfg.sim_type {
            SimType::Support => self.support_sim(i, j),
            SimType::Cosine => self.cosine_sim(i, j),
            SimType::Mse => self.mse_sim(i, j),
            SimType::FactorCosine => self.factor_cosine_sim(i, j),
            SimType::FactorPearson => self.factor_pearson_sim(i, j),
            SimType::FactorMse => self.factor_mse_sim(i, j),
            SimType::FactorDot => self.factor_dot_sim(i, j),
        }
    }
}

impl Regressor for ItemItemModel {
    type Config = ItemItemConfig;

    fn new(tr: &Dataset, _pr: &Dataset, cfg: Self::Config) -> Self {
        // Load similarity stats
        let ds = &tr.name;

        let stat0 = cfg.stat0.map(|stat| read_npy(format!("sim/{}.{}.npy", stat, ds)).unwrap());
        let stat1 = cfg.stat1.map(|stat| read_npy(format!("sim/{}.{}.npy", stat, ds)).unwrap());
        let stat2 = cfg.stat2.map(|stat| read_npy(format!("sim/{}.{}.npy", stat, ds)).unwrap());

        let mse = if let SimType::Mse = cfg.sim_type {
            let stat0_uw: &Array2<f32> = stat0.as_ref().unwrap();
            let stat2_uw: &Array2<f32> = stat2.as_ref().unwrap();
            let mut sse = 0.0;
            let mut cnt = 0.0;
            for i in 0..tr.n_items {
                for j in (i + 1)..tr.n_items {
                    cnt += stat0_uw[[i, j]] as f64;
                    sse += stat2_uw[[i, j]] as f64;
                }
            }
            (sse / cnt) as f32
        }
        else { 0.0 };

        // Precompute per-user rating lists from training data
        let mut ur: Vec<Vec<(usize, f32, f32)>> = vec![Vec::new(); tr.n_users];
        let mut off: usize = 0;
        for u in 0..tr.n_users {
            let cnt = tr.user_cnts[u] as usize;
            let mut v = Vec::with_capacity(cnt);
            for t in 0..cnt {
                let idx = off + t;
                let it = tr.item_idxs[idx] as usize;
                let r = tr.residuals[idx];
                let d = tr.dates[idx] as f32;
                v.push((it, r, d));
            }
            ur[u] = v;
            off += cnt;
        }

        // Load factor vectors if configured
        let (factors, factor_norms, factor_means, factor_cnorms) = if let Some(ff) = cfg.factors {
            let fmat: Array2<f32> = read_npy(format!("{}/{}.{}.npy", cfg.preds_dir, ff, ds)).unwrap();
            let n_items = fmat.nrows();
            let d = fmat.ncols() as f32;
            let mut norms = vec![0.0_f32; n_items];
            let mut means = vec![0.0_f32; n_items];
            let mut cnorms = vec![0.0_f32; n_items];
            for i in 0..n_items {
                let row = fmat.row(i);
                let dot = row.dot(&row);
                norms[i] = dot.sqrt();
                let mu = row.sum() / d;
                means[i] = mu;
                // ||v - mu||^2 = ||v||^2 - 2*mu*sum(v) + D*mu^2 = dot - D*mu^2
                cnorms[i] = (dot - d * mu * mu).max(0.0).sqrt();
            }
            (Some(fmat), norms, means, cnorms)
        } else {
            (None, Vec::new(), Vec::new(), Vec::new())
        };

        Self {cfg, stat0, stat1, stat2, ur, mse, factors, factor_norms, factor_means, factor_cnorms}
    }

    fn n_epochs(&self) -> usize { 0 }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &Dataset, _epoch: usize) { }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let residuals = &self.ur[u];
        let mut neigh: Vec<(f32, f32, f32, usize)> = Vec::with_capacity(residuals.len());
        for &(j, ruj, duj) in residuals.iter() {
            if j == i { continue; }
            let dt = (day as f32 - duj).abs();
            let raw_sim = self.sim(i, j);

            if let Some(th) = self.cfg.sim_threshold {
                if raw_sim < th { continue; }
            }
            let sim = raw_sim.powf(self.cfg.scaling) / (1.0 + self.cfg.tau * dt);
            if sim == 0.0 { continue; }
            neigh.push((sim.abs(), sim, ruj, j));
        }
        if neigh.is_empty() { return 0.0; }

        let k = self.cfg.k;
        if neigh.len() > k {
            neigh.select_nth_unstable_by(k, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            neigh.truncate(k);
        }

        // Regression path: Cholesky solve, fall through to weighted avg on failure
        if self.cfg.regression {
            let kk = neigh.len();
            let mut xtx = vec![0.0_f64; kk * kk];
            let mut xty = vec![0.0_f64; kk];

            let stat0 = self.stat0.as_ref().unwrap();
            let stat2 = self.stat2.as_ref().unwrap();
            let shr = self.cfg.shrinkage;
            for a in 0..kk {
                let ja = neigh[a].3;
                xty[a] = (stat2[[ja, i]] / (stat0[[ja, i]] + shr)) as f64;
                for b in 0..kk {
                    let jb = neigh[b].3;
                    xtx[a * kk + b] = (stat2[[ja, jb]] / (stat0[[ja, jb]] + shr)) as f64;
                }
                xtx[a * kk + a] += self.cfg.regression_lambda as f64;
            }

            let a_mat = DMatrix::<f64>::from_row_slice(kk, kk, &xtx);
            let b_vec = DVector::<f64>::from_row_slice(&xty);

            if let Some(chol) = a_mat.cholesky() {
                let w = chol.solve(&b_vec);
                let mut pred = 0.0_f32;
                for a in 0..kk {
                    pred += w[a] as f32 * neigh[a].2;
                }
                return pred;
            }
            // Cholesky failed — fall through to weighted average
        }

        let mut num = 0.0;
        let mut den = 0.0;
        for &(_, sim, ruj, j) in neigh.iter() {
            if self.cfg.use_stat1 {
                let stat0 = self.stat0.as_ref().unwrap();
                let stat1 = self.stat1.as_ref().unwrap();
                let diff = stat1[[i, j]] / (stat0[[i, j]] + self.cfg.shrinkage);
                num += sim * diff;
            }
            else {
                num += sim * ruj;
            }
            den += sim.abs();
        }

        num / den
    }
}

fn main() {
    // let base = "rbmx-700";
    // let base = "fbias";
    // let base = "tsvd-2048";
    // let base = "rbm-32";
    // let base = "tsvdxx-2048";
    // let base = "als8-8";
    // let base = "0.8*tsvdx4-180f + 0.2*rbmx-700";
    let base = "tsvdx4-4800lm";

    let cfg = ItemItemConfig {
        stat0: Some("rtg_supp"),
        stat1: Some("tsvdx4-4800lm_diff1"),
        stat2: None,
        factors: None,
        sim_type: SimType::Support,
        k: 15,
        sim_threshold: None,
        shrinkage: 100.0,
        scaling: 3.5,
        tau: 0.0,
        use_stat1: true,
        regression: false,
        regression_lambda: 0.1,
        preds_dir: "predsx",
    };

    // Snapshot of one run — `base`, the stat names and the `__knn-*` suffix were
    // edited by hand per experiment. `fit2x` was a wider variant of `fit2`; its six
    // trailing flags no longer have recoverable names (the function predates the
    // repo's git history), so they are left unannotated on purpose.
    gravity::fit2x::<ItemItemModel>(
        cfg,
        format!("{base}").leak(),            // target (residual of `base`)
        format!("{base}__knn-d3").leak(),    // model_name
        false,
        false,
        false,
        false,
        false,
        false,
    );
}
