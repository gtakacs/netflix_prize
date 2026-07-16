use crate::{Dataset, MaskedDataset, Regressor};
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Configuration for KNNF: item-item kNN where similarity is the cosine
/// between item factor vectors (loaded from an external `.npy` file produced
/// by some MF-family model), with a temporal decay applied at predict time.
#[derive(Clone, Copy, Debug)]
pub struct KnnfConfig {
    /// Path prefix for the factor file. The actual loaded file is
    /// `<factors>.<dataset>.npy` (e.g. "predsx/mf-10u.ifeat" + ".train.npy").
    pub factors: &'static str,
    /// Number of nearest neighbors retained per prediction (top-k by similarity).
    pub k: usize,
    /// Exponent applied to raw cosine similarity. >1 sharpens the weights.
    pub scaling: f32,
    /// Temporal decay rate: weight is divided by `(1 + tau * |day_pred - day_rated|)`.
    pub tau: f32,
}

impl KnnfConfig {
    /// Create config with given factor path prefix and default knnf parameters.
    pub fn with_factors(factors: &'static str) -> Self {
        Self { factors, k: 25, scaling: 1.0, tau: 0.025 }
    }
}

/// KNNF model: temporal item-item kNN over a precomputed factor space.
///
/// At predict time, for the target (user u, item i, day d) we visit each item
/// j that u has rated, score sim(i, j) discounted by time gap |d - day(u,j)|,
/// keep the top-k, and return a similarity-weighted mean of u's residuals
/// on those items. Training is just preprocessing — there are no SGD epochs.
pub struct KnnfModel {
    cfg: KnnfConfig,
    /// Per-user rated-item list: `ur[u][t] = (item, residual, day)`.
    ur: Vec<Vec<(usize, f32, f32)>>,
    /// Item factor matrix loaded from disk: shape (n_items, n_feat).
    factors: Array2<f32>,
    /// L2 norms of each row of `factors`, precomputed for cosine similarity.
    factor_norms: Vec<f32>,
}

impl KnnfModel {
    /// Cosine similarity between factor rows i and j, clamped at 0
    /// (negative similarities are treated as "no similarity").
    fn sim(&self, i: usize, j: usize) -> f32 {
        let ni = self.factor_norms[i];
        let nj = self.factor_norms[j];
        if ni == 0.0 || nj == 0.0 { return 0.0; }
        let phi = self.factors.row(i).dot(&self.factors.row(j)) / (ni * nj);
        phi.max(0.0)
    }

    /// Construct from a precomputed factor matrix (n_items × n_feat).
    ///
    /// Bypasses the file load — useful for tests and for callers that already
    /// have the factors in memory. The `Regressor::new` impl is a thin wrapper
    /// that reads the `.npy` file and delegates here.
    pub fn new_with_factors(tr: &Dataset, factors: Array2<f32>, cfg: KnnfConfig) -> Self {
        // Precompute L2 norms once so sim() is O(n_feat) per call instead of
        // re-scanning the rows.
        let n_items = factors.nrows();
        let mut factor_norms = vec![0.0_f32; n_items];
        for i in 0..n_items {
            factor_norms[i] = factors.row(i).dot(&factors.row(i)).sqrt();
        }

        // Per-user rated-item lists. tr is user-sorted, so contiguous slices of
        // length user_cnts[u] starting at the running offset belong to user u.
        let mut ur: Vec<Vec<(usize, f32, f32)>> = vec![Vec::new(); tr.n_users];
        let mut off: usize = 0;
        for u in 0..tr.n_users {
            let cnt = tr.user_cnts[u] as usize;
            let mut v = Vec::with_capacity(cnt);
            for t in 0..cnt {
                let idx = off + t;
                v.push((
                    tr.item_idxs[idx] as usize,
                    tr.residuals[idx],
                    tr.dates[idx] as f32,
                ));
            }
            ur[u] = v;
            off += cnt;
        }

        Self { cfg, ur, factors, factor_norms }
    }
}

impl Regressor for KnnfModel {
    type Config = KnnfConfig;

    fn new(tr: &Dataset, _pr: &MaskedDataset, cfg: Self::Config) -> Self {
        // Naming convention: <prefix>.<dataset_name>.npy — same factor matrix
        // computed once per dataset variant (train vs fulltrain etc.).
        let path = format!("{}.{}.npy", cfg.factors, tr.name);
        let factors: Array2<f32> = read_npy(&path).expect(&path);
        Self::new_with_factors(tr, factors, cfg)
    }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, _epoch: usize) {}

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let residuals = &self.ur[u];
        // Score each of u's rated items as a candidate neighbor of i.
        // Tuple layout: (sim, sim * residual, sim) — second is the numerator
        // contribution, third is the denominator (so the final answer is the
        // similarity-weighted mean of residuals).
        let mut neigh: Vec<(f32, f32, f32)> = Vec::with_capacity(residuals.len());
        for &(j, ruj, duj) in residuals.iter() {
            if j == i { continue; }
            let dt = (day as f32 - duj).abs();
            let raw_sim = self.sim(i, j);
            // Combined weight: sharpened cosine, then divided by temporal-gap
            // penalty. tau ≈ 0.025 ⇒ ~halving every ~40 days.
            let sim = raw_sim.powf(self.cfg.scaling) / (1.0 + self.cfg.tau * dt);
            if sim == 0.0 { continue; }
            neigh.push((sim, sim * ruj, sim));
        }
        if neigh.is_empty() { return 0.0; }

        // Keep only the top-k neighbors by similarity. select_nth_unstable_by
        // is O(n) average, faster than sort when we only need the top-k.
        let k = self.cfg.k;
        if neigh.len() > k {
            neigh.select_nth_unstable_by(k, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            neigh.truncate(k);
        }

        // Similarity-weighted mean: Σ(sim * residual) / Σ(sim).
        let mut num = 0.0;
        let mut den = 0.0;
        for &(_, contrib, w) in neigh.iter() {
            num += contrib;
            den += w;
        }

        num / den
    }
}
