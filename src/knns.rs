use crate::{Dataset, N_USERS, MaskedDataset, Regressor};
use crate::nbstats::build_bin_nbstats;
use ndarray::Array2;
use ndarray_npy::read_npy;

/// Where the support (co-occurrence) matrix comes from.
#[derive(Clone, Copy, Debug)]
pub enum SuppSource {
    /// Load a precomputed `.npy` matrix from this fixed path.
    Path(&'static str),
    /// Load from `sim/bin_supp.<tr.name>.npy` (path follows the standard naming pattern).
    PathPattern,
    /// Compute on the fly from `tr ∪ pr` via `build_bin_nbstats(.., "supp")`.
    Compute,
}

/// Configuration for KNNS: item-item kNN where similarity is a shrunken
/// "lift over independence" computed from a binary co-occurrence support
/// matrix, with a temporal decay applied at predict time.
#[derive(Clone, Copy, Debug)]
pub struct KnnsConfig {
    /// Number of nearest neighbors retained per prediction (top-k by similarity).
    pub k: usize,
    /// Support shrinkage: low-co-occurrence pairs are pulled toward 0
    /// via `phi * n / (n + shrinkage)` (Bell-Koren style).
    pub shrinkage: f32,
    /// Exponent applied to the (already shrunken) similarity. >1 sharpens the weights.
    pub scaling: f32,
    /// Temporal decay rate: weight is divided by `(1 + tau * |day_pred - day_rated|)`.
    pub tau: f32,
    /// How to obtain the (n_items × n_items) support matrix.
    pub supp_source: SuppSource,
}

impl Default for KnnsConfig {
    fn default() -> Self {
        Self {
            k: 25, shrinkage: 100.0, scaling: 1.0, tau: 0.025,
            supp_source: SuppSource::Path("sim/bin_supp.fulltrain.npy"),
        }
    }
}

/// KNNS model: temporal item-item kNN over a precomputed support matrix.
///
/// At predict time, for the target (user u, item i, day d) we visit each item
/// j that u has rated, score sim(i, j) discounted by time gap |d - day(u,j)|,
/// keep the top-k, and return a similarity-weighted mean of u's residuals
/// on those items. Training is just preprocessing — there are no SGD epochs.
pub struct KnnsModel {
    cfg: KnnsConfig,
    /// Co-occurrence support matrix loaded from disk: shape (n_items, n_items).
    /// `supp[i,j]` = number of users who co-rated items i and j.
    /// `supp[i,i]` = total support (number of users who rated item i).
    supp: Array2<f32>,
    /// Per-user rated-item list: `ur[u][t] = (item, residual, day)`.
    ur: Vec<Vec<(usize, f32, f32)>>,
}

impl KnnsModel {
    /// Shrunk lift-over-independence similarity between items i and j.
    ///
    /// phi  = lift over independence: P(co-rated) / [P(rated i) * P(rated j)],
    ///        with N=N_USERS (Netflix Prize user count) plugged in.
    /// sim  = phi shrunk toward 0 for low-support pairs (Bell-Koren).
    fn sim(&self, i: usize, j: usize) -> f32 {
        let n = self.supp[[i, j]];
        let den = self.supp[[i, i]] * self.supp[[j, j]];
        let phi = if den > 0.0 { n * N_USERS as f32 / den } else { 0.0 };
        phi * n / (n + self.cfg.shrinkage)
    }
}

impl Regressor for KnnsModel {
    type Config = KnnsConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let supp: Array2<f32> = match cfg.supp_source {
            SuppSource::Path(p) => read_npy(p).expect(p),
            SuppSource::PathPattern => {
                let path = format!("sim/bin_supp.{}.npy", tr.name);
                read_npy(&path).expect(&path)
            }
            SuppSource::Compute => build_bin_nbstats(tr, pr, "supp"),
        };

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

        Self { cfg, supp, ur }
    }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, _epoch: usize) {}

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let residuals = &self.ur[u];
        // Score each of u's rated items as a candidate neighbor of i.
        // Tuple layout: (sim, residual) — sim is used both as weight and
        // for top-k selection.
        let mut neigh: Vec<(f32, f32)> = Vec::with_capacity(residuals.len());
        for &(j, ruj, duj) in residuals.iter() {
            if j == i { continue; }
            let dt = (day as f32 - duj).abs();
            let raw_sim = self.sim(i, j);
            // Combined weight: sharpened similarity, then divided by temporal-gap
            // penalty. tau ≈ 0.025 ⇒ ~halving every ~40 days.
            let sim = raw_sim.powf(self.cfg.scaling) / (1.0 + self.cfg.tau * dt);
            if sim == 0.0 { continue; }
            neigh.push((sim, ruj));
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
        for &(sim, ruj) in neigh.iter() {
            num += sim * ruj;
            den += sim;
        }

        num / den
    }
}
