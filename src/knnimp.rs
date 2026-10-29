// Item kNN with implicit neighbors (split_3): the `kimp[-tag]` stage of
// `s3 chain`, and its pair statistics, built by `s3 nbstats` (stat `imp`). The
// knobs named below are read from the environment by `s3 chain`.
//
// The usual kNN averages the user's own residuals r_uj over rated neighbors j,
// so only items with a known rating can be neighbors. Here a neighbor j only
// has to be *present* for the user: rated in the training set or merely listed
// in the prediction set (probe_3 / qual, rating unknown). Its contribution is a
// pair statistic learned from the other users instead of the user's r_uj:
//
//   c_ij = (S_ij - M_ij * mu_i) / (M_ij + KI_SHR)
//   S_ij = sum of r_vi over users v with i in their training ratings and j present
//   M_ij = number of such users;  mu_i = mean training residual of i
//
// i.e. how much better or worse than usual i is rated by those who also have j
// (KI_CENTER=0 drops the mu_i term). Prediction for (u, i): over j present for
// u (train + prediction set, j != i), w = sim(i, j)^KI_SCALE / (1 + KI_TAU *
// |day - day_j|), top KI_K by w, sum(w * c_ij) / (sum(w) + KI_DSHR). sim:
//   KI_SIM=supp  the lift of knn.rs's Support sim over KI_STAT0 (default
//                bin_wsupp, which also counts probe/qual presence), KI_WSHR shrinkage
//   KI_SIM=conf  M_ij / (M_ij + KI_WSHR)
//
// Statistics (`save_imp`, by name as `s3 nbstats` gets it):
//   bin_imp            the count matrix M -> <sim>/bin_impcnt.<ds>.npy
//   <base>_imp         the sum matrix S over <base>'s residuals -> <sim>/<base>_impsum.<ds>.npy
// Model (`KimpModel`, the chain stage <base>__kimp-<tag> on <base>'s residual,
// target W*base): reads bin_impcnt and <base>_impsum (KI_SUM overrides the stem).
// Each stats mode writes the train_3 (presence: + probe_3) and the fulltrain
// (presence: + qual) matrix; ~1.3 GB each.

use ndarray::Array2;
use ndarray_npy::{read_npy, write_npy};
use crate::{Dataset, MaskedDataset, Regressor, Split};
use parking_lot::Mutex;
use rayon::prelude::*;

/// Per-user (item, date) lists of a dataset that need not be user-sorted.
fn by_user(n_users: usize, users: &[i32], items: &[i32], dates: &[i16]) -> Vec<Vec<(usize, f32)>> {
    let mut v = vec![Vec::new(); n_users];
    for t in 0..users.len() {
        v[users[t] as usize].push((items[t] as usize, dates[t] as f32));
    }
    v
}

/// S (residual sums, `sum = true`) or M (counts) over tr x (tr + pr presence).
fn build_imp(tr: &Dataset, pr: &MaskedDataset, sum: bool) -> Array2<f32> {
    let n = tr.n_items;
    let pres = by_user(tr.n_users, pr.user_idxs.as_slice().unwrap(),
                       pr.item_idxs.as_slice().unwrap(), pr.dates.as_slice().unwrap());
    let mut starts = vec![0; tr.n_users + 1];
    for u in 0..tr.n_users { starts[u + 1] = starts[u] + tr.user_cnts[u] as usize; }
    let rows: Vec<Mutex<Vec<f32>>> = (0..n).map(|_| Mutex::new(vec![0.0; n])).collect();
    (0..tr.n_users).into_par_iter().for_each(|u| {
        let (a, b) = (starts[u], starts[u + 1]);
        let present: Vec<usize> = (a..b).map(|t| tr.item_idxs[t] as usize)
            .chain(pres[u].iter().map(|&(j, _)| j)).collect();
        for t in a..b {
            let x = if sum { tr.residuals[t] } else { 1.0 };
            let mut row = rows[tr.item_idxs[t] as usize].lock();
            for &j in &present { row[j] += x; }
        }
    });
    let mut out = Array2::<f32>::zeros((n, n));
    for (i, r) in rows.into_iter().enumerate() {
        out.row_mut(i).assign(&ndarray::ArrayView1::from(&r.into_inner()[..]));
    }
    out
}

/// Writes the train_3 and fulltrain matrix of `<rtg>_imp` (rtg = `bin` or a model).
pub fn save_imp(rtg: &str, split: Split) {
    std::fs::create_dir_all(split.sim_dir).unwrap();
    for (tr_set, pr_set) in [(split.tr, split.pr), (split.fulltrain_tr, split.fulltrain_pr)] {
        let sum = rtg != "bin";
        let tr = Dataset::load(tr_set, if sum { rtg } else { "rtg" }, split.preds_dir);
        let pr = Dataset::load(pr_set, "rtg", split.preds_dir);
        let m = build_imp(&tr, &MaskedDataset::from(&pr), sum);
        let path = format!("{}/{}_{}.{}.npy", split.sim_dir, rtg, if sum { "impsum" } else { "impcnt" }, tr_set);
        write_npy(&path, &m).unwrap();
        crate::teeln!("wrote {path}");
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KimpConfig {
    pub sum: &'static str,
    pub supp: Option<&'static str>,
    pub k: usize,
    pub shrinkage: f32,
    pub w_shrinkage: f32,
    pub d_shrinkage: f32,
    pub scaling: f32,
    pub tau: f32,
    pub center: bool,
    pub sim_dir: &'static str,
}

pub struct KimpModel {
    cfg: KimpConfig,
    cnt: Array2<f32>,
    sum: Array2<f32>,
    supp: Option<Array2<f32>>,
    mu: Vec<f32>,
    // per-user present items: [(item, date)], training ratings then prediction set
    pres: Vec<Vec<(usize, f32)>>,
}

impl KimpModel {
    fn weight(&self, i: usize, j: usize) -> f32 {
        match &self.supp {
            Some(s) => {
                let n = s[[i, j]];
                let den = s[[i, i]] * s[[j, j]];
                let phi = if den > 0.0 { n * 480189.0 / den } else { 0.0 };
                phi * n / (n + self.cfg.w_shrinkage)
            }
            None => { let m = self.cnt[[i, j]]; m / (m + self.cfg.w_shrinkage) }
        }
    }
}

impl Regressor for KimpModel {
    type Config = KimpConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let load = |stem: &str| -> Array2<f32> {
            read_npy(format!("{}/{}.{}.npy", cfg.sim_dir, stem, tr.name)).unwrap()
        };
        let cnt = load("bin_impcnt");
        let sum = load(cfg.sum);
        let supp = cfg.supp.map(load);
        let mut s = vec![0.0f64; tr.n_items];
        let mut c = vec![0.0f64; tr.n_items];
        for t in 0..tr.n_ratings {
            s[tr.item_idxs[t] as usize] += tr.residuals[t] as f64;
            c[tr.item_idxs[t] as usize] += 1.0;
        }
        let mu = (0..tr.n_items)
            .map(|i| if cfg.center && c[i] > 0.0 { (s[i] / c[i]) as f32 } else { 0.0 }).collect();
        let mut pres = by_user(tr.n_users, tr.user_idxs.as_slice().unwrap(),
                               tr.item_idxs.as_slice().unwrap(), tr.dates.as_slice().unwrap());
        let prs = by_user(tr.n_users, pr.user_idxs.as_slice().unwrap(),
                          pr.item_idxs.as_slice().unwrap(), pr.dates.as_slice().unwrap());
        for (v, p) in pres.iter_mut().zip(prs) { v.extend(p); }
        Self { cfg, cnt, sum, supp, mu, pres }
    }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, _epoch: usize) {}

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let mut neigh: Vec<(f32, f32)> = Vec::with_capacity(self.pres[u].len());
        for &(j, dj) in &self.pres[u] {
            if j == i { continue; }
            let m = self.cnt[[i, j]];
            if m == 0.0 { continue; }
            let w = self.weight(i, j).powf(self.cfg.scaling)
                / (1.0 + self.cfg.tau * (day as f32 - dj).abs());
            if w <= 0.0 { continue; }
            let cij = (self.sum[[i, j]] - m * self.mu[i]) / (m + self.cfg.shrinkage);
            neigh.push((w, cij));
        }
        if neigh.is_empty() { return 0.0; }
        let k = self.cfg.k;
        if neigh.len() > k {
            neigh.select_nth_unstable_by(k, |x, y| y.0.total_cmp(&x.0));
            neigh.truncate(k);
        }
        let (num, den) = neigh.iter().fold((0.0, 0.0), |(n, d), &(w, c)| (n + w * c, d + w));
        num / (den + self.cfg.d_shrinkage)
    }
}
