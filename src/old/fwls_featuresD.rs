// FWLS voting-feature generator, phase D — 3 item-item similarity aggregates over the
// user's rated set: correlation sum, top-20% concentration, and max similarity.
// Frozen archive — see README.md; superseded by src/vfeat1.rs. Writes
// features/fwls_D{k}.{set}.npy and the shipped files were renamed by hand into
// preds_old/fwls/D{k}.*.npy, the layout the current blenders load.
//
// Phase D features (3 features):
// - f10_log_sim_sum (log of sum of positive item correlations for user's rated items)
// - f14_sim_top20_pct (% of correlation sum from top 20% most correlated items)
// - f20_max_sim (max correlation of movie with any other movie)

use gravity::fwls_common::{FwlsBase, build_user_items, compute_sim_matrix};
use gravity::Dataset;
use indicatif::ParallelProgressIterator;
use ndarray::{Array1, Array2};
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;

fn save_features(features: &[Array1<f32>], set_name: &str) {
    use std::io::Write;
    for (k, feat) in features.iter().enumerate() {
        let path = format!("features/fwls_D{}.{}.npy", k + 1, set_name);
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        feat.write_npy(&mut writer).unwrap();
        writer.flush().unwrap();
        println!("  Saved: {}", path);
    }
}

const N_FEATURES: usize = 3;

/// Phase D feature indices
const F_LOG_SIM_SUM: usize = 0;
const F_SIM_TOP20_PCT: usize = 1;
const F_MAX_SIM: usize = 2;

pub struct FwlsFeaturesD {
    n_users: usize,
    n_items: usize,
    sim_matrix: Array2<f32>,
    max_sim: Array1<f32>,
    user_items: Vec<Vec<usize>>,
}

impl FwlsFeaturesD {
    pub fn new(ds: &Dataset) -> Self {
        println!("Computing FWLS Phase D statistics...");
        let base = FwlsBase::new(ds);

        // Build user item lists
        let user_items = build_user_items(ds, &base.user_starts);

        // Compute similarity matrix
        let (sim_matrix, max_sim) = compute_sim_matrix(ds, &base);

        Self {
            n_users: base.n_users,
            n_items: base.n_items,
            sim_matrix,
            max_sim,
            user_items,
        }
    }

    #[inline]
    pub fn compute(&self, u: usize, i: usize, _day: i16) -> [f32; N_FEATURES] {
        let mut f = [0.0_f32; N_FEATURES];

        // Features 10 and 14: collect positive similarities with user's rated items
        let user_items = &self.user_items[u];
        let mut pos_sims: Vec<f32> = Vec::with_capacity(user_items.len());
        for &j in user_items {
            if j == i { continue; }
            let sim = self.sim_matrix[[i, j]];
            if sim > 0.0 {
                pos_sims.push(sim);
            }
        }
        let total_pos_sim: f32 = pos_sims.iter().sum();

        // Feature 10: log(1 + sum of positive correlations)
        f[F_LOG_SIM_SUM] = (1.0 + total_pos_sim).ln();

        // Feature 14: % of correlation sum from top 20% most correlated items
        if !pos_sims.is_empty() {
            pos_sims.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let mut top_sum: f32 = 0.0;
            for k in 0..pos_sims.len() {
                if k * 5 >= pos_sims.len() { break; }
                top_sum += pos_sims[k];
            }
            f[F_SIM_TOP20_PCT] = top_sum / total_pos_sim;
        }

        // Feature 20: max correlation of movie with any other movie
        f[F_MAX_SIM] = self.max_sim[i];

        f
    }

    pub fn compute_all(&self, ds: &Dataset) -> Vec<Array1<f32>> {
        println!("Computing {} Phase D features for {} ratings...", N_FEATURES, ds.n_ratings);

        let mut features: Vec<Array1<f32>> = (0..N_FEATURES)
            .map(|_| Array1::<f32>::zeros(ds.n_ratings))
            .collect();

        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let ptrs: Vec<SendPtr> = features.iter_mut()
            .map(|arr| SendPtr(arr.as_slice_mut().unwrap().as_mut_ptr()))
            .collect();

        (0..ds.n_ratings).into_par_iter()
            .progress_count(ds.n_ratings as u64)
            .for_each(|idx| {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;
                let day = ds.dates[idx];
                let f = self.compute(u, i, day);
                for k in 0..N_FEATURES {
                    unsafe { *ptrs[k].0.add(idx) = f[k]; }
                }
            });

        features
    }

    pub fn feature_names() -> [&'static str; N_FEATURES] {
        [
            "f10_log_sim_sum",
            "f14_sim_top20_pct",
            "f20_max_sim",
        ]
    }

    pub fn print_summary(&self) {
        println!("\nFWLS Phase D Statistics ({} features):", N_FEATURES);
        println!("  Users: {}", self.n_users);
        println!("  Items: {}", self.n_items);

        // Sample max similarities
        let avg_max_sim: f32 = self.max_sim.iter().sum::<f32>() / self.n_items as f32;
        println!("  Average max similarity: {:.4}", avg_max_sim);

        println!("\nFeatures:");
        for (i, name) in Self::feature_names().iter().enumerate() {
            println!("  [{:2}] {}", i, name);
        }
    }
}

fn main() {
    println!("=== FWLS Meta-Features Generator (Phase D, {} features) ===\n", N_FEATURES);

    std::fs::create_dir_all("features").unwrap();

    // Process train -> probe
    println!("Processing: train -> probe");
    let train = Dataset::load("train", "rtg", "preds");
    let probe = Dataset::load("probe", "rtg", "preds");

    let fwls = FwlsFeaturesD::new(&train);
    fwls.print_summary();

    let probe_features = fwls.compute_all(&probe);
    save_features(&probe_features, "probe");

    // Process fulltrain -> qual
    println!("\nProcessing: fulltrain -> qual");
    let fulltrain = Dataset::load("fulltrain", "rtg", "preds");
    let qual = Dataset::load("qual", "rtg", "preds");

    let fwls_full = FwlsFeaturesD::new(&fulltrain);

    let qual_features = fwls_full.compute_all(&qual);
    save_features(&qual_features, "qual");

    println!("\n=== Done ===");
}
