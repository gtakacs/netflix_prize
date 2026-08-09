// FWLS voting-feature generator, phase B — 2 features built from Bayesian-shrunk
// user and movie means (shrinkage constant BAYESIAN_K).
// Frozen archive — see README.md; superseded by src/vfeat1.rs. Writes
// features/fwls_B{k}.{set}.npy and the shipped files were renamed by hand into
// preds_old/fwls/B{k}.*.npy, the layout the current blenders load.
//
// Phase B features (2 features):
// - f05_bayesian_movie_minus_user (bayesian movie mean - bayesian user mean)
// - f07_bayesian_user_mean (bayesian-shrunk user mean)

use gravity::fwls_common::{FwlsBase, BAYESIAN_K};
use gravity::Dataset;
use indicatif::ParallelProgressIterator;
use ndarray::Array1;
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;

const N_FEATURES: usize = 2;

/// Phase B feature indices
const F_BAYESIAN_MOVIE_MINUS_USER: usize = 0;
const F_BAYESIAN_USER_MEAN: usize = 1;

pub struct FwlsFeaturesB {
    base: FwlsBase,
    user_bayesian_mean: Array1<f32>,
    movie_bayesian_mean: Array1<f32>,
}

impl FwlsFeaturesB {
    pub fn new(ds: &Dataset) -> Self {
        println!("Computing FWLS Phase B statistics...");
        let base = FwlsBase::new(ds);

        // Bayesian-shrunk user means
        let mut user_bayesian_mean = Array1::<f32>::zeros(base.n_users);
        for u in 0..base.n_users {
            let cnt = base.user_rating_counts[u] as f64;
            user_bayesian_mean[u] = ((base.user_sum_ratings[u] + BAYESIAN_K * base.global_user_mean_avg)
                / (cnt + BAYESIAN_K)) as f32;
        }

        // Bayesian-shrunk movie means
        let mut movie_bayesian_mean = Array1::<f32>::zeros(base.n_items);
        for i in 0..base.n_items {
            let cnt = base.movie_rating_counts[i] as f64;
            movie_bayesian_mean[i] = ((base.movie_sum_ratings[i] + BAYESIAN_K * base.global_mean)
                / (cnt + BAYESIAN_K)) as f32;
        }

        Self { base, user_bayesian_mean, movie_bayesian_mean }
    }

    #[inline]
    pub fn compute(&self, u: usize, i: usize, _day: i16) -> [f32; N_FEATURES] {
        let mut f = [0.0_f32; N_FEATURES];

        // Feature 5: bayesian movie mean - bayesian user mean
        f[F_BAYESIAN_MOVIE_MINUS_USER] = self.movie_bayesian_mean[i] - self.user_bayesian_mean[u];

        // Feature 7: bayesian-shrunk user mean
        f[F_BAYESIAN_USER_MEAN] = self.user_bayesian_mean[u];

        f
    }

    pub fn compute_all(&self, ds: &Dataset) -> Vec<Array1<f32>> {
        println!("Computing {} Phase B features for {} ratings...", N_FEATURES, ds.n_ratings);

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
            "f05_bayesian_movie_minus_user",
            "f07_bayesian_user_mean",
        ]
    }

    pub fn print_summary(&self) {
        println!("\nFWLS Phase B Statistics ({} features):", N_FEATURES);
        println!("  Users: {}", self.base.n_users);
        println!("  Items: {}", self.base.n_items);
        println!("  Bayesian K: {}", BAYESIAN_K);
        println!("  Global mean: {:.4}", self.base.global_mean);
        println!("  Global user mean avg: {:.4}", self.base.global_user_mean_avg);

        // Sample Bayesian means
        let active_users: Vec<usize> = (0..self.base.n_users)
            .filter(|&u| self.base.user_rating_counts[u] > 0)
            .take(3)
            .collect();
        if !active_users.is_empty() {
            println!("\n  Bayesian user mean samples:");
            for &u in &active_users {
                println!("    user {}: bayesian_mean={:.3}, count={}",
                    u, self.user_bayesian_mean[u], self.base.user_rating_counts[u]);
            }
        }

        println!("\nFeatures:");
        for (i, name) in Self::feature_names().iter().enumerate() {
            println!("  [{:2}] {}", i, name);
        }
    }
}

fn save_features(features: &[Array1<f32>], set_name: &str) {
    use std::io::Write;
    for (k, feat) in features.iter().enumerate() {
        let path = format!("features/fwls_B{}.{}.npy", k + 1, set_name);
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        feat.write_npy(&mut writer).unwrap();
        writer.flush().unwrap();
        println!("  Saved: {}", path);
    }
}

fn main() {
    println!("=== FWLS Meta-Features Generator (Phase B, {} features) ===\n", N_FEATURES);

    std::fs::create_dir_all("features").unwrap();

    // Process train -> probe
    println!("Processing: train -> probe");
    let train = Dataset::load("train", "rtg", "preds");
    let probe = Dataset::load("probe", "rtg", "preds");

    let fwls = FwlsFeaturesB::new(&train);
    fwls.print_summary();

    let probe_features = fwls.compute_all(&probe);
    save_features(&probe_features, "probe");

    // Process fulltrain -> qual
    println!("\nProcessing: fulltrain -> qual");
    let fulltrain = Dataset::load("fulltrain", "rtg", "preds");
    let qual = Dataset::load("qual", "rtg", "preds");

    let fwls_full = FwlsFeaturesB::new(&fulltrain);

    let qual_features = fwls_full.compute_all(&qual);
    save_features(&qual_features, "qual");

    println!("\n=== Done ===");
}
