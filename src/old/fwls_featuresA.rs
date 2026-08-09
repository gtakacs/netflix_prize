// FWLS voting-feature generator, phase A — 10 features: counts, log-counts,
// rating stddevs, user tenure, and one log(movie_cnt)*log(user_cnt) product.
// Frozen archive — see README.md; superseded by src/vfeat1.rs. Writes
// features/fwls_A{k}.{set}.npy and the shipped files were renamed by hand into
// preds_old/fwls/A{k}.*.npy, the layout the current blenders load.
//
// Phase A features (10 features):
// - f01_constant (1)
// - f02_user_gt3_on_date (user rated >3 movies on this date)
// - f03_log_movie_cnt
// - f04_log_user_dates
// - f06_log_user_cnt
// - f16_user_std
// - f17_movie_std
// - f18_log_user_tenure
// - f19_log_user_date_cnt
// - f21_log_product (f03 * f06)

use gravity::fwls_common::FwlsBase;
use gravity::Dataset;
use indicatif::ParallelProgressIterator;
use ndarray::Array1;
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;

const N_FEATURES: usize = 10;

/// Phase A feature indices
const F_CONSTANT: usize = 0;
const F_USER_GT3_ON_DATE: usize = 1;
const F_LOG_MOVIE_CNT: usize = 2;
const F_LOG_USER_DATES: usize = 3;
const F_LOG_USER_CNT: usize = 4;
const F_USER_STD: usize = 5;
const F_MOVIE_STD: usize = 6;
const F_LOG_USER_TENURE: usize = 7;
const F_LOG_USER_DATE_CNT: usize = 8;
const F_LOG_PRODUCT: usize = 9;

pub struct FwlsFeaturesA {
    base: FwlsBase,
}

impl FwlsFeaturesA {
    pub fn new(ds: &Dataset) -> Self {
        println!("Computing FWLS Phase A statistics...");
        let base = FwlsBase::new(ds);
        Self { base }
    }

    #[inline]
    pub fn compute(&self, u: usize, i: usize, day: i16) -> [f32; N_FEATURES] {
        let mut f = [0.0_f32; N_FEATURES];

        // Feature 1: constant 1
        f[F_CONSTANT] = 1.0;

        // Feature 2: user rated >3 movies on this date (binary)
        let user_day_cnt = self.base.user_date_counts[u].get(&day).copied().unwrap_or(0);
        f[F_USER_GT3_ON_DATE] = if user_day_cnt > 3 { 1.0 } else { 0.0 };

        // Feature 3: log(movie rating count)
        let movie_cnt = self.base.movie_rating_counts[i].max(1) as f32;
        f[F_LOG_MOVIE_CNT] = movie_cnt.ln();

        // Feature 4: log(user distinct dates)
        let user_dates = self.base.user_distinct_dates[u].max(1) as f32;
        f[F_LOG_USER_DATES] = user_dates.ln();

        // Feature 6: log(user rating count)
        let user_cnt = self.base.user_rating_counts[u].max(1) as f32;
        f[F_LOG_USER_CNT] = user_cnt.ln();

        // Feature 16: std dev of user ratings
        f[F_USER_STD] = self.base.user_std_ratings[u];

        // Feature 17: std dev of movie ratings
        f[F_MOVIE_STD] = self.base.movie_std_ratings[i];

        // Feature 18: log(rating date - first user rating date + 1)
        let first_date = self.base.user_first_date[u];
        let tenure = (day - first_date).max(0) as f32 + 1.0;
        f[F_LOG_USER_TENURE] = tenure.ln();

        // Feature 19: log(user ratings on this date + 1)
        f[F_LOG_USER_DATE_CNT] = (user_day_cnt as f32 + 1.0).ln();

        // Feature 21: Feature 3 * Feature 6
        f[F_LOG_PRODUCT] = f[F_LOG_MOVIE_CNT] * f[F_LOG_USER_CNT];

        f
    }

    pub fn compute_all(&self, ds: &Dataset) -> Vec<Array1<f32>> {
        println!("Computing {} Phase A features for {} ratings...", N_FEATURES, ds.n_ratings);

        let mut features: Vec<Array1<f32>> = (0..N_FEATURES)
            .map(|_| Array1::<f32>::zeros(ds.n_ratings))
            .collect();

        // Wrap pointers for thread safety
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
            "f01_constant",
            "f02_user_gt3_on_date",
            "f03_log_movie_cnt",
            "f04_log_user_dates",
            "f06_log_user_cnt",
            "f16_user_std",
            "f17_movie_std",
            "f18_log_user_tenure",
            "f19_log_user_date_cnt",
            "f21_log_product",
        ]
    }

    pub fn print_summary(&self) {
        println!("\nFWLS Phase A Statistics ({} features):", N_FEATURES);
        println!("  Users: {}", self.base.n_users);
        println!("  Items: {}", self.base.n_items);

        let avg_user_cnt = self.base.user_rating_counts.iter()
            .map(|&x| x as f64).sum::<f64>() / self.base.n_users as f64;
        let avg_user_dates = self.base.user_distinct_dates.iter()
            .map(|&x| x as f64).sum::<f64>() / self.base.n_users as f64;
        println!("  Avg ratings per user: {:.2}", avg_user_cnt);
        println!("  Avg distinct dates per user: {:.2}", avg_user_dates);

        println!("\nFeatures:");
        for (i, name) in Self::feature_names().iter().enumerate() {
            println!("  [{:2}] {}", i, name);
        }
    }
}

fn save_features(features: &[Array1<f32>], set_name: &str) {
    use std::io::Write;
    for (k, feat) in features.iter().enumerate() {
        let path = format!("features/fwls_A{}.{}.npy", k + 1, set_name);
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        feat.write_npy(&mut writer).unwrap();
        writer.flush().unwrap();
        println!("  Saved: {}", path);
    }
}

fn main() {
    println!("=== FWLS Meta-Features Generator (Phase A, {} features) ===\n", N_FEATURES);

    // Ensure features directory exists
    std::fs::create_dir_all("features").unwrap();

    // Process train -> probe
    println!("Processing: train -> probe");
    let train = Dataset::load("train", "rtg", "preds");
    let probe = Dataset::load("probe", "rtg", "preds");

    let fwls = FwlsFeaturesA::new(&train);
    fwls.print_summary();

    let probe_features = fwls.compute_all(&probe);
    save_features(&probe_features, "probe");

    // Process fulltrain -> qual
    println!("\nProcessing: fulltrain -> qual");
    let fulltrain = Dataset::load("fulltrain", "rtg", "preds");
    let qual = Dataset::load("qual", "rtg", "preds");

    let fwls_full = FwlsFeaturesA::new(&fulltrain);

    let qual_features = fwls_full.compute_all(&qual);
    save_features(&qual_features, "qual");

    println!("\n=== Done ===");
}
