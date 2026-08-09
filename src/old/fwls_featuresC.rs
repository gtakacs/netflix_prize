// FWLS voting-feature generator, phase C — 5 features: rater-count, movie rating-date
// stddev, user date-bias stddev, single-day rating share, avg movie popularity.
// Frozen archive — see README.md; superseded by src/vfeat1.rs. Writes
// features/fwls_C{k}.{set}.npy and the shipped files were renamed by hand into
// preds_old/fwls/C{k}.*.npy, the layout the current blenders load.
//
// Phase C features (5 features):
// - f12_log_avg_rater_cnt (log of average user rating count for movie's raters)
// - f13_log_movie_date_std (log of std dev of movie's rating dates)
// - f15_user_date_bias_std (std dev of user's date-specific mean ratings)
// - f23_movie_single_day_pct (% of movie ratings that were user's only rating that day)
// - f24_user_avg_movie_pop (avg movie rating count for user's rated movies, regularized)

use gravity::fwls_common::{FwlsBase, BAYESIAN_K};
use gravity::Dataset;
use indicatif::ParallelProgressIterator;
use ndarray::Array1;
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;

const N_FEATURES: usize = 5;

/// Phase C feature indices
const F_LOG_AVG_RATER_CNT: usize = 0;
const F_LOG_MOVIE_DATE_STD: usize = 1;
const F_USER_DATE_BIAS_STD: usize = 2;
const F_MOVIE_SINGLE_DAY_PCT: usize = 3;
const F_USER_AVG_MOVIE_POP: usize = 4;

pub struct FwlsFeaturesC {
    base: FwlsBase,
    movie_avg_rater_cnt: Array1<f32>,
    movie_date_std: Array1<f32>,
    user_date_bias_std: Array1<f32>,
    movie_single_day_pct: Array1<f32>,
    user_avg_movie_pop: Array1<f32>,
}

impl FwlsFeaturesC {
    pub fn new(ds: &Dataset) -> Self {
        println!("Computing FWLS Phase C statistics...");
        let base = FwlsBase::new(ds);
        let n_users = base.n_users;
        let n_items = base.n_items;

        // Feature 13: std dev of movie's rating dates
        println!("  Computing movie date std dev...");
        let mut movie_date_std = Array1::<f32>::zeros(n_items);
        for i in 0..n_items {
            let cnt = base.movie_rating_counts[i] as f64;
            if cnt > 1.0 {
                let mean_d = base.movie_sum_dates[i] / cnt;
                let variance = (base.movie_sum_sq_dates[i] / cnt) - (mean_d * mean_d);
                movie_date_std[i] = variance.max(0.0).sqrt() as f32;
            }
        }

        // Feature 15: std dev of user's date-specific means
        println!("  Computing user date bias std dev...");
        let mut user_date_bias_std = Array1::<f32>::zeros(n_users);
        for u in 0..n_users {
            let date_counts = &base.user_date_counts[u];
            let date_sums = &base.user_date_sum[u];
            let n_dates = date_counts.len();
            if n_dates > 1 {
                let mut sum_of_means = 0.0_f64;
                let mut sum_of_means_sq = 0.0_f64;
                for (&date, &cnt) in date_counts.iter() {
                    let date_mean = date_sums[&date] / cnt as f64;
                    sum_of_means += date_mean;
                    sum_of_means_sq += date_mean * date_mean;
                }
                let mean_of_means = sum_of_means / n_dates as f64;
                let variance = (sum_of_means_sq / n_dates as f64) - (mean_of_means * mean_of_means);
                user_date_bias_std[u] = variance.max(0.0).sqrt() as f32;
            }
        }

        // Pass 2: compute cross-entity aggregations
        println!("  Computing cross-entity aggregations...");
        let mut movie_sum_rater_cnt = Array1::<f64>::zeros(n_items);
        let mut movie_single_day_count = Array1::<u32>::zeros(n_items);
        let mut user_sum_movie_pop = Array1::<f64>::zeros(n_users);

        for idx in 0..ds.n_ratings {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            let date = ds.dates[idx];

            // Feature 12: sum of rater's user_rating_count for each movie
            movie_sum_rater_cnt[i] += base.user_rating_counts[u] as f64;

            // Feature 23: count ratings that were user's only rating of the day
            if base.user_date_counts[u][&date] == 1 {
                movie_single_day_count[i] += 1;
            }

            // Feature 24: sum of movie_rating_count for each user
            user_sum_movie_pop[u] += base.movie_rating_counts[i] as f64;
        }

        // Feature 12: average rater count per movie
        let mut movie_avg_rater_cnt = Array1::<f32>::zeros(n_items);
        for i in 0..n_items {
            let cnt = base.movie_rating_counts[i] as f64;
            if cnt > 0.0 {
                movie_avg_rater_cnt[i] = (movie_sum_rater_cnt[i] / cnt) as f32;
            }
        }

        // Feature 23: single-day percentage per movie
        let mut movie_single_day_pct = Array1::<f32>::zeros(n_items);
        for i in 0..n_items {
            let cnt = base.movie_rating_counts[i];
            if cnt > 0 {
                movie_single_day_pct[i] = movie_single_day_count[i] as f32 / cnt as f32;
            }
        }

        // Feature 24: average movie popularity per user (regularized with K)
        let global_avg_movie_cnt = base.total_ratings / n_items as f64;
        let mut user_avg_movie_pop = Array1::<f32>::zeros(n_users);
        for u in 0..n_users {
            let cnt = base.user_rating_counts[u] as f64;
            if cnt > 0.0 {
                user_avg_movie_pop[u] = ((user_sum_movie_pop[u] + BAYESIAN_K * global_avg_movie_cnt)
                    / (cnt + BAYESIAN_K)) as f32;
            }
        }

        Self {
            base,
            movie_avg_rater_cnt,
            movie_date_std,
            user_date_bias_std,
            movie_single_day_pct,
            user_avg_movie_pop,
        }
    }

    #[inline]
    pub fn compute(&self, u: usize, i: usize, _day: i16) -> [f32; N_FEATURES] {
        let mut f = [0.0_f32; N_FEATURES];

        // Feature 12: log(avg user rating count for movie's raters)
        f[F_LOG_AVG_RATER_CNT] = self.movie_avg_rater_cnt[i].max(1.0).ln();

        // Feature 13: log(std dev of movie's rating dates)
        f[F_LOG_MOVIE_DATE_STD] = (self.movie_date_std[i].max(1.0)).ln();

        // Feature 15: std dev of user's date-specific mean ratings
        f[F_USER_DATE_BIAS_STD] = self.user_date_bias_std[u];

        // Feature 23: % of movie ratings that were user's only rating of the day
        f[F_MOVIE_SINGLE_DAY_PCT] = self.movie_single_day_pct[i];

        // Feature 24: avg movie ratings for user's rated movies (regularized)
        f[F_USER_AVG_MOVIE_POP] = self.user_avg_movie_pop[u];

        f
    }

    pub fn compute_all(&self, ds: &Dataset) -> Vec<Array1<f32>> {
        println!("Computing {} Phase C features for {} ratings...", N_FEATURES, ds.n_ratings);

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
            "f12_log_avg_rater_cnt",
            "f13_log_movie_date_std",
            "f15_user_date_bias_std",
            "f23_movie_single_day_pct",
            "f24_user_avg_movie_pop",
        ]
    }

    pub fn print_summary(&self) {
        println!("\nFWLS Phase C Statistics ({} features):", N_FEATURES);
        println!("  Users: {}", self.base.n_users);
        println!("  Items: {}", self.base.n_items);

        println!("\nFeatures:");
        for (i, name) in Self::feature_names().iter().enumerate() {
            println!("  [{:2}] {}", i, name);
        }
    }
}

fn save_features(features: &[Array1<f32>], set_name: &str) {
    use std::io::Write;
    for (k, feat) in features.iter().enumerate() {
        let path = format!("features/fwls_C{}.{}.npy", k + 1, set_name);
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        feat.write_npy(&mut writer).unwrap();
        writer.flush().unwrap();
        println!("  Saved: {}", path);
    }
}

fn main() {
    println!("=== FWLS Meta-Features Generator (Phase C, {} features) ===\n", N_FEATURES);

    std::fs::create_dir_all("features").unwrap();

    // Process train -> probe
    println!("Processing: train -> probe");
    let train = Dataset::load("train", "rtg", "preds");
    let probe = Dataset::load("probe", "rtg", "preds");

    let fwls = FwlsFeaturesC::new(&train);
    fwls.print_summary();

    let probe_features = fwls.compute_all(&probe);
    save_features(&probe_features, "probe");

    // Process fulltrain -> qual
    println!("\nProcessing: fulltrain -> qual");
    let fulltrain = Dataset::load("fulltrain", "rtg", "preds");
    let qual = Dataset::load("qual", "rtg", "preds");

    let fwls_full = FwlsFeaturesC::new(&fulltrain);

    let qual_features = fwls_full.compute_all(&qual);
    save_features(&qual_features, "qual");

    println!("\n=== Done ===");
}
