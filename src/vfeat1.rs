// Voting feature set #1: 130 per-rating features for FWLS-style blending.
//
// Three internal phases (fx1: 000-044, fx2: 045-089, fx3: 090-129).
// Each phase is self-contained — its statistics are gated on the active
// `Selection` so unused phases (and the expensive sim matrix / SVD inside
// fx1) are skipped entirely.

use crate::{Dataset, Split};
use indicatif::ParallelProgressIterator;
use ndarray::Array1;
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

const BAYESIAN_K: f64 = 25.0;

// ===========================================================================
// Selection
// ===========================================================================

/// Selects which feature names this run should produce.
///
/// - `All`: produce every feature (default).
/// - `AllExcept`: produce every feature except those listed by name.
/// - `Only`: produce only the listed names (start from empty).
#[derive(Debug, Clone)]
pub enum Selection {
    All,
    AllExcept(Vec<&'static str>),
    Only(Vec<&'static str>),
}

impl Selection {
    pub fn includes(&self, name: &str) -> bool {
        match self {
            Selection::All => true,
            Selection::AllExcept(excl) => !excl.iter().any(|e| *e == name),
            Selection::Only(incl) => incl.iter().any(|i| *i == name),
        }
    }

    pub fn includes_any(&self, names: &[&str]) -> bool {
        names.iter().any(|n| self.includes(n))
    }
}

// ===========================================================================
// FX1 — Phase A-F (000-044): 45 features, includes sim matrix + SVD
// ===========================================================================

mod fx1 {
    use super::{BAYESIAN_K, Selection};
    use crate::{Dataset, calc_gbias, rand_array2};
    use indicatif::ParallelProgressIterator;
    use ndarray::{Array1, Array2};
    use rand::{SeedableRng, rngs::StdRng};
    use rayon::prelude::*;
    use std::collections::HashMap;

    const SIM_SHRINKAGE: f32 = 100.0;
    pub const N_FEATURES: usize = 45;

    // Phase A (10)
    const FA_CONSTANT: usize = 0;
    const FA_USER_GT3_ON_DATE: usize = 1;
    const FA_LOG_MOVIE_CNT: usize = 2;
    const FA_LOG_USER_DATES: usize = 3;
    const FA_LOG_USER_CNT: usize = 4;
    const FA_USER_STD: usize = 5;
    const FA_MOVIE_STD: usize = 6;
    const FA_LOG_USER_TENURE: usize = 7;
    const FA_LOG_USER_DATE_CNT: usize = 8;
    const FA_LOG_PRODUCT: usize = 9;
    // Phase B (2)
    const FB_BAYESIAN_MOVIE_MINUS_USER: usize = 10;
    const FB_BAYESIAN_USER_MEAN: usize = 11;
    // Phase C (5)
    const FC_LOG_AVG_RATER_CNT: usize = 12;
    const FC_LOG_MOVIE_DATE_STD: usize = 13;
    const FC_USER_DATE_BIAS_STD: usize = 14;
    const FC_MOVIE_SINGLE_DAY_PCT: usize = 15;
    const FC_USER_AVG_MOVIE_POP: usize = 16;
    // Phase D (3)
    const FD_LOG_SIM_SUM: usize = 17;
    const FD_SIM_TOP20_PCT: usize = 18;
    const FD_MAX_SIM: usize = 19;
    // Phase E (5)
    const FE_SVD_USER_NORM: usize = 20;
    const FE_SVD_MOVIE_NORM: usize = 21;
    const FE_ORDINAL_SVD_STD: usize = 22;
    const FE_USER_OVERLAP: usize = 23;
    const FE_SAMEDAY_CORR: usize = 24;
    // Phase F binary same-day (5)
    const FF_USER_HAS_R1_SAME_DAY: usize = 25;
    const FF_USER_HAS_R2_SAME_DAY: usize = 26;
    const FF_USER_HAS_R3_SAME_DAY: usize = 27;
    const FF_USER_HAS_R4_SAME_DAY: usize = 28;
    const FF_USER_HAS_R5_SAME_DAY: usize = 29;
    // binary other-day (5)
    const FF_USER_HAS_R1_OTHER_DAY: usize = 30;
    const FF_USER_HAS_R2_OTHER_DAY: usize = 31;
    const FF_USER_HAS_R3_OTHER_DAY: usize = 32;
    const FF_USER_HAS_R4_OTHER_DAY: usize = 33;
    const FF_USER_HAS_R5_OTHER_DAY: usize = 34;
    // log same-day (5)
    const FF_USER_LOG_R1_SAME_DAY: usize = 35;
    const FF_USER_LOG_R2_SAME_DAY: usize = 36;
    const FF_USER_LOG_R3_SAME_DAY: usize = 37;
    const FF_USER_LOG_R4_SAME_DAY: usize = 38;
    const FF_USER_LOG_R5_SAME_DAY: usize = 39;
    // log other-day (5)
    const FF_USER_LOG_R1_OTHER_DAY: usize = 40;
    const FF_USER_LOG_R2_OTHER_DAY: usize = 41;
    const FF_USER_LOG_R3_OTHER_DAY: usize = 42;
    const FF_USER_LOG_R4_OTHER_DAY: usize = 43;
    const FF_USER_LOG_R5_OTHER_DAY: usize = 44;

    pub const NAMES: [&'static str; N_FEATURES] = [
        "vf000_constant",
        "vf001_user_gt3_on_date",
        "vf002_log_movie_cnt",
        "vf003_log_user_dates",
        "vf004_log_user_cnt",
        "vf005_user_std",
        "vf006_movie_std",
        "vf007_log_user_tenure",
        "vf008_log_user_date_cnt",
        "vf009_log_product",
        "vf010_bayesian_movie_minus_user",
        "vf011_bayesian_user_mean",
        "vf012_log_avg_rater_cnt",
        "vf013_log_movie_date_std",
        "vf014_user_date_bias_std",
        "vf015_movie_single_day_pct",
        "vf016_user_avg_movie_pop",
        "vf017_log_sim_sum",
        "vf018_sim_top20_pct",
        "vf019_max_sim",
        "vf020_svd_user_norm",
        "vf021_svd_movie_norm",
        "vf022_ordinal_svd_std",
        "vf023_user_overlap",
        "vf024_sameday_corr",
        "vf025_has_r1_same_day",
        "vf026_has_r2_same_day",
        "vf027_has_r3_same_day",
        "vf028_has_r4_same_day",
        "vf029_has_r5_same_day",
        "vf030_has_r1_other_day",
        "vf031_has_r2_other_day",
        "vf032_has_r3_other_day",
        "vf033_has_r4_other_day",
        "vf034_has_r5_other_day",
        "vf035_log_r1_same_day",
        "vf036_log_r2_same_day",
        "vf037_log_r3_same_day",
        "vf038_log_r4_same_day",
        "vf039_log_r5_same_day",
        "vf040_log_r1_other_day",
        "vf041_log_r2_other_day",
        "vf042_log_r3_other_day",
        "vf043_log_r4_other_day",
        "vf044_log_r5_other_day",
    ];

    // Names of features whose computation requires the sim matrix
    const NEEDS_SIM: &[&str] = &[
        "vf017_log_sim_sum",
        "vf018_sim_top20_pct",
        "vf019_max_sim",
        "vf024_sameday_corr",
    ];
    // Names of features whose computation requires the 10-factor SVD
    const NEEDS_SVD10: &[&str] = &["vf020_svd_user_norm", "vf021_svd_movie_norm"];
    // Names of features whose computation requires the 60-factor SVD
    const NEEDS_SVD60: &[&str] = &["vf022_ordinal_svd_std"];
    // Names of features whose computation requires the user-overlap pass
    const NEEDS_OVERLAP: &[&str] = &["vf023_user_overlap"];

	#[inline]
	fn sigmoid64(x: f32) -> f32 {
		1.0 / (1.0 + (-x as f64).exp()) as f32
	}

    fn train_svd(
        ds: &Dataset,
        n_feat: usize,
    ) -> (f32, Array1<f32>, Array1<f32>, Array2<f32>, Array2<f32>) {
        let n_epochs = 16usize;
        let seed = 42u64;
        let lr_u = 0.001f32;
        let lr_i = 0.01f32;
        let lr_ub = 0.0035f32;
        let lr_ib = 0.006f32;
        let reg_u = 0.03f32;
        let reg_i = 0.005f32;
        let sigma_u = 0.04f32;
        let sigma_i = 0.05f32;
        let reset_u_epoch = 2usize;

        let mut rng = StdRng::seed_from_u64(seed);
        let gbias = calc_gbias(ds);
        let mut ubias = Array1::<f32>::zeros(ds.n_users);
        let mut ibias = Array1::<f32>::zeros(ds.n_items);
        let mut ufeat = rand_array2(ds.n_users, n_feat, &mut rng, sigma_u);
        let mut ifeat = rand_array2(ds.n_items, n_feat, &mut rng, sigma_i);

        for epoch in 0..n_epochs {
            if epoch == reset_u_epoch {
                ufeat.fill(0.0);
            }
            for idx in 0..ds.n_ratings {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;
                let r = ds.residuals[idx];
                let pred = gbias + ubias[u] + ibias[i] + ufeat.row(u).dot(&ifeat.row(i));
                let err = pred - r;

                ubias[u] -= lr_ub * err;
                ibias[i] -= lr_ib * err;

                for k in 0..n_feat {
                    let pk = ufeat[[u, k]];
                    let qk = ifeat[[i, k]];
                    ufeat[[u, k]] -= lr_u * (err * qk + reg_u * pk);
                    ifeat[[i, k]] -= lr_i * (err * pk + reg_i * qk);
                }
            }
            crate::teeln!("    SVD-{} epoch {}/{}", n_feat, epoch + 1, n_epochs);
        }

        (gbias, ubias, ibias, ufeat, ifeat)
    }

    pub struct Fx1Stats {
        // Phase A
        user_date_counts: Vec<HashMap<i16, u32>>,
        movie_rating_counts: Vec<u32>,
        user_distinct_dates: Vec<u32>,
        user_rating_counts: Vec<u32>,
        user_std_ratings: Vec<f32>,
        movie_std_ratings: Vec<f32>,
        user_first_date: Vec<i16>,
        // Phase B
        user_bayesian_mean: Array1<f32>,
        movie_bayesian_mean: Array1<f32>,
        // Phase C
        movie_avg_rater_cnt: Array1<f32>,
        movie_date_std: Array1<f32>,
        user_date_bias_std: Array1<f32>,
        movie_single_day_pct: Array1<f32>,
        user_avg_movie_pop: Array1<f32>,
        // Phase D+E
        sim_matrix: Array2<f32>,
        max_sim: Array1<f32>,
        user_items: Vec<Vec<usize>>,
        svd_user_norm: Array1<f32>,
        svd_movie_norm: Array1<f32>,
        svd60_ufeat: Array2<f32>,
        svd60_ifeat: Array2<f32>,
        ordinal_thresholds: [f32; 4],
        movie_user_overlap: Array1<f32>,
        movie_sameday_corr: Array1<f32>,
        // Phase F
        user_date_rating_counts: Vec<HashMap<i16, [u32; 5]>>,
        user_rating_counts_by_value: Vec<[u32; 5]>,
        // Whether sim/SVD/overlap were actually computed (for compute() gating)
        has_sim: bool,
        has_svd10: bool,
        has_svd60: bool,
        has_overlap: bool,
    }

    impl Fx1Stats {
        pub fn new(ds: &Dataset, sel: &Selection) -> Self {
            crate::teeln!("Computing FX1 statistics...");
            let n_users = ds.n_users;
            let n_items = ds.n_items;

            // Pass 1: basic per-user / per-movie stats
            crate::teeln!("  Pass 1: Basic statistics...");
            let mut user_rating_counts = vec![0u32; n_users];
            let mut user_sum_ratings = vec![0.0f64; n_users];
            let mut user_sum_sq_ratings = vec![0.0f64; n_users];
            let mut user_date_counts: Vec<HashMap<i16, u32>> = vec![HashMap::new(); n_users];
            let mut user_date_sum: Vec<HashMap<i16, f64>> = vec![HashMap::new(); n_users];
            let mut user_first_date = vec![i16::MAX; n_users];
            let mut user_date_rating_counts: Vec<HashMap<i16, [u32; 5]>> = vec![HashMap::new(); n_users];
            let mut user_rating_counts_by_value: Vec<[u32; 5]> = vec![[0u32; 5]; n_users];

            let mut movie_rating_counts = vec![0u32; n_items];
            let mut movie_sum_ratings = vec![0.0f64; n_items];
            let mut movie_sum_sq_ratings = vec![0.0f64; n_items];
            let mut movie_sum_dates = vec![0.0f64; n_items];
            let mut movie_sum_sq_dates = vec![0.0f64; n_items];

            let mut total_sum = 0.0f64;
            let mut total_cnt = 0u64;

            for idx in 0..ds.n_ratings {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;
                let r = ds.raw_ratings[idx];
                let rf = r as f64;
                let day = ds.dates[idx];

                user_rating_counts[u] += 1;
                user_sum_ratings[u] += rf;
                user_sum_sq_ratings[u] += rf * rf;
                *user_date_counts[u].entry(day).or_insert(0) += 1;
                *user_date_sum[u].entry(day).or_insert(0.0) += rf;
                if day < user_first_date[u] { user_first_date[u] = day; }
                if r >= 1 && r <= 5 {
                    let rv = (r - 1) as usize;
                    user_rating_counts_by_value[u][rv] += 1;
                    let entry = user_date_rating_counts[u].entry(day).or_insert([0u32; 5]);
                    entry[rv] += 1;
                }

                movie_rating_counts[i] += 1;
                movie_sum_ratings[i] += rf;
                movie_sum_sq_ratings[i] += rf * rf;
                movie_sum_dates[i] += day as f64;
                movie_sum_sq_dates[i] += (day as f64) * (day as f64);

                total_sum += rf;
                total_cnt += 1;
            }

            let global_mean = total_sum / total_cnt as f64;
            let total_ratings = total_sum;

            let user_distinct_dates: Vec<u32> = user_date_counts.iter()
                .map(|m| m.len() as u32).collect();

            let user_std_ratings: Vec<f32> = (0..n_users).map(|u| {
                let cnt = user_rating_counts[u] as f64;
                if cnt > 1.0 {
                    let mean = user_sum_ratings[u] / cnt;
                    let var = (user_sum_sq_ratings[u] / cnt) - (mean * mean);
                    var.max(0.0).sqrt() as f32
                } else { 0.0 }
            }).collect();

            let movie_std_ratings: Vec<f32> = (0..n_items).map(|i| {
                let cnt = movie_rating_counts[i] as f64;
                if cnt > 1.0 {
                    let mean = movie_sum_ratings[i] / cnt;
                    let var = (movie_sum_sq_ratings[i] / cnt) - (mean * mean);
                    var.max(0.0).sqrt() as f32
                } else { 0.0 }
            }).collect();

            // Phase B: Bayesian means
            crate::teeln!("  Phase B: Bayesian means...");
            let mut user_mean_sum = 0.0f64;
            let mut user_mean_cnt = 0u64;
            for u in 0..n_users {
                if user_rating_counts[u] > 0 {
                    user_mean_sum += user_sum_ratings[u] / user_rating_counts[u] as f64;
                    user_mean_cnt += 1;
                }
            }
            let global_user_mean_avg = if user_mean_cnt > 0 { user_mean_sum / user_mean_cnt as f64 } else { global_mean };

            let mut user_bayesian_mean = Array1::<f32>::zeros(n_users);
            for u in 0..n_users {
                let cnt = user_rating_counts[u] as f64;
                user_bayesian_mean[u] = ((user_sum_ratings[u] + BAYESIAN_K * global_user_mean_avg)
                    / (cnt + BAYESIAN_K)) as f32;
            }

            let mut movie_bayesian_mean = Array1::<f32>::zeros(n_items);
            for i in 0..n_items {
                let cnt = movie_rating_counts[i] as f64;
                movie_bayesian_mean[i] = ((movie_sum_ratings[i] + BAYESIAN_K * global_mean)
                    / (cnt + BAYESIAN_K)) as f32;
            }

            // Phase C
            crate::teeln!("  Phase C: Cross-entity stats...");
            let mut movie_date_std = Array1::<f32>::zeros(n_items);
            for i in 0..n_items {
                let cnt = movie_rating_counts[i] as f64;
                if cnt > 1.0 {
                    let mean_d = movie_sum_dates[i] / cnt;
                    let variance = (movie_sum_sq_dates[i] / cnt) - (mean_d * mean_d);
                    movie_date_std[i] = variance.max(0.0).sqrt() as f32;
                }
            }

            let mut user_date_bias_std = Array1::<f32>::zeros(n_users);
            for u in 0..n_users {
                let date_counts = &user_date_counts[u];
                let date_sums = &user_date_sum[u];
                let n_dates = date_counts.len();
                if n_dates > 1 {
                    let mut sum_of_means = 0.0f64;
                    let mut sum_of_means_sq = 0.0f64;
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

            let mut movie_sum_rater_cnt = Array1::<f64>::zeros(n_items);
            let mut movie_single_day_count = Array1::<u32>::zeros(n_items);
            let mut user_sum_movie_pop = Array1::<f64>::zeros(n_users);

            for idx in 0..ds.n_ratings {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;
                let date = ds.dates[idx];

                movie_sum_rater_cnt[i] += user_rating_counts[u] as f64;
                if user_date_counts[u][&date] == 1 {
                    movie_single_day_count[i] += 1;
                }
                user_sum_movie_pop[u] += movie_rating_counts[i] as f64;
            }

            let mut movie_avg_rater_cnt = Array1::<f32>::zeros(n_items);
            for i in 0..n_items {
                let cnt = movie_rating_counts[i] as f64;
                if cnt > 0.0 {
                    movie_avg_rater_cnt[i] = (movie_sum_rater_cnt[i] / cnt) as f32;
                }
            }

            let mut movie_single_day_pct = Array1::<f32>::zeros(n_items);
            for i in 0..n_items {
                let cnt = movie_rating_counts[i];
                if cnt > 0 {
                    movie_single_day_pct[i] = movie_single_day_count[i] as f32 / cnt as f32;
                }
            }

            let global_avg_movie_cnt = total_ratings / n_items as f64;
            let mut user_avg_movie_pop = Array1::<f32>::zeros(n_users);
            for u in 0..n_users {
                let cnt = user_rating_counts[u] as f64;
                if cnt > 0.0 {
                    user_avg_movie_pop[u] = ((user_sum_movie_pop[u] + BAYESIAN_K * global_avg_movie_cnt)
                        / (cnt + BAYESIAN_K)) as f32;
                }
            }

            // user_items (sorted) — needed both for sim Phase D compute and overlap
            let needs_sim = sel.includes_any(NEEDS_SIM);
            let needs_overlap = sel.includes_any(NEEDS_OVERLAP);
            let needs_svd10 = sel.includes_any(NEEDS_SVD10);
            let needs_svd60 = sel.includes_any(NEEDS_SVD60);

            let mut user_items: Vec<Vec<usize>> = vec![vec![]; n_users];
            if needs_sim || needs_overlap {
                for idx in 0..ds.n_ratings {
                    let u = ds.user_idxs[idx] as usize;
                    let i = ds.item_idxs[idx] as usize;
                    user_items[u].push(i);
                }
                for u in 0..n_users {
                    user_items[u].sort_unstable();
                }
            }

            // Phase D+E sim matrix and sameday corr
            let (sim_matrix, max_sim, movie_sameday_corr) = if needs_sim {
                crate::teeln!("  Phase D+E: Building per-user rating index...");

                let movie_means: Vec<f32> = (0..n_items).map(|i| {
                    let cnt = movie_rating_counts[i] as f64;
                    if cnt > 0.0 { (movie_sum_ratings[i] / cnt) as f32 } else { 0.0 }
                }).collect();

                let mut by_user: Vec<Vec<(usize, f32, i16)>> = vec![vec![]; n_users];
                for idx in 0..ds.n_ratings {
                    let u = ds.user_idxs[idx] as usize;
                    let i = ds.item_idxs[idx] as usize;
                    let r = ds.raw_ratings[idx] as f32 - movie_means[i];
                    let d = ds.dates[idx];
                    by_user[u].push((i, r, d));
                }

                let mut movie_raters: Vec<Vec<usize>> = vec![vec![]; n_items];
                for idx in 0..ds.n_ratings {
                    let u = ds.user_idxs[idx] as usize;
                    let i = ds.item_idxs[idx] as usize;
                    movie_raters[i].push(u);
                }

                crate::teeln!("  Phase D+E: Computing item norms...");
                let mut item_self_dot = vec![0.0f32; n_items];
                for u in 0..n_users {
                    for &(i, r, _) in &by_user[u] {
                        item_self_dot[i] += r * r;
                    }
                }
                let item_norms: Vec<f32> = item_self_dot.iter().map(|&x| x.max(0.0).sqrt()).collect();

                crate::teeln!("  Phase D+E: Computing similarity matrix (per-item parallel)...");

                struct SimRow {
                    sim: Vec<f32>,
                    max_sim: f32,
                    sameday_corr: f32,
                }

                let pb = crate::make_pb(n_items as u64);
                let sim_rows: Vec<SimRow> = (0..n_items).into_par_iter()
                    .progress_with(pb)
                    .map(|i| {
                        let mut supp = vec![0.0f32; n_items];
                        let mut prod = vec![0.0f32; n_items];
                        let mut sameday = vec![0.0f32; n_items];

                        for &u in &movie_raters[i] {
                            let mut ri = 0.0f32;
                            let mut di = 0i16;
                            for &(item, r, d) in &by_user[u] {
                                if item == i { ri = r; di = d; break; }
                            }
                            for &(j, rj, dj) in &by_user[u] {
                                supp[j] += 1.0;
                                prod[j] += ri * rj;
                                if di == dj {
                                    sameday[j] += 1.0;
                                }
                            }
                        }

                        let norm_i = item_norms[i];
                        let mut sim = vec![0.0f32; n_items];
                        let mut max_sim = 0.0f32;
                        for j in 0..n_items {
                            if i == j { continue; }
                            let n = supp[j];
                            if n < 2.0 { continue; }
                            let den = norm_i * item_norms[j];
                            let phi = if den > 0.0 { prod[j] / den } else { 0.0 };
                            let s = phi * n / (n + SIM_SHRINKAGE);
                            sim[j] = s;
                            if s > max_sim { max_sim = s; }
                        }

                        let mut cnt = 0u64;
                        let mut sum_x = 0.0f64;
                        let mut sum_y = 0.0f64;
                        let mut sum_xy = 0.0f64;
                        let mut sum_x2 = 0.0f64;
                        let mut sum_y2 = 0.0f64;
                        for j in 0..n_items {
                            if i == j { continue; }
                            let sp = supp[j];
                            if sp < 2.0 { continue; }
                            let x = (sameday[j] / sp) as f64;
                            let y = sim[j] as f64;
                            cnt += 1;
                            sum_x += x;
                            sum_y += y;
                            sum_xy += x * y;
                            sum_x2 += x * x;
                            sum_y2 += y * y;
                        }
                        let sameday_corr = if cnt > 1 {
                            let nf = cnt as f64;
                            let cov = sum_xy / nf - (sum_x / nf) * (sum_y / nf);
                            let var_x = sum_x2 / nf - (sum_x / nf) * (sum_x / nf);
                            let var_y = sum_y2 / nf - (sum_y / nf) * (sum_y / nf);
                            let den = (var_x * var_y).sqrt();
                            if den > 1e-10 { (cov / den) as f32 } else { 0.0 }
                        } else { 0.0 };

                        SimRow { sim, max_sim, sameday_corr }
                    }).collect();

                crate::teeln!("  Assembling similarity matrix...");
                let mut sim_matrix = Array2::<f32>::zeros((n_items, n_items));
                let mut max_sim = Array1::<f32>::zeros(n_items);
                let mut movie_sameday_corr_vec = Vec::with_capacity(n_items);
                for (i, row) in sim_rows.into_iter().enumerate() {
                    for j in 0..n_items {
                        sim_matrix[[i, j]] = row.sim[j];
                    }
                    max_sim[i] = row.max_sim;
                    movie_sameday_corr_vec.push(row.sameday_corr);
                }
                (sim_matrix, max_sim, Array1::from(movie_sameday_corr_vec))
            } else {
                (Array2::zeros((0, 0)), Array1::zeros(0), Array1::zeros(0))
            };

            // SVD 10-factor
            let (svd_user_norm, svd_movie_norm) = if needs_svd10 {
                crate::teeln!("  Phase E: Training 10-factor SVD...");
                let (_, _, _, svd10_ufeat, svd10_ifeat) = train_svd(ds, 10);
                let user_norm = Array1::from_iter(
                    (0..n_users).map(|u| {
                        let row = svd10_ufeat.row(u);
                        row.dot(&row).sqrt()
                    })
                );
                let movie_norm = Array1::from_iter(
                    (0..n_items).map(|i| {
                        let row = svd10_ifeat.row(i);
                        row.dot(&row).sqrt()
                    })
                );
                (user_norm, movie_norm)
            } else {
                (Array1::zeros(0), Array1::zeros(0))
            };

            // SVD 60-factor + ordinal thresholds
            let (svd60_ufeat, svd60_ifeat, ordinal_thresholds) = if needs_svd60 {
                crate::teeln!("  Phase E: Training 60-factor SVD...");
                let (_, _, _, svd60_ufeat, svd60_ifeat) = train_svd(ds, 60);

                crate::teeln!("  Phase E: Estimating ordinal thresholds...");
                let mut rating_dot_sums = [0.0f64; 5];
                let mut rating_counts = [0u64; 5];
                for idx in 0..ds.n_ratings {
                    let u = ds.user_idxs[idx] as usize;
                    let i = ds.item_idxs[idx] as usize;
                    let r = ds.raw_ratings[idx] as usize;
                    if r >= 1 && r <= 5 {
                        let dot = svd60_ufeat.row(u).dot(&svd60_ifeat.row(i)) as f64;
                        rating_dot_sums[r - 1] += dot;
                        rating_counts[r - 1] += 1;
                    }
                }
                let mut rating_means = [0.0f64; 5];
                for k in 0..5 {
                    if rating_counts[k] > 0 {
                        rating_means[k] = rating_dot_sums[k] / rating_counts[k] as f64;
                    }
                }
                let mut thresholds = [0.0f32; 4];
                for k in 0..4 {
                    thresholds[k] = ((rating_means[k] + rating_means[k + 1]) / 2.0) as f32;
                }
                crate::teeln!("  Ordinal thresholds: {:?}", thresholds);
                (svd60_ufeat, svd60_ifeat, thresholds)
            } else {
                (Array2::zeros((0, 0)), Array2::zeros((0, 0)), [0.0; 4])
            };

            // Phase E: user overlap
            let movie_user_overlap = if needs_overlap {
                crate::teeln!("  Phase E: User overlap...");
                let mut movie_raters: Vec<Vec<usize>> = vec![vec![]; n_items];
                for idx in 0..ds.n_ratings {
                    let u = ds.user_idxs[idx] as usize;
                    let i = ds.item_idxs[idx] as usize;
                    movie_raters[i].push(u);
                }
                let pb = crate::make_pb(n_items as u64);
                let overlap: Vec<f32> = (0..n_items).into_par_iter()
                    .progress_with(pb)
                    .map(|m| {
                        let raters = &movie_raters[m];
                        let n_raters = raters.len();
                        if n_raters < 2 { return 0.0; }
                        let max_sample = 50;
                        let sample: Vec<usize> = if n_raters <= max_sample {
                            raters.clone()
                        } else {
                            (0..max_sample).map(|k| raters[k * n_raters / max_sample]).collect()
                        };
                        let mut total_overlap = 0.0f64;
                        let mut n_pairs = 0u64;
                        for a in 0..sample.len() {
                            let s1 = &user_items[sample[a]];
                            for b in (a + 1)..sample.len() {
                                let s2 = &user_items[sample[b]];
                                let mut intersection = 0usize;
                                let (mut p, mut q) = (0, 0);
                                while p < s1.len() && q < s2.len() {
                                    if s1[p] == s2[q] { intersection += 1; p += 1; q += 1; }
                                    else if s1[p] < s2[q] { p += 1; }
                                    else { q += 1; }
                                }
                                let smaller = s1.len().min(s2.len());
                                if smaller > 0 {
                                    total_overlap += intersection as f64 / smaller as f64;
                                }
                                n_pairs += 1;
                            }
                        }
                        if n_pairs > 0 { (total_overlap / n_pairs as f64) as f32 } else { 0.0 }
                    }).collect();
                Array1::from(overlap)
            } else {
                Array1::zeros(0)
            };

            Self {
                user_date_counts,
                movie_rating_counts,
                user_distinct_dates,
                user_rating_counts,
                user_std_ratings,
                movie_std_ratings,
                user_first_date,
                user_bayesian_mean,
                movie_bayesian_mean,
                movie_avg_rater_cnt,
                movie_date_std,
                user_date_bias_std,
                movie_single_day_pct,
                user_avg_movie_pop,
                sim_matrix,
                max_sim,
                user_items,
                svd_user_norm,
                svd_movie_norm,
                svd60_ufeat,
                svd60_ifeat,
                ordinal_thresholds,
                movie_user_overlap,
                movie_sameday_corr,
                user_date_rating_counts,
                user_rating_counts_by_value,
                has_sim: needs_sim,
                has_svd10: needs_svd10,
                has_svd60: needs_svd60,
                has_overlap: needs_overlap,
            }
        }

        #[inline]
        pub fn compute(&self, u: usize, i: usize, day: i16) -> [f32; N_FEATURES] {
            let mut f = [0.0_f32; N_FEATURES];

            // Phase A
            f[FA_CONSTANT] = 1.0;
            let user_day_cnt = self.user_date_counts[u].get(&day).copied().unwrap_or(0);
            f[FA_USER_GT3_ON_DATE] = if user_day_cnt > 3 { 1.0 } else { 0.0 };
            let movie_cnt = self.movie_rating_counts[i].max(1) as f32;
            f[FA_LOG_MOVIE_CNT] = (movie_cnt as f64).ln() as f32;
            let user_dates = self.user_distinct_dates[u].max(1) as f32;
            f[FA_LOG_USER_DATES] = (user_dates as f64).ln() as f32;
            let user_cnt = self.user_rating_counts[u].max(1) as f32;
            f[FA_LOG_USER_CNT] = (user_cnt as f64).ln() as f32;
            f[FA_USER_STD] = self.user_std_ratings[u];
            f[FA_MOVIE_STD] = self.movie_std_ratings[i];
            let first_date = self.user_first_date[u];
            let tenure = (day - first_date).max(0) as f32 + 1.0;
            f[FA_LOG_USER_TENURE] = (tenure as f64).ln() as f32;
            f[FA_LOG_USER_DATE_CNT] = ((user_day_cnt as f32 + 1.0) as f64).ln() as f32;
            f[FA_LOG_PRODUCT] = f[FA_LOG_MOVIE_CNT] * f[FA_LOG_USER_CNT];

            // Phase B
            f[FB_BAYESIAN_MOVIE_MINUS_USER] = self.movie_bayesian_mean[i] - self.user_bayesian_mean[u];
            f[FB_BAYESIAN_USER_MEAN] = self.user_bayesian_mean[u];

            // Phase C
            f[FC_LOG_AVG_RATER_CNT] = (self.movie_avg_rater_cnt[i].max(1.0) as f64).ln() as f32;
            f[FC_LOG_MOVIE_DATE_STD] = (self.movie_date_std[i].max(1.0) as f64).ln() as f32;
            f[FC_USER_DATE_BIAS_STD] = self.user_date_bias_std[u];
            f[FC_MOVIE_SINGLE_DAY_PCT] = self.movie_single_day_pct[i];
            f[FC_USER_AVG_MOVIE_POP] = self.user_avg_movie_pop[u];

            // Phase D
            if self.has_sim {
                let user_items = &self.user_items[u];
                let mut pos_sims: Vec<f32> = Vec::with_capacity(user_items.len());
                for &j in user_items {
                    if j == i as usize { continue; }
                    let sim = self.sim_matrix[[i, j]];
                    if sim > 0.0 {
                        pos_sims.push(sim);
                    }
                }
                let total_pos_sim: f32 = pos_sims.iter().sum();
                f[FD_LOG_SIM_SUM] = ((1.0 + total_pos_sim) as f64).ln() as f32;
                if !pos_sims.is_empty() {
                    pos_sims.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let mut top_sum: f32 = 0.0;
                    for k in 0..pos_sims.len() {
                        if k * 5 >= pos_sims.len() { break; }
                        top_sum += pos_sims[k];
                    }
                    f[FD_SIM_TOP20_PCT] = top_sum / total_pos_sim;
                }
                f[FD_MAX_SIM] = self.max_sim[i];
            }

            // Phase E
            if self.has_svd10 {
                f[FE_SVD_USER_NORM] = self.svd_user_norm[u];
                f[FE_SVD_MOVIE_NORM] = self.svd_movie_norm[i];
            }

            if self.has_svd60 {
                let dot = self.svd60_ufeat.row(u).dot(&self.svd60_ifeat.row(i));
                let mut cum = [0.0f32; 4];
                for k in 0..4 {
                    cum[k] = sigmoid64(self.ordinal_thresholds[k] - dot);
                }
                let p = [
                    cum[0],
                    cum[1] - cum[0],
                    cum[2] - cum[1],
                    cum[3] - cum[2],
                    1.0 - cum[3],
                ];
                let mean = p[0] + 2.0 * p[1] + 3.0 * p[2] + 4.0 * p[3] + 5.0 * p[4];
                let mean_sq = p[0] + 4.0 * p[1] + 9.0 * p[2] + 16.0 * p[3] + 25.0 * p[4];
                f[FE_ORDINAL_SVD_STD] = (mean_sq - mean * mean).max(0.0).sqrt();
            }

            if self.has_overlap {
                f[FE_USER_OVERLAP] = self.movie_user_overlap[i];
            }

            if self.has_sim {
                f[FE_SAMEDAY_CORR] = self.movie_sameday_corr[i];
            }

            // Phase F
            let day_counts = self.user_date_rating_counts[u].get(&day);
            let total_counts = &self.user_rating_counts_by_value[u];
            let dc = day_counts.copied().unwrap_or([0u32; 5]);

            f[FF_USER_HAS_R1_SAME_DAY] = if dc[0] > 0 { 1.0 } else { 0.0 };
            f[FF_USER_HAS_R2_SAME_DAY] = if dc[1] > 0 { 1.0 } else { 0.0 };
            f[FF_USER_HAS_R3_SAME_DAY] = if dc[2] > 0 { 1.0 } else { 0.0 };
            f[FF_USER_HAS_R4_SAME_DAY] = if dc[3] > 0 { 1.0 } else { 0.0 };
            f[FF_USER_HAS_R5_SAME_DAY] = if dc[4] > 0 { 1.0 } else { 0.0 };

            f[FF_USER_LOG_R1_SAME_DAY] = ((1.0 + dc[0] as f32) as f64).ln() as f32;
            f[FF_USER_LOG_R2_SAME_DAY] = ((1.0 + dc[1] as f32) as f64).ln() as f32;
            f[FF_USER_LOG_R3_SAME_DAY] = ((1.0 + dc[2] as f32) as f64).ln() as f32;
            f[FF_USER_LOG_R4_SAME_DAY] = ((1.0 + dc[3] as f32) as f64).ln() as f32;
            f[FF_USER_LOG_R5_SAME_DAY] = ((1.0 + dc[4] as f32) as f64).ln() as f32;

            let other = [
                total_counts[0].saturating_sub(dc[0]),
                total_counts[1].saturating_sub(dc[1]),
                total_counts[2].saturating_sub(dc[2]),
                total_counts[3].saturating_sub(dc[3]),
                total_counts[4].saturating_sub(dc[4]),
            ];

            f[FF_USER_HAS_R1_OTHER_DAY] = if other[0] > 0 { 1.0 } else { 0.0 };
            f[FF_USER_HAS_R2_OTHER_DAY] = if other[1] > 0 { 1.0 } else { 0.0 };
            f[FF_USER_HAS_R3_OTHER_DAY] = if other[2] > 0 { 1.0 } else { 0.0 };
            f[FF_USER_HAS_R4_OTHER_DAY] = if other[3] > 0 { 1.0 } else { 0.0 };
            f[FF_USER_HAS_R5_OTHER_DAY] = if other[4] > 0 { 1.0 } else { 0.0 };

            f[FF_USER_LOG_R1_OTHER_DAY] = ((1.0 + other[0] as f32) as f64).ln() as f32;
            f[FF_USER_LOG_R2_OTHER_DAY] = ((1.0 + other[1] as f32) as f64).ln() as f32;
            f[FF_USER_LOG_R3_OTHER_DAY] = ((1.0 + other[2] as f32) as f64).ln() as f32;
            f[FF_USER_LOG_R4_OTHER_DAY] = ((1.0 + other[3] as f32) as f64).ln() as f32;
            f[FF_USER_LOG_R5_OTHER_DAY] = ((1.0 + other[4] as f32) as f64).ln() as f32;

            f
        }
    }
}

// ===========================================================================
// FX2 — G1-G5 (045-089): 45 features, no sim/SVD
// ===========================================================================

mod fx2 {
    use super::{BAYESIAN_K, Selection};
    use crate::Dataset;
    use std::collections::HashMap;

    pub const N_FEATURES: usize = 45;

    // G1: temporal dynamics
    const G1_MOVIE_LOG_AGE: usize = 0;
    const G1_USER_RATING_VELOCITY: usize = 1;
    const G1_MOVIE_RATING_VELOCITY: usize = 2;
    const G1_USER_DATE_POSITION: usize = 3;
    const G1_MOVIE_DATE_POSITION: usize = 4;
    const G1_LOG_DAY: usize = 5;
    const G1_DAY_SIN7: usize = 6;
    const G1_DAY_SIN30: usize = 7;
    const G1_USER_LOG_AVG_DAY_GAP: usize = 8;
    const G1_MOVIE_LOG_DISTINCT_DATES: usize = 9;
    // G2: distribution shape
    const G2_USER_SKEWNESS: usize = 10;
    const G2_MOVIE_SKEWNESS: usize = 11;
    const G2_USER_KURTOSIS: usize = 12;
    const G2_MOVIE_KURTOSIS: usize = 13;
    const G2_USER_ENTROPY: usize = 14;
    const G2_MOVIE_ENTROPY: usize = 15;
    const G2_USER_PCT_EXTREME: usize = 16;
    const G2_MOVIE_PCT_EXTREME: usize = 17;
    const G2_USER_MODE_FRAC: usize = 18;
    const G2_MOVIE_MODE_FRAC: usize = 19;
    // G3: cross-entity interactions
    const G3_MOVIE_AVG_RATER_TENURE: usize = 20;
    const G3_USER_AVG_MOVIE_DATE_STD: usize = 21;
    const G3_LOG_CNT_GEOMETRIC: usize = 22;
    const G3_LOG_CNT_HARMONIC: usize = 23;
    const G3_USER_STD_X_MOVIE_STD: usize = 24;
    const G3_MOVIE_AVG_RATER_STD: usize = 25;
    const G3_USER_AVG_MOVIE_MEAN: usize = 26;
    const G3_LOG_USER_CNT_OVER_MOVIE_CNT: usize = 27;
    const G3_USER_TENURE_X_LOG_MOVIE_CNT: usize = 28;
    const G3_USER_DATE_BIAS_STD_X_MOVIE_STD: usize = 29;
    // G4: same-day proportions
    const G4_PCT_R1_SAME_DAY: usize = 30;
    const G4_PCT_R2_SAME_DAY: usize = 31;
    const G4_PCT_R3_SAME_DAY: usize = 32;
    const G4_PCT_R4_SAME_DAY: usize = 33;
    const G4_PCT_R5_SAME_DAY: usize = 34;
    // G5: behavioral
    const G5_USER_LOG_AVG_BATCH: usize = 35;
    const G5_USER_MULTI_DAY_PCT: usize = 36;
    const G5_MOVIE_LOG_RATINGS_PER_DATE: usize = 37;
    const G5_USER_LOG_MAX_DAY_CNT: usize = 38;
    const G5_SAME_DAY_MEAN_RATING: usize = 39;
    const G5_SAME_DAY_STD_RATING: usize = 40;
    const G5_USER_PCT_POPULAR: usize = 41;
    const G5_MOVIE_PCT_HEAVY_RATERS: usize = 42;
    const G5_USER_LAST_DATE_DIFF: usize = 43;
    const G5_MOVIE_LAST_DATE_DIFF: usize = 44;

    pub const NAMES: [&'static str; N_FEATURES] = [
        "vf045_movie_log_age",
        "vf046_user_rating_velocity",
        "vf047_movie_rating_velocity",
        "vf048_user_date_position",
        "vf049_movie_date_position",
        "vf050_log_day",
        "vf051_day_sin7",
        "vf052_day_sin30",
        "vf053_user_log_avg_day_gap",
        "vf054_movie_log_distinct_dates",
        "vf055_user_skewness",
        "vf056_movie_skewness",
        "vf057_user_kurtosis",
        "vf058_movie_kurtosis",
        "vf059_user_entropy",
        "vf060_movie_entropy",
        "vf061_user_pct_extreme",
        "vf062_movie_pct_extreme",
        "vf063_user_mode_frac",
        "vf064_movie_mode_frac",
        "vf065_movie_avg_rater_tenure",
        "vf066_user_avg_movie_date_std",
        "vf067_log_cnt_geometric",
        "vf068_log_cnt_harmonic",
        "vf069_user_std_x_movie_std",
        "vf070_movie_avg_rater_std",
        "vf071_user_avg_movie_mean",
        "vf072_log_user_cnt_over_movie_cnt",
        "vf073_user_tenure_x_log_movie_cnt",
        "vf074_user_date_bias_std_x_movie_std",
        "vf075_pct_r1_same_day",
        "vf076_pct_r2_same_day",
        "vf077_pct_r3_same_day",
        "vf078_pct_r4_same_day",
        "vf079_pct_r5_same_day",
        "vf080_user_log_avg_batch",
        "vf081_user_multi_day_pct",
        "vf082_movie_log_ratings_per_date",
        "vf083_user_log_max_day_cnt",
        "vf084_same_day_mean_rating",
        "vf085_same_day_std_rating",
        "vf086_user_pct_popular",
        "vf087_movie_pct_heavy_raters",
        "vf088_user_last_date_diff",
        "vf089_movie_last_date_diff",
    ];

    #[inline]
    fn entropy5(counts: &[u32; 5]) -> f32 {
        let total: u32 = counts.iter().sum();
        if total == 0 { return 0.0; }
        let tf = total as f64;
        let mut h = 0.0f64;
        for &c in counts {
            if c > 0 {
                let p = c as f64 / tf;
                h -= p * p.ln();
            }
        }
        h as f32
    }

    pub struct Fx2Stats {
        user_cnt: Vec<u32>,
        user_first_date: Vec<i16>,
        user_last_date: Vec<i16>,
        user_distinct_dates: Vec<u32>,
        user_avg_day_gap: Vec<f32>,
        user_max_day_cnt: Vec<u32>,
        user_multi_day_pct: Vec<f32>,
        user_mean: Vec<f32>,
        user_std: Vec<f32>,
        user_skewness: Vec<f32>,
        user_kurtosis: Vec<f32>,
        user_entropy: Vec<f32>,
        user_pct_extreme: Vec<f32>,
        user_mode_frac: Vec<f32>,
        user_date_counts: Vec<HashMap<i16, u32>>,
        user_date_sum: Vec<HashMap<i16, f64>>,
        user_date_sum_sq: Vec<HashMap<i16, f64>>,
        user_date_rating_counts: Vec<HashMap<i16, [u32; 5]>>,
        movie_cnt: Vec<u32>,
        movie_first_date: Vec<i16>,
        movie_last_date: Vec<i16>,
        movie_distinct_dates: Vec<u32>,
        movie_std: Vec<f32>,
        movie_skewness: Vec<f32>,
        movie_kurtosis: Vec<f32>,
        movie_entropy: Vec<f32>,
        movie_pct_extreme: Vec<f32>,
        movie_mode_frac: Vec<f32>,
        movie_avg_rater_tenure: Vec<f32>,
        movie_avg_rater_std: Vec<f32>,
        user_avg_movie_mean: Vec<f32>,
        user_avg_movie_date_std: Vec<f32>,
        user_pct_popular: Vec<f32>,
        movie_pct_heavy_raters: Vec<f32>,
        user_date_bias_std: Vec<f32>,
    }

    impl Fx2Stats {
        pub fn new(ds: &Dataset, _sel: &Selection) -> Self {
            crate::teeln!("Computing FX2 statistics...");
            let n_users = ds.n_users;
            let n_items = ds.n_items;

            crate::teeln!("  Pass 1: Basic statistics...");
            let mut user_cnt = vec![0u32; n_users];
            let mut user_sum = vec![0.0f64; n_users];
            let mut user_sum2 = vec![0.0f64; n_users];
            let mut user_sum3 = vec![0.0f64; n_users];
            let mut user_sum4 = vec![0.0f64; n_users];
            let mut user_first_date = vec![i16::MAX; n_users];
            let mut user_last_date = vec![i16::MIN; n_users];
            let mut user_date_counts: Vec<HashMap<i16, u32>> = vec![HashMap::new(); n_users];
            let mut user_date_sum: Vec<HashMap<i16, f64>> = vec![HashMap::new(); n_users];
            let mut user_date_sum_sq: Vec<HashMap<i16, f64>> = vec![HashMap::new(); n_users];
            let mut user_date_rating_counts: Vec<HashMap<i16, [u32; 5]>> = vec![HashMap::new(); n_users];
            let mut user_rating_hist: Vec<[u32; 5]> = vec![[0u32; 5]; n_users];

            let mut movie_cnt = vec![0u32; n_items];
            let mut movie_sum = vec![0.0f64; n_items];
            let mut movie_sum2 = vec![0.0f64; n_items];
            let mut movie_sum3 = vec![0.0f64; n_items];
            let mut movie_sum4 = vec![0.0f64; n_items];
            let mut movie_first_date = vec![i16::MAX; n_items];
            let mut movie_last_date = vec![i16::MIN; n_items];
            let mut movie_date_set: Vec<HashMap<i16, ()>> = vec![HashMap::new(); n_items];
            let mut movie_sum_dates = vec![0.0f64; n_items];
            let mut movie_sum_sq_dates = vec![0.0f64; n_items];
            let mut movie_rating_hist: Vec<[u32; 5]> = vec![[0u32; 5]; n_items];

            let mut total_sum = 0.0f64;
            let mut total_cnt = 0u64;

            for idx in 0..ds.n_ratings {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;
                let r = ds.raw_ratings[idx];
                let rf = r as f64;
                let day = ds.dates[idx];

                user_cnt[u] += 1;
                user_sum[u] += rf;
                user_sum2[u] += rf * rf;
                user_sum3[u] += rf * rf * rf;
                user_sum4[u] += rf * rf * rf * rf;
                if day < user_first_date[u] { user_first_date[u] = day; }
                if day > user_last_date[u] { user_last_date[u] = day; }
                *user_date_counts[u].entry(day).or_insert(0) += 1;
                *user_date_sum[u].entry(day).or_insert(0.0) += rf;
                *user_date_sum_sq[u].entry(day).or_insert(0.0) += rf * rf;
                if r >= 1 && r <= 5 {
                    user_rating_hist[u][(r - 1) as usize] += 1;
                    let entry = user_date_rating_counts[u].entry(day).or_insert([0u32; 5]);
                    entry[(r - 1) as usize] += 1;
                }

                movie_cnt[i] += 1;
                movie_sum[i] += rf;
                movie_sum2[i] += rf * rf;
                movie_sum3[i] += rf * rf * rf;
                movie_sum4[i] += rf * rf * rf * rf;
                if day < movie_first_date[i] { movie_first_date[i] = day; }
                if day > movie_last_date[i] { movie_last_date[i] = day; }
                movie_date_set[i].entry(day).or_insert(());
                movie_sum_dates[i] += day as f64;
                movie_sum_sq_dates[i] += (day as f64) * (day as f64);
                if r >= 1 && r <= 5 {
                    movie_rating_hist[i][(r - 1) as usize] += 1;
                }

                total_sum += rf;
                total_cnt += 1;
            }

            let global_mean = (total_sum / total_cnt as f64) as f32;

            crate::teeln!("  Deriving per-user stats...");
            let user_distinct_dates: Vec<u32> = user_date_counts.iter()
                .map(|m| m.len() as u32).collect();

            let user_mean: Vec<f32> = (0..n_users).map(|u| {
                if user_cnt[u] > 0 { (user_sum[u] / user_cnt[u] as f64) as f32 } else { global_mean }
            }).collect();

            let user_std: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n > 1.0 {
                    let m = user_sum[u] / n;
                    ((user_sum2[u] / n - m * m).max(0.0)).sqrt() as f32
                } else { 0.0 }
            }).collect();

            let user_skewness: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n < 3.0 { return 0.0; }
                let m = user_sum[u] / n;
                let var = (user_sum2[u] / n - m * m).max(1e-12);
                let std = var.sqrt();
                let m3 = user_sum3[u] / n - 3.0 * m * user_sum2[u] / n + 2.0 * m * m * m;
                (m3 / (std * std * std)) as f32
            }).collect();

            let user_kurtosis: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n < 4.0 { return 0.0; }
                let m = user_sum[u] / n;
                let var = (user_sum2[u] / n - m * m).max(1e-12);
                let m4 = user_sum4[u] / n - 4.0 * m * user_sum3[u] / n
                    + 6.0 * m * m * user_sum2[u] / n - 3.0 * m * m * m * m;
                (m4 / (var * var) - 3.0) as f32
            }).collect();

            let user_entropy: Vec<f32> = (0..n_users).map(|u| entropy5(&user_rating_hist[u])).collect();

            let user_pct_extreme: Vec<f32> = (0..n_users).map(|u| {
                let total = user_cnt[u];
                if total == 0 { return 0.0; }
                (user_rating_hist[u][0] + user_rating_hist[u][4]) as f32 / total as f32
            }).collect();

            let user_mode_frac: Vec<f32> = (0..n_users).map(|u| {
                let total = user_cnt[u];
                if total == 0 { return 0.0; }
                let max_c = *user_rating_hist[u].iter().max().unwrap();
                max_c as f32 / total as f32
            }).collect();

            let user_avg_day_gap: Vec<f32> = (0..n_users).map(|u| {
                let nd = user_distinct_dates[u];
                if nd <= 1 { return 0.0; }
                let span = (user_last_date[u] - user_first_date[u]).max(0) as f32;
                span / (nd - 1) as f32
            }).collect();

            let user_max_day_cnt: Vec<u32> = user_date_counts.iter()
                .map(|m| m.values().copied().max().unwrap_or(0)).collect();

            let user_multi_day_pct: Vec<f32> = user_date_counts.iter()
                .map(|m| {
                    let total = m.len();
                    if total == 0 { return 0.0; }
                    let multi = m.values().filter(|&&c| c > 1).count();
                    multi as f32 / total as f32
                }).collect();

            let user_date_bias_std: Vec<f32> = (0..n_users).map(|u| {
                let n_dates = user_date_counts[u].len();
                if n_dates <= 1 { return 0.0; }
                let mut sum_means = 0.0f64;
                let mut sum_means_sq = 0.0f64;
                for (&day, &cnt) in &user_date_counts[u] {
                    let day_mean = user_date_sum[u][&day] / cnt as f64;
                    sum_means += day_mean;
                    sum_means_sq += day_mean * day_mean;
                }
                let avg = sum_means / n_dates as f64;
                let var = (sum_means_sq / n_dates as f64 - avg * avg).max(0.0);
                var.sqrt() as f32
            }).collect();

            crate::teeln!("  Deriving per-movie stats...");
            let movie_distinct_dates: Vec<u32> = movie_date_set.iter()
                .map(|m| m.len() as u32).collect();
            drop(movie_date_set);

            let movie_std: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n > 1.0 {
                    let m = movie_sum[i] / n;
                    ((movie_sum2[i] / n - m * m).max(0.0)).sqrt() as f32
                } else { 0.0 }
            }).collect();

            let movie_skewness: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n < 3.0 { return 0.0; }
                let m = movie_sum[i] / n;
                let var = (movie_sum2[i] / n - m * m).max(1e-12);
                let std = var.sqrt();
                let m3 = movie_sum3[i] / n - 3.0 * m * movie_sum2[i] / n + 2.0 * m * m * m;
                (m3 / (std * std * std)) as f32
            }).collect();

            let movie_kurtosis: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n < 4.0 { return 0.0; }
                let m = movie_sum[i] / n;
                let var = (movie_sum2[i] / n - m * m).max(1e-12);
                let m4 = movie_sum4[i] / n - 4.0 * m * movie_sum3[i] / n
                    + 6.0 * m * m * movie_sum2[i] / n - 3.0 * m * m * m * m;
                (m4 / (var * var) - 3.0) as f32
            }).collect();

            let movie_entropy: Vec<f32> = (0..n_items).map(|i| entropy5(&movie_rating_hist[i])).collect();

            let movie_pct_extreme: Vec<f32> = (0..n_items).map(|i| {
                let total = movie_cnt[i];
                if total == 0 { return 0.0; }
                (movie_rating_hist[i][0] + movie_rating_hist[i][4]) as f32 / total as f32
            }).collect();

            let movie_mode_frac: Vec<f32> = (0..n_items).map(|i| {
                let total = movie_cnt[i];
                if total == 0 { return 0.0; }
                let max_c = *movie_rating_hist[i].iter().max().unwrap();
                max_c as f32 / total as f32
            }).collect();

            let movie_date_std: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n > 1.0 {
                    let m = movie_sum_dates[i] / n;
                    ((movie_sum_sq_dates[i] / n - m * m).max(0.0)).sqrt() as f32
                } else { 0.0 }
            }).collect();

            let movie_bayesian_mean: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                ((movie_sum[i] + BAYESIAN_K * global_mean as f64) / (n + BAYESIAN_K)) as f32
            }).collect();

            let mut sorted_movie_cnts: Vec<u32> = movie_cnt.iter().copied()
                .filter(|&c| c > 0).collect();
            sorted_movie_cnts.sort_unstable();
            let median_movie_cnt = if sorted_movie_cnts.is_empty() { 1 }
                else { sorted_movie_cnts[sorted_movie_cnts.len() / 2] };

            let mut sorted_user_cnts: Vec<u32> = user_cnt.iter().copied()
                .filter(|&c| c > 0).collect();
            sorted_user_cnts.sort_unstable();
            let median_user_cnt = if sorted_user_cnts.is_empty() { 1 }
                else { sorted_user_cnts[sorted_user_cnts.len() / 2] };

            crate::teeln!("  Pass 2: Cross-entity aggregations...");
            let mut movie_sum_rater_tenure = vec![0.0f64; n_items];
            let mut movie_sum_rater_std = vec![0.0f64; n_items];
            let mut movie_heavy_rater_cnt = vec![0u32; n_items];
            let mut user_sum_movie_mean = vec![0.0f64; n_users];
            let mut user_sum_movie_date_std = vec![0.0f64; n_users];
            let mut user_popular_cnt = vec![0u32; n_users];

            for idx in 0..ds.n_ratings {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;

                let tenure = (user_last_date[u] - user_first_date[u]).max(0) as f64 + 1.0;
                movie_sum_rater_tenure[i] += tenure;
                movie_sum_rater_std[i] += user_std[u] as f64;
                if user_cnt[u] > median_user_cnt {
                    movie_heavy_rater_cnt[i] += 1;
                }

                user_sum_movie_mean[u] += movie_bayesian_mean[i] as f64;
                user_sum_movie_date_std[u] += movie_date_std[i] as f64;
                if movie_cnt[i] > median_movie_cnt {
                    user_popular_cnt[u] += 1;
                }
            }

            let movie_avg_rater_tenure: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n > 0.0 {
                    let global_avg_tenure = total_cnt as f64 / n_users as f64;
                    ((movie_sum_rater_tenure[i] + BAYESIAN_K * global_avg_tenure)
                        / (n + BAYESIAN_K)) as f32
                } else { 0.0 }
            }).collect();

            let global_avg_user_std = user_std.iter().map(|&s| s as f64).sum::<f64>()
                / n_users.max(1) as f64;
            let movie_avg_rater_std: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n > 0.0 {
                    ((movie_sum_rater_std[i] + BAYESIAN_K * global_avg_user_std)
                        / (n + BAYESIAN_K)) as f32
                } else { 0.0 }
            }).collect();

            let movie_pct_heavy_raters: Vec<f32> = (0..n_items).map(|i| {
                if movie_cnt[i] > 0 { movie_heavy_rater_cnt[i] as f32 / movie_cnt[i] as f32 }
                else { 0.0 }
            }).collect();

            let global_avg_movie_mean = movie_bayesian_mean.iter().map(|&m| m as f64).sum::<f64>()
                / n_items.max(1) as f64;
            let user_avg_movie_mean: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n > 0.0 {
                    ((user_sum_movie_mean[u] + BAYESIAN_K * global_avg_movie_mean)
                        / (n + BAYESIAN_K)) as f32
                } else { 0.0 }
            }).collect();

            let global_avg_movie_date_std = movie_date_std.iter().map(|&s| s as f64).sum::<f64>()
                / n_items.max(1) as f64;
            let user_avg_movie_date_std: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n > 0.0 {
                    ((user_sum_movie_date_std[u] + BAYESIAN_K * global_avg_movie_date_std)
                        / (n + BAYESIAN_K)) as f32
                } else { 0.0 }
            }).collect();

            let user_pct_popular: Vec<f32> = (0..n_users).map(|u| {
                if user_cnt[u] > 0 { user_popular_cnt[u] as f32 / user_cnt[u] as f32 }
                else { 0.0 }
            }).collect();

            crate::teeln!("  FX2 statistics done.");

            Self {
                user_cnt,
                user_first_date,
                user_last_date,
                user_distinct_dates,
                user_avg_day_gap,
                user_max_day_cnt,
                user_multi_day_pct,
                user_mean,
                user_std,
                user_skewness,
                user_kurtosis,
                user_entropy,
                user_pct_extreme,
                user_mode_frac,
                user_date_counts,
                user_date_sum,
                user_date_sum_sq,
                user_date_rating_counts,
                movie_cnt,
                movie_first_date,
                movie_last_date,
                movie_distinct_dates,
                movie_std,
                movie_skewness,
                movie_kurtosis,
                movie_entropy,
                movie_pct_extreme,
                movie_mode_frac,
                movie_avg_rater_tenure,
                movie_avg_rater_std,
                user_avg_movie_mean,
                user_avg_movie_date_std,
                user_pct_popular,
                movie_pct_heavy_raters,
                user_date_bias_std,
            }
        }

        #[inline]
        pub fn compute(&self, u: usize, i: usize, day: i16) -> [f32; N_FEATURES] {
            let mut f = [0.0_f32; N_FEATURES];

            // G1
            let movie_age = (day - self.movie_first_date[i]).max(0) as f32 + 1.0;
            f[G1_MOVIE_LOG_AGE] = (movie_age as f64).ln() as f32;

            let user_span = (self.user_last_date[u] - self.user_first_date[u]).max(0) as f64 + 1.0;
            f[G1_USER_RATING_VELOCITY] = (self.user_cnt[u] as f64 / user_span).ln() as f32;

            let movie_span = (self.movie_last_date[i] - self.movie_first_date[i]).max(0) as f64 + 1.0;
            f[G1_MOVIE_RATING_VELOCITY] = (self.movie_cnt[i] as f64 / movie_span).ln() as f32;

            let user_range = (self.user_last_date[u] - self.user_first_date[u]) as f32;
            f[G1_USER_DATE_POSITION] = if user_range > 0.0 {
                ((day - self.user_first_date[u]) as f32 / user_range).clamp(0.0, 1.0)
            } else { 0.5 };

            let movie_range = (self.movie_last_date[i] - self.movie_first_date[i]) as f32;
            f[G1_MOVIE_DATE_POSITION] = if movie_range > 0.0 {
                ((day - self.movie_first_date[i]) as f32 / movie_range).clamp(0.0, 1.0)
            } else { 0.5 };

            f[G1_LOG_DAY] = ((day as f32 + 1.0).max(1.0) as f64).ln() as f32;
            f[G1_DAY_SIN7] = (2.0 * std::f64::consts::PI * day as f64 / 7.0).sin() as f32;
            f[G1_DAY_SIN30] = (2.0 * std::f64::consts::PI * day as f64 / 30.44).sin() as f32;
            f[G1_USER_LOG_AVG_DAY_GAP] = ((self.user_avg_day_gap[u] + 1.0) as f64).ln() as f32;
            f[G1_MOVIE_LOG_DISTINCT_DATES] = ((self.movie_distinct_dates[i].max(1) as f32) as f64).ln() as f32;

            // G2
            f[G2_USER_SKEWNESS] = self.user_skewness[u];
            f[G2_MOVIE_SKEWNESS] = self.movie_skewness[i];
            f[G2_USER_KURTOSIS] = self.user_kurtosis[u];
            f[G2_MOVIE_KURTOSIS] = self.movie_kurtosis[i];
            f[G2_USER_ENTROPY] = self.user_entropy[u];
            f[G2_MOVIE_ENTROPY] = self.movie_entropy[i];
            f[G2_USER_PCT_EXTREME] = self.user_pct_extreme[u];
            f[G2_MOVIE_PCT_EXTREME] = self.movie_pct_extreme[i];
            f[G2_USER_MODE_FRAC] = self.user_mode_frac[u];
            f[G2_MOVIE_MODE_FRAC] = self.movie_mode_frac[i];

            // G3
            f[G3_MOVIE_AVG_RATER_TENURE] = ((self.movie_avg_rater_tenure[i].max(1.0)) as f64).ln() as f32;
            f[G3_USER_AVG_MOVIE_DATE_STD] = self.user_avg_movie_date_std[u];
            let uc = self.user_cnt[u].max(1) as f64;
            let mc = self.movie_cnt[i].max(1) as f64;
            f[G3_LOG_CNT_GEOMETRIC] = (uc * mc).sqrt().ln() as f32;
            let harmonic = if uc + mc > 0.0 {
                (2.0 * uc * mc / (uc + mc)).max(1.0).ln()
            } else { 0.0 };
            f[G3_LOG_CNT_HARMONIC] = harmonic as f32;
            f[G3_USER_STD_X_MOVIE_STD] = self.user_std[u] * self.movie_std[i];
            f[G3_MOVIE_AVG_RATER_STD] = self.movie_avg_rater_std[i];
            f[G3_USER_AVG_MOVIE_MEAN] = self.user_avg_movie_mean[u];
            f[G3_LOG_USER_CNT_OVER_MOVIE_CNT] = (uc / mc).ln() as f32;
            let log_tenure = user_span.ln();
            let log_movie_cnt = mc.ln();
            f[G3_USER_TENURE_X_LOG_MOVIE_CNT] = (log_tenure * log_movie_cnt) as f32;
            f[G3_USER_DATE_BIAS_STD_X_MOVIE_STD] = self.user_date_bias_std[u] * self.movie_std[i];

            // G4
            let dc = self.user_date_rating_counts[u].get(&day)
                .copied().unwrap_or([0u32; 5]);
            let dc_total: u32 = dc.iter().sum();
            if dc_total > 0 {
                let inv = 1.0 / dc_total as f32;
                f[G4_PCT_R1_SAME_DAY] = dc[0] as f32 * inv;
                f[G4_PCT_R2_SAME_DAY] = dc[1] as f32 * inv;
                f[G4_PCT_R3_SAME_DAY] = dc[2] as f32 * inv;
                f[G4_PCT_R4_SAME_DAY] = dc[3] as f32 * inv;
                f[G4_PCT_R5_SAME_DAY] = dc[4] as f32 * inv;
            }

            // G5
            let nd = self.user_distinct_dates[u].max(1) as f32;
            f[G5_USER_LOG_AVG_BATCH] = (uc / nd as f64).ln() as f32;
            f[G5_USER_MULTI_DAY_PCT] = self.user_multi_day_pct[u];

            let mdd = self.movie_distinct_dates[i].max(1) as f32;
            f[G5_MOVIE_LOG_RATINGS_PER_DATE] = (mc / mdd as f64).ln() as f32;

            f[G5_USER_LOG_MAX_DAY_CNT] = ((self.user_max_day_cnt[u] as f32 + 1.0) as f64).ln() as f32;

            let day_cnt = self.user_date_counts[u].get(&day).copied().unwrap_or(0);
            if day_cnt > 0 {
                let day_sum = self.user_date_sum[u].get(&day).copied().unwrap_or(0.0);
                let day_mean = day_sum as f32 / day_cnt as f32;
                f[G5_SAME_DAY_MEAN_RATING] = day_mean - self.user_mean[u];

                if day_cnt > 1 {
                    let day_sum_sq = self.user_date_sum_sq[u].get(&day).copied().unwrap_or(0.0);
                    let mean_d = day_sum / day_cnt as f64;
                    let var = (day_sum_sq / day_cnt as f64 - mean_d * mean_d).max(0.0);
                    f[G5_SAME_DAY_STD_RATING] = var.sqrt() as f32;
                }
            }

            f[G5_USER_PCT_POPULAR] = self.user_pct_popular[u];
            f[G5_MOVIE_PCT_HEAVY_RATERS] = self.movie_pct_heavy_raters[i];

            let user_end_diff = (self.user_last_date[u] - day).max(0) as f32 + 1.0;
            f[G5_USER_LAST_DATE_DIFF] = (user_end_diff as f64).ln() as f32;

            let movie_end_diff = (self.movie_last_date[i] - day).max(0) as f32 + 1.0;
            f[G5_MOVIE_LAST_DATE_DIFF] = (movie_end_diff as f64).ln() as f32;

            f
        }
    }
}

// ===========================================================================
// FX3 — H1-H5 (090-129): 40 features, no sim/SVD
// ===========================================================================

mod fx3 {
    use super::{BAYESIAN_K, Selection};
    use crate::Dataset;
    use std::collections::HashMap;

    pub const N_FEATURES: usize = 40;

    // H1: temporal trends
    const H1_MOVIE_RATING_SLOPE: usize = 0;
    const H1_USER_RATING_SLOPE: usize = 1;
    const H1_MOVIE_EARLY_LATE_DIFF: usize = 2;
    const H1_USER_EARLY_LATE_DIFF: usize = 3;
    const H1_MOVIE_TEMPORAL_Z: usize = 4;
    const H1_USER_TEMPORAL_Z: usize = 5;
    // H2: rating concentration
    const H2_USER_HERFINDAHL: usize = 6;
    const H2_MOVIE_HERFINDAHL: usize = 7;
    const H2_USER_N_DISTINCT_RATINGS: usize = 8;
    const H2_USER_TOP2_CONCENTRATION: usize = 9;
    const H2_MOVIE_TOP2_CONCENTRATION: usize = 10;
    const H2_MOVIE_BIMODALITY: usize = 11;
    const H2_USER_BIMODALITY: usize = 12;
    // H3: user-movie alignment
    const H3_USER_MOVIE_MEAN_GAP: usize = 13;
    const H3_USER_MOVIE_STD_RATIO: usize = 14;
    const H3_USER_MOVIE_ENTROPY_DIFF: usize = 15;
    const H3_USER_PCT_AT_MOVIE_MODE: usize = 16;
    const H3_MOVIE_PCT_AT_USER_MODE: usize = 17;
    const H3_USER_MOVIE_DIST_OVERLAP: usize = 18;
    const H3_USER_MOVIE_DIST_COSINE: usize = 19;
    // H4: session context
    const H4_IS_SINGLE_RATING_DAY: usize = 20;
    const H4_SAME_DAY_CNT_NORM: usize = 21;
    const H4_SAME_DAY_ENTROPY: usize = 22;
    const H4_SAME_DAY_RANGE: usize = 23;
    const H4_SAME_DAY_PCT_HIGH: usize = 24;
    const H4_SAME_DAY_PCT_LOW: usize = 25;
    const H4_IS_FIRST_DATE: usize = 26;
    const H4_IS_LAST_DATE: usize = 27;
    const H4_USER_DAY_INDEX_NORM: usize = 28;
    const H4_LOG_MOVIE_SAME_DAY_CNT: usize = 29;
    // H5: item properties & adoption
    const H5_ITEM_YEAR_NORM: usize = 30;
    const H5_LOG_ITEM_AGE_AT_RATING: usize = 31;
    const H5_IS_EARLY_ADOPTER: usize = 32;
    const H5_IS_LATE_ADOPTER: usize = 33;
    const H5_MOVIE_MATURITY: usize = 34;
    const H5_USER_AVG_ITEM_YEAR: usize = 35;
    const H5_USER_ITEM_YEAR_STD: usize = 36;
    const H5_USER_PREFERS_OLD: usize = 37;
    const H5_MOVIE_AVG_RATER_MEAN: usize = 38;
    const H5_MOVIE_AVG_RATER_ENTROPY: usize = 39;

    pub const NAMES: [&'static str; N_FEATURES] = [
        "vf090_movie_rating_slope",
        "vf091_user_rating_slope",
        "vf092_movie_early_late_diff",
        "vf093_user_early_late_diff",
        "vf094_movie_temporal_z",
        "vf095_user_temporal_z",
        "vf096_user_herfindahl",
        "vf097_movie_herfindahl",
        "vf098_user_n_distinct_ratings",
        "vf099_user_top2_concentration",
        "vf100_movie_top2_concentration",
        "vf101_movie_bimodality",
        "vf102_user_bimodality",
        "vf103_user_movie_mean_gap",
        "vf104_user_movie_std_ratio",
        "vf105_user_movie_entropy_diff",
        "vf106_user_pct_at_movie_mode",
        "vf107_movie_pct_at_user_mode",
        "vf108_user_movie_dist_overlap",
        "vf109_user_movie_dist_cosine",
        "vf110_is_single_rating_day",
        "vf111_same_day_cnt_norm",
        "vf112_same_day_entropy",
        "vf113_same_day_range",
        "vf114_same_day_pct_high",
        "vf115_same_day_pct_low",
        "vf116_is_first_date",
        "vf117_is_last_date",
        "vf118_user_day_index_norm",
        "vf119_log_movie_same_day_cnt",
        "vf120_item_year_norm",
        "vf121_log_item_age_at_rating",
        "vf122_is_early_adopter",
        "vf123_is_late_adopter",
        "vf124_movie_maturity",
        "vf125_user_avg_item_year",
        "vf126_user_item_year_std",
        "vf127_user_prefers_old",
        "vf128_movie_avg_rater_mean",
        "vf129_movie_avg_rater_entropy",
    ];

    #[inline]
    fn entropy5(counts: &[u32; 5]) -> f32 {
        let total: u32 = counts.iter().sum();
        if total == 0 { return 0.0; }
        let tf = total as f64;
        let mut h = 0.0f64;
        for &c in counts {
            if c > 0 {
                let p = c as f64 / tf;
                h -= p * p.ln();
            }
        }
        h as f32
    }

    #[inline]
    fn herfindahl5(counts: &[u32; 5]) -> f32 {
        let total: u32 = counts.iter().sum();
        if total == 0 { return 0.0; }
        let tf = total as f64;
        let mut h = 0.0f64;
        for &c in counts {
            let p = c as f64 / tf;
            h += p * p;
        }
        h as f32
    }

    #[inline]
    fn n_distinct5(counts: &[u32; 5]) -> u8 {
        counts.iter().filter(|&&c| c > 0).count() as u8
    }

    #[inline]
    fn top2_conc5(counts: &[u32; 5]) -> f32 {
        let total: u32 = counts.iter().sum();
        if total == 0 { return 0.0; }
        let mut sorted = *counts;
        sorted.sort_unstable();
        (sorted[3] + sorted[4]) as f32 / total as f32
    }

    #[inline]
    fn mode5(counts: &[u32; 5]) -> u8 {
        let mut best = 0u8;
        let mut best_cnt = 0u32;
        for (k, &c) in counts.iter().enumerate() {
            if c > best_cnt {
                best = k as u8;
                best_cnt = c;
            }
        }
        best
    }

    #[inline]
    fn rating_dist5(counts: &[u32; 5]) -> [f32; 5] {
        let total: u32 = counts.iter().sum();
        if total == 0 { return [0.2; 5]; }
        let inv = 1.0 / total as f32;
        [
            counts[0] as f32 * inv,
            counts[1] as f32 * inv,
            counts[2] as f32 * inv,
            counts[3] as f32 * inv,
            counts[4] as f32 * inv,
        ]
    }

    pub struct Fx3Stats {
        movie_rating_slope: Vec<f32>,
        user_rating_slope: Vec<f32>,
        movie_early_late_diff: Vec<f32>,
        user_early_late_diff: Vec<f32>,
        movie_mean_day: Vec<f32>,
        movie_date_std: Vec<f32>,
        user_mean_day: Vec<f32>,
        user_date_std: Vec<f32>,
        user_herfindahl: Vec<f32>,
        movie_herfindahl: Vec<f32>,
        user_n_distinct: Vec<u8>,
        user_top2_conc: Vec<f32>,
        movie_top2_conc: Vec<f32>,
        movie_bimodality: Vec<f32>,
        user_bimodality: Vec<f32>,
        user_mean: Vec<f32>,
        movie_mean: Vec<f32>,
        user_std: Vec<f32>,
        movie_std: Vec<f32>,
        user_entropy: Vec<f32>,
        movie_entropy: Vec<f32>,
        user_mode: Vec<u8>,
        movie_mode: Vec<u8>,
        user_rating_dist: Vec<[f32; 5]>,
        movie_rating_dist: Vec<[f32; 5]>,
        user_first_date: Vec<i16>,
        user_last_date: Vec<i16>,
        user_date_counts: Vec<HashMap<i16, u32>>,
        user_date_rating_counts: Vec<HashMap<i16, [u32; 5]>>,
        user_sorted_dates: Vec<Vec<i16>>,
        movie_date_counts: Vec<HashMap<i16, u32>>,
        item_year_norm: Vec<f32>,
        item_release_day: Vec<f32>,
        movie_first_date: Vec<i16>,
        movie_last_date: Vec<i16>,
        movie_date_cdf: Vec<Vec<(i16, u32)>>,
        user_avg_item_year: Vec<f32>,
        user_item_year_std: Vec<f32>,
        user_prefers_old: Vec<f32>,
        movie_avg_rater_mean: Vec<f32>,
        movie_avg_rater_entropy: Vec<f32>,
    }

    impl Fx3Stats {
        pub fn new(ds: &Dataset, _sel: &Selection) -> Self {
            crate::teeln!("Computing FX3 statistics...");
            let n_users = ds.n_users;
            let n_items = ds.n_items;

            crate::teeln!("  Pass 1: Basic statistics...");
            let mut user_cnt = vec![0u32; n_users];
            let mut user_sum = vec![0.0f64; n_users];
            let mut user_sum2 = vec![0.0f64; n_users];
            let mut user_sum3 = vec![0.0f64; n_users];
            let mut user_sum4 = vec![0.0f64; n_users];
            let mut user_first_date = vec![i16::MAX; n_users];
            let mut user_last_date = vec![i16::MIN; n_users];
            let mut user_sum_day = vec![0.0f64; n_users];
            let mut user_sum_day2 = vec![0.0f64; n_users];
            let mut user_sum_r_day = vec![0.0f64; n_users];
            let mut user_date_counts: Vec<HashMap<i16, u32>> = vec![HashMap::new(); n_users];
            let mut user_date_rating_counts: Vec<HashMap<i16, [u32; 5]>> = vec![HashMap::new(); n_users];
            let mut user_rating_hist: Vec<[u32; 5]> = vec![[0u32; 5]; n_users];
            let mut user_sum_item_year = vec![0.0f64; n_users];
            let mut user_sum_item_year2 = vec![0.0f64; n_users];

            let mut movie_cnt = vec![0u32; n_items];
            let mut movie_sum = vec![0.0f64; n_items];
            let mut movie_sum2 = vec![0.0f64; n_items];
            let mut movie_sum3 = vec![0.0f64; n_items];
            let mut movie_sum4 = vec![0.0f64; n_items];
            let mut movie_first_date = vec![i16::MAX; n_items];
            let mut movie_last_date = vec![i16::MIN; n_items];
            let mut movie_sum_day = vec![0.0f64; n_items];
            let mut movie_sum_day2 = vec![0.0f64; n_items];
            let mut movie_sum_r_day = vec![0.0f64; n_items];
            let mut movie_date_counts: Vec<HashMap<i16, u32>> = vec![HashMap::new(); n_items];
            let mut movie_rating_hist: Vec<[u32; 5]> = vec![[0u32; 5]; n_items];

            let mut total_sum = 0.0f64;
            let mut total_cnt = 0u64;

            for idx in 0..ds.n_ratings {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;
                let r = ds.raw_ratings[idx];
                let rf = r as f64;
                let day = ds.dates[idx];
                let df = day as f64;

                user_cnt[u] += 1;
                user_sum[u] += rf;
                user_sum2[u] += rf * rf;
                user_sum3[u] += rf * rf * rf;
                user_sum4[u] += rf * rf * rf * rf;
                if day < user_first_date[u] { user_first_date[u] = day; }
                if day > user_last_date[u] { user_last_date[u] = day; }
                user_sum_day[u] += df;
                user_sum_day2[u] += df * df;
                user_sum_r_day[u] += rf * df;
                *user_date_counts[u].entry(day).or_insert(0) += 1;
                if r >= 1 && r <= 5 {
                    let rv = (r - 1) as usize;
                    user_rating_hist[u][rv] += 1;
                    let entry = user_date_rating_counts[u].entry(day).or_insert([0u32; 5]);
                    entry[rv] += 1;
                }
                let iy = ds.item_years[i] as f64;
                user_sum_item_year[u] += iy;
                user_sum_item_year2[u] += iy * iy;

                movie_cnt[i] += 1;
                movie_sum[i] += rf;
                movie_sum2[i] += rf * rf;
                movie_sum3[i] += rf * rf * rf;
                movie_sum4[i] += rf * rf * rf * rf;
                if day < movie_first_date[i] { movie_first_date[i] = day; }
                if day > movie_last_date[i] { movie_last_date[i] = day; }
                movie_sum_day[i] += df;
                movie_sum_day2[i] += df * df;
                movie_sum_r_day[i] += rf * df;
                *movie_date_counts[i].entry(day).or_insert(0) += 1;
                if r >= 1 && r <= 5 {
                    movie_rating_hist[i][(r - 1) as usize] += 1;
                }

                total_sum += rf;
                total_cnt += 1;
            }

            let global_mean = (total_sum / total_cnt as f64) as f32;

            crate::teeln!("  Deriving per-user stats...");
            let user_mean: Vec<f32> = (0..n_users).map(|u| {
                if user_cnt[u] > 0 { (user_sum[u] / user_cnt[u] as f64) as f32 } else { global_mean }
            }).collect();

            let user_std: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n > 1.0 {
                    let m = user_sum[u] / n;
                    ((user_sum2[u] / n - m * m).max(0.0)).sqrt() as f32
                } else { 0.0 }
            }).collect();

            let user_rating_slope: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n < 2.0 { return 0.0; }
                let denom = n * user_sum_day2[u] - user_sum_day[u] * user_sum_day[u];
                if denom.abs() < 1e-10 { return 0.0; }
                let numer = n * user_sum_r_day[u] - user_sum[u] * user_sum_day[u];
                (numer / denom) as f32
            }).collect();

            let user_mean_day: Vec<f32> = (0..n_users).map(|u| {
                if user_cnt[u] > 0 { (user_sum_day[u] / user_cnt[u] as f64) as f32 } else { 0.0 }
            }).collect();

            let user_date_std_vec: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n > 1.0 {
                    let m = user_sum_day[u] / n;
                    ((user_sum_day2[u] / n - m * m).max(0.0)).sqrt() as f32
                } else { 0.0 }
            }).collect();

            let user_skewness: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n < 3.0 { return 0.0; }
                let m = user_sum[u] / n;
                let var = (user_sum2[u] / n - m * m).max(1e-12);
                let std = var.sqrt();
                let m3 = user_sum3[u] / n - 3.0 * m * user_sum2[u] / n + 2.0 * m * m * m;
                (m3 / (std * std * std)) as f32
            }).collect();

            let user_kurtosis: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n < 4.0 { return 0.0; }
                let m = user_sum[u] / n;
                let var = (user_sum2[u] / n - m * m).max(1e-12);
                let m4 = user_sum4[u] / n - 4.0 * m * user_sum3[u] / n
                    + 6.0 * m * m * user_sum2[u] / n - 3.0 * m * m * m * m;
                (m4 / (var * var) - 3.0) as f32
            }).collect();

            let user_entropy: Vec<f32> = (0..n_users).map(|u| entropy5(&user_rating_hist[u])).collect();
            let user_herfindahl: Vec<f32> = (0..n_users).map(|u| herfindahl5(&user_rating_hist[u])).collect();
            let user_n_distinct: Vec<u8> = (0..n_users).map(|u| n_distinct5(&user_rating_hist[u])).collect();
            let user_top2_conc: Vec<f32> = (0..n_users).map(|u| top2_conc5(&user_rating_hist[u])).collect();
            let user_mode_val: Vec<u8> = (0..n_users).map(|u| mode5(&user_rating_hist[u])).collect();
            let user_rating_dist_vec: Vec<[f32; 5]> = (0..n_users).map(|u| rating_dist5(&user_rating_hist[u])).collect();

            let user_bimodality: Vec<f32> = (0..n_users).map(|u| {
                let s = user_skewness[u] as f64;
                let k = user_kurtosis[u] as f64;
                let denom = k + 3.0;
                if denom.abs() < 1e-10 { return 0.0; }
                ((s * s + 1.0) / denom) as f32
            }).collect();

            let user_sorted_dates: Vec<Vec<i16>> = user_date_counts.iter().map(|m| {
                let mut dates: Vec<i16> = m.keys().copied().collect();
                dates.sort_unstable();
                dates
            }).collect();

            let user_avg_item_year: Vec<f32> = (0..n_users).map(|u| {
                if user_cnt[u] > 0 {
                    (user_sum_item_year[u] / user_cnt[u] as f64) as f32
                } else { 0.0 }
            }).collect();

            let user_item_year_std: Vec<f32> = (0..n_users).map(|u| {
                let n = user_cnt[u] as f64;
                if n > 1.0 {
                    let m = user_sum_item_year[u] / n;
                    ((user_sum_item_year2[u] / n - m * m).max(0.0)).sqrt() as f32
                } else { 0.0 }
            }).collect();

            crate::teeln!("  Deriving per-movie stats...");
            let movie_mean: Vec<f32> = (0..n_items).map(|i| {
                if movie_cnt[i] > 0 { (movie_sum[i] / movie_cnt[i] as f64) as f32 } else { global_mean }
            }).collect();

            let movie_std: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n > 1.0 {
                    let m = movie_sum[i] / n;
                    ((movie_sum2[i] / n - m * m).max(0.0)).sqrt() as f32
                } else { 0.0 }
            }).collect();

            let movie_rating_slope: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n < 2.0 { return 0.0; }
                let denom = n * movie_sum_day2[i] - movie_sum_day[i] * movie_sum_day[i];
                if denom.abs() < 1e-10 { return 0.0; }
                let numer = n * movie_sum_r_day[i] - movie_sum[i] * movie_sum_day[i];
                (numer / denom) as f32
            }).collect();

            let movie_mean_day: Vec<f32> = (0..n_items).map(|i| {
                if movie_cnt[i] > 0 { (movie_sum_day[i] / movie_cnt[i] as f64) as f32 } else { 0.0 }
            }).collect();

            let movie_date_std_vec: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n > 1.0 {
                    let m = movie_sum_day[i] / n;
                    ((movie_sum_day2[i] / n - m * m).max(0.0)).sqrt() as f32
                } else { 0.0 }
            }).collect();

            let movie_skewness: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n < 3.0 { return 0.0; }
                let m = movie_sum[i] / n;
                let var = (movie_sum2[i] / n - m * m).max(1e-12);
                let std = var.sqrt();
                let m3 = movie_sum3[i] / n - 3.0 * m * movie_sum2[i] / n + 2.0 * m * m * m;
                (m3 / (std * std * std)) as f32
            }).collect();

            let movie_kurtosis: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n < 4.0 { return 0.0; }
                let m = movie_sum[i] / n;
                let var = (movie_sum2[i] / n - m * m).max(1e-12);
                let m4 = movie_sum4[i] / n - 4.0 * m * movie_sum3[i] / n
                    + 6.0 * m * m * movie_sum2[i] / n - 3.0 * m * m * m * m;
                (m4 / (var * var) - 3.0) as f32
            }).collect();

            let movie_entropy: Vec<f32> = (0..n_items).map(|i| entropy5(&movie_rating_hist[i])).collect();
            let movie_herfindahl: Vec<f32> = (0..n_items).map(|i| herfindahl5(&movie_rating_hist[i])).collect();
            let movie_top2_conc: Vec<f32> = (0..n_items).map(|i| top2_conc5(&movie_rating_hist[i])).collect();
            let movie_mode_val: Vec<u8> = (0..n_items).map(|i| mode5(&movie_rating_hist[i])).collect();
            let movie_rating_dist_vec: Vec<[f32; 5]> = (0..n_items).map(|i| rating_dist5(&movie_rating_hist[i])).collect();

            let movie_bimodality: Vec<f32> = (0..n_items).map(|i| {
                let s = movie_skewness[i] as f64;
                let k = movie_kurtosis[i] as f64;
                let denom = k + 3.0;
                if denom.abs() < 1e-10 { return 0.0; }
                ((s * s + 1.0) / denom) as f32
            }).collect();

            crate::teeln!("  Building movie date CDFs...");
            let movie_date_cdf: Vec<Vec<(i16, u32)>> = (0..n_items).map(|i| {
                let mut entries: Vec<(i16, u32)> = movie_date_counts[i].iter()
                    .map(|(&d, &c)| (d, c)).collect();
                entries.sort_unstable_by_key(|&(d, _)| d);
                let mut cum = 0u32;
                for e in entries.iter_mut() {
                    cum += e.1;
                    e.1 = cum;
                }
                entries
            }).collect();

            crate::teeln!("  Item year normalization...");
            let mut year_sum = 0.0f64;
            let mut year_sum2 = 0.0f64;
            let mut year_cnt = 0u64;
            for i in 0..n_items {
                let y = ds.item_years[i] as f64;
                if y > 0.0 {
                    year_sum += y;
                    year_sum2 += y * y;
                    year_cnt += 1;
                }
            }
            let year_mean = if year_cnt > 0 { year_sum / year_cnt as f64 } else { 1990.0 };
            let year_std = if year_cnt > 1 {
                ((year_sum2 / year_cnt as f64 - year_mean * year_mean).max(0.0)).sqrt().max(1.0)
            } else { 1.0 };

            let item_year_norm: Vec<f32> = (0..n_items).map(|i| {
                ((ds.item_years[i] as f64 - year_mean) / year_std) as f32
            }).collect();

            let item_release_day: Vec<f32> = (0..n_items).map(|i| {
                ((ds.item_years[i] as f64 - 1999.0) * 365.25 - 314.0) as f32
            }).collect();

            let mut valid_years: Vec<i32> = (0..n_items)
                .filter(|&i| movie_cnt[i] > 0 && ds.item_years[i] > 0)
                .map(|i| ds.item_years[i])
                .collect();
            valid_years.sort_unstable();
            let median_year = if valid_years.is_empty() { 1990 }
                else { valid_years[valid_years.len() / 2] };

            crate::teeln!("  Pass 2: Early/late diff & cross-entity...");
            let mut movie_early_sum = vec![0.0f64; n_items];
            let mut movie_early_cnt = vec![0u32; n_items];
            let mut movie_late_sum = vec![0.0f64; n_items];
            let mut movie_late_cnt = vec![0u32; n_items];

            let mut user_early_sum = vec![0.0f64; n_users];
            let mut user_early_cnt = vec![0u32; n_users];
            let mut user_late_sum = vec![0.0f64; n_users];
            let mut user_late_cnt = vec![0u32; n_users];

            let mut movie_sum_rater_mean = vec![0.0f64; n_items];
            let mut movie_sum_rater_entropy = vec![0.0f64; n_items];

            let mut user_old_cnt = vec![0u32; n_users];

            for idx in 0..ds.n_ratings {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;
                let r = ds.raw_ratings[idx] as f64;
                let day = ds.dates[idx];

                if (day as f32) <= movie_mean_day[i] {
                    movie_early_sum[i] += r;
                    movie_early_cnt[i] += 1;
                } else {
                    movie_late_sum[i] += r;
                    movie_late_cnt[i] += 1;
                }

                if (day as f32) <= user_mean_day[u] {
                    user_early_sum[u] += r;
                    user_early_cnt[u] += 1;
                } else {
                    user_late_sum[u] += r;
                    user_late_cnt[u] += 1;
                }

                movie_sum_rater_mean[i] += user_mean[u] as f64;
                movie_sum_rater_entropy[i] += user_entropy[u] as f64;

                if ds.item_years[i] > 0 && ds.item_years[i] < median_year {
                    user_old_cnt[u] += 1;
                }
            }

            let movie_early_late_diff: Vec<f32> = (0..n_items).map(|i| {
                let early_m = if movie_early_cnt[i] > 0 {
                    movie_early_sum[i] / movie_early_cnt[i] as f64
                } else { 0.0 };
                let late_m = if movie_late_cnt[i] > 0 {
                    movie_late_sum[i] / movie_late_cnt[i] as f64
                } else { 0.0 };
                (late_m - early_m) as f32
            }).collect();

            let user_early_late_diff: Vec<f32> = (0..n_users).map(|u| {
                let early_m = if user_early_cnt[u] > 0 {
                    user_early_sum[u] / user_early_cnt[u] as f64
                } else { 0.0 };
                let late_m = if user_late_cnt[u] > 0 {
                    user_late_sum[u] / user_late_cnt[u] as f64
                } else { 0.0 };
                (late_m - early_m) as f32
            }).collect();

            let global_avg_user_mean = user_mean.iter().map(|&m| m as f64).sum::<f64>()
                / n_users.max(1) as f64;
            let movie_avg_rater_mean: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n > 0.0 {
                    ((movie_sum_rater_mean[i] + BAYESIAN_K * global_avg_user_mean)
                        / (n + BAYESIAN_K)) as f32
                } else { global_mean }
            }).collect();

            let global_avg_user_entropy = user_entropy.iter().map(|&e| e as f64).sum::<f64>()
                / n_users.max(1) as f64;
            let movie_avg_rater_entropy: Vec<f32> = (0..n_items).map(|i| {
                let n = movie_cnt[i] as f64;
                if n > 0.0 {
                    ((movie_sum_rater_entropy[i] + BAYESIAN_K * global_avg_user_entropy)
                        / (n + BAYESIAN_K)) as f32
                } else { global_avg_user_entropy as f32 }
            }).collect();

            let user_prefers_old: Vec<f32> = (0..n_users).map(|u| {
                if user_cnt[u] > 0 { user_old_cnt[u] as f32 / user_cnt[u] as f32 }
                else { 0.5 }
            }).collect();

            crate::teeln!("  FX3 statistics done.");

            Self {
                movie_rating_slope,
                user_rating_slope,
                movie_early_late_diff,
                user_early_late_diff,
                movie_mean_day,
                movie_date_std: movie_date_std_vec,
                user_mean_day,
                user_date_std: user_date_std_vec,
                user_herfindahl,
                movie_herfindahl,
                user_n_distinct,
                user_top2_conc,
                movie_top2_conc,
                movie_bimodality,
                user_bimodality,
                user_mean,
                movie_mean,
                user_std,
                movie_std,
                user_entropy,
                movie_entropy,
                user_mode: user_mode_val,
                movie_mode: movie_mode_val,
                user_rating_dist: user_rating_dist_vec,
                movie_rating_dist: movie_rating_dist_vec,
                user_first_date,
                user_last_date,
                user_date_counts,
                user_date_rating_counts,
                user_sorted_dates,
                movie_date_counts,
                item_year_norm,
                item_release_day,
                movie_first_date,
                movie_last_date,
                movie_date_cdf,
                user_avg_item_year,
                user_item_year_std,
                user_prefers_old,
                movie_avg_rater_mean,
                movie_avg_rater_entropy,
            }
        }

        #[inline]
        pub fn compute(&self, u: usize, i: usize, day: i16) -> [f32; N_FEATURES] {
            let mut f = [0.0_f32; N_FEATURES];

            // H1
            f[H1_MOVIE_RATING_SLOPE] = self.movie_rating_slope[i];
            f[H1_USER_RATING_SLOPE] = self.user_rating_slope[u];
            f[H1_MOVIE_EARLY_LATE_DIFF] = self.movie_early_late_diff[i];
            f[H1_USER_EARLY_LATE_DIFF] = self.user_early_late_diff[u];

            if self.movie_date_std[i] > 0.0 {
                f[H1_MOVIE_TEMPORAL_Z] = (day as f32 - self.movie_mean_day[i]) / self.movie_date_std[i];
            }
            if self.user_date_std[u] > 0.0 {
                f[H1_USER_TEMPORAL_Z] = (day as f32 - self.user_mean_day[u]) / self.user_date_std[u];
            }

            // H2
            f[H2_USER_HERFINDAHL] = self.user_herfindahl[u];
            f[H2_MOVIE_HERFINDAHL] = self.movie_herfindahl[i];
            f[H2_USER_N_DISTINCT_RATINGS] = self.user_n_distinct[u] as f32;
            f[H2_USER_TOP2_CONCENTRATION] = self.user_top2_conc[u];
            f[H2_MOVIE_TOP2_CONCENTRATION] = self.movie_top2_conc[i];
            f[H2_MOVIE_BIMODALITY] = self.movie_bimodality[i];
            f[H2_USER_BIMODALITY] = self.user_bimodality[u];

            // H3
            f[H3_USER_MOVIE_MEAN_GAP] = (self.user_mean[u] - self.movie_mean[i]).abs();

            let us = self.user_std[u] + 0.01;
            let ms = self.movie_std[i] + 0.01;
            f[H3_USER_MOVIE_STD_RATIO] = ((us / ms) as f64).ln() as f32;

            f[H3_USER_MOVIE_ENTROPY_DIFF] = self.user_entropy[u] - self.movie_entropy[i];

            let mm = self.movie_mode[i] as usize;
            f[H3_USER_PCT_AT_MOVIE_MODE] = self.user_rating_dist[u][mm];

            let um = self.user_mode[u] as usize;
            f[H3_MOVIE_PCT_AT_USER_MODE] = self.movie_rating_dist[i][um];

            let ud = &self.user_rating_dist[u];
            let md = &self.movie_rating_dist[i];
            let mut overlap = 0.0f32;
            for k in 0..5 {
                overlap += ud[k].min(md[k]);
            }
            f[H3_USER_MOVIE_DIST_OVERLAP] = overlap;

            let mut dot = 0.0f32;
            let mut norm_u = 0.0f32;
            let mut norm_m = 0.0f32;
            for k in 0..5 {
                dot += ud[k] * md[k];
                norm_u += ud[k] * ud[k];
                norm_m += md[k] * md[k];
            }
            let denom = (norm_u * norm_m).sqrt();
            f[H3_USER_MOVIE_DIST_COSINE] = if denom > 1e-8 { dot / denom } else { 0.0 };

            // H4
            let day_cnt = self.user_date_counts[u].get(&day).copied().unwrap_or(0);
            f[H4_IS_SINGLE_RATING_DAY] = if day_cnt == 1 { 1.0 } else { 0.0 };
            f[H4_SAME_DAY_CNT_NORM] = (day_cnt as f32).min(50.0) / 50.0;

            let dc = self.user_date_rating_counts[u].get(&day)
                .copied().unwrap_or([0u32; 5]);
            f[H4_SAME_DAY_ENTROPY] = entropy5(&dc);

            let mut min_r = 5u8;
            let mut max_r = 0u8;
            for k in 0..5u8 {
                if dc[k as usize] > 0 {
                    if k < min_r { min_r = k; }
                    max_r = k;
                }
            }
            f[H4_SAME_DAY_RANGE] = if max_r >= min_r { (max_r - min_r) as f32 } else { 0.0 };

            let dc_total: u32 = dc.iter().sum();
            if dc_total > 0 {
                let inv = 1.0 / dc_total as f32;
                f[H4_SAME_DAY_PCT_HIGH] = (dc[3] + dc[4]) as f32 * inv;
                f[H4_SAME_DAY_PCT_LOW] = (dc[0] + dc[1]) as f32 * inv;
            }

            f[H4_IS_FIRST_DATE] = if day == self.user_first_date[u] { 1.0 } else { 0.0 };
            f[H4_IS_LAST_DATE] = if day == self.user_last_date[u] { 1.0 } else { 0.0 };

            let dates = &self.user_sorted_dates[u];
            if dates.len() > 1 {
                let pos = match dates.binary_search(&day) {
                    Ok(p) => p,
                    Err(p) => p.min(dates.len() - 1),
                };
                f[H4_USER_DAY_INDEX_NORM] = pos as f32 / (dates.len() - 1) as f32;
            } else {
                f[H4_USER_DAY_INDEX_NORM] = 0.5;
            }

            let movie_day_cnt = self.movie_date_counts[i].get(&day).copied().unwrap_or(0);
            f[H4_LOG_MOVIE_SAME_DAY_CNT] = ((1.0 + movie_day_cnt as f32) as f64).ln() as f32;

            // H5
            f[H5_ITEM_YEAR_NORM] = self.item_year_norm[i];

            let age = (day as f32 - self.item_release_day[i]).max(1.0);
            f[H5_LOG_ITEM_AGE_AT_RATING] = (age as f64).ln() as f32;

            let movie_range = (self.movie_last_date[i] - self.movie_first_date[i]) as f32;
            if movie_range > 0.0 {
                let pos = (day - self.movie_first_date[i]) as f32 / movie_range;
                f[H5_IS_EARLY_ADOPTER] = if pos <= 0.2 { 1.0 } else { 0.0 };
                f[H5_IS_LATE_ADOPTER] = if pos >= 0.8 { 1.0 } else { 0.0 };
            }

            let cdf = &self.movie_date_cdf[i];
            if !cdf.is_empty() {
                let total = cdf.last().unwrap().1;
                let cum = match cdf.binary_search_by_key(&day, |&(d, _)| d) {
                    Ok(p) => cdf[p].1,
                    Err(0) => 0,
                    Err(p) => cdf[p - 1].1,
                };
                if total > 0 {
                    f[H5_MOVIE_MATURITY] = cum as f32 / total as f32;
                }
            }

            f[H5_USER_AVG_ITEM_YEAR] = self.user_avg_item_year[u];
            f[H5_USER_ITEM_YEAR_STD] = self.user_item_year_std[u];
            f[H5_USER_PREFERS_OLD] = self.user_prefers_old[u];
            f[H5_MOVIE_AVG_RATER_MEAN] = self.movie_avg_rater_mean[i];
            f[H5_MOVIE_AVG_RATER_ENTROPY] = self.movie_avg_rater_entropy[i];

            f
        }
    }
}

// ===========================================================================
// Public wrapper struct
// ===========================================================================

pub const N_FEATURES: usize = 130;

pub struct VotingFeatures1 {
    fx1: Option<fx1::Fx1Stats>,
    fx2: Option<fx2::Fx2Stats>,
    fx3: Option<fx3::Fx3Stats>,
}

impl VotingFeatures1 {
    pub fn new(ds: &Dataset, sel: &Selection) -> Self {
        let needs_fx1 = fx1::NAMES.iter().any(|n| sel.includes(n));
        let needs_fx2 = fx2::NAMES.iter().any(|n| sel.includes(n));
        let needs_fx3 = fx3::NAMES.iter().any(|n| sel.includes(n));

        Self {
            fx1: if needs_fx1 { Some(fx1::Fx1Stats::new(ds, sel)) } else { None },
            fx2: if needs_fx2 { Some(fx2::Fx2Stats::new(ds, sel)) } else { None },
            fx3: if needs_fx3 { Some(fx3::Fx3Stats::new(ds, sel)) } else { None },
        }
    }

    pub fn feature_names() -> [&'static str; N_FEATURES] {
        let mut names: [&'static str; N_FEATURES] = [""; N_FEATURES];
        for (k, n) in fx1::NAMES.iter().enumerate() { names[k] = n; }
        for (k, n) in fx2::NAMES.iter().enumerate() { names[fx1::N_FEATURES + k] = n; }
        for (k, n) in fx3::NAMES.iter().enumerate() {
            names[fx1::N_FEATURES + fx2::N_FEATURES + k] = n;
        }
        names
    }

    #[inline]
    pub fn compute(&self, u: usize, i: usize, day: i16) -> [f32; N_FEATURES] {
        let mut all = [0.0f32; N_FEATURES];
        if let Some(s) = &self.fx1 {
            let f = s.compute(u, i, day);
            all[0..fx1::N_FEATURES].copy_from_slice(&f);
        }
        if let Some(s) = &self.fx2 {
            let f = s.compute(u, i, day);
            all[fx1::N_FEATURES..fx1::N_FEATURES + fx2::N_FEATURES].copy_from_slice(&f);
        }
        if let Some(s) = &self.fx3 {
            let f = s.compute(u, i, day);
            all[fx1::N_FEATURES + fx2::N_FEATURES..N_FEATURES].copy_from_slice(&f);
        }
        all
    }

    pub fn compute_all(&self, ds: &Dataset) -> Vec<Array1<f32>> {
        crate::teeln!("Computing {} features for {} ratings...", N_FEATURES, ds.n_ratings);

        let mut features: Vec<Array1<f32>> = (0..N_FEATURES)
            .map(|_| Array1::<f32>::zeros(ds.n_ratings))
            .collect();

        struct SendPtr(*mut f32);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let ptrs: Vec<SendPtr> = features.iter_mut()
            .map(|arr| SendPtr(arr.as_slice_mut().unwrap().as_mut_ptr()))
            .collect();

        let pb = crate::make_pb(ds.n_ratings as u64);
        (0..ds.n_ratings).into_par_iter()
            .progress_with(pb)
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
}

// ===========================================================================
// Save harness
// ===========================================================================

fn save_features(features: &[Array1<f32>], sel: &Selection, dir: &str, ds_name: &str) {
    let names = VotingFeatures1::feature_names();
    for (k, feat) in features.iter().enumerate() {
        if !sel.includes(names[k]) { continue; }
        let path = format!("{}/{}.{}.npy", dir, names[k], ds_name);
        write_npy(&path, feat).unwrap();
        crate::teeln!("vfeat1: saved {}", path);
    }
}

/// Compute and save voting feature set #1 over the standard split cycle:
/// train on `split.tr` and emit features over `split.pr`, then train on
/// `split.fulltrain_tr` and emit features over `split.fulltrain_pr`.
///
/// `name` is used as the job identifier — the run log is written to
/// `<split.preds_dir>/<name>.out`. `sel` controls which feature names get
/// written; features whose underlying statistics are not needed by `sel`
/// are not computed at all.
pub fn save_vfeat1(name: &str, sel: Selection, split: Split) {
    std::fs::create_dir_all(split.preds_dir).unwrap();

    let owns_log = crate::LOG_FILE.lock().unwrap().is_none();
    if owns_log {
        *crate::LOG_FILE.lock().unwrap() = Some(BufWriter::new(
            File::create(format!("{}/{}.out", split.preds_dir, name)).unwrap()
        ));
    }

    crate::teeln!("[{}]", name);
    crate::teeln!("selection = {:?}", sel);

    crate::teeln!("vfeat1: {} -> {}", split.tr, split.pr);
    let tr = Dataset::load(split.tr, "rtg", split.preds_dir);
    let pr = Dataset::load(split.pr, "rtg", split.preds_dir);
    let stats = VotingFeatures1::new(&tr, &sel);
    let features = stats.compute_all(&pr);
    save_features(&features, &sel, split.preds_dir, split.pr);
    drop(features);
    drop(stats);
    drop(pr);
    drop(tr);

    crate::teeln!("vfeat1: {} -> {}", split.fulltrain_tr, split.fulltrain_pr);
    let fulltrain = Dataset::load(split.fulltrain_tr, "rtg", split.preds_dir);
    let qual = Dataset::load(split.fulltrain_pr, "rtg", split.preds_dir);
    let stats = VotingFeatures1::new(&fulltrain, &sel);
    let features = stats.compute_all(&qual);
    save_features(&features, &sel, split.preds_dir, split.fulltrain_pr);

    if owns_log {
        if let Some(mut lf) = crate::LOG_FILE.lock().unwrap().take() {
            let _ = lf.flush();
        }
    }
}
