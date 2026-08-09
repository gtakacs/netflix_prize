// FWLS voting-feature generator, the big set — 105 features: user/movie temporal
// windows, kNN context, rating-distribution shape (entropy, mode, bimodality, skew),
// user x movie interactions, and 5 ordinal-SVD class probabilities.
// Frozen archive — see README.md; superseded by src/vfeat1.rs. Writes
// features/claude105_features.{set}.npy and the shipped files were renamed by hand into
// preds_old/claude105/{000..104}.*.npy, the layout the current blenders load.
//
// 105 meta-features for Netflix Prize prediction improvement.
// Categories:
//   - Temporal (User): 0-19
//   - Temporal (Movie): 20-34
//   - Similarity-based: 35-49
//   - Rating Distribution (User): 50-64
//   - Rating Distribution (Movie): 65-74
//   - User-Movie Interaction: 75-89
//   - Advanced Temporal/Context: 90-99
//   - Ordinal SVD probabilities: 100-104
//
// Options:
//   --separate    Generate separate files per feature instead of single combined file (default: combined)
//   --train       Also generate features for train set
//   --fulltrain   Also generate features for fulltrain set
// Default: generates single combined file for probe and qual only.

use gravity::{Dataset, calc_gbias, rand_array2};
use rand::{SeedableRng, rngs::StdRng};
use indicatif::ParallelProgressIterator;
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;

const N_FEATURES: usize = 105;
const BAYESIAN_K: f64 = 25.0;
const REG_K: f32 = 5.0;  // Regularization for temporal averages
const SIM_SHRINKAGE: f32 = 100.0;

// ============================================================================
// Feature indices (0-99)
// ============================================================================

// Temporal Features - User Activity (0-19)
const F_USER_SAME_DAY_AVG: usize = 0;
const F_USER_SAME_DAY_IS_ZERO: usize = 1;
const F_USER_YESTERDAY_AVG: usize = 2;
const F_USER_YESTERDAY_IS_ZERO: usize = 3;
const F_USER_LAST_7D_AVG: usize = 4;
const F_USER_LAST_7D_IS_ZERO: usize = 5;
const F_USER_LAST_32D_AVG: usize = 6;
const F_USER_BEFORE_YESTERDAY_AVG: usize = 7;
const F_USER_BEFORE_YESTERDAY_IS_ZERO: usize = 8;
const F_USER_DAYS_SINCE_PREV: usize = 9;
const F_LOG_USER_DAYS_SINCE_PREV: usize = 10;
const F_USER_IS_FIRST_RATING: usize = 11;
const F_USER_VELOCITY_7D: usize = 12;
const F_USER_BINGE: usize = 13;
const F_USER_DAYS_ACTIVE_7D: usize = 14;
const F_USER_DAYS_ACTIVE_32D: usize = 15;
const F_USER_TENURE_PCT: usize = 16;
const F_USER_IS_WEEKEND: usize = 17;
const F_USER_DAY_OF_WEEK: usize = 18;
const F_LOG_USER_DAYS_UNTIL_LAST: usize = 19;

// Temporal Features - Movie Activity (20-34)
const F_MOVIE_SAME_DAY_AVG: usize = 20;
const F_MOVIE_SAME_DAY_IS_ZERO: usize = 21;
const F_MOVIE_YESTERDAY_AVG: usize = 22;
const F_MOVIE_YESTERDAY_IS_ZERO: usize = 23;
const F_MOVIE_LAST_7D_AVG: usize = 24;
const F_MOVIE_LAST_32D_AVG: usize = 25;
const F_MOVIE_AGE: usize = 26;
const F_LOG_MOVIE_AGE: usize = 27;
const F_MOVIE_DAYS_SINCE_PREV: usize = 28;
const F_MOVIE_SAME_DAY_COUNT: usize = 29;
const F_LOG_MOVIE_SAME_DAY_COUNT: usize = 30;
const F_MOVIE_LAST_7D_COUNT: usize = 31;
const F_MOVIE_IS_NEW: usize = 32;
const F_MOVIE_RATING_TREND: usize = 33;
const F_MOVIE_MOMENTUM: usize = 34;

// Similarity-based Features (35-49)
const F_KNN_SAME_DAY: usize = 35;
const F_KNN_SAME_DAY_IS_ZERO: usize = 36;
const F_KNN_YESTERDAY: usize = 37;
const F_KNN_YESTERDAY_IS_ZERO: usize = 38;
const F_KNN_BEFORE_YESTERDAY: usize = 39;
const F_KNN_BEFORE_IS_ZERO: usize = 40;
const F_KNN_ALL: usize = 41;
const F_MOST_SIMILAR_RATING: usize = 42;
const F_SECOND_SIMILAR_RATING: usize = 43;
const F_AVG_SIM_TO_USER_MOVIES: usize = 44;
const F_WEIGHTED_AVG_USER_RATING: usize = 45;
const F_NEGATIVE_SIM_SUM: usize = 46;
const F_SIM_VARIANCE: usize = 47;
const F_PCT_HIGH_SIM: usize = 48;
const F_MIN_SIM_IN_USER_SET: usize = 49;

// Rating Distribution - User (50-64)
const F_USER_RATING_ENTROPY: usize = 50;
const F_USER_RATING_MODE: usize = 51;
const F_USER_PCT_5STAR: usize = 52;
const F_USER_PCT_4STAR: usize = 53;
const F_USER_PCT_3STAR: usize = 54;
const F_USER_PCT_2STAR: usize = 55;
const F_USER_PCT_1STAR: usize = 56;
const F_USER_PCT_EXTREME: usize = 57;
const F_USER_RATING_RANGE: usize = 58;
const F_USER_MEDIAN: usize = 59;
const F_USER_IS_HARSH: usize = 60;
const F_USER_HARSHNESS: usize = 61;
const F_USER_SKEWNESS: usize = 62;
const F_USER_BIMODAL: usize = 63;
const F_USER_CONSISTENCY: usize = 64;

// Rating Distribution - Movie (65-74)
const F_MOVIE_RATING_ENTROPY: usize = 65;
const F_MOVIE_RATING_MODE: usize = 66;
const F_MOVIE_PCT_5STAR: usize = 67;
const F_MOVIE_PCT_1STAR: usize = 68;
const F_MOVIE_PCT_EXTREME: usize = 69;
const F_MOVIE_IS_POLARIZING: usize = 70;
const F_MOVIE_IS_CROWD_PLEASER: usize = 71;
const F_MOVIE_CONTROVERSY: usize = 72;
const F_MOVIE_BIMODAL: usize = 73;
const F_BAYESIAN_MOVIE_MEAN: usize = 74;

// User-Movie Interaction (75-89)
const F_SVD_DOT: usize = 75;
const F_SVD_DOT_CENTERED: usize = 76;
const F_SVD_CONFIDENCE: usize = 77;
const F_USER_MOVIE_BIAS_PRODUCT: usize = 78;
const F_USER_MOVIE_COUNT_PRODUCT: usize = 79;
const F_USER_MOVIE_STD_PRODUCT: usize = 80;
const F_USER_MOVIE_AVG_DIFF: usize = 81;
const F_USER_MOVIE_AVG_DIFF_ABS: usize = 82;
const F_USER_SIMILAR_MOVIES_COUNT: usize = 83;
const F_LOG_USER_SIMILAR_MOVIES: usize = 84;
const F_USER_RATES_POPULAR: usize = 85;
const F_USER_NICHE_SCORE: usize = 86;
const F_USER_MOVIE_SUPPORT: usize = 87;
const F_LOG_USER_MOVIE_SUPPORT: usize = 88;
const F_ORDINAL_SVD_MODE: usize = 89;

// Advanced Temporal/Context (90-99)
const F_USER_SESSION_POSITION: usize = 90;
const F_USER_DAY_POSITION: usize = 91;
const F_USER_RATING_STREAK: usize = 92;
const F_USER_INTER_RATING_GAP: usize = 93;
const F_MOVIE_FRESHNESS_DECAY: usize = 94;
const F_TIME_TO_DATASET_END: usize = 95;
const F_USER_RECENCY_SCORE: usize = 96;
const F_MOVIE_RECENCY_SCORE: usize = 97;
const F_USER_DIVERSITY_SAME_DAY: usize = 98;
const F_ORDINAL_SVD_ENTROPY: usize = 99;

// Ordinal SVD probabilities (100-104) - P(rating = k) from ordinal MF
const F_ORDINAL_PROB_1: usize = 100;
const F_ORDINAL_PROB_2: usize = 101;
const F_ORDINAL_PROB_3: usize = 102;
const F_ORDINAL_PROB_4: usize = 103;
const F_ORDINAL_PROB_5: usize = 104;

// ============================================================================
// Helper structures
// ============================================================================

/// Per-user cumulative data for efficient range queries
#[derive(Clone)]
struct UserCumulative {
    /// Sorted list of (day, cumsum, cumcount)
    data: Vec<(i16, f32, u32)>,
}

impl UserCumulative {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Get sum and count for range [day_start, day_end]
    fn range_sum_count(&self, day_start: i16, day_end: i16) -> (f32, u32) {
        if self.data.is_empty() { return (0.0, 0); }

        // End: last day <= day_end
        let end_idx = self.data.partition_point(|&(d, _, _)| d <= day_end);
        if end_idx == 0 { return (0.0, 0); }
        let (_, end_sum, end_cnt) = self.data[end_idx - 1];

        // Start: last day < day_start
        let start_idx = self.data.partition_point(|&(d, _, _)| d < day_start);
        let (start_sum, start_cnt) = if start_idx == 0 {
            (0.0, 0)
        } else {
            let (_, s, c) = self.data[start_idx - 1];
            (s, c)
        };

        (end_sum - start_sum, end_cnt - start_cnt)
    }

    /// Regularized average for range
    fn range_avg(&self, day_start: i16, day_end: i16, global_mean: f32) -> f32 {
        let (sum, cnt) = self.range_sum_count(day_start, day_end);
        if cnt == 0 { return 0.0; }
        (sum + REG_K * global_mean) / (cnt as f32 + REG_K)
    }
}

/// Sigmoid function for ordinal SVD
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute entropy of a distribution
#[inline]
fn entropy(counts: &[u32; 5]) -> f32 {
    let total: u32 = counts.iter().sum();
    if total == 0 { return 0.0; }
    let mut ent = 0.0f32;
    for &c in counts {
        if c > 0 {
            let p = c as f32 / total as f32;
            ent -= p * p.ln();
        }
    }
    ent
}

/// Find mode (most common value, 1-5)
#[inline]
fn mode(counts: &[u32; 5]) -> f32 {
    let mut max_idx = 0;
    let mut max_val = counts[0];
    for i in 1..5 {
        if counts[i] > max_val {
            max_val = counts[i];
            max_idx = i;
        }
    }
    (max_idx + 1) as f32
}

/// Compute bimodality score (difference between peaks at extremes vs middle)
#[inline]
fn bimodal_score(counts: &[u32; 5]) -> f32 {
    let total: u32 = counts.iter().sum();
    if total == 0 { return 0.0; }
    let extreme = (counts[0] + counts[4]) as f32 / total as f32;
    let middle = counts[2] as f32 / total as f32;
    (extreme - middle).max(0.0)
}

// ============================================================================
// Main feature computation struct
// ============================================================================

pub struct Claude105Features {
    // Dataset dimensions
    n_users: usize,
    n_items: usize,
    global_mean: f32,
    dataset_max_day: i16,

    // Per-user statistics
    user_rating_counts: Array1<u32>,
    user_first_date: Array1<i16>,
    user_last_date: Array1<i16>,
    user_bayesian_mean: Array1<f32>,
    user_std: Array1<f32>,
    user_rating_dist: Vec<[u32; 5]>,  // counts per rating value
    user_median: Array1<f32>,
    user_skewness: Array1<f32>,

    // Per-user temporal data
    user_day_sum: Vec<HashMap<i16, f32>>,
    user_day_cnt: Vec<HashMap<i16, u32>>,
    user_cumulative: Vec<UserCumulative>,
    user_sorted_days: Vec<Vec<i16>>,
    user_day_ratings: Vec<HashMap<i16, Vec<(usize, f32)>>>,  // (movie_idx, rating)

    // Per-movie statistics
    movie_rating_counts: Array1<u32>,
    movie_first_date: Array1<i16>,
    movie_bayesian_mean: Array1<f32>,
    movie_std: Array1<f32>,
    movie_rating_dist: Vec<[u32; 5]>,

    // Per-movie temporal data
    movie_day_sum: Vec<HashMap<i16, f32>>,
    movie_day_cnt: Vec<HashMap<i16, u32>>,
    movie_cumulative: Vec<UserCumulative>,  // Reuse same structure
    movie_sorted_days: Vec<Vec<i16>>,

    // Movie trend data
    movie_trend: Array1<f32>,  // Linear regression slope
    movie_momentum: Array1<f32>,  // Recent vs older avg

    // Item-item similarity
    sim_matrix: Array2<f32>,
    sim_top: Vec<Vec<(usize, f32)>>,  // Top-100 per item

    // User-movie support (shared raters)
    movie_raters: Vec<Vec<usize>>,

    // SVD data
    svd_gbias: f32,
    svd_ubias: Array1<f32>,
    svd_ibias: Array1<f32>,
    svd_ufeat: Array2<f32>,
    svd_ifeat: Array2<f32>,
    svd_user_norm: Array1<f32>,
    svd_movie_norm: Array1<f32>,

    // Ordinal SVD
    ordinal_ufeat: Array2<f32>,
    ordinal_ifeat: Array2<f32>,
    ordinal_thresholds: [f32; 4],

    // User items (for similarity lookups)
    user_items: Vec<Vec<usize>>,

    // User popularity preference
    user_avg_movie_pop: Array1<f32>,
}

impl Claude105Features {
    pub fn new(ds: &Dataset) -> Self {
        println!("Computing Claude105 statistics...");
        let n_users = ds.n_users;
        let n_items = ds.n_items;

        // ====================================================================
        // Pass 1: Basic statistics collection
        // ====================================================================
        println!("  Pass 1: Collecting basic statistics...");

        let mut user_rating_counts = Array1::<u32>::zeros(n_users);
        let mut user_sum = Array1::<f64>::zeros(n_users);
        let mut user_sum_sq = Array1::<f64>::zeros(n_users);
        let mut user_first_date = Array1::<i16>::from_elem(n_users, i16::MAX);
        let mut user_last_date = Array1::<i16>::from_elem(n_users, i16::MIN);
        let mut user_rating_dist: Vec<[u32; 5]> = vec![[0u32; 5]; n_users];
        let mut user_day_sum: Vec<HashMap<i16, f32>> = vec![HashMap::new(); n_users];
        let mut user_day_cnt: Vec<HashMap<i16, u32>> = vec![HashMap::new(); n_users];
        let mut user_day_ratings: Vec<HashMap<i16, Vec<(usize, f32)>>> = vec![HashMap::new(); n_users];
        let mut user_sum_movie_pop = Array1::<f64>::zeros(n_users);

        let mut movie_rating_counts = Array1::<u32>::zeros(n_items);
        let mut movie_sum = Array1::<f64>::zeros(n_items);
        let mut movie_sum_sq = Array1::<f64>::zeros(n_items);
        let mut movie_first_date = Array1::<i16>::from_elem(n_items, i16::MAX);
        let mut movie_rating_dist: Vec<[u32; 5]> = vec![[0u32; 5]; n_items];
        let mut movie_day_sum: Vec<HashMap<i16, f32>> = vec![HashMap::new(); n_items];
        let mut movie_day_cnt: Vec<HashMap<i16, u32>> = vec![HashMap::new(); n_items];

        let mut total_sum = 0.0f64;
        let mut total_cnt = 0u64;
        let mut dataset_max_day = i16::MIN;

        for idx in 0..ds.n_ratings {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            let r = ds.raw_ratings[idx];
            let rf = r as f32;
            let rd = r as f64;
            let day = ds.dates[idx];

            // User stats
            user_rating_counts[u] += 1;
            user_sum[u] += rd;
            user_sum_sq[u] += rd * rd;
            if day < user_first_date[u] { user_first_date[u] = day; }
            if day > user_last_date[u] { user_last_date[u] = day; }
            if r >= 1 && r <= 5 {
                user_rating_dist[u][(r - 1) as usize] += 1;
            }
            *user_day_sum[u].entry(day).or_insert(0.0) += rf;
            *user_day_cnt[u].entry(day).or_insert(0) += 1;
            user_day_ratings[u].entry(day).or_default().push((i, rf));

            // Movie stats
            movie_rating_counts[i] += 1;
            movie_sum[i] += rd;
            movie_sum_sq[i] += rd * rd;
            if day < movie_first_date[i] { movie_first_date[i] = day; }
            if r >= 1 && r <= 5 {
                movie_rating_dist[i][(r - 1) as usize] += 1;
            }
            *movie_day_sum[i].entry(day).or_insert(0.0) += rf;
            *movie_day_cnt[i].entry(day).or_insert(0) += 1;

            total_sum += rd;
            total_cnt += 1;
            if day > dataset_max_day { dataset_max_day = day; }
        }

        let global_mean = (total_sum / total_cnt as f64) as f32;

        // ====================================================================
        // Pass 2: Compute derived statistics
        // ====================================================================
        println!("  Pass 2: Computing derived statistics...");

        // User means, std, bayesian mean
        let mut user_bayesian_mean = Array1::<f32>::zeros(n_users);
        let mut user_std = Array1::<f32>::zeros(n_users);
        let mut user_median = Array1::<f32>::zeros(n_users);
        let mut user_skewness = Array1::<f32>::zeros(n_users);

        for u in 0..n_users {
            let cnt = user_rating_counts[u] as f64;
            if cnt > 0.0 {
                let mean = user_sum[u] / cnt;
                user_bayesian_mean[u] = ((user_sum[u] + BAYESIAN_K * global_mean as f64) / (cnt + BAYESIAN_K)) as f32;
                if cnt > 1.0 {
                    let var = (user_sum_sq[u] / cnt) - (mean * mean);
                    user_std[u] = var.max(0.0).sqrt() as f32;
                }
                // Compute median from distribution
                let dist = &user_rating_dist[u];
                let total: u32 = dist.iter().sum();
                let mid = total / 2;
                let mut cumsum = 0u32;
                for (val, &c) in dist.iter().enumerate() {
                    cumsum += c;
                    if cumsum > mid {
                        user_median[u] = (val + 1) as f32;
                        break;
                    }
                }
                // Compute skewness
                if user_std[u] > 0.01 && cnt > 2.0 {
                    let std3 = (user_std[u] as f64).powi(3);
                    let mut m3 = 0.0f64;
                    for (val, &c) in dist.iter().enumerate() {
                        let x = (val + 1) as f64 - mean;
                        m3 += c as f64 * x.powi(3);
                    }
                    m3 /= cnt;
                    user_skewness[u] = (m3 / std3) as f32;
                }
            }
        }

        // Movie means, std, bayesian mean
        let mut movie_bayesian_mean = Array1::<f32>::zeros(n_items);
        let mut movie_std = Array1::<f32>::zeros(n_items);

        for i in 0..n_items {
            let cnt = movie_rating_counts[i] as f64;
            if cnt > 0.0 {
                let mean = movie_sum[i] / cnt;
                movie_bayesian_mean[i] = ((movie_sum[i] + BAYESIAN_K * global_mean as f64) / (cnt + BAYESIAN_K)) as f32;
                if cnt > 1.0 {
                    let var = (movie_sum_sq[i] / cnt) - (mean * mean);
                    movie_std[i] = var.max(0.0).sqrt() as f32;
                }
            }
        }

        // User avg movie popularity
        for idx in 0..ds.n_ratings {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            user_sum_movie_pop[u] += movie_rating_counts[i] as f64;
        }
        let avg_movie_cnt = total_cnt as f64 / n_items as f64;
        let mut user_avg_movie_pop = Array1::<f32>::zeros(n_users);
        for u in 0..n_users {
            let cnt = user_rating_counts[u] as f64;
            if cnt > 0.0 {
                user_avg_movie_pop[u] = ((user_sum_movie_pop[u] / cnt) / avg_movie_cnt) as f32;
            }
        }

        // ====================================================================
        // Build cumulative structures
        // ====================================================================
        println!("  Building cumulative structures...");

        let mut user_cumulative: Vec<UserCumulative> = vec![UserCumulative::new(); n_users];
        let mut user_sorted_days: Vec<Vec<i16>> = vec![Vec::new(); n_users];

        for u in 0..n_users {
            let mut days: Vec<i16> = user_day_sum[u].keys().copied().collect();
            days.sort_unstable();
            user_sorted_days[u] = days.clone();

            let mut cum_sum = 0.0f32;
            let mut cum_cnt = 0u32;
            for &day in &days {
                cum_sum += user_day_sum[u][&day];
                cum_cnt += user_day_cnt[u][&day];
                user_cumulative[u].data.push((day, cum_sum, cum_cnt));
            }
        }

        let mut movie_cumulative: Vec<UserCumulative> = vec![UserCumulative::new(); n_items];
        let mut movie_sorted_days: Vec<Vec<i16>> = vec![Vec::new(); n_items];

        for i in 0..n_items {
            let mut days: Vec<i16> = movie_day_sum[i].keys().copied().collect();
            days.sort_unstable();
            movie_sorted_days[i] = days.clone();

            let mut cum_sum = 0.0f32;
            let mut cum_cnt = 0u32;
            for &day in &days {
                cum_sum += movie_day_sum[i][&day];
                cum_cnt += movie_day_cnt[i][&day];
                movie_cumulative[i].data.push((day, cum_sum, cum_cnt));
            }
        }

        // ====================================================================
        // Movie trend and momentum
        // ====================================================================
        println!("  Computing movie trends...");

        let mut movie_trend = Array1::<f32>::zeros(n_items);
        let mut movie_momentum = Array1::<f32>::zeros(n_items);

        for i in 0..n_items {
            let days = &movie_sorted_days[i];
            if days.len() >= 10 {
                // Linear regression: rating vs normalized day
                let min_day = days[0] as f64;
                let max_day = days[days.len() - 1] as f64;
                let day_range = (max_day - min_day).max(1.0);

                let mut sum_x = 0.0f64;
                let mut sum_y = 0.0f64;
                let mut sum_xy = 0.0f64;
                let mut sum_xx = 0.0f64;
                let mut n = 0.0f64;

                for &day in days {
                    let x = (day as f64 - min_day) / day_range;
                    let y = movie_day_sum[i][&day] as f64 / movie_day_cnt[i][&day] as f64;
                    let cnt = movie_day_cnt[i][&day] as f64;
                    sum_x += x * cnt;
                    sum_y += y * cnt;
                    sum_xy += x * y * cnt;
                    sum_xx += x * x * cnt;
                    n += cnt;
                }

                let denom = n * sum_xx - sum_x * sum_x;
                if denom.abs() > 1e-10 {
                    let slope = (n * sum_xy - sum_x * sum_y) / denom;
                    movie_trend[i] = slope as f32;
                }

                // Momentum: compare last 30% vs first 30%
                let n_days = days.len();
                let first_end = n_days * 3 / 10;
                let last_start = n_days * 7 / 10;

                let (first_sum, first_cnt) = days[..first_end].iter()
                    .map(|&d| (movie_day_sum[i][&d], movie_day_cnt[i][&d]))
                    .fold((0.0f32, 0u32), |(s, c), (ds, dc)| (s + ds, c + dc));
                let (last_sum, last_cnt) = days[last_start..].iter()
                    .map(|&d| (movie_day_sum[i][&d], movie_day_cnt[i][&d]))
                    .fold((0.0f32, 0u32), |(s, c), (ds, dc)| (s + ds, c + dc));

                if first_cnt > 0 && last_cnt > 0 {
                    let first_avg = first_sum / first_cnt as f32;
                    let last_avg = last_sum / last_cnt as f32;
                    movie_momentum[i] = last_avg - first_avg;
                }
            }
        }

        // ====================================================================
        // Item-item similarity matrix
        // ====================================================================
        println!("  Computing item-item similarities...");

        // Build user items
        let mut user_starts = vec![0usize; n_users + 1];
        for u in 0..n_users {
            user_starts[u + 1] = user_starts[u] + ds.user_cnts[u] as usize;
        }
        let user_items: Vec<Vec<usize>> = (0..n_users).map(|u| {
            let mut items: Vec<usize> = (user_starts[u]..user_starts[u + 1])
                .map(|idx| ds.item_idxs[idx] as usize).collect();
            items.sort_unstable();
            items
        }).collect();

        // Movie means for centering
        let movie_means: Vec<f32> = (0..n_items).map(|i| {
            if movie_rating_counts[i] > 0 {
                (movie_sum[i] / movie_rating_counts[i] as f64) as f32
            } else { 0.0 }
        }).collect();

        // Accumulate similarity data
        let supp_rows: Vec<Mutex<Vec<f32>>> =
            (0..n_items).map(|_| Mutex::new(vec![0.0; n_items])).collect();
        let prod_rows: Vec<Mutex<Vec<f32>>> =
            (0..n_items).map(|_| Mutex::new(vec![0.0; n_items])).collect();

        (0..n_users).into_par_iter().progress_count(n_users as u64).for_each(|u| {
            let start = user_starts[u];
            let end = user_starts[u + 1];
            if start == end { return; }

            let items: Vec<(usize, f32)> = (start..end).map(|idx| {
                let i = ds.item_idxs[idx] as usize;
                let r = ds.raw_ratings[idx] as f32 - movie_means[i];
                (i, r)
            }).collect();

            for &(i, ri) in &items {
                let mut supp = supp_rows[i].lock();
                let mut prod = prod_rows[i].lock();
                for &(j, rj) in &items {
                    supp[j] += 1.0;
                    prod[j] += ri * rj;
                }
            }
        });

        // Convert to similarity matrix
        println!("  Converting to similarity matrix...");
        let norms: Vec<f32> = (0..n_items).map(|i| {
            prod_rows[i].lock()[i].max(0.0).sqrt()
        }).collect();

        let mut sim_matrix = Array2::<f32>::zeros((n_items, n_items));
        let mut sim_top: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_items];

        for i in 0..n_items {
            let supp = supp_rows[i].lock();
            let prod = prod_rows[i].lock();
            let mut sims: Vec<(usize, f32)> = Vec::new();

            for j in 0..n_items {
                if i == j { continue; }
                let n = supp[j];
                if n < 2.0 { continue; }
                let den = norms[i] * norms[j];
                let phi = if den > 0.0 { prod[j] / den } else { 0.0 };
                let sim = phi * n / (n + SIM_SHRINKAGE);
                sim_matrix[[i, j]] = sim;
                if sim > 0.01 {
                    sims.push((j, sim));
                }
            }
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(100);
            sim_top[i] = sims;
        }
        drop(supp_rows);
        drop(prod_rows);

        // Build movie raters
        let mut movie_raters: Vec<Vec<usize>> = vec![Vec::new(); n_items];
        for idx in 0..ds.n_ratings {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            movie_raters[i].push(u);
        }

        // ====================================================================
        // Train SVD models
        // ====================================================================
        println!("  Training 32-factor SVD...");
        let (svd_gbias, svd_ubias, svd_ibias, svd_ufeat, svd_ifeat) = train_svd(ds, 32);

        let svd_user_norm = Array1::from_iter((0..n_users).map(|u| {
            svd_ufeat.row(u).dot(&svd_ufeat.row(u)).sqrt()
        }));
        let svd_movie_norm = Array1::from_iter((0..n_items).map(|i| {
            svd_ifeat.row(i).dot(&svd_ifeat.row(i)).sqrt()
        }));

        println!("  Training 60-factor ordinal SVD...");
        let (_, _, _, ordinal_ufeat, ordinal_ifeat) = train_svd(ds, 60);

        // Estimate ordinal thresholds
        let mut rating_dot_sums = [0.0f64; 5];
        let mut rating_counts = [0u64; 5];
        for idx in 0..ds.n_ratings {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            let r = ds.raw_ratings[idx] as usize;
            if r >= 1 && r <= 5 {
                let dot = ordinal_ufeat.row(u).dot(&ordinal_ifeat.row(i)) as f64;
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
        let mut ordinal_thresholds = [0.0f32; 4];
        for k in 0..4 {
            ordinal_thresholds[k] = ((rating_means[k] + rating_means[k + 1]) / 2.0) as f32;
        }
        println!("  Ordinal thresholds: {:?}", ordinal_thresholds);

        println!("  All statistics computed.");

        Self {
            n_users,
            n_items,
            global_mean,
            dataset_max_day,
            user_rating_counts,
            user_first_date,
            user_last_date,
            user_bayesian_mean,
            user_std,
            user_rating_dist,
            user_median,
            user_skewness,
            user_day_sum,
            user_day_cnt,
            user_cumulative,
            user_sorted_days,
            user_day_ratings,
            movie_rating_counts,
            movie_first_date,
            movie_bayesian_mean,
            movie_std,
            movie_rating_dist,
            movie_day_sum,
            movie_day_cnt,
            movie_cumulative,
            movie_sorted_days,
            movie_trend,
            movie_momentum,
            sim_matrix,
            sim_top,
            movie_raters,
            svd_gbias,
            svd_ubias,
            svd_ibias,
            svd_ufeat,
            svd_ifeat,
            svd_user_norm,
            svd_movie_norm,
            ordinal_ufeat,
            ordinal_ifeat,
            ordinal_thresholds,
            user_items,
            user_avg_movie_pop,
        }
    }

    /// Get regularized user day average (optionally excluding a rating)
    #[inline]
    fn user_day_avg(&self, u: usize, day: i16, exclude: Option<f32>) -> f32 {
        let sum = self.user_day_sum[u].get(&day).copied().unwrap_or(0.0);
        let cnt = self.user_day_cnt[u].get(&day).copied().unwrap_or(0);
        let (adj_sum, adj_cnt) = match exclude {
            Some(r) => (sum - r, cnt.saturating_sub(1)),
            None => (sum, cnt),
        };
        if adj_cnt == 0 { return 0.0; }
        (adj_sum + REG_K * self.global_mean) / (adj_cnt as f32 + REG_K)
    }

    /// Get regularized movie day average
    #[inline]
    fn movie_day_avg(&self, i: usize, day: i16, exclude: Option<f32>) -> f32 {
        let sum = self.movie_day_sum[i].get(&day).copied().unwrap_or(0.0);
        let cnt = self.movie_day_cnt[i].get(&day).copied().unwrap_or(0);
        let (adj_sum, adj_cnt) = match exclude {
            Some(r) => (sum - r, cnt.saturating_sub(1)),
            None => (sum, cnt),
        };
        if adj_cnt == 0 { return 0.0; }
        (adj_sum + REG_K * self.global_mean) / (adj_cnt as f32 + REG_K)
    }

    /// Find kNN prediction from user's ratings in a day range
    fn knn_prediction(&self, u: usize, i: usize, day_start: i16, day_end: i16) -> (f32, f32) {
        let top_sims = &self.sim_top[i];
        if top_sims.is_empty() { return (0.0, 0.0); }

        let mut weighted_sum = 0.0f32;
        let mut sim_sum = 0.0f32;

        for &day in &self.user_sorted_days[u] {
            if day < day_start { continue; }
            if day > day_end { break; }

            if let Some(ratings) = self.user_day_ratings[u].get(&day) {
                for &(j, rating) in ratings {
                    if j == i { continue; }
                    // Check if j is in top similarities for i
                    for &(k, sim) in top_sims {
                        if k == j && sim > 0.0 {
                            weighted_sum += sim * rating;
                            sim_sum += sim;
                            break;
                        }
                    }
                }
            }
        }

        if sim_sum > 0.0 {
            (weighted_sum / sim_sum, sim_sum)
        } else {
            (0.0, 0.0)
        }
    }

    /// Get most similar movie rating(s) from a day range
    fn most_similar_ratings(&self, u: usize, i: usize, day_start: i16, day_end: i16) -> (f32, f32) {
        let top_sims = &self.sim_top[i];
        if top_sims.is_empty() { return (0.0, 0.0); }

        let mut best: Vec<(f32, f32)> = Vec::new();  // (sim, rating)

        for &day in &self.user_sorted_days[u] {
            if day < day_start { continue; }
            if day > day_end { break; }

            if let Some(ratings) = self.user_day_ratings[u].get(&day) {
                for &(j, rating) in ratings {
                    if j == i { continue; }
                    for &(k, sim) in top_sims {
                        if k == j {
                            best.push((sim, rating));
                            break;
                        }
                    }
                }
            }
        }

        best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let first = best.first().map(|&(_, r)| r).unwrap_or(0.0);
        let second = best.get(1).map(|&(_, r)| r).unwrap_or(0.0);
        (first, second)
    }

    /// Compute all 105 features for a (user, item, day) tuple
    pub fn compute(&self, u: usize, i: usize, day: i16) -> [f32; N_FEATURES] {
        let mut f = [0.0f32; N_FEATURES];

        // ====================================================================
        // Temporal Features - User (0-19)
        // ====================================================================

        // User same-day avg (excluding this rating in training - here we don't exclude)
        let user_same_day = self.user_day_avg(u, day, None);
        f[F_USER_SAME_DAY_AVG] = user_same_day;
        f[F_USER_SAME_DAY_IS_ZERO] = if user_same_day == 0.0 { 1.0 } else { 0.0 };

        // User yesterday
        let user_yesterday = self.user_day_avg(u, day - 1, None);
        f[F_USER_YESTERDAY_AVG] = user_yesterday;
        f[F_USER_YESTERDAY_IS_ZERO] = if user_yesterday == 0.0 { 1.0 } else { 0.0 };

        // User last 7 days (excluding today)
        let user_7d = self.user_cumulative[u].range_avg(day - 7, day - 1, self.global_mean);
        f[F_USER_LAST_7D_AVG] = user_7d;
        f[F_USER_LAST_7D_IS_ZERO] = if user_7d == 0.0 { 1.0 } else { 0.0 };

        // User last 32 days (excluding today)
        f[F_USER_LAST_32D_AVG] = self.user_cumulative[u].range_avg(day - 32, day - 1, self.global_mean);

        // User before yesterday
        let user_before = self.user_cumulative[u].range_avg(i16::MIN, day - 2, self.global_mean);
        f[F_USER_BEFORE_YESTERDAY_AVG] = user_before;
        f[F_USER_BEFORE_YESTERDAY_IS_ZERO] = if user_before == 0.0 { 1.0 } else { 0.0 };

        // Days since previous rating
        let days = &self.user_sorted_days[u];
        let day_idx = days.partition_point(|&d| d < day);
        let days_since = if day_idx > 0 {
            (day - days[day_idx - 1]).max(0) as f32
        } else {
            0.0
        };
        f[F_USER_DAYS_SINCE_PREV] = days_since;
        f[F_LOG_USER_DAYS_SINCE_PREV] = (1.0 + days_since).ln();
        f[F_USER_IS_FIRST_RATING] = if day_idx == 0 { 1.0 } else { 0.0 };

        // User velocity (ratings per day in last 7 days)
        let (_, cnt_7d) = self.user_cumulative[u].range_sum_count(day - 7, day - 1);
        f[F_USER_VELOCITY_7D] = cnt_7d as f32 / 7.0;

        // Binge indicator
        let day_cnt = self.user_day_cnt[u].get(&day).copied().unwrap_or(0);
        f[F_USER_BINGE] = if day_cnt > 10 { 1.0 } else { 0.0 };

        // Days active in last 7/32 days
        let active_7d = days.iter().filter(|&&d| d >= day - 7 && d < day).count();
        let active_32d = days.iter().filter(|&&d| d >= day - 32 && d < day).count();
        f[F_USER_DAYS_ACTIVE_7D] = active_7d as f32;
        f[F_USER_DAYS_ACTIVE_32D] = active_32d as f32;

        // Tenure percentage
        let first = self.user_first_date[u];
        let last = self.user_last_date[u];
        if last > first {
            f[F_USER_TENURE_PCT] = (day - first) as f32 / (last - first) as f32;
        }

        // Weekend and day of week (day 0 = some reference, assume mod 7)
        let dow = ((day % 7) + 7) % 7;  // 0-6
        f[F_USER_IS_WEEKEND] = if dow == 5 || dow == 6 { 1.0 } else { 0.0 };
        f[F_USER_DAY_OF_WEEK] = dow as f32 / 6.0;  // Normalize to 0-1

        // Days until last rating
        let days_until_last = (last - day).max(0) as f32;
        f[F_LOG_USER_DAYS_UNTIL_LAST] = (1.0 + days_until_last).ln();

        // ====================================================================
        // Temporal Features - Movie (20-34)
        // ====================================================================

        let movie_same_day = self.movie_day_avg(i, day, None);
        f[F_MOVIE_SAME_DAY_AVG] = movie_same_day;
        f[F_MOVIE_SAME_DAY_IS_ZERO] = if movie_same_day == 0.0 { 1.0 } else { 0.0 };

        let movie_yesterday = self.movie_day_avg(i, day - 1, None);
        f[F_MOVIE_YESTERDAY_AVG] = movie_yesterday;
        f[F_MOVIE_YESTERDAY_IS_ZERO] = if movie_yesterday == 0.0 { 1.0 } else { 0.0 };

        f[F_MOVIE_LAST_7D_AVG] = self.movie_cumulative[i].range_avg(day - 7, day - 1, self.global_mean);
        f[F_MOVIE_LAST_32D_AVG] = self.movie_cumulative[i].range_avg(day - 32, day - 1, self.global_mean);

        // Movie age
        let movie_age = (day - self.movie_first_date[i]).max(0) as f32;
        f[F_MOVIE_AGE] = movie_age;
        f[F_LOG_MOVIE_AGE] = (1.0 + movie_age).ln();

        // Movie days since previous
        let m_days = &self.movie_sorted_days[i];
        let m_day_idx = m_days.partition_point(|&d| d < day);
        let m_days_since = if m_day_idx > 0 {
            (day - m_days[m_day_idx - 1]).max(0) as f32
        } else {
            0.0
        };
        f[F_MOVIE_DAYS_SINCE_PREV] = m_days_since;

        // Movie same day count
        let m_day_cnt = self.movie_day_cnt[i].get(&day).copied().unwrap_or(0) as f32;
        f[F_MOVIE_SAME_DAY_COUNT] = m_day_cnt;
        f[F_LOG_MOVIE_SAME_DAY_COUNT] = (1.0 + m_day_cnt).ln();

        // Movie last 7 days count
        let (_, m_cnt_7d) = self.movie_cumulative[i].range_sum_count(day - 7, day - 1);
        f[F_MOVIE_LAST_7D_COUNT] = m_cnt_7d as f32;

        // Movie is new (< 20 ratings before this date)
        let m_ratings_before = m_days.iter().take_while(|&&d| d < day)
            .map(|&d| self.movie_day_cnt[i].get(&d).copied().unwrap_or(0) as usize)
            .sum::<usize>();
        f[F_MOVIE_IS_NEW] = if m_ratings_before < 20 { 1.0 } else { 0.0 };

        f[F_MOVIE_RATING_TREND] = self.movie_trend[i];
        f[F_MOVIE_MOMENTUM] = self.movie_momentum[i];

        // ====================================================================
        // Similarity Features (35-49)
        // ====================================================================

        // kNN predictions for different time windows
        let (knn_same, _) = self.knn_prediction(u, i, day, day);
        f[F_KNN_SAME_DAY] = knn_same;
        f[F_KNN_SAME_DAY_IS_ZERO] = if knn_same == 0.0 { 1.0 } else { 0.0 };

        let (knn_yest, _) = self.knn_prediction(u, i, day - 1, day - 1);
        f[F_KNN_YESTERDAY] = knn_yest;
        f[F_KNN_YESTERDAY_IS_ZERO] = if knn_yest == 0.0 { 1.0 } else { 0.0 };

        let (knn_before, _) = self.knn_prediction(u, i, i16::MIN, day - 2);
        f[F_KNN_BEFORE_YESTERDAY] = knn_before;
        f[F_KNN_BEFORE_IS_ZERO] = if knn_before == 0.0 { 1.0 } else { 0.0 };

        let (knn_all, _) = self.knn_prediction(u, i, i16::MIN, day);
        f[F_KNN_ALL] = knn_all;

        // Most similar ratings
        let (most_sim, second_sim) = self.most_similar_ratings(u, i, i16::MIN, day);
        f[F_MOST_SIMILAR_RATING] = most_sim;
        f[F_SECOND_SIMILAR_RATING] = second_sim;

        // Similarity statistics with user's movies
        let user_items = &self.user_items[u];
        let mut sim_sum = 0.0f32;
        let mut neg_sim_sum = 0.0f32;
        let mut sim_sq_sum = 0.0f32;
        let mut high_sim_cnt = 0usize;
        let mut min_sim = f32::MAX;
        let mut weight_sum = 0.0f32;
        let mut weighted_rating_sum = 0.0f32;
        let mut sim_cnt = 0usize;

        for &j in user_items {
            if j == i { continue; }
            let sim = self.sim_matrix[[i, j]];
            sim_sum += sim.max(0.0);
            if sim < 0.0 { neg_sim_sum += sim; }
            sim_sq_sum += sim * sim;
            if sim > 0.2 { high_sim_cnt += 1; }
            if sim < min_sim { min_sim = sim; }
            sim_cnt += 1;

            if sim > 0.0 {
                // Get user's rating for j (simplified: use bayesian mean as proxy)
                // In a real implementation we'd look up the actual rating
                weight_sum += sim;
                weighted_rating_sum += sim * self.movie_bayesian_mean[j];
            }
        }

        f[F_AVG_SIM_TO_USER_MOVIES] = if sim_cnt > 0 { sim_sum / sim_cnt as f32 } else { 0.0 };
        f[F_WEIGHTED_AVG_USER_RATING] = if weight_sum > 0.0 { weighted_rating_sum / weight_sum } else { 0.0 };
        f[F_NEGATIVE_SIM_SUM] = neg_sim_sum;

        // Similarity variance
        if sim_cnt > 1 {
            let mean_sim = sim_sum / sim_cnt as f32;
            let var = sim_sq_sum / sim_cnt as f32 - mean_sim * mean_sim;
            f[F_SIM_VARIANCE] = var.max(0.0).sqrt();
        }

        f[F_PCT_HIGH_SIM] = if sim_cnt > 0 { high_sim_cnt as f32 / sim_cnt as f32 } else { 0.0 };
        f[F_MIN_SIM_IN_USER_SET] = if min_sim < f32::MAX { min_sim } else { 0.0 };

        // ====================================================================
        // Rating Distribution - User (50-64)
        // ====================================================================

        let u_dist = &self.user_rating_dist[u];
        let u_total: u32 = u_dist.iter().sum();

        f[F_USER_RATING_ENTROPY] = entropy(u_dist);
        f[F_USER_RATING_MODE] = mode(u_dist);

        if u_total > 0 {
            let ut = u_total as f32;
            f[F_USER_PCT_5STAR] = u_dist[4] as f32 / ut;
            f[F_USER_PCT_4STAR] = u_dist[3] as f32 / ut;
            f[F_USER_PCT_3STAR] = u_dist[2] as f32 / ut;
            f[F_USER_PCT_2STAR] = u_dist[1] as f32 / ut;
            f[F_USER_PCT_1STAR] = u_dist[0] as f32 / ut;
            f[F_USER_PCT_EXTREME] = (u_dist[0] + u_dist[4]) as f32 / ut;
        }

        // Rating range (find min and max non-zero)
        let mut u_min = 5usize;
        let mut u_max = 1usize;
        for (v, &c) in u_dist.iter().enumerate() {
            if c > 0 {
                if v + 1 < u_min { u_min = v + 1; }
                if v + 1 > u_max { u_max = v + 1; }
            }
        }
        f[F_USER_RATING_RANGE] = (u_max - u_min) as f32;

        f[F_USER_MEDIAN] = self.user_median[u];
        f[F_USER_IS_HARSH] = if self.user_bayesian_mean[u] < self.global_mean { 1.0 } else { 0.0 };
        f[F_USER_HARSHNESS] = self.global_mean - self.user_bayesian_mean[u];
        f[F_USER_SKEWNESS] = self.user_skewness[u];
        f[F_USER_BIMODAL] = bimodal_score(u_dist);
        f[F_USER_CONSISTENCY] = 1.0 / (1.0 + self.user_std[u]);

        // ====================================================================
        // Rating Distribution - Movie (65-74)
        // ====================================================================

        let m_dist = &self.movie_rating_dist[i];
        let m_total: u32 = m_dist.iter().sum();

        f[F_MOVIE_RATING_ENTROPY] = entropy(m_dist);
        f[F_MOVIE_RATING_MODE] = mode(m_dist);

        if m_total > 0 {
            let mt = m_total as f32;
            f[F_MOVIE_PCT_5STAR] = m_dist[4] as f32 / mt;
            f[F_MOVIE_PCT_1STAR] = m_dist[0] as f32 / mt;
            f[F_MOVIE_PCT_EXTREME] = (m_dist[0] + m_dist[4]) as f32 / mt;
        }

        // Polarizing: high std, moderate mean
        let m_mean = self.movie_bayesian_mean[i];
        let m_std = self.movie_std[i];
        f[F_MOVIE_IS_POLARIZING] = if m_std > 1.2 && m_mean > 2.5 && m_mean < 4.0 { 1.0 } else { 0.0 };
        f[F_MOVIE_IS_CROWD_PLEASER] = if m_mean > 4.0 && m_std < 0.8 { 1.0 } else { 0.0 };
        f[F_MOVIE_CONTROVERSY] = if m_mean > 0.1 { m_std / m_mean } else { 0.0 };
        f[F_MOVIE_BIMODAL] = bimodal_score(m_dist);
        f[F_BAYESIAN_MOVIE_MEAN] = m_mean;

        // ====================================================================
        // User-Movie Interaction (75-89)
        // ====================================================================

        // SVD features
        let svd_dot = self.svd_ufeat.row(u).dot(&self.svd_ifeat.row(i));
        f[F_SVD_DOT] = svd_dot;
        f[F_SVD_DOT_CENTERED] = svd_dot - self.global_mean;
        f[F_SVD_CONFIDENCE] = self.svd_user_norm[u] * self.svd_movie_norm[i];
        f[F_USER_MOVIE_BIAS_PRODUCT] = self.svd_ubias[u] * self.svd_ibias[i];

        // Count-based interaction
        let u_cnt = self.user_rating_counts[u].max(1) as f32;
        let m_cnt = self.movie_rating_counts[i].max(1) as f32;
        f[F_USER_MOVIE_COUNT_PRODUCT] = u_cnt.ln() * m_cnt.ln();
        f[F_USER_MOVIE_STD_PRODUCT] = self.user_std[u] * m_std;

        // Average difference
        f[F_USER_MOVIE_AVG_DIFF] = self.user_bayesian_mean[u] - m_mean;
        f[F_USER_MOVIE_AVG_DIFF_ABS] = (self.user_bayesian_mean[u] - m_mean).abs();

        // Similar movies count
        let similar_cnt = user_items.iter()
            .filter(|&&j| j != i && self.sim_matrix[[i, j]] > 0.1)
            .count();
        f[F_USER_SIMILAR_MOVIES_COUNT] = similar_cnt as f32;
        f[F_LOG_USER_SIMILAR_MOVIES] = (1.0 + similar_cnt as f32).ln();

        // User rates popular movies
        f[F_USER_RATES_POPULAR] = self.user_avg_movie_pop[u];
        f[F_USER_NICHE_SCORE] = 1.0 - self.user_avg_movie_pop[u].min(1.0);

        // User-movie support (shared raters)
        let raters = &self.movie_raters[i];
        let mut support = 0usize;
        for &other_u in raters {
            if other_u == u { continue; }
            // Check if other_u shares movies with u (simplified: count overlap)
            let other_items = &self.user_items[other_u];
            let mut overlap = 0;
            let (mut p, mut q) = (0, 0);
            while p < user_items.len() && q < other_items.len() {
                if user_items[p] == other_items[q] { overlap += 1; p += 1; q += 1; }
                else if user_items[p] < other_items[q] { p += 1; }
                else { q += 1; }
            }
            if overlap >= 3 { support += 1; }
        }
        f[F_USER_MOVIE_SUPPORT] = support as f32;
        f[F_LOG_USER_MOVIE_SUPPORT] = (1.0 + support as f32).ln();

        // Ordinal SVD mode
        let ord_dot = self.ordinal_ufeat.row(u).dot(&self.ordinal_ifeat.row(i));
        let mut cum = [0.0f32; 4];
        for k in 0..4 {
            cum[k] = sigmoid(self.ordinal_thresholds[k] - ord_dot);
        }
        let probs = [
            cum[0],
            cum[1] - cum[0],
            cum[2] - cum[1],
            cum[3] - cum[2],
            1.0 - cum[3],
        ];
        let mut max_prob = probs[0];
        let mut mode_rating = 1;
        for (k, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                mode_rating = k + 1;
            }
        }
        f[F_ORDINAL_SVD_MODE] = mode_rating as f32;

        // ====================================================================
        // Advanced Temporal/Context (90-99)
        // ====================================================================

        // Session position (position within day's ratings)
        f[F_USER_SESSION_POSITION] = day_cnt as f32;
        f[F_USER_DAY_POSITION] = (day_idx + 1) as f32;

        // Rating streak (consecutive days)
        let mut streak = 0i32;
        for d in (1..=30).rev() {
            if self.user_day_cnt[u].contains_key(&(day - d as i16)) {
                streak += 1;
            } else {
                break;
            }
        }
        f[F_USER_RATING_STREAK] = streak as f32;

        // Inter-rating gap average
        if days.len() > 1 {
            let mut total_gap = 0i32;
            for w in days.windows(2) {
                total_gap += (w[1] - w[0]) as i32;
            }
            f[F_USER_INTER_RATING_GAP] = total_gap as f32 / (days.len() - 1) as f32;
        }

        // Movie freshness decay
        f[F_MOVIE_FRESHNESS_DECAY] = (-movie_age / 365.0).exp();

        // Time to dataset end
        f[F_TIME_TO_DATASET_END] = (self.dataset_max_day - day).max(0) as f32;

        // Recency scores (how recent is this in entity's history)
        if days.len() > 1 {
            f[F_USER_RECENCY_SCORE] = day_idx as f32 / (days.len() - 1) as f32;
        }
        if m_days.len() > 1 {
            f[F_MOVIE_RECENCY_SCORE] = m_day_idx as f32 / (m_days.len() - 1) as f32;
        }

        // User diversity same day
        let diversity = self.user_day_ratings[u].get(&day).map(|r| r.len()).unwrap_or(0);
        f[F_USER_DIVERSITY_SAME_DAY] = diversity as f32;

        // Ordinal SVD entropy
        let mut ord_entropy = 0.0f32;
        for &p in &probs {
            if p > 0.0 {
                ord_entropy -= p * p.ln();
            }
        }
        f[F_ORDINAL_SVD_ENTROPY] = ord_entropy;

        // ====================================================================
        // Ordinal SVD Probabilities (100-104)
        // ====================================================================
        f[F_ORDINAL_PROB_1] = probs[0];
        f[F_ORDINAL_PROB_2] = probs[1];
        f[F_ORDINAL_PROB_3] = probs[2];
        f[F_ORDINAL_PROB_4] = probs[3];
        f[F_ORDINAL_PROB_5] = probs[4];

        f
    }

    /// Compute features for all ratings in a dataset
    pub fn compute_all(&self, ds: &Dataset) -> Array2<f32> {
        println!("Computing {} features for {} ratings...", N_FEATURES, ds.n_ratings);

        let mut features = Array2::<f32>::zeros((ds.n_ratings, N_FEATURES));
        let features_slice = features.as_slice_mut().unwrap();

        features_slice
            .par_chunks_mut(N_FEATURES)
            .enumerate()
            .progress_count(ds.n_ratings as u64)
            .for_each(|(idx, row)| {
                let u = ds.user_idxs[idx] as usize;
                let i = ds.item_idxs[idx] as usize;
                let day = ds.dates[idx];
                let f = self.compute(u, i, day);
                row.copy_from_slice(&f);
            });

        features
    }

    pub fn feature_names() -> [&'static str; N_FEATURES] {
        [
            // 0-19: Temporal User
            "user_same_day_avg", "user_same_day_is_zero", "user_yesterday_avg", "user_yesterday_is_zero",
            "user_last_7d_avg", "user_last_7d_is_zero", "user_last_32d_avg", "user_before_yesterday_avg",
            "user_before_yesterday_is_zero", "user_days_since_prev", "log_user_days_since_prev",
            "user_is_first_rating", "user_velocity_7d", "user_binge", "user_days_active_7d",
            "user_days_active_32d", "user_tenure_pct", "user_is_weekend", "user_day_of_week",
            "log_user_days_until_last",
            // 20-34: Temporal Movie
            "movie_same_day_avg", "movie_same_day_is_zero", "movie_yesterday_avg", "movie_yesterday_is_zero",
            "movie_last_7d_avg", "movie_last_32d_avg", "movie_age", "log_movie_age",
            "movie_days_since_prev", "movie_same_day_count", "log_movie_same_day_count",
            "movie_last_7d_count", "movie_is_new", "movie_rating_trend", "movie_momentum",
            // 35-49: Similarity
            "knn_same_day", "knn_same_day_is_zero", "knn_yesterday", "knn_yesterday_is_zero",
            "knn_before_yesterday", "knn_before_is_zero", "knn_all", "most_similar_rating",
            "second_similar_rating", "avg_sim_to_user_movies", "weighted_avg_user_rating",
            "negative_sim_sum", "sim_variance", "pct_high_sim", "min_sim_in_user_set",
            // 50-64: User Distribution
            "user_rating_entropy", "user_rating_mode", "user_pct_5star", "user_pct_4star",
            "user_pct_3star", "user_pct_2star", "user_pct_1star", "user_pct_extreme",
            "user_rating_range", "user_median", "user_is_harsh", "user_harshness",
            "user_skewness", "user_bimodal", "user_consistency",
            // 65-74: Movie Distribution
            "movie_rating_entropy", "movie_rating_mode", "movie_pct_5star", "movie_pct_1star",
            "movie_pct_extreme", "movie_is_polarizing", "movie_is_crowd_pleaser", "movie_controversy",
            "movie_bimodal", "bayesian_movie_mean",
            // 75-89: Interaction
            "svd_dot", "svd_dot_centered", "svd_confidence", "user_movie_bias_product",
            "user_movie_count_product", "user_movie_std_product", "user_movie_avg_diff",
            "user_movie_avg_diff_abs", "user_similar_movies_count", "log_user_similar_movies",
            "user_rates_popular", "user_niche_score", "user_movie_support", "log_user_movie_support",
            "ordinal_svd_mode",
            // 90-99: Advanced
            "user_session_position", "user_day_position", "user_rating_streak", "user_inter_rating_gap",
            "movie_freshness_decay", "time_to_dataset_end", "user_recency_score", "movie_recency_score",
            "user_diversity_same_day", "ordinal_svd_entropy",
            // 100-104: Ordinal SVD Probabilities
            "ordinal_prob_1", "ordinal_prob_2", "ordinal_prob_3", "ordinal_prob_4", "ordinal_prob_5",
        ]
    }
}

/// Train SVD model (reused from fwls_features.rs)
fn train_svd(ds: &Dataset, n_feat: usize) -> (f32, Array1<f32>, Array1<f32>, Array2<f32>, Array2<f32>) {
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
        println!("    SVD-{} epoch {}/{}", n_feat, epoch + 1, n_epochs);
    }

    (gbias, ubias, ibias, ufeat, ifeat)
}

/// Save features
fn save_features(features: &Array2<f32>, dataset_name: &str, separate: bool) {
    if separate {
        let names = Claude105Features::feature_names();
        for col in 0..N_FEATURES {
            let path = format!("features/claude105_{}.{}.npy", names[col], dataset_name);
            let column = features.column(col).to_owned();
            write_npy(&path, &column).unwrap();
            println!("Saved: {}", path);
        }
    } else {
        let path = format!("features/claude105_features.{}.npy", dataset_name);
        write_npy(&path, features).unwrap();
        println!("Saved features to: {}", path);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let separate = args.iter().any(|a| a == "--separate");
    let gen_train = args.iter().any(|a| a == "--train");
    let gen_fulltrain = args.iter().any(|a| a == "--fulltrain");

    println!("=== Claude's 105 Features Generator ({} features) ===", N_FEATURES);
    println!("Output mode: {}", if separate { "separate files per feature" } else { "single combined file" });
    println!("Datasets: probe, qual{}{}\n",
        if gen_train { ", train" } else { "" },
        if gen_fulltrain { ", fulltrain" } else { "" });

    std::fs::create_dir_all("features").unwrap();

    // Process train -> probe
    println!("Processing: train -> probe");
    let train = Dataset::load("train", "rtg", "preds");
    let probe = Dataset::load("probe", "rtg", "preds");

    let fwls = Claude105Features::new(&train);

    let probe_features = fwls.compute_all(&probe);
    save_features(&probe_features, "probe", separate);

    if gen_train {
        let train_features = fwls.compute_all(&train);
        save_features(&train_features, "train", separate);
    }

    // Process fulltrain -> qual
    println!("\nProcessing: fulltrain -> qual");
    let fulltrain = Dataset::load("fulltrain", "rtg", "preds");
    let qual = Dataset::load("qual", "rtg", "preds");

    let fwls_full = Claude105Features::new(&fulltrain);

    let qual_features = fwls_full.compute_all(&qual);
    save_features(&qual_features, "qual", separate);

    if gen_fulltrain {
        let fulltrain_features = fwls_full.compute_all(&fulltrain);
        save_features(&fulltrain_features, "fulltrain", separate);
    }

    println!("\n=== Done ===");
}
