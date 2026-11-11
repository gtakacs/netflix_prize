// Voting feature set #0: the old split's voting features (voting-old.toml) that vfeat1 does
// not reproduce. The other 44 of its fwls A-F features are vfeat1's vf000-vf044 (bit for bit
// or to float rounding, checked on split_1 against preds_old/fwls/).
//
//   fwls_c5_user_avg_movie_pop        fwls C5 (src/old/fwls_featuresC.rs), with the mean movie
//                                     rating COUNT as prior (vfeat1's vf016 uses the rating sum)
//   c105_015_user_days_active_32d     claude105 (src/old/fwls_features_claude105.rs) features
//   c105_025_movie_last_32d_avg       by their old index and name
//   c105_026_movie_age
//   c105_030_log_movie_same_day_count
//   c105_035_knn_same_day
//   c105_050_user_rating_entropy
//   c105_082_user_movie_avg_diff_abs
//   c105_084_log_user_similar_movies
//   c105_085_user_rates_popular
//   c105_088_log_user_movie_support
//   c105_097_movie_recency_score
//
// The formulas and float types follow the old code. Two things differ by design: the item-item
// similarity (035, 084) is accumulated per item row in a fixed order instead of the old code's
// user-parallel order (which made it run-dependent), and the shared-rater support (088) is
// counted with a bitset of the user's items and an early exit at 3 instead of a full merge per
// rater (same integer result, far cheaper).
//
// No target leakage: the statistics are built from the training set only, and the set the
// features are emitted for reaches `compute_all` only as a `MaskedDataset`.

use crate::{Dataset, MaskedDataset, Split};
use indicatif::ParallelProgressIterator;
use ndarray::Array1;
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

const BAYESIAN_K: f64 = 25.0;  // fwls and claude105 Bayesian shrinkage
const REG_K: f32 = 5.0;        // claude105 temporal averages
const SIM_SHRINKAGE: f32 = 100.0;

pub const N_FEATURES: usize = 12;
pub const NAMES: [&str; N_FEATURES] = [
    "fwls_c5_user_avg_movie_pop",
    "c105_015_user_days_active_32d",
    "c105_025_movie_last_32d_avg",
    "c105_026_movie_age",
    "c105_030_log_movie_same_day_count",
    "c105_035_knn_same_day",
    "c105_050_user_rating_entropy",
    "c105_082_user_movie_avg_diff_abs",
    "c105_084_log_user_similar_movies",
    "c105_085_user_rates_popular",
    "c105_088_log_user_movie_support",
    "c105_097_movie_recency_score",
];

/// Entropy of a 1..5 rating histogram (claude105 `entropy`).
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

/// Per-day running sums of one movie: (day, cumulative sum, cumulative count, count that day).
struct DaySeries {
    days: Vec<(i16, f32, u32, u32)>,
}

impl DaySeries {
    /// Sum and count over the days in [day_start, day_end] (claude105 `range_sum_count`).
    fn range_sum_count(&self, day_start: i16, day_end: i16) -> (f32, u32) {
        let d = &self.days;
        let end = d.partition_point(|x| x.0 <= day_end);
        if end == 0 { return (0.0, 0); }
        let (_, es, ec, _) = d[end - 1];
        let start = d.partition_point(|x| x.0 < day_start);
        let (ss, sc) = if start == 0 { (0.0, 0) } else { (d[start - 1].1, d[start - 1].2) };
        (es - ss, ec - sc)
    }
}

pub struct VotingFeatures0 {
    n_users: usize,
    global_mean: f32,
    user_bayes: Vec<f32>,
    movie_bayes: Vec<f32>,
    user_entropy: Vec<f32>,
    user_pop_c5: Vec<f32>,
    user_pop_c105: Vec<f32>,
    user_days: Vec<Vec<i16>>,                 // distinct rating days, sorted
    user_day_ratings: Vec<Vec<(i16, u16, f32)>>, // (day, item, rating), by day, then data order
    user_items: Vec<Vec<u16>>,                // sorted
    user_items_pop: Vec<Vec<u16>>,            // most rated movies first (088 early exit)
    movie_series: Vec<DaySeries>,
    movie_first_date: Vec<i16>,
    movie_raters: Vec<Vec<u32>>,
    sim_top: Vec<Vec<(u16, f32)>>,            // top 100 with sim > 0.01, by sim desc
    sim_hi: Vec<Vec<u16>>,                    // items with sim > 0.1, sorted
}

impl VotingFeatures0 {
    /// The statistics of the training set `ds` (its ratings are read here, and only here).
    pub fn new(ds: &Dataset) -> Self {
        let (nu, ni) = (ds.n_users, ds.n_items);
        crate::teeln!("vfeat0: statistics over {} ratings ({} users, {} items)", ds.n_ratings, nu, ni);

        let mut user_cnt = vec![0u32; nu];
        let mut user_sum = vec![0.0f64; nu];
        let mut user_dist = vec![[0u32; 5]; nu];
        let mut movie_cnt = vec![0u32; ni];
        let mut movie_sum = vec![0.0f64; ni];
        let mut movie_first_date = vec![i16::MAX; ni];
        let (mut total_sum, mut total_cnt) = (0.0f64, 0u64);
        let mut user_day_ratings: Vec<Vec<(i16, u16, f32)>> = vec![Vec::new(); nu];
        let mut movie_day: Vec<Vec<(i16, f32)>> = vec![Vec::new(); ni];
        for idx in 0..ds.n_ratings {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            let r = ds.raw_ratings[idx];
            let day = ds.dates[idx];
            user_cnt[u] += 1;
            user_sum[u] += r as f64;
            if (1..=5).contains(&r) { user_dist[u][(r - 1) as usize] += 1; }
            movie_cnt[i] += 1;
            movie_sum[i] += r as f64;
            if day < movie_first_date[i] { movie_first_date[i] = day; }
            total_sum += r as f64;
            total_cnt += 1;
            user_day_ratings[u].push((day, i as u16, r as f32));
            movie_day[i].push((day, r as f32));
        }
        let global_mean = (total_sum / total_cnt as f64) as f32;
        let bayes = |s: f64, c: u32| if c > 0 {
            ((s + BAYESIAN_K * global_mean as f64) / (c as f64 + BAYESIAN_K)) as f32
        } else { 0.0 };
        let user_bayes: Vec<f32> = (0..nu).map(|u| bayes(user_sum[u], user_cnt[u])).collect();
        let movie_bayes: Vec<f32> = (0..ni).map(|i| bayes(movie_sum[i], movie_cnt[i])).collect();
        let user_entropy: Vec<f32> = user_dist.iter().map(entropy).collect();

        // movie popularity per user: fwls C5 (prior = mean movie count) and claude105 085
        let mut user_sum_pop = vec![0.0f64; nu];
        for idx in 0..ds.n_ratings {
            user_sum_pop[ds.user_idxs[idx] as usize] += movie_cnt[ds.item_idxs[idx] as usize] as f64;
        }
        let avg_movie_cnt = total_cnt as f64 / ni as f64;
        let user_pop_c5: Vec<f32> = (0..nu).map(|u| if user_cnt[u] > 0 {
            ((user_sum_pop[u] + BAYESIAN_K * avg_movie_cnt) / (user_cnt[u] as f64 + BAYESIAN_K)) as f32
        } else { 0.0 }).collect();
        let user_pop_c105: Vec<f32> = (0..nu).map(|u| if user_cnt[u] > 0 {
            ((user_sum_pop[u] / user_cnt[u] as f64) / avg_movie_cnt) as f32
        } else { 0.0 }).collect();

        // per-day series: users' distinct days, movies' running sums (claude105 accumulates the
        // day sums in data order, then the running sums over the sorted days)
        for v in user_day_ratings.iter_mut() { v.sort_by_key(|x| x.0); }
        let user_days: Vec<Vec<i16>> = user_day_ratings.iter().map(|v| {
            let mut d: Vec<i16> = v.iter().map(|x| x.0).collect();
            d.dedup();
            d
        }).collect();
        let movie_series: Vec<DaySeries> = movie_day.into_iter().map(|mut v| {
            v.sort_by_key(|x| x.0);  // stable: data order within a day
            let mut days: Vec<(i16, f32, u32, u32)> = Vec::new();
            let (mut day_sum, mut day_cnt) = (0.0f32, 0u32);
            let (mut cs, mut cc) = (0.0f32, 0u32);
            for (k, &(d, r)) in v.iter().enumerate() {
                day_sum += r;
                day_cnt += 1;
                if k + 1 == v.len() || v[k + 1].0 != d {
                    cs += day_sum;
                    cc += day_cnt;
                    days.push((d, cs, cc, day_cnt));
                    day_sum = 0.0;
                    day_cnt = 0;
                }
            }
            DaySeries { days }
        }).collect();

        // items per user (sorted, and most rated first), raters per movie
        let mut user_items: Vec<Vec<u16>> = vec![Vec::new(); nu];
        let mut movie_raters: Vec<Vec<u32>> = vec![Vec::new(); ni];
        let mut user_cent: Vec<Vec<(u16, f32)>> = vec![Vec::new(); nu];
        let mut movie_cent: Vec<Vec<(u32, f32)>> = vec![Vec::new(); ni];
        let movie_mean: Vec<f32> = (0..ni).map(|i| if movie_cnt[i] > 0 {
            (movie_sum[i] / movie_cnt[i] as f64) as f32
        } else { 0.0 }).collect();
        for idx in 0..ds.n_ratings {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            let rc = ds.raw_ratings[idx] as f32 - movie_mean[i];
            user_items[u].push(i as u16);
            movie_raters[i].push(u as u32);
            user_cent[u].push((i as u16, rc));
            movie_cent[i].push((u as u32, rc));
        }
        let user_items_pop: Vec<Vec<u16>> = user_items.iter().map(|v| {
            let mut w = v.clone();
            w.sort_by_key(|&j| std::cmp::Reverse(movie_cnt[j as usize]));
            w
        }).collect();
        for v in user_items.iter_mut() { v.sort_unstable(); }

        // item-item similarity on movie-mean-centred ratings (claude105): one item row at a
        // time, its raters in data order, so the result does not depend on the thread count
        crate::teeln!("vfeat0: item-item similarities");
        let norms: Vec<f32> = movie_cent.iter()
            .map(|v| v.iter().fold(0.0f32, |s, &(_, r)| s + r * r).max(0.0).sqrt()).collect();
        let pb = crate::make_pb(ni as u64);
        let rows: Vec<(Vec<(u16, f32)>, Vec<u16>)> = (0..ni).into_par_iter().progress_with(pb).map(|i| {
            let mut supp = vec![0.0f32; ni];
            let mut prod = vec![0.0f32; ni];
            for &(u, ri) in &movie_cent[i] {
                for &(j, rj) in &user_cent[u as usize] {
                    supp[j as usize] += 1.0;
                    prod[j as usize] += ri * rj;
                }
            }
            let mut sims: Vec<(u16, f32)> = Vec::new();
            let mut hi: Vec<u16> = Vec::new();
            for j in 0..ni {
                if j == i { continue; }
                let n = supp[j];
                if n < 2.0 { continue; }
                let den = norms[i] * norms[j];
                let phi = if den > 0.0 { prod[j] / den } else { 0.0 };
                let sim = phi * n / (n + SIM_SHRINKAGE);
                if sim > 0.1 { hi.push(j as u16); }
                if sim > 0.01 { sims.push((j as u16, sim)); }
            }
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(100);
            (sims, hi)
        }).collect();
        let (sim_top, sim_hi): (Vec<_>, Vec<_>) = rows.into_iter().unzip();

        Self {
            n_users: nu, global_mean, user_bayes, movie_bayes, user_entropy, user_pop_c5,
            user_pop_c105, user_days, user_day_ratings, user_items, user_items_pop, movie_series,
            movie_first_date, movie_raters, sim_top, sim_hi,
        }
    }

    pub fn feature_names() -> [&'static str; N_FEATURES] { NAMES }

    /// claude105 088: raters of movie i (other than u) sharing at least 3 movies with u.
    fn support(&self, u: usize, i: usize, mine: &[u64]) -> u32 {
        let mut support = 0u32;
        for &v in &self.movie_raters[i] {
            let v = v as usize;
            if v == u { continue; }
            let mut overlap = 0;
            for &j in &self.user_items_pop[v] {
                if mine[j as usize >> 6] >> (j & 63) & 1 == 1 {
                    overlap += 1;
                    if overlap >= 3 { support += 1; break; }
                }
            }
        }
        support
    }

    fn compute(&self, u: usize, i: usize, day: i16, mine: &[u64]) -> [f32; N_FEATURES] {
        let mut f = [0.0f32; N_FEATURES];
        f[0] = self.user_pop_c5[u];

        let days = &self.user_days[u];
        f[1] = days.iter().filter(|&&d| d >= day - 32 && d < day).count() as f32;

        let ms = &self.movie_series[i];
        let (sum, cnt) = ms.range_sum_count(day - 32, day - 1);
        f[2] = if cnt == 0 { 0.0 } else { (sum + REG_K * self.global_mean) / (cnt as f32 + REG_K) };

        f[3] = (day - self.movie_first_date[i]).max(0) as f32;

        let k = ms.days.partition_point(|x| x.0 < day);
        let m_day_cnt = if k < ms.days.len() && ms.days[k].0 == day { ms.days[k].3 } else { 0 };
        f[4] = (1.0 + m_day_cnt as f32).ln();

        // kNN over the user's same-day ratings (claude105 `knn_prediction(u, i, day, day)`)
        let top = &self.sim_top[i];
        if !top.is_empty() {
            let r = &self.user_day_ratings[u];
            let s = r.partition_point(|x| x.0 < day);
            let (mut ws, mut ss) = (0.0f32, 0.0f32);
            for &(d, j, rating) in &r[s..] {
                if d != day { break; }
                if j as usize == i { continue; }
                for &(t, sim) in top {
                    if t == j && sim > 0.0 { ws += sim * rating; ss += sim; break; }
                }
            }
            if ss > 0.0 { f[5] = ws / ss; }
        }

        f[6] = self.user_entropy[u];
        f[7] = (self.user_bayes[u] - self.movie_bayes[i]).abs();

        // user's movies with sim(i, j) > 0.1
        let (a, b) = (&self.user_items[u], &self.sim_hi[i]);
        let (mut p, mut q, mut similar) = (0, 0, 0u32);
        while p < a.len() && q < b.len() {
            if a[p] == b[q] { if a[p] as usize != i { similar += 1; } p += 1; q += 1; }
            else if a[p] < b[q] { p += 1; } else { q += 1; }
        }
        f[8] = (1.0 + similar as f32).ln();

        f[9] = self.user_pop_c105[u];
        f[10] = (1.0 + self.support(u, i, mine) as f32).ln();
        if ms.days.len() > 1 { f[11] = k as f32 / (ms.days.len() - 1) as f32; }
        f
    }

    /// The features of every (user, item, date) of the prediction set `ds`; masked, so its
    /// targets cannot be read. Rows are processed per user (the user's item bitset for 088).
    pub fn compute_all(&self, ds: &MaskedDataset) -> Vec<Array1<f32>> {
        crate::teeln!("vfeat0: computing {} features for {} ratings...", N_FEATURES, ds.n_ratings);
        let mut by_user: Vec<Vec<usize>> = vec![Vec::new(); self.n_users];
        for idx in 0..ds.n_ratings { by_user[ds.user_idxs[idx] as usize].push(idx); }
        let words = (self.movie_series.len() + 63) / 64;
        let pb = crate::make_pb(self.n_users as u64);
        let rows: Vec<Vec<(usize, [f32; N_FEATURES])>> = by_user.par_iter().enumerate()
            .progress_with(pb)
            .map(|(u, idxs)| {
                if idxs.is_empty() { return Vec::new(); }
                let mut mine = vec![0u64; words];
                for &j in &self.user_items[u] { mine[j as usize >> 6] |= 1 << (j & 63); }
                idxs.iter().map(|&idx| {
                    (idx, self.compute(u, ds.item_idxs[idx] as usize, ds.dates[idx], &mine))
                }).collect()
            }).collect();
        let mut features: Vec<Array1<f32>> = (0..N_FEATURES).map(|_| Array1::zeros(ds.n_ratings)).collect();
        for v in rows {
            for (idx, f) in v {
                for k in 0..N_FEATURES { features[k][idx] = f[k]; }
            }
        }
        features
    }
}

/// Compute and save voting feature set #0 over the standard split cycle (as `save_vfeat1`):
/// train on `split.tr`, emit over `split.pr`; train on `split.fulltrain_tr`, emit over
/// `split.fulltrain_pr`. Files go to `<preds_dir>/vf0/<feature>.<set>.npy`, the run log to
/// `<preds_dir>/<name>.out`.
pub fn save_vfeat0(name: &str, split: Split) {
    let vf_dir = format!("{}/vf0", split.preds_dir);
    std::fs::create_dir_all(&vf_dir).unwrap();
    let owns_log = crate::LOG_FILE.lock().unwrap().is_none();
    if owns_log {
        *crate::LOG_FILE.lock().unwrap() = Some(BufWriter::new(
            File::create(format!("{}/{}.out", split.preds_dir, name)).unwrap()
        ));
    }
    crate::teeln!("[{}]", name);
    for (tr_set, pr_set) in [(split.tr, split.pr), (split.fulltrain_tr, split.fulltrain_pr)] {
        crate::teeln!("vfeat0: {} -> {}", tr_set, pr_set);
        let tr = Dataset::load(tr_set, "rtg", split.preds_dir);
        let stats = VotingFeatures0::new(&tr);
        drop(tr);
        let pr = Dataset::load(pr_set, "rtg", split.preds_dir);
        let features = stats.compute_all(&MaskedDataset::from(&pr));
        for (k, feat) in features.iter().enumerate() {
            let path = format!("{}/{}.{}.npy", vf_dir, NAMES[k], pr_set);
            write_npy(&path, feat).unwrap();
            crate::teeln!("vfeat0: saved {}", path);
        }
    }
    if owns_log {
        if let Some(mut lf) = crate::LOG_FILE.lock().unwrap().take() { let _ = lf.flush(); }
    }
}
