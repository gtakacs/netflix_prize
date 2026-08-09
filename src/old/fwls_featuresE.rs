// FWLS voting-feature generator, phase E — 5 features: SVD factor norms, ordinal-SVD
// stddev, user overlap, same-day correlation. Trains its own throwaway SVDs inline.
// Frozen archive — see README.md; superseded by src/vfeat1.rs. Writes
// features/fwls_E{k}.{set}.npy and the shipped files were renamed by hand into
// preds_old/fwls/E{k}.*.npy, the layout the current blenders load.
//
// Phase E features (5 features):
// - f08_svd_user_norm (norm of user SVD factor vector, 10-factor)
// - f09_svd_movie_norm (norm of movie SVD factor vector, 10-factor)
// - f11_ordinal_svd_std (std dev of 60-factor ordinal SVD prediction)
// - f22_user_overlap (avg user-pair movie set overlap for movie)
// - f25_sameday_corr (same-day probability vs ratings correlation)

use gravity::fwls_common::{FwlsBase, SIM_SHRINKAGE, build_user_items, sigmoid};
use gravity::{Dataset, calc_gbias, rand_array2};
use indicatif::ParallelProgressIterator;
use ndarray::{Array1, Array2};
use ndarray_npy::WriteNpyExt;
use parking_lot::Mutex;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

fn save_features(features: &[Array1<f32>], set_name: &str) {
    use std::io::Write;
    for (k, feat) in features.iter().enumerate() {
        let path = format!("features/fwls_E{}.{}.npy", k + 1, set_name);
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        feat.write_npy(&mut writer).unwrap();
        writer.flush().unwrap();
        println!("  Saved: {}", path);
    }
}

const N_FEATURES: usize = 5;

/// Phase E feature indices
const F_SVD_USER_NORM: usize = 0;
const F_SVD_MOVIE_NORM: usize = 1;
const F_ORDINAL_SVD_STD: usize = 2;
const F_USER_OVERLAP: usize = 3;
const F_SAMEDAY_CORR: usize = 4;

/// Train a matrix factorization SVD model
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
        println!("    SVD-{} epoch {}/{}", n_feat, epoch + 1, n_epochs);
    }

    (gbias, ubias, ibias, ufeat, ifeat)
}

pub struct FwlsFeaturesE {
    n_users: usize,
    n_items: usize,
    svd_user_norm: Array1<f32>,
    svd_movie_norm: Array1<f32>,
    svd60_ufeat: Array2<f32>,
    svd60_ifeat: Array2<f32>,
    ordinal_thresholds: [f32; 4],
    movie_user_overlap: Array1<f32>,
    movie_sameday_corr: Array1<f32>,
}

impl FwlsFeaturesE {
    pub fn new(ds: &Dataset) -> Self {
        println!("Computing FWLS Phase E statistics...");
        let base = FwlsBase::new(ds);
        let n_users = base.n_users;
        let n_items = base.n_items;

        // Build user item lists for overlap calculation
        let user_items = build_user_items(ds, &base.user_starts);

        // Compute similarity matrix and same-day accumulation for feature 25
        println!("  Computing item-item co-rating statistics...");
        let movie_means: Vec<f32> = (0..n_items).map(|i| {
            let cnt = base.movie_rating_counts[i] as f64;
            if cnt > 0.0 { (base.movie_sum_ratings[i] / cnt) as f32 } else { 0.0 }
        }).collect();

        let supp_rows: Vec<Mutex<Vec<f32>>> =
            (0..n_items).map(|_| Mutex::new(vec![0.0; n_items])).collect();
        let prod_rows: Vec<Mutex<Vec<f32>>> =
            (0..n_items).map(|_| Mutex::new(vec![0.0; n_items])).collect();
        let sameday_rows: Vec<Mutex<Vec<f32>>> =
            (0..n_items).map(|_| Mutex::new(vec![0.0; n_items])).collect();

        (0..n_users).into_par_iter().progress_count(n_users as u64).for_each(|u| {
            let start = base.user_starts[u];
            let end = base.user_starts[u + 1];
            if start == end { return; }

            let items: Vec<(usize, f32, i16)> = (start..end).map(|idx| {
                let i = ds.item_idxs[idx] as usize;
                let r = ds.raw_ratings[idx] as f32 - movie_means[i];
                let d = ds.dates[idx];
                (i, r, d)
            }).collect();

            for &(i, ri, di) in &items {
                let mut supp_row = supp_rows[i].lock();
                let mut prod_row = prod_rows[i].lock();
                let mut sameday_row = sameday_rows[i].lock();
                for &(j, rj, dj) in &items {
                    supp_row[j] += 1.0;
                    prod_row[j] += ri * rj;
                    if di == dj {
                        sameday_row[j] += 1.0;
                    }
                }
            }
        });

        // Convert to similarity matrix
        println!("  Converting to similarity matrix...");
        let norms: Vec<f32> = (0..n_items).map(|i| {
            prod_rows[i].lock()[i].max(0.0).sqrt()
        }).collect();

        let mut sim_matrix = Array2::<f32>::zeros((n_items, n_items));
        for i in 0..n_items {
            let supp_row = supp_rows[i].lock();
            let prod_row = prod_rows[i].lock();
            for j in 0..n_items {
                if i == j { continue; }
                let n = supp_row[j];
                if n < 2.0 { continue; }
                let den = norms[i] * norms[j];
                let phi = if den > 0.0 { prod_row[j] / den } else { 0.0 };
                let sim = phi * n / (n + SIM_SHRINKAGE);
                sim_matrix[[i, j]] = sim;
            }
        }
        drop(prod_rows);

        // Feature 25: same-day probability vs ratings correlation
        println!("  Computing feature 25 (same-day correlation)...");
        let movie_sameday_corr: Vec<f32> = (0..n_items).into_par_iter().map(|i| {
            let supp_row = supp_rows[i].lock();
            let sameday_row = sameday_rows[i].lock();
            let mut n = 0u64;
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_xy = 0.0f64;
            let mut sum_x2 = 0.0f64;
            let mut sum_y2 = 0.0f64;
            for j in 0..n_items {
                if i == j { continue; }
                let supp = supp_row[j];
                if supp < 2.0 { continue; }
                let x = (sameday_row[j] / supp) as f64;
                let y = sim_matrix[[i, j]] as f64;
                n += 1;
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
                sum_y2 += y * y;
            }
            if n > 1 {
                let nf = n as f64;
                let cov = sum_xy / nf - (sum_x / nf) * (sum_y / nf);
                let var_x = sum_x2 / nf - (sum_x / nf) * (sum_x / nf);
                let var_y = sum_y2 / nf - (sum_y / nf) * (sum_y / nf);
                let den = (var_x * var_y).sqrt();
                if den > 1e-10 { (cov / den) as f32 } else { 0.0 }
            } else {
                0.0
            }
        }).collect();
        let movie_sameday_corr = Array1::from(movie_sameday_corr);
        drop(supp_rows);
        drop(sameday_rows);
        drop(sim_matrix);

        // Train 10-factor SVD for features 8, 9
        println!("  Training 10-factor SVD...");
        let (_, _, _, svd10_ufeat, svd10_ifeat) = train_svd(ds, 10);
        let svd_user_norm = Array1::from_iter(
            (0..n_users).map(|u| {
                let row = svd10_ufeat.row(u);
                row.dot(&row).sqrt()
            })
        );
        let svd_movie_norm = Array1::from_iter(
            (0..n_items).map(|i| {
                let row = svd10_ifeat.row(i);
                row.dot(&row).sqrt()
            })
        );
        drop(svd10_ufeat);
        drop(svd10_ifeat);

        // Train 60-factor SVD for feature 11
        println!("  Training 60-factor SVD...");
        let (_, _, _, svd60_ufeat, svd60_ifeat) = train_svd(ds, 60);

        // Estimate ordinal thresholds
        println!("  Estimating ordinal thresholds...");
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
        let mut ordinal_thresholds = [0.0f32; 4];
        for k in 0..4 {
            ordinal_thresholds[k] = ((rating_means[k] + rating_means[k + 1]) / 2.0) as f32;
        }
        println!("  Ordinal thresholds: {:?}", ordinal_thresholds);

        // Feature 22: average user-pair movie set overlap
        println!("  Computing feature 22 (user overlap)...");
        let mut movie_users: Vec<Vec<usize>> = vec![vec![]; n_items];
        for idx in 0..ds.n_ratings {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            movie_users[i].push(u);
        }
        let movie_user_overlap: Vec<f32> = (0..n_items).into_par_iter()
            .progress_count(n_items as u64)
            .map(|m| {
                let raters = &movie_users[m];
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
        let movie_user_overlap = Array1::from(movie_user_overlap);

        Self {
            n_users,
            n_items,
            svd_user_norm,
            svd_movie_norm,
            svd60_ufeat,
            svd60_ifeat,
            ordinal_thresholds,
            movie_user_overlap,
            movie_sameday_corr,
        }
    }

    #[inline]
    pub fn compute(&self, u: usize, i: usize, _day: i16) -> [f32; N_FEATURES] {
        let mut f = [0.0_f32; N_FEATURES];

        // Feature 8: norm of user SVD factor vector
        f[F_SVD_USER_NORM] = self.svd_user_norm[u];

        // Feature 9: norm of movie SVD factor vector
        f[F_SVD_MOVIE_NORM] = self.svd_movie_norm[i];

        // Feature 11: std dev of 60-factor ordinal SVD prediction
        let dot = self.svd60_ufeat.row(u).dot(&self.svd60_ifeat.row(i));
        let mut cum = [0.0f32; 4];
        for k in 0..4 {
            cum[k] = sigmoid(self.ordinal_thresholds[k] - dot);
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
        f[F_ORDINAL_SVD_STD] = (mean_sq - mean * mean).max(0.0).sqrt();

        // Feature 22: average user-pair movie set overlap
        f[F_USER_OVERLAP] = self.movie_user_overlap[i];

        // Feature 25: same-day probability vs ratings correlation
        f[F_SAMEDAY_CORR] = self.movie_sameday_corr[i];

        f
    }

    pub fn compute_all(&self, ds: &Dataset) -> Vec<Array1<f32>> {
        println!("Computing {} Phase E features for {} ratings...", N_FEATURES, ds.n_ratings);

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
            "f08_svd_user_norm",
            "f09_svd_movie_norm",
            "f11_ordinal_svd_std",
            "f22_user_overlap",
            "f25_sameday_corr",
        ]
    }

    pub fn print_summary(&self) {
        println!("\nFWLS Phase E Statistics ({} features):", N_FEATURES);
        println!("  Users: {}", self.n_users);
        println!("  Items: {}", self.n_items);
        println!("  Ordinal thresholds: {:?}", self.ordinal_thresholds);

        println!("\nFeatures:");
        for (i, name) in Self::feature_names().iter().enumerate() {
            println!("  [{:2}] {}", i, name);
        }
    }
}

fn main() {
    println!("=== FWLS Meta-Features Generator (Phase E, {} features) ===\n", N_FEATURES);

    std::fs::create_dir_all("features").unwrap();

    // Process train -> probe
    println!("Processing: train -> probe");
    let train = Dataset::load("train", "rtg", "preds");
    let probe = Dataset::load("probe", "rtg", "preds");

    let fwls = FwlsFeaturesE::new(&train);
    fwls.print_summary();

    let probe_features = fwls.compute_all(&probe);
    save_features(&probe_features, "probe");

    // Process fulltrain -> qual
    println!("\nProcessing: fulltrain -> qual");
    let fulltrain = Dataset::load("fulltrain", "rtg", "preds");
    let qual = Dataset::load("qual", "rtg", "preds");

    let fwls_full = FwlsFeaturesE::new(&fulltrain);

    let qual_features = fwls_full.compute_all(&qual);
    save_features(&qual_features, "qual");

    println!("\n=== Done ===");
}
