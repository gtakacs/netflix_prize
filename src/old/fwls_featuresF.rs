// FWLS voting-feature generator, phase F — 20 features encoding "the user gave rating r
// on this / on another day", as binary indicators and as log(1 + count).
// Frozen archive — see README.md; superseded by src/vfeat1.rs. Writes
// features/fwls_F{k}.{set}.npy and the shipped files were renamed by hand into
// preds_old/fwls/F{k}.*.npy, the layout the current blenders load.
//
// Phase F features (20 features):
// Binary features (exists ? 1 : 0):
// - fF1_user_has_r1_same_day ... fF5_user_has_r5_same_day
// - fF6_user_has_r1_other_day ... fF10_user_has_r5_other_day
// Log(1+count) features:
// - fF11_user_log_r1_same_day ... fF15_user_log_r5_same_day
// - fF16_user_log_r1_other_day ... fF20_user_log_r5_other_day

use gravity::fwls_common::FwlsBase;
use gravity::Dataset;
use indicatif::ParallelProgressIterator;
use ndarray::Array1;
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;

fn save_features(features: &[Array1<f32>], set_name: &str) {
    use std::io::Write;
    for (k, feat) in features.iter().enumerate() {
        let path = format!("features/fwls_F{}.{}.npy", k + 1, set_name);
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        feat.write_npy(&mut writer).unwrap();
        writer.flush().unwrap();
        println!("  Saved: {}", path);
    }
}

const N_FEATURES: usize = 20;

/// Phase F feature indices - Binary features
const F_USER_HAS_R1_SAME_DAY: usize = 0;
const F_USER_HAS_R2_SAME_DAY: usize = 1;
const F_USER_HAS_R3_SAME_DAY: usize = 2;
const F_USER_HAS_R4_SAME_DAY: usize = 3;
const F_USER_HAS_R5_SAME_DAY: usize = 4;
const F_USER_HAS_R1_OTHER_DAY: usize = 5;
const F_USER_HAS_R2_OTHER_DAY: usize = 6;
const F_USER_HAS_R3_OTHER_DAY: usize = 7;
const F_USER_HAS_R4_OTHER_DAY: usize = 8;
const F_USER_HAS_R5_OTHER_DAY: usize = 9;
/// Phase F feature indices - Log(1+count) features
const F_USER_LOG_R1_SAME_DAY: usize = 10;
const F_USER_LOG_R2_SAME_DAY: usize = 11;
const F_USER_LOG_R3_SAME_DAY: usize = 12;
const F_USER_LOG_R4_SAME_DAY: usize = 13;
const F_USER_LOG_R5_SAME_DAY: usize = 14;
const F_USER_LOG_R1_OTHER_DAY: usize = 15;
const F_USER_LOG_R2_OTHER_DAY: usize = 16;
const F_USER_LOG_R3_OTHER_DAY: usize = 17;
const F_USER_LOG_R4_OTHER_DAY: usize = 18;
const F_USER_LOG_R5_OTHER_DAY: usize = 19;

pub struct FwlsFeaturesF {
    base: FwlsBase,
}

impl FwlsFeaturesF {
    pub fn new(ds: &Dataset) -> Self {
        println!("Computing FWLS Phase F statistics...");
        let base = FwlsBase::new(ds);
        Self { base }
    }

    #[inline]
    pub fn compute(&self, u: usize, _i: usize, day: i16) -> [f32; N_FEATURES] {
        let mut f = [0.0_f32; N_FEATURES];

        // Get rating counts for this user on this day
        let day_counts = self.base.user_date_rating_counts[u].get(&day);
        let total_counts = &self.base.user_rating_counts_by_value[u];

        let dc = day_counts.map(|x| *x).unwrap_or([0u32; 5]);

        // Binary features: user has rating of value 1..5 on this day
        f[F_USER_HAS_R1_SAME_DAY] = if dc[0] > 0 { 1.0 } else { 0.0 };
        f[F_USER_HAS_R2_SAME_DAY] = if dc[1] > 0 { 1.0 } else { 0.0 };
        f[F_USER_HAS_R3_SAME_DAY] = if dc[2] > 0 { 1.0 } else { 0.0 };
        f[F_USER_HAS_R4_SAME_DAY] = if dc[3] > 0 { 1.0 } else { 0.0 };
        f[F_USER_HAS_R5_SAME_DAY] = if dc[4] > 0 { 1.0 } else { 0.0 };

        // Log(1+count) features for same day
        f[F_USER_LOG_R1_SAME_DAY] = (1.0 + dc[0] as f32).ln();
        f[F_USER_LOG_R2_SAME_DAY] = (1.0 + dc[1] as f32).ln();
        f[F_USER_LOG_R3_SAME_DAY] = (1.0 + dc[2] as f32).ln();
        f[F_USER_LOG_R4_SAME_DAY] = (1.0 + dc[3] as f32).ln();
        f[F_USER_LOG_R5_SAME_DAY] = (1.0 + dc[4] as f32).ln();

        // Other days: total_counts - day_counts
        let other_counts = [
            total_counts[0].saturating_sub(dc[0]),
            total_counts[1].saturating_sub(dc[1]),
            total_counts[2].saturating_sub(dc[2]),
            total_counts[3].saturating_sub(dc[3]),
            total_counts[4].saturating_sub(dc[4]),
        ];

        // Binary features: user has rating of value 1..5 on other days
        f[F_USER_HAS_R1_OTHER_DAY] = if other_counts[0] > 0 { 1.0 } else { 0.0 };
        f[F_USER_HAS_R2_OTHER_DAY] = if other_counts[1] > 0 { 1.0 } else { 0.0 };
        f[F_USER_HAS_R3_OTHER_DAY] = if other_counts[2] > 0 { 1.0 } else { 0.0 };
        f[F_USER_HAS_R4_OTHER_DAY] = if other_counts[3] > 0 { 1.0 } else { 0.0 };
        f[F_USER_HAS_R5_OTHER_DAY] = if other_counts[4] > 0 { 1.0 } else { 0.0 };

        // Log(1+count) features for other days
        f[F_USER_LOG_R1_OTHER_DAY] = (1.0 + other_counts[0] as f32).ln();
        f[F_USER_LOG_R2_OTHER_DAY] = (1.0 + other_counts[1] as f32).ln();
        f[F_USER_LOG_R3_OTHER_DAY] = (1.0 + other_counts[2] as f32).ln();
        f[F_USER_LOG_R4_OTHER_DAY] = (1.0 + other_counts[3] as f32).ln();
        f[F_USER_LOG_R5_OTHER_DAY] = (1.0 + other_counts[4] as f32).ln();

        f
    }

    pub fn compute_all(&self, ds: &Dataset) -> Vec<Array1<f32>> {
        println!("Computing {} Phase F features for {} ratings...", N_FEATURES, ds.n_ratings);

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
            // Binary same-day
            "fF1_user_has_r1_same_day",
            "fF2_user_has_r2_same_day",
            "fF3_user_has_r3_same_day",
            "fF4_user_has_r4_same_day",
            "fF5_user_has_r5_same_day",
            // Binary other-day
            "fF6_user_has_r1_other_day",
            "fF7_user_has_r2_other_day",
            "fF8_user_has_r3_other_day",
            "fF9_user_has_r4_other_day",
            "fF10_user_has_r5_other_day",
            // Log same-day
            "fF11_user_log_r1_same_day",
            "fF12_user_log_r2_same_day",
            "fF13_user_log_r3_same_day",
            "fF14_user_log_r4_same_day",
            "fF15_user_log_r5_same_day",
            // Log other-day
            "fF16_user_log_r1_other_day",
            "fF17_user_log_r2_other_day",
            "fF18_user_log_r3_other_day",
            "fF19_user_log_r4_other_day",
            "fF20_user_log_r5_other_day",
        ]
    }

    pub fn print_summary(&self) {
        println!("\nFWLS Phase F Statistics ({} features):", N_FEATURES);
        println!("  Users: {}", self.base.n_users);
        println!("  Items: {}", self.base.n_items);

        // Sample rating distributions
        let active_users: Vec<usize> = (0..self.base.n_users)
            .filter(|&u| self.base.user_rating_counts[u] > 10)
            .take(3)
            .collect();
        if !active_users.is_empty() {
            println!("\n  Sample user rating distributions:");
            for &u in &active_users {
                let counts = &self.base.user_rating_counts_by_value[u];
                println!("    user {}: [1]={}, [2]={}, [3]={}, [4]={}, [5]={}",
                    u, counts[0], counts[1], counts[2], counts[3], counts[4]);
            }
        }

        println!("\nFeatures:");
        for (i, name) in Self::feature_names().iter().enumerate() {
            println!("  [{:2}] {}", i, name);
        }
    }
}

fn main() {
    println!("=== FWLS Meta-Features Generator (Phase F, {} features) ===\n", N_FEATURES);

    std::fs::create_dir_all("features").unwrap();

    // Process train -> probe
    println!("Processing: train -> probe");
    let train = Dataset::load("train", "rtg", "preds");
    let probe = Dataset::load("probe", "rtg", "preds");

    let fwls = FwlsFeaturesF::new(&train);
    fwls.print_summary();

    let probe_features = fwls.compute_all(&probe);
    save_features(&probe_features, "probe");

    // Process fulltrain -> qual
    println!("\nProcessing: fulltrain -> qual");
    let fulltrain = Dataset::load("fulltrain", "rtg", "preds");
    let qual = Dataset::load("qual", "rtg", "preds");

    let fwls_full = FwlsFeaturesF::new(&fulltrain);

    let qual_features = fwls_full.compute_all(&qual);
    save_features(&qual_features, "qual");

    println!("\n=== Done ===");
}
