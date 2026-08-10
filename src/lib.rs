use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressIterator};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::write_npy;
use rand::{prelude::SliceRandom, rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{fmt::Debug, fs::File, io::{BufWriter, Write}, sync::Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

static NO_PROGRESS: AtomicBool = AtomicBool::new(false);
pub fn suppress_progress() { NO_PROGRESS.store(true, Ordering::Relaxed); }

/// Total number of users in the Netflix Prize dataset.
pub const N_USERS: usize = 480_189;

// ---------------------------------------------------------------------------
// Numeric utilities
// ---------------------------------------------------------------------------

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sample (rows × cols) matrix from Normal(0, sigma)
#[inline]
pub fn rand_array2<R: Rng>(rows: usize, cols: usize, rng: &mut R, sigma: f32) -> Array2<f32> {
    let dist = Normal::<f32>::new(0.0, sigma).unwrap();
    Array2::from_shape_fn((rows, cols), |_| dist.sample(rng))
}

/// Sample (rows × cols × depth) tensor from Normal(0, sigma)
#[inline]
pub fn rand_array3<R: Rng>(rows: usize, cols: usize, depth: usize, rng: &mut R, sigma: f32) -> Array3<f32> {
    let dist = Normal::<f32>::new(0.0, sigma).unwrap();
    Array3::from_shape_fn((rows, cols, depth), |_| dist.sample(rng))
}

/// Sample (rows × cols) matrix from Uniform(-sigma, sigma)
#[inline]
pub fn rand_array2u<R: Rng>(rows: usize, cols: usize, rng: &mut R, sigma: f32,) -> Array2<f32> {
    if sigma == 0.0 {
        Array2::zeros((rows, cols))
    }
    else {
        let dist = Uniform::<f32>::new(-sigma, sigma).unwrap();
        Array2::from_shape_fn((rows, cols), |_| dist.sample(rng))
    }
}

// ---------------------------------------------------------------------------
// Logging and progress macros
// ---------------------------------------------------------------------------

/// Global log file for tee output (stdout + file)
pub static LOG_FILE: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Print to both stdout and the log file (if open)
#[macro_export]
macro_rules! tee {
    ($($arg:tt)*) => {{
        print!($($arg)*);
        if let Some(f) = $crate::LOG_FILE.lock().unwrap().as_mut() {
            use ::std::io::Write as _;
            let _ = write!(f, $($arg)*);
            let _ = f.flush();
        }
    }};
}

/// Print line to both stdout and the log file (if open)
#[macro_export]
macro_rules! teeln {
    () => {{
        println!();
        {
            use ::std::io::Write as _;
            let _ = ::std::io::stdout().flush();
        }
        if let Some(f) = $crate::LOG_FILE.lock().unwrap().as_mut() {
            use ::std::io::Write as _;
            let _ = writeln!(f);
            let _ = f.flush();
        }
    }};
    ($($arg:tt)*) => {{
        println!($($arg)*);
        {
            use ::std::io::Write as _;
            let _ = ::std::io::stdout().flush();
        }
        if let Some(f) = $crate::LOG_FILE.lock().unwrap().as_mut() {
            use ::std::io::Write as _;
            let _ = writeln!(f, $($arg)*);
            let _ = f.flush();
        }
    }};
}

#[macro_export]
macro_rules! progress {
    ($e:expr) => {{
        let __iter = $e;
        if $crate::NO_PROGRESS.load(::std::sync::atomic::Ordering::Relaxed) {
            __iter.progress_with(indicatif::ProgressBar::hidden())
        } else {
            __iter.progress()
        }
    }};
}
#[macro_export]
macro_rules! progress_count {
    ($e:expr, $n:expr) => {
        $e.progress_with(if $crate::NO_PROGRESS.load(::std::sync::atomic::Ordering::Relaxed) {
            indicatif::ProgressBar::hidden()
        } else {
            indicatif::ProgressBar::new($n)
        })
    };
}

/// Create a progress bar that respects the global `NO_PROGRESS` flag
/// (set by tests via `suppress_progress()`). Returns a hidden bar
/// when `NO_PROGRESS` is set; the caller may apply custom styling and
/// a prefix on the returned `ProgressBar`.
pub fn make_pb(total: u64) -> ProgressBar {
    if NO_PROGRESS.load(Ordering::Relaxed) {
        ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::hidden())
    } else {
        ProgressBar::new(total)
    }
}

/// Load numpy array from data/{prefix}/{fname}
macro_rules! read_npy {
    ($prefix:expr, $fname:expr) => {{
        let path = format!("data/{}/{}", $prefix, $fname);
        ::ndarray_npy::read_npy(&path).expect(&path)
    }};
}

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod aex;
pub mod als8;
pub mod asym;
pub mod bk1;
pub mod bk3;
pub mod bknbrx;
pub mod blend;
pub mod cfnade;
pub mod knn;
pub mod knn3;
pub mod knnf;
pub mod knns;
pub mod mf;
pub mod mfrbmx;
pub mod mlp;
pub mod nbstats;
pub mod nlpp;
pub mod rbmx2;
pub mod rx;
pub mod tsvdx4;
pub mod tsvdx4p;
pub mod tsvdx5;
pub mod tsvdx6;
pub mod tx;
pub mod vfeat1;

// ---------------------------------------------------------------------------
// Dataset and target spec
// ---------------------------------------------------------------------------

/// Parse "weight*model" or "model" target spec term into (weight, model_name)
fn parse_target_term(spec: &str) -> (f32, &str) {
    match spec.split_once('*') {
        Some((w_str, name)) => {
            let weight: f32 = w_str.trim().parse().unwrap();
            (weight, name.trim())
        }
        None => (1.0, spec.trim()),
    }
}

/// Rating data and dataset statistics
pub struct Dataset {
    pub user_idxs: Array1<i32>,  // User indices per rating
    pub user_cnts: Array1<i32>,  // Number of ratings per user
    pub item_idxs: Array1<i32>,  // Item indices per rating (i32 to support transposed datasets where user indices become item indices)
    pub item_cnts: Array1<i32>,  // Number of ratings per item
    pub item_years: Array1<i32>, // Item production years
    pub raw_ratings: Array1<i8>, // Raw rating values
    pub residuals: Array1<f32>,  // Prediction target values
    pub dates: Array1<i16>,      // Rating dates
    pub is_test: Array1<i8>,     // Does the rating belong to test set?

    pub name: String,            // Name of the data set
    pub n_users: usize,          // Number of users
    pub n_items: usize,          // Number of items
    pub n_ratings: usize,        // Number of ratings
    pub transposed: bool,
}

impl Dataset {
    /// Load dataset from data/{name}/ directory, computing residuals for target_spec
    /// - target_spec = "rtg": residuals = ratings
    /// - target_spec = "weight*model": residuals = ratings - weight * model_predictions
    /// - target_spec = "0.5*model1 + 0.5*model2": residuals = ratings - linear combination
    /// - preds_dir: directory to load model predictions from (e.g. "preds" or "preds2")
    #[inline]
    pub fn load(name: &str, target_spec: &str, preds_dir: &str) -> Self {
        let user_idxs: Array1<i32> = read_npy!(name, "user_idxs.npy");
        let user_cnts: Array1<i32> = read_npy!(name, "user_cnts.npy");
        let item_idxs_i16: Array1<i16> = read_npy!(name, "item_idxs.npy");
        let item_idxs = item_idxs_i16.mapv(|x| x as i32);
        let item_cnts: Array1<i32> = read_npy!(name, "item_cnts.npy");
        let item_years: Array1<i32> = read_npy!(name, "item_years.npy");

        let raw_ratings: Array1<i8> = read_npy!(name, "ratings.npy");
        let ratings = raw_ratings.mapv(|x| x as f32);
        let residuals = if target_spec == "rtg" {
            ratings
        } else {
            let terms: Vec<&str> = target_spec.split(" + ").collect();
            let mut combined = ratings;
            for term in &terms {
                let (weight, model) = parse_target_term(term);
                let path = format!("{}/{}.{}.npy", preds_dir, model, name);
                let preds: Array1<f32> = ::ndarray_npy::read_npy(&path).expect(&path);
                combined = combined - &preds * weight;
            }
            combined
        };

        let dates: Array1<i16> = read_npy!(name, "dates.npy");
        let is_test: Array1<i8> = read_npy!(name, "is_test.npy");

        let name = name.to_owned();
        let n_users = N_USERS;
        let n_items = 17_770;
        let n_ratings = raw_ratings.len();
        let transposed = false;

        Self {
            user_idxs, user_cnts, item_idxs, item_cnts, item_years,
            raw_ratings, residuals, dates, is_test,
            name, n_users, n_items, n_ratings, transposed,
        }
    }

    /// Transpose dataset: swap user ↔ item roles.
    /// If reorder=true, sort ratings by (new_user, date) so calc_user_offsets works.
    /// (Use reorder=true for training sets, false for probe/qual where original order is needed.)
    pub fn transpose(&self, reorder: bool) -> Self {
        let n = self.n_ratings;

        if reorder {
            // Pack all per-rating fields into a struct, radix sort it, then unpack.
            // All memory access is sequential — no random gather passes.
            #[derive(Clone, Copy)]
            #[repr(C)]
            struct Row {
                key: u32,       // composite sort key: item_idx * n_days + date
                user_idx: i32,  // new user = old item (already encoded in key, but we need old item idx)
                item_idx: i32,  // new item = old user
                residual: f32,
                date: i16,
                raw_rating: i8,
                is_test: i8,
            } // 20 bytes, packed

            let n_days = (*self.dates.iter().max().unwrap() as u32) + 1;

            // Pack (sequential read from 6 arrays, sequential write to 1)
            let mut src: Vec<Row> = (0..n).map(|i| Row {
                key: self.item_idxs[i] as u32 * n_days + self.dates[i] as u32,
                user_idx: self.item_idxs[i],  // swapped
                item_idx: self.user_idxs[i],  // swapped
                residual: self.residuals[i],
                date: self.dates[i],
                raw_rating: self.raw_ratings[i],
                is_test: self.is_test[i],
            }).collect();
            let mut dst: Vec<Row> = vec![Row {
                key: 0, user_idx: 0, item_idx: 0, residual: 0.0,
                date: 0, raw_rating: 0, is_test: 0,
            }; n];

            // Radix sort by key (2-pass, 16-bit radix) — moves entire Row structs
            let mut counts = [0u32; 65536];
            for r in &src { counts[(r.key & 0xFFFF) as usize] += 1; }
            let mut offsets = [0u32; 65536];
            for i in 1..65536 { offsets[i] = offsets[i - 1] + counts[i - 1]; }
            for r in &src {
                let bucket = (r.key & 0xFFFF) as usize;
                dst[offsets[bucket] as usize] = *r;
                offsets[bucket] += 1;
            }

            counts.fill(0);
            for r in &dst { counts[(r.key >> 16) as usize] += 1; }
            offsets.fill(0);
            for i in 1..65536 { offsets[i] = offsets[i - 1] + counts[i - 1]; }
            for r in &dst {
                let bucket = (r.key >> 16) as usize;
                src[offsets[bucket] as usize] = *r;
                offsets[bucket] += 1;
            }

            // Unpack (sequential read from 1, sequential write to 6)
            let mut user_idxs = Array1::<i32>::uninit(n);
            let mut item_idxs = Array1::<i32>::uninit(n);
            let mut raw_ratings = Array1::<i8>::uninit(n);
            let mut residuals = Array1::<f32>::uninit(n);
            let mut dates = Array1::<i16>::uninit(n);
            let mut is_test = Array1::<i8>::uninit(n);
            for (i, r) in src.iter().enumerate() {
                user_idxs[i] = std::mem::MaybeUninit::new(r.user_idx);
                item_idxs[i] = std::mem::MaybeUninit::new(r.item_idx);
                raw_ratings[i] = std::mem::MaybeUninit::new(r.raw_rating);
                residuals[i] = std::mem::MaybeUninit::new(r.residual);
                dates[i] = std::mem::MaybeUninit::new(r.date);
                is_test[i] = std::mem::MaybeUninit::new(r.is_test);
            }
            // SAFETY: all elements initialized in the loop above
            let (user_idxs, item_idxs, raw_ratings, residuals, dates, is_test) = unsafe {(
                user_idxs.assume_init(), item_idxs.assume_init(),
                raw_ratings.assume_init(), residuals.assume_init(),
                dates.assume_init(), is_test.assume_init(),
            )};

            Self {
                user_idxs, item_idxs, raw_ratings, residuals, dates, is_test,
                user_cnts:   self.item_cnts.clone(),
                item_cnts:   self.user_cnts.clone(),
                item_years:  Array1::zeros(self.n_users), // no meaningful years for transposed items (= original users)
                name:        self.name.clone(),
                n_users:     self.n_items,
                n_items:     self.n_users,
                n_ratings:   n,
                transposed:  !self.transposed,
            }
        } else {
            Self {
                user_idxs:   self.item_idxs.clone(),
                item_idxs:   self.user_idxs.clone(),
                raw_ratings: self.raw_ratings.clone(),
                residuals:   self.residuals.clone(),
                dates:       self.dates.clone(),
                is_test:     self.is_test.clone(),
                user_cnts:   self.item_cnts.clone(),
                item_cnts:   self.user_cnts.clone(),
                item_years:  Array1::zeros(self.n_users),
                name:        self.name.clone(),
                n_users:     self.n_items,
                n_items:     self.n_users,
                n_ratings:   n,
                transposed:  !self.transposed,
            }
        }
    }

    /// True if the dataset contains any test examples (`is_test != 0`).
    /// Used to detect held-out sets (e.g. qual) where per-epoch RMSE
    /// should be suppressed.
    #[inline]
    pub fn contains_test_examples(&self) -> bool {
        self.is_test.iter().any(|&x| x != 0)
    }
}

/// `Dataset` view with the prediction targets masked out, for use during training.
///
/// Exposes only the fields a model is allowed to read from a held-out (probe
/// or qual) set during `fit_epoch`: indices, dates, counts, etc. The
/// `raw_ratings` and `residuals` fields are deliberately omitted, so the
/// compiler rejects any accidental peek at held-out targets while training.
/// Evaluation code (`calc_rmse`, `gen_preds*`) keeps using `&Dataset` and
/// can still see all fields.
pub struct MaskedDataset<'a> {
    pub user_idxs:  &'a Array1<i32>,
    pub user_cnts:  &'a Array1<i32>,
    pub item_idxs:  &'a Array1<i32>,
    pub item_cnts:  &'a Array1<i32>,
    pub item_years: &'a Array1<i32>,
    pub dates:      &'a Array1<i16>,
    pub is_test:    &'a Array1<i8>,
    pub name:       &'a str,
    pub n_users:    usize,
    pub n_items:    usize,
    pub n_ratings:  usize,
    pub transposed: bool,
    // raw_ratings, residuals: intentionally omitted
}

impl<'a> MaskedDataset<'a> {
    #[inline]
    pub fn from(ds: &'a Dataset) -> Self {
        Self {
            user_idxs:  &ds.user_idxs,
            user_cnts:  &ds.user_cnts,
            item_idxs:  &ds.item_idxs,
            item_cnts:  &ds.item_cnts,
            item_years: &ds.item_years,
            dates:      &ds.dates,
            is_test:    &ds.is_test,
            name:       &ds.name,
            n_users:    ds.n_users,
            n_items:    ds.n_items,
            n_ratings:  ds.n_ratings,
            transposed: ds.transposed,
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset helpers
// ---------------------------------------------------------------------------

/// Compute global mean of residuals (global bias term)
#[inline]
pub fn calc_gbias(tr: &Dataset) -> f32 {
    let rsum = tr.residuals.iter().map(|&x| x as f64).sum::<f64>();
    (rsum / tr.n_ratings as f64) as f32
}

/// Compute offsets for each user's ratings (cumulative sum of user_cnts)
pub fn calc_user_offsets(tr: &Dataset) -> Array1<usize> {
    for t in 1..tr.n_ratings {
        assert!(
            tr.user_idxs[t] >= tr.user_idxs[t - 1],
            "dataset '{}' is not user-sorted at index {t}: user {} follows user {}",
            tr.name, tr.user_idxs[t], tr.user_idxs[t - 1]
        );
    }
    let mut user_offsets = Array1::zeros(tr.n_users + 1);
    for u in 0..tr.n_users {
        let cnt = tr.user_cnts[u] as usize;
        user_offsets[u + 1] = user_offsets[u] + cnt
    }
    user_offsets
}

/// Generate user indices 0..n, optionally shuffled with epoch-dependent seed
pub fn get_users(n: usize, shuffle: bool, seed: u64, epoch: usize) -> Array1<usize> {
    let mut users: Vec<usize> = (0..n).collect();
    if shuffle {
        let seed2 = seed ^ (epoch as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let mut rng = StdRng::seed_from_u64(seed2);
        users.shuffle(&mut rng);
    }
    Array1::from(users)
}

// ---------------------------------------------------------------------------
// Ordinal regression head
// ---------------------------------------------------------------------------

/// Ordinal regression head configuration
#[derive(Clone, Copy, Debug)]
pub struct OrdinalHeadConfig {
    pub th_init: [f32; 4], // Initial thresholds for converting score to probabilities
    pub th_gap: f32,       // Minimum gap between consecutive thresholds
    pub lr_t: f32,         // Learning rate for threshold updates
    pub reg_t: f32,        // Regularization strength for thresholds
}

/// Ordinal regression head for converting continuous scores to probability distributions over rating categories
#[derive(Clone, Copy, Debug)]
pub struct OrdinalHead {
    cfg: OrdinalHeadConfig,   // Ordinal head configuration
    pub thresholds: [f32; 4], // Thresholds for converting score to probabilities
}

impl OrdinalHead {
    // Enforce ordered thresholds with minimum gap constraint
    #[inline]
    pub fn enforce_sorted_with_gap(&mut self) {
        self.thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 1..4 {
            let min_ok = self.thresholds[k - 1] + self.cfg.th_gap;
            if self.thresholds[k] < min_ok { self.thresholds[k] = min_ok; }
        }
    }

    // Initialize ordinal head with config
    #[inline]
    pub fn new(cfg: OrdinalHeadConfig) -> Self {
        Self { cfg, thresholds: cfg.th_init }
    }

    // Predict probability distribution over ratings [1, 2, 3, 4, 5] from score
    // Uses ordinal regression: P(Y <= k) = sigmoid(threshold[k] - score)
    #[inline]
    pub fn predict_probs(&self, score: f32) -> [f32; 5] {
        // Step 1: Compute cumulative probabilities F[k] = P(Y <= k)
        // Ordinal regression model: P(Y <= k) = sigmoid(c_k - score)
        // where c_k is the k-th threshold
        // - Lower score -> higher sigmoid -> higher P(Y <= k)
        // - Higher score -> lower sigmoid -> lower P(Y <= k)
        let f: [f32; 6] = std::array::from_fn(|k| match k {
            0 => 0.0, // P(Y <= 0) = 0 (impossible)
            5 => 1.0, // P(Y <= 5) = 1 (all ratings)
            _ => sigmoid(self.thresholds[k - 1] - score), // P(Y <= k)
        });

        // Step 2: Convert cumulative to individual probabilities
        // P(Y = k) = P(Y <= k) - P(Y <= k-1) = F[k] - F[k-1]
        // This gives the probability mass for each rating category
        let mut p = [0.0; 5];
        for k in 0..5 { p[k] = f[k + 1] - f[k]; }

        // Step 3: Normalize probabilities to sum to 1
        // (ensures numerical stability in case of floating point errors)
        // let sum: f32 = p.iter().sum();
        // p.map(|x| x / sum)
        p
    }

    // Compute gradient of score and thresholds for negative log-likelihood
    // Loss = -log(P(Y = y)) where y is the true rating
    // P(Y = y) = F[y] - F[y-1] = sigmoid(c_y - score) - sigmoid(c_{y-1} - score)
    #[inline]
    pub fn grad(&self, score: f32, y: usize) -> (f32, [f32; 4]) {
        // Step 1: Compute cumulative probabilities F[k] and sigmoid derivatives a[k]
        // a[k] = dF[k]/d(threshold or score) = F[k] * (1 - F[k]) (sigmoid derivative)
        let f: [f32; 6] = std::array::from_fn(|k| match k {
            0 => 0.0,
            5 => 1.0,
            _ => sigmoid(self.thresholds[k - 1] - score),
        });

        // Sigmoid derivatives: a[k] = sigmoid'(c_k - score) = F[k] * (1 - F[k])
        let a: [f32; 6] = std::array::from_fn(|k| f[k] * (1.0 - f[k]));

        // Probability of true rating: P(Y = y) = F[y] - F[y-1]
        // Clamp to 1e-6 to avoid division by zero
        let p_y = (f[y] - f[y - 1]).max(1e-6);

        // Step 2: Gradient w.r.t. score (g_ds)
        // d(-log P(Y=y))/d(score) = d(-log(F[y] - F[y-1]))/d(score)
        // Using chain rule: = -(1/P(Y=y)) * d(F[y] - F[y-1])/d(score)
        // d(F[y] - F[y-1])/d(score) = -a[y] + a[y-1] (sigmoid' flips sign for -score term)
        // Therefore: g_ds = (a[y] - a[y-1]) / P(Y=y)
        let g_ds = (a[y] - a[y - 1]) / p_y;

        // Step 3: Gradient w.r.t. thresholds (g_t[k] for k in 0..4)
        // d(-log P(Y=y))/d(c_k) = -(1/P(Y=y)) * d(F[y] - F[y-1])/d(c_k)
        // F[y] depends on c_y: dF[y]/dc_y = a[y] (only when k = y-1, i.e., c_k is threshold y-1)
        // F[y-1] depends on c_{y-1}: dF[y-1]/dc_{y-1} = a[y-1] (when k = y-2)
        // So: g_t[y-2] affects F[y-1] (lower boundary): +a[y-1] / P(Y=y)
        //     g_t[y-1] affects F[y] (upper boundary): -a[y] / P(Y=y)
        let mut g_t = [0.0; 4];
        if y >= 2 { g_t[y - 2] += a[y - 1] / p_y; } // Lower threshold (increases P(Y=y))
        if y <= 4 { g_t[y - 1] -= a[y] / p_y; }     // Upper threshold (decreases P(Y=y))

        (g_ds, g_t)
    }

    // Update thresholds with gradient and regularization
    #[inline]
    pub fn update_thresholds(&mut self, g_t: [f32; 4]) {
        for k in 0..4 {
            let tk = self.thresholds[k];
            self.thresholds[k] -= self.cfg.lr_t * (g_t[k] + self.cfg.reg_t * tk);
        }
    }
}

// ---------------------------------------------------------------------------
// Regressor trait
// ---------------------------------------------------------------------------

/// Common interface for collaborative filtering models
pub trait Regressor {
    type Config: Copy + Debug;

    /// Initialize model with training/probe datasets and configuration.
    /// `pr` is a restricted view that hides the probe set's `raw_ratings`
    /// and `residuals` so they cannot be peeked at during training.
    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self;

    /// Return total number of training epochs
    fn n_epochs(&self) -> usize { 0 }

    /// Train one epoch (epoch is 1-based: 1..=n_epochs()).
    /// `pr` is a restricted view that hides the probe set's `raw_ratings`
    /// and `residuals` so they cannot be peeked at during training.
    fn fit_epoch(&mut self, tr: &Dataset, pr: &MaskedDataset, epoch: usize);

    /// Return number of sub-predictions this model generates
    fn n_subscores(&self) -> usize { 0 }

    /// Return names/suffixes for each sub-score (e.g. ["bias", "mf", "nsvd1"])
    /// Default returns ["1", "2", ...] for backward compatibility.
    fn subscore_names(&self) -> Vec<String> {
        (1..=self.n_subscores()).map(|j| j.to_string()).collect()
    }

    /// Predict rating residual for user u, item i, on given day
    fn predict(&self, u: usize, i: usize, day: i32) -> f32;

    /// Predict individual sub-scores (for models with multiple components)
    fn predict_subscores(&self, _u: usize, _i: usize, _day: i32) -> Array1::<f32> {
        Array1::zeros(0)
    }

    /// Return number of factor-scores this model generates (typically ~n_feat)
    fn n_factorscores(&self) -> usize { 0 }

    /// Return names/suffixes for each factor-score.
    /// Default returns ["1", "2", ...] for backward compatibility.
    fn factorscore_names(&self) -> Vec<String> {
        (1..=self.n_factorscores()).map(|j| j.to_string()).collect()
    }

    /// Predict per-factor scores (finer-grained than subscores; length ~= n_feat)
    fn predict_factorscores(&self, _u: usize, _i: usize, _day: i32) -> Array1::<f32> {
        Array1::zeros(0)
    }

    /// Save model-specific artifacts after training (e.g. item factors to preds_dir)
    fn save_artifacts(&self, _model_name: &str, _tr_set: &str, _preds_dir: &str) {}
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

/// Compute RMSE on non-test rows (is_test == 0)
#[inline]
pub fn calc_rmse<M: Regressor>(model: &mut M, ds: &Dataset) -> f64 {
    let mut cnt = 0.0;
    let mut sse = 0.0;
    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 { continue; }
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let r = ds.residuals[idx];
        let day = ds.dates[idx] as i32;
        let err = (model.predict(u, i, day) - r) as f64;
        cnt += 1.0;
        sse += err * err;
    }
    (sse / cnt).sqrt()
}

/// Print epoch number and RMSE. On datasets that contain test examples
/// (e.g. qual), the RMSE is reported only on the final epoch.
#[inline]
pub fn report_rmse<M: Regressor>(model: &mut M, epoch: usize, n_epochs: usize, ds: &Dataset) {
    tee!("Epoch {:02}", epoch);
    if ds.contains_test_examples() && epoch < n_epochs {
        teeln!();
        return;
    }
    let rmse = calc_rmse(model, ds);
    teeln!(" | RMSE {:.5}", rmse);
}

/// Generate predictions for all ratings (sequential) and return as Array1.
#[inline]
pub fn gen_preds<M: Regressor>(model: &mut M, ds: &Dataset) -> Array1<f32> {
    let mut preds = Array1::<f32>::zeros(ds.n_ratings);
    for idx in progress!(0..ds.n_ratings) {
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let day = ds.dates[idx] as i32;
        let offset = ds.raw_ratings[idx] as f32 - ds.residuals[idx];
        preds[idx] = model.predict(u, i, day) + offset;
    }
    preds
}

/// Generate predictions for all ratings (sequential) and save to .npy file.
#[inline]
pub fn save_preds<M: Regressor>(model: &mut M, ds: &Dataset, path: &str) {
    let preds = gen_preds(model, ds);
    write_npy(path, &preds).unwrap();
}

/// Generate predictions in parallel and return as Array1
pub fn gen_preds_parallel<M: Regressor + Sync>(model: &M, ds: &Dataset) -> Array1<f32> {
    let mut preds = Array1::<f32>::zeros(ds.n_ratings);
    let preds_slice = preds.as_slice_mut().unwrap();
    progress_count!(preds_slice.par_iter_mut().enumerate(), ds.n_ratings as u64)
        .for_each(|(idx, out)| {
            let u = ds.user_idxs[idx] as usize;
            let i = ds.item_idxs[idx] as usize;
            let day = ds.dates[idx] as i32;
            let offset = ds.raw_ratings[idx] as f32 - ds.residuals[idx];
            *out = model.predict(u, i, day) + offset;
        });
    preds
}

/// Compute RMSE from precomputed predictions (non-test rows only)
pub fn rmse_from_preds(preds: &Array1<f32>, ds: &Dataset) -> f64 {
    let mut sse = 0.0_f64;
    let mut cnt = 0.0_f64;
    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 { continue; }
        let err = (preds[idx] - ds.raw_ratings[idx] as f32) as f64;
        sse += err * err;
        cnt += 1.0;
    }
    (sse / cnt).sqrt()
}

/// Generate predictions in parallel and save to .npy file
pub fn save_preds_parallel<M: Regressor + Sync>(model: &M, ds: &Dataset, path: &str) {
    let preds = gen_preds_parallel(model, ds);
    write_npy(path, &preds).unwrap();
}

/// Compute all sub-scores for each rating (for models with multiple components)
pub fn calc_subscores<M: Regressor>(model: &M, ds: &Dataset) -> Array2::<f32> {
    let d = model.n_subscores();
    let mut subscores = Array2::<f32>::zeros((ds.n_ratings, d));
    for idx in 0..ds.n_ratings {
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let day = ds.dates[idx] as i32;

        let yhat = model.predict_subscores(u, i, day);
        subscores.row_mut(idx).assign(&yhat);
    }
    subscores
}

/// Compute all factor-scores for each rating (finer-grained than subscores)
pub fn calc_factorscores<M: Regressor>(model: &M, ds: &Dataset) -> Array2::<f32> {
    let d = model.n_factorscores();
    let mut factorscores = Array2::<f32>::zeros((ds.n_ratings, d));
    for idx in 0..ds.n_ratings {
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let day = ds.dates[idx] as i32;

        let yhat = model.predict_factorscores(u, i, day);
        factorscores.row_mut(idx).assign(&yhat);
    }
    factorscores
}

/// Save each factorscore column for `ds` to its own `{model_name}-{name}.{set}.npy`
/// (same convention as subscores). No-op if the model emits none.
fn save_factorscores_npy<M: Regressor>(
    model: &M, ds: &Dataset, preds_dir: &str, model_name: &str, set: &str,
) {
    if model.n_factorscores() == 0 { return; }
    let factorscores = calc_factorscores(model, ds);
    let names = model.factorscore_names();
    for (j, name) in names.iter().enumerate() {
        let path = format!("{preds_dir}/{model_name}-{name}.{set}.npy");
        write_npy(path, &factorscores.column(j)).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Low-level training entry point
// ---------------------------------------------------------------------------

/// Train model on tr_set and generate predictions for pr_set
#[inline]
pub fn fit<M: Regressor + Sync>(
    cfg: M::Config,
    target: &str,
    tr_set: &str,
    pr_set: &str,
    model_name: &str,
    save_subscores: bool,
    save_factorscores: bool,
    save_train: bool,
    save_probe_each_epoch: bool,
    preds_dir: &str,
    transpose: bool,
) {
    teeln!("{} => {}{}", tr_set, pr_set, if transpose { " (transposed)" } else { "" });
    let tr = Dataset::load(tr_set, target, preds_dir);
    let pr = Dataset::load(pr_set, target, preds_dir);
    let (tr, pr) = if transpose { (tr.transpose(true), pr.transpose(false)) } else { (tr, pr) };

    let pr_masked = MaskedDataset::from(&pr);
    let mut model = M::new(&tr, &pr_masked, cfg);
    let n_epochs = model.n_epochs();

    for epoch in 1..=n_epochs {
        model.fit_epoch(&tr, &pr_masked, epoch);
        report_rmse(&mut model, epoch, n_epochs, &pr);
        if save_probe_each_epoch {
            let ep_path = format!("{}/{}_ep{:02}.{}.npy", preds_dir, model_name, epoch, pr_set);
            let ep_preds = gen_preds_parallel(&model, &pr);
            write_npy(&ep_path, &ep_preds).unwrap();
        }
    }

    let pr_preds = gen_preds_parallel(&model, &pr);
    let path = format!("{}/{}.{}.npy", preds_dir, model_name, pr_set);
    write_npy(&path, &pr_preds).unwrap();

    if save_subscores {
        let subscores = calc_subscores(&model, &pr);
        let names = model.subscore_names();
        for (j, name) in names.iter().enumerate() {
            let path = format!("{}/{model_name}-{name}.{pr_set}.npy", preds_dir);
            let col = subscores.column(j);
            write_npy(path, &col).unwrap();
        }
    }

    if save_factorscores {
        save_factorscores_npy(&model, &pr, preds_dir, model_name, pr_set);
    }

    if save_train {
        let path = format!("{}/{}.{}.npy", preds_dir, model_name, tr_set);
        save_preds_parallel(&model, &tr, &path);
    }

    model.save_artifacts(model_name, tr_set, preds_dir);
}

// ---------------------------------------------------------------------------
// Split convention
// ---------------------------------------------------------------------------

/// Train/probe/qual split convention. Constants:
/// - `SPLIT_OLD` = train → probe, fulltrain → qual, preds_dir = "preds"
/// - `SPLIT_NEW` = trainx → probex, fulltrain → qual, preds_dir = "preds_new"
#[derive(Debug, Clone, Copy)]
pub struct Split {
    pub tr: &'static str,
    pub pr: &'static str,
    pub fulltrain_tr: &'static str,
    pub fulltrain_pr: &'static str,
    pub preds_dir: &'static str,
    pub sim_dir: &'static str,
}

pub const SPLIT_OLD: Split = Split {
    tr: "train", pr: "probe",
    fulltrain_tr: "fulltrain", fulltrain_pr: "qual",
    preds_dir: "preds",
    sim_dir: "sim",
};

pub const SPLIT_NEW: Split = Split {
    tr: "trainx", pr: "probex",
    fulltrain_tr: "fulltrain", fulltrain_pr: "qual",
    preds_dir: "preds_new",
    sim_dir: "sim",
};

// ---------------------------------------------------------------------------
// fit2 family
// ---------------------------------------------------------------------------

/// Optional flags for `fit2_inner` / `fit2!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fit2Opts {
    pub save_train: bool,
    pub save_probe_each_epoch: bool,
    pub save_subscores: bool,
    pub save_factorscores: bool,
    /// Skip the fulltrain → fulltrain_pr phase; instead, predict fulltrain_pr
    /// using the phase-1 model.
    pub no_fulltrain: bool,
    pub transpose: bool,
}

/// Call `fit2_inner` with named optional parameters.
/// Usage: `fit2!(Model, cfg, "rtg", "name", SPLIT_OLD, save_subscores: true)`
#[macro_export]
macro_rules! fit2 {
    ($M:ty, $cfg:expr, $target:expr, $model_name:expr, $split:expr $(, $key:ident : $val:expr)* $(,)?) => {
        $crate::fit2_inner::<$M>(
            $cfg, $target, $model_name, $split,
            $crate::Fit2Opts { $($key: $val,)* ..Default::default() },
        )
    };
}

/// Run standard experiment: split.tr → split.pr (with optional epoch saves,
/// subscores, train preds), save config, then split.fulltrain_tr → split.fulltrain_pr
/// (unless `opts.no_fulltrain`, in which case the phase-1 model is used to
/// predict `split.fulltrain_pr` directly).
#[inline]
pub fn fit2_inner<M: Regressor + Sync>(
    cfg: M::Config,
    target: &str,
    model_name: &str,
    split: Split,
    opts: Fit2Opts,
) {
    let Fit2Opts { save_train, save_probe_each_epoch, save_subscores, save_factorscores, no_fulltrain, transpose } = opts;
    let preds_dir = split.preds_dir;

    std::fs::create_dir_all(preds_dir).unwrap();

    // Open log file (only if not already open, e.g. from fit3)
    let owns_log = LOG_FILE.lock().unwrap().is_none();
    if owns_log {
        *LOG_FILE.lock().unwrap() = Some(BufWriter::new(
            File::create(format!("{}/{}.out", preds_dir, model_name)).unwrap()
        ));
    }

    teeln!("[{}]", model_name);
    teeln!("target = {:?}", target);
    teeln!("{:?}", cfg);

    // Phase 1: split.tr → split.pr  (scoped to free model + datasets before phase 2)
    {
        teeln!("{} => {}{}", split.tr, split.pr, if transpose { " (transposed)" } else { "" });
        let tr = Dataset::load(split.tr, target, preds_dir);
        let pr = Dataset::load(split.pr, target, preds_dir);
        let qual_ds = if no_fulltrain {
            Some(Dataset::load(split.fulltrain_pr, target, preds_dir))
        } else { None };
        let (tr, pr, qual_ds) = if transpose {
            (tr.transpose(true), pr.transpose(false), qual_ds.map(|q| q.transpose(false)))
        } else { (tr, pr, qual_ds) };

        let pr_masked = MaskedDataset::from(&pr);
        let mut model = M::new(&tr, &pr_masked, cfg);
        let n_epochs = model.n_epochs();

        for epoch in 1..=n_epochs {
            model.fit_epoch(&tr, &pr_masked, epoch);
            report_rmse(&mut model, epoch, n_epochs, &pr);
            if save_probe_each_epoch {
                let ep_path = format!("{}/{}_ep{:02}.{}.npy", preds_dir, model_name, epoch, split.pr);
                let ep_preds = gen_preds_parallel(&model, &pr);
                write_npy(&ep_path, &ep_preds).unwrap();
                if no_fulltrain {
                    let ep_qual_path = format!("{}/{}_ep{:02}.{}.npy", preds_dir, model_name, epoch, split.fulltrain_pr);
                    let ep_qual_preds = gen_preds_parallel(&model, qual_ds.as_ref().unwrap());
                    write_npy(&ep_qual_path, &ep_qual_preds).unwrap();
                }
            }
        }

        let pr_preds = gen_preds_parallel(&model, &pr);
        let rmse = rmse_from_preds(&pr_preds, &pr);
        teeln!("{} RMSE {:.5}", split.pr, rmse);
        let path = format!("{}/{}.{}.npy", preds_dir, model_name, split.pr);
        write_npy(&path, &pr_preds).unwrap();

        if save_subscores && model.n_subscores() > 0 {
            let subscores = calc_subscores(&model, &pr);
            let names = model.subscore_names();
            for (j, name) in names.iter().enumerate() {
                let path = format!("{}/{model_name}-{name}.{}.npy", preds_dir, split.pr);
                let col = subscores.column(j);
                write_npy(&path, &col).unwrap();
            }
        }

        if save_factorscores {
            save_factorscores_npy(&model, &pr, preds_dir, model_name, split.pr);
        }

        if save_train {
            let path = format!("{}/{}.{}.npy", preds_dir, model_name, split.tr);
            save_preds_parallel(&model, &tr, &path);
        }

        model.save_artifacts(model_name, split.tr, preds_dir);

        // Phase 1 → fulltrain_pr (when skipping the fulltrain phase)
        if no_fulltrain {
            teeln!("{} => {} (no_fulltrain)", split.tr, split.fulltrain_pr);
            let qual = qual_ds.as_ref().unwrap();
            let qual_path = format!("{}/{}.{}.npy", preds_dir, model_name, split.fulltrain_pr);
            save_preds_parallel(&model, qual, &qual_path);
            if save_subscores && model.n_subscores() > 0 {
                let subscores = calc_subscores(&model, qual);
                let names = model.subscore_names();
                for (j, name) in names.iter().enumerate() {
                    let path = format!("{}/{model_name}-{name}.{}.npy", preds_dir, split.fulltrain_pr);
                    let col = subscores.column(j);
                    write_npy(&path, &col).unwrap();
                }
            }
            if save_factorscores {
                save_factorscores_npy(&model, qual, preds_dir, model_name, split.fulltrain_pr);
            }
        }
    } // model, tr, pr freed here

    // Save config — disabled; details live in the per-job src/bin/<name>-<split>.rs binary.
    // let mut f = File::create(format!("{}/{}.cfg", preds_dir, model_name)).unwrap();
    // writeln!(f, "target = {:#?}", target).unwrap();
    // writeln!(f, "cfg = {:#?}", cfg).unwrap();
    // drop(f);

    // Phase 2: split.fulltrain_tr → split.fulltrain_pr
    if !no_fulltrain {
        fit::<M>(cfg, target, split.fulltrain_tr, split.fulltrain_pr, model_name,
                 save_subscores, save_factorscores, save_train, save_probe_each_epoch, preds_dir, transpose);
    }

    teeln!();

    if owns_log {
        if let Some(mut lf) = LOG_FILE.lock().unwrap().take() {
            let _ = lf.flush();
        }
    }
}

/// Blend epoch predictions with non-epoch models using ridge regression.
/// Epoch weights are fitted on the probe set (`split.pr`); non-epoch weights are
/// replaced by their average. Saves `{model_name}__epochs.{split.pr,split.fulltrain_pr}.npy`.
pub fn epoch_blend(model_name: &str, non_epoch_names: &[&str], lambda: f64, split: Split) {
    let preds_dir = split.preds_dir;
    std::fs::create_dir_all(preds_dir).unwrap();

    // Open log file (only if not already open, e.g. from fit3)
    let owns_log = LOG_FILE.lock().unwrap().is_none();
    if owns_log {
        let eblend_name = format!("{}__epochs", model_name);
        *LOG_FILE.lock().unwrap() = Some(BufWriter::new(
            File::create(format!("{}/{}.out", preds_dir, eblend_name)).unwrap()
        ));
    }

    // Discover epoch files
    let mut n_epochs = 0usize;
    loop {
        let path = format!("{}/{}_ep{:02}.{}.npy", preds_dir, model_name, n_epochs + 1, split.pr);
        if !std::path::Path::new(&path).exists() { break; }
        n_epochs += 1;
    }
    if n_epochs == 0 {
        teeln!("eblend: no epoch files found, skipping");
        return;
    }

    // Collect column names and load probe predictions
    let mut col_names: Vec<String> = Vec::new();
    let mut is_epoch: Vec<bool> = Vec::new();
    let mut pr_cols: Vec<Array1<f32>> = Vec::new();

    for ep in 1..=n_epochs {
        let name = format!("{}_ep{:02}", model_name, ep);
        let path = format!("{}/{}.{}.npy", preds_dir, name, split.pr);
        let preds: Array1<f32> = ndarray_npy::read_npy(&path).expect(&path);
        col_names.push(name);
        is_epoch.push(true);
        pr_cols.push(preds);
    }

    for &name in non_epoch_names {
        let path = format!("{}/{}.{}.npy", preds_dir, name, split.pr);
        if !std::path::Path::new(&path).exists() { continue; }
        let preds: Array1<f32> = ndarray_npy::read_npy(&path).expect(&path);
        col_names.push(name.to_string());
        is_epoch.push(false);
        pr_cols.push(preds);
    }

    let d = col_names.len();
    let n = pr_cols[0].len();

    // Log inputs
    let eblend_name = format!("{}__epochs", model_name);
    teeln!("[{}]", eblend_name);
    teeln!("eblend: {} epoch + {} non-epoch columns, lambda={}", n_epochs, d - n_epochs, lambda);
    for (i, name) in col_names.iter().enumerate() {
        teeln!("  col {:2}: {} {}", i, name, if is_epoch[i] { "(epoch)" } else { "" });
    }

    // Load target: raw ratings from probe set
    let ratings_path = format!("data/{}/ratings.npy", split.pr);
    let ratings: Array1<i8> = ndarray_npy::read_npy(&ratings_path).expect(&ratings_path);
    let target: Vec<f64> = ratings.iter().map(|&x| x as f64).collect();

    // Build XtX and Xty
    let mut xtx = DMatrix::<f64>::zeros(d, d);
    let mut xty = DVector::<f64>::zeros(d);

    for i in 0..d {
        for j in i..d {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += pr_cols[i][k] as f64 * pr_cols[j][k] as f64;
            }
            xtx[(i, j)] = s;
            xtx[(j, i)] = s;
        }
        let mut s = 0.0_f64;
        for k in 0..n {
            s += pr_cols[i][k] as f64 * target[k];
        }
        xty[i] = s;
    }

    for i in 0..d {
        xtx[(i, i)] += lambda * n as f64;
    }

    let weights = xtx.cholesky().unwrap().solve(&xty);

    // Log fitted weights
    teeln!("eblend fitted weights:");
    for i in 0..d {
        teeln!("  {:30} = {:+.6}{}", col_names[i], weights[i],
            if is_epoch[i] { "" } else { " (non-epoch)" });
    }

    // Compute RMSE with fitted weights on the probe set
    let mut sse = 0.0_f64;
    for k in 0..n {
        let mut pred = 0.0_f64;
        for i in 0..d {
            pred += weights[i] * pr_cols[i][k] as f64;
        }
        let err = pred - target[k];
        sse += err * err;
    }
    teeln!("eblend fitted RMSE: {:.5}", (sse / n as f64).sqrt());

    // Replace non-epoch columns with their means (constant offset preserves value range
    // but removes user/item-specific information, keeping only epoch-specific patterns)
    let non_epoch_indices: Vec<usize> = (0..d).filter(|&i| !is_epoch[i]).collect();
    let non_epoch_offset: f64 = non_epoch_indices.iter().map(|&i| {
        let mean = pr_cols[i].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        teeln!("  non-epoch {:30} w={:+.6} mean={:.5} contrib={:+.6}",
            col_names[i], weights[i], mean, weights[i] * mean);
        weights[i] * mean
    }).sum();
    teeln!("eblend non-epoch constant offset: {:+.6}", non_epoch_offset);

    // RMSE with epoch weights + constant offset on the probe set
    sse = 0.0;
    for k in 0..n {
        let mut pred = non_epoch_offset;
        for i in 0..d {
            if is_epoch[i] {
                pred += weights[i] * pr_cols[i][k] as f64;
            }
        }
        let err = pred - target[k];
        sse += err * err;
    }
    teeln!("eblend final RMSE:  {:.5}", (sse / n as f64).sqrt());

    // Apply epoch weights + constant offset to probe and qual sets
    drop(pr_cols); // free memory before loading qual

    for dataset in &[split.pr, split.fulltrain_pr] {
        // Only load epoch columns
        let epoch_cols: Vec<Array1<f32>> = col_names.iter().enumerate()
            .filter(|&(i, _)| is_epoch[i])
            .map(|(_, name)| {
                let path = format!("{}/{}.{}.npy", preds_dir, name, dataset);
                ndarray_npy::read_npy(&path).expect(&path)
            }).collect();

        let epoch_weights: Vec<f64> = (0..d).filter(|&i| is_epoch[i])
            .map(|i| weights[i]).collect();

        let n_ds = epoch_cols[0].len();
        let mut preds = Array1::<f32>::zeros(n_ds);
        for k in 0..n_ds {
            let mut pred = non_epoch_offset;
            for i in 0..epoch_cols.len() {
                pred += epoch_weights[i] * epoch_cols[i][k] as f64;
            }
            preds[k] = pred as f32;
        }

        let path = format!("{}/{}.{}.npy", preds_dir, eblend_name, dataset);
        write_npy(&path, &preds).unwrap();
        teeln!("eblend saved {}", path);
    }
    teeln!();

    if owns_log {
        if let Some(mut lf) = LOG_FILE.lock().unwrap().take() {
            use std::io::Write as _;
            let _ = lf.flush();
        }
    }
}

/// Legacy mode for `epoch_blend`: apply pre-supplied epoch weights and a constant
/// offset to per-epoch predictions, without re-fitting. Reads
/// `{preds}/{model_name}_ep<NN>.{split.pr|fulltrain_pr}.npy` for NN in
/// `1..=epoch_weights.len()` and writes `{preds}/{model_name}__epochs.<ds>.npy`.
/// Useful for reproducing the original `__epochs` output from values copied out
/// of an existing `.out` file (note: those weights are typically truncated to
/// ~6 decimal places, so the reproduction is approximate).
pub fn epoch_blend_apply(
    model_name: &str,
    epoch_weights: &[f64],
    constant_offset: f64,
    split: Split,
) {
    let preds_dir = split.preds_dir;
    std::fs::create_dir_all(preds_dir).unwrap();

    let eblend_name = format!("{}__epochs", model_name);
    let owns_log = LOG_FILE.lock().unwrap().is_none();
    if owns_log {
        *LOG_FILE.lock().unwrap() = Some(BufWriter::new(
            File::create(format!("{}/{}.out", preds_dir, eblend_name)).unwrap()
        ));
    }

    teeln!("[{}]", eblend_name);
    teeln!("eblend legacy mode: {} epoch weights, constant offset={:+.6}",
        epoch_weights.len(), constant_offset);
    for (i, w) in epoch_weights.iter().enumerate() {
        teeln!("  epoch {:2} weight = {:+.6}", i + 1, w);
    }

    for dataset in &[split.pr, split.fulltrain_pr] {
        let mut epoch_preds: Vec<Array1<f32>> = Vec::with_capacity(epoch_weights.len());
        for ep in 1..=epoch_weights.len() {
            let path = format!("{}/{}_ep{:02}.{}.npy", preds_dir, model_name, ep, dataset);
            let preds: Array1<f32> = ndarray_npy::read_npy(&path).expect(&path);
            epoch_preds.push(preds);
        }

        let n = epoch_preds[0].len();
        let mut blend = Array1::<f32>::zeros(n);
        for k in 0..n {
            let mut pred = constant_offset;
            for (i, &w) in epoch_weights.iter().enumerate() {
                pred += w * epoch_preds[i][k] as f64;
            }
            blend[k] = pred as f32;
        }

        let out_path = format!("{}/{}.{}.npy", preds_dir, eblend_name, dataset);
        write_npy(&out_path, &blend).unwrap();
        teeln!("eblend saved {}", out_path);
    }
    teeln!();

    if owns_log {
        if let Some(mut lf) = LOG_FILE.lock().unwrap().take() {
            use std::io::Write as _;
            let _ = lf.flush();
        }
    }
}

// ---------------------------------------------------------------------------
// fit3 family
// ---------------------------------------------------------------------------

use knn3::{Knn3Model, Knn3Config};
use knnf::{KnnfModel, KnnfConfig};

/// Options for `fit3_inner` / `fit3!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fit3Opts {
    pub keep_epoch_preds: bool,
    pub keep_train_preds: bool,
    pub save_subscores: bool,
    pub save_factorscores: bool,
    pub no_fulltrain: bool,
    pub transpose: bool,
}

/// Call `fit3_inner` with named optional parameters.
/// Usage: `fit3!(Model, cfg, "rtg", "name", SPLIT_NEW, keep_train_preds: true)`
#[macro_export]
macro_rules! fit3 {
    ($M:ty, $cfg:expr, $target:expr, $model_name:expr, $split:expr $(, $key:ident : $val:expr)* $(,)?) => {
        $crate::fit3_inner::<$M>(
            $cfg, $target, $model_name, $split,
            $crate::Fit3Opts { $($key: $val,)* ..Default::default() },
        )
    };
}

/// Run `fit2_inner` for main model M, then KNNf (if ifeat exists), KNN3
/// (when not transposed), and epoch blend on its residuals.
#[inline]
pub fn fit3_inner<M: Regressor + Sync>(
    cfg: M::Config,
    target: &str,
    model_name: &str,
    split: Split,
    opts: Fit3Opts,
) {
    let Fit3Opts { keep_epoch_preds, keep_train_preds, save_subscores, save_factorscores, no_fulltrain, transpose } = opts;
    let preds_dir = split.preds_dir;

    // Open single log file for the entire fit3 run
    *LOG_FILE.lock().unwrap() = Some(BufWriter::new(
        File::create(format!("{}/{}.out", preds_dir, model_name)).unwrap()
    ));

    // Main model (save_train=true for knn residuals, save_probe_each_epoch=true for eblend)
    fit2_inner::<M>(cfg, target, model_name, split, Fit2Opts {
        save_train: true,
        save_probe_each_epoch: true,
        save_subscores,
        save_factorscores,
        no_fulltrain,
        transpose,
    });

    let base_target: &str = format!("1.0*{}", model_name).leak();

    // KNNf on residuals of main model (only if ifeat files were saved)
    // KNNf can transpose (factor-based, user-neighbor method makes sense)
    let ifeat_prefix: &str = format!("{}/{}.ifeat", preds_dir, model_name).leak();
    let ifeat_check = format!("{}.{}.npy", ifeat_prefix, split.tr);
    let has_knnf = std::path::Path::new(&ifeat_check).exists();
    if has_knnf {
        let knnf_name: &str = format!("{}__knnf", model_name).leak();
        let knnf_cfg = KnnfConfig::with_factors(ifeat_prefix);
        fit2_inner::<KnnfModel>(knnf_cfg, base_target, knnf_name, split, Fit2Opts {
            no_fulltrain, transpose, ..Default::default()
        });
    }

    // KNN3 on residuals of main model (skipped when transposed: train order differs, and
    // n_users×n_users similarity matrix would be too large with 480k users)
    let knn3_name: &str = format!("{}__knn3", model_name).leak();
    let has_knn3 = !transpose;
    if has_knn3 {
        fit2_inner::<Knn3Model>(Knn3Config::default(), base_target, knn3_name, split, Fit2Opts {
            no_fulltrain, ..Default::default()
        });
    }

    // Epoch blend: combine epoch predictions with non-epoch models
    let mut non_epoch: Vec<&str> = vec![model_name];
    if has_knn3 {
        non_epoch.push(knn3_name);
    }
    if has_knnf {
        non_epoch.push(format!("{}__knnf", model_name).leak());
    }
    epoch_blend(model_name, &non_epoch, 10.0, split);

    // Clean up epoch prediction files if not needed
    if !keep_epoch_preds {
        let mut ep = 1;
        loop {
            let mut found = false;
            for dataset in &[split.pr, split.fulltrain_pr, split.tr, split.fulltrain_tr] {
                let path = format!("{}/{}_ep{:02}.{}.npy", preds_dir, model_name, ep, dataset);
                if std::path::Path::new(&path).exists() {
                    std::fs::remove_file(&path).unwrap();
                    found = true;
                }
            }
            if !found { break; }
            ep += 1;
        }
        if ep > 1 {
            teeln!("eblend: removed {} epoch prediction files", (ep - 1) * 2);
        }
    }

    // Clean up train/fulltrain prediction files if not needed
    if !keep_train_preds {
        let mut removed = 0;
        let all_names: Vec<&str> = {
            let mut v = vec![model_name];
            if has_knn3 { v.push(knn3_name); }
            if has_knnf { v.push(format!("{}__knnf", model_name).leak()); }
            v
        };
        for name in &all_names {
            for dataset in &[split.tr, split.fulltrain_tr] {
                let path = format!("{}/{}.{}.npy", preds_dir, name, dataset);
                if std::path::Path::new(&path).exists() {
                    std::fs::remove_file(&path).unwrap();
                    removed += 1;
                }
            }
        }
        if removed > 0 {
            teeln!("fit3: removed {} train/fulltrain prediction files", removed);
        }
    }

    // Close log file
    if let Some(mut lf) = LOG_FILE.lock().unwrap().take() {
        let _ = lf.flush();
    }
}
