//! Factored CF-NADE-S: autoregressive rating model with timeSVD++-style
//! temporal extensions. Models the full 1–5 distribution per item (ordinal
//! head), conditioned on the user's rated set. Train target is always "rtg".

use crate::{Dataset, MaskedDataset, Regressor, get_users, rand_array2, rand_array3};
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2, Array3};
use parking_lot::Mutex;
use rand::{Rng, SeedableRng, prelude::SliceRandom, rngs::StdRng};
use std::collections::HashMap;

/// Transcendental f32 ops. With `--features det-math` they route through the
/// platform-independent `libm` crate (bit-identical across OSes); otherwise
/// they evaluate in f64 via the system libm and round back to f32 (faster, and
/// the f64 intermediate makes cross-OS differences vanishingly unlikely).
/// `sqrt`/`powi` are intentionally NOT wrapped: correctly-rounded / integer
/// power, already deterministic.
mod fmath {
    #[cfg(feature = "det-math")]
    #[inline]
    pub fn exp(x: f32) -> f32 {
        libm::expf(x)
    }
    #[cfg(not(feature = "det-math"))]
    #[inline]
    pub fn exp(x: f32) -> f32 {
        (x as f64).exp() as f32
    }

    #[cfg(feature = "det-math")]
    #[inline]
    pub fn tanh(x: f32) -> f32 {
        libm::tanhf(x)
    }
    #[cfg(not(feature = "det-math"))]
    #[inline]
    pub fn tanh(x: f32) -> f32 {
        (x as f64).tanh() as f32
    }

    #[cfg(feature = "det-math")]
    #[inline]
    pub fn ln(x: f32) -> f32 {
        libm::logf(x)
    }
    #[cfg(not(feature = "det-math"))]
    #[inline]
    pub fn ln(x: f32) -> f32 {
        (x as f64).ln() as f32
    }

    #[cfg(feature = "det-math")]
    #[inline]
    pub fn ln_1p(x: f32) -> f32 {
        libm::log1pf(x)
    }
    #[cfg(not(feature = "det-math"))]
    #[inline]
    pub fn ln_1p(x: f32) -> f32 {
        (x as f64).ln_1p() as f32
    }

    #[cfg(feature = "det-math")]
    #[inline]
    pub fn powf(b: f32, e: f32) -> f32 {
        libm::powf(b, e)
    }
    #[cfg(not(feature = "det-math"))]
    #[inline]
    pub fn powf(b: f32, e: f32) -> f32 {
        (b as f64).powf(e as f64) as f32
    }
}
use fmath::{exp, ln, ln_1p, powf, tanh};

const K: usize = 5;
const MAX_MILESTONES: usize = 4;
const SIDE_FEATS: usize = 7;
const DRIFT_BETA: f32 = 0.4;
const ITEM_TIME_BINS: usize = 64;
const DAY_BINS: usize = 64;
const FREQ_BINS: usize = 8;

type UserRating = (usize, u8, i16);
type UserItemDay = (usize, i16);

/// Hyperparameters and feature toggles for the CF-NADE-S model.
#[derive(Clone, Copy, Debug)]
pub struct CfNadeConfig {
    pub n_hidden: usize,
    pub n_hidden2: usize,
    pub rank: usize,
    pub n_epochs: usize,
    pub seed: u64,
    pub shuffle_users: bool,
    pub batch_size: usize,
    pub lr: f32,
    pub first_layer_lr_mult: f32,
    pub n_milestones: usize,
    pub milestones: [usize; MAX_MILESTONES],
    pub lr_gamma: f32,
    pub weight_decay: f32,
    pub ordinal_lambda: f32,
    pub rmse_weight: f32,
    pub ctx_norm_alpha: f32,
    pub swa_start: usize,
    pub use_ctx_target_day_scale: bool,
    pub use_implicit_probe_ctx: bool,
    pub use_user_bias: bool,
    pub use_user_drift: bool,
    pub use_user_item_scale: bool,
    pub use_user_item_scale_bin: bool,
    pub use_user_day_bias_bin: bool,
    pub use_user_day_scale_bin: bool,
    pub use_day_bias: bool,
    pub use_item_time_bias: bool,
    pub use_day_freq_bias: bool,
    pub use_side_features: bool,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for CfNadeConfig {
    fn default() -> Self {
        Self {
            n_hidden: 320,
            n_hidden2: 0,
            rank: 96,
            n_epochs: 96,
            seed: 42,
            shuffle_users: true,
            batch_size: 8,
            lr: 0.0006,
            first_layer_lr_mult: 1.4,
            n_milestones: 2,
            milestones: [64, 72, 0, 0],
            lr_gamma: 0.9,
            weight_decay: 0.008,
            ordinal_lambda: 1.0,
            rmse_weight: 0.1,
            ctx_norm_alpha: 0.5,
            swa_start: 18,
            use_ctx_target_day_scale: false,
            use_implicit_probe_ctx: false,
            use_user_bias: true,
            use_user_drift: true,
            use_user_item_scale: false,
            use_user_item_scale_bin: false,
            use_user_day_bias_bin: false,
            use_user_day_scale_bin: false,
            use_day_bias: false,
            use_item_time_bias: false,
            use_day_freq_bias: false,
            use_side_features: false,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Per-batch accumulated gradients (sparse for per-user/item params, dense for shared layers).
struct SparseBatchGrads {
    a: Vec<HashMap<usize, Vec<f32>>>,
    p: Vec<HashMap<usize, Vec<f32>>>,
    b: Vec<HashMap<usize, f32>>,
    user_bias: Vec<HashMap<usize, f32>>,
    user_drift: Vec<HashMap<usize, f32>>,
    user_item_scale: HashMap<usize, f32>,
    user_item_scale_drift: HashMap<usize, f32>,
    user_item_scale_bin: HashMap<usize, f32>,
    user_day_bias_bin: HashMap<usize, f32>,
    user_day_scale_bin: HashMap<usize, f32>,
    day_bias: Vec<HashMap<usize, f32>>,
    item_time_bias: Vec<HashMap<usize, f32>>,
    day_freq_bias: Vec<HashMap<usize, f32>>,
    ctx_target_day_scale: Array2<f32>,
    implicit_a: HashMap<usize, Vec<f32>>,
    enc_proj: Array2<f32>,
    hidden2_proj: Array2<f32>,
    dec_proj: Array2<f32>,
    enc_bias: Array1<f32>,
    hidden2_bias: Array1<f32>,
    wide_w: Array2<f32>,
    count: usize,
}

impl SparseBatchGrads {
    /// Allocates empty gradient buffers sized for the given layer widths.
    fn new(hidden: usize, hidden2: usize, rank: usize) -> Self {
        Self {
            a: (0..K).map(|_| HashMap::new()).collect(),
            p: (0..K).map(|_| HashMap::new()).collect(),
            b: (0..K).map(|_| HashMap::new()).collect(),
            user_bias: (0..K).map(|_| HashMap::new()).collect(),
            user_drift: (0..K).map(|_| HashMap::new()).collect(),
            user_item_scale: HashMap::new(),
            user_item_scale_drift: HashMap::new(),
            user_item_scale_bin: HashMap::new(),
            user_day_bias_bin: HashMap::new(),
            user_day_scale_bin: HashMap::new(),
            day_bias: (0..K).map(|_| HashMap::new()).collect(),
            item_time_bias: (0..K).map(|_| HashMap::new()).collect(),
            day_freq_bias: (0..K).map(|_| HashMap::new()).collect(),
            ctx_target_day_scale: Array2::zeros((DAY_BINS, DAY_BINS)),
            implicit_a: HashMap::new(),
            enc_proj: Array2::zeros((hidden, rank)),
            hidden2_proj: Array2::zeros((hidden2, hidden)),
            dec_proj: Array2::zeros((rank, if hidden2 > 0 { hidden2 } else { hidden })),
            enc_bias: Array1::zeros(hidden),
            hidden2_bias: Array1::zeros(hidden2),
            wide_w: Array2::zeros((K, SIDE_FEATS)),
            count: 0,
        }
    }

    /// Resets all accumulators to zero for reuse across batches.
    fn clear(&mut self) {
        for m in &mut self.a {
            m.clear();
        }
        for m in &mut self.p {
            m.clear();
        }
        for m in &mut self.b {
            m.clear();
        }
        for m in &mut self.user_bias {
            m.clear();
        }
        for m in &mut self.user_drift {
            m.clear();
        }
        self.user_item_scale.clear();
        self.user_item_scale_drift.clear();
        self.user_item_scale_bin.clear();
        self.user_day_bias_bin.clear();
        self.user_day_scale_bin.clear();
        for m in &mut self.day_bias {
            m.clear();
        }
        for m in &mut self.item_time_bias {
            m.clear();
        }
        for m in &mut self.day_freq_bias {
            m.clear();
        }
        self.ctx_target_day_scale.fill(0.0);
        self.implicit_a.clear();
        self.enc_proj.fill(0.0);
        self.hidden2_proj.fill(0.0);
        self.dec_proj.fill(0.0);
        self.enc_bias.fill(0.0);
        self.hidden2_bias.fill(0.0);
        self.wide_w.fill(0.0);
        self.count = 0;
    }
}

/// Precomputed per-user/item statistics feeding the side features and temporal terms.
struct SideStats {
    user_mean_ctr: Array1<f32>,
    item_mean_ctr: Array1<f32>,
    user_log_cnt: Array1<f32>,
    item_log_cnt: Array1<f32>,
    item_year_norm: Array1<f32>,
    user_day_mean: Array1<f32>,
    day_min: i32,
    day_mean: f32,
    day_span: f32,
    day_std: f32,
}

/// Factored CF-NADE-S model: parameters, Adam/SWA state, and cached per-user data.
pub struct CfNadeModel {
    cfg: CfNadeConfig,

    // Factored CF-NADE-S parameters.
    // A/P are [rating, item, rank].
    a: Array3<f32>,
    p: Array3<f32>,
    b: Array2<f32>,
    user_bias: Array2<f32>,
    user_drift: Array2<f32>,
    user_item_scale: Array1<f32>,
    user_item_scale_drift: Array1<f32>,
    user_item_scale_bin: Array1<f32>,
    user_day_bias_bin: Array1<f32>,
    user_day_scale_bin: Array1<f32>,
    day_bias: Array2<f32>,
    item_time_bias: Array2<f32>,
    day_freq_bias: Array2<f32>,
    ctx_target_day_scale: Array2<f32>,
    implicit_a: Array2<f32>,
    enc_proj: Array2<f32>, // B in the paper, [hidden, rank]
    hidden2_proj: Array2<f32>,
    dec_proj: Array2<f32>, // Q in the paper, [rank, hidden or hidden2]
    enc_bias: Array1<f32>, // c in the paper
    hidden2_bias: Array1<f32>,
    wide_w: Array2<f32>,
    swa_a: Array3<f32>,
    swa_p: Array3<f32>,
    swa_b: Array2<f32>,
    swa_user_bias: Array2<f32>,
    swa_user_drift: Array2<f32>,
    swa_user_item_scale: Array1<f32>,
    swa_user_item_scale_drift: Array1<f32>,
    swa_user_item_scale_bin: Array1<f32>,
    swa_user_day_bias_bin: Array1<f32>,
    swa_user_day_scale_bin: Array1<f32>,
    swa_day_bias: Array2<f32>,
    swa_item_time_bias: Array2<f32>,
    swa_day_freq_bias: Array2<f32>,
    swa_ctx_target_day_scale: Array2<f32>,
    swa_implicit_a: Array2<f32>,
    swa_enc_proj: Array2<f32>,
    swa_hidden2_proj: Array2<f32>,
    swa_dec_proj: Array2<f32>,
    swa_enc_bias: Array1<f32>,
    swa_hidden2_bias: Array1<f32>,
    swa_wide_w: Array2<f32>,

    // Adam states.
    m_a: Array3<f32>,
    v_a: Array3<f32>,
    m_p: Array3<f32>,
    v_p: Array3<f32>,
    m_b: Array2<f32>,
    v_b: Array2<f32>,
    m_user_bias: Array2<f32>,
    v_user_bias: Array2<f32>,
    m_user_drift: Array2<f32>,
    v_user_drift: Array2<f32>,
    m_user_item_scale: Array1<f32>,
    v_user_item_scale: Array1<f32>,
    m_user_item_scale_drift: Array1<f32>,
    v_user_item_scale_drift: Array1<f32>,
    m_user_item_scale_bin: Array1<f32>,
    v_user_item_scale_bin: Array1<f32>,
    m_user_day_bias_bin: Array1<f32>,
    v_user_day_bias_bin: Array1<f32>,
    m_user_day_scale_bin: Array1<f32>,
    v_user_day_scale_bin: Array1<f32>,
    m_day_bias: Array2<f32>,
    v_day_bias: Array2<f32>,
    m_item_time_bias: Array2<f32>,
    v_item_time_bias: Array2<f32>,
    m_day_freq_bias: Array2<f32>,
    v_day_freq_bias: Array2<f32>,
    m_ctx_target_day_scale: Array2<f32>,
    v_ctx_target_day_scale: Array2<f32>,
    m_implicit_a: Array2<f32>,
    v_implicit_a: Array2<f32>,
    m_enc_proj: Array2<f32>,
    v_enc_proj: Array2<f32>,
    m_hidden2_proj: Array2<f32>,
    v_hidden2_proj: Array2<f32>,
    m_dec_proj: Array2<f32>,
    v_dec_proj: Array2<f32>,
    m_enc_bias: Array1<f32>,
    v_enc_bias: Array1<f32>,
    m_hidden2_bias: Array1<f32>,
    v_hidden2_bias: Array1<f32>,
    m_wide_w: Array2<f32>,
    v_wide_w: Array2<f32>,

    user_ratings: Vec<Vec<UserRating>>,
    user_probe_items: Vec<Vec<UserItemDay>>,
    user_day_freq_bin: Vec<HashMap<i16, u8>>,
    side_stats: SideStats,
    pred_cache: Mutex<(usize, usize, Array1<f32>)>, // cached z = Q h for one user/day-bin
    rng: StdRng,
    timestep: usize,
    swa_count: usize,
    /// True while the live params hold the SWA average (swapped in at the end of
    /// an epoch so the driver's per-epoch predict/save sees the averaged model);
    /// swapped back to the SGD snapshot at the start of the next epoch.
    swapped_in: bool,
}

/// Groups each user's (item, rating, day) triples into a per-user list, skipping test rows.
fn build_user_ratings(ds: &Dataset) -> Vec<Vec<UserRating>> {
    let mut ratings: Vec<Vec<UserRating>> = (0..ds.n_users)
        .map(|u| Vec::with_capacity(ds.user_cnts[u] as usize))
        .collect();

    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 {
            continue;
        }
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let r = ds.raw_ratings[idx].clamp(1, 5) as u8 - 1;
        let day = ds.dates[idx];
        ratings[u].push((i, r, day));
    }

    ratings
}

/// Groups each user's (item, day) pairs from a masked dataset for implicit-feedback context.
fn build_user_item_days(ds: &MaskedDataset) -> Vec<Vec<UserItemDay>> {
    let mut items: Vec<Vec<UserItemDay>> = (0..ds.n_users)
        .map(|u| Vec::with_capacity(ds.user_cnts[u] as usize))
        .collect();

    for idx in 0..ds.n_ratings {
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let day = ds.dates[idx];
        items[u].push((i, day));
    }

    items
}

/// Maps each (user, day) to a log-scaled frequency bin of how many ratings share that day.
fn build_user_day_freq_bins(tr: &Dataset, pr: &MaskedDataset) -> Vec<HashMap<i16, u8>> {
    let mut counts: Vec<HashMap<i16, u16>> = (0..tr.n_users).map(|_| HashMap::new()).collect();

    let mut accumulate = |user_idxs: &Array1<i32>, dates: &Array1<i16>, n: usize| {
        for idx in 0..n {
            let u = user_idxs[idx] as usize;
            let day = dates[idx];
            *counts[u].entry(day).or_insert(0) += 1;
        }
    };
    accumulate(&tr.user_idxs, &tr.dates, tr.n_ratings);
    accumulate(pr.user_idxs, pr.dates, pr.n_ratings);

    counts
        .into_iter()
        .map(|m| {
            m.into_iter()
                .map(|(day, cnt)| {
                    let bin = ((u16::BITS - cnt.max(1).leading_zeros() - 1) as usize)
                        .min(FREQ_BINS - 1) as u8;
                    (day, bin)
                })
                .collect()
        })
        .collect()
}

/// Standardizes an array to zero mean and unit standard deviation.
fn normalize_mean_std(values: &Array1<f32>) -> Array1<f32> {
    let n = values.len().max(1) as f32;
    let mean = values.iter().copied().sum::<f32>() / n;
    let var = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let std = var.sqrt().max(1e-3);
    values.mapv(|v| (v - mean) / std)
}

/// Computes the per-user/item means, counts, year and day statistics used as side features.
fn build_side_stats(ds: &Dataset) -> SideStats {
    let mut user_sum = Array1::<f32>::zeros(ds.n_users);
    let mut item_sum = Array1::<f32>::zeros(ds.n_items);
    let mut user_seen = Array1::<f32>::zeros(ds.n_users);
    let mut item_seen = Array1::<f32>::zeros(ds.n_items);
    let mut user_day_sum = Array1::<f32>::zeros(ds.n_users);

    let mut global_sum = 0.0;
    let mut day_sum = 0.0;
    let mut day_sq_sum = 0.0;
    let mut day_n = 0usize;
    let mut day_min = i32::MAX;
    let mut day_max = i32::MIN;
    let mut year_min = i32::MAX;
    let mut year_max = i32::MIN;

    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 {
            continue;
        }
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let r = ds.raw_ratings[idx] as f32;
        user_sum[u] += r;
        item_sum[i] += r;
        user_seen[u] += 1.0;
        item_seen[i] += 1.0;
        user_day_sum[u] += ds.dates[idx] as f32;
        global_sum += r;

        let day = ds.dates[idx] as f64;
        day_min = day_min.min(ds.dates[idx] as i32);
        day_max = day_max.max(ds.dates[idx] as i32);
        day_sum += day;
        day_sq_sum += day * day;
        day_n += 1;
    }

    let global_mean = global_sum / user_seen.sum().max(1.0);

    let mut user_mean = Array1::<f32>::zeros(ds.n_users);
    let mut item_mean = Array1::<f32>::zeros(ds.n_items);
    for u in 0..ds.n_users {
        if user_seen[u] > 0.0 {
            user_mean[u] = (user_sum[u] / user_seen[u] - global_mean) / 2.0;
        }
    }
    for i in 0..ds.n_items {
        if item_seen[i] > 0.0 {
            item_mean[i] = (item_sum[i] / item_seen[i] - global_mean) / 2.0;
        }
    }

    for &year in ds.item_years.iter() {
        if year > 0 {
            year_min = year_min.min(year);
            year_max = year_max.max(year);
        }
    }
    let year_span = (year_max - year_min).max(1) as f32;
    let item_year_norm = ds.item_years.mapv(|year| {
        if year > 0 {
            ((year - year_min) as f32 / year_span) * 2.0 - 1.0
        } else {
            0.0
        }
    });

    let user_log_raw = user_seen.mapv(|cnt| ln_1p(cnt));
    let item_log_raw = item_seen.mapv(|cnt| ln_1p(cnt));

    let day_mean = (day_sum / day_n.max(1) as f64) as f32;
    let day_var = (day_sq_sum / day_n.max(1) as f64) as f32 - day_mean * day_mean;
    let day_std = day_var.max(1e-3).sqrt();
    let mut user_day_mean = Array1::<f32>::from_elem(ds.n_users, day_mean);
    for u in 0..ds.n_users {
        if user_seen[u] > 0.0 {
            user_day_mean[u] = user_day_sum[u] / user_seen[u];
        }
    }

    SideStats {
        user_mean_ctr: user_mean,
        item_mean_ctr: item_mean,
        user_log_cnt: normalize_mean_std(&user_log_raw),
        item_log_cnt: normalize_mean_std(&item_log_raw),
        item_year_norm,
        user_day_mean,
        day_min,
        day_mean,
        day_span: (day_max - day_min).max(1) as f32,
        day_std,
    }
}

/// Numerically stable softmax over the K rating scores.
#[inline]
fn softmax(scores: &[f32; K]) -> [f32; K] {
    let mut max = f32::NEG_INFINITY;
    for &s in scores {
        if s > max {
            max = s;
        }
    }
    let mut out = [0.0; K];
    let mut sum = 0.0;
    for k in 0..K {
        let e = exp(scores[k] - max);
        out[k] = e;
        sum += e;
    }
    for k in 0..K {
        out[k] /= sum;
    }
    out
}

/// Turns per-rating base logits into cumulative ordinal scores.
#[inline]
fn base_to_scores(base: &[f32; K]) -> [f32; K] {
    let mut scores = [0.0; K];
    for t in 0..K {
        scores[t] = base[t] + if t > 0 { scores[t - 1] } else { 0.0 };
    }
    scores
}

/// Computes the expected rating (1..5) from base logits via the softmax distribution.
#[inline]
fn expected_from_base(base: &[f32; K]) -> f32 {
    let probs = softmax(&base_to_scores(base));
    let mut pred = 0.0;
    for k in 0..K {
        pred += (k as f32 + 1.0) * probs[k];
    }
    pred
}

/// Cross-entropy (softmax) loss and gradient w.r.t. scores for the target rating.
fn regular_loss_grad(scores: &[f32; K], target: usize) -> (f32, [f32; K]) {
    let probs = softmax(scores);
    let mut grad = probs;
    grad[target] -= 1.0;
    let loss = -ln(probs[target].max(1e-12));
    (loss, grad)
}

/// Ordinal loss and gradient: sums prefix/suffix log-partition penalties so scores stay
/// monotone around the target rating, encouraging the model to respect rating ordering.
fn ordinal_loss_grad(scores: &[f32; K], target: usize) -> (f32, [f32; K]) {
    let mut grad = [0.0; K];
    let mut loss = 0.0;

    for j in 0..=target {
        let mut max = f32::NEG_INFINITY;
        for &s in &scores[..=j] {
            if s > max {
                max = s;
            }
        }
        let mut exps = [0.0; K];
        let mut sum = 0.0;
        for t in 0..=j {
            let e = exp(scores[t] - max);
            exps[t] = e;
            sum += e;
        }
        loss += max + ln(sum) - scores[j];
        for t in 0..=j {
            grad[t] += exps[t] / sum;
        }
        grad[j] -= 1.0;
    }

    for j in target..K {
        let mut max = f32::NEG_INFINITY;
        for &s in &scores[j..K] {
            if s > max {
                max = s;
            }
        }
        let mut exps = [0.0; K];
        let mut sum = 0.0;
        for t in j..K {
            let e = exp(scores[t] - max);
            exps[t] = e;
            sum += e;
        }
        loss += max + ln(sum) - scores[j];
        for t in j..K {
            grad[t] += exps[t] / sum;
        }
        grad[j] -= 1.0;
    }

    (loss, grad)
}

/// Squared-error loss and gradient on the expected-rating value w.r.t. scores.
fn rmse_loss_grad(scores: &[f32; K], target: usize) -> (f32, [f32; K]) {
    let probs = softmax(scores);
    let mut pred = 0.0;
    for k in 0..K {
        pred += (k as f32 + 1.0) * probs[k];
    }
    let target_val = target as f32 + 1.0;
    let err = pred - target_val;
    let mut grad = [0.0; K];
    for k in 0..K {
        grad[k] = err * probs[k] * ((k as f32 + 1.0) - pred);
    }
    (0.5 * err * err, grad)
}

/// Initializes the per-item rating biases from smoothed empirical rating frequencies.
fn init_item_biases(ds: &Dataset) -> Array2<f32> {
    let mut counts = vec![[0u32; K]; ds.n_items];
    let mut totals = vec![0u32; ds.n_items];
    let mut global = [0u64; K];
    let mut global_total = 0u64;

    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 {
            continue;
        }
        let item = ds.item_idxs[idx] as usize;
        let rating = ds.raw_ratings[idx].clamp(1, 5) as usize - 1;
        counts[item][rating] += 1;
        totals[item] += 1;
        global[rating] += 1;
        global_total += 1;
    }

    let eps = 1e-3;
    let mut b = Array2::<f32>::zeros((K, ds.n_items));
    for item in 0..ds.n_items {
        let (use_counts, tot) = if totals[item] > 0 {
            (counts[item], totals[item] as f32)
        } else {
            let mut fallback = [0u32; K];
            for k in 0..K {
                fallback[k] = global[k] as u32;
            }
            (fallback, global_total.max(1) as f32)
        };

        let mut prev = 0.0;
        for k in 0..K {
            let p = (use_counts[k] as f32 + eps) / (tot + K as f32 * eps);
            let logp = ln(p);
            b[[k, item]] = if k == 0 { logp } else { logp - prev };
            prev = logp;
        }
    }
    b
}

impl CfNadeModel {
    /// Invalidates the cached per-user/day-bin encoding.
    #[inline]
    fn clear_cache(&self) {
        let mut guard = self.pred_cache.lock();
        guard.0 = usize::MAX;
        guard.1 = usize::MAX;
        guard.2 = Array1::zeros(0);
    }

    /// Folds a 3D parameter array into its running SWA average.
    fn update_swa_3d(avg: &mut Array3<f32>, param: &Array3<f32>, count: usize) {
        let c = count as f32;
        for (avg, param) in avg.iter_mut().zip(param.iter()) {
            *avg += (*param - *avg) / c;
        }
    }

    /// Folds a 2D parameter array into its running SWA average.
    fn update_swa_2d(avg: &mut Array2<f32>, param: &Array2<f32>, count: usize) {
        let c = count as f32;
        for (avg, param) in avg.iter_mut().zip(param.iter()) {
            *avg += (*param - *avg) / c;
        }
    }

    /// Folds a 1D parameter array into its running SWA average.
    fn update_swa_1d(avg: &mut Array1<f32>, param: &Array1<f32>, count: usize) {
        let c = count as f32;
        for (avg, param) in avg.iter_mut().zip(param.iter()) {
            *avg += (*param - *avg) / c;
        }
    }

    /// Updates the SWA running average of all parameters once past `swa_start`.
    fn update_swa(&mut self, epoch: usize) {
        if epoch < self.cfg.swa_start {
            return;
        }
        self.swa_count += 1;
        let count = self.swa_count;
        Self::update_swa_3d(&mut self.swa_a, &self.a, count);
        Self::update_swa_3d(&mut self.swa_p, &self.p, count);
        Self::update_swa_2d(&mut self.swa_b, &self.b, count);
        Self::update_swa_2d(&mut self.swa_user_bias, &self.user_bias, count);
        Self::update_swa_2d(&mut self.swa_user_drift, &self.user_drift, count);
        Self::update_swa_1d(&mut self.swa_user_item_scale, &self.user_item_scale, count);
        Self::update_swa_1d(
            &mut self.swa_user_item_scale_drift,
            &self.user_item_scale_drift,
            count,
        );
        Self::update_swa_1d(
            &mut self.swa_user_item_scale_bin,
            &self.user_item_scale_bin,
            count,
        );
        Self::update_swa_1d(
            &mut self.swa_user_day_bias_bin,
            &self.user_day_bias_bin,
            count,
        );
        Self::update_swa_1d(
            &mut self.swa_user_day_scale_bin,
            &self.user_day_scale_bin,
            count,
        );
        Self::update_swa_2d(&mut self.swa_day_bias, &self.day_bias, count);
        Self::update_swa_2d(&mut self.swa_item_time_bias, &self.item_time_bias, count);
        Self::update_swa_2d(&mut self.swa_day_freq_bias, &self.day_freq_bias, count);
        Self::update_swa_2d(
            &mut self.swa_ctx_target_day_scale,
            &self.ctx_target_day_scale,
            count,
        );
        Self::update_swa_2d(&mut self.swa_implicit_a, &self.implicit_a, count);
        Self::update_swa_2d(&mut self.swa_enc_proj, &self.enc_proj, count);
        Self::update_swa_2d(&mut self.swa_hidden2_proj, &self.hidden2_proj, count);
        Self::update_swa_2d(&mut self.swa_dec_proj, &self.dec_proj, count);
        Self::update_swa_1d(&mut self.swa_enc_bias, &self.enc_bias, count);
        Self::update_swa_1d(&mut self.swa_hidden2_bias, &self.hidden2_bias, count);
        Self::update_swa_2d(&mut self.swa_wide_w, &self.wide_w, count);
    }

    /// Swaps the live parameters with the SWA averages so prediction uses the averaged model.
    fn swap_with_swa(&mut self) {
        if self.swa_count == 0 {
            return;
        }
        std::mem::swap(&mut self.a, &mut self.swa_a);
        std::mem::swap(&mut self.p, &mut self.swa_p);
        std::mem::swap(&mut self.b, &mut self.swa_b);
        std::mem::swap(&mut self.user_bias, &mut self.swa_user_bias);
        std::mem::swap(&mut self.user_drift, &mut self.swa_user_drift);
        std::mem::swap(&mut self.user_item_scale, &mut self.swa_user_item_scale);
        std::mem::swap(
            &mut self.user_item_scale_drift,
            &mut self.swa_user_item_scale_drift,
        );
        std::mem::swap(
            &mut self.user_item_scale_bin,
            &mut self.swa_user_item_scale_bin,
        );
        std::mem::swap(&mut self.user_day_bias_bin, &mut self.swa_user_day_bias_bin);
        std::mem::swap(
            &mut self.user_day_scale_bin,
            &mut self.swa_user_day_scale_bin,
        );
        std::mem::swap(&mut self.day_bias, &mut self.swa_day_bias);
        std::mem::swap(&mut self.item_time_bias, &mut self.swa_item_time_bias);
        std::mem::swap(&mut self.day_freq_bias, &mut self.swa_day_freq_bias);
        std::mem::swap(
            &mut self.ctx_target_day_scale,
            &mut self.swa_ctx_target_day_scale,
        );
        std::mem::swap(&mut self.implicit_a, &mut self.swa_implicit_a);
        std::mem::swap(&mut self.enc_proj, &mut self.swa_enc_proj);
        std::mem::swap(&mut self.hidden2_proj, &mut self.swa_hidden2_proj);
        std::mem::swap(&mut self.dec_proj, &mut self.swa_dec_proj);
        std::mem::swap(&mut self.enc_bias, &mut self.swa_enc_bias);
        std::mem::swap(&mut self.hidden2_bias, &mut self.swa_hidden2_bias);
        std::mem::swap(&mut self.wide_w, &mut self.swa_wide_w);
        self.clear_cache();
    }

    /// Normalization factor that downweights the context sum by its length.
    #[inline]
    fn ctx_scale(&self, ctx_len: usize) -> f32 {
        if ctx_len == 0 {
            1.0
        } else {
            powf(ctx_len as f32, -self.cfg.ctx_norm_alpha)
        }
    }

    /// Scales the context sum in place by the length-based factor and returns it.
    #[inline]
    fn apply_ctx_scale(&self, ctx_sum: &mut [f64], ctx_len: usize) -> f32 {
        let scale = self.ctx_scale(ctx_len);
        if scale != 1.0 {
            for v in ctx_sum.iter_mut() {
                *v *= scale as f64;
            }
        }
        scale
    }

    /// Runs the context sum through the encoder/decoder layers to produce the latent vector z.
    fn encode_ctx_sum(&self, ctx_sum: &[f64]) -> Array1<f32> {
        let rank = self.cfg.rank;
        let hidden = self.cfg.n_hidden;
        let hidden2 = self.cfg.n_hidden2;

        let mut h = vec![0.0; hidden];
        for i in 0..hidden {
            let mut s = self.enc_bias[i] as f64;
            for j in 0..rank {
                s += self.enc_proj[[i, j]] as f64 * ctx_sum[j];
            }
            h[i] = tanh(s as f32);
        }

        let top = if hidden2 > 0 {
            let mut h2 = vec![0.0; hidden2];
            for i in 0..hidden2 {
                let mut s = self.hidden2_bias[i] as f64;
                for j in 0..hidden {
                    s += self.hidden2_proj[[i, j]] as f64 * h[j] as f64;
                }
                h2[i] = tanh(s as f32);
            }
            h2
        } else {
            h
        };

        let mut z = Array1::zeros(rank);
        for j in 0..rank {
            let mut s = 0.0;
            for i in 0..top.len() {
                s += self.dec_proj[[j, i]] as f64 * top[i] as f64;
            }
            z[j] = s as f32;
        }
        z
    }

    /// Sums the A factors of rated items into per-day-bin context vectors.
    fn build_ctx_day_bin_sums(&self, ratings: &[UserRating]) -> Vec<Vec<f64>> {
        let rank = self.cfg.rank;
        let mut ctx_bin_sums = vec![vec![0.0; rank]; DAY_BINS];
        for &(item, rating, day) in ratings {
            let ctx_day_index = self.day_index(day as i32);
            let bin_sum = &mut ctx_bin_sums[ctx_day_index];
            for t in 0..=rating as usize {
                for j in 0..rank {
                    bin_sum[j] += self.a[[t, item, j]] as f64;
                }
            }
        }
        ctx_bin_sums
    }

    /// Sums the implicit-feedback factors of items into per-day-bin context vectors.
    fn build_implicit_ctx_day_bin_sums(&self, items: &[UserItemDay]) -> Vec<Vec<f64>> {
        let rank = self.cfg.rank;
        let mut ctx_bin_sums = vec![vec![0.0; rank]; DAY_BINS];
        if !self.cfg.use_implicit_probe_ctx {
            return ctx_bin_sums;
        }
        for &(item, day) in items {
            let ctx_day_index = self.day_index(day as i32);
            let bin_sum = &mut ctx_bin_sums[ctx_day_index];
            for j in 0..rank {
                bin_sum[j] += self.implicit_a[[item, j]] as f64;
            }
        }
        ctx_bin_sums
    }

    /// Combines per-day-bin context sums with target-day weights and encodes them into z.
    fn encode_day_binned_ctx(
        &self,
        ctx_bin_sums: &[Vec<f64>],
        implicit_ctx_bin_sums: Option<&[Vec<f64>]>,
        ctx_len: usize,
        target_day_index: usize,
    ) -> Array1<f32> {
        let rank = self.cfg.rank;
        let mut ctx_sum = vec![0.0; rank];
        for ctx_day_index in 0..DAY_BINS {
            let ctx_weight = self.ctx_target_day_weight(target_day_index, ctx_day_index) as f64;
            let bin_sum = &ctx_bin_sums[ctx_day_index];
            for j in 0..rank {
                ctx_sum[j] += ctx_weight * bin_sum[j];
            }
            if let Some(implicit_ctx_bin_sums) = implicit_ctx_bin_sums {
                let implicit_bin_sum = &implicit_ctx_bin_sums[ctx_day_index];
                for j in 0..rank {
                    ctx_sum[j] += ctx_weight * implicit_bin_sum[j];
                }
            }
        }
        self.apply_ctx_scale(&mut ctx_sum, ctx_len);

        self.encode_ctx_sum(&ctx_sum)
    }

    /// Builds the latent z for a user at a given target day-bin from their full rated set.
    fn encode_user_for_day_index(
        &self,
        u: usize,
        ratings: &[UserRating],
        target_day_index: usize,
    ) -> Array1<f32> {
        if ratings.is_empty()
            && (!self.cfg.use_implicit_probe_ctx || self.user_probe_items[u].is_empty())
        {
            return Array1::zeros(self.cfg.rank);
        }
        let ctx_bin_sums = self.build_ctx_day_bin_sums(ratings);
        let implicit_ctx_bin_sums = if self.cfg.use_implicit_probe_ctx {
            Some(self.build_implicit_ctx_day_bin_sums(&self.user_probe_items[u]))
        } else {
            None
        };
        self.encode_day_binned_ctx(
            &ctx_bin_sums,
            implicit_ctx_bin_sums.as_deref(),
            ratings.len(),
            target_day_index,
        )
    }

    /// Builds the latent z for a user on a specific calendar day.
    fn encode_user_for_day(&self, u: usize, ratings: &[UserRating], day: i32) -> Array1<f32> {
        self.encode_user_for_day_index(u, ratings, self.day_index(day))
    }

    /// Assembles the side-feature vector for a (user, item, day) prediction.
    #[inline]
    fn side_features(&self, u: usize, item: usize, day: i32) -> [f32; SIDE_FEATS] {
        [
            1.0,
            self.side_stats.user_mean_ctr[u],
            self.side_stats.item_mean_ctr[item],
            self.side_stats.user_log_cnt[u],
            self.side_stats.item_log_cnt[item],
            (day as f32 - self.side_stats.day_mean) / self.side_stats.day_std,
            self.side_stats.item_year_norm[item],
        ]
    }

    /// Signed temporal deviation of a day from the user's mean rating day (timeSVD++ devₜ).
    #[inline]
    fn user_time_dev(&self, u: usize, day: i32) -> f32 {
        let delta = day as f32 - self.side_stats.user_day_mean[u];
        if delta == 0.0 {
            0.0
        } else {
            delta.signum() * powf(delta.abs() / self.side_stats.day_std, DRIFT_BETA)
        }
    }

    /// Flat index into the per-item time-bias array for an item on a given day.
    #[inline]
    fn item_time_index(&self, item: usize, day: i32) -> usize {
        let rel = (day - self.side_stats.day_min) as f32 / self.side_stats.day_span;
        let bin = (rel.clamp(0.0, 1.0) * ITEM_TIME_BINS as f32) as usize;
        let bin = bin.min(ITEM_TIME_BINS - 1);
        item * ITEM_TIME_BINS + bin
    }

    /// Maps a calendar day to its global day-bin index.
    #[inline]
    fn day_index(&self, day: i32) -> usize {
        let rel = (day - self.side_stats.day_min) as f32 / self.side_stats.day_span;
        let bin = (rel.clamp(0.0, 1.0) * DAY_BINS as f32) as usize;
        bin.min(DAY_BINS - 1)
    }

    /// Looks up the frequency bin for a user's rating day (0 if unseen).
    #[inline]
    fn day_freq_index(&self, u: usize, day: i32) -> usize {
        self.user_day_freq_bin[u]
            .get(&(day as i16))
            .copied()
            .unwrap_or(0) as usize
    }

    /// Learned weight relating a context day-bin to a target day-bin (1.0 when disabled).
    #[inline]
    fn ctx_target_day_weight(&self, target_day_index: usize, ctx_day_index: usize) -> f32 {
        if self.cfg.use_ctx_target_day_scale {
            1.0 + self.ctx_target_day_scale[[target_day_index, ctx_day_index]]
        } else {
            1.0
        }
    }

    /// Flat index into the per-user/day-bin item-scale array.
    #[inline]
    fn user_item_scale_bin_index(&self, u: usize, day_index: usize) -> usize {
        u * DAY_BINS + day_index
    }

    /// Flat index into the per-user/day-bin bias array.
    #[inline]
    fn user_day_bias_bin_index(&self, u: usize, day_index: usize) -> usize {
        u * DAY_BINS + day_index
    }

    /// Flat index into the per-user/day-bin scale array.
    #[inline]
    fn user_day_scale_bin_index(&self, u: usize, day_index: usize) -> usize {
        u * DAY_BINS + day_index
    }

    /// Per-rating item bias combining the base, day, item-time, and day-frequency terms.
    #[inline]
    fn item_bias_term(
        &self,
        t: usize,
        item: usize,
        day_index: usize,
        item_time_index: usize,
        day_freq_index: usize,
    ) -> f32 {
        let mut s = self.b[[t, item]];
        if self.cfg.use_day_bias {
            s += self.day_bias[[t, day_index]];
        }
        if self.cfg.use_item_time_bias {
            s += self.item_time_bias[[t, item_time_index]];
        }
        if self.cfg.use_day_freq_bias {
            s += self.day_freq_bias[[t, day_freq_index]];
        }
        s
    }

    /// Per-user/day-bin additive bias (0.0 when disabled).
    #[inline]
    fn user_day_bias_bin_value(&self, u: usize, day_index: usize) -> f32 {
        if self.cfg.use_user_day_bias_bin {
            self.user_day_bias_bin[self.user_day_bias_bin_index(u, day_index)]
        } else {
            0.0
        }
    }

    /// Per-user/day-bin multiplicative scale on the logits (1.0 when disabled).
    #[inline]
    fn user_day_scale_factor(&self, u: usize, day_index: usize) -> f32 {
        if self.cfg.use_user_day_scale_bin {
            1.0 + self.user_day_scale_bin[self.user_day_scale_bin_index(u, day_index)]
        } else {
            1.0
        }
    }

    /// Per-user multiplicative scale on item-bias terms, with drift and optional day-bin (1.0 when disabled).
    #[inline]
    fn user_item_scale_factor(&self, u: usize, user_dev: f32, day_index: usize) -> f32 {
        if self.cfg.use_user_item_scale {
            let mut scale =
                1.0 + self.user_item_scale[u] + self.user_item_scale_drift[u] * user_dev;
            if self.cfg.use_user_item_scale_bin {
                scale += self.user_item_scale_bin[self.user_item_scale_bin_index(u, day_index)];
            }
            scale
        } else {
            1.0
        }
    }

    /// Assembles the K base logits for one rating from the latent z, biases, and side features.
    fn base_logits_from_parts(
        &self,
        z: &[f32],
        u: usize,
        item: usize,
        day: i32,
        user_dev: f32,
        side: Option<&[f32; SIDE_FEATS]>,
    ) -> [f32; K] {
        let mut base = [0.0; K];
        let day_index = self.day_index(day);
        let item_time_index = self.item_time_index(item, day);
        let day_freq_index = self.day_freq_index(u, day);
        let item_scale = self.user_item_scale_factor(u, user_dev, day_index);
        let user_day_bias = self.user_day_bias_bin_value(u, day_index);
        let user_day_scale = self.user_day_scale_factor(u, day_index);
        for t in 0..K {
            let mut s = (item_scale
                * self.item_bias_term(t, item, day_index, item_time_index, day_freq_index))
                as f64;
            if self.cfg.use_user_bias {
                s += self.user_bias[[t, u]] as f64;
            }
            if self.cfg.use_user_drift {
                s += (self.user_drift[[t, u]] * user_dev) as f64;
            }
            s += user_day_bias as f64;
            for j in 0..self.cfg.rank {
                s += self.p[[t, item, j]] as f64 * z[j] as f64;
            }
            if let Some(side) = side {
                for f in 0..SIDE_FEATS {
                    s += (self.wide_w[[t, f]] * side[f]) as f64;
                }
            }
            base[t] = (user_day_scale as f64 * s) as f32;
        }
        base
    }

    /// Predicts the expected rating given a precomputed latent z.
    fn predict_with_z(&self, z: &[f32], u: usize, item: usize, day: i32) -> f32 {
        let side = if self.cfg.use_side_features {
            Some(self.side_features(u, item, day))
        } else {
            None
        };
        let user_dev = if self.cfg.use_user_drift {
            self.user_time_dev(u, day)
        } else {
            0.0
        };
        let base = self.base_logits_from_parts(z, u, item, day, user_dev, side.as_ref());
        expected_from_base(&base)
    }

    /// One autoregressive training pass for a user: randomly splits ratings into
    /// context/suffix, encodes the context, then accumulates the forward+backward
    /// gradients for every suffix rating into `grads`.
    fn process_user(&mut self, u: usize, grads: &mut SparseBatchGrads) {
        let ratings = &self.user_ratings[u];
        let d = ratings.len();
        if d == 0 {
            return;
        }

        let rank = self.cfg.rank;
        let hidden = self.cfg.n_hidden;
        let hidden2 = self.cfg.n_hidden2;

        let mut order: Vec<usize> = (0..d).collect();
        order.shuffle(&mut self.rng);
        let split = self.rng.random_range(1..=d);
        let ctx_len = split - 1;
        let suffix_len = d - ctx_len;
        let scale = d as f32 / suffix_len as f32;

        let mut ctx_bin_sums = vec![vec![0.0; rank]; DAY_BINS];
        for &oi in &order[..ctx_len] {
            let (item, rating, day) = ratings[oi];
            let ctx_day_index = self.day_index(day as i32);
            let bin_sum = &mut ctx_bin_sums[ctx_day_index];
            for t in 0..=rating as usize {
                for j in 0..rank {
                    bin_sum[j] += self.a[[t, item, j]] as f64;
                }
            }
        }
        let mut suffix_by_day: Vec<Vec<usize>> = (0..DAY_BINS).map(|_| Vec::new()).collect();
        for &oi in &order[ctx_len..] {
            let target_day_index = self.day_index(ratings[oi].2 as i32);
            suffix_by_day[target_day_index].push(oi);
        }
        let implicit_items: Vec<UserItemDay> = if self.cfg.use_implicit_probe_ctx {
            order[ctx_len..]
                .iter()
                .map(|&oi| {
                    let (item, _, day) = ratings[oi];
                    (item, day)
                })
                .collect()
        } else {
            Vec::new()
        };
        let implicit_ctx_bin_sums = if self.cfg.use_implicit_probe_ctx {
            Some(self.build_implicit_ctx_day_bin_sums(&implicit_items))
        } else {
            None
        };
        let mut grad_ctx_bin_sums = vec![vec![0.0; rank]; DAY_BINS];
        let mut grad_implicit_ctx_bin_sums = vec![vec![0.0; rank]; DAY_BINS];

        for target_day_index in 0..DAY_BINS {
            if suffix_by_day[target_day_index].is_empty() {
                continue;
            }

            let mut ctx_sum = vec![0.0; rank];
            for ctx_day_index in 0..DAY_BINS {
                let ctx_weight =
                    self.ctx_target_day_weight(target_day_index, ctx_day_index) as f64;
                let bin_sum = &ctx_bin_sums[ctx_day_index];
                for j in 0..rank {
                    ctx_sum[j] += ctx_weight * bin_sum[j];
                }
                if let Some(implicit_ctx_bin_sums) = implicit_ctx_bin_sums.as_ref() {
                    let implicit_bin_sum = &implicit_ctx_bin_sums[ctx_day_index];
                    for j in 0..rank {
                        ctx_sum[j] += ctx_weight * implicit_bin_sum[j];
                    }
                }
            }
            let ctx_scale = self.apply_ctx_scale(&mut ctx_sum, ctx_len);

            let mut h = vec![0.0; hidden];
            for i in 0..hidden {
                let mut s = self.enc_bias[i] as f64;
                for j in 0..rank {
                    s += self.enc_proj[[i, j]] as f64 * ctx_sum[j];
                }
                h[i] = tanh(s as f32);
            }

            let top = if hidden2 > 0 {
                let mut h2 = vec![0.0; hidden2];
                for i in 0..hidden2 {
                    let mut s = self.hidden2_bias[i] as f64;
                    for j in 0..hidden {
                        s += self.hidden2_proj[[i, j]] as f64 * h[j] as f64;
                    }
                    h2[i] = tanh(s as f32);
                }
                h2
            } else {
                h.clone()
            };

            let mut z = vec![0.0; rank];
            for j in 0..rank {
                let mut s = 0.0;
                for i in 0..top.len() {
                    s += self.dec_proj[[j, i]] as f64 * top[i] as f64;
                }
                z[j] = s as f32;
            }

            let mut grad_z = vec![0.0; rank];

            for &oi in &suffix_by_day[target_day_index] {
                let (item, rating, day) = ratings[oi];
                let target = rating as usize;
                let user_dev = if self.cfg.use_user_drift {
                    self.user_time_dev(u, day as i32)
                } else {
                    0.0
                };
                let side = if self.cfg.use_side_features {
                    Some(self.side_features(u, item, day as i32))
                } else {
                    None
                };
                let day_index = self.day_index(day as i32);
                let item_time_index = self.item_time_index(item, day as i32);
                let day_freq_index = self.day_freq_index(u, day as i32);
                let item_scale = self.user_item_scale_factor(u, user_dev, day_index);
                let user_item_scale_bin_index = self.user_item_scale_bin_index(u, day_index);
                let user_day_bias_bin_index = self.user_day_bias_bin_index(u, day_index);
                let user_day_scale_bin_index = self.user_day_scale_bin_index(u, day_index);
                let user_day_bias = self.user_day_bias_bin_value(u, day_index);
                let user_day_scale = self.user_day_scale_factor(u, day_index);

                let base =
                    self.base_logits_from_parts(&z, u, item, day as i32, user_dev, side.as_ref());
                let scores = base_to_scores(&base);

                let (_, grad_reg) = regular_loss_grad(&scores, target);
                let (_, grad_ord) = ordinal_loss_grad(&scores, target);
                let (_, grad_rmse) = rmse_loss_grad(&scores, target);
                let lambda = self.cfg.ordinal_lambda;
                let rmse_weight = self.cfg.rmse_weight;
                let mut grad_scores = [0.0; K];
                for t in 0..K {
                    let ordinal_grad = (1.0 - lambda) * grad_reg[t] + lambda * grad_ord[t];
                    grad_scores[t] =
                        scale * ((1.0 - rmse_weight) * ordinal_grad + rmse_weight * grad_rmse[t]);
                }

                let mut grad_base = [0.0; K];
                let mut suffix = 0.0;
                for t in (0..K).rev() {
                    suffix += grad_scores[t];
                    grad_base[t] = suffix;
                }

                let mut user_day_bias_grad = 0.0;
                let mut user_day_scale_grad = 0.0;
                for t in 0..K {
                    let item_bias =
                        self.item_bias_term(t, item, day_index, item_time_index, day_freq_index);
                    let mut raw_score = (item_scale * item_bias) as f64;
                    if self.cfg.use_user_bias {
                        raw_score += self.user_bias[[t, u]] as f64;
                    }
                    if self.cfg.use_user_drift {
                        raw_score += (self.user_drift[[t, u]] * user_dev) as f64;
                    }
                    raw_score += user_day_bias as f64;
                    for j in 0..rank {
                        raw_score += self.p[[t, item, j]] as f64 * z[j] as f64;
                    }
                    if let Some(side) = side.as_ref() {
                        for f in 0..SIDE_FEATS {
                            raw_score += (self.wide_w[[t, f]] * side[f]) as f64;
                        }
                    }

                    let raw_grad = grad_base[t] * user_day_scale;
                    let item_bias_grad = raw_grad * item_scale;
                    user_day_bias_grad += raw_grad as f64;
                    user_day_scale_grad += grad_base[t] as f64 * raw_score;
                    *grads.b[t].entry(item).or_insert(0.0) += item_bias_grad;
                    if self.cfg.use_user_bias {
                        *grads.user_bias[t].entry(u).or_insert(0.0) += raw_grad;
                    }
                    if self.cfg.use_user_drift {
                        *grads.user_drift[t].entry(u).or_insert(0.0) += raw_grad * user_dev;
                    }
                    if self.cfg.use_user_item_scale {
                        *grads.user_item_scale.entry(u).or_insert(0.0) += raw_grad * item_bias;
                        *grads.user_item_scale_drift.entry(u).or_insert(0.0) +=
                            raw_grad * item_bias * user_dev;
                        if self.cfg.use_user_item_scale_bin {
                            *grads
                                .user_item_scale_bin
                                .entry(user_item_scale_bin_index)
                                .or_insert(0.0) += raw_grad * item_bias;
                        }
                    }
                    if self.cfg.use_day_bias {
                        *grads.day_bias[t].entry(day_index).or_insert(0.0) += item_bias_grad;
                    }
                    if self.cfg.use_item_time_bias {
                        *grads.item_time_bias[t]
                            .entry(item_time_index)
                            .or_insert(0.0) += item_bias_grad;
                    }
                    if self.cfg.use_day_freq_bias {
                        *grads.day_freq_bias[t].entry(day_freq_index).or_insert(0.0) +=
                            item_bias_grad;
                    }
                    if let Some(side) = side.as_ref() {
                        for f in 0..SIDE_FEATS {
                            grads.wide_w[[t, f]] += raw_grad * side[f];
                        }
                    }

                    let row = grads.p[t].entry(item).or_insert_with(|| vec![0.0; rank]);
                    for j in 0..rank {
                        row[j] += raw_grad * z[j];
                        grad_z[j] += raw_grad as f64 * self.p[[t, item, j]] as f64;
                    }
                }
                if self.cfg.use_user_day_bias_bin {
                    *grads
                        .user_day_bias_bin
                        .entry(user_day_bias_bin_index)
                        .or_insert(0.0) += user_day_bias_grad as f32;
                }
                if self.cfg.use_user_day_scale_bin {
                    *grads
                        .user_day_scale_bin
                        .entry(user_day_scale_bin_index)
                        .or_insert(0.0) += user_day_scale_grad as f32;
                }
            }

            for j in 0..rank {
                for i in 0..top.len() {
                    grads.dec_proj[[j, i]] += (grad_z[j] * top[i] as f64) as f32;
                }
            }

            let mut grad_h = vec![0.0; hidden];
            if hidden2 > 0 {
                let mut grad_h2 = vec![0.0; hidden2];
                for i in 0..hidden2 {
                    let mut s = 0.0;
                    for j in 0..rank {
                        s += self.dec_proj[[j, i]] as f64 * grad_z[j];
                    }
                    grad_h2[i] = s * (1.0 - top[i] as f64 * top[i] as f64);
                    grads.hidden2_bias[i] += grad_h2[i] as f32;
                }
                for i in 0..hidden2 {
                    for j in 0..hidden {
                        grads.hidden2_proj[[i, j]] += (grad_h2[i] * h[j] as f64) as f32;
                        grad_h[j] += self.hidden2_proj[[i, j]] as f64 * grad_h2[i];
                    }
                }
                for i in 0..hidden {
                    grad_h[i] *= 1.0 - h[i] as f64 * h[i] as f64;
                    grads.enc_bias[i] += grad_h[i] as f32;
                }
            } else {
                for i in 0..hidden {
                    let mut s = 0.0;
                    for j in 0..rank {
                        s += self.dec_proj[[j, i]] as f64 * grad_z[j];
                    }
                    grad_h[i] = s * (1.0 - h[i] as f64 * h[i] as f64);
                    grads.enc_bias[i] += grad_h[i] as f32;
                }
            }

            for j in 0..rank {
                for i in 0..hidden {
                    grads.enc_proj[[i, j]] += (grad_h[i] * ctx_sum[j]) as f32;
                }
            }

            let mut grad_ctx_sum = vec![0.0; rank];
            for j in 0..rank {
                let mut s = 0.0;
                for i in 0..hidden {
                    s += self.enc_proj[[i, j]] as f64 * grad_h[i];
                }
                grad_ctx_sum[j] = s * ctx_scale as f64;
            }

            for ctx_day_index in 0..DAY_BINS {
                let ctx_weight =
                    self.ctx_target_day_weight(target_day_index, ctx_day_index) as f64;
                let bin_sum = &ctx_bin_sums[ctx_day_index];
                let grad_bin = &mut grad_ctx_bin_sums[ctx_day_index];
                let implicit_bin_sum = implicit_ctx_bin_sums
                    .as_ref()
                    .map(|bins| &bins[ctx_day_index]);
                let grad_implicit_bin = &mut grad_implicit_ctx_bin_sums[ctx_day_index];
                let mut weight_grad = 0.0;
                for j in 0..rank {
                    let weighted_grad = ctx_weight * grad_ctx_sum[j];
                    grad_bin[j] += weighted_grad;
                    weight_grad += grad_ctx_sum[j] * bin_sum[j];
                    if let Some(implicit_bin_sum) = implicit_bin_sum {
                        grad_implicit_bin[j] += weighted_grad;
                        weight_grad += grad_ctx_sum[j] * implicit_bin_sum[j];
                    }
                }
                if self.cfg.use_ctx_target_day_scale {
                    grads.ctx_target_day_scale[[target_day_index, ctx_day_index]] +=
                        weight_grad as f32;
                }
            }
        }

        for &oi in &order[..ctx_len] {
            let (item, rating, day) = ratings[oi];
            let ctx_day_index = self.day_index(day as i32);
            let grad_bin = &grad_ctx_bin_sums[ctx_day_index];
            for t in 0..=rating as usize {
                let row = grads.a[t].entry(item).or_insert_with(|| vec![0.0; rank]);
                for j in 0..rank {
                    row[j] += grad_bin[j] as f32;
                }
            }
        }
        if self.cfg.use_implicit_probe_ctx {
            for &(item, day) in &implicit_items {
                let ctx_day_index = self.day_index(day as i32);
                let grad_bin = &grad_implicit_ctx_bin_sums[ctx_day_index];
                let row = grads
                    .implicit_a
                    .entry(item)
                    .or_insert_with(|| vec![0.0; rank]);
                for j in 0..rank {
                    row[j] += grad_bin[j] as f32;
                }
            }
        }

        grads.count += 1;
    }

    /// Adam step (with weight decay) for a dense 2D parameter array.
    fn adam_update_dense_2d(
        param: &mut Array2<f32>,
        m: &mut Array2<f32>,
        v: &mut Array2<f32>,
        grad: &Array2<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        timestep: usize,
        batch_count: usize,
    ) {
        let bc = batch_count.max(1) as f32;
        let t = timestep as i32;
        let corr1 = 1.0 - beta1.powi(t);
        let corr2 = 1.0 - beta2.powi(t);
        for ((p, m), (v, g)) in param
            .iter_mut()
            .zip(m.iter_mut())
            .zip(v.iter_mut().zip(grad.iter()))
        {
            let g = *g / bc;
            *m = beta1 * *m + (1.0 - beta1) * g;
            *v = beta2 * *v + (1.0 - beta2) * g * g;
            let mhat = *m / corr1;
            let vhat = *v / corr2;
            *p -= lr * mhat / (vhat.sqrt() + eps);
            *p -= lr * weight_decay * *p;
        }
    }

    /// Adam step (with weight decay) for a dense 1D parameter array.
    fn adam_update_dense_1d(
        param: &mut Array1<f32>,
        m: &mut Array1<f32>,
        v: &mut Array1<f32>,
        grad: &Array1<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        timestep: usize,
        batch_count: usize,
    ) {
        let bc = batch_count.max(1) as f32;
        let t = timestep as i32;
        let corr1 = 1.0 - beta1.powi(t);
        let corr2 = 1.0 - beta2.powi(t);
        for ((p, m), (v, g)) in param
            .iter_mut()
            .zip(m.iter_mut())
            .zip(v.iter_mut().zip(grad.iter()))
        {
            let g = *g / bc;
            *m = beta1 * *m + (1.0 - beta1) * g;
            *v = beta2 * *v + (1.0 - beta2) * g * g;
            let mhat = *m / corr1;
            let vhat = *v / corr2;
            *p -= lr * mhat / (vhat.sqrt() + eps);
            *p -= lr * weight_decay * *p;
        }
    }

    /// Adam step for sparse per-(rating, item) factor rows of a 3D array (e.g. A/P).
    fn apply_sparse_row_updates(
        param: &mut Array3<f32>,
        m: &mut Array3<f32>,
        v: &mut Array3<f32>,
        grads: &[HashMap<usize, Vec<f32>>],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        timestep: usize,
        batch_count: usize,
    ) {
        let bc = batch_count.max(1) as f32;
        let t = timestep as i32;
        let corr1 = 1.0 - beta1.powi(t);
        let corr2 = 1.0 - beta2.powi(t);
        for rating in 0..K {
            for (&item, row_grad) in &grads[rating] {
                for j in 0..row_grad.len() {
                    let p = &mut param[[rating, item, j]];
                    let m_t = &mut m[[rating, item, j]];
                    let v_t = &mut v[[rating, item, j]];
                    let g = row_grad[j] / bc;
                    *m_t = beta1 * *m_t + (1.0 - beta1) * g;
                    *v_t = beta2 * *v_t + (1.0 - beta2) * g * g;
                    let mhat = *m_t / corr1;
                    let vhat = *v_t / corr2;
                    *p -= lr * mhat / (vhat.sqrt() + eps);
                    *p -= lr * weight_decay * *p;
                }
            }
        }
    }

    /// Adam step for sparse rows of a 2D matrix keyed by index (e.g. implicit_a).
    fn apply_sparse_matrix_updates(
        param: &mut Array2<f32>,
        m: &mut Array2<f32>,
        v: &mut Array2<f32>,
        grads: &HashMap<usize, Vec<f32>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        timestep: usize,
        batch_count: usize,
    ) {
        let bc = batch_count.max(1) as f32;
        let t = timestep as i32;
        let corr1 = 1.0 - beta1.powi(t);
        let corr2 = 1.0 - beta2.powi(t);
        for (&row_idx, row_grad) in grads {
            for j in 0..row_grad.len() {
                let p = &mut param[[row_idx, j]];
                let m_t = &mut m[[row_idx, j]];
                let v_t = &mut v[[row_idx, j]];
                let g = row_grad[j] / bc;
                *m_t = beta1 * *m_t + (1.0 - beta1) * g;
                *v_t = beta2 * *v_t + (1.0 - beta2) * g * g;
                let mhat = *m_t / corr1;
                let vhat = *v_t / corr2;
                *p -= lr * mhat / (vhat.sqrt() + eps);
                *p -= lr * weight_decay * *p;
            }
        }
    }

    /// Adam step (no weight decay) for sparse per-(rating, index) scalar biases in a 2D array.
    fn apply_sparse_bias_updates(
        param: &mut Array2<f32>,
        m: &mut Array2<f32>,
        v: &mut Array2<f32>,
        grads: &[HashMap<usize, f32>],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        timestep: usize,
        batch_count: usize,
    ) {
        let bc = batch_count.max(1) as f32;
        let t = timestep as i32;
        let corr1 = 1.0 - beta1.powi(t);
        let corr2 = 1.0 - beta2.powi(t);
        for rating in 0..K {
            for (&item, &g0) in &grads[rating] {
                let p = &mut param[[rating, item]];
                let m_t = &mut m[[rating, item]];
                let v_t = &mut v[[rating, item]];
                let g = g0 / bc;
                *m_t = beta1 * *m_t + (1.0 - beta1) * g;
                *v_t = beta2 * *v_t + (1.0 - beta2) * g * g;
                let mhat = *m_t / corr1;
                let vhat = *v_t / corr2;
                *p -= lr * mhat / (vhat.sqrt() + eps);
            }
        }
    }

    /// Adam step for sparse scalar entries of a 1D vector keyed by index.
    fn apply_sparse_vector_updates(
        param: &mut Array1<f32>,
        m: &mut Array1<f32>,
        v: &mut Array1<f32>,
        grads: &HashMap<usize, f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        timestep: usize,
        batch_count: usize,
    ) {
        let bc = batch_count.max(1) as f32;
        let t = timestep as i32;
        let corr1 = 1.0 - beta1.powi(t);
        let corr2 = 1.0 - beta2.powi(t);
        for (&idx, &g0) in grads {
            let p = &mut param[idx];
            let m_t = &mut m[idx];
            let v_t = &mut v[idx];
            let g = g0 / bc;
            *m_t = beta1 * *m_t + (1.0 - beta1) * g;
            *v_t = beta2 * *v_t + (1.0 - beta2) * g * g;
            let mhat = *m_t / corr1;
            let vhat = *v_t / corr2;
            *p -= lr * mhat / (vhat.sqrt() + eps);
            *p -= lr * weight_decay * *p;
        }
    }

    /// Applies one Adam optimizer step over all parameters using the accumulated batch gradients.
    fn apply_batch_with_lr(&mut self, grads: &SparseBatchGrads, lr: f32) {
        if grads.count == 0 {
            return;
        }
        self.timestep += 1;
        let cfg = self.cfg;
        let first_lr = lr * cfg.first_layer_lr_mult;

        Self::adam_update_dense_2d(
            &mut self.enc_proj,
            &mut self.m_enc_proj,
            &mut self.v_enc_proj,
            &grads.enc_proj,
            first_lr,
            cfg.beta1,
            cfg.beta2,
            cfg.epsilon,
            cfg.weight_decay,
            self.timestep,
            grads.count,
        );
        if cfg.n_hidden2 > 0 {
            Self::adam_update_dense_2d(
                &mut self.hidden2_proj,
                &mut self.m_hidden2_proj,
                &mut self.v_hidden2_proj,
                &grads.hidden2_proj,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                self.timestep,
                grads.count,
            );
            Self::adam_update_dense_1d(
                &mut self.hidden2_bias,
                &mut self.m_hidden2_bias,
                &mut self.v_hidden2_bias,
                &grads.hidden2_bias,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                0.0,
                self.timestep,
                grads.count,
            );
        }
        Self::adam_update_dense_2d(
            &mut self.dec_proj,
            &mut self.m_dec_proj,
            &mut self.v_dec_proj,
            &grads.dec_proj,
            lr,
            cfg.beta1,
            cfg.beta2,
            cfg.epsilon,
            cfg.weight_decay,
            self.timestep,
            grads.count,
        );
        if cfg.use_side_features {
            Self::adam_update_dense_2d(
                &mut self.wide_w,
                &mut self.m_wide_w,
                &mut self.v_wide_w,
                &grads.wide_w,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                self.timestep,
                grads.count,
            );
        }
        Self::adam_update_dense_1d(
            &mut self.enc_bias,
            &mut self.m_enc_bias,
            &mut self.v_enc_bias,
            &grads.enc_bias,
            lr,
            cfg.beta1,
            cfg.beta2,
            cfg.epsilon,
            0.0,
            self.timestep,
            grads.count,
        );

        Self::apply_sparse_row_updates(
            &mut self.a,
            &mut self.m_a,
            &mut self.v_a,
            &grads.a,
            first_lr,
            cfg.beta1,
            cfg.beta2,
            cfg.epsilon,
            cfg.weight_decay,
            self.timestep,
            grads.count,
        );
        Self::apply_sparse_row_updates(
            &mut self.p,
            &mut self.m_p,
            &mut self.v_p,
            &grads.p,
            lr,
            cfg.beta1,
            cfg.beta2,
            cfg.epsilon,
            cfg.weight_decay,
            self.timestep,
            grads.count,
        );
        Self::apply_sparse_bias_updates(
            &mut self.b,
            &mut self.m_b,
            &mut self.v_b,
            &grads.b,
            lr,
            cfg.beta1,
            cfg.beta2,
            cfg.epsilon,
            self.timestep,
            grads.count,
        );
        if cfg.use_user_bias {
            Self::apply_sparse_bias_updates(
                &mut self.user_bias,
                &mut self.m_user_bias,
                &mut self.v_user_bias,
                &grads.user_bias,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                self.timestep,
                grads.count,
            );
        }
        if cfg.use_user_drift {
            Self::apply_sparse_bias_updates(
                &mut self.user_drift,
                &mut self.m_user_drift,
                &mut self.v_user_drift,
                &grads.user_drift,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                self.timestep,
                grads.count,
            );
        }
        if cfg.use_user_item_scale {
            Self::apply_sparse_vector_updates(
                &mut self.user_item_scale,
                &mut self.m_user_item_scale,
                &mut self.v_user_item_scale,
                &grads.user_item_scale,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                self.timestep,
                grads.count,
            );
            Self::apply_sparse_vector_updates(
                &mut self.user_item_scale_drift,
                &mut self.m_user_item_scale_drift,
                &mut self.v_user_item_scale_drift,
                &grads.user_item_scale_drift,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                self.timestep,
                grads.count,
            );
            if cfg.use_user_item_scale_bin {
                Self::apply_sparse_vector_updates(
                    &mut self.user_item_scale_bin,
                    &mut self.m_user_item_scale_bin,
                    &mut self.v_user_item_scale_bin,
                    &grads.user_item_scale_bin,
                    lr,
                    cfg.beta1,
                    cfg.beta2,
                    cfg.epsilon,
                    cfg.weight_decay,
                    self.timestep,
                    grads.count,
                );
            }
        }
        if cfg.use_user_day_bias_bin {
            Self::apply_sparse_vector_updates(
                &mut self.user_day_bias_bin,
                &mut self.m_user_day_bias_bin,
                &mut self.v_user_day_bias_bin,
                &grads.user_day_bias_bin,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                self.timestep,
                grads.count,
            );
        }
        if cfg.use_user_day_scale_bin {
            Self::apply_sparse_vector_updates(
                &mut self.user_day_scale_bin,
                &mut self.m_user_day_scale_bin,
                &mut self.v_user_day_scale_bin,
                &grads.user_day_scale_bin,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                self.timestep,
                grads.count,
            );
        }
        if cfg.use_day_bias {
            Self::apply_sparse_bias_updates(
                &mut self.day_bias,
                &mut self.m_day_bias,
                &mut self.v_day_bias,
                &grads.day_bias,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                self.timestep,
                grads.count,
            );
        }
        if cfg.use_item_time_bias {
            Self::apply_sparse_bias_updates(
                &mut self.item_time_bias,
                &mut self.m_item_time_bias,
                &mut self.v_item_time_bias,
                &grads.item_time_bias,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                self.timestep,
                grads.count,
            );
        }
        if cfg.use_day_freq_bias {
            Self::apply_sparse_bias_updates(
                &mut self.day_freq_bias,
                &mut self.m_day_freq_bias,
                &mut self.v_day_freq_bias,
                &grads.day_freq_bias,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                self.timestep,
                grads.count,
            );
        }
        if cfg.use_ctx_target_day_scale {
            Self::adam_update_dense_2d(
                &mut self.ctx_target_day_scale,
                &mut self.m_ctx_target_day_scale,
                &mut self.v_ctx_target_day_scale,
                &grads.ctx_target_day_scale,
                lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                self.timestep,
                grads.count,
            );
        }
        if cfg.use_implicit_probe_ctx {
            Self::apply_sparse_matrix_updates(
                &mut self.implicit_a,
                &mut self.m_implicit_a,
                &mut self.v_implicit_a,
                &grads.implicit_a,
                first_lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                self.timestep,
                grads.count,
            );
        }
    }

    /// Learning rate for an epoch after applying the milestone decay schedule.
    fn effective_lr(&self, epoch: usize) -> f32 {
        let mut lr = self.cfg.lr;
        for &ms in self.cfg.milestones[..self.cfg.n_milestones].iter() {
            if epoch > ms {
                lr *= self.cfg.lr_gamma;
            }
        }
        lr
    }
}

impl Regressor for CfNadeModel {
    type Config = CfNadeConfig;

    /// Initializes parameters, Adam/SWA buffers, and per-user indexes from the datasets.
    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let sigma_item = (2.0 / cfg.rank as f32).sqrt() * 0.05;
        let sigma_proj = (2.0 / (cfg.rank + cfg.n_hidden) as f32).sqrt();
        let top_hidden = if cfg.n_hidden2 > 0 {
            cfg.n_hidden2
        } else {
            cfg.n_hidden
        };
        // Optional features are sized to 0 when disabled so their parameter and
        // Adam/SWA buffers cost nothing. The accessors that index these arrays
        // are all guarded by the same `cfg.use_*` flag, so 0-length is safe.
        let user_n = if cfg.use_user_bias { tr.n_users } else { 0 };
        let drift_n = if cfg.use_user_drift { tr.n_users } else { 0 };
        let scale_n = if cfg.use_user_item_scale { tr.n_users } else { 0 };
        let scale_bin_n = if cfg.use_user_item_scale_bin { tr.n_users * DAY_BINS } else { 0 };
        let day_bias_bin_n = if cfg.use_user_day_bias_bin { tr.n_users * DAY_BINS } else { 0 };
        let day_scale_bin_n = if cfg.use_user_day_scale_bin { tr.n_users * DAY_BINS } else { 0 };
        let item_time_n = if cfg.use_item_time_bias { tr.n_items * ITEM_TIME_BINS } else { 0 };
        let implicit_n = if cfg.use_implicit_probe_ctx { tr.n_items } else { 0 };
        crate::teeln!(
            "CF-NADE init: sampling A/P factors (2 x {} x {} x {})...",
            K, tr.n_items, cfg.rank
        );
        let a = rand_array3(K, tr.n_items, cfg.rank, &mut rng, sigma_item);
        let p = rand_array3(K, tr.n_items, cfg.rank, &mut rng, sigma_item);
        let implicit_a = Array2::zeros((implicit_n, cfg.rank));
        crate::teeln!("CF-NADE init: computing item rating-bias priors...");
        let b = init_item_biases(tr);
        let user_bias = Array2::zeros((K, user_n));
        let user_drift = Array2::zeros((K, drift_n));
        let user_item_scale = Array1::zeros(scale_n);
        let user_item_scale_drift = Array1::zeros(scale_n);
        let user_item_scale_bin = Array1::zeros(scale_bin_n);
        let user_day_bias_bin = Array1::zeros(day_bias_bin_n);
        let user_day_scale_bin = Array1::zeros(day_scale_bin_n);
        let day_bias = Array2::zeros((K, DAY_BINS));
        let item_time_bias = Array2::zeros((K, item_time_n));
        let day_freq_bias = Array2::zeros((K, FREQ_BINS));
        let ctx_target_day_scale = Array2::zeros((DAY_BINS, DAY_BINS));
        let enc_proj = rand_array2(cfg.n_hidden, cfg.rank, &mut rng, sigma_proj);
        let hidden2_proj = rand_array2(cfg.n_hidden2, cfg.n_hidden, &mut rng, sigma_proj);
        let dec_proj = rand_array2(cfg.rank, top_hidden, &mut rng, sigma_proj);
        let enc_bias = Array1::zeros(cfg.n_hidden);
        let hidden2_bias = Array1::zeros(cfg.n_hidden2);
        let wide_w = Array2::zeros((K, SIDE_FEATS));

        crate::teeln!("CF-NADE init: indexing user ratings...");
        let user_ratings = build_user_ratings(tr);
        crate::teeln!("CF-NADE init: indexing probe items for implicit context...");
        let user_probe_items = build_user_item_days(pr);
        crate::teeln!("CF-NADE init: building per-user day-frequency bins...");
        let user_day_freq_bin = build_user_day_freq_bins(tr, pr);
        crate::teeln!("CF-NADE init: computing side-feature statistics...");
        let side_stats = build_side_stats(tr);

        Self {
            cfg,
            swa_a: a.clone(),
            swa_p: p.clone(),
            swa_b: b.clone(),
            swa_user_bias: user_bias.clone(),
            swa_user_drift: user_drift.clone(),
            swa_user_item_scale: user_item_scale.clone(),
            swa_user_item_scale_drift: user_item_scale_drift.clone(),
            swa_user_item_scale_bin: user_item_scale_bin.clone(),
            swa_user_day_bias_bin: user_day_bias_bin.clone(),
            swa_user_day_scale_bin: user_day_scale_bin.clone(),
            swa_day_bias: day_bias.clone(),
            swa_item_time_bias: item_time_bias.clone(),
            swa_day_freq_bias: day_freq_bias.clone(),
            swa_ctx_target_day_scale: ctx_target_day_scale.clone(),
            swa_implicit_a: implicit_a.clone(),
            swa_enc_proj: enc_proj.clone(),
            swa_hidden2_proj: hidden2_proj.clone(),
            swa_dec_proj: dec_proj.clone(),
            swa_enc_bias: enc_bias.clone(),
            swa_hidden2_bias: hidden2_bias.clone(),
            swa_wide_w: wide_w.clone(),
            a,
            p,
            b,
            user_bias,
            user_drift,
            user_item_scale,
            user_item_scale_drift,
            user_item_scale_bin,
            user_day_bias_bin,
            user_day_scale_bin,
            day_bias,
            item_time_bias,
            day_freq_bias,
            ctx_target_day_scale,
            implicit_a,
            enc_proj,
            hidden2_proj,
            dec_proj,
            enc_bias,
            hidden2_bias,
            wide_w,
            m_a: Array3::zeros((K, tr.n_items, cfg.rank)),
            v_a: Array3::zeros((K, tr.n_items, cfg.rank)),
            m_p: Array3::zeros((K, tr.n_items, cfg.rank)),
            v_p: Array3::zeros((K, tr.n_items, cfg.rank)),
            m_b: Array2::zeros((K, tr.n_items)),
            v_b: Array2::zeros((K, tr.n_items)),
            m_user_bias: Array2::zeros((K, user_n)),
            v_user_bias: Array2::zeros((K, user_n)),
            m_user_drift: Array2::zeros((K, drift_n)),
            v_user_drift: Array2::zeros((K, drift_n)),
            m_user_item_scale: Array1::zeros(scale_n),
            v_user_item_scale: Array1::zeros(scale_n),
            m_user_item_scale_drift: Array1::zeros(scale_n),
            v_user_item_scale_drift: Array1::zeros(scale_n),
            m_user_item_scale_bin: Array1::zeros(scale_bin_n),
            v_user_item_scale_bin: Array1::zeros(scale_bin_n),
            m_user_day_bias_bin: Array1::zeros(day_bias_bin_n),
            v_user_day_bias_bin: Array1::zeros(day_bias_bin_n),
            m_user_day_scale_bin: Array1::zeros(day_scale_bin_n),
            v_user_day_scale_bin: Array1::zeros(day_scale_bin_n),
            m_day_bias: Array2::zeros((K, DAY_BINS)),
            v_day_bias: Array2::zeros((K, DAY_BINS)),
            m_item_time_bias: Array2::zeros((K, item_time_n)),
            v_item_time_bias: Array2::zeros((K, item_time_n)),
            m_day_freq_bias: Array2::zeros((K, FREQ_BINS)),
            v_day_freq_bias: Array2::zeros((K, FREQ_BINS)),
            m_ctx_target_day_scale: Array2::zeros((DAY_BINS, DAY_BINS)),
            v_ctx_target_day_scale: Array2::zeros((DAY_BINS, DAY_BINS)),
            m_implicit_a: Array2::zeros((implicit_n, cfg.rank)),
            v_implicit_a: Array2::zeros((implicit_n, cfg.rank)),
            m_enc_proj: Array2::zeros((cfg.n_hidden, cfg.rank)),
            v_enc_proj: Array2::zeros((cfg.n_hidden, cfg.rank)),
            m_hidden2_proj: Array2::zeros((cfg.n_hidden2, cfg.n_hidden)),
            v_hidden2_proj: Array2::zeros((cfg.n_hidden2, cfg.n_hidden)),
            m_dec_proj: Array2::zeros((cfg.rank, top_hidden)),
            v_dec_proj: Array2::zeros((cfg.rank, top_hidden)),
            m_enc_bias: Array1::zeros(cfg.n_hidden),
            v_enc_bias: Array1::zeros(cfg.n_hidden),
            m_hidden2_bias: Array1::zeros(cfg.n_hidden2),
            v_hidden2_bias: Array1::zeros(cfg.n_hidden2),
            m_wide_w: Array2::zeros((K, SIDE_FEATS)),
            v_wide_w: Array2::zeros((K, SIDE_FEATS)),
            user_ratings,
            user_probe_items,
            user_day_freq_bin,
            side_stats,
            pred_cache: Mutex::new((usize::MAX, usize::MAX, Array1::zeros(0))),
            rng,
            timestep: 0,
            swa_count: 0,
            swapped_in: false,
        }
    }

    /// Number of training epochs from the config.
    fn n_epochs(&self) -> usize {
        self.cfg.n_epochs
    }

    /// Runs one epoch: batched SGD over all users, then updates (and finally swaps in) the SWA model.
    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        self.clear_cache();
        // If the previous epoch swapped the SWA average in for the driver's
        // per-epoch predict/save, restore the SGD snapshot before training so we
        // keep averaging points along the SGD trajectory, not from the average.
        if self.swapped_in {
            self.swap_with_swa();
            self.swapped_in = false;
        }
        let users = get_users(
            self.user_ratings.len(),
            self.cfg.shuffle_users,
            self.cfg.seed,
            epoch,
        );
        let mut grads = SparseBatchGrads::new(self.cfg.n_hidden, self.cfg.n_hidden2, self.cfg.rank);
        let lr = self.effective_lr(epoch);

        for &u in crate::progress!(users.iter()) {
            if self.user_ratings[u].is_empty() {
                continue;
            }
            self.process_user(u, &mut grads);
            if grads.count >= self.cfg.batch_size {
                self.apply_batch_with_lr(&grads, lr);
                grads.clear();
            }
        }

        if grads.count > 0 {
            self.apply_batch_with_lr(&grads, lr);
        }

        // SWA: accumulate the running average, then swap it in as the live model
        // so the driver's post-epoch predict/save (per-epoch and final) uses the
        // averaged weights. The next epoch swaps the snapshot back before training;
        // the final epoch leaves the average in place.
        self.update_swa(epoch);
        if self.swa_count > 0 {
            self.swap_with_swa();
            self.swapped_in = true;
        }
    }

    /// Predicts a rating, reusing the cached per-user/day-bin latent z when possible.
    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let ratings = &self.user_ratings[u];
        if ratings.is_empty()
            && (!self.cfg.use_implicit_probe_ctx || self.user_probe_items[u].is_empty())
        {
            let zero_z = vec![0.0; self.cfg.rank];
            return self.predict_with_z(&zero_z, u, i, day);
        }

        let day_index = self.day_index(day);
        let mut guard = self.pred_cache.lock();
        if guard.0 != u || guard.1 != day_index || guard.2.is_empty() {
            guard.0 = u;
            guard.1 = day_index;
            guard.2 = self.encode_user_for_day(u, ratings, day);
        }
        self.predict_with_z(guard.2.as_slice().unwrap(), u, i, day)
    }
}
