// Near-verbatim copy of rbmb.rs with Bernoulli hidden units swapped for noisy ReLU
// (Nair & Hinton 2010): h_j = max(0, x_j + N(0, sqrt(sigmoid(x_j)))), whose mean is
// x*Phi(x/sigma) + sigma*phi(x/sigma). Everything else is identical to rbmb.rs — the
// two files are the worst duplication in this directory.
// Produced: rbmr-512, rbmr-2048, rbmr-4096.
// Frozen archive — see README.md; superseded by src/rx.rs HiddenType::NReLU.
use gravity::{Dataset, Regressor, get_users};
use indicatif::{ProgressBar, ProgressIterator};
use ndarray::{Array1, Array2, Array3};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use parking_lot::Mutex;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

const K: usize = 5; // rating categories 1..=5

#[derive(Clone, Copy, Debug)]
#[allow(non_snake_case)]
struct RbmConfig {
    n_hidden: usize,
    n_epochs: usize,
    seed: u64,
    shuffle_users: bool,

    batch_size: usize,
    lr: f32,
    momentum: f32,
    weight_decay: f32,

    // Per-user visible bias
    lr_bu: f32,
    wd_bu: f32,

    // Per-user-day visible bias
    lr_but: f32,
    wd_but: f32,

    // CD-k schedule
    cd_start: usize,
    cd_inc_every: usize,
    cd_inc_by: usize,
    cd_max: usize,

    // Conditional RBM: include rated/unrated vector r
    use_conditional: bool,
    // If true, include all pr-set pairs in r (train + pr). If false, only pr is_test pairs.
    r_include_pr_all: bool,

    // Speed-ups (toggleable)
    speed_up_cache_r: bool,
    speed_up_index_ratings: bool,
    speed_up_cache_hidden: bool,
    parallel: bool,
}

struct RbmModel {
    cfg: RbmConfig,
    n_users: usize,
    n_items: usize,

    // Parameters
    w: Array3<f32>,  // [item, rating, hidden]
    bv: Array2<f32>, // [item, rating]
    bh: Array1<f32>, // [hidden]
    d: Option<Array2<f32>>, // [item, hidden]
    bu: Array2<f32>, // [user, K]
    but: Vec<Vec<[f32; K]>>,  // [user][day_idx][k]

    // Momentum buffers
    mw: Array3<f32>,
    mbv: Array2<f32>,
    mbh: Array1<f32>,
    md: Option<Array2<f32>>,
    mbu: Array2<f32>, // [user, K]
    mbut: Vec<Vec<[f32; K]>>, // [user][day_idx][k]

    // Training data by user: (item, rating, day_idx)
    user_ratings: Vec<Vec<(usize, u8, usize)>>,
    // Sorted distinct days per user (for day_idx lookup)
    user_days: Vec<Vec<i16>>,
    // Rated/unrated items (train + pr test pairs)
    user_r_items: Option<Vec<Vec<usize>>>,
    ratings_zero_based: bool,

    // Cached hidden means for prediction
    pred_cache: Option<Mutex<Vec<Option<Vec<f32>>>>>,

    rng: StdRng,
}

struct ThreadGrads {
    w: Array3<f32>,
    bv: Array2<f32>,
    bh: Array1<f32>,
    d: Option<Array2<f32>>,
}

impl ThreadGrads {
    fn new(n_items: usize, n_hidden: usize, use_d: bool) -> Self {
        Self {
            w: Array3::zeros((n_items, K, n_hidden)),
            bv: Array2::zeros((n_items, K)),
            bh: Array1::zeros(n_hidden),
            d: if use_d { Some(Array2::zeros((n_items, n_hidden))) } else { None },
        }
    }

    fn zero(&mut self) {
        self.w.fill(0.0);
        self.bv.fill(0.0);
        self.bh.fill(0.0);
        if let Some(d) = self.d.as_mut() { d.fill(0.0); }
    }
}

impl RbmModel {
    fn cd_steps_for_epoch(&self, epoch: usize) -> usize {
        if self.cfg.cd_inc_every == 0 {
            return self.cfg.cd_start.max(1);
        }
        let incs = (epoch.saturating_sub(1)) / self.cfg.cd_inc_every;
        let steps = self.cfg.cd_start + incs * self.cfg.cd_inc_by;
        steps.min(self.cfg.cd_max).max(1)
    }

    fn zero_grads(
        grad_w: &mut Array3<f32>,
        grad_bv: &mut Array2<f32>,
        grad_bh: &mut Array1<f32>,
        grad_d: &mut Option<Array2<f32>>,
    ) {
        grad_w.fill(0.0);
        grad_bv.fill(0.0);
        grad_bh.fill(0.0);
        if let Some(d) = grad_d.as_mut() {
            d.fill(0.0);
        }
    }

    fn apply_grads(
        &mut self,
        batch_count: f32,
        grad_w: &Array3<f32>,
        grad_bv: &Array2<f32>,
        grad_bh: &Array1<f32>,
        grad_d: &Option<Array2<f32>>,
    ) {
        let lr = self.cfg.lr / batch_count.max(1.0);
        let mom = self.cfg.momentum;
        let wd = self.cfg.weight_decay;

        // W with weight decay
        for ((w, mw), gw) in self.w.iter_mut().zip(self.mw.iter_mut()).zip(grad_w.iter()) {
            *mw = mom * *mw + lr * (*gw - wd * *w);
            *w += *mw;
        }
        // Visible bias
        for ((b, mb), gb) in self.bv.iter_mut().zip(self.mbv.iter_mut()).zip(grad_bv.iter()) {
            *mb = mom * *mb + lr * *gb;
            *b += *mb;
        }
        // Hidden bias
        for ((b, mb), gb) in self.bh.iter_mut().zip(self.mbh.iter_mut()).zip(grad_bh.iter()) {
            *mb = mom * *mb + lr * *gb;
            *b += *mb;
        }
        // Conditional D with weight decay
        if let (Some(d), Some(md), Some(gd)) = (self.d.as_mut(), self.md.as_mut(), grad_d.as_ref()) {
            for ((w, mw), gw) in d.iter_mut().zip(md.iter_mut()).zip(gd.iter()) {
                *mw = mom * *mw + lr * (*gw - wd * *w);
                *w += *mw;
            }
        }
    }

    fn apply_bu_grad(&mut self, u: usize, grad_bu: &[f32; K]) {
        let mom = self.cfg.momentum;
        let lr_bu = self.cfg.lr_bu;
        let wd_bu = self.cfg.wd_bu;
        for k in 0..K {
            let m = mom * self.mbu[[u, k]] + lr_bu * (grad_bu[k] - wd_bu * self.bu[[u, k]]);
            self.mbu[[u, k]] = m;
            self.bu[[u, k]] += m;
        }
    }

    fn apply_but_grad(&mut self, u: usize, grad_but: &[[f32; K]]) {
        let mom = self.cfg.momentum;
        let lr_but = self.cfg.lr_but;
        let wd_but = self.cfg.wd_but;
        for (di, g) in grad_but.iter().enumerate() {
            for k in 0..K {
                let m = mom * self.mbut[u][di][k] + lr_but * (g[k] - wd_but * self.but[u][di][k]);
                self.mbut[u][di][k] = m;
                self.but[u][di][k] += m;
            }
        }
    }

    fn clear_pred_cache(&self) {
        if let Some(cache) = &self.pred_cache {
            let mut guard = cache.lock();
            for slot in guard.iter_mut() {
                *slot = None;
            }
        }
    }

    /// Compute effective bias bu[u,k] + but[u,day_idx,k] for each of this user's distinct days
    fn effective_bu(&self, u: usize) -> Vec<[f32; K]> {
        let n_days = self.user_days[u].len();
        let mut eff = Vec::with_capacity(n_days);
        for di in 0..n_days {
            let mut eb = [0.0f32; K];
            for k in 0..K {
                eb[k] = self.bu[[u, k]] + self.but[u][di][k];
            }
            eff.push(eb);
        }
        eff
    }

    /// Compute effective bias for a single day (for prediction)
    fn effective_bu_day(&self, u: usize, day: i16) -> [f32; K] {
        let mut eb = [0.0f32; K];
        for k in 0..K {
            eb[k] = self.bu[[u, k]];
        }
        if let Ok(di) = self.user_days[u].binary_search(&day) {
            for k in 0..K {
                eb[k] += self.but[u][di][k];
            }
        }
        eb
    }

    fn fit_epoch_sequential(&mut self, epoch: usize) {
        let users = get_users(self.n_users, self.cfg.shuffle_users, self.cfg.seed, epoch);
        let cd_steps = self.cd_steps_for_epoch(epoch);

        let mut grad_w = Array3::<f32>::zeros((self.n_items, K, self.cfg.n_hidden));
        let mut grad_bv = Array2::<f32>::zeros((self.n_items, K));
        let mut grad_bh = Array1::<f32>::zeros(self.cfg.n_hidden);
        let mut grad_d = self.d.as_ref().map(|_| Array2::<f32>::zeros((self.n_items, self.cfg.n_hidden)));

        let mut batch_count = 0usize;

        for &u in users.iter().progress() {
            let ratings = &self.user_ratings[u];
            if ratings.is_empty() { continue; }

            let r_items = self.user_r_items.as_ref().map(|v| v[u].as_slice());
            let r_contrib = if self.cfg.speed_up_cache_r {
                compute_r_contrib(r_items, self.d.as_ref(), self.cfg.n_hidden)
            } else {
                None
            };
            let eff_bu = self.effective_bu(u);
            let n_days = self.user_days[u].len();
            let (grad_bu, grad_but) = accumulate_user_grad(
                ratings,
                r_items,
                cd_steps,
                self.cfg.n_hidden,
                &self.w,
                &self.bv,
                &self.bh,
                self.d.as_ref(),
                &eff_bu,
                n_days,
                &mut self.rng,
                self.ratings_zero_based,
                r_contrib.as_deref(),
                &mut grad_w,
                &mut grad_bv,
                &mut grad_bh,
                &mut grad_d,
            );

            // Per-user bu/but update (not batched)
            self.apply_bu_grad(u, &grad_bu);
            self.apply_but_grad(u, &grad_but);

            batch_count += 1;
            if batch_count >= self.cfg.batch_size {
                self.apply_grads(batch_count as f32, &grad_w, &grad_bv, &grad_bh, &grad_d);
                RbmModel::zero_grads(&mut grad_w, &mut grad_bv, &mut grad_bh, &mut grad_d);
                batch_count = 0;
            }
        }

        if batch_count > 0 {
            self.apply_grads(batch_count as f32, &grad_w, &grad_bv, &grad_bh, &grad_d);
        }
    }

    fn fit_epoch_parallel(&mut self, epoch: usize) {
        let users = get_users(self.n_users, self.cfg.shuffle_users, self.cfg.seed, epoch);
        let cd_steps = self.cd_steps_for_epoch(epoch);

        // Pre-filter to non-empty users, preserving shuffled order
        let active_users: Vec<usize> = users.iter()
            .filter(|&&u| !self.user_ratings[u].is_empty())
            .copied()
            .collect();

        let n_threads = rayon::current_num_threads();
        let mut tl: Vec<ThreadGrads> = (0..n_threads)
            .map(|_| ThreadGrads::new(self.n_items, self.cfg.n_hidden, self.d.is_some()))
            .collect();

        // Copy scalars to avoid borrowing self in closures
        let n_hidden = self.cfg.n_hidden;
        let cache_r = self.cfg.speed_up_cache_r;
        let seed = self.cfg.seed;
        let ratings_zero_based = self.ratings_zero_based;
        let batch_size = self.cfg.batch_size;

        let pb = ProgressBar::new(active_users.len() as u64);
        for (batch_idx, batch) in active_users.chunks(batch_size).enumerate() {
            for tg in tl.iter_mut() { tg.zero(); }

            let chunk_size = batch.len().div_ceil(n_threads).max(1);
            let batch_chunks: Vec<&[usize]> = batch.chunks(chunk_size).collect();
            let n_chunks = batch_chunks.len();

            // Collect per-user bu/but gradients from each thread
            let user_grads: Vec<Vec<(usize, [f32; K], Vec<[f32; K]>)>>;
            {
                let w = &self.w;
                let bv = &self.bv;
                let bh = &self.bh;
                let d = self.d.as_ref();
                let user_ratings = &self.user_ratings;
                let user_r_items = &self.user_r_items;
                let bu = &self.bu;
                let but = &self.but;
                let user_days = &self.user_days;

                user_grads = batch_chunks.into_par_iter()
                    .zip(tl[..n_chunks].par_iter_mut())
                    .enumerate()
                    .map(|(tid, (chunk, tg))| {
                        let rng_seed = seed
                            ^ (epoch as u64).wrapping_mul(0x9E3779B97F4A7C15)
                            ^ (batch_idx as u64).wrapping_mul(0x517CC1B727220A95)
                            ^ (tid as u64).wrapping_mul(0x6C62272E07BB0142);
                        let mut rng = StdRng::seed_from_u64(rng_seed);
                        let mut ug = Vec::with_capacity(chunk.len());
                        for &u in chunk {
                            let ratings = &user_ratings[u];
                            let r_items = user_r_items.as_ref().map(|v| v[u].as_slice());
                            let r_contrib = if cache_r {
                                compute_r_contrib(r_items, d, n_hidden)
                            } else {
                                None
                            };
                            // Compute effective bias per day
                            let n_days = user_days[u].len();
                            let mut eff_bu = Vec::with_capacity(n_days);
                            for di in 0..n_days {
                                let mut eb = [0.0f32; K];
                                for k in 0..K {
                                    eb[k] = bu[[u, k]] + but[u][di][k];
                                }
                                eff_bu.push(eb);
                            }
                            let (grad_bu, grad_but) = accumulate_user_grad(
                                ratings, r_items, cd_steps, n_hidden,
                                w, bv, bh, d, &eff_bu, n_days,
                                &mut rng, ratings_zero_based,
                                r_contrib.as_deref(),
                                &mut tg.w, &mut tg.bv, &mut tg.bh, &mut tg.d,
                            );
                            ug.push((u, grad_bu, grad_but));
                        }
                        ug
                    })
                    .collect();
            }

            // Reduce: sum thread-local gradients into first buffer
            {
                let (first, rest) = tl[..n_chunks].split_first_mut().unwrap();
                for tg in rest.iter() {
                    first.w += &tg.w;
                    first.bv += &tg.bv;
                    first.bh += &tg.bh;
                    if let (Some(d0), Some(dt)) = (first.d.as_mut(), tg.d.as_ref()) {
                        *d0 += dt;
                    }
                }
            }

            self.apply_grads(batch.len() as f32, &tl[0].w, &tl[0].bv, &tl[0].bh, &tl[0].d);

            // Apply per-user bu/but updates sequentially
            for thread_ug in &user_grads {
                for (u, grad_bu, grad_but) in thread_ug {
                    self.apply_bu_grad(*u, grad_bu);
                    self.apply_but_grad(*u, grad_but);
                }
            }

            pb.inc(batch.len() as u64);
        }
        pb.finish_and_clear();
    }

    fn get_hidden_means(&self, u: usize) -> Vec<f32> {
        let ratings = &self.user_ratings[u];
        let r_items = self.user_r_items.as_ref().map(|v| v[u].as_slice());
        if let Some(cache) = &self.pred_cache {
            let mut guard = cache.lock();
            if let Some(hm) = &guard[u] {
                return hm.clone();
            }
            let v_state: Vec<usize> = if self.ratings_zero_based {
                ratings.iter().map(|(_, r, _)| *r as usize).collect()
            } else {
                ratings.iter().map(|(_, r, _)| (*r as usize) - 1).collect()
            };
            let r_contrib = if self.cfg.speed_up_cache_r {
                compute_r_contrib(r_items, self.d.as_ref(), self.cfg.n_hidden)
            } else {
                None
            };
            let acts = hidden_acts_from_state(
                &self.bh, &self.w, self.d.as_ref(), ratings, &v_state,
                r_items, r_contrib.as_deref(), self.cfg.n_hidden,
            );
            let means = hidden_means_from_acts(&acts);
            guard[u] = Some(means.clone());
            return means;
        }

        let v_state: Vec<usize> = if self.ratings_zero_based {
            ratings.iter().map(|(_, r, _)| *r as usize).collect()
        } else {
            ratings.iter().map(|(_, r, _)| (*r as usize) - 1).collect()
        };
        let r_contrib = if self.cfg.speed_up_cache_r {
            compute_r_contrib(r_items, self.d.as_ref(), self.cfg.n_hidden)
        } else {
            None
        };
        let acts = hidden_acts_from_state(
            &self.bh, &self.w, self.d.as_ref(), ratings, &v_state,
            r_items, r_contrib.as_deref(), self.cfg.n_hidden,
        );
        hidden_means_from_acts(&acts)
    }

    fn predict_probs(&self, u: usize, i: usize, day: i16) -> Array1<f32> {
        let h_mean = self.get_hidden_means(u);
        let eff_bu = self.effective_bu_day(u, day);
        probs_with_ph(&self.bv, &self.w, i, &h_mean, self.cfg.n_hidden, &eff_bu)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi)
fn norm_pdf(x: f32) -> f32 {
    const INV_SQRT_2PI: f32 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF using Abramowitz & Stegun 7.1.26 erf approximation
fn norm_cdf(x: f32) -> f32 {
    // erf approximation: max error ~1.5e-7
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let erf = 1.0 - (0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3
        - 1.453152027 * t4 + 1.061405429 * t5) * (-a * a).exp();
    let cdf = 0.5 * (1.0 + erf);
    if x >= 0.0 { cdf } else { 1.0 - cdf }
}

/// Expected value of max(0, x + N(0, sigma)) where sigma = sqrt(sigmoid(x))
/// = x * Phi(x/sigma) + sigma * phi(x/sigma)
fn nrelu_mean_scalar(x: f32) -> f32 {
    let sig = sigmoid(x);
    let sigma = sig.sqrt();
    if sigma < 1e-10 {
        return x.max(0.0);
    }
    let z = x / sigma;
    x * norm_cdf(z) + sigma * norm_pdf(z)
}

/// Compute raw pre-activations (no sigmoid) for hidden units given visible state
fn hidden_acts_from_state(
    bh: &Array1<f32>,
    w: &Array3<f32>,
    d: Option<&Array2<f32>>,
    items: &[(usize, u8, usize)],
    v_state: &[usize],
    r_items: Option<&[usize]>,
    r_contrib: Option<&[f32]>,
    n_hidden: usize,
) -> Vec<f32> {
    let mut act = bh.to_vec();

    for (idx, (item, _, _)) in items.iter().enumerate() {
        let k = v_state[idx];
        for j in 0..n_hidden {
            act[j] += w[[*item, k, j]];
        }
    }

    if let Some(rc) = r_contrib {
        for j in 0..n_hidden {
            act[j] += rc[j];
        }
    } else if let (Some(r_items), Some(d)) = (r_items, d) {
        for &item in r_items {
            for j in 0..n_hidden {
                act[j] += d[[item, j]];
            }
        }
    }

    // No sigmoid — return raw pre-activations
    act
}

/// Compute E[max(0, x + N(0, sqrt(sigmoid(x))))] for each pre-activation
fn hidden_means_from_acts(acts: &[f32]) -> Vec<f32> {
    acts.iter().map(|&x| nrelu_mean_scalar(x)).collect()
}

/// Sample noisy ReLU: h_j = max(0, act_j + N(0, sqrt(sigmoid(act_j))))
fn sample_nrelu(rng: &mut StdRng, acts: &[f32]) -> Vec<f32> {
    let std_normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut h = Vec::with_capacity(acts.len());
    for &a in acts {
        let sigma = sigmoid(a).sqrt();
        let noise = std_normal.sample(rng) * sigma;
        h.push((a + noise).max(0.0));
    }
    h
}

fn sample_softmax(rng: &mut StdRng, logits: [f32; K]) -> usize {
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v > max { max = v; }
    }
    let mut exps = [0.0f32; K];
    let mut sum = 0.0f32;
    for k in 0..K {
        let e = (logits[k] - max).exp();
        exps[k] = e;
        sum += e;
    }
    let mut r = rng.random::<f32>() * sum;
    for k in 0..K {
        if r <= exps[k] {
            return k;
        }
        r -= exps[k];
    }
    K - 1
}

fn item_logits(bv: &Array2<f32>, w: &Array3<f32>, item: usize, h: &[f32], n_hidden: usize, bu_eff: &[f32]) -> [f32; K] {
    let mut logits = [0.0f32; K];
    for k in 0..K {
        let mut s = bv[[item, k]] + bu_eff[k];
        for j in 0..n_hidden {
            s += h[j] * w[[item, k, j]];
        }
        logits[k] = s;
    }
    logits
}

fn compute_r_contrib(
    r_items: Option<&[usize]>,
    d: Option<&Array2<f32>>,
    n_hidden: usize,
) -> Option<Vec<f32>> {
    let r_items = r_items?;
    let d = d?;
    let mut rc = vec![0.0f32; n_hidden];
    for &item in r_items {
        for j in 0..n_hidden {
            rc[j] += d[[item, j]];
        }
    }
    Some(rc)
}

/// Returns (grad_bu [K], grad_but [n_days][K])
fn accumulate_user_grad(
    ratings: &[(usize, u8, usize)],
    r_items: Option<&[usize]>,
    cd_steps: usize,
    n_hidden: usize,
    w: &Array3<f32>,
    bv: &Array2<f32>,
    bh: &Array1<f32>,
    d: Option<&Array2<f32>>,
    eff_bu: &[[f32; K]],  // [day_idx] -> effective bu+but per k
    n_days: usize,
    rng: &mut StdRng,
    ratings_zero_based: bool,
    r_contrib: Option<&[f32]>,
    grad_w: &mut Array3<f32>,
    grad_bv: &mut Array2<f32>,
    grad_bh: &mut Array1<f32>,
    grad_d: &mut Option<Array2<f32>>,
) -> ([f32; K], Vec<[f32; K]>) {
    let mut grad_bu = [0.0f32; K];
    let mut grad_but = vec![[0.0f32; K]; n_days];
    if ratings.is_empty() { return (grad_bu, grad_but); }

    let mut v_state: Vec<usize> = if ratings_zero_based {
        ratings.iter().map(|(_, r, _)| *r as usize).collect()
    } else {
        ratings.iter().map(|(_, r, _)| (*r as usize) - 1).collect()
    };

    // Positive phase: compute acts and means from data
    let act_pos = hidden_acts_from_state(bh, w, d, ratings, &v_state, r_items, r_contrib, n_hidden);
    let mean_pos = hidden_means_from_acts(&act_pos);

    // CD loop using noisy ReLU sampling
    let mut act = act_pos.clone();
    for _ in 0..cd_steps {
        let h_sample = sample_nrelu(rng, &act);
        for (idx, (item, _, day_idx)) in ratings.iter().enumerate() {
            let logits = item_logits(bv, w, *item, &h_sample, n_hidden, &eff_bu[*day_idx]);
            let k = sample_softmax(rng, logits);
            v_state[idx] = k;
        }
        act = hidden_acts_from_state(bh, w, d, ratings, &v_state, r_items, r_contrib, n_hidden);
    }
    let mean_neg = hidden_means_from_acts(&act);

    for (idx, (item, rating, day_idx)) in ratings.iter().enumerate() {
        let k_data = if ratings_zero_based { *rating as usize } else { (*rating as usize) - 1 };
        let k_model = v_state[idx];
        for j in 0..n_hidden {
            grad_w[[*item, k_data, j]] += mean_pos[j];
            grad_w[[*item, k_model, j]] -= mean_neg[j];
        }
        grad_bv[[*item, k_data]] += 1.0;
        grad_bv[[*item, k_model]] -= 1.0;

        // Per-user visible bias gradient
        grad_bu[k_data] += 1.0;
        grad_bu[k_model] -= 1.0;

        // Per-user-day visible bias gradient
        grad_but[*day_idx][k_data] += 1.0;
        grad_but[*day_idx][k_model] -= 1.0;
    }

    for j in 0..n_hidden {
        grad_bh[j] += mean_pos[j] - mean_neg[j];
    }

    if let (Some(r_items), Some(gd)) = (r_items, grad_d.as_mut()) {
        for &item in r_items {
            for j in 0..n_hidden {
                gd[[item, j]] += mean_pos[j] - mean_neg[j];
            }
        }
    }

    (grad_bu, grad_but)
}

fn build_user_ratings(ds: &Dataset, zero_based: bool) -> (Vec<Vec<(usize, u8, usize)>>, Vec<Vec<i16>>) {
    // First pass: collect distinct days per user
    let mut user_day_set: Vec<Vec<i16>> = vec![Vec::new(); ds.n_users];
    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 { continue; }
        let u = ds.user_idxs[idx] as usize;
        user_day_set[u].push(ds.dates[idx]);
    }
    for days in user_day_set.iter_mut() {
        days.sort_unstable();
        days.dedup();
    }

    // Second pass: build ratings with day indices
    let mut ratings: Vec<Vec<(usize, u8, usize)>> = (0..ds.n_users)
        .map(|u| Vec::with_capacity(ds.user_cnts[u] as usize))
        .collect();
    for idx in 0..ds.n_ratings {
        if ds.is_test[idx] != 0 { continue; }
        let u = ds.user_idxs[idx] as usize;
        let i = ds.item_idxs[idx] as usize;
        let mut r = ds.raw_ratings[idx] as u8;
        if zero_based { r = r.saturating_sub(1); }
        let day = ds.dates[idx];
        let di = user_day_set[u].binary_search(&day).unwrap();
        ratings[u].push((i, r, di));
    }

    (ratings, user_day_set)
}

fn build_r_items(tr: &Dataset, pr: &Dataset, include_pr_all: bool) -> Vec<Vec<usize>> {
    let mut items: Vec<Vec<usize>> = vec![Vec::new(); tr.n_users];

    for idx in 0..tr.n_ratings {
        let u = tr.user_idxs[idx] as usize;
        let i = tr.item_idxs[idx] as usize;
        items[u].push(i);
    }

    for idx in 0..pr.n_ratings {
        if !include_pr_all && pr.is_test[idx] == 0 { continue; }
        let u = pr.user_idxs[idx] as usize;
        let i = pr.item_idxs[idx] as usize;
        items[u].push(i);
    }

    for u in 0..items.len() {
        items[u].sort_unstable();
        items[u].dedup();
    }

    items
}

fn init_visible_biases(tr: &Dataset) -> Array2<f32> {
    let mut counts = vec![[0u32; K]; tr.n_items];
    let mut totals = vec![0u32; tr.n_items];
    let mut global = [0u64; K];
    let mut global_total = 0u64;

    for idx in 0..tr.n_ratings {
        if tr.is_test[idx] != 0 { continue; }
        let i = tr.item_idxs[idx] as usize;
        let r = tr.raw_ratings[idx] as usize;
        let k = r.saturating_sub(1).min(K - 1);
        counts[i][k] += 1;
        totals[i] += 1;
        global[k] += 1;
        global_total += 1;
    }

    let eps = 1e-6f32;
    let mut bv = Array2::<f32>::zeros((tr.n_items, K));
    for i in 0..tr.n_items {
        let tot_i = totals[i] as f32;
        let (use_counts, tot) = if tot_i > 0.0 {
            (counts[i], tot_i)
        } else {
            let mut tmp = [0u32; K];
            for k in 0..K { tmp[k] = global[k] as u32; }
            (tmp, global_total as f32)
        };
        for k in 0..K {
            let c = use_counts[k] as f32;
            let p = (c + eps) / (tot + (K as f32) * eps);
            bv[[i, k]] = p.ln();
        }
    }
    bv
}

fn rand_array3(rows: usize, cols: usize, depth: usize, rng: &mut StdRng, sigma: f32) -> Array3<f32> {
    let dist = Normal::<f32>::new(0.0, sigma).unwrap();
    Array3::from_shape_fn((rows, cols, depth), |_| dist.sample(rng))
}

fn rand_array2(rows: usize, cols: usize, rng: &mut StdRng, sigma: f32) -> Array2<f32> {
    let dist = Normal::<f32>::new(0.0, sigma).unwrap();
    Array2::from_shape_fn((rows, cols), |_| dist.sample(rng))
}

impl Regressor for RbmModel {
    type Config = RbmConfig;

    fn new(tr: &Dataset, pr: &Dataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        let w = rand_array3(tr.n_items, K, cfg.n_hidden, &mut rng, 0.001);
        let bv = init_visible_biases(tr);
        let bh = Array1::<f32>::zeros(cfg.n_hidden);

        let d = if cfg.use_conditional {
            Some(rand_array2(tr.n_items, cfg.n_hidden, &mut rng, 0.001))
        } else {
            None
        };

        let mw = Array3::<f32>::zeros((tr.n_items, K, cfg.n_hidden));
        let mbv = Array2::<f32>::zeros((tr.n_items, K));
        let mbh = Array1::<f32>::zeros(cfg.n_hidden);
        let md = d.as_ref().map(|_| Array2::<f32>::zeros((tr.n_items, cfg.n_hidden)));

        let bu = Array2::<f32>::zeros((tr.n_users, K));
        let mbu = Array2::<f32>::zeros((tr.n_users, K));

        let (user_ratings, user_days) = build_user_ratings(tr, cfg.speed_up_index_ratings);

        // Init but/mbut: zero for each (user, day_idx) pair
        let but: Vec<Vec<[f32; K]>> = user_days.iter()
            .map(|days| vec![[0.0f32; K]; days.len()])
            .collect();
        let mbut: Vec<Vec<[f32; K]>> = user_days.iter()
            .map(|days| vec![[0.0f32; K]; days.len()])
            .collect();

        let user_r_items = if cfg.use_conditional {
            Some(build_r_items(tr, pr, cfg.r_include_pr_all))
        } else {
            None
        };
        let pred_cache = if cfg.speed_up_cache_hidden {
            Some(Mutex::new(vec![None; tr.n_users]))
        } else {
            None
        };

        Self {
            cfg,
            n_users: tr.n_users,
            n_items: tr.n_items,
            w,
            bv,
            bh,
            d,
            bu,
            but,
            mw,
            mbv,
            mbh,
            md,
            mbu,
            mbut,
            user_ratings,
            user_days,
            user_r_items,
            ratings_zero_based: cfg.speed_up_index_ratings,
            pred_cache,
            rng,
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &Dataset, epoch: usize) {
        if self.cfg.speed_up_cache_hidden {
            self.clear_pred_cache();
        }
        if self.cfg.parallel {
            self.fit_epoch_parallel(epoch);
        } else {
            self.fit_epoch_sequential(epoch);
        }
    }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let h_mean = self.get_hidden_means(u);
        let eff_bu = self.effective_bu_day(u, day as i16);
        predict_with_ph(&self.bv, &self.w, i, &h_mean, self.cfg.n_hidden, &eff_bu)
    }

    fn n_subscores(&self) -> usize { K }

    fn predict_subscores(&self, u: usize, i: usize, day: i32) -> Array1<f32> {
        self.predict_probs(u, i, day as i16)
    }
}

fn predict_with_ph(bv: &Array2<f32>, w: &Array3<f32>, i: usize, p_h: &[f32], n_hidden: usize, bu_eff: &[f32]) -> f32 {
    let logits = compute_logits(bv, w, i, p_h, n_hidden, bu_eff);
    let probs = softmax(&logits);
    let mut exp_rating = 0.0f32;
    for k in 0..K {
        exp_rating += (k as f32 + 1.0) * probs[k];
    }
    exp_rating
}

fn probs_with_ph(bv: &Array2<f32>, w: &Array3<f32>, i: usize, p_h: &[f32], n_hidden: usize, bu_eff: &[f32]) -> Array1<f32> {
    let logits = compute_logits(bv, w, i, p_h, n_hidden, bu_eff);
    let probs = softmax(&logits);
    Array1::from_vec(probs.to_vec())
}

fn compute_logits(bv: &Array2<f32>, w: &Array3<f32>, i: usize, p_h: &[f32], n_hidden: usize, bu_eff: &[f32]) -> [f32; K] {
    let mut logits = [0.0f32; K];
    for k in 0..K {
        let mut s = bv[[i, k]] + bu_eff[k];
        for j in 0..n_hidden {
            s += p_h[j] * w[[i, k, j]];
        }
        logits[k] = s;
    }
    logits
}

fn softmax(logits: &[f32; K]) -> [f32; K] {
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter() { if v > max { max = v; } }
    let mut exps = [0.0f32; K];
    let mut sum = 0.0f32;
    for k in 0..K {
        let e = (logits[k] - max).exp();
        exps[k] = e;
        sum += e;
    }
    for k in 0..K {
        exps[k] /= sum;
    }
    exps
}

fn main() {
    let cfg = RbmConfig {
        n_hidden: 32,
        n_epochs: 30,
        seed: 128,
        shuffle_users: true,

        batch_size: 500,
        lr: 0.005,
        momentum: 0.9,
        weight_decay: 0.001,

        lr_bu: 0.002,
        wd_bu: 0.01,

        lr_but: 0.001,
        wd_but: 0.01,

        cd_start: 1,
        cd_inc_every: 5,
        cd_inc_by: 1,
        cd_max: 5,

        use_conditional: true,
        r_include_pr_all: true,

        speed_up_cache_r: true,
        speed_up_index_ratings: true,
        speed_up_cache_hidden: true,
        parallel: false,
    };

    // Snapshot of one run. Note this calls the low-level `fit` (a single tr -> pr pass)
    // against the long-gone train8/probe8 split, not `fit2` — so it never produced a
    // qual prediction. The shipped rbmr-* predictors came from `fit2` runs.
    gravity::fit::<RbmModel>(
        cfg,
        "rtg",      // target
        "train8",   // tr_set
        "probe8",   // pr_set
        "rbmr",     // model_name
        false,      // save_subscores
        false,      // save_train
        false,      // save_probe_each_epoch
        "preds",    // preds_dir
        false,      // transpose
    );
}
