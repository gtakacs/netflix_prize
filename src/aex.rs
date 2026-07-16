// Autoencoder model with unified MSE loss.
// Loss = (expected_value + ubias + udbias - target)²
// Gradient flows through softmax → logits → W_dec → h → W_enc.

use crate::{
    Dataset, MaskedDataset, Regressor,
    calc_user_offsets, get_users, rand_array2, sigmoid,
};
use crate::tx::SparseUD;
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct AexConfig {
    pub n_feat: usize,       // Hidden dimension
    pub n_epochs: usize,
    pub seed: u64,
    pub shuffle_users: bool,
    pub lr: f32,             // Learning rate for W_enc, W_dec, b_enc, b_dec
    pub reg: f32,            // L2 regularization
    pub sigma: f32,          // Init std dev
    pub normalize: bool,     // Normalize encoder sum by 1/sqrt(|rated|)
    pub use_implicit: bool,  // Add implicit feedback (sum of y[i] for all rated items)
    pub dropout: f32,        // Input dropout rate for denoising (0.0 = no dropout)
    pub lr_ubias: f32,       // Learning rate for user bias (0.0 = disabled)
    pub lr_udbias: f32,      // Learning rate for user×day bias (0.0 = disabled)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct AexModel {
    cfg: AexConfig,
    w_enc: Array2<f32>,  // (n_items * 5, n_feat) - encoder weights
    b_enc: Array1<f32>,  // (n_feat,) - encoder bias
    w_dec: Array2<f32>,  // (n_feat, n_items * 5) - decoder weights
    b_dec: Array2<f32>,  // (n_items, 5) - decoder bias
    y_enc: Array2<f32>,  // (n_items, n_feat) - implicit feedback weights
    hcache: Array2<f32>, // (n_users, n_feat) - cached hidden activations
    ubias: Array1<f32>,  // (n_users,) - user bias
    udbias: Vec<f32>,    // (n_ud,) - user×day bias, indexed by SparseUD
    ud: SparseUD,        // Sparse (user, day) index
    rng: StdRng,
}

impl AexModel {
    /// Encoder: sum W_enc rows for rated items + implicit feedback, add bias, sigmoid
    fn encode_masked(&self, tr: &Dataset, start: usize, end: usize, mask: Option<&[bool]>) -> Array1<f32> {
        let mut h = self.b_enc.clone();
        let mut cnt = 0usize;
        for t in start..end {
            if let Some(m) = mask {
                if !m[t - start] { continue; }
            }
            let i = tr.item_idxs[t] as usize;
            let k = tr.raw_ratings[t] as usize - 1;
            let row = i * 5 + k;
            for f in 0..self.cfg.n_feat {
                h[f] += self.w_enc[[row, f]];
            }
            if self.cfg.use_implicit {
                for f in 0..self.cfg.n_feat {
                    h[f] += self.y_enc[[i, f]];
                }
            }
            cnt += 1;
        }
        if self.cfg.normalize && cnt > 0 {
            let scale = 1.0 / (cnt as f32).sqrt();
            h *= scale;
        }
        h.mapv(sigmoid)
    }

    fn encode(&self, tr: &Dataset, start: usize, end: usize) -> Array1<f32> {
        self.encode_masked(tr, start, end, None)
    }

    /// Decoder: compute logits for a single item, return softmax probs
    fn decode_item(&self, h: ArrayView1<f32>, i: usize) -> [f32; 5] {
        let mut logits = [0.0f32; 5];
        for k in 0..5 {
            let col = i * 5 + k;
            let mut s = self.b_dec[[i, k]];
            for f in 0..self.cfg.n_feat {
                s += h[f] * self.w_dec[[f, col]];
            }
            logits[k] = s;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for k in 0..5 {
            logits[k] = (logits[k] - max).exp();
            sum += logits[k];
        }
        if sum > 0.0 {
            for k in 0..5 { logits[k] /= sum; }
        }
        logits
    }

    fn rebuild_hcache(&mut self, tr: &Dataset) {
        let user_offsets = calc_user_offsets(tr);
        for u in 0..tr.n_users {
            let start = user_offsets[u];
            let end = user_offsets[u + 1];
            if start == end { continue; }
            let h = self.encode(tr, start, end);
            self.hcache.row_mut(u).assign(&h);
        }
    }
}

// ---------------------------------------------------------------------------
// Regressor impl
// ---------------------------------------------------------------------------

impl Regressor for AexModel {
    type Config = AexConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let n_items = tr.n_items;
        let dim = n_items * 5;

        let w_enc = rand_array2(dim, cfg.n_feat, &mut rng, cfg.sigma);
        let b_enc = Array1::<f32>::zeros(cfg.n_feat);
        let w_dec = rand_array2(cfg.n_feat, dim, &mut rng, cfg.sigma);
        let b_dec = Array2::<f32>::zeros((n_items, 5));
        let y_enc = Array2::<f32>::zeros((n_items, cfg.n_feat));
        let ud = SparseUD::new(tr, pr);
        let udbias = vec![0.0f32; ud.n_total()];

        let mut model = Self {
            cfg, w_enc, b_enc, w_dec, b_dec, y_enc,
            hcache: Array2::<f32>::zeros((tr.n_users, cfg.n_feat)),
            ubias: Array1::<f32>::zeros(tr.n_users),
            udbias, ud,
            rng: StdRng::seed_from_u64(cfg.seed.wrapping_add(1)),
        };
        model.rebuild_hcache(tr);
        model
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let probs = self.decode_item(self.hcache.row(u), i);
        let ev = probs[0] + 2.0 * probs[1] + 3.0 * probs[2] + 4.0 * probs[3] + 5.0 * probs[4];
        let bu = self.ubias[u];
        let bud = self.ud.index(u, day as i16).map_or(0.0, |idx| self.udbias[idx]);
        ev + bu + bud
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        let cfg = self.cfg;
        let n_feat = cfg.n_feat;
        let lr = cfg.lr;
        let reg = cfg.reg;

        let user_offsets = calc_user_offsets(tr);
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        for &u in crate::progress!(users.iter()) {
            let start = user_offsets[u];
            let end = user_offsets[u + 1];
            if start == end { continue; }

            // Generate dropout mask
            let mask: Option<Vec<bool>> = if cfg.dropout > 0.0 {
                let m: Vec<bool> = (start..end)
                    .map(|_| self.rng.random::<f32>() >= cfg.dropout)
                    .collect();
                Some(m)
            } else {
                None
            };

            // Forward pass (with dropout mask if denoising)
            let h = self.encode_masked(tr, start, end, mask.as_deref());

            // Decode all rated items, compute MSE gradient through softmax
            // d_h = gradient w.r.t. hidden layer (accumulated)
            let mut d_h = Array1::<f32>::zeros(n_feat);
            let mut ubias_grad = 0.0f32;

            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                let day = tr.dates[t];

                // Decode this item → softmax probs → expected value
                let probs = self.decode_item(h.view(), i);
                let ev = probs[0] + 2.0 * probs[1] + 3.0 * probs[2] + 4.0 * probs[3] + 5.0 * probs[4];

                // MSE error: predicted - target
                let bud = self.ud.index(u, day).map_or(0.0, |idx| self.udbias[idx]);
                let err = ev + self.ubias[u] + bud - tr.residuals[t];

                // MSE gradient through softmax:
                // d_loss/d_logit[k] = err * probs[k] * ((k+1) - ev)
                for k in 0..5 {
                    let col = i * 5 + k;
                    let d_logit = err * probs[k] * ((k + 1) as f32 - ev);

                    // Update b_dec
                    self.b_dec[[i, k]] -= lr * d_logit;

                    // Update W_dec and accumulate d_h
                    for f in 0..n_feat {
                        d_h[f] += d_logit * self.w_dec[[f, col]];
                        self.w_dec[[f, col]] -= lr * (d_logit * h[f] + reg * self.w_dec[[f, col]]);
                    }
                }

                // Update user×day bias
                if cfg.lr_udbias > 0.0 {
                    if let Some(idx) = self.ud.index(u, day) {
                        self.udbias[idx] -= cfg.lr_udbias * err;
                    }
                }

                ubias_grad += err;
            }

            // Update user bias
            if cfg.lr_ubias > 0.0 {
                self.ubias[u] -= cfg.lr_ubias * ubias_grad;
            }

            // Backprop through sigmoid: d_pre = d_h * h * (1 - h)
            let mut d_pre = Array1::<f32>::zeros(n_feat);
            for f in 0..n_feat {
                d_pre[f] = d_h[f] * h[f] * (1.0 - h[f]);
            }
            if cfg.normalize {
                let enc_cnt = if let Some(ref m) = mask {
                    m.iter().filter(|&&x| x).count()
                } else {
                    end - start
                };
                if enc_cnt > 0 {
                    let scale = 1.0 / (enc_cnt as f32).sqrt();
                    d_pre *= scale;
                }
            }

            // Update b_enc
            for f in 0..n_feat {
                self.b_enc[f] -= lr * d_pre[f];
            }

            // Update W_enc and y_enc (only items in encoder input)
            for t in start..end {
                if let Some(ref m) = mask {
                    if !m[t - start] { continue; }
                }
                let i = tr.item_idxs[t] as usize;
                let k = tr.raw_ratings[t] as usize - 1;
                let row = i * 5 + k;
                for f in 0..n_feat {
                    self.w_enc[[row, f]] -= lr * (d_pre[f] + reg * self.w_enc[[row, f]]);
                }
                if cfg.use_implicit {
                    for f in 0..n_feat {
                        self.y_enc[[i, f]] -= lr * (d_pre[f] + reg * self.y_enc[[i, f]]);
                    }
                }
            }

            // Update hcache (without dropout)
            let h_cache = self.encode(tr, start, end);
            self.hcache.row_mut(u).assign(&h_cache);
        }
    }
}
