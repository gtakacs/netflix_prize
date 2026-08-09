// Conditional RBM for collaborative filtering, softmax visible units over the 5
// rating categories. Plain per-user SGD — no mini-batching, no momentum — with a
// CD-1 -> CD-3 switch at `gibbs3_after_epoch`. Exposes the 5 category
// probabilities as subscores.
// Produced: rbm-32, rbm-64.  Frozen archive — see README.md, does not build.
use ndarray::{Array1, Array2};
use gravity::{Dataset, Regressor};
use gravity::{calc_user_offsets, get_users, rand_array2u};
use rand::{rngs::StdRng, Rng, SeedableRng};
use indicatif::ProgressIterator;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_inplace(logits: &mut Array1<f32>) {
    let m = logits.iter().copied().reduce(f32::max).unwrap();
    let mut tmp = logits.mapv(|x| (x - m).exp());
    let s = tmp.sum();
    if s > 0.0 {
        tmp /= s;
    }
    logits.assign(&tmp);
}

#[derive(Clone, Copy, Debug)]
struct RbmConfig {
    n_hidden: usize,
    n_levels: usize,
    n_epochs: usize,
    seed: u64,
    shuffle_users: bool,

    // after this epoch, run 3 Gibbs steps instead of 1 (CD-3 vs CD-1)
    gibbs3_after_epoch: usize,

    lr_w: f32,
    lr_a: f32,
    lr_b: f32,
    lr_c: f32,

    reg_w: f32,
    reg_a: f32,
    reg_b: f32,
    reg_c: f32,

    sigma_w: f32,
    sigma_a: f32,
    sigma_b: f32,
    sigma_c: f32,

    use_logprobs: bool,
}

struct RbmModel {
    cfg: RbmConfig,
    rng: StdRng,
    a: Array2<f32>,      // Visible biases: [n_items, n_levels]
    b: Array1<f32>,      // Hidden biases: [n_hidden]
    c: Array2<f32>,      // Conditional biases: [n_items, n_hidden]
    w: Array2<f32>,      // Weights: [(n_items*n_levels), n_hidden]
    hcache: Array2<f32>, // Hidden activations: [n_users, n_hidden]
    logprobs: Array1<f32>,
}

impl RbmModel {
    #[inline]
    fn row_idx(&self, item: usize, level: usize) -> usize {
        item * self.cfg.n_levels + level
    }

    fn hidden_pre_from_items(&self, tr: &Dataset, start: usize, end: usize) -> Array1<f32> {
        let mut pre = self.b.clone();
        for t in start..end {
            let i = tr.item_idxs[t] as usize;
            pre += &self.c.row(i);
        }
        pre
    }

    fn forward(&self, tr: &Dataset, start: usize, end: usize) -> Array1<f32> {
        let mut pre = self.hidden_pre_from_items(tr, start, end);
        for t in start..end {
            let i = tr.item_idxs[t] as usize;
            let k = tr.raw_ratings[t] as usize - 1;
            pre += &self.w.row(self.row_idx(i, k));
        }
        pre.mapv(sigmoid)
    }

    fn rebuild_hcache(&mut self, tr: &Dataset) {
        self.hcache.fill(0.0);
        let mut idx = 0;
        for u in 0..tr.n_users {
            let cnt = tr.user_cnts[u] as usize;
            if cnt == 0 {
                continue;
            }
            let start = idx;
            let end = idx + cnt;
            idx = end;

            let h = self.forward(tr, start, end);
            self.hcache.row_mut(u).assign(&h);
        }
    }

    #[inline]
    fn sample_bernoulli(rng: &mut StdRng, p: &Array1<f32>) -> Array1<f32> {
        let mut x = Array1::<f32>::zeros(p.len());
        for i in 0..p.len() {
            x[i] = if rng.random::<f32>() < p[i] { 1.0 } else { 0.0 };
        }
        x
    }

    fn predict_probs(&self, u: usize, i: usize) -> Array1::<f32> {
        let h = self.hcache.row(u);
        let mut probs = Array1::<f32>::zeros(self.cfg.n_levels);
        for k in 0..self.cfg.n_levels {
            let w = self.w.row(self.row_idx(i, k));
            probs[k] = self.a[[i, k]] + w.dot(&h);
        }
        softmax_inplace(&mut probs);
        probs
    }
}

impl Regressor for RbmModel {
    type Config = RbmConfig;

    fn new(tr: &Dataset, _pr: &Dataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        let mut p = Array1::<f64>::zeros(cfg.n_levels);
        for i in 0..tr.n_ratings {
            let l = tr.raw_ratings[i] as usize - 1;
            p[l] += 1.0
        }
        p /= p.sum();
        let logprobs = p.mapv(|x| x.ln() as f32);

        let a = if cfg.use_logprobs {
            let mut a = Array2::<f32>::zeros((tr.n_items, cfg.n_levels));
            for i in 0..tr.n_items {
                for l in 0..cfg.n_levels { a[[i, l]] = logprobs[l]; }
            }
            a
        }
        else {
            rand_array2u(tr.n_items, cfg.n_levels, &mut rng, cfg.sigma_a)
        };

        let b = Array1::from_shape_fn(cfg.n_hidden, |_| {
            (rng.random::<f32>() * 2.0 - 1.0) * cfg.sigma_b
        });
        let c = rand_array2u(tr.n_items, cfg.n_hidden, &mut rng, cfg.sigma_c);
        let w = rand_array2u(tr.n_items * cfg.n_levels, cfg.n_hidden, &mut rng, cfg.sigma_w);

        let mut model = Self {
            cfg, rng, a, b, c, w,
            hcache: Array2::<f32>::zeros((tr.n_users, cfg.n_hidden)),
            logprobs
        };

        model.rebuild_hcache(tr);
        model
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    fn predict(&self, u: usize, i: usize, _day: i32) -> f32 {
        let probs = self.predict_probs(u, i);
        let rvals = Array1::from_shape_fn(self.cfg.n_levels, |k| k as f32 + 1.0);
        probs.dot(&rvals)
    }

    fn n_subscores(&self) -> usize { self.cfg.n_levels }

    fn predict_subscores(&self, u: usize, i: usize, _day: i32) -> Array1::<f32> {
        self.predict_probs(u, i)
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        let use_cd3 = epoch > cfg.gibbs3_after_epoch;

        let user_offsets = calc_user_offsets(tr);
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        // let mut idx = 0;
        // for u in (0..tr.n_users).progress() {
        for &u in users.iter().progress() {
            let start = user_offsets[u];
            let end = user_offsets[u + 1];
            let cnt = tr.user_cnts[u] as usize;
            if cnt == 0 { continue; }
            // let start = idx;
            // let end = idx + cnt;
            // idx = end;

            // Positive phase
            let h0_prob = self.forward(tr, start, end);
            let h0 = RbmModel::sample_bernoulli(&mut self.rng, &h0_prob);

            let mut logits = Array1::<f32>::zeros(cfg.n_levels);

            let b = &self.b;
            let c = &self.c;
            let a = &self.a;
            let w = &self.w;

            let hidden_pre_from_items = |tr: &Dataset, start: usize, end: usize| -> Array1<f32> {
                let mut pre = b.clone();
                for t in start..end {
                    let i = tr.item_idxs[t] as usize;
                    pre += &c.row(i);
                }
                pre
            };

            let mut gibbs_from_h = |h_bin: &Array1<f32>| -> (Vec<Array1<f32>>, Array1<f32>) {
                let mut vps = Vec::with_capacity(end - start);
                let mut pre = hidden_pre_from_items(tr, start, end);

                for t in start..end {
                    let i = tr.item_idxs[t] as usize;

                    for k in 0..cfg.n_levels {
                        let wr = w.row(i * cfg.n_levels + k);
                        logits[k] = a[[i, k]] + wr.dot(h_bin);
                    }
                    softmax_inplace(&mut logits);
                    vps.push(logits.clone());

                    for k in 0..cfg.n_levels {
                        let wr = w.row(i * cfg.n_levels + k);
                        pre.scaled_add(logits[k], &wr);
                    }
                }
                (vps, pre.mapv(sigmoid))
            };

            // Negative phase
            let (h_neg_prob, vneg_probs_by_t) = if !use_cd3 {
                // CD-1
                let (v1, h1_prob) = gibbs_from_h(&h0);
                (h1_prob, v1)
            } else {
                // CD-3
                let (_v1, h1_prob) = gibbs_from_h(&h0);
                let h1 = RbmModel::sample_bernoulli(&mut self.rng, &h1_prob);

                let (_v2, h2_prob) = gibbs_from_h(&h1);
                let h2 = RbmModel::sample_bernoulli(&mut self.rng, &h2_prob);

                let (v3, h3_prob) = gibbs_from_h(&h2);

                (h3_prob, v3)
            };

            // Update b
            self.b =
                &self.b * (1.0 - cfg.lr_b * cfg.reg_b) + cfg.lr_b * (&h0_prob - &h_neg_prob);

            // Update c
            for t in start..end {
                let i = tr.item_idxs[t] as usize;
                for h in 0..cfg.n_hidden {
                    let c_ih = self.c[[i, h]];
                    self.c[[i, h]] = c_ih
                        + cfg.lr_c * (h0_prob[h] - h_neg_prob[h])
                        - cfg.lr_c * cfg.reg_c * c_ih;
                }
            }

            // Update a and w
            for (offset, t) in (start..end).enumerate() {
                let i = tr.item_idxs[t] as usize;
                let k_pos = tr.raw_ratings[t] as usize - 1;

                let vneg = &vneg_probs_by_t[offset];

                for k in 0..cfg.n_levels {
                    let row = self.row_idx(i, k);
                    let v0 = if k == k_pos { 1.0 } else { 0.0 };
                    let v1 = vneg[k];

                    // Update a
                    let a_ik = self.a[[i, k]];
                    let a_ik_target = if cfg.use_logprobs { a_ik - self.logprobs[k] } else { a_ik };
                    self.a[[i, k]] = a_ik + cfg.lr_a * (v0 - v1) - cfg.lr_a * cfg.reg_a * a_ik_target;

                    // Update w
                    for h in 0..cfg.n_hidden {
                        let w_rh = self.w[[row, h]];
                        let grad = v0 * h0_prob[h] - v1 * h_neg_prob[h];
                        self.w[[row, h]] = w_rh + cfg.lr_w * grad - cfg.lr_w * cfg.reg_w * w_rh;
                    }
                }
            }
        }

        self.rebuild_hcache(tr);
    }
}

fn main() {
    let cfg = RbmConfig {
        n_hidden: 120,
        n_levels: 5,
        n_epochs: 15,
        seed: 42,
        shuffle_users: true,
        gibbs3_after_epoch: 8,
        lr_w: 0.0025,
        lr_a: 0.0029,
        lr_b: 0.0029,
        lr_c: 0.0029,
        // reg_w: 0.0017 + 0.0018,
        // reg_w: 0.005,
        reg_w: 0.016,
        reg_a: 0.0005,
        reg_b: 0.0005,
        reg_c: 0.0005,
        sigma_w: 0.01,
        sigma_a: 0.01,
        sigma_b: 0.01,
        sigma_c: 0.01,
        use_logprobs: false,
    };

    // Snapshot of one run — n_hidden/model_name were edited by hand per experiment.
    // The commented tr_set/pr_set are leftovers from the earlier `fit` signature.
    gravity::fit2::<RbmModel>(
        cfg,
        "rtg",      // target
        // "train8",   // tr_set
        // "probe8",   // pr_set
        "rbm-120",  // model_name
        false,      // save_subscores
        false,      // save_train
        false,      // save_probe_each_epoch
    );
}
