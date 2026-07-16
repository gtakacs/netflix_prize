// Integrated MF + RBM model: predicts rating as
//     w_mf · mf_score + w_rbm · rbm_score
// MF and RBM are trained jointly per user — MF first (with RBM held fixed),
// then RBM on the residual after MF.

use crate::{
    Dataset, MaskedDataset, Regressor,
    calc_gbias, calc_user_offsets, rand_array2, rand_array2u, sigmoid,
};
use indicatif::ProgressIterator;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, Normal};

// ---------------------------------------------------------------------------
// MF (inner)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct MfConfig {
    pub n_feat: usize,
    pub n_epochs: usize,
    pub seed: u64,

    pub lr_u: f32,
    pub lr_i: f32,
    pub lr_ub: f32,
    pub lr_ib: f32,

    pub reg_u: f32,
    pub reg_i: f32,

    pub sigma_u: f32,
    pub sigma_i: f32,

    pub reset_u_epoch: usize,
}

struct MfModel {
    cfg: MfConfig,
    gbias: f32,
    ubias: Array1<f32>,
    ibias: Array1<f32>,
    ufeat: Array2<f32>,
    ifeat: Array2<f32>,
}

impl MfModel {
    fn new(tr: &Dataset, cfg: MfConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        Self {
            cfg,
            gbias: calc_gbias(tr),
            ubias: Array1::<f32>::zeros(tr.n_users),
            ibias: Array1::<f32>::zeros(tr.n_items),
            ufeat: rand_array2(tr.n_users, cfg.n_feat, &mut rng, cfg.sigma_u),
            ifeat: rand_array2(tr.n_items, cfg.n_feat, &mut rng, cfg.sigma_i),
        }
    }

    #[inline]
    fn predict(&self, u: usize, i: usize) -> f32 {
        let pu = self.ufeat.row(u);
        let qi = self.ifeat.row(i);
        self.gbias + self.ubias[u] + self.ibias[i] + pu.dot(&qi)
    }

    #[inline]
    fn reset_epoch_if_needed(&mut self, epoch: usize) {
        if epoch == self.cfg.reset_u_epoch {
            self.ufeat.fill(0.0);
        }
    }

    fn fit_user_block(
        &mut self,
        tr: &Dataset,
        start: usize,
        end: usize,
        rbm_pred: &impl Fn(usize, usize) -> f32,
        w_mf: f32,
        w_rbm: f32,
    ) {
        let cfg = self.cfg;

        for t in start..end {
            let u = tr.user_idxs[t] as usize;
            let i = tr.item_idxs[t] as usize;
            let r = tr.residuals[t];

            let mf_pred = self.predict(u, i);
            let rbm_p = rbm_pred(u, i);

            let pred = w_mf * mf_pred + w_rbm * rbm_p;
            let err = pred - r;

            self.ubias[u] -= cfg.lr_ub * err;
            self.ibias[i] -= cfg.lr_ib * err;

            for k in 0..cfg.n_feat {
                let pk = self.ufeat[[u, k]];
                let qk = self.ifeat[[i, k]];
                self.ufeat[[u, k]] -= cfg.lr_u * (err * qk + cfg.reg_u * pk);
                self.ifeat[[i, k]] -= cfg.lr_i * (err * pk + cfg.reg_i * qk);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RBM with Gaussian visible units (inner)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct RbmGvConfig {
    pub n_hidden: usize,
    pub n_epochs: usize,
    pub seed: u64,

    pub gibbs3_after_epoch: usize,

    pub lr_w: f32,
    pub lr_c: f32,

    pub reg_w: f32,
    pub reg_c: f32,

    pub sigma_w: f32,

    pub sigma_v: f32,

    pub v_shift: f32,
    pub v_scale: f32,

    pub sample_visible: bool,
}

struct RbmGvModel {
    cfg: RbmGvConfig,
    rng: StdRng,

    c: Array2<f32>,      // [n_items, n_hidden]
    w: Array2<f32>,      // [n_items, n_hidden]

    hcache: Array2<f32>, // [n_users, n_hidden]
}

impl RbmGvModel {
    fn new(tr: &Dataset, cfg: RbmGvConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let c = Array2::zeros((tr.n_items, cfg.n_hidden));
        let w = rand_array2u(tr.n_items, cfg.n_hidden, &mut rng, cfg.sigma_w);
        let mut model = Self {
            cfg,
            rng,
            c,
            w,
            hcache: Array2::<f32>::zeros((tr.n_users, cfg.n_hidden)),
        };
        model.rebuild_hcache(tr);
        model
    }

    #[inline]
    fn v_from_residual(&self, r: f32) -> f32 {
        (r - self.cfg.v_shift) * self.cfg.v_scale
    }

    #[inline]
    fn residual_from_v(&self, v: f32) -> f32 {
        v / self.cfg.v_scale + self.cfg.v_shift
    }

    fn forward_hprob(&self, tr: &Dataset, start: usize, end: usize, vvals: Option<&[f32]>) -> Array1<f32> {
        let mut pre = Array1::zeros(self.cfg.n_hidden);
        for t in start..end {
            let i = tr.item_idxs[t] as usize;
            pre += &self.c.row(i);
        }
        let inv_sig2 = 1.0 / (self.cfg.sigma_v * self.cfg.sigma_v).max(1e-8);

        match vvals {
            None => {
                for t in start..end {
                    let i = tr.item_idxs[t] as usize;
                    let v = self.v_from_residual(tr.residuals[t]) * inv_sig2;
                    pre.scaled_add(v, &self.w.row(i));
                }
            }
            Some(vs) => {
                for (offset, t) in (start..end).enumerate() {
                    let i = tr.item_idxs[t] as usize;
                    let v = vs[offset] * inv_sig2;
                    pre.scaled_add(v, &self.w.row(i));
                }
            }
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

            let h = self.forward_hprob(tr, start, end, None);
            self.hcache.row_mut(u).assign(&h);
        }
    }

    #[inline]
    fn sample_bernoulli(rng: &mut StdRng, p: &Array1<f32>) -> Array1<f32> {
        let mut x = Array1::<f32>::zeros(p.len());
        for j in 0..p.len() {
            x[j] = if rng.random::<f32>() < p[j] { 1.0 } else { 0.0 };
        }
        x
    }

    fn reconstruct_v_for_user(&mut self, tr: &Dataset, start: usize, end: usize, h_bin: &Array1<f32>) -> Vec<f32> {
        let cfg = self.cfg;
        let normal = if cfg.sample_visible {
            Some(Normal::new(0.0, cfg.sigma_v.max(1e-8)).unwrap())
        } else {
            None
        };

        let mut v1 = Vec::<f32>::with_capacity(end - start);
        for t in start..end {
            let i = tr.item_idxs[t] as usize;
            let mu = self.w.row(i).dot(h_bin);
            let v = if let Some(n01) = normal.as_ref() {
                mu + n01.sample(&mut self.rng)
            } else {
                mu
            };
            v1.push(v);
        }
        v1
    }

    #[inline]
    fn predict(&self, u: usize, i: usize) -> f32 {
        let h = self.hcache.row(u);
        let v_hat = self.w.row(i).dot(&h);
        self.residual_from_v(v_hat)
    }

    fn fit_user_block(
        &mut self,
        tr: &Dataset,
        u: usize,
        start: usize,
        end: usize,
        epoch: usize,
        v0s: &[f32],
    ) {
        let cfg = self.cfg;
        let use_cd3 = epoch > cfg.gibbs3_after_epoch;

        let h0_prob = self.forward_hprob(tr, start, end, Some(v0s));
        let h0 = RbmGvModel::sample_bernoulli(&mut self.rng, &h0_prob);

        let (h_neg_prob, v_neg): (Array1<f32>, Vec<f32>) = if !use_cd3 {
            let v1 = self.reconstruct_v_for_user(tr, start, end, &h0);
            let h1_prob = self.forward_hprob(tr, start, end, Some(&v1));
            (h1_prob, v1)
        } else {
            let v1 = self.reconstruct_v_for_user(tr, start, end, &h0);
            let h1_prob = self.forward_hprob(tr, start, end, Some(&v1));
            let h1 = RbmGvModel::sample_bernoulli(&mut self.rng, &h1_prob);

            let v2 = self.reconstruct_v_for_user(tr, start, end, &h1);
            let h2_prob = self.forward_hprob(tr, start, end, Some(&v2));
            let h2 = RbmGvModel::sample_bernoulli(&mut self.rng, &h2_prob);

            let v3 = self.reconstruct_v_for_user(tr, start, end, &h2);
            let h3_prob = self.forward_hprob(tr, start, end, Some(&v3));

            (h3_prob, v3)
        };

        for (offset, t) in (start..end).enumerate() {
            let i = tr.item_idxs[t] as usize;

            let v0 = v0s[offset];
            let v1 = v_neg[offset];

            for h in 0..cfg.n_hidden {
                let c_ih = self.c[[i, h]];
                self.c[[i, h]] = c_ih
                    + cfg.lr_c * (h0_prob[h] - h_neg_prob[h])
                    - cfg.lr_c * cfg.reg_c * c_ih;

                let w_ih = self.w[[i, h]];
                let grad = v0 * h0_prob[h] - v1 * h_neg_prob[h];
                self.w[[i, h]] = w_ih + cfg.lr_w * grad - cfg.lr_w * cfg.reg_w * w_ih;
            }
        }

        let h_new = self.forward_hprob(tr, start, end, Some(v0s));
        self.hcache.row_mut(u).assign(&h_new);
    }
}

// ---------------------------------------------------------------------------
// Combined MF + RBM
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct MfRbmxConfig {
    pub mf: MfConfig,
    pub rbm: RbmGvConfig,

    pub w_mf: f32,
    pub w_rbm: f32,
}

pub struct MfRbmxModel {
    cfg: MfRbmxConfig,
    mf: MfModel,
    rbm: RbmGvModel,
}

impl Regressor for MfRbmxModel {
    type Config = MfRbmxConfig;

    fn new(tr: &Dataset, _pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let mf = MfModel::new(tr, cfg.mf);
        let rbm = RbmGvModel::new(tr, cfg.rbm);
        Self { cfg, mf, rbm }
    }

    fn n_epochs(&self) -> usize {
        self.cfg.mf.n_epochs.max(self.cfg.rbm.n_epochs)
    }

    fn n_subscores(&self) -> usize { 2 }

    fn subscore_names(&self) -> Vec<String> {
        ["mf", "rbm"].iter().map(|s| s.to_string()).collect()
    }

    fn predict_subscores(&self, u: usize, i: usize, _day: i32) -> Array1<f32> {
        let mf_s = self.mf.predict(u, i);
        let rbm_s = self.rbm.predict(u, i);
        Array1::from_vec(vec![
            self.cfg.w_mf * mf_s,
            self.cfg.w_rbm * rbm_s,
        ])
    }

    fn predict(&self, u: usize, i: usize, _day: i32) -> f32 {
        let mf_s = self.cfg.w_mf * self.mf.predict(u, i);
        let rbm_s = self.cfg.w_rbm * self.rbm.predict(u, i);
        mf_s + rbm_s
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        // Legacy shuffle: seed = epoch (no model seed mixed in). Differs from
        // the standard `get_users()` helper — preserved for byte-reproducibility
        // of the predsx outputs.
        let mut rng = StdRng::seed_from_u64(epoch as u64);
        let mut users: Vec<usize> = (0..tr.n_users).collect();
        users.shuffle(&mut rng);

        self.mf.reset_epoch_if_needed(epoch);

        let offsets = calc_user_offsets(tr);

        let w_mf = self.cfg.w_mf;
        let w_rbm = self.cfg.w_rbm;

        for &u in crate::progress!(users.iter()) {
            let start = offsets[u];
            let end = offsets[u + 1];
            if start == end {
                continue;
            }

            // 1) MF update (RBM fixed)
            {
                let rbm_pred = |uu: usize, ii: usize| -> f32 { self.rbm.predict(uu, ii) };
                self.mf.fit_user_block(tr, start, end, &rbm_pred, w_mf, w_rbm);
            }

            // 2) RBM update on residuals of MF
            let mut v0s = Vec::<f32>::with_capacity(end - start);
            for t in start..end {
                let uu = tr.user_idxs[t] as usize;
                let ii = tr.item_idxs[t] as usize;

                let r = tr.residuals[t];
                let mf_only = w_mf * self.mf.predict(uu, ii);
                let res_for_rbm = r - mf_only;

                v0s.push(self.rbm.v_from_residual(res_for_rbm));
            }

            self.rbm.fit_user_block(tr, u, start, end, epoch, &v0s);
        }
    }
}
