// 5-class multinomial-softmax classifier MF (explicitly NOT ordinal): every rating
// category gets its own (gbias, ubias, ibias, p_u, q_i) head and the scalar
// prediction is E[r] = sum c * p_c. Cross-entropy with label smoothing `eps`.
// Produced: catmf-16.  Frozen archive, see README.md; does not build.
use ndarray::{Array2};
use gravity::{rand_array2, Dataset, Regressor};
use rand::{rngs::StdRng, SeedableRng};
use indicatif::ProgressIterator;

// 5-class classifier MF (multinomial softmax, NOT ordinal).
// Each class has its own (gbias, ubias, ibias, ufeat, ifeat) "head".

const C: usize = 5;

// Hyperparameters
#[derive(Clone, Copy, Debug)]
struct ClfMfConfig {
    n_feat: usize,
    n_epochs: usize,
    seed: u64,

    lr_gb: f32,
    lr_ub: f32,
    lr_ib: f32,
    lr_u: f32,
    lr_i: f32,

    reg_u: f32,
    reg_i: f32,

    sigma_u: f32,
    sigma_i: f32,

    eps: f32,
    reset_u_epoch: usize,
}

struct ClfMfModel {
    cfg: ClfMfConfig,

    // Log-prior init: gbias[c] = ln(freq_c / total)
    gbias: [f32; C],

    // (n_users, 5), (n_items, 5)
    ubias: Array2<f32>,
    ibias: Array2<f32>,

    // (n_users, 5*n_feat), (n_items, 5*n_feat)
    ufeat: Array2<f32>,
    ifeat: Array2<f32>,
}

impl ClfMfModel {
    #[inline]
    fn logits(&self, u: usize, i: usize, _day: i32) -> [f32; C] {
        let k = self.cfg.n_feat;
        let row_u = u * (C * k);
        let row_i = i * (C * k);

        let ufeat = self.ufeat.as_slice().unwrap();
        let ifeat = self.ifeat.as_slice().unwrap();

        let mut s = [0.0f32; C];

        // s_c = gbias_c + ubias_{u,c} + ibias_{i,c} + dot(p_{u,c}, q_{i,c})
        for c in 0..C {
            let mut sc = self.gbias[c]
                + self.ubias[[u, c]]
                + self.ibias[[i, c]];

            let base_u = row_u + c * k;
            let base_i = row_i + c * k;

            let mut dot = 0.0f32;
            for t in 0..k {
                dot += ufeat[base_u + t] * ifeat[base_i + t];
            }
            sc += dot;

            s[c] = sc;
        }

        s
    }

    #[inline]
    pub fn probs(&self, u: usize, i: usize, day: i32) -> [f32; C] {
        let s = self.logits(u, i, day);

        // softmax
        let mut ex = [0.0f32; C];
        let mut z = 0.0f32;
        for c in 0..C {
            let e = s[c].exp();
            ex[c] = e;
            z += e;
        }
        for c in 0..C {
            ex[c] /= z;
        }
        ex
    }

    #[inline]
    pub fn expected(&self, u: usize, i: usize, day: i32) -> f32 {
        let p = self.probs(u, i, day);
        // E[r] = sum_{c=1..5} c * p_c
        (1.0 * p[0]) + (2.0 * p[1]) + (3.0 * p[2]) + (4.0 * p[3]) + (5.0 * p[4])
    }

    // #[inline]
    // fn init_gbias_log_prior(tr: &Dataset) -> [f32; C] {
    //     let mut cnt = [0u32; C];
    //     for &r in tr.raw_ratings.iter() {
    //         let rr = r as i32;
    //         if (1..=5).contains(&rr) {
    //             cnt[(rr - 1) as usize] += 1;
    //         }
    //     }
    //     let total = cnt.iter().copied().sum::<u32>() as f32;
    //     let mut gb = [0.0f32; C];
    //     for c in 0..C {
    //         gb[c] = (cnt[c] as f32 / total).ln();
    //     }
    //     gb
    // }
}

impl Regressor for ClfMfModel {
    type Config = ClfMfConfig;

    fn new(tr: &Dataset, _pr: &Dataset, cfg: Self::Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let k5 = C * cfg.n_feat;

        Self {
            cfg,
            // gbias: Self::init_gbias_log_prior(tr),
            gbias: [0.0; 5],
            ubias: Array2::<f32>::zeros((tr.n_users, C)),
            ibias: Array2::<f32>::zeros((tr.n_items, C)),
            ufeat: rand_array2(tr.n_users, k5, &mut rng, cfg.sigma_u),
            ifeat: rand_array2(tr.n_items, k5, &mut rng, cfg.sigma_i),
        }
    }

    fn n_epochs(&self) -> usize {
        self.cfg.n_epochs
    }

    // Framework expects a scalar prediction -> we return E[rating].
    #[inline]
    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        self.expected(u, i, day)
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &Dataset, epoch: usize) {
        let cfg = self.cfg;
        if epoch == cfg.reset_u_epoch { self.ufeat.fill(0.0); }

        let k = cfg.n_feat;
        let k5 = C * k;

        // Raw slice access for tight updates
        let ufeat = self.ufeat.as_slice_mut().unwrap();
        let ifeat = self.ifeat.as_slice_mut().unwrap();

        for idx in (0..tr.n_ratings).progress() {
            let u = tr.user_idxs[idx] as usize;
            let i = tr.item_idxs[idx] as usize;

            // true class y in 0..5
            let y = (tr.raw_ratings[idx] as i32 - 1) as usize;

            // logits
            let row_u = u * k5;
            let row_i = i * k5;

            let mut s = [0.0f32; C];
            for c in 0..C {
                let sc = self.gbias[c] + self.ubias[[u, c]] + self.ibias[[i, c]];
                let base_u = row_u + c * k;
                let base_i = row_i + c * k;
                let mut dot = 0.0f32;
                for t in 0..k {
                    dot += ufeat[base_u + t] * ifeat[base_i + t];
                }
                s[c] = sc + dot;
            }

            // softmax probs
            let mut p = [0.0f32; C];
            let mut z = 0.0f32;
            for c in 0..C {
                let e = s[c].exp();
                p[c] = e;
                z += e;
            }
            for c in 0..C {
                p[c] /= z;
            }

            // gradient: g_c = p_c - I[c==y]
            for c in 0..C {
                // let g = p[c] - if c == y { 1.0 } else { 0.0 }
                //let eps = 0.025;
                //let eps = 0.01;
                let g = p[c] - if c == y { 1.0 - 4.0 * cfg.eps } else { cfg.eps };

                self.gbias[c] -= cfg.lr_gb * g;
                self.ubias[[u, c]] -= cfg.lr_ub * g;
                self.ibias[[i, c]] -= cfg.lr_ib * g;

                let base_u = row_u + c * k;
                let base_i = row_i + c * k;

                for t in 0..k {
                    let pu = ufeat[base_u + t];
                    let qi = ifeat[base_i + t];

                    ufeat[base_u + t] = pu - cfg.lr_u * (g * qi + cfg.reg_u * pu);
                    ifeat[base_i + t] = qi - cfg.lr_i * (g * pu + cfg.reg_i * qi);
                }
            }
        }
    }
}

fn main() {
    let cfg = ClfMfConfig {
        n_feat: 16,
        n_epochs: 9,
        seed: 42,

        lr_gb: 0.002,
        lr_ub: 0.1,
        lr_ib: 0.01,
        lr_u: 0.6,
        lr_i: 0.12,

        reg_u: 0.01,
        reg_i: 0.025,

        sigma_u: 0.004,
        sigma_i: 0.004,

        eps: 0.025,
        reset_u_epoch: 1024,
    };

    gravity::fit2::<ClfMfModel>(
        cfg,
        "rtg",        // target
        "catmf-16",   // model_name
        false,        // save_subscores
        true,         // save_train
        false,        // save_probe_each_epoch
    );
}
