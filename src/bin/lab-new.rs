// Experiment dispatcher. Everything here reuses the existing library models
// unmodified; the point is the *wiring*: which model runs on whose residual,
// with which loss, and how far down the chain. All jobs are deterministic
// (`n_threads: 1`) so anything that pays can go straight into the standard set.

use netflix_prize::{
    OrdinalHeadConfig, SPLIT_NEW,
    asym::{AsymConfig, AsymModel},
    attn::{AttnConfig, AttnModel},
    dnn::{DnnConfig, DnnModel},
    fit2,
    knn3::{Knn3Config, Knn3Model},
    knnf::{KnnfConfig, KnnfModel},
    nlpp::{NlppConfig, NlppModel},
    tx::{TxConfig, TxModel},
};
use std::env;

fn ev<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// NLPP regularisation borrowed from the tuned tsvdx5-1000 fit: a sane prior
/// for a base of similar accuracy, not re-optimised per model.
fn nlpp_cfg(base: &'static str) -> NlppConfig {
    NlppConfig {
        base_model: base,
        preds_dir: SPLIT_NEW.preds_dir,
        n_als_iters: 2,
        reg_a: [(4.74449, 0.489646), (1.86309, 4.83715e5),
                (118.959, 1.19996e-6), (2.37045e5, 6.54512e-4)],
        reg_b: [(3.74578e-4, 0.452495), (0.638521, 5.08715e3),
                (3.36975e-8, 0.161125), (4.45546e3, 459.028)],
        shrinkage_u: 10.0,
        shrinkage_i: 25.0,
        regs_path: None,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    // Post-processing chains dispatch on the suffix alone, so they work for any
    // base, including bases produced by the other dispatchers.
    if let Some(base) = job_name.strip_suffix("__knn3") {
        let target = format!("1.0*{}", base);
        fit2!(Knn3Model, Knn3Config::default(), &target, job_name, SPLIT_NEW);
        return;
    }
    // `__knnf`: neighborhood from the base model's own *item factors* (cosine
    // similarity in latent space) instead of kNN3's rating co-occurrence. Needs
    // the base to have written `.ifeat.<ds>.npy`, which only AsymModel does.
    // `b` sharpens the cosine weights over a wider neighborhood with no time
    // decay (the `rbmx2-500__knnf` shape); `c` is narrow, flat and time-local.
    if let Some((base, width)) = ['\0', 'b', 'c']
        .iter()
        .find_map(|w| {
            let suffix = if *w == '\0' { "__knnf".to_string() } else { format!("__knnf{w}") };
            job_name.strip_suffix(&suffix).map(|b| (b, *w))
        })
    {
        let factors: &'static str = format!("{}/{}.ifeat", SPLIT_NEW.preds_dir, base).leak();
        let cfg = match width {
            'b' => KnnfConfig { factors, k: 60, scaling: 2.0, tau: 0.0 },
            'c' => KnnfConfig { factors, k: 10, scaling: 0.5, tau: 0.06 },
            _ => KnnfConfig::with_factors(factors),
        };
        let target = format!("1.0*{}", base);
        fit2!(KnnfModel, cfg, &target, job_name, SPLIT_NEW);
        return;
    }

    // `__knn3p`: the default config on a *half* residual. Every chain in the
    // project hands kNN3 the whole residual; the partial-residual trick is what
    // paid for the dnn and asym stages, and it had never been tried here.
    if let Some(base) = job_name.strip_suffix("__knn3p") {
        let target = format!("0.5*{}", base);
        fit2!(Knn3Model, Knn3Config::default(), &target, job_name, SPLIT_NEW);
        return;
    }
    // `__knn3b` / `__knn3c`: the same chain at a far wider and a far narrower
    // neighborhood. Every chain in the project so far uses the default config,
    // so two kNN3 columns off one base had never been tried. The point is a
    // decorrelated sibling, not a better one. `b` was worth 17e-6 on a deep
    // asym chain and nothing at all on the shallower bases.
    if let Some((base, width)) = ['b', 'c', 'd', 'e']
        .iter()
        .find_map(|w| job_name.strip_suffix(&format!("__knn3{w}")).map(|b| (b, *w)))
    {
        let cfg = match width {
            'b' => Knn3Config {
                threshold: 0.10, k_min: 20, k_max: 150, shrinkage: 5000.0, x: 0.6,
                ..Knn3Config::default()
            },
            'c' => Knn3Config {
                threshold: 0.40, k_min: 5, k_max: 25, shrinkage: 50000.0, x: 1.0,
                ..Knn3Config::default()
            },
            'd' => Knn3Config {
                threshold: 0.05, k_min: 40, k_max: 250, shrinkage: 1000.0, x: 0.4,
                ..Knn3Config::default()
            },
            _ => Knn3Config {
                threshold: 0.15, k_min: 30, k_max: 100, shrinkage: 200.0, x: 1.3,
                ..Knn3Config::default()
            },
        };
        let target = format!("1.0*{}", base);
        fit2!(Knn3Model, cfg, &target, job_name, SPLIT_NEW);
        return;
    }
    if let Some(base) = job_name.strip_suffix("__nlpp") {
        let cfg = nlpp_cfg(base.to_string().leak());
        fit2!(NlppModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        return;
    }

    let attn = AttnConfig { n_threads: 1, ..Default::default() };
    let dnn = DnnConfig { n_threads: 1, ..Default::default() };

    match job_name {
        // A fresh dnn base that also writes its train-set predictions, so the
        // chains below have a second low-correlation base to work from.
        // (`dnn-16`'s own job does not save them.)
        "lab-d32" => {
            let cfg = DnnConfig {
                n_feat: 32, h1: 96, h2: 48, n_mf: 192, n_epochs: 7, lr_decay: 0.85, ..dnn
            };
            fit2!(DnnModel, cfg, "rtg", job_name, SPLIT_NEW,
                  save_train: true, save_probe_each_epoch: true, save_subscores: true);
        }
        "lab-d16" => {
            let cfg = DnnConfig { n_feat: 16, h1: 64, h2: 32, n_mf: 192, n_epochs: 14, ..dnn };
            fit2!(DnnModel, cfg, "rtg", job_name, SPLIT_NEW,
                  save_train: true, save_probe_each_epoch: true, save_subscores: true);
        }

        // Attention over a dnn's residual: a *learned* neighborhood second
        // stage, where the existing chains offer only fixed-similarity kNN3 and
        // a per-user/item polynomial. `-h` uses a Huber loss, `-p` leaves half
        // the base in place so the second stage explains a partial residual.
        "lab-a24r" | "lab-a24h" | "lab-a24p" => {
            let cfg = AttnConfig {
                n_feat: 32, n_pool: 32, n_mf: 0, n_epochs: 10,
                huber: if job_name == "lab-a24h" { 0.7 } else { 0.0 },
                ..attn
            };
            let target = if job_name == "lab-a24p" { "0.5*dnn-24" } else { "1.0*dnn-24" };
            fit2!(AttnModel, cfg, target, job_name, SPLIT_NEW, save_train: true);
        }

        // The partial-residual recipe at the capacity of its own base: does a
        // d=24 second stage beat the d=16 that earned its place? Epoch snapshots
        // are kept because at ~25 min an epoch this is the family's most
        // expensive fit and the best epoch is not known in advance.
        "dnn-24__dnn-24p50" => {
            let cfg = DnnConfig {
                n_feat: 24, h1: 96, h2: 64, n_mf: 256, n_epochs: 8, lr_decay: 0.87, ..dnn
            };
            fit2!(DnnModel, cfg, "0.5*dnn-24", job_name, SPLIT_NEW,
                  save_train: true, save_probe_each_epoch: true);
        }

        // The reverse direction: a small integrated model on the dnn's partial
        // residual. Config copied verbatim from the proven `rbmx2-10__tsvdx4-10lm`
        // fit (n_feat 10, neighborhood disabled). The result is not a plain
        // tsvdx. It is `0.5*dnn-24` plus a tsvdx over the other half, so its
        // residuals are half dnn, which is the point.
        // Idea 3: the same fit under a different loss geometry. The ordinal head
        // (src/lib.rs, used by the tsvdx4/tsvdx5 jobs) replaces squared error
        // with cumulative-link thresholds over the five rating categories, so
        // the errors land differently, especially at the tails, where a
        // mean-regression model is systematically pulled inward.
        "dnn-24__tsvdx4-10lmp" | "lab-tx10ord" => {
            let cfg = TxConfig {
        n_feat: 10,
        n_epochs: 12,
        seed: 42,
        shuffle_users: true,
        n_time_bins: 29,
        beta: 0.3,
        n_freq_bins: 16,
        lr_u: 0.003,
        lr_ud: 0.00125,
        lr_u2: 7e-6,
        lr_ub: 0.0031,
        lr_ubd: 0.003,
        lr_i: 0.0036,
        lr_ib: 0.0036,
        lr_y: 0.0005,
        lr_yb: 2.5e-5,
        lr_yd: 0.000267,
        lr_tu: 0.0,
        lr_ti: 0.000225,
        lr_ta: 2.25e-5,
        lr_ibf: 5e-5,
        lr_iqf: 5e-6,
        lr_cu: 0.002,
        reg_iqf: 0.007,
        reg_cu: 0.01,
        reg_u: 0.0504,
        reg_u2: 0.4,
        reg_ud: 0.04,
        reg_i: 0.00735,
        reg_y: 0.04,
        reg_yd: 0.02667,
        sigma_iqf: 0.005,
        sigma_u: 0.0015,
        sigma_i: 0.005,
        sigma_y: 0.00333,
        sigma_yd: 0.009,
        reset_u_epoch: 13,
        // Neighborhood disabled (tx4 mode)
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        // No sub-model lr multipliers
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false,
        lambda1: 0.0,
        lambda2: 0.0,
        ordinal_head: if job_name == "lab-tx10ord" {
            Some(OrdinalHeadConfig { th_init: [0.5, 1.25, 3.75, 5.5], th_gap: 0.001,
                                     lr_t: 0.0, reg_t: 0.0 })
        } else {
            None
        },
        save_ifeat: true,
        low_memory: true,
        full_su: true,
            };
            fit2!(TxModel, cfg, "0.5*dnn-24", job_name, SPLIT_NEW,
                  save_train: true, save_probe_each_epoch: true);
        }

        // Residual against a *mixture* of two roots. The target spec takes a
        // linear combination, which nothing in the project had used. The point
        // is orthogonality to two families at once: a quarter of a dnn plus a
        // quarter of a foreign model leaves a residual that neither explains,
        // where every earlier chain was orthogonal to one root only.
        "lab-mix1" | "lab-mix2" => {
            let cfg = DnnConfig { n_feat: 16, h1: 64, h2: 32, n_mf: 192, n_epochs: 8, ..dnn };
            let target = match job_name {
                "lab-mix1" => "0.25*dnn-24 + 0.25*tsvdx5-120o",
                _ => "0.25*dnn-24 + 0.25*rbmx2-500",
            };
            fit2!(DnnModel, cfg, target, job_name, SPLIT_NEW,
                  save_train: true, save_probe_each_epoch: true);
        }

        // The one untested sibling at the most productive spot in the tree: a
        // third *dnn* on the partial residual of the partial-residual dnn, where
        // so far only asym has stood. The full-residual third stage was measured
        // at zero, but the partial one at this depth never was.
        "dnn-24__dnn-16p50__dnn-16p50" => {
            let cfg = DnnConfig { n_feat: 16, h1: 64, h2: 32, n_mf: 192, n_epochs: 8, ..dnn };
            fit2!(DnnModel, cfg, "0.5*dnn-24__dnn-16p50", job_name, SPLIT_NEW,
                  save_train: true, save_probe_each_epoch: true);
        }

        // Generic `<base>__asym-<d>[pNN]`, for trying the recipe on bases the
        // dnn dispatcher does not list yet. Anything that pays moves there.
        n if n.contains("__asym-") => {
            let (residual_base, spec) = n.rsplit_once("__asym-").unwrap();
            let (n_feat, weight) = match spec.split_once('p') {
                None => (spec, 1.0),
                Some((d, "")) => (d, 0.5),
                Some((d, pct)) => (d, pct.parse::<f32>().expect("bad weight suffix") / 100.0),
            };
            let cfg = AsymConfig {
                n_feat: n_feat.parse().expect("asym job name must end in <d>[pNN]"),
                n_epochs: 10,
                seed: 42,
                shuffle_users: true,
                lr_ub: 0.0031,
                lr_i: 0.0036,
                lr_ib: 0.0036,
                lr_y: 0.0005,
                lr_us: 0.0,
                reg_i: 0.007,
                reg_y: 0.04,
                reg_us: 0.0,
                sigma_i: 0.005,
                sigma_y: 0.005,
                init_with_user_std: false,
                save_ifeat: true, // item factors for the `__knnf` chains below
            };
            let target: &'static str = format!("{}*{}", weight, residual_base).leak();
            fit2!(AsymModel, cfg, target, job_name, SPLIT_NEW, save_train: true);
        }

        // Free-form probe: model, target and every knob from the environment.
        n if n.starts_with("lab-x") => {
            let cfg = AttnConfig {
                n_feat: ev("D", 32),
                n_pool: ev("POOL", 32),
                n_mf: ev("NMF", 0),
                n_epochs: ev("EPOCHS", 8),
                huber: ev("HUBER", 0.0),
                lr_q: ev("LR_Q", attn.lr_q),
                lr_c: ev("LR_C", attn.lr_c),
                out_scale: ev("OSCALE", attn.out_scale),
                train_frac: ev("FRAC", 0.999),
                ..attn
            };
            let target: &'static str = env::var("TARGET").unwrap_or_else(|_| "rtg".into()).leak();
            fit2!(AttnModel, cfg, target, job_name, SPLIT_NEW, no_fulltrain: true);
        }
        _ => panic!("invalid job name: {}", job_name),
    }
}
