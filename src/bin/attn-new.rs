use netflix_prize::{
    SPLIT_NEW,
    attn::{AttnConfig, AttnModel},
    fit2,
    knn3::{Knn3Config, Knn3Model},
    nlpp::{NlppConfig, NlppModel},
};
use std::env;

/// Environment override for a tuning knob, used by the `attn-x*` probes only.
fn ev<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    // Post-processing chains, as in the other model families. Both read the
    // base model's saved train-set predictions, so the base job must have run
    // with `save_train`. NLPP keeps writing them so a `__nlpp__knn3` chain works.
    // Each job name is spelled out, so searching the source for a name from the
    // pipeline lands on the logic that runs it.
    if job_name == "attn-32__nlpp__knn3" {
        let base = job_name.strip_suffix("__knn3").unwrap();
        let target = format!("1.0*{}", base);
        fit2!(Knn3Model, Knn3Config::default(), &target, job_name, SPLIT_NEW);
        return;
    }
    if job_name == "attn-32__nlpp" {
        let base = job_name.strip_suffix("__nlpp").unwrap();
        // Regularisation borrowed from the tuned tsvdx5-1000 fit: a sane prior
        // for a base of similar accuracy; not re-optimised per model.
        let cfg = NlppConfig {
            base_model: base.to_string().leak(),
            preds_dir: SPLIT_NEW.preds_dir,
            n_als_iters: 2,
            reg_a: [(4.74449, 0.489646), (1.86309, 4.83715e5),
                    (118.959, 1.19996e-6), (2.37045e5, 6.54512e-4)],
            reg_b: [(3.74578e-4, 0.452495), (0.638521, 5.08715e3),
                    (3.36975e-8, 0.161125), (4.45546e3, 459.028)],
            shrinkage_u: 10.0,
            shrinkage_i: 25.0,
            regs_path: None,
        };
        fit2!(NlppModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        return;
    }

    let base = AttnConfig::default();
    let cfg = match job_name {
        // Fast probe: every knob overridable from the environment, and with
        // FRAC < 1 only a slice of the users. Any `attn-x*` name shares this
        // arm so several probes can run at once.
        n if n.starts_with("attn-x") => AttnConfig {
            n_feat: ev("D", base.n_feat),
            n_pool: ev("POOL", base.n_pool),
            n_mf: ev("NMF", base.n_mf),
            n_epochs: ev("EPOCHS", 6),
            seed: ev("SEED", base.seed),
            lr_q: ev("LR_Q", base.lr_q),
            lr_c: ev("LR_C", base.lr_c),
            beta: ev("BETA", base.beta),
            lr_bias: ev("LR_BIAS", base.lr_bias),
            lr_decay: ev("LR_DECAY", base.lr_decay),
            reg_q: ev("REG_Q", base.reg_q),
            reg_c: ev("REG_C", base.reg_c),
            out_scale: ev("OSCALE", base.out_scale),
            emb_cap: ev("CAP", base.emb_cap),
            huber: ev("HUBER", base.huber),
            n_threads: ev("THREADS", base.n_threads),
            train_frac: ev("FRAC", 0.999),
            ..base
        },
        // The measured configuration: attention alone converges slowly and adds
        // nothing to the blend, but paired with the wide bilinear term it beats
        // every dnn base model standalone and still adds on top of them.
        "attn-32" => AttnConfig {
            n_feat: 32, n_pool: 32, n_mf: 192, n_epochs: 12, n_threads: 1, ..base
        },
        // User-based attention: transposing swaps the roles, so the pool
        // becomes "the other users who rated this film" and the softmax weighs
        // *users* by learned similarity. The ensemble has no user-kNN at all,
        // because a 480k × 480k similarity matrix is out of reach, but attention never
        // materialises one, so the gap is reachable from here.
        "attn-32-t" => AttnConfig {
            n_feat: 32, n_pool: 32, n_mf: 192, n_epochs: 12, n_threads: 1, ..base
        },
        "attn-48" => AttnConfig {
            n_feat: 48, n_pool: 64, n_mf: 192, n_epochs: 12, n_threads: 1, ..base
        },
        _ => panic!("invalid job name: {}", job_name),
    };

    if job_name.starts_with("attn-x") {
        fit2!(AttnModel, cfg, "rtg", job_name, SPLIT_NEW, no_fulltrain: true);
    } else if job_name.ends_with("-t") {
        fit2!(AttnModel, cfg, "rtg", job_name, SPLIT_NEW,
              transpose: true, save_train: true, save_probe_each_epoch: true);
    } else {
        fit2!(AttnModel, cfg, "rtg", job_name, SPLIT_NEW,
              save_train: true, save_probe_each_epoch: true, save_subscores: true);
    }
}
