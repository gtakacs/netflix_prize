use netflix_prize::{
    SPLIT_NEW,
    asym::{AsymConfig, AsymModel},
    attn::{AttnConfig, AttnModel},
    dnn::{DnnConfig, DnnModel},
    fit2,
    knn3::{Knn3Config, Knn3Model},
};
use std::env;

/// Environment override for a tuning knob, used by the `dnn-dbg` probe only.
fn ev<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Split a `<d>` or `<d>pNN` size spec into the factor count and the residual
/// weight: bare = 1.0, `p50` = 0.5, `p25` = 0.25. The weight is the axis that
/// moves the blend most. The less of the base a chained model is handed, the
/// more of the signal it has to rebuild in its own terms, and the further its
/// residuals sit from the base's. The step from 1.0 to 0.5 is large; the
/// quarter-steps around it are worth about a millionth each.
///
/// A bare `p` used to mean 0.5 and is now an error, so that every weight in a
/// job name reads as a percentage.
fn split_weight(spec: &str) -> (&str, f32) {
    match spec.split_once('p') {
        None => (spec, 1.0),
        Some((d, pct)) => (d, pct.parse::<f32>().expect("bad weight suffix") / 100.0),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    // kNN3 over a base model's residuals is its own job, as in the other model
    // families: it reads the base model's saved train predictions rather than
    // retraining it, which is what `extras = ["@train_preds"]` gates on. The
    // chains share one body, but every job name is spelled out so that searching
    // the source for a name from the pipeline lands on the logic that runs it.
    if matches!(
        job_name,
        "dnn-24__knn3"
            | "dnn-24__dnn-16__knn3"
            | "dnn-24__dnn-16p50__knn3"
            | "dnn-24__asym-16p75__knn3"
    ) {
        let base = job_name.strip_suffix("__knn3").unwrap();
        let target = format!("1.0*{}", base);
        fit2!(Knn3Model, Knn3Config::default(), &target, job_name, SPLIT_NEW);
        return;
    }

    // Two more kNN3 columns off one base, at a much wider (`b`) and much
    // narrower (`c`) neighborhood than the default. The kNN3 *configuration*
    // turns out to be a diversity axis of its own: on this deep chain the two
    // siblings are worth 17e-6 and 14e-6, more than most base models. It only
    // works on a base that is itself novel. The same trick on `dnn-24`,
    // `attn-32__nlpp` or `dnn-24__dnn-16p50` is worth exactly nothing.
    if matches!(
        job_name,
        "dnn-24__dnn-16p50__asym-16p75__knn3b" | "dnn-24__dnn-16p50__asym-16p75__knn3c"
    ) {
        let wide = job_name.ends_with('b');
        let cfg = if wide {
            Knn3Config {
                threshold: 0.10, k_min: 20, k_max: 150, shrinkage: 5000.0, x: 0.6,
                ..Knn3Config::default()
            }
        } else {
            Knn3Config {
                threshold: 0.40, k_min: 5, k_max: 25, shrinkage: 50000.0, x: 1.0,
                ..Knn3Config::default()
            }
        };
        let base = job_name[..job_name.len() - 1].strip_suffix("__knn3").unwrap();
        let target = format!("1.0*{}", base);
        fit2!(Knn3Model, cfg, &target, job_name, SPLIT_NEW);
        return;
    }

    // Attention over the user's own other ratings, on dnn-24's residual. The
    // pool is a fixed strided sample of the user's history and the softmax
    // weighs it by learned item similarity and rating-date distance; with
    // `n_mf: 0` there is no bilinear term, so the column is pure neighborhood.
    // Deterministic at `n_threads: 1`, since every parameter is per-user or per-item
    // so there is no shared dense matrix for threads to race on.
    if job_name == "dnn-24__attn-32" {
        let cfg = AttnConfig {
            n_feat: 32, n_pool: 32, n_mf: 0, n_epochs: 10, n_threads: 1,
            ..AttnConfig::default()
        };
        fit2!(AttnModel, cfg, "1.0*dnn-24", job_name, SPLIT_NEW);
        return;
    }

    // `<base>__asym-<d>[pNN]`: NSVD1-style asymmetric factorisation on a
    // residual, where a user is represented by the items they rated rather than
    // by a free vector. Parameters from the proven `mf-61__asym-16` fit; fully
    // sequential, so deterministic without an `n_threads` knob, and an order of
    // magnitude cheaper than a dnn. On a *partial* residual it is the strongest
    // chain base found: `dnn-24__asym-16p50__knn3` reaches probe 0.88696. It saves
    // its train predictions because a kNN3 chain is what makes it pay.
    if matches!(
        job_name,
        "dnn-24__asym-16p75"
            | "dnn-24__dnn-16p50__asym-16p50"
            | "dnn-24__dnn-16p50__asym-16p75"
    ) {
        let (residual_base, spec) = job_name.rsplit_once("__asym-").unwrap();
        let (n_feat, weight) = split_weight(spec);
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
            save_ifeat: false,
        };
        let target: &'static str = format!("{}*{}", weight, residual_base).leak();
        fit2!(AsymModel, cfg, target, job_name, SPLIT_NEW, save_train: true);
        return;
    }

    let base = DnnConfig::default();
    let (cfg, target) = match job_name {
        // Fast hyperparameter probe: a slice of the users, probe phase only,
        // every knob overridable from the environment.
        // Any `dnn-dbg*` name shares this arm, so two probes can run at once
        // without clobbering each other's prediction files.
        n if n.starts_with("dnn-x") => (
            DnnConfig {
                n_feat: ev("D", base.n_feat),
                h1: ev("H1", base.h1),
                h2: ev("H2", base.h2),
                n_epochs: ev("EPOCHS", 8),
                seed: ev("SEED", base.seed),
                lr_mlp: ev("LR_MLP", base.lr_mlp),
                lr_emb: ev("LR_EMB", base.lr_emb),
                lr_bias: ev("LR_BIAS", base.lr_bias),
                lr_decay: ev("LR_DECAY", base.lr_decay),
                reg_emb: ev("REG_EMB", base.reg_emb),
                reg_bias: ev("REG_BIAS", base.reg_bias),
                reg_mlp: ev("REG_MLP", base.reg_mlp),
                reg_bu_day: ev("REG_BD", base.reg_bu_day),
                grad_clip: ev("CLIP", base.grad_clip),
                block_users: ev("BLOCK", base.block_users),
                n_threads: ev("THREADS", base.n_threads),
                emb_cap: ev("CAP", base.emb_cap),
                out_scale: ev("OSCALE", base.out_scale),
                n_mf: ev("NMF", base.n_mf),
                lr_mf: ev("LR_MF", base.lr_mf),
                reg_mf: ev("REG_MF", base.reg_mf),
                train_frac: ev("FRAC", 0.15),
                ..base
            },
            "rtg",
        ),
        "dnn-16"  => (
            DnnConfig { n_feat: 16, h1: 64, h2: 32, n_mf: 192, n_epochs: 14, n_threads: 1, ..base },
            "rtg",
        ),
        // Deliberately off the grid of the three above, with a different aspect
        // ratio, a wider bilinear term and another seed, so its residuals are
        // its own, which is what the `dnn-24__knn3` chain then works on.
        "dnn-24" => (
            DnnConfig {
                n_feat: 24, h1: 96, h2: 64, n_mf: 256, seed: 43,
                n_epochs: 8, lr_decay: 0.87, n_threads: 1, ..base
            },
            "rtg",
        ),
        // More capacity overfits sooner, so decay faster over fewer epochs. The
        // epoch count is exactly what the blend consumes: the learning-rate
        // schedule depends only on the epoch index, so truncating to K leaves
        // the final prediction bit-identical to what epoch K produced before.
        // A d=32 sibling used to sit here and was dropped. Between d=16 and
        // d=64 it added no function class the blend could not already reach.
        "dnn-64"  => (
            DnnConfig {
                n_feat: 64, h1: 96, h2: 48, n_mf: 192,
                n_epochs: 5, lr_decay: 0.85, n_threads: 1, ..base
            },
            "rtg",
        ),
        // A second dnn on dnn-24's residual, named after the chain convention
        // `<base>__<model>` (cf. `rbmx2-12__asym-16`). The boosting step stays
        // inside the family's low correlation with the ensemble, unlike kNN3 and
        // NLPP, which pull a model toward the consensus. `p50` leaves half the
        // base in place, so the chained model has to rebuild half the signal
        // itself, it changes the base more and is the more valuable of the two.
        // A third stage on top of `dnn-24__dnn-16` was measured and adds nothing.
        "dnn-24__dnn-16" | "dnn-24__dnn-16p50" => (
            DnnConfig { n_feat: 16, h1: 64, h2: 32, n_mf: 192, n_epochs: 8, n_threads: 1, ..base },
            if job_name.ends_with("p50") { "0.5*dnn-24" } else { "1.0*dnn-24" },
        ),
        // The one chain that does not start from a dnn. A small dnn on half of
        // `tsvdx5-120o`, chosen from the candidate roots by in-sample optimism
        // (probe residual minus train residual, since the target is built from
        // the root's *train* predictions): 0.183 here against dnn-24's own
        // 0.173, while tsvdx5-1000 at 0.311 and tsvdx4-60 at 0.254 are far too
        // overfit to learn a residual from. A second foreign root (rbmx2-500,
        // 0.186) adds nothing on top of this one. They substitute for each
        // other. Its own kNN3 chains are worth nothing either: half the root is
        // a tsvdx, so it sits closer to the ensemble consensus (correlation
        // 0.945 against the dnn family's 0.922) than a chain can pay off from.
        //
        // Six epochs, not the eight it was found with: the run was searched with
        // `save_probe_each_epoch` and epoch 6 was worth twice the final epoch.
        // The decay is `lr_decay^(epoch-1)`, independent of `n_epochs`, so this
        // reproduces that snapshot exactly.
        "tsvdx5-120o__dnn-16p50" => (
            DnnConfig { n_feat: 16, h1: 64, h2: 32, n_mf: 192, n_epochs: 6, n_threads: 1, ..base },
            "0.5*tsvdx5-120o",
        ),
        // The same partial residual at the capacity of its own base. It is more
        // accurate than the d=16 sibling (best epoch 0.90638 against 0.90792),
        // but what the blend takes is the *overfit* final epoch, 0.91047. As
        // everywhere in this family, idiosyncratic structure beats accuracy.
        // Every mid-training snapshot was measured and is worth nothing.
        "dnn-24__dnn-24p50" => (
            DnnConfig {
                n_feat: 24, h1: 96, h2: 64, n_mf: 256,
                n_epochs: 8, lr_decay: 0.87, n_threads: 1, ..base
            },
            "0.5*dnn-24",
        ),
        _ => panic!("invalid job name: {}", job_name),
    };
    if job_name.starts_with("dnn-x") {
        fit2!(DnnModel, cfg, target, job_name, SPLIT_NEW, no_fulltrain: true);
    } else if job_name == "dnn-24" {
        // The `__knn3` chain needs this model's train-set predictions; its own
        // epoch snapshots are not used by the blend, so they are not written.
        fit2!(DnnModel, cfg, target, job_name, SPLIT_NEW,
              save_train: true, save_subscores: true);
    } else if matches!(job_name, "dnn-24__dnn-16" | "dnn-24__dnn-16p50") {
        // Both variants feed a kNN3 chain, and in both cases the chain is what
        // the blend takes. `dnn-24__dnn-16p50` is fully redundant next to its own
        // `__knn3`, so only the train predictions matter here.
        fit2!(DnnModel, cfg, target, job_name, SPLIT_NEW, save_train: true);
    } else if matches!(job_name, "dnn-24__dnn-24p50" | "tsvdx5-120o__dnn-16p50") {
        // Nothing chains off these, so they write predictions and nothing else.
        fit2!(DnnModel, cfg, target, job_name, SPLIT_NEW);
    } else {
        fit2!(DnnModel, cfg, target, job_name, SPLIT_NEW, save_probe_each_epoch: true, save_subscores: true);
    }
}
