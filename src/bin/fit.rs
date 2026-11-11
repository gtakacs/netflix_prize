// split_3 model runs: one dispatcher with a subcommand per model family. Each
// subcommand builds the config of a library model from the environment (every
// knob is an env var, so a parameter search needs no recompile) and runs it on
// the split given by -p. The arguments after the subcommand are the lab ones
// (NAME, -p, --final, --preds, --target, --smoke, --sample).
//
//   s3 <subcommand> NAME -p pipeline-3.toml [--final] [--preds DIR] ...
//
//   tx       TxModel (the tsvdx4/5 family)
//   rbm      the conditional RBM (RxModel)
//   misc     the smaller families, picked by the name's prefix (mf, als, aex, bk3,
//            bk1, bknbrx, mfrbmx)
//   chain    second stages on another column's residual (knn-*, knn3, knnf, knns,
//            nlpp, asym-<d>, kasym, kimp, uknn)
//   attn     the attention model
//   dnn      the dnn (embedding MLP + MF branch)
//   nade     CF-NADE
//   nbstats  item-item sim matrices for the knn-<tag> and kimp-<tag> chain stages
//   vfeat1   the voting features
//   integ    the integrated model trained by ALS (IntegModel; NSVD1 with TERMS=y)
//
// The sections below are the former lab-s3-<subcommand> binaries, unchanged
// apart from their `use` lines moving into the function (integ: its model code
// moved to src/integ.rs).
//
//   F=50 EPOCHS=20 cargo run --release --bin s3 -- tx s3-tx50 -p pipeline-3.toml

const SUBCOMMANDS: &[(&str, fn())] = &[
    ("tx", tx), ("rbm", rbm), ("misc", misc), ("chain", chain), ("attn", attn),
    ("dnn", dnn), ("nade", nade), ("nbstats", nbstats), ("vfeat1", vfeat1), ("integ", integ),
];

fn main() {
    let sub = std::env::args().nth(1).unwrap_or_default();
    match SUBCOMMANDS.iter().find(|(n, _)| *n == sub) {
        Some((_, run)) => run(),
        None => {
            let names: Vec<&str> = SUBCOMMANDS.iter().map(|(n, _)| *n).collect();
            eprintln!("usage: s3 <subcommand> NAME -p PIPELINE [--final] [--preds DIR] ...");
            eprintln!("subcommands: {}", names.join(", "));
            std::process::exit(2);
        }
    }
}

// ----------------------------------------------------------------------------
// tx
// ----------------------------------------------------------------------------
// split_3 experiments: TxModel (the tsvdx4/5 family) with every knob from the
// environment, so a parameter search needs no recompile. IFEAT=1 / UFEAT=1 save
// the item / user factors (`.ifeat.<ds>.npy`, `.ufeat.<ds>.npy`). Defaults are the
// tsvdx5 base recipe, with the neighborhood off unless NBR > 0. QUAL_ONLY=1
// runs only the fulltrain -> qual phase of an existing probe-only column.
//
//   F=50 EPOCHS=20 cargo run --release --bin s3 -- tx s3-tx50 -p pipeline-3.toml

fn tx() {
    use netflix_prize::{
        OrdinalHeadConfig, fit2,
        lab::{LabArgs, ev},
        tx::{TxConfig, TxModel},
    };

    let args = LabArgs::parse();

    let ord = ev("ORD", 0) != 0;
    let wf: f32 = ev("WF", 1.0);
    // NBR = 0 turns the neighborhood off (Tx itself reads 0 as "no cap").
    let nbr: usize = ev("NBR", 0);
    let lr_w: f32 = if nbr > 0 { ev("LR_W", 0.0015) } else { 0.0 };
    let cfg = TxConfig {
        n_feat: ev("F", 50),
        n_epochs: if args.sampled() { 2 } else { ev("EPOCHS", 20) },
        seed: ev("SEED", 42),
        shuffle_users: true,
        n_time_bins: ev("TBINS", 32),
        beta: ev("BETA", 0.3),
        n_freq_bins: ev("FBINS", 16),
        lr_u: ev("LR_U", 0.003), lr_ud: ev("LR_UD", 0.00125), lr_u2: ev("LR_U2", 7e-6),
        lr_ub: ev("LR_UB", 0.0031), lr_ubd: ev("LR_UBD", 0.003),
        lr_i: ev("LR_I", 0.0036), lr_ib: ev("LR_IB", 0.0036),
        lr_y: ev("LR_Y", 0.0005), lr_yb: ev("LR_YB", 2.5e-5), lr_yd: ev("LR_YD", 0.000267),
        lr_tu: ev("LR_TU", 0.0), lr_ti: ev("LR_TI", 0.000225), lr_ta: ev("LR_TA", 2.25e-5),
        lr_ibf: ev("LR_IBF", 5e-5), lr_iqf: ev("LR_IQF", 5e-6), lr_cu: ev("LR_CU", 0.002),
        reg_iqf: ev("REG_IQF", 0.007), reg_cu: ev("REG_CU", 0.01),
        reg_u: ev("REG_U", 0.0504), reg_u2: ev("REG_U2", 0.4), reg_ud: ev("REG_UD", 0.04),
        reg_i: ev("REG_I", 0.00735), reg_y: ev("REG_Y", 0.04), reg_yd: ev("REG_YD", 0.02667),
        sigma_iqf: 0.005, sigma_u: ev("SIGMA_U", 0.0015), sigma_i: ev("SIGMA_I", 0.005),
        sigma_y: ev("SIGMA_Y", 0.00333), sigma_yd: 0.009,
        reset_u_epoch: ev("RESET", 1024),
        max_neighbors: nbr,
        lr_w, lr_c: lr_w,
        reg_w: ev("REG_W", 0.002), reg_c: ev("REG_W", 0.002),
        // Same-day neighborhood (tsvdx6 when > 0, e.g. LR_WD=0.001 REG_WD=0.002).
        lr_w_day: ev("LR_WD", 0.0), lr_c_day: ev("LR_WD", 0.0),
        reg_w_day: ev("REG_WD", 0.0), reg_c_day: ev("REG_WD", 0.0),
        w_bias: ev("WB", wf), w_factor: wf, w_nbr: ev("WN", wf),
        sum_err_bug: false,
        lambda1: 25.0, lambda2: 10.0,
        ordinal_head: ord.then_some(OrdinalHeadConfig {
            th_init: [0.5, 1.25, 3.75, 5.5], th_gap: 0.001, lr_t: 0.0, reg_t: 0.0,
        }),
        save_ifeat: ev("IFEAT", 0) != 0,
        save_ufeat: ev("UFEAT", 0) != 0,
        low_memory: ev("LM", 0) != 0,
        full_su: true,
    };

    if ev("QUAL_ONLY", 0) != 0 {
        // Only the fulltrain -> qual phase, for a column whose probe_3 preds
        // already exist (same knobs as the probe run); probe files stay as they are.
        let split = args.split();
        netflix_prize::fit::<TxModel>(cfg, &args.target, split.fulltrain_tr, split.fulltrain_pr,
            &args.name, false, true, ev("EPSAVE", 0) != 0, split.preds_dir, false, false);
        return;
    }

    fit2!(TxModel, cfg, &args.target, &args.name, args.split(),
          save_train: true, save_probe_each_epoch: ev("EPSAVE", 0) != 0,
          skip_fulltrain: !args.final_run);
}

// ----------------------------------------------------------------------------
// rbm
// ----------------------------------------------------------------------------
// split_3 experiments: the conditional RBM (RxModel) with every knob from the
// environment. Defaults are the rbmx2-500 recipe at H hidden units.
// HT selects the hidden type: b(ernoulli), p (bipolar), r (NReLU), e (TruncExp
// on [EA, EB]).
// TRANSPOSE=1 swaps users and items (an item-based RBM).
// QUAL_ONLY=1 runs only the fulltrain -> qual phase (probe files are left alone).
// UFEAT=1 saves the expected hidden units per user (`.ufeat.<ds>.npy`, blend features).
//
//   H=100 EPOCHS=20 cargo run --release --bin s3 -- rbm s3-rbm100 -p pipeline-3.toml

fn rbm() {
    use netflix_prize::{
        fit2,
        lab::{LabArgs, ev},
        rx::{HiddenType, RxConfig, RxModel, VisibleType},
    };

    let args = LabArgs::parse();

    let hidden_type = match ev("HT", "b".to_string()).as_str() {
        "b" => HiddenType::Bernoulli,
        "p" => HiddenType::Bipolar,
        "r" => HiddenType::NReLU,
        "e" => HiddenType::TruncExp(ev("EA", 0.0), ev("EB", 1.0)),
        t => panic!("unknown hidden type '{t}'"),
    };
    let nf: usize = ev("NFACT", 0);
    let cfg = RxConfig {
        hidden_type,
        visible_type: VisibleType::Softmax,
        temperature: 1.0,
        n_hidden: ev("H", 100),
        n_epochs: if args.sampled() { 1 } else { ev("EPOCHS", 30) },
        seed: ev("SEED", 128),
        shuffle_users: true,
        init_sigma: 0.01,
        batch_size: ev("BATCH", 500),
        lr: ev("LR", 0.02),
        momentum: ev("MOM", 0.9),
        weight_decay: ev("WD", 0.001),
        lr_bu: ev("LR_BU", 0.001),
        wd_bu: 0.01,
        lr_but: ev("LR_BUT", 0.0005),
        wd_but: 0.01,
        cd_start: 1,
        cd_inc_every: ev("CD_EVERY", 5),
        cd_inc_by: 1,
        cd_max: ev("CD_MAX", 5),
        use_conditional: ev("COND", 1) != 0,
        r_include_pr_all: true,
        save_w: true,
        save_ufeat: ev("UFEAT", 0) != 0,
        n_factors: (nf > 0).then_some(nf),
        n_freq_bins: 8,
        lr_bif: ev("LR_BIF", 0.0),
        wd_bif: 0.01,
    };

    if ev("QUAL_ONLY", 0) != 0 {
        // Only the fulltrain -> qual phase, for a column whose probe_3 preds
        // already exist (same knobs as the probe run); probe files stay as they are.
        // Nothing in the RBM schedule depends on EPOCHS, so EPOCHS=n gives the
        // probe run's first n epochs.
        assert!(ev("TRANSPOSE", 0) == 0, "QUAL_ONLY is not set up for TRANSPOSE=1");
        let split = args.split();
        netflix_prize::fit::<RxModel>(cfg, &args.target, split.fulltrain_tr, split.fulltrain_pr,
            &args.name, false, true, ev("EPSAVE", 0) != 0, split.preds_dir, false, true);
        return;
    }

    fit2!(RxModel, cfg, &args.target, &args.name, args.split(),
          save_train: true, save_probe_each_epoch: ev("EPSAVE", 0) != 0,
          skip_fulltrain: !args.final_run, transpose: ev("TRANSPOSE", 0) != 0,
          train_in_rating_order: true);
}

// ----------------------------------------------------------------------------
// misc
// ----------------------------------------------------------------------------
// split_3 experiments: the smaller model families, picked by the name's
// prefix, every knob from the environment.
//
//   s3-mf*   biased SGD MF (mf-60's rates)          F, EPOCHS, RESET, REG_U, REG_I
//   s3-als*  als8                                   F, EPOCHS, REG_U, REG_I, SHRINK
//   s3-aex*  autoencoder                            F, EPOCHS, LR, REG, DROP,
//                                                   TRANSPOSE=1 (item-based)
//   s3-bk3*  BellKor time-aware SVD++ (bk3-20's)    F, EPOCHS, LR_SCALE
//   s3-bk4*  Pragmatic Theory BK4 (bk4.rs)          F, EPOCHS, NBR, SIGF, TIME, ENV,
//                                                   LR_SCALE, DECAY, ... (bk4::config_from_env)
//   s3-bk1*  Pragmatic Theory integrated (bk1-10's) F, EPOCHS, NBR, LR_SCALE, LR_NBR,
//                                                   LR_DECAY, REG_FACT, REG_NBR
//   s3-bknbrx*  BellKor neighborhood, eq 18          NBR, EPOCHS, LR_W, LR_WD, LR_ALPHA,
//               (bknbrx-50's)                       LR_BETA, REG_W, REG_ALPHA, THREADS
//   s3-mfrbmx*  joint MF + RBM (mfrbmx-50's)        F, H, EPOCHS, W_MF, W_RBM,
//                                                   LR_W, REG_U, REG_I
//
//   F=8 cargo run --release --bin s3 -- misc s3-als8 -p pipeline-3.toml

fn misc() {
    use netflix_prize::{
        aex::{AexConfig, AexModel},
        als8::{Als8Config, Als8Model},
        bk1::{Bk1Config, Bk1Model},
        bk3::{Bk3Config, Bk3Model},
        bk4::{self, Bk4Model},
        bknbrx::{BknbrxConfig, BknbrxModel, ParallelMode},
        fit2,
        lab::{LabArgs, ev},
        mf::{MfConfig, MfModel},
        mfrbmx::{self, MfRbmxConfig, MfRbmxModel, RbmGvConfig},
    };

    let args = LabArgs::parse();
    let split = args.split();
    let n = &args.name;
    let t = &args.target;
    let fin = !args.final_run;
    let ep = |d: usize| if args.sampled() { 1 } else { ev("EPOCHS", d) };
    let epsave = ev("EPSAVE", 0) != 0;

    if n.starts_with("s3-mf") && !n.starts_with("s3-mfrbmx") {
        let cfg = MfConfig {
            n_feat: ev("F", 60),
            n_epochs: ep(20),
            seed: ev("SEED", 42),
            shuffle_users: true,
            lr_u: 0.0031, lr_i: 0.0036, lr_ub: 0.0031, lr_ib: 0.0036,
            reg_u: ev("REG_U", 0.03),
            reg_i: ev("REG_I", 0.005),
            sigma_u: 0.004, sigma_i: 0.005,
            reset_u_epoch: ev("RESET", 0),
            item_feat_npy: None,
            ordinal_head: None,
            save_ifeat: false,
        };
        fit2!(MfModel, cfg, t, n, split, save_train: true,
              save_probe_each_epoch: epsave, skip_fulltrain: fin);
    } else if n.starts_with("s3-als") {
        let cfg = Als8Config {
            n_epochs: ep(10),
            seed: ev("SEED", 42),
            n_feat: ev("F", 8),
            sigma_u: 0.005, sigma_i: 0.005,
            reg_ub: 20.0, reg_ib: 20.0,
            reg_u: ev("REG_U", 0.05),
            reg_i: ev("REG_I", 0.01),
            shrink_m: ev("SHRINK", 40.0),
            use_probe: true,
            n_bias_used: 8,
        };
        fit2!(Als8Model, cfg, t, n, split, save_train: true,
              save_probe_each_epoch: epsave, skip_fulltrain: fin);
    } else if n.starts_with("s3-aex") {
        let cfg = AexConfig {
            n_feat: ev("F", 64),
            n_epochs: ep(20),
            seed: ev("SEED", 42),
            shuffle_users: true,
            lr: ev("LR", 0.001),
            reg: ev("REG", 0.002),
            sigma: 0.01,
            normalize: true,
            use_implicit: true,
            dropout: ev("DROP", 0.05),
            lr_ubias: 0.0001,
            lr_udbias: 5e-5,
        };
        fit2!(AexModel, cfg, t, n, split, save_train: true,
              save_probe_each_epoch: epsave, skip_fulltrain: fin,
              transpose: ev("TRANSPOSE", 0) != 0, train_in_rating_order: true);
    } else if n.starts_with("s3-bk4") {
        let mut cfg = bk4::config_from_env();
        if args.sampled() { cfg.n_epochs = 1; }
        fit2!(Bk4Model, cfg, t, n, split, save_train: true,
              save_probe_each_epoch: epsave, skip_fulltrain: fin);
    } else if n.starts_with("s3-bk3") {
        let s: f32 = ev("LR_SCALE", 1.0);
        let cfg = Bk3Config {
            n_feat: ev("F", 20),
            n_epochs: ep(10),
            seed: ev("SEED", 42),
            shuffle_users: true,
            n_time_bins: 30, n_time_bins_fact: 8,
            n_freq_bins_bias: 30, n_freq_bins_fact: 8,
            beta: 0.4, k1: 0.042979, k2: 1.0, dev_mean: false,
            k_neighbors: ev("NBR", 0),
            alpha_rho: 100.0, lambda1: 25.0, lambda2: 10.0,
            lr_bias: 0.00790541 * s, lr_fact: 0.00525646 * s, lr_qt: 0.00525646 * s,
            lr_qf: 0.0, lr_nbr: 0.000199005 * s, lr_h: 0.0406664 * s,
            lr_decay: 0.11587,
            reg_bias: 0.00138669, reg_fact: ev("REG_FACT", 0.0092125),
            reg_qt: 0.0092125, reg_qf: 0.0, reg_nbr: 0.030711, reg_h: 0.000373603,
        };
        fit2!(Bk3Model, cfg, t, n, split, save_train: true,
              save_probe_each_epoch: epsave, skip_fulltrain: fin);
    } else if n.starts_with("s3-bk1") {
        let s: f32 = ev("LR_SCALE", 1.0);
        let cfg = Bk1Config {
            n_feat: ev("F", 10),
            n_epochs: ep(30),
            seed: ev("SEED", 42),
            shuffle_users: true,
            n_time_bins: 30, beta: 0.4, k1: 0.0363636, k2: 0.909091, dev_mean: false,
            // 0 = no neighborhood term (bk1-10 ran without one).
            k_neighbors: ev("NBR", 0),
            alpha_rho: 100.0, lambda1: 25.0, lambda2: 10.0,
            lr_bias: 0.007 * s, lr_fact: 0.007 * s, lr_nbr: ev("LR_NBR", 0.001) * s, lr_h: 0.0,
            lr_decay: ev("LR_DECAY", 0.0785714),
            reg_bias: 0.005, reg_fact: ev("REG_FACT", 0.015), reg_nbr: ev("REG_NBR", 0.015),
            reg_h: 0.0,
        };
        fit2!(Bk1Model, cfg, t, n, split, save_train: true,
              save_probe_each_epoch: epsave, skip_fulltrain: fin);
    } else if n.starts_with("s3-bknbrx") {
        let cfg = BknbrxConfig {
            n_epochs: ep(25),
            seed: ev("SEED", 42),
            shuffle_users: true,
            parallel_mode: ParallelMode::Hogwild,
            n_threads: ev("THREADS", 8),
            n_time_bins: 30,
            beta: 0.4,
            lr_bu: 0.003, lr_but: 0.0025,
            lr_alpha: ev("LR_ALPHA", 1e-5),
            lr_bi: 0.002, lr_bit: 5e-5,
            lr_cu: 0.008, lr_cut: 0.002,
            reg_bu: 0.03, reg_but: 0.005,
            reg_alpha: ev("REG_ALPHA", 50.0),
            reg_bi: 0.03, reg_bit: 0.1,
            reg_cu: 0.01, reg_cut: 0.005,
            max_neighbors: ev("NBR", 50),
            lr_w: ev("LR_W", 0.005), lr_c: ev("LR_W", 0.005),
            lr_beta_u: ev("LR_BETA", 1e-7),
            reg_w: ev("REG_W", 0.002), reg_c: ev("REG_W", 0.002),
            reg_beta_u: 0.01,
            lr_w_day: ev("LR_WD", 0.004), lr_c_day: ev("LR_WD", 0.004),
            reg_w_day: ev("REG_WD", 0.002), reg_c_day: ev("REG_WD", 0.002),
            lambda1: 25.0, lambda2: 10.0,
        };
        fit2!(BknbrxModel, cfg, t, n, split, save_train: true,
              save_probe_each_epoch: epsave, skip_fulltrain: fin);
    } else if n.starts_with("s3-mfrbmx") {
        let e = ep(9);
        let cfg = MfRbmxConfig {
            mf: mfrbmx::MfConfig {
                n_feat: ev("F", 25),
                n_epochs: e,
                seed: ev("SEED", 42),
                lr_u: 0.005, lr_i: 0.03, lr_ub: 0.006, lr_ib: 0.03,
                reg_u: ev("REG_U", 0.07), reg_i: ev("REG_I", 0.004),
                sigma_u: 0.004, sigma_i: 0.005,
                reset_u_epoch: 1024,
            },
            rbm: RbmGvConfig {
                n_hidden: ev("H", 25),
                n_epochs: e,
                seed: ev("SEED", 42),
                gibbs3_after_epoch: 1024,
                lr_w: ev("LR_W", 6e-5), lr_c: 0.005,
                reg_w: 0.004, reg_c: 0.0,
                sigma_w: 0.02, sigma_v: 0.8,
                v_shift: 0.0, v_scale: 1.0,
                sample_visible: true,
            },
            w_mf: ev("W_MF", 0.5),
            w_rbm: ev("W_RBM", 1.0),
        };
        fit2!(MfRbmxModel, cfg, t, n, split, save_train: true,
              save_probe_each_epoch: epsave, skip_fulltrain: fin);
    } else {
        panic!("unknown model family in '{n}'");
    }
}

// ----------------------------------------------------------------------------
// chain
// ----------------------------------------------------------------------------
// split_3 experiments: second stages on another column's residual. The stage
// is the last `__` segment of the name, the base is everything before it, so
// chains nest (`s3-tx50__nlpp__knn3`). Knobs come from the environment, and
// `W` (default 1.0) is the weight of the base in the target (`W*base`).
//
//   K3_KMAX=100 cargo run --release --bin s3 -- chain s3-tx50__knn3b -p pipeline-3.toml
//
// Stages: knn3[tag], knnf[tag] (needs the base's `.ifeat`), knns[tag],
// nlpp[tag], asym-<d>[tag], and
//   knn-<tag>    the stat-matrix item kNN of knn.rs: KN_SIM = support|cosine|mse|
//                fcos|fpearson|fmse|fdot, KN_STAT0/1/2 = stems in the split's sim
//                dir (`none` = unset), KN_FACT = factor stem in the preds dir for
//                the f* sims, e.g. `s3-rbm400.ifeat`
//   kasym[-tag]  knn.rs with the asymmetric SimType::Asym: KA_SIM = supp|prod,
//                KA_A, KA_B, KA_STAT0 (rtg_supp), KA_STAT2 (a `*_prod`, prod only),
//                KA_K (30), KA_SCALE (2), KA_SHR, KA_TAU
//   kimp[-tag]   the implicit-neighbor kNN of knnimp.rs over bin_impcnt and
//                <base>_impsum (from `s3 nbstats bin_imp` / `<base>_imp`): KI_SIM =
//                supp|conf, KI_STAT0 (bin_wsupp), KI_SUM, KI_K, KI_SHR, KI_WSHR,
//                KI_DSHR, KI_SCALE, KI_TAU, KI_CENTER
//   uknn[-tag]   the user-user kNN of uknn.rs: UK_VEC = pres|res|coday, UK_FACT =
//                item-factor stem in the preds dir (s3-tx50n.ifeat), UK_K, UK_KP,
//                UK_SCALE, UK_SHR, UK_TAU, UK_DIM, UK_WIN, UK_CAP, IMPL

fn chain() {
    use netflix_prize::{
        asym::{AsymConfig, AsymModel},
        fit2,
        knn::{KnnConfig, KnnModel, SimType},
        knnimp::{KimpConfig, KimpModel},
        knn3::{Knn3Config, Knn3Model},
        knnf::{KnnfConfig, KnnfModel},
        knns::{KnnsConfig, KnnsModel, SuppSource},
        lab::{LabArgs, ev},
        nlpp::{NlppConfig, NlppModel},
        teeln,
        uknn::{UknnConfig, UknnModel},
    };

    let args = LabArgs::parse();
    let split = args.split();
    let name = args.name.trim_end_matches("-smoke").to_string();
    let (base, stage) = name.rsplit_once("__").expect("name must be <base>__<stage>");
    let w: f32 = ev("W", 1.0);
    let target = format!("{w}*{base}");
    let base: &'static str = base.to_string().leak();
    let n = &args.name;
    let fin = !args.final_run;

    if stage.starts_with("knn-") {
        let stem = |key: &str, d: &str| -> Option<&'static str> {
            let v: String = ev(key, d.to_string());
            (v != "none").then(|| &*v.leak())
        };
        let sim_type = match ev("KN_SIM", "support".to_string()).as_str() {
            "support" => SimType::Support,
            "cosine" => SimType::Cosine,
            "mse" => SimType::Mse,
            "fcos" => SimType::FactorCosine,
            "fpearson" => SimType::FactorPearson,
            "fmse" => SimType::FactorMse,
            "fdot" => SimType::FactorDot,
            s => panic!("unknown KN_SIM '{s}'"),
        };
        let thr: f32 = ev("KN_THR", f32::NAN);
        let cfg = KnnConfig {
            stat0: stem("KN_STAT0", "bin_wsupp"),
            stat1: stem("KN_STAT1", "none"),
            stat2: stem("KN_STAT2", "none"),
            factors: stem("KN_FACT", "none"),
            sim_type,
            k: ev("KN_K", 50),
            sim_threshold: (!thr.is_nan()).then_some(thr),
            shrinkage: ev("KN_SHR", 100.0),
            scaling: ev("KN_SCALE", 1.5),
            tau: ev("KN_TAU", 0.05),
            use_stat1: ev("KN_USE1", 0) != 0,
            regression: ev("KN_REG", 0) != 0,
            regression_lambda: ev("KN_RLAM", 0.1),
            sim_dir: split.sim_dir,
            preds_dir: split.preds_dir,
        };
        fit2!(KnnModel, cfg, &target, n, split, skip_fulltrain: fin);
    } else if stage.starts_with("knn3") {
        let d = Knn3Config::default();
        let cfg = Knn3Config {
            threshold: ev("K3_THR", d.threshold),
            k_min: ev("K3_KMIN", d.k_min),
            k_max: ev("K3_KMAX", d.k_max),
            shrinkage: ev("K3_SHR", d.shrinkage),
            reg: ev("K3_REG", d.reg),
            x: ev("K3_X", d.x),
            bl_reg_m: ev("K3_BLM", d.bl_reg_m),
            bl_reg_u: ev("K3_BLU", d.bl_reg_u),
        };
        fit2!(Knn3Model, cfg, &target, n, split, skip_fulltrain: fin);
    } else if stage.starts_with("knnf") {
        let factors: &'static str = format!("{}/{base}.ifeat", split.preds_dir).leak();
        let cfg = KnnfConfig {
            factors,
            k: ev("KF_K", 25),
            scaling: ev("KF_SCALE", 1.0),
            tau: ev("KF_TAU", 0.025),
        };
        fit2!(KnnfModel, cfg, &target, n, split, skip_fulltrain: fin);
    } else if stage.starts_with("knns") {
        let cfg = KnnsConfig {
            k: ev("KS_K", 25),
            shrinkage: ev("KS_SHR", 100.0),
            scaling: ev("KS_SCALE", 1.0),
            tau: ev("KS_TAU", 0.025),
            supp_source: SuppSource::Compute,
        };
        fit2!(KnnsModel, cfg, &target, n, split, skip_fulltrain: fin);
    } else if stage.starts_with("nlpp") {
        // Regularisation of the tuned tsvdx5-1000 fit, as in lab-new.
        let cfg = NlppConfig {
            base_model: base,
            preds_dir: split.preds_dir,
            n_als_iters: ev("NL_ITERS", 2),
            reg_a: [(4.74449, 0.489646), (1.86309, 4.83715e5),
                    (118.959, 1.19996e-6), (2.37045e5, 6.54512e-4)],
            reg_b: [(3.74578e-4, 0.452495), (0.638521, 5.08715e3),
                    (3.36975e-8, 0.161125), (4.45546e3, 459.028)],
            shrinkage_u: ev("NL_SHU", 10.0),
            shrinkage_i: ev("NL_SHI", 25.0),
            regs_path: None,
        };
        fit2!(NlppModel, cfg, "rtg", n, split, save_train: true, skip_fulltrain: fin);
    } else if let Some(spec) = stage.strip_prefix("asym-") {
        let d: String = spec.chars().take_while(|c| c.is_ascii_digit()).collect();
        let cfg = AsymConfig {
            n_feat: d.parse().expect("asym stage must be asym-<d>[tag]"),
            n_epochs: if args.sampled() { 1 } else { ev("EPOCHS", 10) },
            seed: ev("SEED", 42),
            shuffle_users: true,
            lr_ub: ev("LR_UB", 0.0031),
            lr_i: ev("LR_I", 0.0036),
            lr_ib: ev("LR_IB", 0.0036),
            lr_y: ev("LR_Y", 0.0005),
            lr_us: 0.0,
            reg_i: ev("REG_I", 0.007),
            reg_y: ev("REG_Y", 0.04),
            reg_us: 0.0,
            sigma_i: 0.005,
            sigma_y: 0.005,
            init_with_user_std: false,
            save_ifeat: true,
        };
        fit2!(AsymModel, cfg, &target, n, split, save_train: true,
              save_probe_each_epoch: ev("EPSAVE", 0) != 0, skip_fulltrain: fin);
    } else if stage.starts_with("kasym") {
        let stem = |key: &str, d: &str| -> Option<&'static str> {
            let v: String = ev(key, d.to_string());
            (v != "none").then(|| &*v.leak())
        };
        let prod = match ev("KA_SIM", "supp".to_string()).as_str() {
            "supp" => false,
            "prod" => true,
            s => panic!("unknown KA_SIM '{s}'"),
        };
        let stat2 = stem("KA_STAT2", "none");
        assert!(!prod || stat2.is_some(), "KA_SIM=prod needs KA_STAT2");
        let cfg = KnnConfig {
            stat0: Some(stem("KA_STAT0", "rtg_supp").expect("KA_STAT0")),
            stat1: None,
            // only the prod sim reads stat2
            stat2: if prod { stat2 } else { None },
            factors: None,
            sim_type: SimType::Asym { a: ev("KA_A", 1.0), b: ev("KA_B", 1.0), prod },
            k: ev("KA_K", 30),
            sim_threshold: None,
            shrinkage: ev("KA_SHR", 100.0),
            scaling: ev("KA_SCALE", 2.0),
            tau: ev("KA_TAU", 0.05),
            use_stat1: false,
            regression: false,
            regression_lambda: 0.0,
            sim_dir: split.sim_dir,
            preds_dir: split.preds_dir,
        };
        fit2!(KnnModel, cfg, &target, n, split, skip_fulltrain: fin);
    } else if stage.starts_with("kimp") {
        let stem = |key: &str, d: String| -> Option<&'static str> {
            let v: String = ev(key, d);
            (v != "none").then(|| &*v.leak())
        };
        let sim = ev("KI_SIM", "supp".to_string());
        let cfg = KimpConfig {
            sum: stem("KI_SUM", format!("{base}_impsum")).unwrap(),
            supp: match sim.as_str() {
                "supp" => Some(stem("KI_STAT0", "bin_wsupp".into()).expect("KI_STAT0")),
                "conf" => None,
                s => panic!("unknown KI_SIM '{s}'"),
            },
            k: ev("KI_K", 50),
            shrinkage: ev("KI_SHR", 100.0),
            w_shrinkage: ev("KI_WSHR", 100.0),
            d_shrinkage: ev("KI_DSHR", 0.0),
            scaling: ev("KI_SCALE", 1.5),
            tau: ev("KI_TAU", 0.05),
            center: ev("KI_CENTER", 1) != 0,
            sim_dir: split.sim_dir,
        };
        fit2!(KimpModel, cfg, &target, n, split, skip_fulltrain: fin);
    } else if stage.starts_with("uknn") {
        let vec: String = ev("UK_VEC", "pres".to_string());
        let cfg = UknnConfig {
            fact: ev("UK_FACT", "s3-tx50n.ifeat".to_string()).leak(),
            res_vec: vec == "res",
            coday: match vec.as_str() {
                "pres" | "res" => false,
                "coday" => true,
                v => panic!("unknown UK_VEC '{v}'"),
            },
            win: ev("UK_WIN", 1),
            cap: ev("UK_CAP", 1000),
            dim: ev("UK_DIM", 64),
            k: ev("UK_K", 300),
            kp: ev("UK_KP", 50),
            scaling: ev("UK_SCALE", 2.0),
            shrinkage: ev("UK_SHR", 10.0),
            tau: ev("UK_TAU", 0.0),
            implicit: ev("IMPL", 1) != 0,
            preds_dir: split.preds_dir,
        };
        teeln!("{cfg:?}");
        fit2!(UknnModel, cfg, &target, n, split, save_train: true, skip_fulltrain: fin);
    } else {
        panic!("unknown stage '{stage}'");
    }
}

// ----------------------------------------------------------------------------
// attn
// ----------------------------------------------------------------------------
// split_3 experiments: the attention model (attn.rs) with every knob from the
// environment. Defaults are attn-32's (D=32 POOL=32 NMF=192 EPOCHS=12).
// TRANSPOSE=1 gives the user-based variant (attn-32-t). THREADS=1 is
// deterministic, more threads are faster but not reproducible. TR_LEGACY=1 (with
// TRANSPOSE=1) saves the train preds in the old, transposed order, needed only
// to regenerate s3-attn32t__knn3 and s3-attn32t__nlpp.
//
//   EPSAVE=1 cargo run --release --bin s3 -- attn s3-attn32 -p pipeline-3.toml

fn attn() {
    use netflix_prize::{
        attn::{AttnConfig, AttnModel},
        fit2,
        lab::{LabArgs, ev},
    };

    let args = LabArgs::parse();

    let b = AttnConfig::default();
    let cfg = AttnConfig {
        n_feat: ev("D", 32),
        n_pool: ev("POOL", 32),
        n_mf: ev("NMF", 192),
        n_epochs: if args.sampled() { 1 } else { ev("EPOCHS", 12) },
        seed: ev("SEED", b.seed),
        lr_q: ev("LR_Q", b.lr_q),
        lr_c: ev("LR_C", b.lr_c),
        beta: ev("BETA", b.beta),
        lr_bias: ev("LR_BIAS", b.lr_bias),
        lr_decay: ev("DECAY", b.lr_decay),
        reg_q: ev("REG_Q", b.reg_q),
        reg_c: ev("REG_C", b.reg_c),
        n_threads: ev("THREADS", 1),
        ..b
    };

    let epsave = ev("EPSAVE", 0) != 0;
    if ev("TRANSPOSE", 0) != 0 {
        fit2!(AttnModel, cfg, &args.target, &args.name, args.split(),
              transpose: true, save_train: true, save_probe_each_epoch: epsave,
              train_in_rating_order: ev("TR_LEGACY", 0) == 0,
              skip_fulltrain: !args.final_run);
    } else {
        fit2!(AttnModel, cfg, &args.target, &args.name, args.split(),
              save_train: true, save_probe_each_epoch: epsave,
              skip_fulltrain: !args.final_run);
    }
}

// ----------------------------------------------------------------------------
// dnn
// ----------------------------------------------------------------------------
// split_3 experiments: the dnn (embedding MLP + MF branch) with every knob from
// the environment. Defaults are dnn-16's. THREADS=1 is deterministic, more
// threads are faster but not reproducible.
// TRANSPOSE=1 swaps users and items.
//
//   D=16 NMF=192 EPOCHS=10 cargo run --release --bin s3 -- dnn s3-dnn16 -p pipeline-3.toml

fn dnn() {
    use netflix_prize::{
        dnn::{DnnConfig, DnnModel},
        fit2,
        lab::{LabArgs, ev},
    };

    let args = LabArgs::parse();

    let b = DnnConfig::default();
    let cfg = DnnConfig {
        n_feat: ev("D", b.n_feat),
        h1: ev("H1", b.h1),
        h2: ev("H2", b.h2),
        n_mf: ev("NMF", 192),
        n_epochs: if args.sampled() { 1 } else { ev("EPOCHS", b.n_epochs) },
        seed: ev("SEED", b.seed),
        lr_mlp: ev("LR_MLP", b.lr_mlp),
        lr_emb: ev("LR_EMB", b.lr_emb),
        lr_mf: ev("LR_MF", b.lr_mf),
        lr_bias: ev("LR_BIAS", b.lr_bias),
        lr_decay: ev("DECAY", b.lr_decay),
        reg_emb: ev("REG_EMB", b.reg_emb),
        reg_mf: ev("REG_MF", b.reg_mf),
        reg_bias: ev("REG_BIAS", b.reg_bias),
        reg_mlp: ev("REG_MLP", b.reg_mlp),
        n_threads: ev("THREADS", 1),
        ..b
    };

    fit2!(DnnModel, cfg, &args.target, &args.name, args.split(),
          save_train: true, save_probe_each_epoch: ev("EPSAVE", 0) != 0,
          skip_fulltrain: !args.final_run, transpose: ev("TRANSPOSE", 0) != 0,
          train_in_rating_order: true);
}

// ----------------------------------------------------------------------------
// nade
// ----------------------------------------------------------------------------
// split_3 experiments: CF-NADE with the main knobs from the environment.
// Defaults are cfnade-48's shape; the lr milestones sit at 2/3 and 3/4 of the
// run, as in the 96-epoch preset, unless MS1/MS2 say otherwise.
//
//   H=160 RANK=48 EPOCHS=30 cargo run --release --bin s3 -- nade s3-nade48 -p pipeline-3.toml

fn nade() {
    use netflix_prize::{
        cfnade::{CfNadeConfig, CfNadeModel},
        fit2,
        lab::{LabArgs, ev},
    };

    let args = LabArgs::parse();

    let b = CfNadeConfig::default();
    let n_epochs = if args.sampled() { 1 } else { ev("EPOCHS", 30) };
    let cfg = CfNadeConfig {
        n_hidden: ev("H", 160),
        rank: ev("RANK", 48),
        n_epochs,
        seed: ev("SEED", b.seed),
        batch_size: ev("BATCH", b.batch_size),
        lr: ev("LR", b.lr),
        milestones: [ev("MS1", n_epochs * 2 / 3), ev("MS2", n_epochs * 3 / 4), 0, 0],
        weight_decay: ev("WD", b.weight_decay),
        ordinal_lambda: ev("ORDL", b.ordinal_lambda),
        rmse_weight: ev("RMSEW", b.rmse_weight),
        swa_start: ev("SWA", n_epochs / 5),
        use_user_item_scale: ev("UIS", 0) != 0,
        use_day_bias: ev("DAYB", 0) != 0,
        use_item_time_bias: ev("ITB", 0) != 0,
        ..b
    };

    fit2!(CfNadeModel, cfg, &args.target, &args.name, args.split(),
          save_train: ev("SAVETR", 0) != 0, save_probe_each_epoch: ev("EPSAVE", 0) != 0,
          skip_fulltrain: !args.final_run);
}

// ----------------------------------------------------------------------------
// nbstats
// ----------------------------------------------------------------------------
// split_3 sim matrices for the knn-<tag> chain stage: item-item statistics over
// train_3 and fulltrain, into the split's sim dir as <rtg>_<stat>.<ds>.npy. The
// name is <rtg>_<stat> as for nbstats-new: rtg = `rtg` (ratings), `bin` (who
// rated what, probe/qual included) or a model whose residuals are used (read
// from --preds); stat = supp, wsupp, prod, diff1, diff2, or imp: the pair
// statistics of the kimp-<tag> chain stage (knnimp.rs; bin_imp -> bin_impcnt,
// <base>_imp -> <base>_impsum).
//
//   cargo run --release --bin s3 -- nbstats rtg_supp -p pipeline-3.toml

fn nbstats() {
    use netflix_prize::{knnimp, lab::LabArgs, nbstats};

    let args = LabArgs::parse();
    let (rtg, stat) = args.name.rsplit_once('_')
        .unwrap_or_else(|| panic!("name must be of form <rtg>_<stat>: {}", args.name));
    if stat == "imp" {
        knnimp::save_imp(rtg, args.split());
    } else {
        nbstats::save_nbstats_split(rtg, stat, args.split());
    }
}

// ----------------------------------------------------------------------------
// vfeat1
// ----------------------------------------------------------------------------
// split_3 voting features: vfeat1 (Selection::All, the `all` group of
// voting-new.toml) on the lab split, into preds_lab/vf/<feature>.{probe_3,qual}.npy.
// The qual side depends only on fulltrain -> qual, so it must match
// preds_new/vf/*.qual.npy; that is the check of a run.
//
//   cargo run --release --bin s3 -- vfeat1 s3-vfeat1 -p pipeline-3.toml

fn vfeat1() {
    use netflix_prize::{
        lab::LabArgs,
        vfeat1::{Selection, save_vfeat1},
    };

    let args = LabArgs::parse();
    save_vfeat1(&args.name, Selection::All, args.split());
}

// ----------------------------------------------------------------------------
// integ
// ----------------------------------------------------------------------------
// split_3 experiments: the integrated model trained by ALS (src/integ.rs) with
// every knob from the environment.
//
// Knobs: TERMS (p+y), F, EPOCHS, ALPHA (0.5), REG_BU, REG_BI, REG_P, REG_Q, REG_Y,
// SIGMA (init std of p, q, y; without p, q starts at 0), CG_ITERS, CG_TOL, PRECOND,
// IMPL, SEED, EPSAVE, LOSSCHK, SORT (1: each user's ratings sorted by item, which
// nb, ud, dr and fq need; 0: input order, as the old NSVD1 code summed them);
// dr: REG_DR, DR_BETA (0.4); fq: REG_FQ, FQ_BASE (6.76);
// nb: NB_K (50), NB_START (first epoch that fits w, c; 1), NB_FREEZE (1: p, q, y
// stay fixed from NB_START on, only the biases and w, c are fitted), REG_NW, REG_NC, NB_SIM, NB_STAT0, NB_STAT2, NB_SHR (100); ud: REG_UD.
//
//   TERMS=p+y F=20 EPOCHS=5 cargo run --release --bin s3 -- integ s3-integ20 -p pipeline-3.toml --smoke

fn integ() {
    use netflix_prize::{
        fit2,
        integ::{IntegConfig, IntegModel},
        lab::{LabArgs, ev},
        teeln,
    };

    let args = LabArgs::parse();
    let split = args.split();
    let terms: String = ev("TERMS", "p+y".to_string());
    let has = |t: &str| terms.split('+').any(|x| x == t);
    for t in terms.split('+') { assert!(["p", "y", "nb", "ud", "dr", "fq"].contains(&t), "unknown term '{t}' in TERMS"); }
    let leak = |key: &str, d: &str| -> &'static str { ev(key, d.to_string()).leak() };
    let cfg = IntegConfig {
        use_p: has("p"),
        use_y: has("y"),
        use_nb: has("nb"),
        use_ud: has("ud"),
        use_dr: has("dr"),
        use_fq: has("fq"),
        n_feat: ev("F", 20),
        n_epochs: ev("EPOCHS", 10),
        alpha: ev("ALPHA", 0.5),
        reg_bu: ev("REG_BU", 10.0),
        reg_bi: ev("REG_BI", 10.0),
        reg_p: ev("REG_P", 10.0),
        reg_q: ev("REG_Q", 10.0),
        reg_y: ev("REG_Y", 10.0),
        reg_nw: ev("REG_NW", 10.0),
        reg_nc: ev("REG_NC", 10.0),
        reg_ud: ev("REG_UD", 10.0),
        reg_dr: ev("REG_DR", 1000.0),
        dr_beta: ev("DR_BETA", 0.4),
        reg_fq: ev("REG_FQ", 10.0),
        fq_base: ev("FQ_BASE", 6.76),
        nb_k: ev("NB_K", 50),
        nb_start: ev("NB_START", 1),
        nb_freeze: ev("NB_FREEZE", 0) != 0,
        nb_mse: match ev("NB_SIM", "mse".to_string()).as_str() {
            "mse" => true,
            "cos" => false,
            s => panic!("unknown NB_SIM '{s}'"),
        },
        nb_stat0: leak("NB_STAT0", "rtg_supp"),
        nb_stat2: leak("NB_STAT2", "rtg_diff2"),
        nb_shr: ev("NB_SHR", 100.0),
        sim_dir: split.sim_dir,
        sigma: ev("SIGMA", 0.01),
        cg_iters: ev("CG_ITERS", 20),
        cg_tol: ev("CG_TOL", 1e-4),
        precond: ev("PRECOND", 1) != 0,
        implicit: ev("IMPL", 0) != 0,
        seed: ev("SEED", 42),
        loss_check: ev("LOSSCHK", 1) != 0,
        sort: ev("SORT", 1) != 0,
    };
    assert!(cfg.sort || !(cfg.use_nb || cfg.use_ud || cfg.use_dr || cfg.use_fq),
            "SORT=0 works only with TERMS from p, y (nb, ud, dr, fq need item-sorted rows)");
    teeln!("{cfg:?}");
    let epsave = ev("EPSAVE", 0) != 0;
    fit2!(IntegModel, cfg, &args.target, &args.name, split, save_train: true,
          save_probe_each_epoch: epsave, skip_fulltrain: !args.final_run);
}
