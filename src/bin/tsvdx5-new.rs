use std::env;
use std::fs::{self, File};
use std::io::BufWriter;

use ndarray::Array1;
use ndarray_npy::write_npy;
use netflix_prize::{
    Dataset, OrdinalHeadConfig, SPLIT_NEW,
    epoch_blend_apply, fit2,
    knn::{KnnConfig, KnnModel, SimType},
    knn3::{Knn3Config, Knn3Model},
    knnf::{KnnfConfig, KnnfModel},
    nlpp::{NlppConfig, NlppModel},
    tx::{TxModel, TxConfig},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    match job_name {
        // === Base TxModel (no ordinal head) ===
        "tsvdx5-20"     => run_base(   20, 20, 25,  0.002,  false, 1024, false, true,  false, false, job_name),
        "tsvdx5-40"     => run_base(   40, 25, 25,  0.002,  false, 1024, false, true,  false, false, job_name),
        "tsvdx5-60"     => run_base(   60, 23, 50,  0.002,  false, 1024, false, false, false, false, job_name),
        "tsvdx5-120"    => run_base(  120, 23, 100, 0.0015, true,  1024, false, false, false, false, job_name),
        "tsvdx5-150"    => run_base(  150, 23, 100, 0.0015, false, 1024, false, true,  false, false, job_name),
        "tsvdx5-300"    => run_base(  300, 23, 100, 0.0015, false,   10, false, false, true,  false, job_name),
        "tsvdx5-400"    => run_base(  400, 26, 100, 0.0015, false,   10, false, false, false, false, job_name),
        "tsvdx5-1000"   => run_base( 1000, 30, 100, 0.0015, false,   10, false, false, false, false, job_name),
        "tsvdx5-1200lm" => run_base( 1200, 23, 100, 0.0015, false, 1024, true,  true,  true,  false, job_name),
        "tsvdx5-10fs"   => run_base(   10,  3, 25,  0.002,  false, 1024, false, false, false, true,  job_name),
        "tsvdx5-40fs"   => run_base(   40, 25, 25,  0.002,  false, 1024, false, false, false, true,  job_name),

        // === Base TxModel with ordinal head ===
        "tsvdx5-120o" => run_ord(120, 28, 100, 0.8, job_name),
        "tsvdx5-150o" => run_ord(150, 25, 120, 1.2, job_name),

        // === NLPP chains (base + regs from paropt) ===
        "tsvdx5-150__nlpp" | "tsvdx5-300__nlpp" | "tsvdx5-400__nlpp"
        | "tsvdx5-1000__nlpp" => run_nlpp(job_name),

        // === NLPP paropt (Nelder-Mead reg search) ===
        "tsvdx5-150__nlpp_paropt" => run_paropt(
            [(8.02449,    0.44670206),
             (39.480854,  39764.363),
             (116.59052,  2.9861756e-6),
             (430561.13,  5.7766872e-5)],
            [(0.0028033287, 0.24241549),
             (1.304104,     1689.8875),
             (6.817317e-8,  0.12698959),
             (4096.8228,    1490.0216)],
            job_name,
        ),
        "tsvdx5-300__nlpp_paropt" => run_paropt(
            [(7.0363865,   0.3988001),
             (2.5923166,   10831.524),
             (61.713535,   5.4500247e-6),
             (245677.72,   0.0012987673)],
            [(0.017870171,  0.24616595),
             (2.397998,     6.3064556),
             (1.6548915e-7, 0.069282666),
             (8500.04,      1451.2802)],
            job_name,
        ),
        // tsvdx5-1000 starts from the same simplex as tsvdx5-400 — it is only a
        // Nelder-Mead seed, and the 400 run is the closest sibling.
        "tsvdx5-400__nlpp_paropt" | "tsvdx5-1000__nlpp_paropt" => run_paropt(
            [(6.7, 0.9), (13.8, 15070.0), (84.4, 3e-6), (100478.9, 0.0007)],
            [(0.02, 0.68), (1.4, 27.3), (3e-7, 0.003), (14112.5, 1042.1)],
            job_name,
        ),

        // === Knn3 chains ===
        "tsvdx5-120o__knn3"
        | "tsvdx5-300__nlpp__knn3"
        | "tsvdx5-400__nlpp__knn3" => run_knn3(job_name, Knn3Config::default()),

        // Tuned for the 1000-factor NLPP base; every field differs from the
        // default, so it is spelled out in full.
        "tsvdx5-1000__nlpp__knn3" => run_knn3(job_name, Knn3Config {
            threshold: 0.5786969,
            k_min: 9,
            k_max: 106,
            shrinkage: 29774.094,
            reg: 0.5322302,
            x: 0.865259,
            bl_reg_m: 1.0029083,
            bl_reg_u: 5.5354156,
        }),

        // === Knnf (factor cosine) — uses tsvdx5-120 factors regardless of base ===
        "tsvdx5-150__nlpp__knnf" => run_knnf(job_name),

        // === Knn-w (binary support sim) ===
        "tsvdx5-300__nlpp__knn-w" => run_knn_w(job_name),

        // === Epoch blends (apply pre-computed weights, no training) ===
        "tsvdx5-120o__epochs" => run_eblend(
            &[
                0.027859, 0.028633, 0.029052, 0.029406, 0.029674,
                0.030008, 0.030212, 0.030423, 0.030624, 0.030802,
                0.030952, 0.031114, 0.031269, 0.031435, 0.031551,
                0.031699, 0.031830, 0.031948, 0.032081, 0.032203,
                0.032305, 0.032421, 0.032529, 0.032643, 0.032748,
                0.032844, 0.032927, 0.033024,
            ],
            0.381012,
            job_name,
        ),
        "tsvdx5-300__epochs" => run_eblend(
            &[
                0.034800, 0.035376, 0.035884, 0.036167, 0.036416,
                0.036645, 0.036883, 0.037075, 0.037255, 0.036797,
                0.037004, 0.037221, 0.037410, 0.037604, 0.037767,
                0.037958, 0.038098, 0.038248, 0.038387, 0.038531,
                0.038679, 0.038773, 0.038903,
            ],
            0.440369,
            job_name,
        ),

        _ => panic!("invalid job name: {}", job_name),
    }
}

fn run_base(
    n_feat: usize,
    n_epochs: usize,
    max_neighbors: usize,
    lr_w_c: f32,
    save_ifeat: bool,
    reset_u_epoch: usize,
    low_memory: bool,
    save_subscores: bool,
    save_probe_each_epoch: bool,
    save_factorscores: bool,
    job_name: &str,
) {
    let cfg = TxConfig {
        n_feat, n_epochs,
        seed: 42, shuffle_users: true,
        n_time_bins: 32, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.003, lr_ud: 0.00125, lr_u2: 7e-6,
        lr_ub: 0.0031, lr_ubd: 0.003,
        lr_i: 0.0036, lr_ib: 0.0036,
        lr_y: 0.0005, lr_yb: 2.5e-5, lr_yd: 0.000267,
        lr_tu: 0.0, lr_ti: 0.000225, lr_ta: 2.25e-5,
        lr_ibf: 5e-5, lr_iqf: 5e-6, lr_cu: 0.002,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.0504, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.00735, reg_y: 0.04, reg_yd: 0.02667,
        sigma_iqf: 0.005, sigma_u: 0.0015, sigma_i: 0.005,
        sigma_y: 0.00333, sigma_yd: 0.009,
        reset_u_epoch,
        max_neighbors,
        lr_w: lr_w_c, lr_c: lr_w_c, reg_w: 0.002, reg_c: 0.002,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 0.8, w_factor: 0.8, w_nbr: 0.8,
        sum_err_bug: false,
        lambda1: 25.0, lambda2: 10.0,
        ordinal_head: None,
        save_ifeat,
        low_memory,
        full_su: true,
    };
    match (save_subscores, save_probe_each_epoch) {
        (false, false) => fit2!(
            TxModel, cfg, "rtg", job_name, SPLIT_NEW,
            save_train: true, save_factorscores: save_factorscores
        ),
        (true, false) => fit2!(
            TxModel, cfg, "rtg", job_name, SPLIT_NEW,
            save_train: true, save_subscores: true, save_factorscores: save_factorscores
        ),
        (false, true) => fit2!(
            TxModel, cfg, "rtg", job_name, SPLIT_NEW,
            save_train: true, save_probe_each_epoch: true, save_factorscores: save_factorscores
        ),
        (true, true) => fit2!(
            TxModel, cfg, "rtg", job_name, SPLIT_NEW,
            save_train: true, save_subscores: true, save_probe_each_epoch: true, save_factorscores: save_factorscores
        ),
    }
}

fn run_ord(n_feat: usize, n_epochs: usize, max_neighbors: usize, w_factor: f32, job_name: &str) {
    let cfg = TxConfig {
        n_feat, n_epochs,
        seed: 64, shuffle_users: true,
        n_time_bins: 20, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.003, lr_ud: 0.0015, lr_u2: 1e-5,
        lr_ub: 0.006, lr_ubd: 0.006,
        lr_i: 0.0072, lr_ib: 0.0072,
        lr_y: 0.0012, lr_yb: 2e-6, lr_yd: 0.000267,
        lr_tu: 0.0, lr_ti: 0.0001, lr_ta: 1.5e-5,
        lr_ibf: 5e-5, lr_iqf: 5e-6, lr_cu: 0.002,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.04, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.007, reg_y: 0.04, reg_yd: 0.04,
        sigma_iqf: 0.005, sigma_u: 0.0, sigma_i: 0.005,
        sigma_y: 0.005, sigma_yd: 0.005,
        reset_u_epoch: 1024,
        max_neighbors,
        lr_w: 0.0015, lr_c: 0.0015, reg_w: 0.002, reg_c: 0.002,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: w_factor, w_factor, w_nbr: w_factor,
        sum_err_bug: false,
        lambda1: 25.0, lambda2: 10.0,
        ordinal_head: Some(OrdinalHeadConfig {
            th_init: [0.5, 1.25, 3.75, 5.5],
            th_gap: 0.001,
            lr_t: 0.0,
            reg_t: 0.0,
        }),
        save_ifeat: false,
        low_memory: false,
        full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true, save_subscores: true, save_probe_each_epoch: true
    );
}

fn run_nlpp(job_name: &str) {
    let base_model: &'static str =
        job_name.strip_suffix("__nlpp").unwrap().to_string().leak();
    let regs_path: &'static str =
        format!("{}/{}__nlpp_regs.npy", SPLIT_NEW.preds_dir, base_model).leak();
    let cfg = NlppConfig {
        base_model,
        preds_dir: SPLIT_NEW.preds_dir,
        n_als_iters: 2,
        // Placeholder values; overridden at construction by `regs_path`.
        reg_a: [(0.0, 0.0); 4],
        reg_b: [(0.0, 0.0); 4],
        shrinkage_u: 10.0,
        shrinkage_i: 25.0,
        regs_path: Some(regs_path),
    };
    fit2!(NlppModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
}

fn run_paropt(reg_a: [(f32, f32); 4], reg_b: [(f32, f32); 4], job_name: &str) {
    let base_model: &'static str =
        job_name.strip_suffix("__nlpp_paropt").unwrap().to_string().leak();

    let init_cfg = NlppConfig {
        base_model,
        preds_dir: SPLIT_NEW.preds_dir,
        n_als_iters: 2,
        reg_a, reg_b,
        shrinkage_u: 10.0,
        shrinkage_i: 25.0,
        regs_path: None,
    };

    // paropt bypasses fit2_inner so we manage the log file manually.
    fs::create_dir_all(init_cfg.preds_dir).unwrap();
    let log_path = format!("{}/{}__nlpp_paropt.out", init_cfg.preds_dir, init_cfg.base_model);
    *netflix_prize::LOG_FILE.lock().unwrap() = Some(BufWriter::new(
        File::create(&log_path).unwrap()
    ));

    netflix_prize::teeln!("[{}__nlpp_paropt]", init_cfg.base_model);
    netflix_prize::teeln!("{:?}", init_cfg);

    let tr = Dataset::load(SPLIT_NEW.tr, "rtg", init_cfg.preds_dir);
    let pr = Dataset::load(SPLIT_NEW.pr, "rtg", init_cfg.preds_dir);
    let mut model = NlppModel::new_with_dataset(&tr, &pr, init_cfg);
    let (best_reg_a, best_reg_b) = model.fit_with_nm(&tr, &pr, 150, 0.5);

    let mut regs = vec![0.0_f32; 16];
    for k in 0..4 {
        regs[2 * k]         = best_reg_a[k].0;
        regs[2 * k + 1]     = best_reg_a[k].1;
        regs[8 + 2 * k]     = best_reg_b[k].0;
        regs[8 + 2 * k + 1] = best_reg_b[k].1;
    }
    let regs_path = format!("{}/{}__nlpp_regs.npy", init_cfg.preds_dir, init_cfg.base_model);
    write_npy(&regs_path, &Array1::from(regs)).unwrap();
    netflix_prize::teeln!("Saved regs to {}", regs_path);

    if let Some(mut lf) = netflix_prize::LOG_FILE.lock().unwrap().take() {
        use std::io::Write as _;
        let _ = lf.flush();
    }
}

fn run_knn3(job_name: &str, cfg: Knn3Config) {
    let base = job_name.strip_suffix("__knn3").unwrap();
    let target = format!("1.0*{}", base);
    fit2!(Knn3Model, cfg, &target, job_name, SPLIT_NEW);
}

fn run_knnf(job_name: &str) {
    let factors_path: &'static str =
        format!("{}/tsvdx5-120.ifeat", SPLIT_NEW.preds_dir).leak();
    let cfg = KnnfConfig {
        factors: factors_path,
        k: 25,
        scaling: 3.5,
        tau: 0.1,
    };
    fit2!(KnnfModel, cfg, "1.0*tsvdx5-150__nlpp", job_name, SPLIT_NEW);
}

fn run_knn_w(job_name: &str) {
    let cfg = KnnConfig {
        stat0: Some("bin_wsupp"),
        stat1: None,
        stat2: None,
        factors: None,
        sim_type: SimType::Support,
        k: 50,
        sim_threshold: None,
        shrinkage: 100.0,
        scaling: 1.5,
        tau: 0.05,
        use_stat1: false,
        regression: false,
        regression_lambda: 0.1,
        sim_dir: SPLIT_NEW.sim_dir,
        preds_dir: SPLIT_NEW.preds_dir,
    };
    fit2!(KnnModel, cfg, "1.0*tsvdx5-300__nlpp", job_name, SPLIT_NEW);
}

fn run_eblend(weights: &[f64], offset: f64, job_name: &str) {
    // Weights and offset extracted from predsx/<base>.out (truncated to 6 decimals).
    // Original eblend included non-epoch columns; their contribution is absorbed
    // into the constant offset (means × weights).
    let base = job_name.strip_suffix("__epochs").unwrap();
    epoch_blend_apply(base, weights, offset, SPLIT_NEW);
}
