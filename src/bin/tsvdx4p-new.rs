use std::env;
use netflix_prize::{
    fit2,
    knnf::{KnnfConfig, KnnfModel},
    tsvdx4p::{Tsvdx4pConfig, Tsvdx4pModel},
    SPLIT_NEW,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    match job_name {
        "tsvdx4p-1000lm"  => run_lm(1000,  true,  [1, 10, 20, 0, 0], true,  job_name),
        "tsvdx4p-2000lm"  => run_lm(2000,  true,  [1, 10, 20, 0, 0], true,  job_name),
        "tsvdx4p-4000lm"  => run_lm(4000,  false, [1, 10, 20, 0, 0], false, job_name),
        "tsvdx4p-6000lm"  => run_lm(6000,  true,  [0, 0, 0, 0, 0],   true,  job_name),
        "tsvdx4p-8000lm"  => run_lm(8000,  true,  [1, 10, 20, 0, 0], true,  job_name),
        "tsvdx4p-16000lm" => run_lm(16000, false, [1, 10, 20, 0, 0], false, job_name),

        "tsvdx4p-1000lm__knnf"
        | "tsvdx4p-2000lm__knnf"
        | "tsvdx4p-6000lm__knnf"
        | "tsvdx4p-8000lm__knnf" => run_knnf(job_name),

        _ => panic!("invalid job name: {}", job_name),
    }
}

fn run_lm(
    n_feat: usize,
    save_ifeat: bool,
    seq_epochs: [usize; 5],
    save_train: bool,
    job_name: &str,
) {
    let cfg = Tsvdx4pConfig {
        n_feat,
        n_epochs: 20,
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
        lr_yb: 0.0002,
        lr_yd: 0.000267,
        lr_tu: 0.0,
        lr_ti: 0.000225,
        lr_ta: 2.25e-5,
        lr_ibf: 5e-5,
        lr_iqf: 5e-6,
        reg_iqf: 0.007,
        sigma_iqf: 0.005,
        lr_cu: 0.002,
        reg_cu: 0.01,
        reg_u: 0.0504,
        reg_u2: 0.4,
        reg_ud: 0.04,
        reg_i: 0.00735,
        reg_y: 0.04,
        reg_yd: 0.02667,
        sigma_u: 0.0015,
        sigma_i: 0.005,
        sigma_y: 0.00333,
        sigma_yd: 0.009,
        reset_u_epoch: 1024,
        save_ifeat,
        low_memory: true,
        full_su: true,
        nsvd_norm_exp: 1.0,
        n_threads: 16,
        seq_epochs,
    };
    if save_train {
        fit2!(Tsvdx4pModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
    } else {
        fit2!(
            Tsvdx4pModel, cfg, "rtg", job_name, SPLIT_NEW,
            save_probe_each_epoch: true
        );
    }
}

fn run_knnf(job_name: &str) {
    let base = job_name.strip_suffix("__knnf").unwrap();
    // Leak the formatted path so it satisfies KnnfConfig's `&'static str`
    // requirement. The leak is one-shot per process — negligible.
    let factors_path: &'static str =
        format!("{}/{}.ifeat", SPLIT_NEW.preds_dir, base).leak();
    let target = format!("1.0*{}", base);
    fit2!(
        KnnfModel,
        KnnfConfig::with_factors(factors_path),
        &target,
        job_name,
        SPLIT_NEW
    );
}
