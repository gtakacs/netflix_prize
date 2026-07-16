use std::env;
use netflix_prize::{
    OrdinalHeadConfig, SPLIT_NEW,
    epoch_blend_apply, fit2,
    knn::{KnnConfig, KnnModel, SimType},
    knn3::{Knn3Config, Knn3Model},
    knnf::{KnnfConfig, KnnfModel},
    knns::{KnnsConfig, KnnsModel, SuppSource},
    tx::{TxConfig, TxModel},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    match job_name {
        // === lm cluster: shared baseline (60-style lr/reg/sigma) ===
        "tsvdx4-60"     => run_lm(   60, 20,   42, 29, false, false, true,  false, true,  job_name),
        "tsvdx4-400lm"  => run_lm(  400, 20,   42, 29, true,  true,  true,  true,  true,  job_name),
        "tsvdx4-1600lm" => run_lm( 1600,  2, 1600, 29, false, true,  false, false, true,  job_name),
        "tsvdx4-2400lm" => run_lm( 2400, 25, 2400, 32, false, true,  true,  true,  false, job_name),
        "tsvdx4-4800lm" => run_lm( 4800, 27, 2400, 32, true,  true,  true,  false, true,  job_name),

        // === 60o/80o/160o cluster: shared baseline + ord head, only n_feat differs ===
        "tsvdx4-60o"  => run_small_o( 60, true,  job_name),
        "tsvdx4-80o"  => run_small_o( 80, false, job_name),
        "tsvdx4-160o" => run_small_o(160, false, job_name),

        // === Other base trainers (each tuned uniquely) ===
        "tsvdx4-d0tb"  => run_d0tb(job_name),
        "tsvdx4-f1"    => run_f1(job_name),
        "tsvdx4-h8nb"  => run_h8nb(job_name),
        "tsvdx4-l32nb" => run_l32nb(job_name),
        "tsvdx4-m1n"   => run_m1n(job_name),
        "tsvdx4-q16tb" => run_q16tb(job_name),
        "tsvdx4-s20fo" => run_s20fo(job_name),
        "tsvdx4-t8cu"  => run_t8cu(job_name),
        "tsvdx4-16-t"  => run_16_t(job_name),
        "tsvdx4-n8nbo" => run_n8nbo(job_name),
        "tsvdx4-p0tbo" => run_p0tbo(job_name),

        // === Knn3 chains ===
        "tsvdx4-d0tb__knn3"
        | "tsvdx4-h8nb__knn3"
        | "tsvdx4-n8nbo__knn3"
        | "tsvdx4-p0tbo__knn3" => run_knn3(job_name),

        // === Knns chains ===
        "tsvdx4-60__knns"     => run_knns(25, 1.0,  0.025, "1.0", job_name),
        "tsvdx4-60o__knns"    => run_knns(50, 0.5,  0.0,   "1.0", job_name),
        "tsvdx4-2400lm__knns" => run_knns(15, 1.0,  0.5,   "1.2", job_name),
        "tsvdx4-4800lm__knns" => run_knns(40, 3.0,  0.05,  "0.9", job_name),

        // === Knnf chains (ifeat from base) ===
        "tsvdx4-16-t__knnf"
        | "tsvdx4-400lm__knnf"
        | "tsvdx4-4800lm__knnf" => run_knnf(job_name),

        // === Knn-d (residual stat-based kNN with sim deps) ===
        "tsvdx4-4800lm__knn-d" => run_knn_d(job_name),

        // === Epoch blends (apply pre-computed weights, no training) ===
        "tsvdx4-60__epochs" => run_eblend(
            &[
                0.037902, 0.038349, 0.038870, 0.039168, 0.039484,
                0.039743, 0.040036, 0.040299, 0.040508, 0.040723,
                0.040934, 0.041124, 0.039802, 0.040261, 0.040536,
                0.040834, 0.041041, 0.041260, 0.041453, 0.041609,
            ],
            0.628000,
            job_name,
        ),
        "tsvdx4-60o__epochs" => run_eblend(
            &[
                0.027400, 0.028007, 0.028270, 0.028537, 0.028736,
                0.029028, 0.029214, 0.029395, 0.029596, 0.029754,
                0.029904, 0.030050, 0.030198, 0.030342, 0.030463,
                0.030626, 0.030730, 0.030845, 0.030969, 0.031096,
                0.031185, 0.031290, 0.031392, 0.031510, 0.031593,
                0.031697, 0.031757, 0.031838,
            ],
            0.487734,
            job_name,
        ),
        "tsvdx4-d0tb__epochs" => run_eblend(
            &[0.127633, 0.128381, 0.128978, 0.129290, 0.129347],
            0.978328,
            job_name,
        ),
        "tsvdx4-f1__epochs" => run_eblend(
            &[0.127902, 0.128805, 0.129345, 0.129645, 0.129651],
            0.972784,
            job_name,
        ),
        "tsvdx4-q16tb__epochs" => run_eblend(
            &[
                0.091397, 0.092004, 0.092560, 0.093097, 0.093596,
                0.094026, 0.094293, 0.094378,
            ],
            0.702795,
            job_name,
        ),
        "tsvdx4-400lm__epochs" => run_eblend(
            &[
                0.039362, 0.039928, 0.040560, 0.040957, 0.041354,
                0.041709, 0.042063, 0.042378, 0.042626, 0.042909,
                0.043160, 0.043400, 0.041114, 0.041567, 0.041924,
                0.042314, 0.042636, 0.042944, 0.043212, 0.043466,
            ],
            0.498104,
            job_name,
        ),

        _ => panic!("invalid job name: {}", job_name),
    }
}

fn run_lm(
    n_feat: usize,
    n_epochs: usize,
    seed: u64,
    n_time_bins: usize,
    save_ifeat: bool,
    low_memory: bool,
    save_train: bool,
    save_subscores: bool,
    save_probe_each_epoch: bool,
    job_name: &str,
) {
    let cfg = TxConfig {
        n_feat, n_epochs, seed,
        shuffle_users: true,
        n_time_bins,
        beta: 0.3, n_freq_bins: 16,
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
        reset_u_epoch: 13,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat, low_memory, full_su: true,
    };
    match (save_train, save_subscores, save_probe_each_epoch) {
        (true,  false, true)  => fit2!(TxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_probe_each_epoch: true),
        (true,  true,  true)  => fit2!(TxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_subscores: true, save_probe_each_epoch: true),
        (false, false, true)  => fit2!(TxModel, cfg, "rtg", job_name, SPLIT_NEW, save_probe_each_epoch: true),
        (true,  true,  false) => fit2!(TxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_subscores: true),
        (true,  false, false) => fit2!(TxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true),
        _ => panic!("unsupported opts combination for {}", job_name),
    }
}

fn run_small_o(n_feat: usize, save_train: bool, job_name: &str) {
    let cfg = TxConfig {
        n_feat, n_epochs: 28, seed: 64,
        shuffle_users: true,
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
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
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
    if save_train {
        fit2!(TxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_probe_each_epoch: true);
    } else {
        fit2!(TxModel, cfg, "rtg", job_name, SPLIT_NEW, save_subscores: true);
    }
}

fn run_d0tb(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 0, n_epochs: 5, seed: 8003,
        shuffle_users: true,
        n_time_bins: 29, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.0, lr_ud: 0.0, lr_u2: 0.0,
        lr_ub: 0.012, lr_ubd: 0.012,
        lr_i: 0.0, lr_ib: 0.012,
        lr_y: 0.0, lr_yb: 0.0, lr_yd: 0.0,
        lr_tu: 0.003, lr_ti: 0.001, lr_ta: 0.0001,
        lr_ibf: 0.0003, lr_iqf: 0.0, lr_cu: 0.006,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.0504, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.00735, reg_y: 0.04, reg_yd: 0.02667,
        sigma_iqf: 0.0, sigma_u: 0.0, sigma_i: 0.0,
        sigma_y: 0.0, sigma_yd: 0.0,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true, save_probe_each_epoch: true
    );
}

fn run_f1(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 1, n_epochs: 5, seed: 8005,
        shuffle_users: true,
        n_time_bins: 29, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.009, lr_ud: 0.0, lr_u2: 2.1e-5,
        lr_ub: 0.0093, lr_ubd: 0.009,
        lr_i: 0.0108, lr_ib: 0.0108,
        lr_y: 0.0015, lr_yb: 7.5e-5, lr_yd: 0.0,
        lr_tu: 0.0, lr_ti: 0.000675, lr_ta: 6.75e-5,
        lr_ibf: 0.00015, lr_iqf: 1.5e-5, lr_cu: 0.006,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.0504, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.00735, reg_y: 0.04, reg_yd: 0.02667,
        sigma_iqf: 0.005, sigma_u: 0.0015, sigma_i: 0.005,
        sigma_y: 0.00333, sigma_yd: 0.0,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_probe_each_epoch: true
    );
}

fn run_h8nb(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 8, n_epochs: 5, seed: 8007,
        shuffle_users: true,
        n_time_bins: 29, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.012, lr_ud: 0.0, lr_u2: 3e-5,
        lr_ub: 0.0, lr_ubd: 0.0,
        lr_i: 0.015, lr_ib: 0.0,
        lr_y: 0.003, lr_yb: 0.0, lr_yd: 0.0,
        lr_tu: 0.0, lr_ti: 0.0, lr_ta: 0.0,
        lr_ibf: 0.0, lr_iqf: 0.0, lr_cu: 0.0,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.03, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.005, reg_y: 0.03, reg_yd: 0.02667,
        sigma_iqf: 0.0, sigma_u: 0.003, sigma_i: 0.008,
        sigma_y: 0.005, sigma_yd: 0.0,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true, save_probe_each_epoch: true
    );
}

fn run_l32nb(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 32, n_epochs: 5, seed: 8011,
        shuffle_users: true,
        n_time_bins: 29, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.01, lr_ud: 0.0, lr_u2: 2.5e-5,
        lr_ub: 0.0, lr_ubd: 0.0,
        lr_i: 0.02, lr_ib: 0.0,
        lr_y: 0.002, lr_yb: 0.0, lr_yd: 0.0,
        lr_tu: 0.0, lr_ti: 0.0, lr_ta: 0.0,
        lr_ibf: 0.0, lr_iqf: 0.0, lr_cu: 0.0,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.03, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.004, reg_y: 0.03, reg_yd: 0.02667,
        sigma_iqf: 0.0, sigma_u: 0.003, sigma_i: 0.008,
        sigma_y: 0.005, sigma_yd: 0.0,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_probe_each_epoch: true
    );
}

fn run_m1n(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 1, n_epochs: 5, seed: 8012,
        shuffle_users: true,
        n_time_bins: 29, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.0, lr_ud: 0.0, lr_u2: 0.0,
        lr_ub: 0.015, lr_ubd: 0.015,
        lr_i: 0.0108, lr_ib: 0.0108,
        lr_y: 0.004, lr_yb: 0.0002, lr_yd: 0.0,
        lr_tu: 0.0, lr_ti: 0.000675, lr_ta: 6.75e-5,
        lr_ibf: 0.00015, lr_iqf: 1.5e-5, lr_cu: 0.006,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.0504, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.00735, reg_y: 0.03, reg_yd: 0.02667,
        sigma_iqf: 0.005, sigma_u: 0.0, sigma_i: 0.005,
        sigma_y: 0.008, sigma_yd: 0.0,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(TxModel, cfg, "rtg", job_name, SPLIT_NEW);
}

fn run_q16tb(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 16, n_epochs: 8, seed: 9001,
        shuffle_users: true,
        n_time_bins: 48, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.005, lr_ud: 0.002, lr_u2: 1e-5,
        lr_ub: 0.005, lr_ubd: 0.005,
        lr_i: 0.006, lr_ib: 0.006,
        lr_y: 0.001, lr_yb: 5e-5, lr_yd: 0.0004,
        lr_tu: 0.0003, lr_ti: 0.0004, lr_ta: 4e-5,
        lr_ibf: 0.0001, lr_iqf: 1e-5, lr_cu: 0.004,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.04, reg_u2: 0.4, reg_ud: 0.03,
        reg_i: 0.006, reg_y: 0.03, reg_yd: 0.02,
        sigma_iqf: 0.005, sigma_u: 0.002, sigma_i: 0.006,
        sigma_y: 0.004, sigma_yd: 0.01,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true, save_probe_each_epoch: true
    );
}

fn run_s20fo(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 20, n_epochs: 10, seed: 9003,
        shuffle_users: true,
        n_time_bins: 25, beta: 0.3, n_freq_bins: 24,
        lr_u: 0.004, lr_ud: 0.002, lr_u2: 1.5e-5,
        lr_ub: 0.007, lr_ubd: 0.007,
        lr_i: 0.008, lr_ib: 0.008,
        lr_y: 0.0015, lr_yb: 5e-6, lr_yd: 0.0004,
        lr_tu: 0.0, lr_ti: 0.00015, lr_ta: 2e-5,
        lr_ibf: 8e-5, lr_iqf: 8e-6, lr_cu: 0.003,
        reg_iqf: 0.005, reg_cu: 0.008,
        reg_u: 0.02, reg_u2: 0.3, reg_ud: 0.02,
        reg_i: 0.004, reg_y: 0.02, reg_yd: 0.02,
        sigma_iqf: 0.005, sigma_u: 0.0, sigma_i: 0.005,
        sigma_y: 0.005, sigma_yd: 0.005,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: Some(OrdinalHeadConfig {
            th_init: [0.5, 1.25, 3.75, 5.5],
            th_gap: 0.001,
            lr_t: 0.0,
            reg_t: 0.0,
        }),
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true, save_probe_each_epoch: true
    );
}

fn run_t8cu(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 8, n_epochs: 8, seed: 9004,
        shuffle_users: true,
        n_time_bins: 29, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.006, lr_ud: 0.0, lr_u2: 1.5e-5,
        lr_ub: 0.006, lr_ubd: 0.006,
        lr_i: 0.008, lr_ib: 0.008,
        lr_y: 0.005, lr_yb: 0.0005, lr_yd: 0.0,
        lr_tu: 0.0, lr_ti: 0.0003, lr_ta: 3e-5,
        lr_ibf: 0.0001, lr_iqf: 1e-5, lr_cu: 0.05,
        reg_iqf: 0.007, reg_cu: 0.003,
        reg_u: 0.04, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.006, reg_y: 0.02, reg_yd: 0.02667,
        sigma_iqf: 0.005, sigma_u: 0.003, sigma_i: 0.008,
        sigma_y: 0.008, sigma_yd: 0.0,
        reset_u_epoch: 3,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true, save_probe_each_epoch: true
    );
}

fn run_16_t(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 16, n_epochs: 4, seed: 9001,
        shuffle_users: true,
        n_time_bins: 22, beta: 0.35, n_freq_bins: 12,
        lr_u: 0.004, lr_ud: 0.0015, lr_u2: 1e-5,
        lr_ub: 0.004, lr_ubd: 0.0035,
        lr_i: 0.0025, lr_ib: 0.0028,
        lr_y: 0.0004, lr_yb: 2e-5, lr_yd: 0.0002,
        lr_tu: 0.0, lr_ti: 0.0002, lr_ta: 2e-5,
        lr_ibf: 4e-5, lr_iqf: 0.0, lr_cu: 0.0015,
        reg_iqf: 0.007, reg_cu: 0.008,
        reg_u: 0.008, reg_u2: 0.3, reg_ud: 0.035,
        reg_i: 0.045, reg_y: 0.035, reg_yd: 0.025,
        sigma_iqf: 0.0, sigma_u: 0.006, sigma_i: 0.0012,
        sigma_y: 0.003, sigma_yd: 0.008,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat: true, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true, save_probe_each_epoch: true, transpose: true
    );
}

fn run_n8nbo(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 8, n_epochs: 5, seed: 8013,
        shuffle_users: true,
        n_time_bins: 20, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.015, lr_ud: 0.0, lr_u2: 0.0,
        lr_ub: 0.0, lr_ubd: 0.0,
        lr_i: 0.018, lr_ib: 0.0,
        lr_y: 0.003, lr_yb: 0.0, lr_yd: 0.0,
        lr_tu: 0.0, lr_ti: 0.0, lr_ta: 0.0,
        lr_ibf: 0.0, lr_iqf: 0.0, lr_cu: 0.0,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.03, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.005, reg_y: 0.03, reg_yd: 0.02667,
        sigma_iqf: 0.0, sigma_u: 0.003, sigma_i: 0.008,
        sigma_y: 0.005, sigma_yd: 0.0,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: Some(OrdinalHeadConfig {
            th_init: [0.5, 1.25, 3.75, 5.5],
            th_gap: 0.001,
            lr_t: 0.0,
            reg_t: 0.0,
        }),
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true
    );
}

fn run_p0tbo(job_name: &str) {
    let cfg = TxConfig {
        n_feat: 0, n_epochs: 5, seed: 8014,
        shuffle_users: true,
        n_time_bins: 29, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.0, lr_ud: 0.0, lr_u2: 0.0,
        lr_ub: 0.018, lr_ubd: 0.018,
        lr_i: 0.0, lr_ib: 0.015,
        lr_y: 0.0, lr_yb: 0.0, lr_yd: 0.0,
        lr_tu: 0.003, lr_ti: 0.001, lr_ta: 0.0001,
        lr_ibf: 0.0003, lr_iqf: 0.0, lr_cu: 0.006,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.0504, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.00735, reg_y: 0.04, reg_yd: 0.02667,
        sigma_iqf: 0.0, sigma_u: 0.0, sigma_i: 0.0,
        sigma_y: 0.0, sigma_yd: 0.0,
        reset_u_epoch: 1024,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: Some(OrdinalHeadConfig {
            th_init: [0.5, 1.25, 3.75, 5.5],
            th_gap: 0.001,
            lr_t: 0.0,
            reg_t: 0.0,
        }),
        save_ifeat: false, low_memory: true, full_su: true,
    };
    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true
    );
}

fn run_knn3(job_name: &str) {
    let base = job_name.strip_suffix("__knn3").unwrap();
    let target = format!("1.0*{}", base);
    fit2!(Knn3Model, Knn3Config::default(), &target, job_name, SPLIT_NEW);
}

fn run_knns(k: usize, scaling: f32, tau: f32, target_weight: &str, job_name: &str) {
    let cfg = KnnsConfig {
        k,
        shrinkage: 100.0,
        scaling,
        tau,
        supp_source: SuppSource::Compute,
    };
    let base = job_name.strip_suffix("__knns").unwrap();
    let target = format!("{}*{}", target_weight, base);
    fit2!(KnnsModel, cfg, &target, job_name, SPLIT_NEW);
}

fn run_knnf(job_name: &str) {
    let base = job_name.strip_suffix("__knnf").unwrap();
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

fn run_knn_d(job_name: &str) {
    let cfg = KnnConfig {
        stat0: Some("rtg_supp"),
        stat1: Some("tsvdx4-4800lm_diff1"),
        stat2: None,
        factors: None,
        sim_type: SimType::Support,
        k: 15,
        sim_threshold: None,
        shrinkage: 100.0,
        scaling: 3.5,
        tau: 0.0,
        use_stat1: true,
        regression: false,
        regression_lambda: 0.1,
        sim_dir: SPLIT_NEW.sim_dir,
        preds_dir: SPLIT_NEW.preds_dir,
    };
    fit2!(KnnModel, cfg, "1.0*tsvdx4-4800lm", job_name, SPLIT_NEW);
}

fn run_eblend(weights: &[f64], offset: f64, job_name: &str) {
    // Weights and offset extracted from predsx/<base>.out (truncated to 6 decimals).
    let base = job_name.strip_suffix("__epochs").unwrap();
    epoch_blend_apply(base, weights, offset, SPLIT_NEW);
}
