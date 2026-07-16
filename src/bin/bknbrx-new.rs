use std::env;
use netflix_prize::{
    bknbrx::{BknbrxConfig, BknbrxModel, ParallelMode},
    fit2,
    SPLIT_NEW,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    match job_name {
        "bknbrx-50"  => run_bknbrx(50,  25, 1e-5, 1e-7, job_name),
        "bknbrx-100" => run_bknbrx(100,  5, 0.0,  0.0,  job_name),

        _ => panic!("invalid job name: {}", job_name),
    }
}

fn run_bknbrx(max_neighbors: usize, n_epochs: usize, lr_alpha: f32, lr_beta_u: f32, job_name: &str) {
    let cfg = BknbrxConfig {
        n_epochs,
        seed: 42,
        shuffle_users: true,
        parallel_mode: ParallelMode::Hogwild,
        n_threads: 8,

        n_time_bins: 30,
        beta: 0.4,

        lr_bu: 0.003,
        lr_but: 0.0025,
        lr_alpha,
        lr_bi: 0.002,
        lr_bit: 5e-5,
        lr_cu: 0.008,
        lr_cut: 0.002,

        reg_bu: 0.03,
        reg_but: 0.005,
        reg_alpha: 50.0,
        reg_bi: 0.03,
        reg_bit: 0.1,
        reg_cu: 0.01,
        reg_cut: 0.005,

        max_neighbors,
        lr_w: 0.005,
        lr_c: 0.005,
        lr_beta_u,
        reg_w: 0.002,
        reg_c: 0.002,
        reg_beta_u: 0.01,

        lr_w_day: 0.004,
        lr_c_day: 0.004,
        reg_w_day: 0.002,
        reg_c_day: 0.002,

        lambda1: 25.0,
        lambda2: 10.0,
    };
    fit2!(
        BknbrxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_probe_each_epoch: true
    );
}
