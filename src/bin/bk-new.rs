use std::env;
use netflix_prize::{
    bk1::{Bk1Config, Bk1Model},
    bk3::{Bk3Config, Bk3Model},
    fit2,
    knns::{KnnsConfig, KnnsModel, SuppSource},
    SPLIT_NEW,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    match job_name {
        "bk1-10" => {
            let cfg = Bk1Config {
                n_feat: 10,
                n_epochs: 12,
                seed: 42,
                shuffle_users: true,
                n_time_bins: 30,
                beta: 0.4,
                k1: 0.0363636,
                k2: 0.909091,
                dev_mean: false,
                k_neighbors: 0,
                alpha_rho: 100.0,
                lambda1: 25.0,
                lambda2: 10.0,
                lr_bias: 0.007,
                lr_fact: 0.007,
                lr_nbr: 0.001,
                lr_h: 0.0,
                lr_decay: 0.0785714,
                reg_bias: 0.005,
                reg_fact: 0.015,
                reg_nbr: 0.015,
                reg_h: 0.0,
            };
            fit2!(
                Bk1Model, cfg, "rtg", job_name, SPLIT_NEW,
                save_train: true,
                save_probe_each_epoch: true
            );
        }

        "bk3-20" => {
            let cfg = Bk3Config {
                n_feat: 20,
                n_epochs: 10,
                seed: 42,
                shuffle_users: true,
                n_time_bins: 30,
                n_time_bins_fact: 8,
                n_freq_bins_bias: 30,
                n_freq_bins_fact: 8,
                beta: 0.4,
                k1: 0.042979,
                k2: 1.0,
                dev_mean: false,
                k_neighbors: 0,
                alpha_rho: 100.0,
                lambda1: 25.0,
                lambda2: 10.0,
                lr_bias: 0.00790541,
                lr_fact: 0.00525646,
                lr_qt: 0.00525646,
                lr_qf: 0.0,
                lr_nbr: 0.000199005,
                lr_h: 0.0406664,
                lr_decay: 0.11587,
                reg_bias: 0.00138669,
                reg_fact: 0.0092125,
                reg_qt: 0.0092125,
                reg_qf: 0.0,
                reg_nbr: 0.030711,
                reg_h: 0.000373603,
            };
            fit2!(Bk3Model, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "bk3-30" => {
            let cfg = Bk3Config {
                n_feat: 30,
                n_epochs: 4,
                seed: 42,
                shuffle_users: true,
                n_time_bins: 30,
                n_time_bins_fact: 8,
                n_freq_bins_bias: 30,
                n_freq_bins_fact: 8,
                beta: 0.4,
                k1: 0.042979,
                k2: 1.0,
                dev_mean: false,
                k_neighbors: 0,
                alpha_rho: 100.0,
                lambda1: 25.0,
                lambda2: 10.0,
                lr_bias: 0.003952705,
                lr_fact: 0.00262823,
                lr_qt: 0.00262823,
                lr_qf: 0.0,
                lr_nbr: 9.95025e-5,
                lr_h: 0.0203332,
                lr_decay: 0.057935,
                reg_bias: 0.00138669,
                reg_fact: 0.0092125,
                reg_qt: 0.0092125,
                reg_qf: 0.0,
                reg_nbr: 0.030711,
                reg_h: 0.000373603,
            };
            fit2!(Bk3Model, cfg, "rtg", job_name, SPLIT_NEW);
        }

        "bk3-20__knns" => {
            let cfg = KnnsConfig {
                k: 50,
                shrinkage: 100.0,
                scaling: 1.5,
                tau: 0.05,
                supp_source: SuppSource::Compute,
            };
            fit2!(KnnsModel, cfg, "0.8*bk3-20", job_name, SPLIT_NEW);
        }

        _ => panic!("invalid job name: {}", job_name),
    }
}
