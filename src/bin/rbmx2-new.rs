#![allow(deprecated)]
use std::env;
use netflix_prize::{
    SPLIT_NEW,
    asym::{AsymConfig, AsymModel},
    epoch_blend_apply, fit2,
    knn3::{Knn3Config, Knn3Model},
    knnf::{KnnfConfig, KnnfModel},
    knns::{KnnsConfig, KnnsModel, SuppSource},
    rbmx2::{Rbmx2Config, Rbmx2Model},
    rx::{HiddenType, RxConfig, RxModel, VisibleType},
    tx::{TxConfig, TxModel},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    match job_name {
        // === RxModel base jobs ===
        "rbmx2-0" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 0,
            n_epochs: 4,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.005,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: true,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0002,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "rbmx2-8e" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::TruncExp(0.0, 1.0),
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 8,
            n_epochs: 4,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.005,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.0,
            wd_bu: 0.01,
            lr_but: 0.0,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: true,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "rbmx2-12" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 12,
            n_epochs: 8,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0002,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "rbmx2-14e" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::TruncExp(-1.0, 1.0),
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 14,
            n_epochs: 4,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.005,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0002,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW);
        }

        "rbmx2-16-t" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::TruncExp(-1.0, 1.0),
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 16,
            n_epochs: 8,
            seed: 9002,
            shuffle_users: true,
            init_sigma: 0.008,
            batch_size: 100,
            lr: 0.004,
            momentum: 0.9,
            weight_decay: 0.0012,
            lr_bu: 0.0,
            wd_bu: 0.01,
            lr_but: 0.0,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            n_freq_bins: 0,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, transpose: true);
        }

        "rbmx2-32-t" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 32,
            n_epochs: 1,
            seed: 9002,
            shuffle_users: true,
            init_sigma: 0.008,
            batch_size: 50,
            lr: 0.004,
            momentum: 0.9,
            weight_decay: 0.0012,
            lr_bu: 0.0,
            wd_bu: 0.01,
            lr_but: 0.0,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            n_freq_bins: 0,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, transpose: true);
        }

        "rbmx2-100e" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::TruncExp(0.0, 1.0),
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 100,
            n_epochs: 15,
            seed: 200,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.008,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.002,
            wd_bu: 0.01,
            lr_but: 0.001,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: true,
            n_factors: None,
            n_freq_bins: 12,
            lr_bif: 0.0003,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_probe_each_epoch: true);
        }

        "rbmx2-160f" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 160,
            n_epochs: 20,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: Some(40),
            n_freq_bins: 8,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_probe_each_epoch: true);
        }

        "rbmx2-300s" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 300,
            n_epochs: 60,
            seed: 256,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 1000,
            lr: 0.015,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 4,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: true,
            n_factors: None,
            n_freq_bins: 12,
            lr_bif: 0.0005,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_probe_each_epoch: true);
        }

        "rbmx2-350e" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::TruncExp(0.0, 1.0),
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 350,
            n_epochs: 20,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 250,
            lr: 0.005,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 4,
            cd_inc_by: 1,
            cd_max: 4,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_probe_each_epoch: true);
        }

        "rbmx2-400" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 400,
            n_epochs: 30,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.02,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 5,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: true,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_subscores: true);
        }

        "rbmx2-500" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 500,
            n_epochs: 30,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.02,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 5,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: true,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "rbmx2-500bp" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bipolar,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 500,
            n_epochs: 30,
            seed: 350,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.015,
            momentum: 0.9,
            weight_decay: 0.0008,
            lr_bu: 0.002,
            wd_bu: 0.01,
            lr_but: 0.001,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 4,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: true,
            n_factors: None,
            n_freq_bins: 12,
            lr_bif: 0.0003,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true, save_probe_each_epoch: true);
        }

        "rbmx2-600" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 600,
            n_epochs: 27,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.02,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 5,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_probe_each_epoch: true);
        }

        "rbmx2-800bp" => {
            let cfg = RxConfig {
    hidden_type: HiddenType::Bipolar,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 800,
            n_epochs: 3,
            seed: 700,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 1000,
            lr: 0.02,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 5,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            n_freq_bins: 8,
            lr_bif: 0.0,
            wd_bif: 0.01,
            };
            fit2!(RxModel, cfg, "rtg", job_name, SPLIT_NEW, save_probe_each_epoch: true);
        }

        // === Rbmx2Model base jobs (deprecated) ===
        "rbmx2-10" => {
            let cfg = Rbmx2Config {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 10,
            n_epochs: 30,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 3,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            mf_dim: 0,
            lr_mf_u: 0.001,
            lr_mf_i: 0.0005,
            wd_mf: 0.01,
            n_freq_bins: 8,
            lr_bif: 0.001,
            wd_bif: 0.01,
            lr_bif_bug: true,
            };
            fit2!(Rbmx2Model, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "rbmx2-40" => {
            let cfg = Rbmx2Config {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 40,
            n_epochs: 30,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.04,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 5,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            mf_dim: 0,
            lr_mf_u: 0.001,
            lr_mf_i: 0.0005,
            wd_mf: 0.01,
            n_freq_bins: 8,
            lr_bif: 0.001,
            wd_bif: 0.01,
            lr_bif_bug: true,
            };
            fit2!(Rbmx2Model, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "rbmx2-60" => {
            let cfg = Rbmx2Config {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 60,
            n_epochs: 30,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.04,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 5,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            mf_dim: 0,
            lr_mf_u: 0.001,
            lr_mf_i: 0.0005,
            wd_mf: 0.01,
            n_freq_bins: 8,
            lr_bif: 0.001,
            wd_bif: 0.01,
            lr_bif_bug: true,
            };
            fit2!(Rbmx2Model, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "rbmx2-150" => {
            let cfg = Rbmx2Config {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 150,
            n_epochs: 30,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.03,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 5,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: true,
            n_factors: None,
            mf_dim: 0,
            lr_mf_u: 0.001,
            lr_mf_i: 0.0005,
            wd_mf: 0.01,
            n_freq_bins: 8,
            lr_bif: 0.001,
            wd_bif: 0.01,
            lr_bif_bug: true,
            };
            fit2!(Rbmx2Model, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        "rbmx2-200" => {
            let cfg = Rbmx2Config {
    hidden_type: HiddenType::Bernoulli,
            visible_type: VisibleType::Softmax,
            temperature: 1.0,
            n_hidden: 200,
            n_epochs: 30,
            seed: 128,
            shuffle_users: true,
            init_sigma: 0.01,
            batch_size: 500,
            lr: 0.025,
            momentum: 0.9,
            weight_decay: 0.001,
            lr_bu: 0.001,
            wd_bu: 0.01,
            lr_but: 0.0005,
            wd_but: 0.01,
            cd_start: 1,
            cd_inc_every: 5,
            cd_inc_by: 1,
            cd_max: 5,
            use_conditional: true,
            r_include_pr_all: true,
            save_w: false,
            n_factors: None,
            mf_dim: 0,
            lr_mf_u: 0.001,
            lr_mf_i: 0.0005,
            wd_mf: 0.01,
            n_freq_bins: 8,
            lr_bif: 0.001,
            wd_bif: 0.01,
            lr_bif_bug: true,
            };
            fit2!(Rbmx2Model, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
        }

        // === TxModel residual chain ===
        "rbmx2-10__tsvdx4-10lm" => {
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
            ordinal_head: None,
            save_ifeat: true,
            low_memory: true,
            full_su: true,
            };
            fit2!(TxModel, cfg, "0.9*rbmx2-10", job_name, SPLIT_NEW, save_train: true, save_subscores: true, save_probe_each_epoch: true);
        }

        // === AsymModel chain ===
        "rbmx2-12__asym-16" => {
            let cfg = AsymConfig {
    n_feat: 16,
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
            save_ifeat: true,
            };
            fit2!(AsymModel, cfg, "rbmx2-12", job_name, SPLIT_NEW, save_train: true);
        }

        // === Knn3 chains ===
        "rbmx2-10__knn3"          => run_knn3("1.0*rbmx2-10",         job_name),
        "rbmx2-12__asym-16__knn3" => run_knn3("1.0*rbmx2-12__asym-16", job_name),
        "rbmx2-40__knn3"          => run_knn3("1.0*rbmx2-40",         job_name),
        "rbmx2-60__knn3"          => run_knn3("1.0*rbmx2-60",         job_name),
        "rbmx2-200__knn3"         => run_knn3("rbmx2-200",            job_name),
        "rbmx2-300s__knn3"        => run_knn3("1.0*rbmx2-300s",       job_name),
        "rbmx2-500__knn3"         => run_knn3("1.0*rbmx2-500",        job_name),

        // === Knnf chains ===
        "rbmx2-12__asym-16__knnf" => {
            let factors_path: &'static str =
                format!("{}/rbmx2-12__asym-16.ifeat", SPLIT_NEW.preds_dir).leak();
            let cfg = KnnfConfig::with_factors(factors_path);
            fit2!(KnnfModel, cfg, "1.0*rbmx2-12__asym-16", job_name, SPLIT_NEW);
        }

        "rbmx2-150__knnf" => {
            let factors_path: &'static str =
                format!("{}/rbmx2-150.ifeat", SPLIT_NEW.preds_dir).leak();
            let cfg = KnnfConfig::with_factors(factors_path);
            fit2!(KnnfModel, cfg, "rbmx2-150", job_name, SPLIT_NEW);
        }

        "rbmx2-300s__knnf" => {
            let factors_path: &'static str =
                format!("{}/rbmx2-300s.ifeat", SPLIT_NEW.preds_dir).leak();
            let cfg = KnnfConfig::with_factors(factors_path);
            fit2!(KnnfModel, cfg, "1.0*rbmx2-300s", job_name, SPLIT_NEW);
        }

        "rbmx2-400__knnf" => {
            let factors_path: &'static str =
                format!("{}/rbmx2-400.ifeat", SPLIT_NEW.preds_dir).leak();
            let cfg = KnnfConfig::with_factors(factors_path);
            fit2!(KnnfModel, cfg, "1.0*rbmx2-400", job_name, SPLIT_NEW);
        }

        "rbmx2-500__knnf" => {
            let factors_path: &'static str =
                format!("{}/rbmx2-500.ifeat", SPLIT_NEW.preds_dir).leak();
            let cfg = KnnfConfig {
    factors: factors_path,
            k: 15,
            scaling: 1.5,
            tau: 0.0,
            };
            fit2!(KnnfModel, cfg, "1.1*rbmx2-500", job_name, SPLIT_NEW);
        }

        "rbmx2-500bp__knnf" => {
            let factors_path: &'static str =
                format!("{}/rbmx2-500bp.ifeat", SPLIT_NEW.preds_dir).leak();
            let cfg = KnnfConfig::with_factors(factors_path);
            fit2!(KnnfModel, cfg, "1.0*rbmx2-500bp", job_name, SPLIT_NEW);
        }

        "rbmx2-8e__knnf" => {
            let factors_path: &'static str =
                format!("{}/rbmx2-8e.ifeat", SPLIT_NEW.preds_dir).leak();
            let cfg = KnnfConfig::with_factors(factors_path);
            fit2!(KnnfModel, cfg, "1.0*rbmx2-8e", job_name, SPLIT_NEW);
        }

        // === Knns chain ===
        "rbmx2-150__knns" => {
            let cfg = KnnsConfig {
    k: 50,
            shrinkage: 100.0,
            scaling: 0.5,
            tau: 0.025,
            supp_source: SuppSource::Compute,
            };
            fit2!(KnnsModel, cfg, "0.9*rbmx2-150", job_name, SPLIT_NEW);
        }

        // === Epoch blends ===
        "rbmx2-10__tsvdx4-10lm__epochs" => run_eblend(
            &[
                0.062234, 0.062410, 0.062627, 0.062817, 0.063013,
                0.063148, 0.063364, 0.063531, 0.063660, 0.063804,
                0.063966, 0.064068,
            ],
            0.726987,
            job_name,
        ),
        "rbmx2-160f__epochs" => run_eblend(
            &[
                0.039839, 0.040495, 0.041063, 0.041410, 0.041169,
                0.041443, 0.041796, 0.041525, 0.042036, 0.041783,
                0.042111, 0.042405, 0.042201, 0.042275, 0.042449,
                0.042727, 0.042892, 0.042685, 0.042857, 0.042883,
            ],
            0.494455,
            job_name,
        ),
        "rbmx2-350e__epochs" => run_eblend(
            &[
                0.038945, 0.039426, 0.039901, 0.040269, 0.040546,
                0.040924, 0.041190, 0.041466, 0.041739, 0.041973,
                0.042230, 0.042485, 0.042655, 0.042879, 0.043074,
                0.043291, 0.043423, 0.043583, 0.043773, 0.043853,
            ],
            0.511822,
            job_name,
        ),

        _ => panic!("invalid job name: {}", job_name),
    }
}

fn run_knn3(target: &str, job_name: &str) {
    fit2!(Knn3Model, Knn3Config::default(), target, job_name, SPLIT_NEW);
}

fn run_eblend(weights: &[f64], offset: f64, job_name: &str) {
    let base = job_name.strip_suffix("__epochs").unwrap();
    epoch_blend_apply(base, weights, offset, SPLIT_NEW);
}
