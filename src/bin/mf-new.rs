use std::env;
use netflix_prize::{
    asym::{AsymConfig, AsymModel},
    fit2,
    knn3::{Knn3Config, Knn3Model},
    mf::{MfConfig, MfModel},
    SPLIT_NEW,
};

fn run_mf(n_feat: usize, n_epochs: usize, job_name: &str) {
    let cfg = MfConfig {
        n_feat,
        n_epochs,
        seed: 42,
        shuffle_users: true,
        lr_u: 0.0031,
        lr_i: 0.0036,
        lr_ub: 0.0031,
        lr_ib: 0.0036,
        reg_u: 0.03,
        reg_i: 0.005,
        sigma_u: 0.004,
        sigma_i: 0.005,
        reset_u_epoch: 10,
        item_feat_npy: None,
        ordinal_head: None,
        save_ifeat: false,
    };
    fit2!(MfModel, cfg, "rtg", job_name, SPLIT_NEW, save_train: true);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    match job_name {
        "mf-60" => run_mf(60, 12, job_name),
        "mf-61" => run_mf(61, 20, job_name),

        "mf-60__knn3" | "mf-61__knn3" => {
            let target = format!("1.0*{}", job_name.strip_suffix("__knn3").unwrap());
            fit2!(Knn3Model, Knn3Config::default(), &target, job_name, SPLIT_NEW);
        }

        "mf-61__asym-16" => {
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
                save_ifeat: false,
            };
            fit2!(
                AsymModel, cfg, "mf-61", job_name, SPLIT_NEW,
                save_probe_each_epoch: true
            );
        }

        _ => panic!("invalid job name: {}", job_name),
    }
}
