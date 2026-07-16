use std::env;
use netflix_prize::{aex::{AexConfig, AexModel}, fit2, SPLIT_NEW};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    let (n_feat, reg, dropout) = match job_name {
        "aex-16"  => (16,  0.001, 0.0),
        "aex-32"  => (32,  0.001, 0.0),
        "aex-64"  => (64,  0.002, 0.05),
        "aex-128" => (128, 0.002, 0.1),
        _ => panic!("invalid job name: {}", job_name),
    };

    let cfg = AexConfig {
        n_feat,
        n_epochs: 20,
        seed: 42,
        shuffle_users: true,
        lr: 0.001,
        reg,
        sigma: 0.01,
        normalize: true,
        use_implicit: true,
        dropout,
        lr_ubias: 0.0001,
        lr_udbias: 5e-5,
    };
    fit2!(AexModel, cfg, "rtg", job_name, SPLIT_NEW);
}
