use std::env;
use netflix_prize::{
    als8::{Als8Config, Als8Model},
    fit2,
    knn3::{Knn3Config, Knn3Model},
    SPLIT_NEW,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args[1].as_str();

    match job_name {
        "als8-8" => {
            let cfg = Als8Config {
                n_epochs: 10,
                seed: 42,
                n_feat: 8,
                sigma_u: 0.005,
                sigma_i: 0.005,
                reg_ub: 20.0,
                reg_ib: 20.0,
                reg_u: 0.05,
                reg_i: 0.01,
                shrink_m: 40.0,
                use_probe: true,
                n_bias_used: 8,
            };
            fit2!(Als8Model, cfg, "rtg", "als8-8", SPLIT_NEW, save_train: true);
        }
        "als8-8__knn3" => {
            fit2!(Knn3Model, Knn3Config::default(), "1.0*als8-8", "als8-8__knn3", SPLIT_NEW);
        }
        _ => panic!("invalid job name: {}", job_name),
    }
}
