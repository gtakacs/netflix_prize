use netflix_prize::{
    fit2,
    mfrbmx::{MfConfig, MfRbmxConfig, MfRbmxModel, RbmGvConfig},
    SPLIT_NEW,
};

fn main() {
    let cfg = MfRbmxConfig {
        mf: MfConfig {
            n_feat: 25,
            n_epochs: 9,
            seed: 42,
            lr_u: 0.005,
            lr_i: 0.03,
            lr_ub: 0.006,
            lr_ib: 0.03,
            reg_u: 0.07,
            reg_i: 0.004,
            sigma_u: 0.004,
            sigma_i: 0.005,
            reset_u_epoch: 1024,
        },
        rbm: RbmGvConfig {
            n_hidden: 25,
            n_epochs: 9,
            seed: 42,
            gibbs3_after_epoch: 1024,
            lr_w: 6e-5,
            lr_c: 0.005,
            reg_w: 0.004,
            reg_c: 0.0,
            sigma_w: 0.02,
            sigma_v: 0.8,
            v_shift: 0.0,
            v_scale: 1.0,
            sample_visible: true,
        },
        w_mf: 0.5,
        w_rbm: 1.0,
    };
    fit2!(
        MfRbmxModel, cfg, "rtg", "mfrbmx-50", SPLIT_NEW,
        save_subscores: true
    );
}
