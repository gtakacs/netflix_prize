mod helpers;
use helpers::{make_tiny_train, make_tiny_probe};

use netflix_prize::{MaskedDataset, Regressor, calc_rmse, gen_preds, gen_preds_parallel};
#[allow(deprecated)] use netflix_prize::tsvdx4::{Tsvdx4Config, into_tx_config as cfg4};
#[allow(deprecated)] use netflix_prize::tsvdx5::{Tsvdx5Config, into_tx_config as cfg5};
#[allow(deprecated)] use netflix_prize::tsvdx6::{Tsvdx6Config, into_tx_config as cfg6};
use netflix_prize::tsvdx4p::{Tsvdx4pModel, Tsvdx4pConfig};
use netflix_prize::tx::TxModel;
use netflix_prize::asym::{AsymModel, AsymConfig};
use netflix_prize::knn3::{Knn3Model, Knn3Config};
use netflix_prize::knnf::{KnnfModel, KnnfConfig};
use netflix_prize::knns::{KnnsModel, KnnsConfig, SuppSource};
use netflix_prize::aex::{AexModel, AexConfig};
use netflix_prize::als8::{Als8Model, Als8Config};
use netflix_prize::bk1::{Bk1Model, Bk1Config};
use netflix_prize::cfnade::{CfNadeModel, CfNadeConfig};
use netflix_prize::dnn::{DnnModel, DnnConfig};
use netflix_prize::mf::{MfModel, MfConfig};
use netflix_prize::mfrbmx::{
    MfRbmxModel, MfRbmxConfig,
    MfConfig as MfRbmxMfConfig, RbmGvConfig as MfRbmxRbmConfig,
};
#[allow(deprecated)] use netflix_prize::rbmx2::{Rbmx2Model, Rbmx2Config};
use netflix_prize::rx::{RxModel, RxConfig, HiddenType, VisibleType};
use ndarray::Array2;

macro_rules! cfg {
    ($ty:ident { $($extra:tt)* }) => {
        $ty {
            n_feat: 10,
            n_epochs: 3,
            seed: 42,
            shuffle_users: true,
            n_time_bins: 4,
            beta: 0.3,
            n_freq_bins: 4,
            lr_u:   0.003,
            lr_ud:  0.001,
            lr_u2:  7e-6,
            lr_ub:  0.003,
            lr_ubd: 0.003,
            lr_i:   0.003,
            lr_ib:  0.003,
            lr_y:   0.0005,
            lr_yb:  2.5e-5,
            lr_yd:  0.0002,
            lr_tu:  0.0001,
            lr_ti:  0.0002,
            lr_ta:  2e-5,
            lr_ibf: 5e-5,
            lr_iqf: 5e-6,
            reg_iqf: 0.007,
            sigma_iqf: 0.005,
            lr_cu:  0.002,
            reg_cu: 0.01,
            reg_u:  0.0504,
            reg_u2: 0.4,
            reg_ud: 0.04,
            reg_i:  0.007,
            reg_y:  0.04,
            reg_yd: 0.02,
            sigma_u: 0.001,
            sigma_i: 0.005,
            sigma_y: 0.003,
            sigma_yd: 0.009,
            save_ifeat: false,
            low_memory: false,
            full_su: true,
            $($extra)*
        }
    };
}

#[test]
#[allow(deprecated)]
fn tsvdx4_probe_rmse_regression() {
    let cfg = cfg! { Tsvdx4Config {
        reset_u_epoch: 1,
        ordinal_head: None,
        sum_err_bug: false,
    } };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = TxModel::new(&tr, &pr_masked, cfg4(cfg));
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.020340388903108);
}

#[test]
#[allow(deprecated)]
fn tsvdx5_probe_rmse_regression() {
    let cfg = cfg! { Tsvdx5Config {
        reset_u_epoch: 100,
        ordinal_head: None,
        max_neighbors: 5,
        lr_w:  0.0015,
        lr_c:  0.0015,
        reg_w: 0.002,
        reg_c: 0.002,
        w_bias:   0.8,
        w_factor: 0.8,
        w_nbr:    0.8,
        lambda1: 25.0,
        lambda2: 10.0,
    } };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = TxModel::new(&tr, &pr_masked, cfg5(cfg));
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.02023124170746);
}

#[test]
#[allow(deprecated)]
fn tsvdx6_probe_rmse_regression() {
    let cfg = cfg! { Tsvdx6Config {
        reset_u_epoch: 100,
        ordinal_head: None,
        max_neighbors: 5,
        lr_w:      0.0015,
        lr_c:      0.0015,
        reg_w:     0.002,
        reg_c:     0.002,
        lr_w_day:  0.0015,
        lr_c_day:  0.0015,
        reg_w_day: 0.002,
        reg_c_day: 0.002,
        w_bias:    0.8,
        w_factor:  0.8,
        w_nbr:     0.8,
        lambda1: 25.0,
        lambda2: 10.0,
    } };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = TxModel::new(&tr, &pr_masked, cfg6(cfg));
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.0199379853756045);
}

#[test]
fn tsvdx4p_probe_rmse_regression() {
    let cfg = cfg! { Tsvdx4pConfig {
        reset_u_epoch: 1,
        nsvd_norm_exp: 0.5,
        n_threads: 1,
        seq_epochs: [0, 0, 0, 0, 0],
    } };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = Tsvdx4pModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.02037806773939);
}

#[test]
fn knn3_probe_rmse_regression() {
    let cfg = Knn3Config {
        threshold: 0.25,
        k_min: 2,
        k_max: 5,
        shrinkage: 5.0,
        reg: 0.01,
        x: 0.8,
        bl_reg_m: 2.0,
        bl_reg_u: 1.0,
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = Knn3Model::new(&tr, &pr_masked, cfg);
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 3.0427174790608045);
}

#[test]
fn gen_preds_seq_parallel_equivalence() {
    // Each prediction is independent (no cross-rating reduction), so the
    // sequential and parallel generators must produce bit-identical outputs.
    let cfg = Knn3Config::default();
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = Knn3Model::new(&tr, &pr_masked, cfg);

    let preds_seq = gen_preds(&mut model, &pr);
    let preds_par = gen_preds_parallel(&model, &pr);

    assert_eq!(preds_seq.len(), preds_par.len());
    for i in 0..preds_seq.len() {
        assert_eq!(preds_seq[i], preds_par[i], "mismatch at index {}", i);
    }
}

#[test]
fn knns_probe_rmse_regression() {
    let cfg = KnnsConfig {
        k: 3,
        shrinkage: 5.0,
        scaling: 1.0,
        tau: 0.025,
        supp_source: SuppSource::Compute,
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = KnnsModel::new(&tr, &pr_masked, cfg);
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.2349219913552398);
}

#[test]
fn asym_probe_rmse_regression() {
    let cfg = AsymConfig {
        n_feat: 8,
        n_epochs: 5,
        seed: 42,
        shuffle_users: true,
        lr_ub: 0.003,
        lr_i: 0.005,
        lr_ib: 0.003,
        lr_y: 0.0005,
        lr_us: 0.001,
        reg_i: 0.02,
        reg_y: 0.04,
        reg_us: 0.01,
        sigma_i: 0.005,
        sigma_y: 0.005,
        init_with_user_std: true,
        save_ifeat: false,
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = AsymModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.0220890364096071);
}

#[test]
fn mf_probe_rmse_regression() {
    let cfg = MfConfig {
        n_feat: 8,
        n_epochs: 5,
        seed: 42,
        shuffle_users: true,
        lr_u: 0.005,
        lr_i: 0.005,
        lr_ub: 0.003,
        lr_ib: 0.003,
        reg_u: 0.02,
        reg_i: 0.02,
        sigma_u: 0.005,
        sigma_i: 0.005,
        reset_u_epoch: 2,
        item_feat_npy: None,
        ordinal_head: None,
        save_ifeat: false,
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = MfModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.0221017098146403);
}

#[test]
fn als8_probe_rmse_regression() {
    let cfg = Als8Config {
        n_epochs: 3,
        seed: 42,
        n_feat: 4,
        sigma_u: 0.005,
        sigma_i: 0.005,
        reg_ub: 5.0,
        reg_ib: 5.0,
        reg_u: 0.05,
        reg_i: 0.01,
        shrink_m: 5.0,
        use_probe: true,
        n_bias_used: 8,
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = Als8Model::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.184768956911258);
}

#[test]
fn mfrbmx_probe_rmse_regression() {
    let cfg = MfRbmxConfig {
        mf: MfRbmxMfConfig {
            n_feat: 4,
            n_epochs: 3,
            seed: 42,
            lr_u: 0.005,
            lr_i: 0.03,
            lr_ub: 0.006,
            lr_ib: 0.03,
            reg_u: 0.07,
            reg_i: 0.004,
            sigma_u: 0.004,
            sigma_i: 0.005,
            reset_u_epoch: 100,
        },
        rbm: MfRbmxRbmConfig {
            n_hidden: 4,
            n_epochs: 3,
            seed: 42,
            gibbs3_after_epoch: 100,
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
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = MfRbmxModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.7380981190102633);
}

#[test]
fn aex_probe_rmse_regression() {
    let cfg = AexConfig {
        n_feat: 4,
        n_epochs: 3,
        seed: 42,
        shuffle_users: true,
        lr: 0.01,
        reg: 0.001,
        sigma: 0.01,
        normalize: true,
        use_implicit: true,
        dropout: 0.1,
        lr_ubias: 0.001,
        lr_udbias: 5e-4,
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = AexModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.1648546760603902);
}

#[test]
fn knnf_probe_rmse_regression() {
    // Synthetic factor matrix: 10 items × 4 features. Deterministic, mild variance.
    let n_items = 10;
    let n_feat = 4;
    let factors = Array2::from_shape_fn((n_items, n_feat), |(i, k)| {
        ((i * 13 + k * 7 + 3) % 17) as f32 / 17.0 - 0.5
    });

    let cfg = KnnfConfig {
        factors: "",  // unused (we bypass file load via new_with_factors)
        k: 3,
        scaling: 1.0,
        tau: 0.025,
    };

    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let mut model = KnnfModel::new_with_factors(&tr, factors, cfg);
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.9509877627957246);
}

/// Base Rbmx2Model config with all optional components enabled (per-user/day biases,
/// frequency bin bias, conditional RBM, factored RBM). Mini MF is toggled per test.
#[allow(deprecated)]
fn base_rbmx2_cfg() -> Rbmx2Config {
    Rbmx2Config {
        hidden_type: HiddenType::Bernoulli,
        visible_type: VisibleType::Softmax,
        temperature: 1.0,
        n_hidden: 4,
        n_epochs: 2,
        seed: 42,
        shuffle_users: true,
        init_sigma: 0.01,
        batch_size: 5,
        lr: 0.005,
        momentum: 0.9,
        weight_decay: 0.001,
        // Per-user visible bias (enabled)
        lr_bu: 0.001,
        wd_bu: 0.01,
        // Per-user-day visible bias (enabled)
        lr_but: 0.0005,
        wd_but: 0.01,
        cd_start: 1,
        cd_inc_every: 3,
        cd_inc_by: 1,
        cd_max: 5,
        // Conditional RBM (enabled)
        use_conditional: true,
        r_include_pr_all: true,
        save_w: false,
        // Factored RBM — overridden per test
        n_factors: None,
        // Mini MF — overridden per test
        mf_dim: 0,
        lr_mf_u: 0.001,
        lr_mf_i: 0.0005,
        wd_mf: 0.01,
        // Frequency bin bias (enabled)
        n_freq_bins: 4,
        lr_bif: 0.0002,
        wd_bif: 0.01,
        lr_bif_bug: false,
    }
}

#[test]
#[allow(deprecated)]
fn rbmx2_probe_rmse_regression() {
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let cases = [
        (HiddenType::Bernoulli,            VisibleType::Softmax,            None,    1.1369043574154352),
        (HiddenType::Bipolar,              VisibleType::Softmax,            None,    1.136411360157275),
        (HiddenType::NReLU,                VisibleType::Softmax,            None,    1.1374276757803843),
        (HiddenType::TruncExp(-1.0, 1.0),  VisibleType::Softmax,            None,    1.136373838937567),
        (HiddenType::Bernoulli,            VisibleType::TruncExp(1.0, 5.0), None,    1.1205197203663857),
        (HiddenType::Bernoulli,            VisibleType::Softmax,            Some(2), 1.1371796683637851),
    ];
    for (hidden_type, visible_type, n_factors, expected) in cases {
        let cfg = Rbmx2Config { hidden_type, visible_type, n_factors, ..base_rbmx2_cfg() };
        let mut model = Rbmx2Model::new(&tr, &pr_masked, cfg);
        for epoch in 1..=model.n_epochs() {
            model.fit_epoch(&tr, &pr_masked, epoch);
        }
        let rmse = calc_rmse(&mut model, &pr);
        println!("hidden={:?} visible={:?} n_factors={:?} probe RMSE {}",
                 hidden_type, visible_type, n_factors, rmse);
        assert_eq!(rmse, expected, "RMSE mismatch for {:?}/{:?}/{:?}",
                   hidden_type, visible_type, n_factors);
    }
}

/// Build the equivalent RxConfig from base_rbmx2_cfg() (drops MF + lr_bif_bug fields).
#[allow(deprecated)]
fn base_rx_cfg() -> RxConfig {
    let b = base_rbmx2_cfg();
    RxConfig {
        hidden_type: HiddenType::Bernoulli,  // overridden per case
        visible_type: VisibleType::Softmax,  // overridden per case
        temperature: b.temperature,
        n_hidden: b.n_hidden, n_epochs: b.n_epochs, seed: b.seed,
        shuffle_users: b.shuffle_users, init_sigma: b.init_sigma,
        batch_size: b.batch_size, lr: b.lr, momentum: b.momentum, weight_decay: b.weight_decay,
        lr_bu: b.lr_bu, wd_bu: b.wd_bu, lr_but: b.lr_but, wd_but: b.wd_but,
        cd_start: b.cd_start, cd_inc_every: b.cd_inc_every, cd_inc_by: b.cd_inc_by, cd_max: b.cd_max,
        use_conditional: b.use_conditional, r_include_pr_all: b.r_include_pr_all,
        save_w: b.save_w, n_factors: b.n_factors,
        n_freq_bins: b.n_freq_bins, lr_bif: b.lr_bif, wd_bif: b.wd_bif,
    }
}

#[test]
fn rx_probe_rmse_regression() {
    // Identical to rbmx2_probe_rmse_regression but using RxModel — RMSE values
    // must match (rx is an MF-free, lr_bif_bug-free reimplementation).
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let cases = [
        (HiddenType::Bernoulli,            VisibleType::Softmax,            None,    1.1369043574154352),
        (HiddenType::Bipolar,              VisibleType::Softmax,            None,    1.136411360157275),
        (HiddenType::NReLU,                VisibleType::Softmax,            None,    1.1374276757803843),
        (HiddenType::TruncExp(-1.0, 1.0),  VisibleType::Softmax,            None,    1.136373838937567),
        (HiddenType::Bernoulli,            VisibleType::TruncExp(1.0, 5.0), None,    1.1205197203663857),
        (HiddenType::Bernoulli,            VisibleType::Softmax,            Some(2), 1.1371796683637851),
    ];
    for (hidden_type, visible_type, n_factors, expected) in cases {
        let cfg = RxConfig { hidden_type, visible_type, n_factors, ..base_rx_cfg() };
        let mut model = RxModel::new(&tr, &pr_masked, cfg);
        for epoch in 1..=model.n_epochs() {
            model.fit_epoch(&tr, &pr_masked, epoch);
        }
        let rmse = calc_rmse(&mut model, &pr);
        println!("[rx] hidden={:?} visible={:?} n_factors={:?} probe RMSE {}",
                 hidden_type, visible_type, n_factors, rmse);
        assert_eq!(rmse, expected, "RMSE mismatch for {:?}/{:?}/{:?}",
                   hidden_type, visible_type, n_factors);
    }
}

#[test]
#[allow(deprecated)]
fn rbmx2_with_mf_probe_rmse_regression() {
    let cfg = Rbmx2Config { mf_dim: 3, ..base_rbmx2_cfg() };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = Rbmx2Model::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.1400299371358271);
}

/// Base CF-NADE-S config with small layer widths for fast, deterministic tests.
/// Only the always-on temporal terms (user bias + drift) are enabled; the optional
/// feature toggles are flipped on in the `_full_` variant below.
fn base_cfnade_cfg() -> CfNadeConfig {
    CfNadeConfig {
        n_hidden: 8,
        n_hidden2: 0,
        rank: 4,
        n_epochs: 4,
        seed: 42,
        shuffle_users: true,
        batch_size: 5,
        lr: 0.005,
        first_layer_lr_mult: 1.4,
        n_milestones: 1,
        milestones: [3, 0, 0, 0],
        lr_gamma: 0.9,
        weight_decay: 0.008,
        ordinal_lambda: 1.0,
        rmse_weight: 0.1,
        ctx_norm_alpha: 0.5,
        swa_start: 2,
        use_ctx_target_day_scale: false,
        use_implicit_probe_ctx: false,
        use_user_bias: true,
        use_user_drift: true,
        use_user_item_scale: false,
        use_user_item_scale_bin: false,
        use_user_day_bias_bin: false,
        use_user_day_scale_bin: false,
        use_day_bias: false,
        use_item_time_bias: false,
        use_day_freq_bias: false,
        use_side_features: false,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    }
}

#[test]
fn cfnade_probe_rmse_regression() {
    let cfg = base_cfnade_cfg();
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = CfNadeModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.1388040748818262);
}

#[test]
fn cfnade_full_probe_rmse_regression() {
    // All optional temporal/side feature paths enabled plus a second hidden
    // layer and implicit-feedback context, to cover the branches the base
    // config leaves off.
    let cfg = CfNadeConfig {
        n_hidden2: 6,
        use_ctx_target_day_scale: true,
        use_implicit_probe_ctx: true,
        use_user_item_scale: true,
        use_user_item_scale_bin: true,
        use_user_day_bias_bin: true,
        use_user_day_scale_bin: true,
        use_day_bias: true,
        use_item_time_bias: true,
        use_day_freq_bias: true,
        use_side_features: true,
        ..base_cfnade_cfg()
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = CfNadeModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.1553573447550949);
}

#[test]
fn bk1_probe_rmse_regression() {
    // k_neighbors = 0 disables the neighborhood term so this test
    // doesn't depend on `sim/rtg_{prod,supp}.tiny_train.npy`.
    let cfg = Bk1Config {
        n_feat: 4,
        n_epochs: 3,
        seed: 42,
        shuffle_users: true,
        n_time_bins: 4,
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
        lr_h: 0.005,
        lr_decay: 0.05,
        reg_bias: 0.005,
        reg_fact: 0.015,
        reg_nbr: 0.015,
        reg_h: 0.005,
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = Bk1Model::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.018273739458307);
}

#[test]
fn dnn_probe_rmse_regression() {
    // `n_threads: 1` is what makes the fit reproducible at all: the MLP update
    // is single-threaded either way, but the user blocks are not.
    //
    // Verified to bite: a 0.5% change to `lr_emb` breaks both dnn RMSE asserts.
    // Not covered by the fixture: the bounded output head, which stays in its
    // linear regime because 56 ratings over 3 epochs never drive it to saturate,
    // so `out_scale` has no effect here.
    let cfg = DnnConfig {
        n_feat: 4, h1: 8, h2: 4, n_mf: 0, n_bins: 4,
        n_epochs: 3, seed: 42, n_threads: 1, block_users: 8,
        ..DnnConfig::default()
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = DnnModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.0484216878929749);
}

#[test]
fn dnn_with_mf_probe_rmse_regression() {
    // The wide bilinear term is a separate code path — plain SGD beside the
    // network — and the two are only ever exercised together by accident.
    //
    // `n_bins` is large on purpose. Bins are cut over the *whole* 2244-day span
    // while the fixture's dates only run 9..39, so anything under ~250 collapses
    // every rating into bin 0 and the per-bin item biases never differ. At 512
    // the ratings land in bins 2..8, which does exercise them — but note that a
    // one-day change to the span still moves no boundary over a range this
    // narrow, so these tests do not pin the span itself.
    let cfg = DnnConfig {
        n_feat: 4, h1: 8, h2: 4, n_mf: 6, n_bins: 512,
        n_epochs: 3, seed: 42, n_threads: 1, block_users: 8,
        ..DnnConfig::default()
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = DnnModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    let rmse = calc_rmse(&mut model, &pr);
    println!("probe RMSE {}", rmse);
    assert!(rmse == 1.0443630034640592);
}

#[test]
fn dnn_subscores_sum_to_the_prediction() {
    // `predict_subscores` splits the output into baseline and network; the two
    // must reconstruct `predict` exactly, since the blender consumes them as
    // separate columns.
    let cfg = DnnConfig {
        n_feat: 4, h1: 8, h2: 4, n_mf: 6, n_bins: 4,
        n_epochs: 2, seed: 7, n_threads: 1, block_users: 8,
        ..DnnConfig::default()
    };
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = DnnModel::new(&tr, &pr_masked, cfg);
    for epoch in 1..=model.n_epochs() {
        model.fit_epoch(&tr, &pr_masked, epoch);
    }
    assert_eq!(model.n_subscores(), 2);
    for idx in 0..pr.n_ratings {
        let (u, i) = (pr.user_idxs[idx] as usize, pr.item_idxs[idx] as usize);
        let day = pr.dates[idx] as i32;
        let parts = model.predict_subscores(u, i, day);
        let diff = (parts[0] + parts[1] - model.predict(u, i, day)).abs();
        assert!(diff < 1e-5, "subscores drift by {diff} at rating {idx}");
    }
}
