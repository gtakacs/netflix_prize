// Regression tests for the split_3 ALS integrated model (integ.rs) and the
// user-user kNN chain stage (uknn.rs) on the tiny train / probe sets. Both models
// sum in parallel, so every test runs in a one-thread rayon pool, which makes the
// result bit-reproducible; each test pins the probe RMSE and a checksum over all
// probe predictions.

mod helpers;
use helpers::{make_tiny_probe, make_tiny_train};

use ndarray::Array2;
use ndarray_npy::write_npy;
use netflix_prize::integ::{IntegConfig, IntegModel};
use netflix_prize::nbstats::build_rtg_nbstats;
use netflix_prize::uknn::{UknnConfig, UknnModel};
use netflix_prize::{Dataset, MaskedDataset, Regressor, calc_rmse};

/// A fresh scratch directory under the system temp dir, as a 'static str (the
/// model configs hold their file directories as &'static str).
fn scratch_dir(tag: &str) -> &'static str {
    let d = std::env::temp_dir().join(format!("netflix_prize_test_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.to_str().unwrap().to_string().leak()
}

/// Probe RMSE and a position-weighted checksum of every probe prediction.
fn score<M: Regressor>(model: &mut M, pr: &Dataset) -> (f64, f64) {
    let rmse = calc_rmse(model, pr);
    let sum = (0..pr.n_ratings).map(|t| {
        let p = model.predict(pr.user_idxs[t] as usize, pr.item_idxs[t] as usize, pr.dates[t] as i32);
        p as f64 * (t + 1) as f64
    }).sum();
    (rmse, sum)
}

/// Fit `M` on tiny_train in a one-thread pool, score on tiny_probe. With `center`
/// the target is rating − 3.6 on both sets (a chain stage fits a residual).
fn run<M: Regressor + Send>(cfg: M::Config, center: bool) -> (f64, f64)
where
    M::Config: Send + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let (mut tr, mut pr) = (make_tiny_train(), make_tiny_probe());
        if center {
            tr.residuals.mapv_inplace(|r| r - 3.6);
            pr.residuals.mapv_inplace(|r| r - 3.6);
        }
        let pr_masked = MaskedDataset::from(&pr);
        let mut model = M::new(&tr, &pr_masked, cfg);
        for epoch in 1..=model.n_epochs() {
            model.fit_epoch(&tr, &pr_masked, epoch);
        }
        score(&mut model, &pr)
    })
}

// ---------------------------------------------------------------------------
// integ
// ---------------------------------------------------------------------------

/// The `s3 integ` defaults with F = 4 and 3 epochs, no terms switched on.
fn integ_cfg() -> IntegConfig {
    IntegConfig {
        use_p: false, use_y: false, use_nb: false, use_ud: false, use_dr: false, use_fq: false,
        n_feat: 4, n_epochs: 3, alpha: 0.5,
        reg_bu: 10.0, reg_bi: 10.0, reg_p: 10.0, reg_q: 10.0, reg_y: 10.0,
        reg_nw: 10.0, reg_nc: 10.0, reg_ud: 10.0, reg_dr: 1000.0, dr_beta: 0.4,
        reg_fq: 10.0, fq_base: 6.76,
        nb_k: 50, nb_start: 1, nb_freeze: false, nb_mse: true,
        nb_stat0: "rtg_supp", nb_stat2: "rtg_diff2", nb_shr: 100.0, sim_dir: "",
        sigma: 0.01, cg_iters: 20, cg_tol: 1e-4, precond: true, implicit: false,
        seed: 42, loss_check: true, sort: true,
    }
}

#[test]
fn integ_nsvd1_regression() {
    // TERMS=y SORT=0: the NSVD1 case (lab-s3-nsvd's model)
    let (rmse, sum) = run::<IntegModel>(IntegConfig { use_y: true, sort: false, ..integ_cfg() }, false);
    println!("integ nsvd1: rmse {rmse:?} sum {sum:?}");
    assert_eq!((rmse, sum), (1.0566619228510383, 750.6052415370941));
}

#[test]
fn integ_svdpp_implicit_regression() {
    // TERMS=p+y IMPL=1: SVD++ with N(u) including the probe items
    let (rmse, sum) = run::<IntegModel>(IntegConfig { use_p: true, use_y: true, implicit: true, ..integ_cfg() }, false);
    println!("integ svd++: rmse {rmse:?} sum {sum:?}");
    assert_eq!((rmse, sum), (1.0566619011501732, 750.60524559021));
}

#[test]
fn integ_time_terms_regression() {
    // TERMS=p+y+ud+dr+fq IMPL=1: the s3-integ40/80-pyuddf recipe
    let cfg = IntegConfig {
        use_p: true, use_y: true, use_ud: true, use_dr: true, use_fq: true, implicit: true,
        reg_p: 30.0, reg_q: 30.0, reg_y: 30.0, ..integ_cfg()
    };
    let (rmse, sum) = run::<IntegModel>(cfg, false);
    println!("integ p+y+ud+dr+fq: rmse {rmse:?} sum {sum:?}");
    assert_eq!((rmse, sum), (1.0914583849795771, 750.3578655719757));
}

#[test]
fn integ_nb_regression() {
    // TERMS=p+y+nb: the neighbourhood term over sim matrices built from tiny_train
    let dir = scratch_dir("integ_nb");
    let tr = make_tiny_train();
    for stat in ["supp", "diff2"] {
        write_npy(format!("{dir}/rtg_{stat}.{}.npy", tr.name), &build_rtg_nbstats(&tr, stat)).unwrap();
    }
    let cfg = IntegConfig { use_p: true, use_y: true, use_nb: true, nb_k: 3, nb_shr: 1.0, sim_dir: dir, ..integ_cfg() };
    let (rmse, sum) = run::<IntegModel>(cfg, false);
    println!("integ p+y+nb: rmse {rmse:?} sum {sum:?}");
    assert_eq!((rmse, sum), (1.0912507537415093, 748.3923637866974));
}

#[test]
fn integ_sort_only_changes_summation_order() {
    // SORT only reorders each user's ratings, so TERMS=y gives the same model up
    // to rounding with and without it.
    let a = run::<IntegModel>(IntegConfig { use_y: true, sort: true, ..integ_cfg() }, false);
    let b = run::<IntegModel>(IntegConfig { use_y: true, sort: false, ..integ_cfg() }, false);
    assert!((a.0 - b.0).abs() < 1e-9, "rmse {} vs {}", a.0, b.0);
}

// ---------------------------------------------------------------------------
// uknn
// ---------------------------------------------------------------------------

/// Writes a deterministic 10 × 8 item-factor matrix as `<dir>/tf.tiny_train.npy`.
fn uknn_cfg(dir: &'static str) -> UknnConfig {
    let f = Array2::from_shape_fn((10, 8), |(i, k)| ((i * 13 + k * 7 + 3) % 17) as f32 / 17.0 - 0.5);
    write_npy(format!("{dir}/tf.tiny_train.npy"), &f).unwrap();
    UknnConfig {
        fact: "tf", res_vec: false, coday: false, win: 1, cap: 1000, dim: 64,
        k: 6, kp: 3, scaling: 2.0, shrinkage: 10.0, tau: 0.0, implicit: true, preds_dir: dir,
    }
}

#[test]
fn uknn_presence_regression() {
    // UK_VEC=pres: user vectors from the factors of the items present for the user
    let cfg = uknn_cfg(scratch_dir("uknn_pres"));
    let (rmse, sum) = run::<UknnModel>(cfg, true);
    println!("uknn pres: rmse {rmse:?} sum {sum:?}");
    assert_eq!((rmse, sum), (1.01518709856166, 5.187869109446183));
}

#[test]
fn uknn_residual_projected_regression() {
    // UK_VEC=res with the factors projected to UK_DIM=4 dims, UK_TAU > 0
    let cfg = UknnConfig { res_vec: true, dim: 4, tau: 0.05, ..uknn_cfg(scratch_dir("uknn_res")) };
    let (rmse, sum) = run::<UknnModel>(cfg, true);
    println!("uknn res: rmse {rmse:?} sum {sum:?}");
    assert_eq!((rmse, sum), (1.0097320211082994, 9.227413386106491));
}

#[test]
fn uknn_coday_regression() {
    // UK_VEC=coday: users compared by co-rating in (item, day / UK_WIN) buckets
    let cfg = UknnConfig { coday: true, win: 7, ..uknn_cfg(scratch_dir("uknn_coday")) };
    let (rmse, sum) = run::<UknnModel>(cfg, true);
    println!("uknn coday: rmse {rmse:?} sum {sum:?}");
    assert_eq!((rmse, sum), (1.0090546999564487, -6.819758262485266));
}
