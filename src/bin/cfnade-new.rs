//! CF-NADE dispatcher (SPLIT_NEW). `args[1]` selects a job:
//!   cfnade-96   -> canonical preset, run through the standard fit2 pipeline
//!   --test ...  -> flag-driven tuning loop on trainx/probex (no qual/train save)

use std::env;
use std::time::Instant;

use netflix_prize::{
    Dataset, MaskedDataset, Regressor, calc_rmse, fit2, save_preds, SPLIT_NEW,
    cfnade::{CfNadeConfig, CfNadeModel},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let job_name = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match job_name {
        // Capacity-scaled presets (hidden, rank, n_epochs); other knobs are the
        // canonical defaults (swa_start 18, ms [64,72]).
        "cfnade-96" => run_preset(job_name, 320, 96, 95),
        "cfnade-48" => run_preset(job_name, 160, 48, 95),
        "cfnade-12" => run_preset(job_name, 40, 12, 40),

        // Same as cfnade-96 but with SWA averaging disabled (raw SGD weights).
        "cfnade-96r" => run_preset_noswa(job_name, 320, 96, 95),

        "--test" => run_test(),

        _ => panic!(
            "unknown job '{job_name}' (try cfnade-{{96,48,24,12}} or '--test')"
        ),
    }
}

/// Run a capacity-scaled preset through the standard fit2 pipeline.
/// `save_probe_each_epoch` writes per-epoch predictions: `{name}_ep{NN}.probex.npy`
/// in phase 1 and `{name}_ep{NN}.qual.npy` in the fulltrain phase.
fn run_preset(job_name: &str, n_hidden: usize, rank: usize, n_epochs: usize) {
    run_preset_swa(job_name, n_hidden, rank, n_epochs, CfNadeConfig::default().swa_start);
}

/// Like `run_preset` but with SWA averaging disabled: `swa_start > n_epochs`
/// keeps `swa_count` at 0, so `update_swa`/`swap_with_swa` are no-ops and every
/// prediction (per-epoch and final) uses the raw SGD weights.
fn run_preset_noswa(job_name: &str, n_hidden: usize, rank: usize, n_epochs: usize) {
    run_preset_swa(job_name, n_hidden, rank, n_epochs, usize::MAX);
}

fn run_preset_swa(job_name: &str, n_hidden: usize, rank: usize, n_epochs: usize, swa_start: usize) {
    let cfg = CfNadeConfig { n_hidden, rank, n_epochs, swa_start, ..CfNadeConfig::default() };
    fit2!(CfNadeModel, cfg, "rtg", job_name, SPLIT_NEW, save_probe_each_epoch: true);
}

fn cli_arg(key: &str) -> Option<String> {
    let args: Vec<String> = env::args().collect();
    args.iter().position(|a| a == key).map(|i| args[i + 1].clone())
}

fn has_flag(key: &str) -> bool {
    env::args().any(|a| a == key)
}

/// Flag-driven tuning loop: trains on trainx (or train{N} via --data) and
/// reports probex RMSE each `--eval-every` epochs. Final-epoch RMSE reflects
/// the SWA-averaged weights (the model swaps to them inside fit_epoch).
fn run_test() {
    let mut cfg = CfNadeConfig::default();
    let set = |v: Option<String>, f: &mut dyn FnMut(f32)| {
        if let Some(s) = v {
            f(s.parse().unwrap());
        }
    };
    set(cli_arg("--hidden"), &mut |v| cfg.n_hidden = v as usize);
    set(cli_arg("--hidden2"), &mut |v| cfg.n_hidden2 = v as usize);
    set(cli_arg("--rank"), &mut |v| cfg.rank = v as usize);
    set(cli_arg("--epochs"), &mut |v| cfg.n_epochs = v as usize);
    set(cli_arg("--seed"), &mut |v| cfg.seed = v as u64);
    set(cli_arg("--bs"), &mut |v| cfg.batch_size = v as usize);
    set(cli_arg("--lr"), &mut |v| cfg.lr = v);
    set(cli_arg("--flr"), &mut |v| cfg.first_layer_lr_mult = v);
    set(cli_arg("--gamma"), &mut |v| cfg.lr_gamma = v);
    set(cli_arg("--wd"), &mut |v| cfg.weight_decay = v);
    set(cli_arg("--lambda"), &mut |v| cfg.ordinal_lambda = v);
    set(cli_arg("--rmse-w"), &mut |v| cfg.rmse_weight = v);
    set(cli_arg("--ctx-alpha"), &mut |v| cfg.ctx_norm_alpha = v);
    set(cli_arg("--swa-start"), &mut |v| cfg.swa_start = v as usize);

    if let Some(ms_str) = cli_arg("--ms") {
        let vals: Vec<usize> = ms_str
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect();
        cfg.n_milestones = vals.len().min(cfg.milestones.len());
        cfg.milestones = [0; 4];
        for (i, &v) in vals.iter().take(cfg.milestones.len()).enumerate() {
            cfg.milestones[i] = v;
        }
    }

    cfg.use_ctx_target_day_scale = has_flag("--ctx-target-day-scale");
    cfg.use_implicit_probe_ctx = has_flag("--implicit-probe-ctx");
    cfg.use_user_drift = !has_flag("--no-user-drift");
    cfg.use_user_item_scale = has_flag("--user-item-scale");
    cfg.use_user_item_scale_bin = has_flag("--user-item-scale-bin");
    cfg.use_user_day_bias_bin = has_flag("--user-day-bias-bin");
    cfg.use_user_day_scale_bin = has_flag("--user-day-scale-bin");
    cfg.use_day_bias = has_flag("--day-bias");
    cfg.use_item_time_bias = has_flag("--item-time-bias");
    cfg.use_day_freq_bias = has_flag("--day-freq-bias");
    cfg.use_side_features = has_flag("--wide");

    let eval_every: usize = cli_arg("--eval-every").map(|s| s.parse().unwrap()).unwrap_or(1);
    let save_prefix = cli_arg("--save");
    let dataset = cli_arg("--data").unwrap_or_else(|| "x".to_string());
    let (tr_name, pr_name) = match dataset.as_str() {
        "x" => ("trainx".to_string(), "probex".to_string()),
        "full" => ("train".to_string(), "probe".to_string()),
        "qual" => ("fulltrain".to_string(), "qual".to_string()),
        n => (format!("train{n}"), format!("probe{n}")),
    };

    println!(
        "=== TEST CF-NADE h={} h2={} r={} lr={} flr={} ms={:?} g={} wd={} bs={} ep={} swa={} \
         lambda={} rmsew={} ctxa={} | drift={} uscale={} usbin={} udaybin={} udayscale={} \
         daybias={} itbias={} dayfreq={} ctxtday={} implicit={} wide={} data={} ===",
        cfg.n_hidden, cfg.n_hidden2, cfg.rank, cfg.lr, cfg.first_layer_lr_mult,
        &cfg.milestones[..cfg.n_milestones], cfg.lr_gamma, cfg.weight_decay, cfg.batch_size,
        cfg.n_epochs, cfg.swa_start, cfg.ordinal_lambda, cfg.rmse_weight, cfg.ctx_norm_alpha,
        cfg.use_user_drift, cfg.use_user_item_scale, cfg.use_user_item_scale_bin,
        cfg.use_user_day_bias_bin, cfg.use_user_day_scale_bin, cfg.use_day_bias,
        cfg.use_item_time_bias, cfg.use_day_freq_bias, cfg.use_ctx_target_day_scale,
        cfg.use_implicit_probe_ctx, cfg.use_side_features, dataset,
    );

    let tr = Dataset::load(&tr_name, "rtg", SPLIT_NEW.preds_dir);
    let pr = Dataset::load(&pr_name, "rtg", SPLIT_NEW.preds_dir);
    let pr_masked = MaskedDataset::from(&pr);
    let mut model = CfNadeModel::new(&tr, &pr_masked, cfg);

    let mut best = f64::INFINITY;
    for epoch in 1..=cfg.n_epochs {
        let t0 = Instant::now();
        model.fit_epoch(&tr, &pr_masked, epoch);
        let secs = t0.elapsed().as_secs_f64();
        if epoch % eval_every == 0 || epoch == cfg.n_epochs {
            let rmse = calc_rmse(&mut model, &pr);
            println!("Epoch {epoch:02} | train {secs:.1}s | RMSE {rmse:.4}");
            if rmse < best {
                best = rmse;
                if let Some(prefix) = save_prefix.as_ref() {
                    let path = format!("{}/{}.{}.npy", SPLIT_NEW.preds_dir, prefix, pr_name);
                    save_preds(&mut model, &pr, &path);
                }
            }
        } else {
            println!("Epoch {epoch:02} | train {secs:.1}s | eval skipped");
        }
    }
    println!("BEST_RMSE={best:.4}");
}
