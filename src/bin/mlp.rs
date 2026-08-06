//! MLP cross-fit blending of base predictors + voting features. Counterpart to
//! `gbm.rs`; one job per named blend (mirrors the Python `blending/mlp.py`
//! `mlpr*` jobs). Builds with pure-Rust matmul by default; `--features blas`
//! routes the dense products through OpenBLAS (`cblas_dgemm`).

#[cfg(feature = "blas")]
extern crate blas_src;

use netflix_prize::blend::{
    build_xy, close_log, cvk_blend, flatten_groups, load_models_toml, load_quiz_mask, log_columns,
    open_log, resolve_voting, save_preds, select_groups,
};
use netflix_prize::mlp::{MlpBlender, MlpCfg, MlpHead};
use netflix_prize::teeln;
use std::collections::HashMap;
use std::process::ExitCode;

struct Args {
    name: String,
    pipeline: String,
    models: String,
    groups: Vec<String>,
    model_manual: Vec<String>,
    exclude: Vec<String>,
    voting_models: String,
    voting: Vec<String>,
    seeds: Vec<u64>,
}

/// MLP + cross-fit parameters for a named blend. Add a branch per blend.
#[derive(Debug)]
struct BlendParams {
    hidden: Vec<usize>,
    alpha: f64,
    lr: f64,
    iters: usize,
    batch: usize,
    momentum: f64,
    tol: f64,
    n_iter_no_change: usize,
    folds: usize,
    /// Output head: `Regression` (linear + squared loss) for `mlpr*` jobs, or
    /// `Ordinal` (4 sigmoid threshold heads + BCE) for `mlpc*` jobs.
    head: MlpHead,
}

/// Baseline = the `mlpr1` config (create_nn: 64×64, alpha 0.05, lr 0.0004, 64
/// iters, sklearn defaults for momentum/tol/early-stop, 2-fold). Each
/// `blend_config` branch overrides only what differs via struct-update.
impl Default for BlendParams {
    fn default() -> Self {
        Self {
            hidden: vec![64, 64],
            alpha: 0.05,
            lr: 0.0004,
            iters: 64,
            batch: 200,
            momentum: 0.9,
            tol: 1e-4,
            n_iter_no_change: 10,
            folds: 2,
            head: MlpHead::Regression,
        }
    }
}

fn blend_config(name: &str) -> BlendParams {
    match name {
        // create_nn: (64, 64), alpha 0.05, lr 0.0004, 64 iters, 2-fold.
        "mlpr1" => BlendParams::default(),

        // Bayesian-optimized MLP.
        "mlpr2o" => BlendParams {
            hidden: vec![32, 32], alpha: 0.24101624563575427, lr: 0.0005339808669500829,
            iters: 80, momentum: 0.9148116502629432, tol: 2.890613343488106e-5,
            n_iter_no_change: 9, ..Default::default()
        },

        // Bayesian-optimized ordinal MLP.
        "mlpc3o" => BlendParams {
            hidden: vec![64, 32], alpha: 0.9723373672604836, lr: 0.000953329728567159,
            iters: 80, batch: 400, momentum: 0.9380845452666436, tol: 1.4640841816087311e-5,
            n_iter_no_change: 7, head: MlpHead::Ordinal, ..Default::default()
        },
        _ => panic!("unknown blend job '{name}' (add a branch in blend_config)"),
    }
}

fn print_help() {
    println!("Usage: mlp NAME -p FILE -t FILE --voting-models FILE --voting G,... (--seeds N,N,... | --seed N)");
    println!();
    println!("  NAME                       blend name; MLP params come from");
    println!("                             blend_config(NAME); produces NAME-s<seed>.*");
    println!("  -p FILE, --pipeline FILE   pipeline TOML (required; carries the split)");
    println!("  -t FILE, --models FILE     base-predictor models TOML (required)");
    println!("  --groups G,G,...           model groups (default: the TOML's `all`; omit with -m for manual-only)");
    println!("  -m NAME, --model NAME      add a single base predictor (repeatable; combines with --groups)");
    println!("  -x NAME, --exclude NAME    drop a model by name (repeatable; brace-expanded)");
    println!("  --voting-models FILE       voting-feature groups TOML (required)");
    println!("  --voting G,G,...           voting-feature groups to use (required; 'all' if the TOML defines it)");
    println!("  --seeds N,N,...            net+fold seeds; one output NAME-s<N> per seed (data loaded once)");
    println!("  --seed N                   add a single seed (repeatable)");
    println!("  -h, --help                 show this help");
}

fn need(argv: &[String], i: usize) -> String {
    if i + 1 >= argv.len() {
        eprintln!("error: '{}' requires an argument", argv[i]);
        std::process::exit(2);
    }
    argv[i + 1].clone()
}

fn parse_args() -> Args {
    let mut a = Args {
        name: String::new(),
        pipeline: String::new(),
        models: String::new(),
        groups: Vec::new(),
        model_manual: Vec::new(),
        exclude: Vec::new(),
        voting_models: String::new(),
        voting: Vec::new(),
        seeds: Vec::new(),
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            "-p" | "--pipeline" => { a.pipeline = need(&argv, i); i += 2; }
            "-t" | "--models" => { a.models = need(&argv, i); i += 2; }
            "--groups" => {
                for tok in need(&argv, i).split(',') {
                    a.groups.push(tok.trim().to_string());
                }
                i += 2;
            }
            "-m" | "--model" => { a.model_manual.push(need(&argv, i)); i += 2; }
            "-x" | "--exclude" => { a.exclude.push(need(&argv, i)); i += 2; }
            "--voting-models" => { a.voting_models = need(&argv, i); i += 2; }
            "--voting" => {
                for tok in need(&argv, i).split(',') {
                    a.voting.push(tok.trim().to_string());
                }
                i += 2;
            }
            "--seed" => { a.seeds.push(need(&argv, i).parse().expect("bad --seed")); i += 2; }
            "--seeds" => {
                for tok in need(&argv, i).split(',') {
                    a.seeds.push(tok.trim().parse().expect("bad --seeds value"));
                }
                i += 2;
            }
            s if s.starts_with('-') => {
                eprintln!("error: unknown flag '{}'", s);
                print_help();
                std::process::exit(2);
            }
            s => {
                if !a.name.is_empty() {
                    eprintln!("error: only one NAME argument allowed");
                    std::process::exit(2);
                }
                a.name = s.to_string();
                i += 1;
            }
        }
    }
    if a.name.is_empty() {
        eprintln!("error: NAME argument required");
        print_help();
        std::process::exit(2);
    }
    for (val, flag) in [
        (&a.pipeline, "-p FILE (pipeline TOML)"),
        (&a.models, "-t FILE (models TOML)"),
        (&a.voting_models, "--voting-models FILE"),
    ] {
        if val.is_empty() {
            eprintln!("error: {flag} is required");
            print_help();
            std::process::exit(2);
        }
    }
    if a.seeds.is_empty() {
        eprintln!("error: provide --seeds N,N,... or --seed N");
        print_help();
        std::process::exit(2);
    }
    if a.voting.is_empty() {
        eprintln!("error: --voting is required (name a group from the voting TOML, e.g. 'all')");
        print_help();
        std::process::exit(2);
    }
    a
}

fn load_pipeline_split(path: &str) -> HashMap<String, String> {
    #[derive(serde::Deserialize)]
    struct P {
        #[serde(default)]
        split: HashMap<String, String>,
    }
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let p: P = toml::from_str(&s).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    p.split
}

fn main() -> ExitCode {
    let args = parse_args();

    let split = load_pipeline_split(&args.pipeline);
    let pr = split.get("pr").expect("[split].pr missing").clone();
    let fulltrain_pr = split.get("fulltrain_pr").expect("[split].fulltrain_pr missing").clone();
    let preds = split.get("preds").expect("[split].preds missing").clone();
    let split_name = split.get("name").cloned().unwrap_or_else(|| "?".to_string());

    let p = blend_config(&args.name);
    // Models: from --groups of the models TOML, or purely from -m when no group
    // is named (manual-only skips loading the TOML entirely).
    let mut groups = if args.groups.is_empty() && !args.model_manual.is_empty() {
        indexmap::IndexMap::new()
    } else {
        select_groups(&load_models_toml(&args.models), &args.groups)
    };
    if !args.model_manual.is_empty() {
        groups.insert("manual".to_string(), args.model_manual.clone());
    }
    let base = flatten_groups(&groups, &args.exclude).specs();
    assert!(!base.is_empty(), "no models selected (use --groups or -m)");
    let voting = resolve_voting(&args.voting_models, &args.voting);

    open_log(&preds, &args.name);
    let seeds_str = args.seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
    let models_src = if args.groups.is_empty() && !args.model_manual.is_empty() {
        "(manual)".to_string()
    } else { args.models.clone() };
    let groups_str = if !args.groups.is_empty() { args.groups.join(",") }
        else if args.model_manual.is_empty() { "all".to_string() } else { "manual".to_string() };
    let voting_str = args.voting.join(",");
    teeln!("[{}]", args.name);
    teeln!("Pipeline:  {} (split = {})", args.pipeline, split_name);
    teeln!("Models:    {} ({} base predictors, groups: {})", models_src, base.len(), groups_str);
    if !args.exclude.is_empty() {
        teeln!("Excluded:  {} name(s): {}", args.exclude.len(), args.exclude.join(", "));
    }
    teeln!("Voting:    {} ({} features, groups: {})", args.voting_models, voting.len(), voting_str);
    teeln!("Seeds:     {} (net init + fold permutation)", seeds_str);
    teeln!("Params:    {:?}", p);
    log_columns(&base, &voting);
    teeln!();

    println!("Loading probe set ({})...", pr);
    let (x_pr, y_pr) = build_xy(&base, &voting, &preds, &preds, &pr);
    println!("Loading qual set ({})...", fulltrain_pr);
    let (x_ql, y_ql) = build_xy(&base, &voting, &preds, &preds, &fulltrain_pr);
    let qz = load_quiz_mask(&fulltrain_pr);
    println!("Probe: {} rows, Qual: {} rows", y_pr.len(), y_ql.len());

    // Feature matrices are loaded once and reused across all seeds.
    for &seed in &args.seeds {
        teeln!();
        teeln!("=== seed {} ===", seed);
        let cfg = MlpCfg {
            hidden: p.hidden.clone(),
            alpha: p.alpha,
            lr: p.lr,
            max_iter: p.iters,
            batch_size: p.batch,
            momentum: p.momentum,
            tol: p.tol,
            n_iter_no_change: p.n_iter_no_change,
            seed,
            head: p.head,
            ..Default::default()
        };
        let (p_pr, p_ql) = cvk_blend::<MlpBlender>(
            &x_pr,
            y_pr.as_slice().unwrap(),
            &x_ql,
            y_ql.as_slice().unwrap(),
            qz.as_slice().unwrap(),
            p.folds,
            seed,
            &cfg,
        );
        let pr_path = format!("{preds}/{}-s{seed}.{pr}.npy", args.name);
        let ql_path = format!("{preds}/{}-s{seed}.{fulltrain_pr}.npy", args.name);
        save_preds(&pr_path, &p_pr);
        save_preds(&ql_path, &p_ql);
        teeln!("Saved {} / {}", pr_path, ql_path);
    }

    close_log();
    ExitCode::SUCCESS
}
