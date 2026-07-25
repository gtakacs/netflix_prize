//! LightGBM cross-fit blending of base predictors + voting features, with a
//! regression (`gbr*`) or multiclass-classifier (`gbc*`) objective per job.
//! Counterpart to `ridge.rs`; only functional when built with `--features lgbm`.

#[cfg(feature = "lgbm")]
fn main() -> std::process::ExitCode {
    real::run()
}

#[cfg(not(feature = "lgbm"))]
fn main() {
    eprintln!("gbm-new requires the 'lgbm' feature:");
    eprintln!("  cargo build --release --features lgbm --bin gbm-new");
    std::process::exit(2);
}

#[cfg(feature = "lgbm")]
mod real {
    use netflix_prize::blend::{
        build_xy, cvk_blend, flatten_groups, load_models_toml, load_quiz_mask, save_preds,
        select_groups, LgbmBlender, LgbmCfg, LgbmMode,
    };
    use std::collections::HashMap;
    use std::process::ExitCode;

    struct Args {
        name: String,
        pipeline: String,
        models: String,
        groups: Vec<String>,
        exclude: Vec<String>,
        seeds: Vec<u64>,
    }

    /// LightGBM + cross-fit parameters for a named blend. Add a branch per blend
    /// (mirrors the base-predictor dispatchers in `tsvdx4-new.rs` etc.).
    #[derive(Debug)]
    struct BlendParams {
        num_leaves: usize,
        iters: usize,
        lr: f64,
        min_data_in_leaf: usize,
        feature_fraction: f64,
        bagging_fraction: f64,
        bagging_freq: usize,
        lambda_l1: f64,
        lambda_l2: f64,
        max_bin: usize,
        folds: usize,
        threads: usize,
        /// `false` -> regression objective; `true` -> 5-class multiclass with the
        /// prediction collapsed to the rating expectation Σ_k p_k·k (mirrors the
        /// Python `gbc*` LGBMClassifier blends).
        multiclass: bool,
    }

    /// Baseline: LightGBM defaults + the hand-set blends' shared knobs
    /// (num_leaves 63, 200 iters, lr 0.1, 2-fold, 16 threads, regression). Each
    /// `blend_config` branch overrides only what differs via struct-update.
    impl Default for BlendParams {
        fn default() -> Self {
            Self {
                num_leaves: 63, iters: 200, lr: 0.1,
                min_data_in_leaf: 20, feature_fraction: 1.0,
                bagging_fraction: 1.0, bagging_freq: 0,
                lambda_l1: 0.0, lambda_l2: 0.0, max_bin: 255,
                folds: 2, threads: 16, multiclass: false,
            }
        }
    }

    fn blend_config(name: &str) -> BlendParams {
        match name {
            "gbr1" => BlendParams::default(),
            "gbr2" => BlendParams { iters: 50, ..Default::default() },
            "gbr3" => BlendParams { iters: 10, ..Default::default() },
            "gbc1" => BlendParams { multiclass: true, ..Default::default() },
            // Bayesian-optimized regression GBR (tuned with the cfnade predictors in the set).
            "gbr_opt" => BlendParams {
                num_leaves: 95, iters: 192, lr: 0.044121174451031774,
                min_data_in_leaf: 388, feature_fraction: 0.9740037968504494,
                bagging_fraction: 0.9171367599808904, bagging_freq: 3,
                lambda_l1: 4.074486360693566, lambda_l2: 4.156792180633296e-7,
                max_bin: 127, ..Default::default()
            },
            _ => panic!("unknown blend job '{name}' (add a branch in blend_config)"),
        }
    }

    fn print_help() {
        println!("Usage: gbm-new NAME [-n | -p FILE] [-t FILE] (--seeds N,N,... | --seed N)");
        println!();
        println!("  NAME                       blend name; LightGBM params come from");
        println!("                             blend_config(NAME); produces NAME-s<seed>.*");
        println!("  -n, --new                  use pipeline-new.toml for [split]");
        println!("  -p FILE, --pipeline FILE   pipeline TOML (default: pipeline-old.toml)");
        println!("  -t FILE, --models FILE     base-predictor models TOML (default: models-new.toml)");
        println!("  --groups G,G,...           model groups to use (default: all groups in the TOML)");
        println!("  -x NAME, --exclude NAME    drop a model by name (repeatable; brace-expanded)");
        println!("  --seeds N,N,...            fold seeds; one output NAME-s<N> per seed (data loaded once)");
        println!("  --seed N                   add a single fold seed (repeatable)");
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
            pipeline: "pipeline-old.toml".to_string(),
            models: "models-new.toml".to_string(),
            groups: Vec::new(),
            exclude: Vec::new(),
            seeds: Vec::new(),
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            match argv[i].as_str() {
                "-h" | "--help" => { print_help(); std::process::exit(0); }
                "-n" | "--new" => { a.pipeline = "pipeline-new.toml".to_string(); i += 1; }
                "-p" | "--pipeline" => { a.pipeline = need(&argv, i); i += 2; }
                "-t" | "--models" => { a.models = need(&argv, i); i += 2; }
                "--groups" => {
                    for tok in need(&argv, i).split(',') {
                        a.groups.push(tok.trim().to_string());
                    }
                    i += 2;
                }
                "-x" | "--exclude" => { a.exclude.push(need(&argv, i)); i += 2; }
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
        if a.seeds.is_empty() {
            eprintln!("error: provide --seeds N,N,... or --seed N");
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


    /// All voting-feature column names in `preds_dir` for `dataset`: files named
    /// `vf<digit>...{dataset}.npy`, with the `.{dataset}.npy` suffix stripped,
    /// sorted by name for deterministic column order.
    fn glob_voting(preds_dir: &str, dataset: &str) -> Vec<String> {
        let suffix = format!(".{dataset}.npy");
        let mut names: Vec<String> = std::fs::read_dir(preds_dir)
            .unwrap_or_else(|e| panic!("read_dir {preds_dir}: {e}"))
            .flatten()
            .filter_map(|e| {
                let f = e.file_name().to_string_lossy().into_owned();
                let is_vf = f.starts_with("vf")
                    && f.as_bytes().get(2).is_some_and(u8::is_ascii_digit);
                if is_vf && f.ends_with(&suffix) {
                    Some(f[..f.len() - suffix.len()].to_string())
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        names
    }

    pub fn run() -> ExitCode {
        let args = parse_args();

        let split = load_pipeline_split(&args.pipeline);
        let pr = split.get("pr").expect("[split].pr missing").clone();
        let fulltrain_pr = split.get("fulltrain_pr").expect("[split].fulltrain_pr missing").clone();
        let preds = split.get("preds").expect("[split].preds missing").clone();
        let split_name = split.get("name").cloned().unwrap_or_else(|| "?".to_string());

        let p = blend_config(&args.name);
        let mg = load_models_toml(&args.models);
        let groups = select_groups(&mg, &args.groups);
        let base = flatten_groups(&groups, &args.exclude).specs();
        let voting = glob_voting(&preds, &pr);

        let seeds_str = args.seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
        let groups_str = if args.groups.is_empty() { "all".to_string() } else { args.groups.join(",") };
        println!("Pipeline:  {} (split = {})", args.pipeline, split_name);
        println!("Models:    {} ({} base predictors, groups: {})", args.models, base.len(), groups_str);
        if !args.exclude.is_empty() {
            println!("Excluded:  {} name(s): {}", args.exclude.len(), args.exclude.join(", "));
        }
        println!("Voting:    {} features (glob {}/vf*.{}.npy)", voting.len(), preds, pr);
        println!("Columns:   {}", base.len() + voting.len());
        println!("Seeds:     {} (fold permutation)", seeds_str);
        println!("Params:    {:?}", p);
        println!();

        println!("Loading probe set ({})...", pr);
        let (x_pr, y_pr) = build_xy(&base, &voting, &preds, &preds, &pr);
        println!("Loading qual set ({})...", fulltrain_pr);
        let (x_ql, y_ql) = build_xy(&base, &voting, &preds, &preds, &fulltrain_pr);
        let qz = load_quiz_mask(&fulltrain_pr);
        println!("Probe: {} rows, Qual: {} rows", y_pr.len(), y_ql.len());

        let mode = if p.multiclass {
            LgbmMode::Multiclass { values: vec![1.0, 2.0, 3.0, 4.0, 5.0] }
        } else {
            LgbmMode::Regression
        };
        let cfg = LgbmCfg {
            mode,
            num_iterations: p.iters,
            num_leaves: p.num_leaves,
            learning_rate: p.lr,
            min_data_in_leaf: p.min_data_in_leaf,
            feature_fraction: p.feature_fraction,
            bagging_fraction: p.bagging_fraction,
            bagging_freq: p.bagging_freq,
            lambda_l1: p.lambda_l1,
            lambda_l2: p.lambda_l2,
            max_bin: p.max_bin,
            num_threads: p.threads,
            ..Default::default()
        };

        // Feature matrices are loaded once and reused across all seeds.
        for &seed in &args.seeds {
            println!();
            println!("=== seed {} ===", seed);
            let (p_pr, p_ql) = cvk_blend::<LgbmBlender>(
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
            println!("Saved {} / {}", pr_path, ql_path);
        }

        ExitCode::SUCCESS
    }
}
