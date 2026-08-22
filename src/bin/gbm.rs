//! LightGBM cross-fit blending of base predictors + voting features, with a
//! regression (`gbr*`) or multiclass-classifier (`gbc*`) objective per job.
//! Counterpart to `ridge.rs`; only functional when built with `--features lgbm`.

#[cfg(feature = "lgbm")]
fn main() -> std::process::ExitCode {
    real::run()
}

#[cfg(not(feature = "lgbm"))]
fn main() {
    eprintln!("gbm requires the 'lgbm' feature:");
    eprintln!("  cargo build --release --features lgbm --bin gbm");
    std::process::exit(2);
}

#[cfg(feature = "lgbm")]
mod real {
    use netflix_prize::blend::{
        build_xy, close_log, cvk_blend, flatten_groups, load_models_toml, load_quiz_mask, log_columns,
        open_log, resolve_voting, save_preds, select_groups, LgbmBlender, LgbmCfg, LgbmMode,
        CLIP_MAX, CLIP_MIN,
    };
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
        in_clip: (f32, f32),
        voting_models: String,
        voting: Vec<String>,
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
            "gbr1g" | "gbr1g1" | "gbr1g2" | "gbr1g3" | "gbr1g4" => BlendParams::default(),
            "gbr2" | "gbr2x" => BlendParams { iters: 50, ..Default::default() },
            "gbr3" | "gbr3x" => BlendParams { iters: 10, ..Default::default() },

            // Bayesian-optimized regression GBR.
            "gbr4o" | "gbr4ox" => BlendParams {
                num_leaves: 95, iters: 192, lr: 0.044121174451031774,
                min_data_in_leaf: 388, feature_fraction: 0.9740037968504494,
                bagging_fraction: 0.9171367599808904, bagging_freq: 3,
                lambda_l1: 4.074486360693566, lambda_l2: 4.156792180633296e-7,
                max_bin: 127, ..Default::default()
            },
            "gbr4g1" | "gbr4g2" | "gbr4g3" | "gbr4g4" | "gbr4x" => BlendParams {
                num_leaves: 95, iters: 192, lr: 0.044121174451031774,
                min_data_in_leaf: 388, feature_fraction: 0.9740037968504494,
                bagging_fraction: 0.9171367599808904, bagging_freq: 3,
                lambda_l1: 4.074486360693566, lambda_l2: 4.156792180633296e-7,
                max_bin: 127, ..Default::default()
            },

            "gbc1" | "gbc1x" | "gbc1y" => BlendParams { multiclass: true, ..Default::default() },

            // Bayesian-optimized multiclass classifier GBC.
            "gbc2o" | "gbc2x" => BlendParams {
                num_leaves: 191, iters: 400, lr: 0.019929113104398585,
                min_data_in_leaf: 527, feature_fraction: 0.6706351917324398,
                bagging_fraction: 0.9755609862783436, bagging_freq: 3,
                lambda_l1: 2.0251302591612124e-7, lambda_l2: 5.4822642770934344e-6,
                max_bin: 127, multiclass: true, ..Default::default()
            },
            _ => panic!("unknown blend job '{name}' (add a branch in blend_config)"),
        }
    }

    fn print_help() {
        println!("Usage: gbm NAME -p FILE -t FILE --voting-models FILE --voting G,... (--seeds N,N,... | --seed N)");
        println!();
        println!("  NAME                       blend name; LightGBM params come from");
        println!("                             blend_config(NAME); produces NAME-s<seed>.*");
        println!("  -p FILE, --pipeline FILE   pipeline TOML (required; carries the split)");
        println!("  -t FILE, --models FILE     base-predictor models TOML (required)");
        println!("  --groups G,G,...           model groups (default: the TOML's `all`; omit with -m for manual-only)");
        println!("  -m NAME, --model NAME      add a single base predictor (repeatable; combines with --groups)");
        println!("  -x NAME, --exclude NAME    drop a model by name (repeatable; brace-expanded)");
        println!("  --in-clip MIN,MAX          clip range for clipped model columns (default {CLIP_MIN},{CLIP_MAX})");
        println!("  --voting-models FILE       voting-feature groups TOML (required)");
        println!("  --voting G,G,...           voting-feature groups to use (required; 'all' if the TOML defines it)");
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
            pipeline: String::new(),
            models: String::new(),
            groups: Vec::new(),
            model_manual: Vec::new(),
            exclude: Vec::new(),
            in_clip: (CLIP_MIN, CLIP_MAX),
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
                "--in-clip" => {
                    let raw = need(&argv, i);
                    let (lo, hi) = raw.split_once(',').unwrap_or_else(|| {
                        eprintln!("error: --in-clip expects MIN,MAX (got '{raw}')");
                        std::process::exit(2)
                    });
                    a.in_clip = (
                        lo.trim().parse().expect("bad --in-clip MIN"),
                        hi.trim().parse().expect("bad --in-clip MAX"),
                    );
                    i += 2;
                }
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


    pub fn run() -> ExitCode {
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
        teeln!("In-clip:   [{}, {}] (clipped model columns)", args.in_clip.0, args.in_clip.1);
        if !args.exclude.is_empty() {
            teeln!("Excluded:  {} name(s): {}", args.exclude.len(), args.exclude.join(", "));
        }
        teeln!("Voting:    {} ({} features, groups: {})", args.voting_models, voting.len(), voting_str);
        teeln!("Seeds:     {} (fold permutation)", seeds_str);
        teeln!("Params:    {:?}", p);
        log_columns(&base, &voting);
        teeln!();

        println!("Loading probe set ({})...", pr);
        let (x_pr, y_pr) = build_xy(&base, &voting, &preds, &preds, &pr, args.in_clip);
        println!("Loading qual set ({})...", fulltrain_pr);
        let (x_ql, y_ql) = build_xy(&base, &voting, &preds, &preds, &fulltrain_pr, args.in_clip);
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
            teeln!();
            teeln!("=== seed {} ===", seed);
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
            teeln!("Saved {} / {}", pr_path, ql_path);
        }

        close_log();
        ExitCode::SUCCESS
    }
}
