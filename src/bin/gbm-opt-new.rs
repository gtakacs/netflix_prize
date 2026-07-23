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
    use indexmap::IndexMap;
    use lightgbm3::{Booster, Dataset};
    use netflix_prize::blend::{
        build_xy, cvk_blend, expand_specs, load_quiz_mask, save_preds, Blender, LgbmBlender, LgbmCfg,
        LgbmMode, NOCLIP_OP,
    };
    use serde_json::json;
    use std::collections::HashMap;
    use std::process::ExitCode;

    struct Args {
        name: String,
        pipeline: String,
        models: String,
        seeds: Vec<u64>,
    }

    /// LightGBM + cross-fit parameters for a named blend. Add a branch per blend
    /// (mirrors the base-predictor dispatchers in `tsvdx4-new.rs` etc.).
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
        ordinal: bool,
    }

    fn blend_config(name: &str) -> BlendParams {
        match name {
            "gbr1" => BlendParams { num_leaves: 63, iters: 200, lr: 0.1, min_data_in_leaf: 20, feature_fraction: 1.0, bagging_fraction: 1.0, bagging_freq: 0, lambda_l1: 0.0, lambda_l2: 0.0, max_bin: 255, folds: 2, threads: 16, multiclass: false, ordinal: false },
            "gbr2" => BlendParams { num_leaves: 63, iters: 50,  lr: 0.1, min_data_in_leaf: 20, feature_fraction: 1.0, bagging_fraction: 1.0, bagging_freq: 0, lambda_l1: 0.0, lambda_l2: 0.0, max_bin: 255, folds: 2, threads: 16, multiclass: false, ordinal: false },
            "gbr3" => BlendParams { num_leaves: 63, iters: 10,  lr: 0.1, min_data_in_leaf: 20, feature_fraction: 1.0, bagging_fraction: 1.0, bagging_freq: 0, lambda_l1: 0.0, lambda_l2: 0.0, max_bin: 255, folds: 2, threads: 16, multiclass: false, ordinal: false },
            "gbc1" => BlendParams { num_leaves: 63, iters: 200, lr: 0.1, min_data_in_leaf: 20, feature_fraction: 1.0, bagging_fraction: 1.0, bagging_freq: 0, lambda_l1: 0.0, lambda_l2: 0.0, max_bin: 255, folds: 2, threads: 16, multiclass: true, ordinal: false },
            "gbc1_opt" | "gbc1_opt_cfnade8" => BlendParams { num_leaves: 191, iters: 400, lr: 0.019929113104398585, min_data_in_leaf: 527, feature_fraction: 0.6706351917324398, bagging_fraction: 0.9755609862783436, bagging_freq: 3, lambda_l1: 2.0251302591612124e-7, lambda_l2: 5.4822642770934344e-6, max_bin: 127, folds: 2, threads: 16, multiclass: true, ordinal: false },
            "gbc_ord" | "gbc_ord_cfnade8" => BlendParams { num_leaves: 63, iters: 200, lr: 0.1, min_data_in_leaf: 20, feature_fraction: 1.0, bagging_fraction: 1.0, bagging_freq: 0, lambda_l1: 0.0, lambda_l2: 0.0, max_bin: 255, folds: 2, threads: 16, multiclass: false, ordinal: true },
            "gbc_ord_opt" | "gbc_ord_opt_cfnade8" => BlendParams { num_leaves: 95, iters: 364, lr: 0.03700103023848226, min_data_in_leaf: 59, feature_fraction: 0.9267960564266805, bagging_fraction: 0.8499096112078639, bagging_freq: 5, lambda_l1: 5.2509871516768145, lambda_l2: 0.028622756447435036, max_bin: 127, folds: 2, threads: 16, multiclass: false, ordinal: true },
            "gbr_opt" | "gbr_opt_cfnade8" => BlendParams { num_leaves: 95, iters: 192, lr: 0.044121174451031774, min_data_in_leaf: 388, feature_fraction: 0.9740037968504494, bagging_fraction: 0.9171367599808904, bagging_freq: 3, lambda_l1: 4.074486360693566, lambda_l2: 4.156792180633296e-7, max_bin: 127, folds: 2, threads: 16, multiclass: false, ordinal: false },
            _ => panic!("unknown blend job '{name}' (add a branch in blend_config)"),
        }
    }

    struct OrdinalLgbmBlender {
        boosters: Vec<Booster>,
    }

    impl Blender for OrdinalLgbmBlender {
        type Cfg = LgbmCfg;

        fn fit(x: &[f32], y: &[f32], n_features: usize, cfg: &LgbmCfg) -> Self {
            let params = json!({
                "objective": "binary",
                "metric": "binary_logloss",
                "num_iterations": cfg.num_iterations,
                "num_leaves": cfg.num_leaves,
                "learning_rate": cfg.learning_rate,
                "min_data_in_leaf": cfg.min_data_in_leaf,
                "feature_fraction": cfg.feature_fraction,
                "bagging_fraction": cfg.bagging_fraction,
                "bagging_freq": cfg.bagging_freq,
                "lambda_l1": cfg.lambda_l1,
                "lambda_l2": cfg.lambda_l2,
                "max_bin": cfg.max_bin,
                "num_threads": cfg.num_threads,
                "seed": cfg.seed,
                "verbosity": cfg.verbosity,
            });

            let mut boosters = Vec::with_capacity(4);
            for threshold in 2..=5 {
                let label: Vec<f32> = y
                    .iter()
                    .map(|&r| if r >= threshold as f32 { 1.0 } else { 0.0 })
                    .collect();
                let dataset = Dataset::from_slice(x, &label, n_features as i32, true)
                    .expect("lightgbm ordinal Dataset::from_slice");
                let booster = Booster::train(dataset, &params).expect("lightgbm ordinal Booster::train");
                boosters.push(booster);
            }
            Self { boosters }
        }

        fn predict(&self, x: &[f32], n_features: usize) -> Vec<f32> {
            let n = x.len() / n_features;
            let mut out = vec![1.0_f32; n];
            for booster in &self.boosters {
                let p = booster
                    .predict(x, n_features as i32, true)
                    .expect("lightgbm ordinal predict");
                assert_eq!(p.len(), n, "binary ordinal prediction length mismatch");
                for (dst, &prob) in out.iter_mut().zip(&p) {
                    *dst += prob as f32;
                }
            }
            out
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

    fn load_models_toml(path: &str) -> IndexMap<String, Vec<String>> {
        let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        toml::from_str(&s).unwrap_or_else(|e| panic!("parse {path}: {e}"))
    }

    /// Flatten all model groups into a single spec list, deduplicating by the
    /// clip-stripped name. If a name appears both clipped and `>`-prefixed, the
    /// no-clip variant wins (matching the ridge registry semantics).
    fn flatten_models(groups: &IndexMap<String, Vec<String>>) -> Vec<String> {
        let mut idx_of: HashMap<String, usize> = HashMap::new();
        let mut out: Vec<String> = Vec::new();
        for specs in groups.values() {
            for raw in specs {
                for spec in expand_specs(raw) {
                    let noclip = spec.starts_with(NOCLIP_OP);
                    let name = spec.trim_start_matches(NOCLIP_OP).to_string();
                    match idx_of.get(&name) {
                        Some(&i) => {
                            if noclip && !out[i].starts_with(NOCLIP_OP) {
                                out[i] = format!("{NOCLIP_OP}{name}");
                            }
                        }
                        None => {
                            idx_of.insert(name, out.len());
                            out.push(spec);
                        }
                    }
                }
            }
        }
        out
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
        let groups = load_models_toml(&args.models);
        let base = flatten_models(&groups);
        let voting = glob_voting(&preds, &pr);

        let seeds_str = args.seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
        println!("Pipeline:  {} (split = {})", args.pipeline, split_name);
        println!("Models:    {} ({} base predictors)", args.models, base.len());
        println!("Voting:    {} features (glob {}/vf*.{}.npy)", voting.len(), preds, pr);
        println!("Columns:   {}", base.len() + voting.len());
        println!("Seeds:     {} (fold permutation)", seeds_str);
        println!("Folds:     {}", p.folds);
        let objective = if p.ordinal {
            "ordinal-binary-thresholds"
        } else if p.multiclass {
            "multiclass"
        } else {
            "regression"
        };
        println!(
            "LightGBM:  objective={} num_leaves={} iters={} lr={} min_data_in_leaf={} feature_fraction={} bagging_fraction={} bagging_freq={} lambda_l1={} lambda_l2={} max_bin={} threads={}",
            objective, p.num_leaves, p.iters, p.lr, p.min_data_in_leaf, p.feature_fraction,
            p.bagging_fraction, p.bagging_freq, p.lambda_l1, p.lambda_l2, p.max_bin, p.threads,
        );
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
            let (p_pr, p_ql) = if p.ordinal {
                cvk_blend::<OrdinalLgbmBlender>(
                    &x_pr,
                    y_pr.as_slice().unwrap(),
                    &x_ql,
                    y_ql.as_slice().unwrap(),
                    qz.as_slice().unwrap(),
                    p.folds,
                    seed,
                    &cfg,
                )
            } else {
                cvk_blend::<LgbmBlender>(
                    &x_pr,
                    y_pr.as_slice().unwrap(),
                    &x_ql,
                    y_ql.as_slice().unwrap(),
                    qz.as_slice().unwrap(),
                    p.folds,
                    seed,
                    &cfg,
                )
            };
            let pr_path = format!("{preds}/{}-s{seed}.{pr}.npy", args.name);
            let ql_path = format!("{preds}/{}-s{seed}.{fulltrain_pr}.npy", args.name);
            save_preds(&pr_path, &p_pr);
            save_preds(&ql_path, &p_ql);
            println!("Saved {} / {}", pr_path, ql_path);
        }

        ExitCode::SUCCESS
    }
}
