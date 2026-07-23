//! Ordinal-classifier MLP cross-fit blending of base predictors + voting features.
//!
//! This is an experimental sibling of `mlp-new.rs` that keeps all changes inside
//! `src/bin`. It uses shared ReLU hidden layers with four sigmoid heads trained
//! by binary cross-entropy for thresholds y>=2, y>=3, y>=4, y>=5. Predictions are
//! converted back to ratings as `1 + sum(sigmoid(heads))`.

#[cfg(feature = "blas")]
extern crate blas_src;

use indexmap::IndexMap;
use ndarray::{Array1, Array2, Axis};
use netflix_prize::blend::{
    build_xy, cvk_blend, expand_specs, load_quiz_mask, save_preds, Blender, NOCLIP_OP,
};
use indicatif::ProgressIterator;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use std::collections::HashMap;
use std::process::ExitCode;

struct Args {
    name: String,
    pipeline: String,
    models: String,
    seeds: Vec<u64>,
}

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
}

fn blend_config(name: &str) -> BlendParams {
    match name {
        // First try: reuse the mlpr_opt hyperparameters, but train as a
        // 4-head ordinal classifier with BCE/logloss.
        "mlpc_ord" => BlendParams { hidden: vec![32, 32], alpha: 0.24101624563575427, lr: 0.0005339808669500829, iters: 80, batch: 200, momentum: 0.9148116502629432, tol: 2.890613343488106e-5, n_iter_no_change: 9, folds: 2 },
        "mlpc_ord_opt" | "mlpc_ord_opt_cfnade8" => BlendParams { hidden: vec![64, 32], alpha: 0.9723373672604836, lr: 0.000953329728567159, iters: 80, batch: 400, momentum: 0.9380845452666436, tol: 1.4640841816087311e-5, n_iter_no_change: 7, folds: 2 },
        // Baseline mlpr1-sized net, kept handy for a cheap contrast.
        "mlpc_ord_base" => BlendParams { hidden: vec![64, 64], alpha: 0.05, lr: 0.0004, iters: 64, batch: 200, momentum: 0.9, tol: 1e-4, n_iter_no_change: 10, folds: 2 },
        _ => panic!("unknown blend job '{name}' (add a branch in blend_config)"),
    }
}

fn print_help() {
    println!("Usage: mlp-ordinal-new NAME [-n | -p FILE] [-t FILE] (--seeds N,N,... | --seed N)");
    println!();
    println!("  NAME                       ordinal MLP blend name; produces NAME-s<seed>.*");
    println!("  -n, --new                  use pipeline-new.toml for [split]");
    println!("  -p FILE, --pipeline FILE   pipeline TOML (default: pipeline-old.toml)");
    println!("  -t FILE, --models FILE     base-predictor models TOML (default: models-new.toml)");
    println!("  --seeds N,N,...            net+fold seeds; one output NAME-s<N> per seed");
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
        pipeline: "pipeline-old.toml".to_string(),
        models: "models-new.toml".to_string(),
        seeds: Vec::new(),
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-n" | "--new" => {
                a.pipeline = "pipeline-new.toml".to_string();
                i += 1;
            }
            "-p" | "--pipeline" => {
                a.pipeline = need(&argv, i);
                i += 2;
            }
            "-t" | "--models" => {
                a.models = need(&argv, i);
                i += 2;
            }
            "--seed" => {
                a.seeds.push(need(&argv, i).parse().expect("bad --seed"));
                i += 2;
            }
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

fn glob_voting(preds_dir: &str, dataset: &str) -> Vec<String> {
    let suffix = format!(".{dataset}.npy");
    let mut names: Vec<String> = std::fs::read_dir(preds_dir)
        .unwrap_or_else(|e| panic!("read_dir {preds_dir}: {e}"))
        .flatten()
        .filter_map(|e| {
            let f = e.file_name().to_string_lossy().into_owned();
            let is_vf = f.starts_with("vf") && f.as_bytes().get(2).is_some_and(u8::is_ascii_digit);
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

#[derive(Clone, Debug)]
struct OrdinalMlpCfg {
    hidden: Vec<usize>,
    alpha: f64,
    lr: f64,
    max_iter: usize,
    batch_size: usize,
    momentum: f64,
    tol: f64,
    n_iter_no_change: usize,
    seed: u64,
}

impl Default for OrdinalMlpCfg {
    fn default() -> Self {
        Self {
            hidden: vec![64, 64],
            alpha: 0.05,
            lr: 0.0004,
            max_iter: 64,
            batch_size: 200,
            momentum: 0.9,
            tol: 1e-4,
            n_iter_no_change: 10,
            seed: 1,
        }
    }
}

struct Scaler {
    mean: Array1<f64>,
    scale: Array1<f64>,
}

impl Scaler {
    fn fit(x: &Array2<f64>) -> Self {
        let mean = x.mean_axis(Axis(0)).unwrap();
        let meanb = mean.broadcast(x.raw_dim()).unwrap();
        let var = (x - &meanb).mapv(|v| v * v).mean_axis(Axis(0)).unwrap();
        let scale = var.mapv(|v| {
            let s = v.sqrt();
            if s == 0.0 { 1.0 } else { s }
        });
        Self { mean, scale }
    }

    fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let meanb = self.mean.broadcast(x.raw_dim()).unwrap();
        let scaleb = self.scale.broadcast(x.raw_dim()).unwrap();
        (x - &meanb) / scaleb
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn forward(
    coefs: &[Array2<f64>],
    intercepts: &[Array1<f64>],
    x0: Array2<f64>,
) -> Vec<Array2<f64>> {
    let l = coefs.len();
    let mut acts = Vec::with_capacity(l + 1);
    acts.push(x0);
    for li in 0..l {
        let mut z = acts[li].dot(&coefs[li]);
        let b = intercepts[li].broadcast(z.raw_dim()).unwrap();
        z += &b;
        if li < l - 1 {
            z.mapv_inplace(|v| v.max(0.0));
        } else {
            z.mapv_inplace(sigmoid);
        }
        acts.push(z);
    }
    acts
}

fn ordinal_targets(y: f64) -> [f64; 4] {
    [
        if y >= 2.0 { 1.0 } else { 0.0 },
        if y >= 3.0 { 1.0 } else { 0.0 },
        if y >= 4.0 { 1.0 } else { 0.0 },
        if y >= 5.0 { 1.0 } else { 0.0 },
    ]
}

fn ordinal_prediction(row: ndarray::ArrayView1<'_, f64>) -> f64 {
    (1.0 + row.sum()).clamp(1.0, 5.0)
}

struct OrdinalMlpBlender {
    scaler: Scaler,
    coefs: Vec<Array2<f64>>,
    intercepts: Vec<Array1<f64>>,
}

impl Blender for OrdinalMlpBlender {
    type Cfg = OrdinalMlpCfg;

    fn fit(x: &[f32], y: &[f32], n_features: usize, cfg: &OrdinalMlpCfg) -> Self {
        let d = n_features;
        let n = y.len();
        assert_eq!(x.len(), n * d, "x length {} != {n}*{d}", x.len());

        let x64 = Array2::from_shape_fn((n, d), |(i, j)| x[i * d + j] as f64);
        let scaler = Scaler::fit(&x64);
        let xs = scaler.transform(&x64);
        drop(x64);

        let mut sizes = vec![d];
        sizes.extend(cfg.hidden.iter().copied());
        sizes.push(4);
        let l = sizes.len() - 1;

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut coefs: Vec<Array2<f64>> = Vec::with_capacity(l);
        let mut intercepts: Vec<Array1<f64>> = Vec::with_capacity(l);
        for li in 0..l {
            let (fan_in, fan_out) = (sizes[li], sizes[li + 1]);
            let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let u = Uniform::new(-bound, bound).unwrap();
            let w = Array2::from_shape_fn((fan_in, fan_out), |_| u.sample(&mut rng));
            let b = Array1::from_shape_fn(fan_out, |_| u.sample(&mut rng));
            coefs.push(w);
            intercepts.push(b);
        }

        let mut v_coefs: Vec<Array2<f64>> = coefs.iter().map(|c| Array2::zeros(c.raw_dim())).collect();
        let mut v_inter: Vec<Array1<f64>> = intercepts.iter().map(|b| Array1::zeros(b.raw_dim())).collect();

        let mut idx: Vec<usize> = (0..n).collect();
        let mut best_loss = f64::INFINITY;
        let mut no_improve = 0usize;

        for _epoch in (0..cfg.max_iter).progress() {
            idx.shuffle(&mut rng);
            let mut accumulated = 0.0f64;

            for batch in idx.chunks(cfg.batch_size) {
                let m = batch.len();
                let nb = m as f64;
                let xb = Array2::from_shape_fn((m, d), |(i, j)| xs[[batch[i], j]]);
                let yb = Array2::from_shape_fn((m, 4), |(i, j)| {
                    ordinal_targets(y[batch[i]] as f64)[j]
                });

                let acts = forward(&coefs, &intercepts, xb);
                let eps = 1e-12;
                let bce = acts[l]
                    .iter()
                    .zip(yb.iter())
                    .map(|(&q, &t)| {
                        let q = q.clamp(eps, 1.0 - eps);
                        -(t * q.ln() + (1.0 - t) * (1.0 - q).ln())
                    })
                    .sum::<f64>() / nb;
                let l2: f64 = coefs.iter().map(|c| c.iter().map(|w| w * w).sum::<f64>()).sum();
                let batch_loss = bce + 0.5 * cfg.alpha * l2 / nb;
                accumulated += batch_loss * nb;

                let mut coef_grads: Vec<Array2<f64>> = vec![Array2::zeros((0, 0)); l];
                let mut inter_grads: Vec<Array1<f64>> = vec![Array1::zeros(0); l];
                // BCE with sigmoid output gives the simple logit gradient q - t.
                let mut delta = &acts[l] - &yb;

                let mut layer = l - 1;
                loop {
                    let mut cg = acts[layer].t().dot(&delta);
                    cg.scaled_add(cfg.alpha, &coefs[layer]);
                    cg.mapv_inplace(|v| v / nb);
                    inter_grads[layer] = delta.mean_axis(Axis(0)).unwrap();
                    coef_grads[layer] = cg;

                    if layer == 0 {
                        break;
                    }
                    let mut prev = delta.dot(&coefs[layer].t());
                    prev.zip_mut_with(&acts[layer], |g, &a| {
                        if a == 0.0 {
                            *g = 0.0;
                        }
                    });
                    delta = prev;
                    layer -= 1;
                }

                for li in 0..l {
                    v_coefs[li].mapv_inplace(|v| v * cfg.momentum);
                    v_coefs[li].scaled_add(-cfg.lr, &coef_grads[li]);
                    let mut upd = &v_coefs[li] * cfg.momentum;
                    upd.scaled_add(-cfg.lr, &coef_grads[li]);
                    coefs[li] += &upd;

                    v_inter[li].mapv_inplace(|v| v * cfg.momentum);
                    v_inter[li].scaled_add(-cfg.lr, &inter_grads[li]);
                    let mut updb = &v_inter[li] * cfg.momentum;
                    updb.scaled_add(-cfg.lr, &inter_grads[li]);
                    intercepts[li] += &updb;
                }
            }

            let loss = accumulated / n as f64;
            if loss > best_loss - cfg.tol {
                no_improve += 1;
            } else {
                no_improve = 0;
            }
            if loss < best_loss {
                best_loss = loss;
            }
            if no_improve > cfg.n_iter_no_change {
                break;
            }
        }

        Self { scaler, coefs, intercepts }
    }

    fn predict(&self, x: &[f32], n_features: usize) -> Vec<f32> {
        let d = n_features;
        let n = x.len() / d;
        let l = self.coefs.len();
        const CHUNK: usize = 100_000;
        let mut out = Vec::with_capacity(n);
        let mut start = 0;
        while start < n {
            let m = (n - start).min(CHUNK);
            let xb = Array2::from_shape_fn((m, d), |(i, j)| x[(start + i) * d + j] as f64);
            let xs = self.scaler.transform(&xb);
            let acts = forward(&self.coefs, &self.intercepts, xs);
            out.extend(acts[l].rows().into_iter().map(|row| ordinal_prediction(row) as f32));
            start += m;
        }
        out
    }
}

fn main() -> ExitCode {
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
    println!("Seeds:     {} (net init + fold permutation)", seeds_str);
    println!("Folds:     {}", p.folds);
    println!(
        "MLP:       hidden={:?} alpha={} lr={} iters={} batch={} momentum={} tol={} patience={} output=1+sum(sigmoid(4 heads)) loss=BCE",
        p.hidden, p.alpha, p.lr, p.iters, p.batch, p.momentum, p.tol, p.n_iter_no_change,
    );
    println!();

    println!("Loading probe set ({})...", pr);
    let (x_pr, y_pr) = build_xy(&base, &voting, &preds, &preds, &pr);
    println!("Loading qual set ({})...", fulltrain_pr);
    let (x_ql, y_ql) = build_xy(&base, &voting, &preds, &preds, &fulltrain_pr);
    let qz = load_quiz_mask(&fulltrain_pr);
    println!("Probe: {} rows, Qual: {} rows", y_pr.len(), y_ql.len());

    for &seed in &args.seeds {
        println!();
        println!("=== seed {} ===", seed);
        let cfg = OrdinalMlpCfg {
            hidden: p.hidden.clone(),
            alpha: p.alpha,
            lr: p.lr,
            max_iter: p.iters,
            batch_size: p.batch,
            momentum: p.momentum,
            tol: p.tol,
            n_iter_no_change: p.n_iter_no_change,
            seed,
            ..Default::default()
        };
        let (p_pr, p_ql) = cvk_blend::<OrdinalMlpBlender>(
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
