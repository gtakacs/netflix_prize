//! FWLS (Feature-Weighted Linear Stacking) cross-fit blend: ridge regression on
//! the interaction features (model prediction × voting feature) with a 2-fold
//! split, mirroring `blending/fwls.py`. The Gram matrix over the `D = M·P + 1`
//! interaction columns is accumulated in row blocks via BLAS `dsyrk`. Dispatcher
//! shape matches `gbm`/`mlp`; requires the `blas` feature (OpenBLAS).

extern crate blas;
extern crate blas_src;

use blas::dsyrk;
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use ndarray_npy::read_npy;
use netflix_prize::blend::{
    close_log, flatten_groups, load_models_toml, log_columns, open_log, resolve_voting, save_preds,
    select_groups, CLIP_MAX, CLIP_MIN,
};
use netflix_prize::teeln;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::process::ExitCode;

/// Row block for the streamed qual prediction passes.
const ROW_BLOCK: usize = 100_000;
/// Target byte size of one `D × blen` f64 interaction block during the Gram.
const GRAM_BYTES: usize = 256_000_000;

// ---------------------------------------------------------------------------
// Partial .npy reader for 1-D float32 arrays (qual streaming)
// ---------------------------------------------------------------------------

struct NpyF32Reader {
    file: BufReader<File>,
    data_offset: u64,
    len: usize,
}

impl NpyF32Reader {
    fn open(path: &str) -> Self {
        let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
        let mut r = BufReader::new(file);
        let mut magic = [0u8; 6];
        r.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"\x93NUMPY", "bad magic in {}", path);
        let mut ver = [0u8; 2];
        r.read_exact(&mut ver).unwrap();
        let (hlen, preamble) = if ver[0] == 1 {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).unwrap();
            (u16::from_le_bytes(b) as u64, 10u64)
        } else {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).unwrap();
            (u32::from_le_bytes(b) as u64, 12u64)
        };
        let mut header_bytes = vec![0u8; hlen as usize];
        r.read_exact(&mut header_bytes).unwrap();
        let header = std::str::from_utf8(&header_bytes).expect("non-utf8 npy header");
        assert!(
            header.contains("'<f4'") || header.contains("'descr': '<f4'"),
            "dtype not <f4 in {}: {}", path, header.trim(),
        );
        let shape_idx = header.find("'shape':").expect("no shape field");
        let after = &header[shape_idx..];
        let open = after.find('(').expect("no (");
        let close = after.find(')').expect("no )");
        let inside = &after[open + 1..close];
        let len: usize = inside
            .split(',').next().unwrap().trim()
            .parse().unwrap_or_else(|_| panic!("bad shape in {}", path));
        Self { file: r, data_offset: preamble + hlen, len }
    }

    fn read_block(&mut self, start: usize, count: usize, out: &mut [f32]) {
        assert_eq!(out.len(), count);
        assert!(start + count <= self.len, "out-of-range read in npy file");
        let byte_offset = self.data_offset + (start as u64) * 4;
        self.file.seek(SeekFrom::Start(byte_offset)).unwrap();
        let mut buf = vec![0u8; count * 4];
        self.file.read_exact(&mut buf).unwrap();
        for (i, chunk) in buf.chunks_exact(4).enumerate() {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    name: String,
    pipeline: String,
    models: String,
    groups: Vec<String>,
    model_manual: Vec<String>,
    exclude: Vec<String>,
    voting_models: String,
    voting: Vec<String>,
    feature_manual: Vec<String>,
    lambda: Option<f64>,
    seeds: Vec<u64>,
}

/// FWLS parameters for a named blend. Add a branch per blend.
struct BlendParams {
    lambda: f64,
}

/// Preset FWLS parameters for a known blend name, or `None` for an ad-hoc name
/// (which then requires `--lambda`).
fn blend_config(name: &str) -> Option<BlendParams> {
    match name {
        "fwls1" => Some(BlendParams { lambda: 10000.0 }),
        _ => None,
    }
}

fn print_help() {
    println!("Usage: fwls NAME [-n | -p FILE] [-t FILE] [--groups G,...] (--seeds N,... | --seed N)");
    println!();
    println!("  NAME                       blend name; a preset (blend_config) or any ad-hoc name");
    println!("                             (ad-hoc requires --lambda). Output goes to NAME-s<seed>.*");
    println!("  -n, --new                  use pipeline-new.toml for [split]");
    println!("  -p FILE, --pipeline FILE   pipeline TOML (default: pipeline-old.toml)");
    println!("  -t FILE, --models FILE     base-predictor models TOML (default: models-new.toml)");
    println!("  --groups G,G,...           model groups to use (default: all; omit with -m for manual-only)");
    println!("  -m NAME, --model NAME      add a single model (repeatable; combines with --groups)");
    println!("  -x NAME, --exclude NAME    drop a model by name (repeatable; brace-expanded)");
    println!("  --voting-models FILE       voting-feature groups TOML (default: voting-new.toml)");
    println!("  --voting G,G,...           voting-feature groups ('all' = every group); optional if -f given");
    println!("  -f NAME, --feature NAME    add a single voting feature (repeatable; may be the only source)");
    println!("  --lambda VALUE             ridge λ; overrides the preset, required for an ad-hoc name");
    println!("  --seeds N,N,...            fold seeds; one output NAME-s<N> per seed (data loaded once)");
    println!("  --seed N                   add a single fold seed (repeatable)");
    println!("  -h, --help                 show this help");
    println!();
    println!("  Context (voting) features come from the --voting-models TOML (names relative");
    println!("  to the preds dir, e.g. vf/<name> or a bare predictor).");
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
        model_manual: Vec::new(),
        exclude: Vec::new(),
        voting_models: "voting-new.toml".to_string(),
        voting: Vec::new(),
        feature_manual: Vec::new(),
        lambda: None,
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
            "-m" | "--model" => { a.model_manual.push(need(&argv, i)); i += 2; }
            "-f" | "--feature" => { a.feature_manual.push(need(&argv, i)); i += 2; }
            "--lambda" => { a.lambda = Some(need(&argv, i).parse().expect("bad --lambda value")); i += 2; }
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
    if a.seeds.is_empty() {
        eprintln!("error: provide --seeds N,N,... or --seed N");
        print_help();
        std::process::exit(2);
    }
    if a.voting.is_empty() && a.feature_manual.is_empty() {
        eprintln!("error: provide voting features via --voting GROUPS or -f NAME");
        print_help();
        std::process::exit(2);
    }
    a
}

// ---------------------------------------------------------------------------
// Pipeline / models TOML
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// Read `names` columns fully into memory (one Vec<f32> each), clipping column
/// `i` to [CLIP_MIN, CLIP_MAX] when `clip[i]`. Columns are loaded in parallel.
fn load_cols(names: &[String], clip: &[bool], preds_dir: &str, ds: &str, n: usize) -> Vec<Vec<f32>> {
    names
        .par_iter()
        .enumerate()
        .map(|(i, name)| {
            let path = format!("{preds_dir}/{name}.{ds}.npy");
            let a: Array1<f32> = read_npy(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            assert_eq!(a.len(), n, "{path}: len {} != {n}", a.len());
            let mut v = a.to_vec();
            if clip[i] {
                for x in v.iter_mut() {
                    *x = x.clamp(CLIP_MIN, CLIP_MAX);
                }
            }
            v
        })
        .collect()
}

fn load_ratings_f64(dataset: &str) -> Vec<f64> {
    let path = format!("data/{dataset}/ratings.npy");
    let r: Array1<i8> = read_npy(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    r.iter().map(|&v| v as f64).collect()
}

// ---------------------------------------------------------------------------
// FWLS fit / predict over interaction features
// ---------------------------------------------------------------------------

/// Accumulate the ridge system over the interaction features for `rows`:
/// `A = ZᵀZ + λI` (bias unregularized), `b = Zᵀy`, where each `Z` column is the
/// per-rating outer product `flatten(X[:,k] ⊗ F[:,k])` plus a 1.0 bias entry.
/// Built in row blocks; `dsyrk` fills the column-major lower triangle.
fn build_gram_fold(
    rows: &[usize],
    xpr: &[Vec<f32>],
    fpr: &[Vec<f32>],
    y: &[f64],
    m: usize,
    p: usize,
    lambda: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let d = m * p + 1;
    let mut amat = DMatrix::<f64>::zeros(d, d);
    let mut b = DVector::<f64>::zeros(d);
    let blen = (GRAM_BYTES / (8 * d)).clamp(256, rows.len().max(1));
    let mut z = vec![0.0f64; blen * d];
    let mut frow = vec![0.0f64; p];

    let nb = rows.len().div_ceil(blen);
    let mut start = 0;
    let mut bi = 0;
    while start < rows.len() {
        let bl = (rows.len() - start).min(blen);
        for kk in 0..bl {
            let row = rows[start + kk];
            for j in 0..p {
                frow[j] = fpr[j][row] as f64;
            }
            let zoff = kk * d;
            for i in 0..m {
                let xi = xpr[i][row] as f64;
                let ioff = zoff + i * p;
                for j in 0..p {
                    z[ioff + j] = xi * frow[j];
                }
            }
            z[zoff + m * p] = 1.0; // bias
        }

        // A += Z Zᵀ over this block's columns (lower triangle, column-major).
        unsafe {
            dsyrk(
                b'L', b'N',
                d as i32, bl as i32,
                1.0, &z[..bl * d], d as i32,
                1.0, amat.as_mut_slice(), d as i32,
            );
        }
        // b += Z y
        for kk in 0..bl {
            let yk = y[rows[start + kk]];
            let col = &z[kk * d..kk * d + d];
            for (dd, &c) in col.iter().enumerate() {
                b[dd] += c * yk;
            }
        }

        start += bl;
        bi += 1;
        eprint!("\r    gram block {}/{}", bi, nb);
    }
    eprintln!();

    // Mirror the lower triangle into the upper so the matrix is fully symmetric,
    // then ridge-regularize all but the bias diagonal entry.
    for c in 0..d {
        for r in (c + 1)..d {
            amat[(c, r)] = amat[(r, c)];
        }
    }
    for i in 0..(d - 1) {
        amat[(i, i)] += lambda;
    }
    (amat, b)
}

fn solve_fold(amat: DMatrix<f64>, b: DVector<f64>) -> Vec<f64> {
    let chol = amat.cholesky().expect("FWLS Gram matrix is not positive definite");
    chol.solve(&b).iter().copied().collect()
}

/// Predict in-memory `rows` from fitted weights `w` (length `D`), optionally
/// clipping. `yhat = bias + Σ_i x_i · (Σ_j w[i·P+j]·f_j)`.
fn predict_rows(
    rows: &[usize],
    xpr: &[Vec<f32>],
    fpr: &[Vec<f32>],
    w: &[f64],
    m: usize,
    p: usize,
    clip: bool,
) -> Vec<f64> {
    let bias = w[m * p];
    rows.par_iter()
        .map(|&row| {
            let mut yhat = bias;
            for i in 0..m {
                let xi = xpr[i][row] as f64;
                let woff = i * p;
                let mut gi = 0.0;
                for j in 0..p {
                    gi += w[woff + j] * (fpr[j][row] as f64);
                }
                yhat += xi * gi;
            }
            if clip {
                yhat = yhat.clamp(CLIP_MIN as f64, CLIP_MAX as f64);
            }
            yhat
        })
        .collect()
}

/// Stream the qual set in row blocks, predict with `w` (clipped), and add into
/// `acc` (length `n_q`). Used once per fold; the caller averages afterwards.
#[allow(clippy::too_many_arguments)]
fn predict_qual_acc(
    xr: &mut [NpyF32Reader],
    fr: &mut [NpyF32Reader],
    xclip: &[bool],
    w: &[f64],
    m: usize,
    p: usize,
    n_q: usize,
    acc: &mut [f64],
) {
    let bias = w[m * p];
    let mut xbuf: Vec<Vec<f32>> = (0..m).map(|_| vec![0.0f32; ROW_BLOCK]).collect();
    let mut fbuf: Vec<Vec<f32>> = (0..p).map(|_| vec![0.0f32; ROW_BLOCK]).collect();

    let nb = n_q.div_ceil(ROW_BLOCK);
    let mut start = 0;
    let mut bi = 0;
    while start < n_q {
        let bl = (n_q - start).min(ROW_BLOCK);
        xr.par_iter_mut().zip(xbuf.par_iter_mut()).enumerate().for_each(|(i, (r, buf))| {
            r.read_block(start, bl, &mut buf[..bl]);
            if xclip[i] {
                for v in buf[..bl].iter_mut() {
                    *v = v.clamp(CLIP_MIN, CLIP_MAX);
                }
            }
        });
        fr.par_iter_mut().zip(fbuf.par_iter_mut()).for_each(|(r, buf)| {
            r.read_block(start, bl, &mut buf[..bl]);
        });

        acc[start..start + bl].par_iter_mut().enumerate().for_each(|(k, a)| {
            let mut yhat = bias;
            for i in 0..m {
                let xi = xbuf[i][k] as f64;
                let woff = i * p;
                let mut gi = 0.0;
                for j in 0..p {
                    gi += w[woff + j] * (fbuf[j][k] as f64);
                }
                yhat += xi * gi;
            }
            *a += yhat.clamp(CLIP_MIN as f64, CLIP_MAX as f64);
        });

        start += bl;
        bi += 1;
        eprint!("\r    qual block {}/{}", bi, nb);
    }
    eprintln!();
}

fn rmse_sel(yhat: &[f64], y: &[f64], rows: &[usize]) -> f64 {
    let mut sse = 0.0;
    for (ii, &row) in rows.iter().enumerate() {
        let e = yhat[ii] - y[row];
        sse += e * e;
    }
    (sse / rows.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let args = parse_args();

    let split = load_pipeline_split(&args.pipeline);
    let pr = split.get("pr").expect("[split].pr missing").clone();
    let fulltrain_pr = split.get("fulltrain_pr").expect("[split].fulltrain_pr missing").clone();
    let preds = split.get("preds").expect("[split].preds missing").clone();
    let split_name = split.get("name").cloned().unwrap_or_else(|| "?".to_string());

    // Params from the named preset (overridable by --lambda), or fully from
    // --lambda for an ad-hoc blend name.
    let params = match blend_config(&args.name) {
        Some(mut p) => { if let Some(l) = args.lambda { p.lambda = l; } p }
        None => BlendParams {
            lambda: args.lambda.unwrap_or_else(|| {
                eprintln!("error: unknown blend '{}': provide --lambda VALUE", args.name);
                std::process::exit(2);
            }),
        },
    };
    // Models: from --groups of the models TOML, or purely from -m when no group is
    // named (manual-only skips loading the TOML entirely).
    let mut groups = if args.groups.is_empty() && !args.model_manual.is_empty() {
        indexmap::IndexMap::new()
    } else {
        select_groups(&load_models_toml(&args.models), &args.groups)
    };
    if !args.model_manual.is_empty() {
        groups.insert("manual".to_string(), args.model_manual.clone());
    }
    let flat = flatten_groups(&groups, &args.exclude);
    let (model_names, model_clip) = (flat.names, flat.clip);
    assert!(!model_names.is_empty(), "no models selected (use --groups or -m)");
    // Voting: from --voting of the voting TOML, plus any -f manual features. When
    // no group is named the voting TOML is not loaded (manual-only).
    let mut voting = if args.voting.is_empty() {
        Vec::new()
    } else {
        resolve_voting(&args.voting_models, &args.voting)
    };
    voting.extend(args.feature_manual.iter().cloned());
    assert!(!voting.is_empty(), "no voting features selected (use --voting or -f)");

    let m = model_names.len();
    let p = voting.len();
    let d = m * p + 1;
    let models_manual_only = args.groups.is_empty() && !args.model_manual.is_empty();
    let models_src = if models_manual_only { "(manual)".to_string() } else { args.models.clone() };
    let groups_str = if args.groups.is_empty() {
        if args.model_manual.is_empty() { "all".to_string() } else { "manual".to_string() }
    } else { args.groups.join(",") };
    let voting_src = if args.voting.is_empty() { "(manual)".to_string() } else { args.voting_models.clone() };
    let voting_str = if args.voting.is_empty() { "manual".to_string() } else { args.voting.join(",") };
    let seeds_str = args.seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(",");

    open_log(&preds, &args.name);
    teeln!("[{}]", args.name);
    teeln!("Pipeline:  {} (split = {})", args.pipeline, split_name);
    teeln!("Models:    {} ({} predictors, groups: {})", models_src, m, groups_str);
    if !args.exclude.is_empty() {
        teeln!("Excluded:  {} name(s): {}", args.exclude.len(), args.exclude.join(", "));
    }
    teeln!("Voting:    {} ({} context features, groups: {})", voting_src, p, voting_str);
    teeln!("Interact:  D = M·P + 1 = {}·{} + 1 = {}", m, p, d);
    teeln!("Lambda:    {}", params.lambda);
    teeln!("Seeds:     {} (2-fold split per seed)", seeds_str);
    log_columns(&model_names, &voting);
    teeln!();

    // Probe data: predictions + context features fully in memory (for fold
    // gathering); ratings as f64.
    println!("Loading probe set ({})...", pr);
    let y_pr = load_ratings_f64(&pr);
    let n = y_pr.len();
    let no_clip = vec![false; p];
    let xpr = load_cols(&model_names, &model_clip, &preds, &pr, n);
    let fpr = load_cols(&voting, &no_clip, &preds, &pr, n);

    // Qual data: ratings + quiz mask in memory; predictions streamed per block.
    let y_ql: Array1<i8> = read_npy(format!("data/{fulltrain_pr}/ratings.npy"))
        .unwrap_or_else(|e| panic!("read data/{fulltrain_pr}/ratings.npy: {e}"));
    let is_test: Array1<i8> = read_npy(format!("data/{fulltrain_pr}/is_test.npy"))
        .unwrap_or_else(|e| panic!("read data/{fulltrain_pr}/is_test.npy: {e}"));
    let n_q = y_ql.len();
    let quiz_n = is_test.iter().filter(|&&t| t == 0).count();
    println!("Probe: {} rows, Qual: {} rows ({} quiz)", n, n_q, quiz_n);

    let open_readers = |names: &[String], dir: &str| -> Vec<NpyF32Reader> {
        names
            .iter()
            .map(|name| {
                let path = format!("{dir}/{name}.{fulltrain_pr}.npy");
                let r = NpyF32Reader::open(&path);
                assert_eq!(r.len, n_q, "{}: len {} != qual {}", path, r.len, n_q);
                r
            })
            .collect()
    };

    for &seed in &args.seeds {
        teeln!();
        teeln!("=== seed {} ===", seed);

        let mut idxs: Vec<usize> = (0..n).collect();
        idxs.shuffle(&mut StdRng::seed_from_u64(seed));
        let half = n / 2;
        let folds = [idxs[..half].to_vec(), idxs[half..].to_vec()];

        let mut yhat_pr = vec![0.0f64; n];
        let mut yhat_ql = vec![0.0f64; n_q];

        // Qual readers reused across folds (re-seeked per block).
        let mut xr = open_readers(&model_names, &preds);
        let mut fr = open_readers(&voting, &preds);

        for (k, &(tr, te)) in [(0usize, 1usize), (1, 0)].iter().enumerate() {
            let train = &folds[tr];
            let test = &folds[te];
            teeln!("  fold {}/2: train {} predict {}", k + 1, train.len(), test.len());

            let (amat, b) = build_gram_fold(train, &xpr, &fpr, &y_pr, m, p, params.lambda);
            let w = solve_fold(amat, b);

            let p_te = predict_rows(test, &xpr, &fpr, &w, m, p, true);
            teeln!("    fold RMSE {:.5}", rmse_sel(&p_te, &y_pr, test));
            for (ii, &row) in test.iter().enumerate() {
                yhat_pr[row] = p_te[ii];
            }

            predict_qual_acc(&mut xr, &mut fr, &model_clip, &w, m, p, n_q, &mut yhat_ql);
        }

        for v in yhat_ql.iter_mut() {
            *v *= 0.5;
        }

        // Metrics
        let probe_sse: f64 = yhat_pr.iter().zip(&y_pr).map(|(&h, &y)| (h - y) * (h - y)).sum();
        let probe_rmse = (probe_sse / n as f64).sqrt();
        let mut quiz_sse = 0.0;
        for k in 0..n_q {
            if is_test[k] == 0 {
                let e = yhat_ql[k] - y_ql[k] as f64;
                quiz_sse += e * e;
            }
        }
        let quiz_rmse = (quiz_sse / quiz_n as f64).sqrt();
        teeln!(" ProbeRMSE: {:.5}", probe_rmse);
        teeln!("  QuizRMSE: {:.5}", quiz_rmse);

        let pr_path = format!("{preds}/{}-s{seed}.{pr}.npy", args.name);
        let ql_path = format!("{preds}/{}-s{seed}.{fulltrain_pr}.npy", args.name);
        save_preds(&pr_path, &Array1::from_iter(yhat_pr.iter().map(|&v| v as f32)));
        save_preds(&ql_path, &Array1::from_iter(yhat_ql.iter().map(|&v| v as f32)));
        teeln!("Saved {} / {}", pr_path, ql_path);
    }

    close_log();
    ExitCode::SUCCESS
}
