//! FWLS + forward feature-selection experiment (standalone; does not touch the
//! `run` pipeline). Takes a set of M models and F features, forms the union
//! U = M ∪ F (any column usable as both a model and a feature factor), and
//! considers every pairwise product u_i·u_j as a candidate FWLS interaction
//! column. The symmetric interaction Gram over all U×U pairs is precomputed once
//! on the probe(x) set (accumulated in f64, blocked, stored as the packed upper
//! triangle in f32). Forward selection then greedily adds interaction columns by
//! the in-sample (Gram-recovered) probe RMSE — or, with `--cv-folds K`, by the
//! K-fold CV RMSE recovered from K per-fold Grams. `--force-models K` narrows the
//! first K steps to plain model terms; each prefix is scored on probe and quiz.

extern crate blas;
extern crate blas_src;

use nalgebra::{DMatrix, DVector};
use ndarray::{s, Array1, Array2, ArrayView1};
use ndarray_npy::{read_npy, write_npy};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use netflix_prize::blend::{
    flatten_groups, load_models_toml, permuted_folds, resolve_voting, select_groups,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::process::ExitCode;

const IN_CLIP_MIN: f64 = 0.0;
const IN_CLIP_MAX: f64 = 6.0;
const OUT_CLIP_MIN: f64 = 1.05;
const OUT_CLIP_MAX: f64 = 4.95;
const PIPELINE_OLD: &str = "pipeline-old.toml";
const PIPELINE_NEW: &str = "pipeline-new.toml";
const MODELS_OLD: &str = "models-old.toml";
const MODELS_NEW: &str = "models-new.toml";
const VOTING_OLD: &str = "voting-old.toml";
const VOTING_NEW: &str = "voting-new.toml";

// Row-block for the Gram / eval streaming passes, and column-panel width for the
// blocked outer product (bounds the transient f64 working set).
const ROW_BLOCK: usize = 1024;
const COL_PANEL: usize = 1024;

// ---------------------------------------------------------------------------
// Partial .npy reader for 1-D float32 arrays (shared shape with ridge/fwls)
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

/// Forward-selection granularity.
#[derive(Clone, Copy, PartialEq)]
enum SelectMode {
    /// Add one interaction column u_i·u_j per step (default).
    Pairs,
    /// Add one U column per step, either as a model or as a feature — pulling in
    /// its whole block of interactions with the other set.
    Column,
}

struct Args {
    pipeline: String,
    split_hint: &'static str, // "old"/"new"/"?" — picks default model/voting TOMLs
    models_toml: Option<String>,
    model_groups: Vec<String>,
    model_manual: Vec<String>,
    features_toml: Option<String>,
    feature_groups: Vec<String>,
    feature_manual: Vec<String>,
    exclude: Vec<String>,
    lambda: f64,
    max_features: Option<usize>,
    in_clip_min: f64,
    in_clip_max: f64,
    out_clip_min: f64,
    out_clip_max: f64,
    probe_frac: f64,
    seed: u64,
    cv_folds: usize,
    cv_seed: Option<u64>,
    cv_patience: Option<usize>,
    select: SelectMode,
    force_models: usize,
    save_state: Option<String>,
    load_state: Option<String>,
    no_score: bool,
}

fn print_help() {
    println!("Usage: fwls-fs [-o|-n|-p FILE] [models sel] [feature sel] [--lambda L] [--max-features K]");
    println!();
    println!("  Split (default {}):", PIPELINE_OLD);
    println!("    -o | -n                  pipeline-old / pipeline-new");
    println!("    -p FILE                  explicit pipeline TOML");
    println!();
    println!("  Models M (clipped prediction factors):");
    println!("    -t FILE                  models TOML (default: models-<split>.toml)");
    println!("    -g G,G,...               model groups (default: all); a group may be sliced");
    println!("                             Python-style over its expanded model list, e.g.");
    println!("                             'integrated[:5]', 'integrated[5:]', 'integrated[3:7]'");
    println!("    -m NAME                  add a single model (repeatable)");
    println!();
    println!("  Features F (unclipped context factors):");
    println!("    -T FILE                  features/voting TOML (default: voting-<split>.toml)");
    println!("    -G G,G,...               feature groups (default: all); sliceable like -g");
    println!("    -f NAME                  add a single feature (repeatable)");
    println!();
    println!("    -x NAME                  drop a model/feature by name (repeatable)");
    println!("    --lambda L               ridge λ, as a fraction of a column's centered");
    println!("                             variance (the bias is projected out first), so it");
    println!("                             shrinks every column type alike: λ=0.01 ≈ 1%");
    println!("                             shrinkage (default 0.01)");
    println!("    --select pairs|column    forward-selection granularity (default pairs):");
    println!("                             pairs = one interaction u_i·u_j per step; column =");
    println!("                             one U column per step, as a model or a feature");
    println!("    --max-features K         stop after K selected steps (interactions or columns)");
    println!("    --force-models K         restrict the first K steps to plain model terms");
    println!("                             (model × <const>) — still chosen greedily, just from");
    println!("                             a narrowed candidate set, so the blend starts from");
    println!("                             real predictors (default 0)");
    println!();
    println!("  Cross-validated selection (K× Gram memory, no extra data pass):");
    println!("    --cv-folds K             select by K-fold CV RMSE over the probe subset instead");
    println!("                             of in-sample RMSE (default 1 = in-sample)");
    println!("    --cv-seed S              RNG seed for the fold assignment (default: --seed)");
    println!("    --cv-patience P          stop after P steps without a CV improvement");
    println!();
    println!("  State save/resume (skip the expensive Gram recomputation):");
    println!("    --save-state DIR         after the Gram pass, save {{gram,b}}.npy + meta.toml to DIR");
    println!("    --load-state DIR         load the saved Gram state; run selection/scoring with the");
    println!("                             current experiment args (lambda/select/max-features/…).");
    println!("                             The Gram-defining args (split, models/features, probe-frac,");
    println!("                             seed, in-clip) come from the saved state.");
    println!("    --no-score               skip probe/qual scoring (in-sample RMSE only; fast sweeps)");
    println!("    --probe-frac F           use only a fraction F of probe rows for the Gram +");
    println!("                             probe scoring (0<F≤1, default 1); qual stays full");
    println!("    --seed S                 RNG seed for --probe-frac subsampling (default 0)");
    println!("    --in-clip MIN,MAX        input clip for model columns (default {IN_CLIP_MIN},{IN_CLIP_MAX})");
    println!("    --out-clip MIN,MAX       output clip before RMSE (default {OUT_CLIP_MIN},{OUT_CLIP_MAX})");
    println!("    -h, --help               show this help");
    println!();
    println!("  U = M ∪ F ∪ {{const}}; candidate columns = every product u_i·u_j (i≤j).");
    println!("  The bias (const×const) is always in; forward selection then adds columns by");
    println!("  in-sample (or K-fold CV) probe RMSE, and scores each prefix on probe & qual.");
}

fn need(argv: &[String], i: usize) -> String {
    if i + 1 >= argv.len() {
        eprintln!("error: '{}' requires an argument", argv[i]);
        std::process::exit(2);
    }
    argv[i + 1].clone()
}

fn parse_clip(s: &str, flag: &str) -> (f64, f64) {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        eprintln!("error: '{}' expects MIN,MAX (got '{}')", flag, s);
        std::process::exit(2);
    }
    let lo = parts[0].trim().parse::<f64>().unwrap_or_else(|_| { eprintln!("bad {} MIN", flag); std::process::exit(2); });
    let hi = parts[1].trim().parse::<f64>().unwrap_or_else(|_| { eprintln!("bad {} MAX", flag); std::process::exit(2); });
    (lo, hi)
}

fn parse_args() -> Args {
    let mut a = Args {
        pipeline: PIPELINE_OLD.to_string(),
        split_hint: "old",
        models_toml: None,
        model_groups: Vec::new(),
        model_manual: Vec::new(),
        features_toml: None,
        feature_groups: Vec::new(),
        feature_manual: Vec::new(),
        exclude: Vec::new(),
        lambda: 0.01,
        max_features: None,
        in_clip_min: IN_CLIP_MIN,
        in_clip_max: IN_CLIP_MAX,
        out_clip_min: OUT_CLIP_MIN,
        out_clip_max: OUT_CLIP_MAX,
        probe_frac: 1.0,
        seed: 0,
        cv_folds: 1,
        cv_seed: None,
        cv_patience: None,
        select: SelectMode::Pairs,
        force_models: 0,
        save_state: None,
        load_state: None,
        no_score: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            "-o" | "--old" => { a.pipeline = PIPELINE_OLD.to_string(); a.split_hint = "old"; i += 1; }
            "-n" | "--new" => { a.pipeline = PIPELINE_NEW.to_string(); a.split_hint = "new"; i += 1; }
            "-p" | "--pipeline" => { a.pipeline = need(&argv, i); a.split_hint = "?"; i += 2; }
            "-t" | "--models" => { a.models_toml = Some(need(&argv, i)); i += 2; }
            "-g" | "--groups" => { for t in need(&argv, i).split(',') { a.model_groups.push(t.trim().to_string()); } i += 2; }
            "-m" | "--model" => { a.model_manual.push(need(&argv, i)); i += 2; }
            "-T" | "--features-toml" => { a.features_toml = Some(need(&argv, i)); i += 2; }
            "-G" | "--feature-groups" => { for t in need(&argv, i).split(',') { a.feature_groups.push(t.trim().to_string()); } i += 2; }
            "-f" | "--feature" => { a.feature_manual.push(need(&argv, i)); i += 2; }
            "-x" | "--exclude" => { a.exclude.push(need(&argv, i)); i += 2; }
            "--lambda" => { a.lambda = need(&argv, i).parse().expect("bad --lambda"); i += 2; }
            "--max-features" => { a.max_features = Some(need(&argv, i).parse().expect("bad --max-features")); i += 2; }
            "--force-models" => { a.force_models = need(&argv, i).parse().expect("bad --force-models"); i += 2; }
            "--cv-folds" => { a.cv_folds = need(&argv, i).parse().expect("bad --cv-folds"); i += 2; }
            "--cv-seed" => { a.cv_seed = Some(need(&argv, i).parse().expect("bad --cv-seed")); i += 2; }
            "--cv-patience" => { a.cv_patience = Some(need(&argv, i).parse().expect("bad --cv-patience")); i += 2; }
            "--save-state" => { a.save_state = Some(need(&argv, i)); i += 2; }
            "--load-state" => { a.load_state = Some(need(&argv, i)); i += 2; }
            "--no-score" => { a.no_score = true; i += 1; }
            "--select" => {
                a.select = match need(&argv, i).as_str() {
                    "pairs" => SelectMode::Pairs,
                    "column" | "columns" => SelectMode::Column,
                    s => { eprintln!("error: --select expects 'pairs' or 'column' (got '{}')", s); std::process::exit(2); }
                };
                i += 2;
            }
            "--probe-frac" => { a.probe_frac = need(&argv, i).parse().expect("bad --probe-frac"); i += 2; }
            "--seed" => { a.seed = need(&argv, i).parse().expect("bad --seed"); i += 2; }
            "--in-clip" => { let (lo, hi) = parse_clip(&need(&argv, i), "--in-clip"); a.in_clip_min = lo; a.in_clip_max = hi; i += 2; }
            "--out-clip" => { let (lo, hi) = parse_clip(&need(&argv, i), "--out-clip"); a.out_clip_min = lo; a.out_clip_max = hi; i += 2; }
            s => { eprintln!("error: unknown arg '{}'", s); print_help(); std::process::exit(2); }
        }
    }
    if !(0.0 < a.probe_frac && a.probe_frac <= 1.0) {
        eprintln!("error: --probe-frac must be in (0, 1] (got {})", a.probe_frac);
        std::process::exit(2);
    }
    if a.cv_folds < 1 {
        eprintln!("error: --cv-folds must be ≥ 1 (got {})", a.cv_folds);
        std::process::exit(2);
    }
    a
}

fn load_pipeline_split(path: &str) -> HashMap<String, String> {
    #[derive(serde::Deserialize)]
    struct P { #[serde(default)] split: HashMap<String, String> }
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let p: P = toml::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {}", path, e));
    p.split
}

// ---------------------------------------------------------------------------
// Union of models + features
// ---------------------------------------------------------------------------

/// A column of the union U: its name, clip flag (models clipped, features not),
/// and provenance (came from the model set / the feature set / both).
struct UCol {
    name: String,
    clip: bool,
    is_model: bool,
    is_feature: bool,
}

/// Build U = M ∪ F, with a synthetic all-ones `const` feature appended last.
/// Models come first (their count is N — the forced "first N as models"), then
/// features not already present. Returns the columns and N.
fn build_union(args: &Args) -> (Vec<UCol>, usize) {
    // Models: clip flag honours the '>' no-clip prefix via flatten_groups.
    let models_toml = args.models_toml.clone().unwrap_or_else(|| match args.split_hint {
        "new" => MODELS_NEW.to_string(),
        _ => MODELS_OLD.to_string(),
    });
    let mut mgroups = select_groups(&load_models_toml(&models_toml), &args.model_groups);
    if !args.model_manual.is_empty() {
        mgroups.insert("manual".to_string(), args.model_manual.clone());
    }
    let mflat = flatten_groups(&mgroups, &args.exclude);

    // Features: unclipped context columns (predictors usable here too).
    let features_toml = args.features_toml.clone().unwrap_or_else(|| match args.split_hint {
        "new" => VOTING_NEW.to_string(),
        _ => VOTING_OLD.to_string(),
    });
    let mut fnames = resolve_voting(&features_toml, &args.feature_groups);
    fnames.extend(args.feature_manual.iter().cloned());

    let mut cols: Vec<UCol> = Vec::new();
    let mut idx: HashMap<String, usize> = HashMap::new();
    for (nm, cl) in mflat.names.iter().zip(mflat.clip.iter()) {
        idx.insert(nm.clone(), cols.len());
        cols.push(UCol { name: nm.clone(), clip: *cl, is_model: true, is_feature: false });
    }
    let n_models = cols.len();
    for f in &fnames {
        // '>' no-clip prefix on a feature is meaningless (features never clip);
        // strip it so names match the on-disk files.
        let name = f.trim_start_matches('>').to_string();
        match idx.get(&name) {
            Some(&c) => { cols[c].is_feature = true; } // already a model → both roles
            None => {
                idx.insert(name.clone(), cols.len());
                cols.push(UCol { name, clip: false, is_model: false, is_feature: true });
            }
        }
    }
    // Synthetic constant feature (all ones), appended last.
    cols.push(UCol { name: "<const>".to_string(), clip: false, is_model: false, is_feature: true });
    (cols, n_models)
}

// ---------------------------------------------------------------------------
// Column loading (clip applied per provenance)
// ---------------------------------------------------------------------------

fn clip_vec(v: &mut [f32], clip: bool, lo: f32, hi: f32) {
    if clip {
        for x in v.iter_mut() { *x = x.clamp(lo, hi); }
    }
}

/// Load every U column over `dataset` fully into memory (const → all-ones).
fn load_u_columns(cols: &[UCol], preds_dir: &str, dataset: &str, n: usize, lo: f32, hi: f32) -> Vec<Vec<f32>> {
    cols.iter().map(|c| {
        if c.name == "<const>" {
            vec![1.0f32; n]
        } else {
            let path = format!("{}/{}.{}.npy", preds_dir, c.name, dataset);
            let arr: Array1<f32> = read_npy(&path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
            assert_eq!(arr.len(), n, "{}: length {} != {}", path, arr.len(), n);
            let mut v = arr.to_vec();
            clip_vec(&mut v, c.clip, lo, hi);
            v
        }
    }).collect()
}

// ---------------------------------------------------------------------------
// Packed upper-triangle indexing for the interaction Gram
// ---------------------------------------------------------------------------

/// Interaction column c ↔ unordered pair (i, j), i ≤ j, over U (incl const).
/// Packed column-major upper triangle: colindex(i, j) = i + j*(j+1)/2.
#[inline]
fn colindex(i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    i + j * (j + 1) / 2
}

// ---------------------------------------------------------------------------
// Ridge subsystem solve (shared with the forward-selection scoring)
// ---------------------------------------------------------------------------

/// Slice the packed Gram `g` / rhs `b` into the (k+? ) subsystem for `cols`.
/// `cols` already includes the bias column, so no extra bias row is added.
fn build_subsystem(g: &[f32], b: &[f64], cols: &[usize]) -> (DMatrix<f64>, DVector<f64>) {
    let k = cols.len();
    let mut a = DMatrix::<f64>::zeros(k, k);
    let mut rhs = DVector::<f64>::zeros(k);
    for (ii, &ci) in cols.iter().enumerate() {
        rhs[ii] = b[ci];
        for (jj, &cj) in cols.iter().enumerate() {
            a[(ii, jj)] = g[colindex(ci, cj)] as f64;
        }
    }
    (a, rhs)
}

/// Ridge-solve the `cols` subsystem with the bias projected out first: every
/// other column is centered against it (`z̃_c = z_c − mean(z_c)·1`). That makes
/// the normal equations block-diagonal — the intercept is exactly ȳ — and leaves
/// a centered block that is scaled to unit diagonal (`s_c = √(centered SS)`)
/// before λ is added. λ is therefore a scale-free fraction of the variance each
/// column can actually explain, uniform across column types.
///
/// Normalizing by the *uncentered* `√A[c,c]` instead would put λ on the raw
/// second moment, of which a prediction column's usable (centered) share is only
/// ~4% while a product column's is 10–25% — the same λ then shrank the two by
/// 27% vs 9% and biased forward selection toward products. Rescaling alone would
/// not fix it either: it leaves the bias cross-term at `mean/std` ≈ 5 against a
/// unit diagonal, which is not positive definite. The centering is what makes λ
/// comparable, and it improves the conditioning of the f32-precision Gram.
///
/// Weights come back in the original basis, with `w_bias = ȳ − Σ w_c·mean(z_c)`.
/// A zero-variance column (a constant feature's plain term) is decoupled from the
/// block and gets weight 0. `None` means the block is singular — two exactly
/// collinear columns, which λ = 0 cannot separate; callers drop the candidate.
fn solve_subsystem(a: &DMatrix<f64>, b: &DVector<f64>, cols: &[usize], bias_col: usize, lambda: f64) -> Option<DVector<f64>> {
    let k = cols.len();
    let bi = cols.iter().position(|&c| c == bias_col)
        .expect("subsystem must contain the bias column");
    let n = a[(bi, bi)]; // Σ1² over the fitted rows
    let mean: Vec<f64> = (0..k).map(|i| a[(i, bi)] / n).collect();
    let free: Vec<usize> = (0..k).filter(|&i| i != bi).collect();
    let p = free.len();

    // Centered block, normalized to unit diagonal, plus λ.
    let css: Vec<f64> = free.iter()
        .map(|&i| (a[(i, i)] - n * mean[i] * mean[i]).max(0.0))
        .collect();
    let s: Vec<f64> = css.iter().map(|&v| if v > 0.0 { v.sqrt() } else { 1.0 }).collect();
    let mut cn = DMatrix::<f64>::zeros(p, p);
    let mut rhs = DVector::<f64>::zeros(p);
    for ii in 0..p {
        if css[ii] <= 0.0 {
            cn[(ii, ii)] = 1.0; // constant column: isolate it, rhs stays 0
            continue;
        }
        let i = free[ii];
        rhs[ii] = (b[i] - mean[i] * b[bi]) / s[ii];
        for jj in 0..p {
            if css[jj] <= 0.0 { continue; }
            let j = free[jj];
            cn[(ii, jj)] = (a[(i, j)] - n * mean[i] * mean[j]) / (s[ii] * s[jj]);
        }
        cn[(ii, ii)] += lambda;
    }

    // Solve and map back: the free weights unscale, the bias absorbs the means.
    let mut w = DVector::<f64>::zeros(k);
    let mut bias = b[bi] / n; // ȳ
    if p > 0 {
        let wt = cn.cholesky()?.solve(&rhs);
        for (ii, &i) in free.iter().enumerate() {
            w[i] = wt[ii] / s[ii];
            bias -= w[i] * mean[i];
        }
    }
    w[bi] = bias;
    Some(w)
}

/// Sum of the per-fold subsystems over `cols` — the full-P system.
fn total_subsystem(g: &[Vec<f32>], b: &[Vec<f64>], cols: &[usize]) -> (DMatrix<f64>, DVector<f64>) {
    let (mut a, mut rhs) = build_subsystem(&g[0], &b[0], cols);
    for k in 1..g.len() {
        let (ak, bk) = build_subsystem(&g[k], &b[k], cols);
        a += ak;
        rhs += bk;
    }
    (a, rhs)
}

/// SSE of the fit `w` against an (unregularized) Gram system:
/// SSE = yᵀy − 2 wᵀ b + wᵀ A w.
fn sse_of(w: &DVector<f64>, a: &DMatrix<f64>, b: &DVector<f64>, yty: f64) -> f64 {
    yty - 2.0 * w.dot(b) + w.dot(&(a * w))
}

/// (in-sample RMSE over all of P, K-fold CV RMSE) for `cols`, recovered from the
/// per-fold Grams alone. Fold k trains on `A − A_k` and is scored on its own
/// held-out `A_k` — the subtraction is safe because the folds are a random split,
/// so `A_k ≈ A/K` and there is no cancellation. With a single fold the CV value
/// degenerates to the in-sample one and only one solve is done. A singular
/// subsystem scores as `+∞`, so forward selection just skips that candidate.
fn eval_subset(
    g: &[Vec<f32>], b: &[Vec<f64>], yty: &[f64], n: usize,
    cols: &[usize], bias_col: usize, lambda: f64,
) -> (f64, f64) {
    const SINGULAR: (f64, f64) = (f64::INFINITY, f64::INFINITY);
    let (a_tot, b_tot) = total_subsystem(g, b, cols);
    let Some(w) = solve_subsystem(&a_tot, &b_tot, cols, bias_col, lambda) else { return SINGULAR };
    let in_rmse = (sse_of(&w, &a_tot, &b_tot, yty.iter().sum()) / n as f64).sqrt();
    if g.len() < 2 {
        return (in_rmse, in_rmse);
    }
    let mut sse_cv = 0.0f64;
    for k in 0..g.len() {
        let (ak, bk) = build_subsystem(&g[k], &b[k], cols);
        let Some(wk) = solve_subsystem(&(&a_tot - &ak), &(&b_tot - &bk), cols, bias_col, lambda)
            else { return SINGULAR };
        sse_cv += sse_of(&wk, &ak, &bk, yty[k]);
    }
    (in_rmse, (sse_cv / n as f64).sqrt())
}

// ---------------------------------------------------------------------------
// Clipped RMSE evaluation for a fit, over an in-memory / streamed column set
// ---------------------------------------------------------------------------

/// Compute the clipped SSE + count of a fit over rows `[0,n)`, forming each
/// selected interaction value `u_i·u_j` from the per-column blocks `get(col)`.
/// When `mask` is given, only rows with `mask==0` are counted (quiz subset).
#[allow(clippy::too_many_arguments)]
fn fit_sse(
    pairs: &[(u32, u32)],
    fit_cols: &[usize],
    w: &[f64],
    bias_col: usize,
    n: usize,
    y: &[i8],
    mask: Option<&[i8]>,
    out_lo: f64,
    out_hi: f64,
    col_at: &mut dyn FnMut(usize, usize) -> Vec<Vec<f32>>, // (start, blen) → per-U-column block
) -> (f64, usize) {
    let mut sse = 0.0f64;
    let mut cnt = 0usize;
    let mut start = 0;
    while start < n {
        let bl = (n - start).min(ROW_BLOCK);
        let ublk = col_at(start, bl); // ublk[u][k]
        for k in 0..bl {
            let row = start + k;
            if let Some(m) = mask { if m[row] != 0 { continue; } }
            let mut yhat = 0.0f64;
            for (wi, &c) in fit_cols.iter().enumerate() {
                let val = if c == bias_col {
                    1.0
                } else {
                    let (i, j) = pairs[c];
                    (ublk[i as usize][k] as f64) * (ublk[j as usize][k] as f64)
                };
                yhat += w[wi] * val;
            }
            let yh = yhat.clamp(out_lo, out_hi);
            let e = yh - y[row] as f64;
            sse += e * e;
            cnt += 1;
        }
        start += bl;
    }
    (sse, cnt)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// The reusable Gram state: everything a resumed experiment needs (any λ /
/// --select / --max-features / --force-models), minus the raw U columns (cheaply
/// reloaded for scoring). `cst = nu - 1`; `pairs` is rebuilt from `nu`. The Gram
/// is kept per CV fold (one entry when `--cv-folds 1`); the full-P system is the
/// elementwise sum, so both criteria come out of the same state.
struct State {
    ucols: Vec<UCol>,
    n_models: usize,
    nu: usize,
    dpair: usize,
    g: Vec<Vec<f32>>,
    bvec: Vec<Vec<f64>>,
    yty: Vec<f64>,
    n_fold: Vec<usize>,
    cv_seed: u64,
    n_used: usize,
    n_pr: usize,
    sel_rows: Vec<usize>,
    preds: String,
    pr: String,
    qual: String,
    split_name: String,
    pipeline: String,
    in_clip_min: f64,
    in_clip_max: f64,
    probe_frac: f64,
    seed: u64,
}

/// v1 = single Gram in gram.npy/b.npy; v2 = per-fold gram.k{i}.npy/b.k{i}.npy.
const STATE_VERSION: u32 = 2;

#[derive(serde::Serialize, serde::Deserialize)]
struct MetaCol { name: String, clip: bool, is_model: bool, is_feature: bool }

/// The scalar/metadata sidecar written alongside gram.npy / b.npy.
#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    version: u32,
    pipeline: String,
    split_name: String,
    preds: String,
    pr: String,
    qual: String,
    in_clip_min: f64,
    in_clip_max: f64,
    probe_frac: f64,
    seed: u64,
    n_used: usize,
    n_pr: usize,
    nu: usize,
    n_models: usize,
    dpair: usize,
    yty: f64, // total over all folds (the only yᵀy a v1 state has)
    #[serde(default = "one")]
    k_folds: usize,
    #[serde(default)]
    cv_seed: u64,
    #[serde(default)]
    fold_yty: Vec<f64>, // v2 only; empty in a v1 state
    #[serde(default)]
    fold_n: Vec<usize>, // v2 only; empty in a v1 state
    columns: Vec<MetaCol>,
}

fn one() -> usize { 1 }

/// Deterministic (seeded) probe subsample, sorted for locality.
fn build_sel_rows(n_pr: usize, n_used: usize, seed: u64) -> Vec<usize> {
    if n_used == n_pr {
        (0..n_pr).collect()
    } else {
        let mut idx: Vec<usize> = (0..n_pr).collect();
        idx.shuffle(&mut StdRng::seed_from_u64(seed));
        idx.truncate(n_used);
        idx.sort_unstable();
        idx
    }
}

/// Split the (already subsampled) probe rows into `k` CV folds. Row-level random
/// assignment, seeded; each fold's row list is sorted for locality. `k == 1`
/// keeps the whole set as a single fold, i.e. the in-sample criterion.
fn build_folds(sel_rows: &[usize], k: usize, seed: u64) -> Vec<Vec<usize>> {
    if k < 2 {
        return vec![sel_rows.to_vec()];
    }
    let mut folds = permuted_folds(sel_rows.len(), k, seed);
    for f in folds.iter_mut() {
        for x in f.iter_mut() {
            *x = sel_rows[*x];
        }
        f.sort_unstable();
    }
    folds
}

#[allow(clippy::too_many_arguments)]
fn print_header(args: &Args, ucols: &[UCol], nu: usize, n_models: usize, dpair: usize,
                pipeline: &str, split_name: &str, in_clip_min: f64, in_clip_max: f64,
                n_pr: usize, n_used: usize, seed: u64, k_folds: usize, cv_seed: u64) {
    println!("Pipeline:  {} (split = {})", pipeline, split_name);
    let both = ucols.iter().filter(|c| c.is_model && c.is_feature).count();
    println!("Union U:   {} columns ({} models + {} features + const; {} of the models are also features)",
        nu, n_models, nu - 1 - n_models, both);
    println!("Interact.: {} candidate columns (all u_i·u_j pairs)", dpair);
    println!("Lambda λ:  {}", args.lambda);
    println!("In-clip:   [{}, {}]   Out-clip: [{}, {}]",
        in_clip_min, in_clip_max, args.out_clip_min, args.out_clip_max);
    let gram_len = dpair * (dpair + 1) / 2;
    let gram_mib = (gram_len as f64 * 4.0) / (1024.0 * 1024.0);
    if k_folds < 2 {
        println!("Gram:      packed upper-tri f32, {:.1} MiB", gram_mib);
        println!("Criterion: in-sample RMSE (no CV)");
    } else {
        println!("Gram:      packed upper-tri f32, {:.1} MiB × {} folds = {:.1} MiB",
            gram_mib, k_folds, gram_mib * k_folds as f64);
        println!("Criterion: {}-fold CV RMSE (fold seed {}{})", k_folds, cv_seed,
            args.cv_patience.map(|p| format!(", patience {p}")).unwrap_or_default());
    }
    if n_used == n_pr {
        println!("Probe set: {} ratings", n_pr);
    } else {
        println!("Probe set: {} / {} rows ({:.1}%, seed {})",
            n_used, n_pr, 100.0 * n_used as f64 / n_pr as f64, seed);
    }
}

/// Compute the Gram state from scratch (union → probe columns → Gram). Returns
/// the state plus the in-memory probe columns + ratings (reused for scoring).
fn compute_state(args: &Args) -> (State, Vec<Vec<f32>>, Array1<i8>) {
    let split = load_pipeline_split(&args.pipeline);
    let pr = split.get("pr").expect("[split].pr missing").clone();
    let qual = split.get("fulltrain_pr").expect("[split].fulltrain_pr missing").clone();
    let preds = split.get("preds").expect("[split].preds missing").clone();
    let split_name = split.get("name").cloned().unwrap_or_else(|| "?".to_string());

    let (ucols, n_models) = build_union(args);
    let nu = ucols.len();
    let dpair = nu * (nu + 1) / 2;

    let y_pr: Array1<i8> = read_npy(format!("data/{}/ratings.npy", pr))
        .unwrap_or_else(|e| panic!("read data/{}/ratings.npy: {}", pr, e));
    let n_pr = y_pr.len();
    let u_pr = load_u_columns(&ucols, &preds, &pr, n_pr, args.in_clip_min as f32, args.in_clip_max as f32);

    let n_used = if args.probe_frac >= 1.0 { n_pr }
                 else { ((args.probe_frac * n_pr as f64).round() as usize).clamp(1, n_pr) };
    let sel_rows = build_sel_rows(n_pr, n_used, args.seed);
    let cv_seed = args.cv_seed.unwrap_or(args.seed);
    let k_folds = args.cv_folds.min(n_used);
    let folds = build_folds(&sel_rows, k_folds, cv_seed);

    print_header(args, &ucols, nu, n_models, dpair, &args.pipeline, &split_name,
        args.in_clip_min, args.in_clip_max, n_pr, n_used, args.seed, k_folds, cv_seed);

    // Interaction pairs (packed order): c = colindex(i, j), i ≤ j.
    let mut pairs = vec![(0u32, 0u32); dpair];
    for j in 0..nu {
        for i in 0..=j { pairs[colindex(i, j)] = (i as u32, j as u32); }
    }

    // --- Gram over probe: A = ZᵀZ (packed f32 upper-tri), b = Zᵀy, yᵀy — one set
    // per CV fold. Every row is still touched exactly once, so the (dominant)
    // cost of this pass is independent of K; only the memory scales. ---
    println!("Building interaction Gram over {} pairs × {} ratings ({} fold(s))...",
        dpair, n_used, k_folds);
    let gram_len = dpair * (dpair + 1) / 2;
    let mut g = vec![vec![0.0f32; gram_len]; k_folds];
    let mut bvec = vec![vec![0.0f64; dpair]; k_folds];
    let mut yty = vec![0.0f64; k_folds];
    let n_fold: Vec<usize> = folds.iter().map(|f| f.len()).collect();
    let n_blocks: usize = n_fold.iter().map(|n| n.div_ceil(ROW_BLOCK)).sum();
    let mut bidx = 0;
    for (kf, rows) in folds.iter().enumerate() {
        let gk = &mut g[kf];
        let bk = &mut bvec[kf];
        let mut start = 0;
        while start < rows.len() {
            let bl = (rows.len() - start).min(ROW_BLOCK);
            let mut z = Array2::<f64>::zeros((bl, dpair));
            {
                let zs = z.as_slice_mut().unwrap();
                zs.par_chunks_mut(dpair).enumerate().for_each(|(k, zrow)| {
                    let row = rows[start + k];
                    for (c, &(i, j)) in pairs.iter().enumerate() {
                        zrow[c] = (u_pr[i as usize][row] as f64) * (u_pr[j as usize][row] as f64);
                    }
                });
            }
            let yb: Array1<f64> = Array1::from_iter((0..bl).map(|k| y_pr[rows[start + k]] as f64));
            yty[kf] += yb.dot(&yb);
            let bt = z.t().dot(&yb);
            for c in 0..dpair { bk[c] += bt[c]; }
            let mut c0 = 0;
            while c0 < dpair {
                let c1 = (c0 + COL_PANEL).min(dpair);
                let panel = z.slice(s![.., c0..c1]);
                let cp = z.t().dot(&panel);
                for jj in 0..(c1 - c0) {
                    let jcol = c0 + jj;
                    let base = jcol * (jcol + 1) / 2;
                    for i in 0..=jcol {
                        gk[base + i] = (gk[base + i] as f64 + cp[[i, jj]]) as f32;
                    }
                }
                c0 = c1;
            }
            start += bl;
            bidx += 1;
            eprint!("\r  gram block {}/{}", bidx, n_blocks);
        }
    }
    eprintln!();

    let st = State {
        ucols, n_models, nu, dpair, g, bvec, yty, n_fold, cv_seed, n_used, n_pr, sel_rows,
        preds, pr, qual, split_name, pipeline: args.pipeline.clone(),
        in_clip_min: args.in_clip_min, in_clip_max: args.in_clip_max,
        probe_frac: args.probe_frac, seed: args.seed,
    };
    (st, u_pr, y_pr)
}

/// Load a previously saved Gram state (gram.npy + b.npy + meta.toml).
fn load_state(dir: &str, args: &Args) -> State {
    let meta_s = std::fs::read_to_string(format!("{dir}/meta.toml"))
        .unwrap_or_else(|e| panic!("read {dir}/meta.toml: {e}"));
    let meta: Meta = toml::from_str(&meta_s).unwrap_or_else(|e| panic!("parse {dir}/meta.toml: {e}"));
    assert!(meta.version <= STATE_VERSION, "state version {} > {}", meta.version, STATE_VERSION);
    let gram_len = meta.dpair * (meta.dpair + 1) / 2;

    // v1 states hold a single Gram in gram.npy/b.npy; v2 splits it per CV fold.
    let (g, bvec, yty, n_fold) = if meta.version < 2 {
        let g_arr: Array1<f32> = read_npy(format!("{dir}/gram.npy"))
            .unwrap_or_else(|e| panic!("read {dir}/gram.npy: {e}"));
        let b_arr: Array1<f64> = read_npy(format!("{dir}/b.npy"))
            .unwrap_or_else(|e| panic!("read {dir}/b.npy: {e}"));
        assert_eq!(g_arr.len(), gram_len, "gram.npy length {} != dpair*(dpair+1)/2 {}", g_arr.len(), gram_len);
        assert_eq!(b_arr.len(), meta.dpair, "b.npy length {} != dpair {}", b_arr.len(), meta.dpair);
        (vec![g_arr.to_vec()], vec![b_arr.to_vec()], vec![meta.yty], vec![meta.n_used])
    } else {
        assert_eq!(meta.fold_yty.len(), meta.k_folds, "fold_yty length != k_folds");
        assert_eq!(meta.fold_n.len(), meta.k_folds, "fold_n length != k_folds");
        let mut g = Vec::with_capacity(meta.k_folds);
        let mut bv = Vec::with_capacity(meta.k_folds);
        for k in 0..meta.k_folds {
            let g_arr: Array1<f32> = read_npy(format!("{dir}/gram.k{k}.npy"))
                .unwrap_or_else(|e| panic!("read {dir}/gram.k{k}.npy: {e}"));
            let b_arr: Array1<f64> = read_npy(format!("{dir}/b.k{k}.npy"))
                .unwrap_or_else(|e| panic!("read {dir}/b.k{k}.npy: {e}"));
            assert_eq!(g_arr.len(), gram_len, "gram.k{}.npy length {} != {}", k, g_arr.len(), gram_len);
            assert_eq!(b_arr.len(), meta.dpair, "b.k{}.npy length {} != {}", k, b_arr.len(), meta.dpair);
            g.push(g_arr.to_vec());
            bv.push(b_arr.to_vec());
        }
        (g, bv, meta.fold_yty.clone(), meta.fold_n.clone())
    };

    let ucols: Vec<UCol> = meta.columns.iter()
        .map(|c| UCol { name: c.name.clone(), clip: c.clip, is_model: c.is_model, is_feature: c.is_feature })
        .collect();
    let sel_rows = build_sel_rows(meta.n_pr, meta.n_used, meta.seed);

    println!("Loaded state from {dir}/ (v{})", meta.version);
    if args.cv_folds != g.len() {
        println!("Note:      --cv-folds {} ignored; the saved state has {} fold(s)",
            args.cv_folds, g.len());
    }
    print_header(args, &ucols, meta.nu, meta.n_models, meta.dpair, &meta.pipeline, &meta.split_name,
        meta.in_clip_min, meta.in_clip_max, meta.n_pr, meta.n_used, meta.seed, g.len(), meta.cv_seed);

    State {
        ucols, n_models: meta.n_models, nu: meta.nu, dpair: meta.dpair,
        g, bvec, yty, n_fold, cv_seed: meta.cv_seed,
        n_used: meta.n_used, n_pr: meta.n_pr, sel_rows,
        preds: meta.preds, pr: meta.pr, qual: meta.qual, split_name: meta.split_name,
        pipeline: meta.pipeline, in_clip_min: meta.in_clip_min, in_clip_max: meta.in_clip_max,
        probe_frac: meta.probe_frac, seed: meta.seed,
    }
}

/// Write the Gram state to `dir/` as gram.k{i}.npy + b.k{i}.npy + meta.toml.
fn save_state(dir: &str, st: &State) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {dir}: {e}"));
    for (k, (gk, bk)) in st.g.iter().zip(st.bvec.iter()).enumerate() {
        write_npy(format!("{dir}/gram.k{k}.npy"), &ArrayView1::from(gk.as_slice()))
            .unwrap_or_else(|e| panic!("write {dir}/gram.k{k}.npy: {e}"));
        write_npy(format!("{dir}/b.k{k}.npy"), &ArrayView1::from(bk.as_slice()))
            .unwrap_or_else(|e| panic!("write {dir}/b.k{k}.npy: {e}"));
    }
    let meta = Meta {
        version: STATE_VERSION,
        pipeline: st.pipeline.clone(), split_name: st.split_name.clone(), preds: st.preds.clone(),
        pr: st.pr.clone(), qual: st.qual.clone(),
        in_clip_min: st.in_clip_min, in_clip_max: st.in_clip_max,
        probe_frac: st.probe_frac, seed: st.seed, n_used: st.n_used, n_pr: st.n_pr,
        nu: st.nu, n_models: st.n_models, dpair: st.dpair,
        yty: st.yty.iter().sum(),
        k_folds: st.g.len(), cv_seed: st.cv_seed,
        fold_yty: st.yty.clone(), fold_n: st.n_fold.clone(),
        columns: st.ucols.iter()
            .map(|c| MetaCol { name: c.name.clone(), clip: c.clip, is_model: c.is_model, is_feature: c.is_feature })
            .collect(),
    };
    std::fs::write(format!("{dir}/meta.toml"), toml::to_string(&meta).expect("serialize meta"))
        .unwrap_or_else(|e| panic!("write {dir}/meta.toml: {e}"));
    println!("Saved state to {dir}/ ({} × gram.k*.npy {:.1} MiB + b.k*.npy + meta.toml)",
        st.g.len(), (st.g[0].len() as f64 * 4.0) / (1024.0 * 1024.0));
}

fn main() -> ExitCode {
    let args = parse_args();

    // --- Phase A: acquire the Gram state (compute from scratch, or load) ---
    let (st, u_pr_opt, y_pr_opt): (State, Option<Vec<Vec<f32>>>, Option<Array1<i8>>) =
        if let Some(dir) = &args.load_state {
            (load_state(dir, &args), None, None)
        } else {
            let (st, u_pr, y_pr) = compute_state(&args);
            (st, Some(u_pr), Some(y_pr))
        };
    if let Some(dir) = &args.save_state {
        save_state(dir, &st);
    }

    // Own the state fields under the names the experiment stage expects.
    let State {
        ucols, n_models, nu, dpair, g, bvec, yty, n_used, n_pr, sel_rows,
        preds, pr, qual, in_clip_min, in_clip_max, ..
    } = st;
    // Checked here, not in parse_args: with --load-state the fold count comes
    // from the saved state, not from --cv-folds.
    let k_folds = g.len();
    let show_cv = k_folds > 1;
    if args.cv_patience.is_some() && !show_cv {
        eprintln!("error: --cv-patience needs ≥ 2 CV folds (in-sample RMSE is monotone)");
        return ExitCode::from(2);
    }
    let cst = nu - 1;
    let mut pairs = vec![(0u32, 0u32); dpair];
    for j in 0..nu {
        for i in 0..=j { pairs[colindex(i, j)] = (i as u32, j as u32); }
    }

    // The bias (const×const) is always in and is never a candidate; the first
    // `n_force` steps after it are chosen greedily, but restricted to plain model
    // terms (model×const) so the blend starts from real predictors.
    let n_force = args.force_models.min(n_models);
    let bias_col = colindex(cst, cst);
    let mut in_sel = vec![false; dpair];

    // --- Forward selection on the probe Gram ---
    // The greedy criterion is the CV RMSE (== the in-sample one when K == 1);
    // `best_cv`/`since_best` drive the optional --cv-patience early stop.
    let target = args.max_features.unwrap_or(dpair).max(1 + n_force);
    let mut selected: Vec<usize> = Vec::new();
    let mut steps: Vec<(String, f64, f64)> = Vec::new(); // (label, in-sample, CV)
    let mut fits: Vec<Vec<usize>> = Vec::new();
    let mut best_cv = f64::INFINITY;
    let mut best_step = 0usize;
    let mut since_best = 0usize;

    // Record a step and report whether --cv-patience says to stop.
    macro_rules! record {
        ($label:expr, $in_rmse:expr, $cv:expr) => {{
            if $cv < best_cv {
                best_cv = $cv;
                best_step = steps.len();
                since_best = 0;
            } else {
                since_best += 1;
            }
            steps.push(($label, $in_rmse, $cv));
            fits.push(selected.clone());
            matches!(args.cv_patience, Some(p) if since_best >= p)
        }};
    }

    // Interaction column c as "u_i × u_j", and the one-line step trace.
    let pair_label = |c: usize| {
        let (i, j) = pairs[c];
        format!("{} × {}", ucols[i as usize].name, ucols[j as usize].name)
    };
    // `step` is the index this row will get in the final table, so the live trace
    // and the summary can be read against each other.
    let show_step = |step: usize, label: &str, ncols: Option<usize>, in_rmse: f64, cv: f64| {
        let head = match ncols {
            None => format!("{:<50}", label),
            Some(n) => format!("{:<44} [{:>2} cols]", label, n),
        };
        if show_cv {
            eprintln!("{:>4}  {} in-sample {:.6}  cv {:.6}", step, head, in_rmse, cv);
        } else {
            eprintln!("{:>4}  {} in-sample {:.6}", step, head, in_rmse);
        }
    };
    // In column mode a forced step *is* an ordinary "add this model" step: with
    // the feature set still `{<const>}`, a model's whole interaction block is the
    // single column model×const. Label it the way the free column steps are.
    let column_mode = args.select == SelectMode::Column;

    // Step 0: the bias alone.
    selected.push(bias_col);
    in_sel[bias_col] = true;
    {
        let (in_rmse, cv) = eval_subset(&g, &bvec, &yty, n_used, &selected, bias_col, args.lambda);
        println!("Forward:   bias + {} forced plain term(s)...", n_force);
        show_step(steps.len(), "<bias>", None, in_rmse, cv);
        let _ = record!("<bias>".to_string(), in_rmse, cv);
    }

    // The forced prefix: ordinary greedy steps over a candidate set narrowed to
    // the models' plain terms. Patience counts free steps only, so `since_best`
    // is reset once the prefix is in.
    let mut forced_models: Vec<usize> = Vec::with_capacity(n_force);
    for _ in 0..n_force {
        let cands: Vec<usize> = (0..n_models)
            .map(|m| colindex(m, cst))
            .filter(|c| !in_sel[*c])
            .collect();
        if cands.is_empty() { break; }
        let (c, in_rmse, cv) = cands.par_iter().map(|&c| {
            let mut trial = selected.clone();
            trial.push(c);
            let (in_rmse, cv) = eval_subset(&g, &bvec, &yty, n_used, &trial, bias_col, args.lambda);
            (c, in_rmse, cv)
        }).min_by(|x, y| x.2.partial_cmp(&y.2).expect("NaN RMSE")).unwrap();
        if !cv.is_finite() { eprintln!("  stop: every remaining plain term is singular"); break; }
        in_sel[c] = true;
        selected.push(c);
        let m = pairs[c].0 as usize;
        forced_models.push(m);
        let (label, ncols) = if column_mode {
            (format!("{} (model)", ucols[m].name), Some(1))
        } else {
            (pair_label(c), None)
        };
        show_step(steps.len(), &label, ncols, in_rmse, cv);
        let _ = record!(label, in_rmse, cv);
    }
    since_best = 0;

    match args.select {
        // One interaction column u_i·u_j per step.
        SelectMode::Pairs => {
            println!("Selecting interactions (pairs mode, target {} columns)...", target);
            while selected.len() < target {
                let candidates: Vec<usize> = (0..dpair).filter(|c| !in_sel[*c]).collect();
                if candidates.is_empty() { break; }
                let best = candidates.par_iter().map(|&c| {
                    let mut trial = selected.clone();
                    trial.push(c);
                    let (in_rmse, cv) = eval_subset(&g, &bvec, &yty, n_used, &trial, bias_col, args.lambda);
                    (c, in_rmse, cv)
                }).min_by(|x, y| x.2.partial_cmp(&y.2).expect("NaN RMSE")).unwrap();
                let (c, in_rmse, cv) = best;
                if !cv.is_finite() { eprintln!("  stop: every remaining candidate is singular"); break; }
                in_sel[c] = true;
                selected.push(c);
                let label = pair_label(c);
                show_step(steps.len(), &label, None, in_rmse, cv);
                if record!(label, in_rmse, cv) {
                    eprintln!("  stop: no CV improvement for {} steps (best step {})",
                        since_best, best_step);
                    break;
                }
            }
        }
        // One U column per step, added as a model or a feature — pulling in its
        // block of interactions with the currently-selected other set.
        SelectMode::Column => {
            let mut m_sel = vec![false; nu];
            let mut f_sel = vec![false; nu];
            for &m in &forced_models { m_sel[m] = true; }
            f_sel[cst] = true;
            let step_target = args.max_features.unwrap_or(usize::MAX);
            println!("Selecting columns (column mode, target {} steps)...",
                args.max_features.map(|k| k.to_string()).unwrap_or_else(|| "all".into()));
            let mut n_steps = forced_models.len();
            while n_steps < step_target {
                // Candidate (column, as_model) options still available in that role.
                let mut opts: Vec<(usize, bool)> = Vec::new();
                for c in 0..nu {
                    if !m_sel[c] { opts.push((c, true)); }
                    if !f_sel[c] { opts.push((c, false)); }
                }
                let best = opts.par_iter().filter_map(|&(c, as_model)| {
                    // New interaction block: c against the whole other set.
                    let new_ix: Vec<usize> = if as_model {
                        (0..nu).filter(|&f| f_sel[f]).map(|f| colindex(c, f))
                            .filter(|ix| !in_sel[*ix]).collect()
                    } else {
                        (0..nu).filter(|&mm| m_sel[mm]).map(|mm| colindex(mm, c))
                            .filter(|ix| !in_sel[*ix]).collect()
                    };
                    if new_ix.is_empty() { return None; }
                    let mut trial = selected.clone();
                    trial.extend_from_slice(&new_ix);
                    let (in_rmse, cv) = eval_subset(&g, &bvec, &yty, n_used, &trial, bias_col, args.lambda);
                    Some((c, as_model, new_ix, in_rmse, cv))
                }).min_by(|a, b| a.4.partial_cmp(&b.4).expect("NaN RMSE"));
                let Some((c, as_model, new_ix, in_rmse, cv)) = best else { break; };
                if !cv.is_finite() { eprintln!("  stop: every remaining column is singular"); break; }
                if as_model { m_sel[c] = true; } else { f_sel[c] = true; }
                for &ix in &new_ix { in_sel[ix] = true; }
                selected.extend_from_slice(&new_ix);
                let role = if as_model { "model" } else { "feature" };
                let label = format!("{} ({})", ucols[c].name, role);
                show_step(steps.len(), &label, Some(new_ix.len()), in_rmse, cv);
                if record!(label, in_rmse, cv) {
                    eprintln!("  stop: no CV improvement for {} steps (best step {})",
                        since_best, best_step);
                    break;
                }
                n_steps += 1;
            }
        }
    }

    // CV column headers/cells, present only when there is more than one fold.
    let cv_hdr = if show_cv { format!(" {:>12}", "cv_pr") } else { String::new() };
    let cv_cell = |v: f64| if show_cv { format!(" {:>12.6}", v) } else { String::new() };
    if show_cv {
        println!("Best CV:   {:.6} at step {} ({} columns)", best_cv, best_step, fits[best_step].len());
    }

    // --no-score: Gram-recovered RMSE only (fast λ/selection sweeps), no reload.
    if args.no_score {
        println!();
        println!("{:>4}  {:<52} {:>13}{}", "step", "added", "insample_pr", cv_hdr);
        for (k, (label, in_rmse, cv)) in steps.iter().enumerate() {
            println!("{:>4}  {:<52} {:>13.6}{}", k, label, in_rmse, cv_cell(*cv));
        }
        return ExitCode::SUCCESS;
    }

    // Refit weights for every prefix on the full probe subset (for the clipped
    // probe/qual scoring) — the CV split only drives the selection criterion.
    let weights: Vec<DVector<f64>> = fits.iter()
        .map(|cols| {
            let (a, rhs) = total_subsystem(&g, &bvec, cols);
            solve_subsystem(&a, &rhs, cols, bias_col, args.lambda)
                .expect("selected subsystem is singular")
        })
        .collect();

    // Probe columns + ratings: reuse those from compute_state, or reload them
    // (load-state path) using the state's own in-clip bounds so they match the Gram.
    let (u_pr, y_pr) = match (u_pr_opt, y_pr_opt) {
        (Some(u), Some(y)) => (u, y),
        _ => {
            let y: Array1<i8> = read_npy(format!("data/{}/ratings.npy", pr))
                .unwrap_or_else(|e| panic!("read data/{}/ratings.npy: {}", pr, e));
            let u = load_u_columns(&ucols, &preds, &pr, n_pr, in_clip_min as f32, in_clip_max as f32);
            (u, y)
        }
    };

    // --- Clipped probe RMSE over the selected subset (from in-memory U) ---
    println!();
    println!("Scoring {} prefixes on probe + qual...", fits.len());
    let probe_scores: Vec<f64> = (0..fits.len()).into_par_iter().map(|fi| {
        let cols = &fits[fi];
        let w = &weights[fi];
        let mut sse = 0.0f64;
        for &row in &sel_rows {
            let mut yhat = 0.0f64;
            for (wi, &c) in cols.iter().enumerate() {
                let val = if c == bias_col {
                    1.0
                } else {
                    let (i, j) = pairs[c];
                    (u_pr[i as usize][row] as f64) * (u_pr[j as usize][row] as f64)
                };
                yhat += w[wi] * val;
            }
            let yh = yhat.clamp(args.out_clip_min, args.out_clip_max);
            let e = yh - y_pr[row] as f64;
            sse += e * e;
        }
        (sse / n_used as f64).sqrt()
    }).collect();
    drop(u_pr);

    // --- Clipped qual quiz RMSE (streamed) ---
    let y_q: Array1<i8> = read_npy(format!("data/{}/ratings.npy", qual))
        .unwrap_or_else(|e| panic!("read data/{}/ratings.npy: {}", qual, e));
    let is_test_q: Array1<i8> = read_npy(format!("data/{}/is_test.npy", qual))
        .unwrap_or_else(|e| panic!("read data/{}/is_test.npy: {}", qual, e));
    let n_q = y_q.len();
    let mut q_readers: Vec<Option<NpyF32Reader>> = ucols.iter().map(|c| {
        if c.name == "<const>" { None } else {
            let path = format!("{}/{}.{}.npy", preds, c.name, qual);
            let r = NpyF32Reader::open(&path);
            assert_eq!(r.len, n_q, "{}: length {} != qual {}", path, r.len, n_q);
            Some(r)
        }
    }).collect();
    let clips: Vec<(bool, f32, f32)> = ucols.iter()
        .map(|c| (c.clip, in_clip_min as f32, in_clip_max as f32)).collect();
    let mut qual_col = |start: usize, bl: usize| -> Vec<Vec<f32>> {
        q_readers.iter_mut().enumerate().map(|(ui, r)| {
            match r {
                None => vec![1.0f32; bl],
                Some(rd) => {
                    let mut buf = vec![0.0f32; bl];
                    rd.read_block(start, bl, &mut buf);
                    let (cl, lo, hi) = clips[ui];
                    clip_vec(&mut buf, cl, lo, hi);
                    buf
                }
            }
        }).collect()
    };
    let quiz_scores: Vec<f64> = fits.iter().zip(weights.iter()).map(|(cols, w)| {
        let (sse, cnt) = fit_sse(&pairs, cols, w.as_slice(), bias_col, n_q,
            y_q.as_slice().unwrap(), Some(is_test_q.as_slice().unwrap()),
            args.out_clip_min, args.out_clip_max, &mut qual_col);
        (sse / cnt as f64).sqrt()
    }).collect();

    // --- Report ---
    println!();
    println!("{:>4}  {:<52} {:>13}{} {:>12} {:>12}",
        "step", "interaction added", "insample_pr", cv_hdr, "clip_probe", "quiz");
    for (k, ((label, in_rmse, cv), (p, q))) in
        steps.iter().zip(probe_scores.iter().zip(quiz_scores.iter())).enumerate()
    {
        println!("{:>4}  {:<52} {:>13.6}{} {:>12.6} {:>12.6}",
            k, label, in_rmse, cv_cell(*cv), p, q);
    }

    ExitCode::SUCCESS
}
