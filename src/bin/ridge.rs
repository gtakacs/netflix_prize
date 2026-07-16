//! Linear blending of model predictions. Computes a single shared Gram
//! matrix over all listed models on the probe split, then solves a ridge
//! least-squares fit per group (and across all groups) by slicing
//! submatrices. Probe and quiz RMSE are evaluated after clipping the
//! blended predictions to [CLIP_MIN, CLIP_MAX].

extern crate blas;
extern crate blas_src;

use blas::dsyrk;
use indexmap::IndexMap;
use netflix_prize::blend::{expand_specs, load_models_toml, select_groups};
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use ndarray_npy::read_npy;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::process::ExitCode;

const NOCLIP_OP: char = '>';
// Default clip bounds: inputs are clipped wide (mostly to tame outliers), the
// blended output is clipped tight to the rating scale.
const IN_CLIP_MIN: f64 = 0.0;
const IN_CLIP_MAX: f64 = 6.0;
const OUT_CLIP_MIN: f64 = 1.05;
const OUT_CLIP_MAX: f64 = 4.95;
const ROW_BLOCK: usize = 100_000;
const PIPELINE_OLD: &str = "pipeline-old.toml";
const PIPELINE_NEW: &str = "pipeline-new.toml";
const MODELS_OLD: &str = "models-old.toml";
const MODELS_NEW: &str = "models-new.toml";

// ---------------------------------------------------------------------------
// Partial .npy reader for 1-D float32 arrays
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
    pipeline: String,
    models_toml: Option<String>,
    models_manual: Vec<String>,
    models_exclude: Vec<String>,
    groups: Vec<String>,
    lambda: f64,
    forward: bool,
    max_features: Option<usize>,
    fixed_group: Option<String>,
    in_clip_min: f64,
    in_clip_max: f64,
    out_clip_min: f64,
    out_clip_max: f64,
    quiz_blend: bool,
    decimals: i32,
}

fn print_help() {
    println!("Usage: ridge [SPLIT] [MODELS] [-m NAME ...] [--lambda VALUE]");
    println!();
    println!("  Split selection (default: {}):", PIPELINE_OLD);
    println!("    -o, --old                shortcut for -p {}", PIPELINE_OLD);
    println!("    -n, --new                shortcut for -p {}", PIPELINE_NEW);
    println!("    -p FILE, --pipeline FILE explicit pipeline TOML");
    println!();
    println!("  Models selection (at least one of -O/-N/-t/-m required):");
    println!("    -O                       -o + default -t {}", MODELS_OLD);
    println!("    -N                       -n + default -t {}", MODELS_NEW);
    println!("    -t FILE, --models FILE   models TOML (groups: list per key)");
    println!("    -g GRP1,GRP2,..., --groups  include only these TOML groups (default: all;");
    println!("                             e.g. -g integrated,rbm,other for base predictors only)");
    println!("    -m NAME, --model NAME    add a single model (repeatable; combines with -t)");
    println!("    -x NAME, --exclude NAME  drop a model by name (repeatable; applied after -t/-m)");
    println!();
    println!("    --lambda VALUE           ridge regularization λ (default 10)");
    println!("    --in-clip MIN,MAX        clip input predictions (skips '>' columns; default {IN_CLIP_MIN},{IN_CLIP_MAX})");
    println!("    --out-clip MIN,MAX       clip the blended output before RMSE (default {OUT_CLIP_MIN},{OUT_CLIP_MAX})");
    println!();
    println!("  Forward feature selection (Gram computed once, then submatrix slicing):");
    println!("    --forward                greedily add models by in-sample (Gram) probe RMSE");
    println!("    --max-features K         stop after K total selected features (incl. --fixed)");
    println!("    --fixed GROUP            pre-select all models in GROUP, search over the rest");
    println!();
    println!("  Quiz blending (fit on the qual labels recovered from RMSE probing):");
    println!("    --quiz-blend             build the Gram over the full qual set and recover");
    println!("                             Z'y from rounded per-model + constant RMSE probes");
    println!("    --decimals N             RMSE feedback precision for --quiz-blend (default 4)");
    println!();
    println!("    -h, --help               show this help");
}

fn set_models_toml(a: &mut Args, path: String, flag: &str) {
    if a.models_toml.is_some() {
        eprintln!("error: '{}' conflicts with an earlier models TOML selection", flag);
        std::process::exit(2);
    }
    a.models_toml = Some(path);
}

fn parse_args() -> Args {
    let mut a = Args {
        pipeline: PIPELINE_OLD.to_string(),
        models_toml: None,
        models_manual: Vec::new(),
        models_exclude: Vec::new(),
        groups: Vec::new(),
        lambda: 10.0,
        forward: false,
        max_features: None,
        fixed_group: None,
        in_clip_min: IN_CLIP_MIN,
        in_clip_max: IN_CLIP_MAX,
        out_clip_min: OUT_CLIP_MIN,
        out_clip_max: OUT_CLIP_MAX,
        quiz_blend: false,
        decimals: 4,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            "-o" | "--old" => { a.pipeline = PIPELINE_OLD.to_string(); i += 1; }
            "-n" | "--new" => { a.pipeline = PIPELINE_NEW.to_string(); i += 1; }
            "-O" => {
                a.pipeline = PIPELINE_OLD.to_string();
                set_models_toml(&mut a, MODELS_OLD.to_string(), "-O");
                i += 1;
            }
            "-N" => {
                a.pipeline = PIPELINE_NEW.to_string();
                set_models_toml(&mut a, MODELS_NEW.to_string(), "-N");
                i += 1;
            }
            "-p" | "--pipeline" => { a.pipeline = need(&argv, i); i += 2; }
            "-t" | "--models" => {
                let path = need(&argv, i);
                set_models_toml(&mut a, path, &argv[i]);
                i += 2;
            }
            "-g" | "--groups" => {
                for tok in need(&argv, i).split(',') { a.groups.push(tok.trim().to_string()); }
                i += 2;
            }
            "-m" | "--model" => { a.models_manual.push(need(&argv, i)); i += 2; }
            "-x" | "--exclude" => { a.models_exclude.push(need(&argv, i)); i += 2; }
            "--lambda" => {
                a.lambda = need(&argv, i).parse().expect("bad --lambda value");
                i += 2;
            }
            "--forward" => { a.forward = true; i += 1; }
            "--max-features" => {
                a.max_features = Some(need(&argv, i).parse().expect("bad --max-features value"));
                i += 2;
            }
            "--fixed" => { a.fixed_group = Some(need(&argv, i)); i += 2; }
            "--quiz-blend" => { a.quiz_blend = true; i += 1; }
            "--decimals" => {
                a.decimals = need(&argv, i).parse().expect("bad --decimals value");
                i += 2;
            }
            "--in-clip" => {
                let (lo, hi) = parse_clip(&need(&argv, i), "--in-clip");
                a.in_clip_min = lo; a.in_clip_max = hi; i += 2;
            }
            "--out-clip" => {
                let (lo, hi) = parse_clip(&need(&argv, i), "--out-clip");
                a.out_clip_min = lo; a.out_clip_max = hi; i += 2;
            }
            s => { eprintln!("error: unknown arg '{}'", s); print_help(); std::process::exit(2); }
        }
    }
    if a.models_toml.is_none() && a.models_manual.is_empty() {
        eprintln!("error: provide -N/-O, -t MODELS_TOML, and/or -m NAME");
        std::process::exit(2);
    }
    if !a.groups.is_empty() && a.models_toml.is_none() {
        eprintln!("error: -g/--groups requires a models TOML (-t/-N/-O)");
        std::process::exit(2);
    }
    if !a.forward && (a.max_features.is_some() || a.fixed_group.is_some()) {
        eprintln!("warning: --max-features/--fixed have no effect without --forward");
    }
    if a.quiz_blend && a.forward {
        eprintln!("error: --quiz-blend cannot be combined with --forward");
        std::process::exit(2);
    }
    a
}

fn need(argv: &[String], i: usize) -> String {
    if i + 1 >= argv.len() {
        eprintln!("error: '{}' requires an argument", argv[i]);
        std::process::exit(2);
    }
    argv[i + 1].clone()
}

/// Parse a `MIN,MAX` clip-bound pair (e.g. `1.0,4.95`).
fn parse_clip(s: &str, flag: &str) -> (f64, f64) {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        eprintln!("error: '{}' expects MIN,MAX (got '{}')", flag, s);
        std::process::exit(2);
    }
    let lo = parts[0].trim().parse::<f64>()
        .unwrap_or_else(|_| { eprintln!("error: bad {} MIN", flag); std::process::exit(2); });
    let hi = parts[1].trim().parse::<f64>()
        .unwrap_or_else(|_| { eprintln!("error: bad {} MAX", flag); std::process::exit(2); });
    if lo > hi {
        eprintln!("error: '{}' MIN ({}) > MAX ({})", flag, lo, hi);
        std::process::exit(2);
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// Pipeline / models TOML
// ---------------------------------------------------------------------------

fn load_pipeline_split(path: &str) -> HashMap<String, String> {
    #[derive(serde::Deserialize)]
    struct P { #[serde(default)] split: HashMap<String, String> }
    let s = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let p: P = toml::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {}", path, e));
    p.split
}

/// Build the model registry: unique names, clip flags, and group → indices.
fn build_registry(
    args: &Args,
) -> (Vec<String>, Vec<bool>, IndexMap<String, Vec<usize>>) {
    let mut groups: IndexMap<String, Vec<String>> = if let Some(p) = &args.models_toml {
        select_groups(&load_models_toml(p), &args.groups)
    } else {
        IndexMap::new()
    };
    if !args.models_manual.is_empty() {
        groups.insert("manual".to_string(), args.models_manual.clone());
    }

    // Normalized set of names to drop (each --exclude arg is brace-expanded and
    // stripped of a leading '>' so it matches the resolved registry name).
    let mut excluded: std::collections::HashSet<String> = std::collections::HashSet::new();
    for raw in &args.models_exclude {
        for spec in expand_specs(raw) {
            let name = spec.strip_prefix(NOCLIP_OP).unwrap_or(&spec).to_string();
            excluded.insert(name);
        }
    }
    let mut exclude_hit: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut unique: Vec<String> = Vec::new();
    let mut clip: Vec<bool> = Vec::new();
    let mut idx_of: HashMap<String, usize> = HashMap::new();
    let mut group_indices: IndexMap<String, Vec<usize>> = IndexMap::new();

    for (gname, specs) in &groups {
        let mut idxs = Vec::new();
        for raw in specs {
            for spec in expand_specs(raw) {
                let (name, c) = if let Some(rest) = spec.strip_prefix(NOCLIP_OP) {
                    (rest.to_string(), false)
                } else {
                    (spec.clone(), true)
                };
                if excluded.contains(&name) {
                    exclude_hit.insert(name);
                    continue;
                }
                let i = *idx_of.entry(name.clone()).or_insert_with(|| {
                    unique.push(name.clone());
                    clip.push(c);
                    unique.len() - 1
                });
                // If the same name appears with conflicting clip flags, prefer no-clip
                // (any '>' wins). This matches the Python load_preds semantics where
                // '>' on the spec disables clipping at load time.
                if !c { clip[i] = false; }
                idxs.push(i);
            }
        }
        // Drop groups emptied by exclusion rather than emit a degenerate
        // bias-only fit row downstream.
        if idxs.is_empty() {
            eprintln!("warning: group '{}' is empty after exclusion — skipping", gname);
            continue;
        }
        group_indices.insert(gname.clone(), idxs);
    }

    for name in excluded.difference(&exclude_hit) {
        eprintln!("warning: --exclude '{}' matched no model", name);
    }

    (unique, clip, group_indices)
}

// ---------------------------------------------------------------------------
// Gram matrix accumulation (single pass over predictions)
// ---------------------------------------------------------------------------

/// Parallel per-block prediction loader. Each model's slice is read into its
/// own `col_bufs[i]` from a rayon worker (one syscall per model in flight);
/// then the strided assembly into the column-major `zt_f32` is done serially.
/// `zt_f32`'s bias row (i == m) is assumed pre-initialised to 1.0.
#[allow(clippy::too_many_arguments)]
fn load_block_parallel(
    readers: &mut [NpyF32Reader],
    clip: &[bool],
    in_clip_min: f32,
    in_clip_max: f32,
    col_bufs: &mut [Vec<f32>],
    zt_f32: &mut [f32],
    start: usize,
    blen: usize,
    dim: usize,
) {
    readers
        .par_iter_mut()
        .zip(col_bufs.par_iter_mut())
        .enumerate()
        .for_each(|(i, (r, buf))| {
            r.read_block(start, blen, &mut buf[..blen]);
            if clip[i] {
                for k in 0..blen {
                    buf[k] = buf[k].clamp(in_clip_min, in_clip_max);
                }
            }
        });
    for i in 0..readers.len() {
        let buf = &col_bufs[i];
        for k in 0..blen {
            zt_f32[i + k * dim] = buf[k];
        }
    }
}

/// Returns (A, b) where:
///   A = Zᵀ Z   ((m+1)×(m+1), row-major, last col/row is bias)
///   b = Zᵀ y   (m+1)
/// Z has shape (n × (m+1)) with the last column all-ones for the bias.
fn build_gram(
    y: &[f64],
    readers: &mut [NpyF32Reader],
    clip: &[bool],
    in_clip_min: f32,
    in_clip_max: f32,
) -> (Vec<f64>, Vec<f64>) {
    let m = readers.len();
    let dim = m + 1;
    let n = y.len();
    let mut a = vec![0.0f64; dim * dim];
    let mut b = vec![0.0f64; dim];

    // Persistent f32 block buffer: column-major (dim × blen, leading dim = dim).
    // Cell (i + k*dim) holds model i's prediction at row k; the bias row is at
    // i == m and stays 1.0 across blocks.
    let mut zt_f32 = vec![0.0f32; ROW_BLOCK * dim];
    for k in 0..ROW_BLOCK {
        zt_f32[m + k * dim] = 1.0;
    }
    // Per-block f64 buffer fed to BLAS — populated by an explicit cast right
    // before each dsyrk/dgemv pair.
    let mut zt_f64 = vec![0.0f64; ROW_BLOCK * dim];
    // Per-model scratch buffers, populated in parallel and then strided into
    // zt_f32. Decouples I/O parallelism from the column-major BLAS layout.
    let mut col_bufs: Vec<Vec<f32>> = (0..m).map(|_| vec![0.0f32; ROW_BLOCK]).collect();

    let n_blocks = n.div_ceil(ROW_BLOCK);
    let mut start = 0;
    let mut bidx = 0;
    while start < n {
        let blen = (n - start).min(ROW_BLOCK);

        load_block_parallel(readers, clip, in_clip_min, in_clip_max, &mut col_bufs, &mut zt_f32, start, blen, dim);

        // Cast the active part of the block to f64 for BLAS.
        let len = blen * dim;
        for k in 0..len {
            zt_f64[k] = zt_f32[k] as f64;
        }

        // dsyrk uplo='L' fills the column-major lower triangle, which is the
        // row-major upper triangle in our linear buffer; the post-loop mirror
        // copies that into the row-major lower half.
        unsafe {
            dsyrk(
                b'L',
                b'N',
                dim as i32,
                blen as i32,
                1.0,
                &zt_f64,
                dim as i32,
                1.0,
                &mut a,
                dim as i32,
            );
        }

        // b += Zᵀ y  (column-major (dim × blen) · blen-vector → dim-vector)
        for k in 0..blen {
            let y_k = y[start + k];
            let col = &zt_f64[k * dim..(k + 1) * dim];
            for i in 0..dim {
                b[i] += col[i] * y_k;
            }
        }

        start += blen;
        bidx += 1;
        eprint!("\r  gram block {}/{}", bidx, n_blocks);
    }
    eprintln!();

    // Mirror row-major upper → row-major lower so downstream row-major
    // submatrix slicing reads a fully symmetric matrix.
    for i in 0..dim {
        for j in (i + 1)..dim {
            a[j * dim + i] = a[i * dim + j];
        }
    }
    (a, b)
}

// ---------------------------------------------------------------------------
// Per-group ridge solve
// ---------------------------------------------------------------------------

/// Slice the shared Gram into the `(k×k)` subsystem for `indices` (+ bias as the
/// last row/col). Returns the un-regularized `A_sub` and `b_sub`.
fn build_subsystem(
    a: &[f64],
    b: &[f64],
    dim: usize,
    indices: &[usize],
) -> (DMatrix<f64>, DVector<f64>) {
    let bias_idx = dim - 1;
    let k = indices.len() + 1;
    let mut sub_idx: Vec<usize> = indices.to_vec();
    sub_idx.push(bias_idx);

    let mut a_sub = DMatrix::<f64>::zeros(k, k);
    let mut b_sub = DVector::<f64>::zeros(k);
    for (ii, &i) in sub_idx.iter().enumerate() {
        b_sub[ii] = b[i];
        for (jj, &j) in sub_idx.iter().enumerate() {
            a_sub[(ii, jj)] = a[i * dim + j];
        }
    }
    (a_sub, b_sub)
}

/// Ridge-solve the subsystem: add `lambda` to the first `n_feat` diagonal
/// entries (the bias term stays unregularized) and Cholesky-solve.
fn solve_subsystem(
    a_sub: &DMatrix<f64>,
    b_sub: &DVector<f64>,
    n_feat: usize,
    lambda: f64,
) -> DVector<f64> {
    let mut a_reg = a_sub.clone();
    for i in 0..n_feat {
        a_reg[(i, i)] += lambda;
    }
    let chol = a_reg.cholesky().expect("Gram matrix is not positive definite");
    chol.solve(b_sub)
}

fn solve_group(
    a: &[f64],
    b: &[f64],
    dim: usize,
    indices: &[usize],
    lambda: f64,
) -> DVector<f64> {
    let (a_sub, b_sub) = build_subsystem(a, b, dim, indices);
    solve_subsystem(&a_sub, &b_sub, indices.len(), lambda)
}

// ---------------------------------------------------------------------------
// Forward feature selection (Gram-only criterion)
// ---------------------------------------------------------------------------

/// One forward-selection step: the model added and the in-sample (probe) RMSE
/// of the resulting prefix, recovered purely from the Gram matrix.
struct ForwardStep {
    added: String,
    in_sample_rmse: f64,
}

/// Solve the ridge subsystem for `indices` and return `(w, in_sample_rmse)`.
/// The RMSE is the unclipped probe RMSE recovered from the Gram alone:
///   SSE = yᵀy − 2·wᵀ·b_sub + wᵀ·A_sub·w   (A_sub WITHOUT the λ ridge term).
fn eval_subset(
    a: &[f64],
    b: &[f64],
    yty: f64,
    dim: usize,
    indices: &[usize],
    lambda: f64,
    n: usize,
) -> (DVector<f64>, f64) {
    let (a_sub, b_sub) = build_subsystem(a, b, dim, indices);
    let w = solve_subsystem(&a_sub, &b_sub, indices.len(), lambda);
    let aw = &a_sub * &w;
    let sse = yty - 2.0 * w.dot(&b_sub) + w.dot(&aw);
    (w, (sse / n as f64).sqrt())
}

/// Greedy forward selection driven by the Gram-only in-sample probe RMSE.
/// `fixed` is pre-selected and never dropped. Returns one prefix fit per step
/// (for downstream clipped probe/quiz evaluation) plus per-step metadata.
/// Candidate evaluation within each step runs in parallel over Rayon.
fn forward_select(
    a: &[f64],
    b: &[f64],
    yty: f64,
    dim: usize,
    n: usize,
    m: usize,
    unique: &[String],
    fixed: &[usize],
    max_features: Option<usize>,
    lambda: f64,
) -> (Vec<(String, Vec<usize>, DVector<f64>)>, Vec<ForwardStep>) {
    let mut in_fixed = vec![false; m];
    for &i in fixed {
        in_fixed[i] = true;
    }
    let mut selected: Vec<usize> = fixed.to_vec();
    let mut remaining: Vec<usize> = (0..m).filter(|&i| !in_fixed[i]).collect();

    let mut fits: Vec<(String, Vec<usize>, DVector<f64>)> = Vec::new();
    let mut steps: Vec<ForwardStep> = Vec::new();

    let target = max_features.unwrap_or(m).min(m);

    // Baseline row for the pre-selected (fixed) set, if any.
    if !selected.is_empty() {
        let (w, rmse) = eval_subset(a, b, yty, dim, &selected, lambda, n);
        eprintln!("  baseline ({} fixed): in-sample {:.6}", selected.len(), rmse);
        steps.push(ForwardStep {
            added: format!("<baseline: {} fixed>", selected.len()),
            in_sample_rmse: rmse,
        });
        fits.push(("base".to_string(), selected.clone(), w));
    }

    while !remaining.is_empty() && selected.len() < target {
        let (pos, cand, rmse, w) = remaining
            .par_iter()
            .enumerate()
            .map(|(pos, &c)| {
                let mut trial = selected.clone();
                trial.push(c);
                let (w, rmse) = eval_subset(a, b, yty, dim, &trial, lambda, n);
                (pos, c, rmse, w)
            })
            .min_by(|x, y| x.2.partial_cmp(&y.2).expect("NaN RMSE in candidate eval"))
            .expect("non-empty remaining");

        remaining.remove(pos);
        selected.push(cand);
        eprintln!(
            "  step {}/{}: + {} (in-sample {:.6})",
            selected.len(), target, unique[cand], rmse,
        );
        steps.push(ForwardStep { added: unique[cand].clone(), in_sample_rmse: rmse });
        fits.push((format!("k={}", selected.len()), selected.clone(), w));
    }
    (fits, steps)
}

// ---------------------------------------------------------------------------
// BLAS backend info
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn print_blas_info() {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_void};

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }
    unsafe extern "C" {
        fn openblas_get_config() -> *const c_char;
        fn openblas_get_corename() -> *const c_char;
        fn openblas_get_num_threads() -> c_int;
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }
    unsafe {
        let cfg = CStr::from_ptr(openblas_get_config()).to_string_lossy();
        let core = CStr::from_ptr(openblas_get_corename()).to_string_lossy();
        let nt = openblas_get_num_threads();
        println!("BLAS:      OpenBLAS (core = {core}, threads = {nt})");
        println!("           {cfg}");

        // Locate the file providing the BLAS symbols at runtime.
        let mut info: DlInfo = std::mem::zeroed();
        let addr = openblas_get_config as *const c_void;
        if dladdr(addr, &mut info) != 0 && !info.dli_fname.is_null() {
            let path = CStr::from_ptr(info.dli_fname).to_string_lossy().into_owned();
            let exe = std::env::current_exe().ok();
            let is_self = exe
                .as_deref()
                .and_then(|p| std::fs::canonicalize(p).ok())
                .zip(std::fs::canonicalize(&path).ok())
                .map(|(a, b)| a == b)
                .unwrap_or(false);
            if is_self {
                println!("           lib: {path} (statically linked into executable)");
            } else {
                println!("           lib: {path}");
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn print_blas_info() {
    println!("BLAS:      Apple Accelerate");
}

// ---------------------------------------------------------------------------
// Quiz-blend: recover Zᵀy from rounded RMSE probes
// ---------------------------------------------------------------------------

/// Recover `b = [Xᵀy ; Σy]` from the per-model and constant-prediction RMSE
/// values that the Netflix leaderboard exposed (rounded to `decimals`), rather
/// than from the hidden qual labels directly. This is the offline simulation of
/// the "RMSE probing" attack: each exact RMSE is computed from the true labels,
/// rounded to the feedback precision, then the RMSE identity is inverted.
///
/// `a` is ZᵀZ (with the bias row/col), `b_true` the true Zᵀy, `yty` the true
/// yᵀy — all over the full qual set. Returns the recovered `b`.
fn recover_quiz_b(a: &[f64], b_true: &[f64], yty: f64, n: usize, m: usize, decimals: i32) -> Vec<f64> {
    let dim = m + 1;
    let nf = n as f64;
    let round = |x: f64| {
        let p = 10f64.powi(decimals);
        (x * p).round() / p
    };

    // Step 1: recover mean(y) and yᵀy from two constant probes (c=1, c=5).
    let sum_y_true = b_true[m]; // bias row of Zᵀy is 1ᵀy = Σy
    let rmse_const = |c: f64| ((nf * c * c - 2.0 * c * sum_y_true + yty) / nf).sqrt();
    let (c1, c2) = (1.0f64, 5.0f64);
    let (r1, r2) = (round(rmse_const(c1)), round(rmse_const(c2)));
    let ymean = ((c1 * c1 - c2 * c2) - (r1 * r1 - r2 * r2)) / (2.0 * (c1 - c2));
    let ey2 = r1 * r1 - c1 * c1 + 2.0 * c1 * ymean;
    let yty_rec = nf * ey2;
    let sum_y_rec = nf * ymean;

    // Step 2: recover Xᵀy per model from its RMSE probe.
    let mut b = vec![0.0f64; dim];
    let (mut max_abs, mut sum_abs, mut max_rel, mut sum_rel) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for j in 0..m {
        let xjxj = a[j * dim + j];
        let rmse_j = ((xjxj - 2.0 * b_true[j] + yty) / nf).sqrt();
        let rj = round(rmse_j);
        let xty = (xjxj + yty_rec - nf * rj * rj) / 2.0;
        b[j] = xty;
        let abs = (xty - b_true[j]).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        if b_true[j] != 0.0 {
            let rel = abs / b_true[j].abs();
            max_rel = max_rel.max(rel);
            sum_rel += rel;
        }
    }
    b[m] = sum_y_rec;

    // Diagnostics: recovered vs. true (the labels are available in this offline run).
    let ymean_true = sum_y_true / nf;
    println!();
    println!("Qual label statistics (recovered vs true):");
    println!("  mean(y): {:.6} vs {:.6} (err={:.2e})", ymean, ymean_true, (ymean - ymean_true).abs());
    println!("  y'y:     {:.1} vs {:.1} (err={:.1})", yty_rec, yty, (yty_rec - yty).abs());
    println!("X'y recovery ({} models):", m);
    println!("  Max absolute error:  {:.1}", max_abs);
    println!("  Mean absolute error: {:.1}", sum_abs / m as f64);
    println!("  Max relative error:  {:.2e}", max_rel);
    println!("  Mean relative error: {:.2e}", sum_rel / m as f64);
    println!();
    b
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let args = parse_args();

    let split = load_pipeline_split(&args.pipeline);
    let pr = split.get("pr").expect("pipeline [split].pr missing").clone();
    let preds_dir = split.get("preds").expect("pipeline [split].preds missing").clone();
    let split_name = split.get("name").cloned().unwrap_or_else(|| "?".to_string());

    let (unique, clip, group_indices) = build_registry(&args);
    let m = unique.len();
    if m == 0 {
        eprintln!("error: no models left after exclusion");
        return ExitCode::from(2);
    }

    println!("Pipeline:  {} (split = {})", args.pipeline, split_name);
    match (&args.models_toml, args.models_manual.len()) {
        (Some(t), 0) => println!("Models:    {} ({} unique across {} group(s))",
            t, m, group_indices.len()),
        (Some(t), k) => println!("Models:    {} + {} manual ({} unique across {} group(s))",
            t, k, m, group_indices.len()),
        (None, k) => println!("Models:    {} manual model(s)", k),
    }
    if args.models_toml.is_some() {
        let g = if args.groups.is_empty() { "all".to_string() } else { args.groups.join(",") };
        println!("Groups:    {}", g);
    }
    if !args.models_exclude.is_empty() {
        println!("Excluded:  {} name(s): {}",
            args.models_exclude.len(), args.models_exclude.join(", "));
    }
    println!("Lambda λ:  {}", args.lambda);
    println!("In-clip:   [{}, {}] (skips '>' columns)", args.in_clip_min, args.in_clip_max);
    println!("Out-clip:  [{}, {}]", args.out_clip_min, args.out_clip_max);
    print_blas_info();
    println!();

    // Load probe ratings (i8) → f32 (for the probe evaluation pass).
    let y_path = format!("data/{}/ratings.npy", pr);
    let y_i8: Array1<i8> = read_npy(&y_path).unwrap_or_else(|e| panic!("read {}: {}", y_path, e));
    let y: Vec<f64> = y_i8.iter().map(|&r| r as f64).collect();
    let n_probe = y.len();
    println!("Probe set: {} ratings ({})", n_probe, y_path);

    // Open one partial probe reader per unique model.
    let mut readers: Vec<NpyF32Reader> = Vec::with_capacity(m);
    for name in &unique {
        let path = format!("{}/{}.{}.npy", preds_dir, name, pr);
        let r = NpyF32Reader::open(&path);
        assert_eq!(r.len, n_probe, "{}: length {} != probe {}", path, r.len, n_probe);
        readers.push(r);
    }

    // Load qual ratings + is_test (for the quiz evaluation pass; in quiz-blend
    // mode the Gram is also built here, over the full qual set).
    let qual = split.get("fulltrain_pr").expect("pipeline [split].fulltrain_pr missing").clone();
    let y_q_i8: Array1<i8> =
        read_npy(format!("data/{}/ratings.npy", qual))
            .unwrap_or_else(|e| panic!("read data/{}/ratings.npy: {}", qual, e));
    let is_test_q: Array1<i8> =
        read_npy(format!("data/{}/is_test.npy", qual))
            .unwrap_or_else(|e| panic!("read data/{}/is_test.npy: {}", qual, e));
    let n_q = y_q_i8.len();
    let quiz_n_expected = is_test_q.iter().filter(|&&t| t == 0).count();
    println!("Quiz set:  {} of {} qual ratings", quiz_n_expected, n_q);

    // Open one partial qual reader per unique model.
    let mut qual_readers: Vec<NpyF32Reader> = Vec::with_capacity(m);
    for name in &unique {
        let path = format!("{}/{}.{}.npy", preds_dir, name, qual);
        let r = NpyF32Reader::open(&path);
        assert_eq!(r.len, n_q, "{}: length {} != qual {}", path, r.len, n_q);
        qual_readers.push(r);
    }

    // Build the shared Gram (A = ZᵀZ), its right-hand side (b = Zᵀy) and yty.
    // Normal mode fits on the probe labels; quiz-blend mode fits on the full
    // qual labels recovered from rounded per-model + constant RMSE probes.
    let dim = m + 1;
    let (a, b, yty, n): (Vec<f64>, Vec<f64>, f64, usize) = if args.quiz_blend {
        let y_q: Vec<f64> = y_q_i8.iter().map(|&r| r as f64).collect();
        println!(
            "Quiz-blend: building Gram over full qual ({} ratings), decimals = {}",
            n_q, args.decimals,
        );
        let (a, b_true) = build_gram(&y_q, &mut qual_readers, &clip,
            args.in_clip_min as f32, args.in_clip_max as f32);
        let yty: f64 = y_q.iter().map(|v| v * v).sum();
        let b = recover_quiz_b(&a, &b_true, yty, n_q, m, args.decimals);
        (a, b, yty, n_q)
    } else {
        println!("Building Gram matrix over {} model(s) × {} ratings...", m, n_probe);
        let (a, b) = build_gram(&y, &mut readers, &clip,
            args.in_clip_min as f32, args.in_clip_max as f32);
        let yty: f64 = y.iter().map(|v| v * v).sum();
        (a, b, yty, n_probe)
    };

    // Build the fits to evaluate: either forward-selection prefixes (one fit per
    // step) or the per-group (and 'all') ridge solves.
    let (fits, fwd_steps): (Vec<(String, Vec<usize>, DVector<f64>)>, Option<Vec<ForwardStep>>) =
        if args.forward {
            let fixed: Vec<usize> = match &args.fixed_group {
                Some(g) => group_indices.get(g).cloned().unwrap_or_else(|| {
                    eprintln!("error: --fixed group '{}' not found in models", g);
                    std::process::exit(2);
                }),
                None => Vec::new(),
            };
            println!(
                "Forward:   λ={} candidates={} fixed={} max_features={}",
                args.lambda,
                m - fixed.len(),
                fixed.len(),
                args.max_features.map(|k| k.to_string()).unwrap_or_else(|| "all".to_string()),
            );
            println!();
            println!("Phase 1/2: forward selection — at each step greedily add the predictor");
            println!("           that most lowers the in-sample (Gram-only, unclipped) probe RMSE.");
            let (fits, steps) =
                forward_select(&a, &b, yty, dim, n, m, &unique, &fixed, args.max_features, args.lambda);
            (fits, Some(steps))
        } else {
            // Solve per group, plus 'all' if more than one group
            let mut fits: Vec<(String, Vec<usize>, DVector<f64>)> = Vec::new();
            for (gname, gidxs) in &group_indices {
                let w = solve_group(&a, &b, dim, gidxs, args.lambda);
                fits.push((gname.clone(), gidxs.clone(), w));
            }
            if group_indices.len() > 1 {
                let all_idxs: Vec<usize> = (0..m).collect();
                let w = solve_group(&a, &b, dim, &all_idxs, args.lambda);
                fits.push(("all".to_string(), all_idxs, w));
            }
            (fits, None)
        };

    if fwd_steps.is_some() {
        println!();
        println!("Phase 2/2: streaming probe + qual passes to score the actual clipped");
        println!("           probe & quiz RMSE for every selected prefix ({} fits).", fits.len());
    }

    // Second pass on probe: compute clipped RMSE per fit
    let (probe_sse, probe_n) = compute_clipped_sse(
        &mut readers, &clip,
        args.in_clip_min as f32, args.in_clip_max as f32, args.out_clip_min, args.out_clip_max,
        &y_i8, None, &fits, m, "probe",
    );

    // Third pass on qual: compute clipped quiz RMSE per fit (mask via is_test).
    // y_q_i8 / is_test_q / qual_readers were loaded above; reuse them (the
    // readers re-seek on every block, so quiz-blend's Gram pass left them usable).
    let (quiz_sse, quiz_n) = compute_clipped_sse(
        &mut qual_readers, &clip,
        args.in_clip_min as f32, args.in_clip_max as f32, args.out_clip_min, args.out_clip_max,
        &y_q_i8, Some(&is_test_q), &fits, m, "quiz",
    );

    println!();
    match &fwd_steps {
        Some(steps) => {
            println!(
                "{:>4}  {:<40} {:>13} {:>11} {:>11}  {:>9}",
                "step", "model added", "insample_pr", "clip_probe", "quiz", "delta",
            );
            let mut prev = f64::INFINITY;
            for (i, (step, (p_sse, q_sse))) in
                steps.iter().zip(probe_sse.iter().zip(quiz_sse.iter())).enumerate()
            {
                let p_rmse = (p_sse / probe_n as f64).sqrt();
                let q_rmse = (q_sse / quiz_n as f64).sqrt();
                let delta = if prev.is_finite() { step.in_sample_rmse - prev } else { 0.0 };
                prev = step.in_sample_rmse;
                println!(
                    "{:>4}  {:<40} {:>13.6} {:>11.6} {:>11.6}  {:>+9.6}",
                    i + 1, step.added, step.in_sample_rmse, p_rmse, q_rmse, delta,
                );
            }
        }
        None => {
            println!("{:<16} {:>8} {:>14} {:>14}", "group", "models", "probe_rmse", "quiz_rmse");
            for ((name, gidxs, _w), (p_sse, q_sse)) in
                fits.iter().zip(probe_sse.iter().zip(quiz_sse.iter()))
            {
                let p_rmse = (p_sse / probe_n as f64).sqrt();
                let q_rmse = (q_sse / quiz_n as f64).sqrt();
                println!("{:<16} {:>8} {:>14.6} {:>14.6}", name, gidxs.len(), p_rmse, q_rmse);
            }
        }
    }

    ExitCode::SUCCESS
}

/// Second-pass SSE evaluator: streams predictions in row-blocks (input-clipped
/// per the `>` flag to `[in_clip_min, in_clip_max]`), applies the fit weights,
/// clips the blended prediction to `[out_clip_min, out_clip_max]`, and
/// accumulates SSE per fit. When `mask` is `Some`, only rows where `mask == 0`
/// are counted (used for quiz scoring against `is_test`).
#[allow(clippy::too_many_arguments)]
fn compute_clipped_sse(
    readers: &mut [NpyF32Reader],
    clip: &[bool],
    in_clip_min: f32,
    in_clip_max: f32,
    out_clip_min: f64,
    out_clip_max: f64,
    y: &Array1<i8>,
    mask: Option<&Array1<i8>>,
    fits: &[(String, Vec<usize>, DVector<f64>)],
    m: usize,
    label: &str,
) -> (Vec<f64>, usize) {
    let n = y.len();
    let n_fits = fits.len();
    let dim = m + 1;
    let mut sse = vec![0.0f64; n_fits];
    let mut count = 0usize;

    // Persistent f32 prediction buffer (column-major dim × ROW_BLOCK) + bias row.
    let mut zt_f32 = vec![0.0f32; ROW_BLOCK * dim];
    for k in 0..ROW_BLOCK {
        zt_f32[m + k * dim] = 1.0;
    }
    let mut col_bufs: Vec<Vec<f32>> = (0..m).map(|_| vec![0.0f32; ROW_BLOCK]).collect();

    let n_blocks = n.div_ceil(ROW_BLOCK);
    let mut start = 0;
    let mut bidx = 0;
    while start < n {
        let blen = (n - start).min(ROW_BLOCK);

        load_block_parallel(readers, clip, in_clip_min, in_clip_max, &mut col_bufs, &mut zt_f32, start, blen, dim);

        for (fi, (_, gidxs, w)) in fits.iter().enumerate() {
            let bias = w[gidxs.len()];
            let mut acc = 0.0f64;
            for k in 0..blen {
                if let Some(mk) = mask {
                    if mk[start + k] != 0 { continue; }
                }
                let mut yhat = bias;
                for (jj, &gi) in gidxs.iter().enumerate() {
                    yhat += w[jj] * zt_f32[gi + k * dim] as f64;
                }
                let yh = yhat.clamp(out_clip_min, out_clip_max);
                let err = yh - y[start + k] as f64;
                acc += err * err;
            }
            sse[fi] += acc;
        }

        for k in 0..blen {
            if let Some(mk) = mask {
                if mk[start + k] != 0 { continue; }
            }
            count += 1;
        }

        start += blen;
        bidx += 1;
        eprint!("\r  {} block {}/{}", label, bidx, n_blocks);
    }
    eprintln!();

    (sse, count)
}
