//! Disk-based FWLS (Feature-Weighted Linear Stacking) cross-fit blend.
//!
//! Same math as `fwls`, but restructured to be **memory-bound** for a large
//! interaction dimension `D = M·P + 1` (e.g. M=1000 models × P=100 voting →
//! D≈100k, whose Gram is 80 GB dense / 40 GB packed). Two ideas:
//!
//!  1. **Packed lower triangle** (`D(D+1)/2` f64, column-major LAPACK 'L'
//!     layout) is the only `D`-sized object ever held — on disk and in RAM.
//!     The build fills it via BLAS `dgemm` in column panels (never a dense
//!     `D×D`); the Cholesky factorizes it in place. So peak RAM ≈ one packed
//!     matrix (~40 GB at D=100k) plus the in-memory probe columns.
//!
//!  2. **Sufficient-statistics differencing for K-fold CV.** The Gram is
//!     additive over rows, so we build each of the K folds' `(AᵀA, Aᵀy)` once,
//!     store them under `./tmp/fwls-diskbased-<ts>/`, and sum to a `total`.
//!     For held-out fold n the training system is `total − chunk_n` — exact up
//!     to f64 rounding (see note below), so no fold is rebuilt K−1 times.
//!
//! Probe RMSE is the out-of-fold estimate; qual is predicted once from the
//! full `total` fit. Unlike `fwls`, the bias column is regularized like every
//! other column (`AᵀA + λI` on the full diagonal), which keeps the packed
//! Cholesky well-conditioned even when constant columns are present.
//!
//! Requires the `blas` feature (Accelerate on macOS, OpenBLAS on Linux).
//!
//! Rounding / leakage note: `total − chunk_n` differs from the exact Gram of
//! the other K−1 folds only by ~1e-16 relative rounding, and with K≥2 each
//! chunk is a bounded fraction of the total so there is no catastrophic
//! cancellation. That perturbation is symmetric numerical noise, not a
//! function of fold n's targets, so it does not leak fold-n information into
//! its own out-of-fold prediction.

extern crate blas;
extern crate blas_src;

use blas::dgemm;
use ndarray::Array1;
use ndarray_npy::read_npy;
use netflix_prize::blend::{
    close_log, expand_globs, flatten_groups, load_models_toml, log_columns, open_log, resolve_voting, save_preds,
    select_groups, CLIP_MAX, CLIP_MIN,
};
use netflix_prize::teeln;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::process::{Command, ExitCode};

/// Row block for the streamed qual prediction passes.
const ROW_BLOCK: usize = 100_000;
/// Target byte size of one `D × blen` f64 interaction block during the Gram.
const GRAM_BYTES: usize = 256_000_000;
/// Target byte size of the `(D−j0) × panel_w` f64 dgemm scratch during the Gram.
const PANEL_BYTES: usize = 256_000_000;
/// f64 block for streamed packed-matrix add/subtract from disk (64 MB).
const PACKED_IO_BLOCK: usize = 8_000_000;

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
    in_clip: (f32, f32),
    folds: usize,
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
    println!("Usage: fwls-diskbased NAME [-n | -p FILE] [-t FILE] [--groups G,...] --folds K (--seeds N,... | --seed N)");
    println!();
    println!("  Disk-based, memory-bound FWLS: packed AᵀA, per-fold Gram cached under");
    println!("  ./tmp/fwls-diskbased-<timestamp>/ and combined by differencing.");
    println!();
    println!("  NAME                       blend name; a preset (blend_config) or any ad-hoc name");
    println!("                             (ad-hoc requires --lambda). Output goes to NAME-s<seed>.*");
    println!("  -n, --new                  use pipeline-new.toml for [split]");
    println!("  -p FILE, --pipeline FILE   pipeline TOML (default: pipeline-old.toml)");
    println!("  -t FILE, --models FILE     base-predictor models TOML (default: models-new.toml)");
    println!("  --groups G,G,...           model groups (default: the TOML's `all`; omit with -m for manual-only)");
    println!("  -m NAME, --model NAME      add a single model (repeatable; combines with --groups; globs)");
    println!("  -x NAME, --exclude NAME    drop a model by name (repeatable; brace-expanded)");
    println!("  --voting-models FILE       voting-feature groups TOML (default: voting-new.toml)");
    println!("  --voting G,G,...           voting-feature groups from the TOML; optional if -f given");
    println!("  -f NAME, --feature NAME    add a single voting feature (repeatable; may be the only source)");
    println!("  --lambda VALUE             ridge λ (regularizes the full diagonal, bias included);");
    println!("                             overrides the preset, required for an ad-hoc name");
    println!("  --in-clip MIN,MAX          clip range for clipped model columns (default {CLIP_MIN},{CLIP_MAX})");
    println!("  --folds K                  K-fold cross-fit per seed (default 2); out-of-fold probe preds,");
    println!("                             qual predicted once from the full-data fit");
    println!("  --seeds N,N,...            fold seeds; one output NAME-s<N> per seed");
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
        model_manual: Vec::new(),
        exclude: Vec::new(),
        voting_models: "voting-new.toml".to_string(),
        voting: Vec::new(),
        feature_manual: Vec::new(),
        lambda: None,
        in_clip: (CLIP_MIN, CLIP_MAX),
        folds: 2,
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
            "--in-clip" => {
                let raw = need(&argv, i);
                let (lo, hi) = raw.split_once(',')
                    .unwrap_or_else(|| { eprintln!("error: --in-clip expects MIN,MAX (got '{raw}')"); std::process::exit(2) });
                a.in_clip = (
                    lo.trim().parse().expect("bad --in-clip MIN"),
                    hi.trim().parse().expect("bad --in-clip MAX"),
                );
                i += 2;
            }
            "-x" | "--exclude" => { a.exclude.push(need(&argv, i)); i += 2; }
            "--voting-models" => { a.voting_models = need(&argv, i); i += 2; }
            "--voting" => {
                for tok in need(&argv, i).split(',') {
                    a.voting.push(tok.trim().to_string());
                }
                i += 2;
            }
            "--folds" => {
                a.folds = need(&argv, i).parse().expect("bad --folds value");
                if a.folds < 2 { eprintln!("error: --folds must be >= 2"); std::process::exit(2); }
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
/// `i` to `[lo, hi]` when `clip[i]`. Columns are loaded in parallel.
fn load_cols(names: &[String], clip: &[bool], preds_dir: &str, ds: &str, n: usize, lo: f32, hi: f32) -> Vec<Vec<f32>> {
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
                    *x = x.clamp(lo, hi);
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
// Packed symmetric storage (lower triangle, column-major "L" layout)
//   index(i, j) for i >= j  =  col_start[j] + (i - j)
// ---------------------------------------------------------------------------

/// Column start offsets into the packed array, plus the packed length.
fn packed_layout(d: usize) -> (Vec<usize>, usize) {
    let mut col_start = vec![0usize; d];
    let mut off = 0usize;
    for j in 0..d {
        col_start[j] = off;
        off += d - j;
    }
    (col_start, off)
}

/// Dump a raw f64 slice to disk (native endianness; same machine round-trips).
fn write_packed(path: &str, v: &[f64]) {
    let mut f = BufWriter::new(File::create(path).unwrap_or_else(|e| panic!("create {path}: {e}")));
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    f.write_all(bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

/// Read a raw f64 file fully into `v` (must be pre-sized to the file's length).
fn read_packed_into(path: &str, v: &mut [f64]) {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) };
    f.read_exact(bytes).unwrap_or_else(|e| panic!("read {path}: {e}"));
}

/// Stream a raw f64 file in blocks and add `sign * file` into `v` in place —
/// so combining two D-sized matrices never needs a second D-sized buffer.
fn accumulate_from_file(path: &str, v: &mut [f64], sign: f64) {
    let mut f = BufReader::new(File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}")));
    let mut raw = vec![0u8; PACKED_IO_BLOCK * 8];
    let mut off = 0usize;
    while off < v.len() {
        let cnt = (v.len() - off).min(PACKED_IO_BLOCK);
        f.read_exact(&mut raw[..cnt * 8]).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let fv = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f64, cnt) };
        for (a, &x) in v[off..off + cnt].iter_mut().zip(fv) {
            *a += sign * x;
        }
        off += cnt;
    }
}

// ---------------------------------------------------------------------------
// FWLS Gram build (packed) / packed Cholesky / solve
// ---------------------------------------------------------------------------

/// Accumulate `AᵀA` (packed lower triangle) and `Aᵀy` for `rows`, where each
/// column of `A` is `flatten(X[:,r] ⊗ F[:,r])` plus a trailing 1.0 bias. The
/// Gram is built in row blocks; within a block a BLAS `dgemm` computes each
/// column panel of `ZZᵀ` (never a dense `D×D`) which is scattered into the
/// packed lower triangle. `ap`/`bvec` must be zeroed by the caller.
#[allow(clippy::too_many_arguments)]
fn build_chunk_packed(
    rows: &[usize],
    xpr: &[Vec<f32>],
    fpr: &[Vec<f32>],
    y: &[f64],
    m: usize,
    p: usize,
    d: usize,
    col_start: &[usize],
    ap: &mut [f64],
    bvec: &mut [f64],
) {
    let blen = (GRAM_BYTES / (8 * d)).clamp(64, rows.len().max(1));
    let panel_w = (PANEL_BYTES / (8 * d)).clamp(1, d);
    let mut z = vec![0.0f64; blen * d]; // column kk = one rating's D-vector
    let mut temp = vec![0.0f64; d * panel_w];

    let nb = rows.len().div_ceil(blen);
    let mut start = 0;
    let mut bi = 0;
    while start < rows.len() {
        let bl = (rows.len() - start).min(blen);
        for kk in 0..bl {
            let row = rows[start + kk];
            let base = kk * d;
            for i in 0..m {
                let xi = xpr[i][row] as f64;
                let ioff = base + i * p;
                for j in 0..p {
                    z[ioff + j] = xi * (fpr[j][row] as f64);
                }
            }
            z[base + m * p] = 1.0; // bias
        }

        // b += Z y
        for kk in 0..bl {
            let yk = y[rows[start + kk]];
            let col = &z[kk * d..kk * d + d];
            for (dd, &c) in col.iter().enumerate() {
                bvec[dd] += c * yk;
            }
        }

        // AᵀA += ZZᵀ, one column panel at a time. For panel columns [j0, j0+w)
        // the needed block is rows [j0, D): C = Z[j0:D,:] · Z[j0:j0+w,:]ᵀ.
        let mut j0 = 0;
        while j0 < d {
            let w = (d - j0).min(panel_w);
            let mrows = d - j0;
            unsafe {
                dgemm(
                    b'N', b'T',
                    mrows as i32, w as i32, bl as i32,
                    1.0, &z[j0..], d as i32,
                    &z[j0..], d as i32,
                    0.0, &mut temp[..mrows * w], mrows as i32,
                );
            }
            // Scatter the lower part (global i >= j, i.e. ii >= jj) into packed.
            for jj in 0..w {
                let cs = col_start[j0 + jj];
                let col = &temp[jj * mrows..jj * mrows + mrows];
                for ii in jj..mrows {
                    ap[cs + (ii - jj)] += col[ii];
                }
            }
            j0 += w;
        }

        start += bl;
        bi += 1;
        eprint!("\r    gram block {}/{}", bi, nb);
    }
    eprintln!();
}

/// In-place packed Cholesky `A = L Lᵀ` (Cholesky–Banachiewicz over the lower
/// triangle). Returns false if a non-positive pivot appears (not PD).
fn chol_packed(ap: &mut [f64], d: usize, col_start: &[usize]) -> bool {
    for i in 0..d {
        for j in 0..=i {
            let mut s = ap[col_start[j] + (i - j)];
            for k in 0..j {
                s -= ap[col_start[k] + (i - k)] * ap[col_start[k] + (j - k)];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                ap[col_start[i]] = s.sqrt();
            } else {
                ap[col_start[j] + (i - j)] = s / ap[col_start[j]];
            }
        }
    }
    true
}

/// Solve `A x = b` given the packed Cholesky factor `l` (L in packed lower):
/// forward-solve `L y = b`, then back-solve `Lᵀ x = y`.
fn solve_packed(l: &[f64], b: &[f64], d: usize, col_start: &[usize]) -> Vec<f64> {
    let mut y = vec![0.0f64; d];
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= l[col_start[k] + (i - k)] * y[k];
        }
        y[i] = s / l[col_start[i]];
    }
    let mut x = vec![0.0f64; d];
    for i in (0..d).rev() {
        let mut s = y[i];
        for k in (i + 1)..d {
            s -= l[col_start[i] + (k - i)] * x[k];
        }
        x[i] = s / l[col_start[i]];
    }
    x
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

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

/// Stream the qual set in row blocks and predict with `w` (clipped) into `out`.
#[allow(clippy::too_many_arguments)]
fn predict_qual(
    xr: &mut [NpyF32Reader],
    fr: &mut [NpyF32Reader],
    xclip: &[bool],
    xlo: f32,
    xhi: f32,
    w: &[f64],
    m: usize,
    p: usize,
    n_q: usize,
    out: &mut [f64],
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
                    *v = v.clamp(xlo, xhi);
                }
            }
        });
        fr.par_iter_mut().zip(fbuf.par_iter_mut()).for_each(|(r, buf)| {
            r.read_block(start, bl, &mut buf[..bl]);
        });

        out[start..start + bl].par_iter_mut().enumerate().for_each(|(k, a)| {
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
            *a = yhat.clamp(CLIP_MIN as f64, CLIP_MAX as f64);
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

/// Local timestamp `YYYYMMDD-HHMMSS` for the scratch directory (falls back to a
/// fixed name if `date` is unavailable).
fn timestamp() -> String {
    Command::new("date")
        .arg("+%Y%m%d-%H%M%S")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "run".to_string())
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

    let params = match blend_config(&args.name) {
        Some(mut p) => { if let Some(l) = args.lambda { p.lambda = l; } p }
        None => BlendParams {
            lambda: args.lambda.unwrap_or_else(|| {
                eprintln!("error: unknown blend '{}': provide --lambda VALUE", args.name);
                std::process::exit(2);
            }),
        },
    };

    let mut groups = if args.groups.is_empty() && !args.model_manual.is_empty() {
        indexmap::IndexMap::new()
    } else {
        select_groups(&load_models_toml(&args.models), &args.groups)
    };
    if !args.model_manual.is_empty() {
        groups.insert("manual".to_string(), expand_globs(&args.model_manual, &preds, &pr));
    }
    let flat = flatten_groups(&groups, &args.exclude);
    let (model_names, model_clip) = (flat.names, flat.clip);
    assert!(!model_names.is_empty(), "no models selected (use --groups or -m)");

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
    let (col_start, plen) = packed_layout(d);

    let models_manual_only = args.groups.is_empty() && !args.model_manual.is_empty();
    let models_src = if models_manual_only { "(manual)".to_string() } else { args.models.clone() };
    let groups_str = if args.groups.is_empty() {
        if args.model_manual.is_empty() { "all".to_string() } else { "manual".to_string() }
    } else { args.groups.join(",") };
    let voting_src = if args.voting.is_empty() { "(manual)".to_string() } else { args.voting_models.clone() };
    let voting_str = if args.voting.is_empty() { "manual".to_string() } else { args.voting.join(",") };
    let seeds_str = args.seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(",");

    let scratch = format!("./tmp/fwls-diskbased-{}", timestamp());
    std::fs::create_dir_all(&scratch).unwrap_or_else(|e| panic!("create {scratch}: {e}"));

    open_log(&preds, &args.name);
    teeln!("[{}] (disk-based)", args.name);
    teeln!("Pipeline:  {} (split = {})", args.pipeline, split_name);
    teeln!("Models:    {} ({} predictors, groups: {})", models_src, m, groups_str);
    if !args.exclude.is_empty() {
        teeln!("Excluded:  {} name(s): {}", args.exclude.len(), args.exclude.join(", "));
    }
    teeln!("Voting:    {} ({} context features, groups: {})", voting_src, p, voting_str);
    teeln!("Interact:  D = M·P + 1 = {}·{} + 1 = {}", m, p, d);
    teeln!("Packed:    {:.2} GB per matrix ({} f64, lower triangle)", (plen * 8) as f64 / 1e9, plen);
    teeln!("Lambda:    {} (regularizes full diagonal, bias included)", params.lambda);
    teeln!("In-clip:   [{}, {}] (clipped model columns)", args.in_clip.0, args.in_clip.1);
    teeln!("Folds:     {} per seed (out-of-fold probe; qual from full fit)", args.folds);
    teeln!("Seeds:     {}", seeds_str);
    teeln!("Scratch:   {}", scratch);
    log_columns(&model_names, &voting);
    teeln!();

    // Probe data: predictions + context features fully in memory; ratings f64.
    println!("Loading probe set ({})...", pr);
    let y_pr = load_ratings_f64(&pr);
    let n = y_pr.len();
    let no_clip = vec![false; p];
    let (xlo, xhi) = args.in_clip;
    let xpr = load_cols(&model_names, &model_clip, &preds, &pr, n, xlo, xhi);
    let fpr = load_cols(&voting, &no_clip, &preds, &pr, n, xlo, xhi);

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

    // The single D-sized working buffer (packed matrix), reused across every
    // phase so peak RAM is one packed triangle + the probe columns.
    let mut ap = vec![0.0f64; plen];
    let mut bvec = vec![0.0f64; d];
    let mut chunk_b = vec![0.0f64; d];

    for &seed in &args.seeds {
        teeln!();
        teeln!("=== seed {} ===", seed);

        let k_folds = args.folds;
        let mut idxs: Vec<usize> = (0..n).collect();
        idxs.shuffle(&mut StdRng::seed_from_u64(seed));
        let bounds: Vec<usize> = (0..=k_folds).map(|f| f * n / k_folds).collect();
        let folds: Vec<Vec<usize>> = (0..k_folds)
            .map(|f| idxs[bounds[f]..bounds[f + 1]].to_vec())
            .collect();

        let ata_path = |k: usize| format!("{scratch}/s{seed}_chunk{k}.ata");
        let b_path = |k: usize| format!("{scratch}/s{seed}_chunk{k}.b");
        let total_ata = format!("{scratch}/s{seed}_total.ata");
        let total_b = format!("{scratch}/s{seed}_total.b");

        // Phase 1: build each fold's Gram once and cache it to disk.
        for k in 0..k_folds {
            ap.iter_mut().for_each(|x| *x = 0.0);
            bvec.iter_mut().for_each(|x| *x = 0.0);
            teeln!("  chunk {}/{}: {} rows", k + 1, k_folds, folds[k].len());
            build_chunk_packed(&folds[k], &xpr, &fpr, &y_pr, m, p, d, &col_start, &mut ap, &mut bvec);
            write_packed(&ata_path(k), &ap);
            write_packed(&b_path(k), &bvec);
        }

        // Phase 2: total = Σ chunks (accumulated in the one buffer, then cached).
        ap.iter_mut().for_each(|x| *x = 0.0);
        bvec.iter_mut().for_each(|x| *x = 0.0);
        for k in 0..k_folds {
            accumulate_from_file(&ata_path(k), &mut ap, 1.0);
            accumulate_from_file(&b_path(k), &mut bvec, 1.0);
        }
        write_packed(&total_ata, &ap);
        write_packed(&total_b, &bvec);

        // Phase 3: out-of-fold probe. Train system for held-out fold n is
        // total − chunk_n, reconstructed in-place from disk.
        let mut yhat_pr = vec![0.0f64; n];
        for te in 0..k_folds {
            read_packed_into(&total_ata, &mut ap);
            accumulate_from_file(&ata_path(te), &mut ap, -1.0);
            read_packed_into(&total_b, &mut bvec);
            read_packed_into(&b_path(te), &mut chunk_b);
            for dd in 0..d {
                bvec[dd] -= chunk_b[dd];
            }
            for i in 0..d {
                ap[col_start[i]] += params.lambda; // regularize full diagonal (bias included)
            }
            teeln!("  fold {}/{}: fit {} rows, predict {}",
                   te + 1, k_folds, n - folds[te].len(), folds[te].len());
            if !chol_packed(&mut ap, d, &col_start) {
                panic!("fold {te}: training Gram not positive definite");
            }
            let w = solve_packed(&ap, &bvec, d, &col_start);
            let p_te = predict_rows(&folds[te], &xpr, &fpr, &w, m, p, true);
            teeln!("    fold RMSE {:.5}", rmse_sel(&p_te, &y_pr, &folds[te]));
            for (ii, &row) in folds[te].iter().enumerate() {
                yhat_pr[row] = p_te[ii];
            }
        }
        let probe_sse: f64 = yhat_pr.iter().zip(&y_pr).map(|(&h, &y)| (h - y) * (h - y)).sum();
        let probe_rmse = (probe_sse / n as f64).sqrt();
        teeln!(" ProbeRMSE: {:.5}", probe_rmse);

        // Qual: single fit from the full-data total.
        read_packed_into(&total_ata, &mut ap);
        read_packed_into(&total_b, &mut bvec);
        for i in 0..d {
            ap[col_start[i]] += params.lambda;
        }
        if !chol_packed(&mut ap, d, &col_start) {
            panic!("qual: full Gram not positive definite");
        }
        let w_full = solve_packed(&ap, &bvec, d, &col_start);

        let mut yhat_ql = vec![0.0f64; n_q];
        let mut xr = open_readers(&model_names, &preds);
        let mut fr = open_readers(&voting, &preds);
        predict_qual(&mut xr, &mut fr, &model_clip, xlo, xhi, &w_full, m, p, n_q, &mut yhat_ql);

        let mut quiz_sse = 0.0;
        for k in 0..n_q {
            if is_test[k] == 0 {
                let e = yhat_ql[k] - y_ql[k] as f64;
                quiz_sse += e * e;
            }
        }
        let quiz_rmse = (quiz_sse / quiz_n as f64).sqrt();
        teeln!("  QuizRMSE: {:.5}", quiz_rmse);

        let pr_path = format!("{preds}/{}-s{seed}.{pr}.npy", args.name);
        let ql_path = format!("{preds}/{}-s{seed}.{fulltrain_pr}.npy", args.name);
        save_preds(&pr_path, &Array1::from_iter(yhat_pr.iter().map(|&v| v as f32)));
        save_preds(&ql_path, &Array1::from_iter(yhat_ql.iter().map(|&v| v as f32)));
        teeln!("Saved {} / {}", pr_path, ql_path);
    }

    teeln!();
    teeln!("Gram cache left under {} ({} files)", scratch, args.seeds.len() * (2 * args.folds + 2));
    close_log();
    ExitCode::SUCCESS
}
