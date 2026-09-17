//! Linear blending of model predictions. Computes a single shared Gram
//! matrix over all listed models on the probe split, then solves a ridge
//! least-squares fit per group (and `all*` = every selected column) by slicing
//! submatrices. Probe and quiz RMSE are evaluated after clipping the
//! blended predictions to [CLIP_MIN, CLIP_MAX].

extern crate blas;
extern crate blas_src;

use blas::dsyrk;
use indexmap::IndexMap;
use flate2::read::GzDecoder;
use netflix_prize::blend::{flatten_groups, load_models_toml, permuted_folds, select_groups};
use netflix_prize::preds_path;
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use ndarray_npy::read_npy;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::process::ExitCode;

// Default clip bounds: inputs are clipped wide (mostly to tame outliers), the
// blended output is clipped tight to the rating scale.
const IN_CLIP_MIN: f64 = 0.0;
const IN_CLIP_MAX: f64 = 6.0;
const OUT_CLIP_MIN: f64 = 1.05;
const OUT_CLIP_MAX: f64 = 4.95;
// Quiz blending probes each column as its own leaderboard submission, so the
// default input clip is the legal rating range rather than the wide outlier clip.
const QUIZ_IN_CLIP_MIN: f64 = 1.0;
const QUIZ_IN_CLIP_MAX: f64 = 5.0;
const ROW_BLOCK: usize = 100_000;
const CV_SEED: u64 = 42;
const PIPELINE_OLD: &str = "pipeline-old.toml";
const PIPELINE_NEW: &str = "pipeline-new.toml";
const MODELS_OLD: &str = "models-old.toml";
const MODELS_NEW: &str = "models-new.toml";
const ENSEMBLES_TOML: &str = "ensembles.toml";
const QUAL_RATINGS_CSV_GZ: &str = "data/qual_ratings/qual_ratings.csv.gz";
const N_QUAL: usize = 2_817_131;
const ENSEMBLE_ROW: &str = "ensemble";

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

/// One predictor source: a split's pipeline (→ preds dir) plus the models TOML
/// and group/manual/exclude selection to pull from it. Single-split runs have
/// exactly one source; `--from` builds one per split for cross-split blending.
struct Source {
    label: String,               // "old" / "new" (split name, for reporting)
    pipeline: String,            // pipeline-<split>.toml
    models_toml: Option<String>, // None → default models-<label>.toml (cross-split) or manual-only (legacy)
    models_manual: Vec<String>,
    models_exclude: Vec<String>,
    groups: Vec<String>,
}

struct Args {
    sources: Vec<Source>,
    cross_split: bool, // true when --from was used (multi-source quiz blend)
    lambda: f64,
    forward: bool,
    max_features: Option<usize>,
    fixed_group: Option<String>,
    cv_folds: usize,
    cv_seed: u64,
    cv_patience: Option<usize>,
    in_clip_min: f64,
    in_clip_max: f64,
    out_clip_min: f64,
    out_clip_max: f64,
    quiz_blend: bool,
    raw_probes: bool,
    decimals: i32,
    /// The stored ensemble this run reproduces (`--ensemble`).
    ensemble: Option<EnsembleDef>,
    /// Columns measured on top of that ensemble (`-m` in ensemble mode).
    extras: Vec<String>,
    /// Calibration lines printed next to a measured delta.
    scale: Vec<String>,
    update_expected: bool,
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
    println!("    -g GRP1,GRP2,..., --groups  include only these TOML groups (default: the");
    println!("                             TOML's `all` group; e.g. -g integrated,rbm,other). A row");
    println!("                             is fitted per group, plus `all*` = every selected group");
    println!("                             and -m predictor together.");
    println!("    -m NAME, --model NAME    add a single model (repeatable; combines with -t)");
    println!("    -x NAME, --exclude NAME  drop a model by name (repeatable; applied after -t/-m)");
    println!();
    println!("    --lambda VALUE           ridge regularization λ (default 10)");
    println!("    --in-clip MIN,MAX        clip input predictions (skips '>' columns; default {IN_CLIP_MIN},{IN_CLIP_MAX})");
    println!("    --out-clip MIN,MAX       clip the blended output before RMSE (default {OUT_CLIP_MIN},{OUT_CLIP_MAX})");
    println!();
    println!("  Forward feature selection (Gram computed once, then submatrix slicing):");
    println!("    --forward                greedily add models by in-sample (Gram) RMSE — probe by");
    println!("                             default, the recovered qual system under --quiz-blend/--from");
    println!("    --max-features K         stop after K total selected features (incl. --fixed)");
    println!("    --fixed GROUP            pre-select all models in GROUP, search over the rest");
    println!("    --cv-folds K             select by K-fold CV RMSE instead, recovered from K");
    println!("                             per-fold Grams built in the same streaming pass");
    println!("    --cv-seed S              RNG seed for the fold assignment (default {CV_SEED})");
    println!("    --cv-patience P          stop after P steps without a CV improvement");
    println!();
    println!("  Quiz blending (fit on the qual labels recovered from RMSE probing):");
    println!("    --quiz-blend             build the Gram over the full qual set and recover");
    println!("                             Z'y from rounded per-model + constant RMSE probes");
    println!("    --decimals N             RMSE feedback precision for --quiz-blend (default 4)");
    println!("                             Combines with --forward: selection then costs no extra");
    println!("                             probes, and only the final prefix gets a clipped pass.");
    println!("                             Every column stands in for one submitted rating vector,");
    println!("                             so '>' (no-clip) columns are dropped and the input clip");
    println!("                             defaults to [{QUIZ_IN_CLIP_MIN}, {QUIZ_IN_CLIP_MAX}] instead of [{IN_CLIP_MIN}, {IN_CLIP_MAX}].");
    println!("    --raw-probes             opt out of both: keep '>' columns and probe them raw");
    println!();
    println!("  Cross-split quiz blending (combine qual.npy predictors from BOTH splits):");
    println!("    --from SPLIT             open a source scope for SPLIT (old|new); the models");
    println!("                             flags -t/-g/-m/-x after it apply to that source.");
    println!("                             -t defaults to models-<split>.toml. Repeat --from to");
    println!("                             mix splits. Implies --quiz-blend; the Gram is built");
    println!("                             over the full qual set from each source's preds dir.");
    println!("                             Example: ridge --quiz-blend --from old -g integrated \\");
    println!("                                            --from new -g integrated,rbm,other");
    println!();
    println!();
    println!("  Stored ensembles ({}):", ENSEMBLES_TOML);
    println!("    --ensemble [NAME]        run a stored blend and check it against the numbers");
    println!("                             recorded for it (default: the one marked default).");
    println!("                             With -m NAME it also fits the same blend WITH that");
    println!("                             column and reports the gain, so the reference is the");
    println!("                             control condition of the measurement. -m takes a bare");
    println!("                             name (looked up in the last source's preds dir) or a");
    println!("                             dir/name path, e.g. -m preds_lab/lab-foo.");
    println!("                             Exits non-zero if the reference does not reproduce.");
    println!("    --update-expected        write this run's numbers back into {}", ENSEMBLES_TOML);
    println!();
    println!("    -h, --help               show this help");
}

fn set_models_toml(dst: &mut Option<String>, path: String, flag: &str) {
    if dst.is_some() {
        eprintln!("error: '{}' conflicts with an earlier models TOML selection", flag);
        std::process::exit(2);
    }
    *dst = Some(path);
}

/// A fresh legacy (single-source) selection, defaulting to the old pipeline.
fn legacy_source() -> Source {
    Source {
        label: "old".to_string(),
        pipeline: PIPELINE_OLD.to_string(),
        models_toml: None,
        models_manual: Vec::new(),
        models_exclude: Vec::new(),
        groups: Vec::new(),
    }
}

fn parse_args() -> Args {
    let mut lambda = 10.0;
    let mut forward = false;
    let mut max_features: Option<usize> = None;
    let mut fixed_group: Option<String> = None;
    let mut cv_folds = 1usize;
    let mut cv_seed = CV_SEED;
    let mut cv_patience: Option<usize> = None;
    let (mut in_clip_min, mut in_clip_max) = (IN_CLIP_MIN, IN_CLIP_MAX);
    let (mut out_clip_min, mut out_clip_max) = (OUT_CLIP_MIN, OUT_CLIP_MAX);
    let mut quiz_blend = false;
    let mut raw_probes = false;
    let mut in_clip_set = false;
    let mut decimals = 4;
    let mut ensemble_name: Option<String> = None;
    let mut update_expected = false;

    // Legacy single-source accumulator (used when no --from is given), plus the
    // list of --from sources. The two are mutually exclusive.
    let mut legacy = legacy_source();
    let mut legacy_touched = false; // any top-level split/models flag seen?
    let mut from_sources: Vec<Source> = Vec::new();
    let mut using_from = false;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        // Selection flags (-t/-g/-m/-x) target the current --from source, or the
        // legacy accumulator when no --from is active.
        let flag = argv[i].as_str();
        match flag {
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            "--from" => {
                let s = need(&argv, i);
                let pipeline = match s.as_str() {
                    "old" => PIPELINE_OLD,
                    "new" => PIPELINE_NEW,
                    _ => { eprintln!("error: --from expects 'old' or 'new' (got '{}')", s); std::process::exit(2); }
                };
                if legacy_touched {
                    eprintln!("error: cannot mix top-level model/split flags with --from");
                    std::process::exit(2);
                }
                using_from = true;
                from_sources.push(Source {
                    label: s.clone(),
                    pipeline: pipeline.to_string(),
                    models_toml: None,
                    models_manual: Vec::new(),
                    models_exclude: Vec::new(),
                    groups: Vec::new(),
                });
                i += 2;
            }
            "-o" | "--old" | "-n" | "--new" | "-O" | "-N" | "-p" | "--pipeline" => {
                if using_from {
                    eprintln!("error: '{}' cannot be combined with --from", flag);
                    std::process::exit(2);
                }
                legacy_touched = true;
                match flag {
                    "-o" | "--old" => { legacy.pipeline = PIPELINE_OLD.to_string(); legacy.label = "old".into(); i += 1; }
                    "-n" | "--new" => { legacy.pipeline = PIPELINE_NEW.to_string(); legacy.label = "new".into(); i += 1; }
                    "-O" => {
                        legacy.pipeline = PIPELINE_OLD.to_string(); legacy.label = "old".into();
                        set_models_toml(&mut legacy.models_toml, MODELS_OLD.to_string(), "-O");
                        i += 1;
                    }
                    "-N" => {
                        legacy.pipeline = PIPELINE_NEW.to_string(); legacy.label = "new".into();
                        set_models_toml(&mut legacy.models_toml, MODELS_NEW.to_string(), "-N");
                        i += 1;
                    }
                    _ => { legacy.pipeline = need(&argv, i); i += 2; }
                }
            }
            "-t" | "--models" | "-g" | "--groups" | "-m" | "--model" | "-x" | "--exclude" => {
                let src = if using_from {
                    from_sources.last_mut().expect("--from source present")
                } else {
                    legacy_touched = true;
                    &mut legacy
                };
                match flag {
                    "-t" | "--models" => {
                        let path = need(&argv, i);
                        // In a --from scope -t overrides the split default; the
                        // conflict guard only applies to legacy -t/-O/-N clashes.
                        if using_from { src.models_toml = Some(path); }
                        else { set_models_toml(&mut src.models_toml, path, flag); }
                        i += 2;
                    }
                    "-g" | "--groups" => {
                        for tok in need(&argv, i).split(',') { src.groups.push(tok.trim().to_string()); }
                        i += 2;
                    }
                    "-m" | "--model" => { src.models_manual.push(need(&argv, i)); i += 2; }
                    _ => { src.models_exclude.push(need(&argv, i)); i += 2; }
                }
            }
            "--ensemble" => {
                // Optional value: `--ensemble` takes the default one, `--ensemble NAME`
                // the named one. Nothing else in this CLI is positional, so a bare
                // word after the flag is unambiguous.
                let name = match argv.get(i + 1) {
                    Some(a) if !a.starts_with('-') => { i += 1; a.clone() }
                    _ => String::new(),
                };
                ensemble_name = Some(name);
                i += 1;
            }
            "--update-expected" => { update_expected = true; i += 1; }
            "--lambda" => { lambda = need(&argv, i).parse().expect("bad --lambda value"); i += 2; }
            "--forward" => { forward = true; i += 1; }
            "--max-features" => {
                max_features = Some(need(&argv, i).parse().expect("bad --max-features value"));
                i += 2;
            }
            "--fixed" => { fixed_group = Some(need(&argv, i)); i += 2; }
            "--cv-folds" => { cv_folds = need(&argv, i).parse().expect("bad --cv-folds value"); i += 2; }
            "--cv-seed" => { cv_seed = need(&argv, i).parse().expect("bad --cv-seed value"); i += 2; }
            "--cv-patience" => {
                cv_patience = Some(need(&argv, i).parse().expect("bad --cv-patience value"));
                i += 2;
            }
            "--quiz-blend" => { quiz_blend = true; i += 1; }
            "--raw-probes" => { raw_probes = true; i += 1; }
            "--decimals" => { decimals = need(&argv, i).parse().expect("bad --decimals value"); i += 2; }
            "--in-clip" => {
                let (lo, hi) = parse_clip(&need(&argv, i), "--in-clip");
                in_clip_min = lo; in_clip_max = hi; in_clip_set = true; i += 2;
            }
            "--out-clip" => {
                let (lo, hi) = parse_clip(&need(&argv, i), "--out-clip");
                out_clip_min = lo; out_clip_max = hi; i += 2;
            }
            s => { eprintln!("error: unknown arg '{}'", s); print_help(); std::process::exit(2); }
        }
    }

    if !forward && (max_features.is_some() || fixed_group.is_some() || cv_folds > 1) {
        eprintln!("warning: --max-features/--fixed/--cv-folds have no effect without --forward");
    }
    if cv_folds < 1 {
        eprintln!("error: --cv-folds must be at least 1 (got {})", cv_folds);
        std::process::exit(2);
    }
    // The in-sample RMSE falls monotonically as columns are added, so there is
    // nothing for patience to wait for without a held-out criterion.
    if cv_patience.is_some() && cv_folds < 2 {
        eprintln!("error: --cv-patience needs --cv-folds >= 2");
        std::process::exit(2);
    }

    // `--ensemble` replaces the whole selection: sources, groups and lambda come
    // from ensembles.toml, and the only thing the caller adds is extra columns.
    let mut ensemble: Option<EnsembleDef> = None;
    let mut extras: Vec<String> = Vec::new();
    let mut scale: Vec<String> = Vec::new();
    if let Some(name) = &ensemble_name {
        let rejected: &[(&str, bool)] = &[
            ("--from", using_from),
            ("-t/-g/-x", legacy.models_toml.is_some() || !legacy.groups.is_empty()
                || !legacy.models_exclude.is_empty()),
            ("--forward", forward),
            ("--quiz-blend", quiz_blend),
            ("--cv-folds", cv_folds > 1),
        ];
        if let Some((flag, _)) = rejected.iter().find(|(_, hit)| *hit) {
            eprintln!("error: '{}' cannot be combined with --ensemble", flag);
            eprintln!("       --ensemble runs a stored blend verbatim; only -m adds to it");
            std::process::exit(2);
        }
        let file = load_ensembles();
        let def = pick_ensemble(&file, name);
        if def.source.is_empty() {
            eprintln!("error: ensemble '{}' lists no sources", def.name);
            std::process::exit(2);
        }
        extras = std::mem::take(&mut legacy.models_manual);
        scale = file.scale.lines.clone();
        lambda = def.lambda;
        using_from = def.source.len() > 1;
        if using_from { quiz_blend = true; }
        from_sources = def.source.iter().map(|src| Source {
            label: src.split.clone(),
            pipeline: format!("pipeline-{}.toml", src.split),
            models_toml: Some(format!("models-{}.toml", src.split)),
            models_manual: src.models.clone(),
            models_exclude: Vec::new(),
            groups: src.groups.clone(),
        }).collect();
        // A single-source ensemble is an ordinary one-split run, so it keeps the
        // unprefixed group names and fits on the probe labels.
        if !using_from {
            legacy = from_sources.remove(0);
        }
        ensemble = Some(def);
    } else if update_expected {
        eprintln!("error: --update-expected needs --ensemble");
        std::process::exit(2);
    }

    let sources = if using_from {
        // Cross-split blending always fits on qual → enable quiz-blend implicitly.
        quiz_blend = true;
        from_sources
    } else {
        if legacy.models_toml.is_none() && legacy.models_manual.is_empty() {
            eprintln!("error: provide -N/-O, -t MODELS_TOML, and/or -m NAME (or --from SPLIT ...)");
            std::process::exit(2);
        }
        if !legacy.groups.is_empty() && legacy.models_toml.is_none() {
            eprintln!("error: -g/--groups requires a models TOML (-t/-N/-O)");
            std::process::exit(2);
        }
        vec![legacy]
    };

    // A quiz-blend column stands in for a submitted rating vector, so unless the
    // caller overrides it the input clip is the legal rating range.
    if quiz_blend && !raw_probes && !in_clip_set {
        in_clip_min = QUIZ_IN_CLIP_MIN;
        in_clip_max = QUIZ_IN_CLIP_MAX;
    }
    if raw_probes && !quiz_blend {
        eprintln!("warning: --raw-probes has no effect without --quiz-blend/--from");
    }

    // Quiz-blend recovers b from RMSEs published over the whole qual set, which
    // cannot be reproduced fold by fold, so the CV criterion is unavailable there.
    if quiz_blend && cv_folds > 1 {
        eprintln!("error: --cv-folds cannot be combined with --quiz-blend/--from");
        std::process::exit(2);
    }

    Args {
        sources,
        cross_split: using_from,
        lambda,
        forward,
        max_features,
        fixed_group,
        cv_folds,
        cv_seed,
        cv_patience,
        in_clip_min,
        in_clip_max,
        out_clip_min,
        out_clip_max,
        quiz_blend,
        raw_probes,
        decimals,
        ensemble,
        extras,
        scale,
        update_expected,
    }
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
// Named ensembles (ensembles.toml)
// ---------------------------------------------------------------------------

/// One stored ensemble: the sources it blends and the numbers a complete
/// prediction set reproduces. `--ensemble` runs it as a check; `--ensemble -m
/// NAME` measures what an extra column does to it, against the same reference
/// in the same run.
#[derive(serde::Deserialize, Clone)]
struct EnsembleDef {
    name: String,
    #[serde(default, rename = "default")]
    is_default: bool,
    #[serde(default)]
    description: String,
    lambda: f64,
    #[serde(default = "default_tolerance")]
    tolerance: f64,
    #[serde(default)]
    source: Vec<EnsembleSourceDef>,
    /// Row name → the RMSEs it is expected to produce.
    #[serde(default)]
    expected: IndexMap<String, ExpectedRow>,
}

#[derive(serde::Deserialize, Clone)]
struct EnsembleSourceDef {
    split: String,
    #[serde(default)]
    groups: Vec<String>,
    #[serde(default)]
    models: Vec<String>,
}

#[derive(serde::Deserialize, Clone, Copy, Default)]
struct ExpectedRow {
    #[serde(default)]
    probe: Option<f64>,
    #[serde(default)]
    quiz: Option<f64>,
}

#[derive(serde::Deserialize, Default)]
struct ScaleNote {
    #[serde(default)]
    lines: Vec<String>,
}

#[derive(serde::Deserialize)]
struct EnsembleFile {
    #[serde(default)]
    ensemble: Vec<EnsembleDef>,
    #[serde(default)]
    scale: ScaleNote,
}

fn default_tolerance() -> f64 { 1e-5 }

fn load_ensembles() -> EnsembleFile {
    let s = std::fs::read_to_string(ENSEMBLES_TOML).unwrap_or_else(|e| {
        eprintln!("error: read {}: {}", ENSEMBLES_TOML, e);
        std::process::exit(2);
    });
    toml::from_str(&s).unwrap_or_else(|e| {
        eprintln!("error: parse {}: {}", ENSEMBLES_TOML, e);
        std::process::exit(2);
    })
}

/// The named ensemble, or the one marked `default = true`.
fn pick_ensemble(file: &EnsembleFile, name: &str) -> EnsembleDef {
    let found = if name.is_empty() {
        file.ensemble.iter().find(|e| e.is_default).or_else(|| file.ensemble.first())
    } else {
        file.ensemble.iter().find(|e| e.name == name)
    };
    match found {
        Some(e) => e.clone(),
        None => {
            let names: Vec<&str> = file.ensemble.iter().map(|e| e.name.as_str()).collect();
            eprintln!("error: no ensemble '{}' in {} (have: {})",
                name, ENSEMBLES_TOML, names.join(", "));
            std::process::exit(2);
        }
    }
}

/// Qual labels. The `.npy` arrays when `ingest` has run, otherwise the
/// `rating,is_test` CSV that ships with the repo: `ingest` writes one straight
/// from the other, row for row, so a blend can be scored on a clone that has
/// never downloaded the dataset.
fn load_qual_labels(qual: &str) -> (Array1<i8>, Array1<i8>) {
    let y_path = format!("data/{}/ratings.npy", qual);
    let t_path = format!("data/{}/is_test.npy", qual);
    if std::path::Path::new(&y_path).exists() && std::path::Path::new(&t_path).exists() {
        let y: Array1<i8> = read_npy(&y_path).unwrap_or_else(|e| panic!("read {}: {}", y_path, e));
        let t: Array1<i8> = read_npy(&t_path).unwrap_or_else(|e| panic!("read {}: {}", t_path, e));
        return (y, t);
    }
    if qual != "qual" || !std::path::Path::new(QUAL_RATINGS_CSV_GZ).exists() {
        eprintln!("error: {} not found, and no fallback for dataset '{}'", y_path, qual);
        eprintln!("       run: ./target/release/run -n ingest");
        std::process::exit(2);
    }
    println!("Qual labels: {} (data/{}/ not ingested)", QUAL_RATINGS_CSV_GZ, qual);
    let f = File::open(QUAL_RATINGS_CSV_GZ)
        .unwrap_or_else(|e| panic!("open {}: {}", QUAL_RATINGS_CSV_GZ, e));
    let mut y: Vec<i8> = Vec::with_capacity(N_QUAL);
    let mut t: Vec<i8> = Vec::with_capacity(N_QUAL);
    for (i, line) in BufReader::new(GzDecoder::new(f)).lines().enumerate() {
        let line = line.unwrap_or_else(|e| panic!("read {}: {}", QUAL_RATINGS_CSV_GZ, e));
        if i == 0 { continue; } // rating,is_test
        let (r, is_test) = line.split_once(',')
            .unwrap_or_else(|| panic!("{}: bad row {}", QUAL_RATINGS_CSV_GZ, i));
        y.push(r.trim().parse().expect("bad rating"));
        t.push(is_test.trim().parse().expect("bad is_test"));
    }
    (Array1::from(y), Array1::from(t))
}

/// Prediction files a run needs but does not have, as `(column, path)` pairs.
fn missing_columns(
    display: &[String],
    unique: &[String],
    preds_dirs: &[String],
    datasets: &[&str],
) -> Vec<(String, String)> {
    let mut missing = Vec::new();
    for ds in datasets {
        for ((name, dir), shown) in unique.iter().zip(preds_dirs.iter()).zip(display.iter()) {
            let path = preds_path(dir, name, ds);
            if !std::path::Path::new(&path).exists() {
                missing.push((shown.clone(), path));
            }
        }
    }
    missing
}

/// Report missing prediction files and how to get them, then exit. A column the
/// caller added is their own to produce; a stored one is a download away.
fn abort_on_missing(missing: &[(String, String)], total: usize, added_only: bool) -> ! {
    eprintln!();
    eprintln!("error: {} of {} prediction files are missing, for example:", missing.len(), total);
    for (name, path) in missing.iter().take(5) {
        eprintln!("         {} → {}", name, path);
    }
    eprintln!();
    if added_only {
        eprintln!("       These are columns given with -m, not part of the ensemble. Check the");
        eprintln!("       name, and that the run producing them has finished writing.");
    } else {
        eprintln!("       The stored ensembles are blends of the project's own predictions.");
        eprintln!("       Fetch them with:");
        eprintln!("         cargo build --release --bin preds");
        eprintln!("         ./target/release/preds pull 'preds_*/*.qual.npy'   # the qual columns");
        eprintln!("       (or ./target/release/preds pull for everything, including the probe sets)");
    }
    std::process::exit(2);
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

/// Build the merged model registry across all sources: unique names, clip flags,
/// per-model preds dir, and group → indices. In cross-split mode group keys are
/// prefixed with the source label (`old/integrated`); a single legacy source
/// keeps bare group names. Dedup is per-source, so the same predictor name in
/// two splits yields two distinct columns from two preds dirs.
fn build_registry(
    args: &Args,
) -> (Vec<String>, Vec<bool>, Vec<String>, Vec<String>, IndexMap<String, Vec<usize>>) {
    let mut names: Vec<String> = Vec::new();
    let mut clip: Vec<bool> = Vec::new();
    let mut preds_dirs: Vec<String> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut group_indices: IndexMap<String, Vec<usize>> = IndexMap::new();

    for src in &args.sources {
        let preds_dir = load_pipeline_split(&src.pipeline)
            .get("preds")
            .unwrap_or_else(|| panic!("{}: [split].preds missing", src.pipeline))
            .clone();

        // In a --from scope -t defaults to models-<label>.toml; a legacy
        // manual-only source keeps no models TOML.
        let models_toml: Option<String> = src.models_toml.clone().or_else(|| {
            if args.cross_split {
                Some(match src.label.as_str() {
                    "new" => MODELS_NEW.to_string(),
                    _ => MODELS_OLD.to_string(),
                })
            } else {
                None
            }
        });

        let mut groups: IndexMap<String, Vec<String>> = if let Some(p) = &models_toml {
            select_groups(&load_models_toml(p), &src.groups)
        } else {
            IndexMap::new()
        };
        if !src.models_manual.is_empty() {
            groups.insert("manual".to_string(), src.models_manual.clone());
        }

        // Flatten groups → (name, clip, group→indices), applying --exclude. Each
        // exclude arg is brace-expanded and '>'-stripped before matching; models
        // are deduplicated by clip-stripped name with any '>' winning.
        let flat = flatten_groups(&groups, &src.models_exclude);
        let offset = names.len();
        for (nm, cl) in flat.names.iter().zip(flat.clip.iter()) {
            names.push(nm.clone());
            clip.push(*cl);
            preds_dirs.push(preds_dir.clone());
            labels.push(src.label.clone());
        }
        for (gname, idxs) in &flat.group_indices {
            let key = if args.cross_split { format!("{}/{}", src.label, gname) } else { gname.clone() };
            let shifted: Vec<usize> = idxs.iter().map(|&x| x + offset).collect();
            group_indices.insert(key, shifted);
        }
    }

    (names, clip, preds_dirs, labels, group_indices)
}

/// Drop the `>` (no-clip) columns from a flattened registry, reindexing the
/// groups and discarding any that end up empty. Quiz blending recovers each
/// column's `Zᵀy` from the RMSE of that column submitted on its own, and a raw
/// factor/bias component is not a rating vector anyone could have submitted.
/// Returns the dropped names, in registry order.
fn drop_noclip(
    names: &mut Vec<String>,
    clip: &mut Vec<bool>,
    preds_dirs: &mut Vec<String>,
    labels: &mut Vec<String>,
    group_indices: &mut IndexMap<String, Vec<usize>>,
) -> Vec<String> {
    let keep = clip.clone();
    if keep.iter().all(|&k| k) {
        return Vec::new();
    }
    let dropped: Vec<String> = names
        .iter()
        .zip(keep.iter())
        .filter(|&(_, &k)| !k)
        .map(|(n, _)| n.clone())
        .collect();

    // Old index → new index, for the columns that survive.
    let mut remap = vec![usize::MAX; keep.len()];
    let mut next = 0usize;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            remap[i] = next;
            next += 1;
        }
    }

    let mut it = keep.iter();
    names.retain(|_| *it.next().expect("keep covers names"));
    let mut it = keep.iter();
    clip.retain(|_| *it.next().expect("keep covers clip"));
    let mut it = keep.iter();
    preds_dirs.retain(|_| *it.next().expect("keep covers preds_dirs"));
    let mut it = keep.iter();
    labels.retain(|_| *it.next().expect("keep covers labels"));

    group_indices.retain(|_, idxs| {
        idxs.retain(|&i| remap[i] != usize::MAX);
        for i in idxs.iter_mut() {
            *i = remap[*i];
        }
        !idxs.is_empty()
    });

    dropped
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

/// Fold index per probe row, from a seeded random permutation. `k == 1` skips
/// the shuffle: every row is in the single fold anyway, and the shuffle would
/// only cost an allocation the size of the probe set.
fn assign_folds(n: usize, k: usize, seed: u64) -> Vec<u32> {
    let mut fold_of = vec![0u32; n];
    if k > 1 {
        for (f, rows) in permuted_folds(n, k, seed).iter().enumerate() {
            for &r in rows {
                fold_of[r] = f as u32;
            }
        }
    }
    fold_of
}

/// One fold's normal-equation system, restricted to that fold's rows:
///   A = Zᵀ Z   ((m+1)×(m+1), row-major, last col/row is bias)
///   b = Zᵀ y   (m+1)
/// Z has shape (n × (m+1)) with the last column all-ones for the bias.
/// A single-fold run holds the whole probe set, which is the non-CV case.
struct GramFold {
    a: Vec<f64>,
    b: Vec<f64>,
    yty: f64,
    n: usize,
}

impl GramFold {
    fn zeros(dim: usize) -> Self {
        GramFold { a: vec![0.0; dim * dim], b: vec![0.0; dim], yty: 0.0, n: 0 }
    }

    /// Element-wise sum, i.e. the system over the union of the folds' rows.
    fn sum(folds: &[GramFold], dim: usize) -> GramFold {
        let mut t = GramFold::zeros(dim);
        for f in folds {
            for i in 0..dim * dim { t.a[i] += f.a[i]; }
            for i in 0..dim { t.b[i] += f.b[i]; }
            t.yty += f.yty;
            t.n += f.n;
        }
        t
    }
}

/// Build one Gram system per CV fold in a single streaming pass, with row `i`
/// contributing to fold `fold_of[i]`. Inside a row block the rows of one fold
/// are gathered into the scratch buffer before its `dsyrk`, so the flop count is
/// the same as for a single Gram and the extra cost is one pass of index checks
/// per fold. With `k_folds == 1` the gather is the identity and this reduces to
/// the plain single-Gram build, bit for bit.
fn build_grams(
    y: &[f64],
    readers: &mut [NpyF32Reader],
    clip: &[bool],
    in_clip_min: f32,
    in_clip_max: f32,
    fold_of: &[u32],
    k_folds: usize,
) -> Vec<GramFold> {
    let m = readers.len();
    let dim = m + 1;
    let n = y.len();
    let mut folds: Vec<GramFold> = (0..k_folds).map(|_| GramFold::zeros(dim)).collect();

    // yᵀy and the row count come from a separate pass in row order, so that a
    // single-fold run reproduces the plain `y.iter().map(v*v).sum()` exactly.
    for (i, &yi) in y.iter().enumerate() {
        let f = &mut folds[fold_of[i] as usize];
        f.yty += yi * yi;
        f.n += 1;
    }

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

        // One pass per fold: gather that fold's rows of the block into the front
        // of the f64 scratch (casting on the way), then one dsyrk over just
        // those columns. `zt_f64` is reused across folds, so the block memory is
        // the same as for a single Gram.
        for (f, fold) in folds.iter_mut().enumerate() {
            let mut cnt = 0usize;
            for k in 0..blen {
                if k_folds > 1 && fold_of[start + k] as usize != f {
                    continue;
                }
                let src = &zt_f32[k * dim..(k + 1) * dim];
                let dst = &mut zt_f64[cnt * dim..(cnt + 1) * dim];
                for i in 0..dim {
                    dst[i] = src[i] as f64;
                }
                // b += Zᵀ y, one row at a time while the row is at hand.
                let y_k = y[start + k];
                for i in 0..dim {
                    fold.b[i] += dst[i] * y_k;
                }
                cnt += 1;
            }
            if cnt == 0 {
                continue;
            }
            // dsyrk uplo='L' fills the column-major lower triangle, which is the
            // row-major upper triangle in our linear buffer; the post-loop mirror
            // copies that into the row-major lower half.
            unsafe {
                dsyrk(
                    b'L',
                    b'N',
                    dim as i32,
                    cnt as i32,
                    1.0,
                    &zt_f64,
                    dim as i32,
                    1.0,
                    &mut fold.a,
                    dim as i32,
                );
            }
        }

        start += blen;
        bidx += 1;
        eprint!("\r  gram block {}/{}", bidx, n_blocks);
    }
    eprintln!();

    // Mirror row-major upper → row-major lower so downstream row-major
    // submatrix slicing reads a fully symmetric matrix.
    for fold in folds.iter_mut() {
        for i in 0..dim {
            for j in (i + 1)..dim {
                fold.a[j * dim + i] = fold.a[i * dim + j];
            }
        }
    }
    folds
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
/// entries (the bias term stays unregularized) and Cholesky-solve. `None` when
/// the regularized system is not positive definite.
fn try_solve_subsystem(
    a_sub: &DMatrix<f64>,
    b_sub: &DVector<f64>,
    n_feat: usize,
    lambda: f64,
) -> Option<DVector<f64>> {
    let mut a_reg = a_sub.clone();
    for i in 0..n_feat {
        a_reg[(i, i)] += lambda;
    }
    Some(a_reg.cholesky()?.solve(b_sub))
}

fn solve_subsystem(
    a_sub: &DMatrix<f64>,
    b_sub: &DVector<f64>,
    n_feat: usize,
    lambda: f64,
) -> DVector<f64> {
    try_solve_subsystem(a_sub, b_sub, n_feat, lambda)
        .expect("Gram matrix is not positive definite")
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

/// One forward-selection step: the model added, and the in-sample and K-fold CV
/// probe RMSE of the resulting prefix, both recovered purely from the Grams.
/// Without CV the two values are the same number.
struct ForwardStep {
    added: String,
    in_sample_rmse: f64,
    cv_rmse: f64,
}

/// SSE of the fit `w` against an unregularized Gram system:
///   SSE = yᵀy − 2·wᵀ·b + wᵀ·A·w
fn sse_of(w: &DVector<f64>, a: &DMatrix<f64>, b: &DVector<f64>, yty: f64) -> f64 {
    yty - 2.0 * w.dot(b) + w.dot(&(a * w))
}

/// Solve the ridge subsystem for `indices` over the pooled system and return
/// `(w, in_sample_rmse, cv_rmse)`. Both RMSEs are the unclipped probe RMSE
/// recovered from the Grams alone, with `A_sub` taken WITHOUT the λ ridge term.
///
/// Fold k is fitted on `A − A_k` and scored on its own held-out `A_k`. The
/// subtraction is exact arithmetic on the accumulated systems and is safe
/// because the folds are a random row split, so `A_k ≈ A/K` and no cancellation
/// occurs. A singular fold complement scores the whole candidate as `+∞`, which
/// simply drops it from the greedy step.
fn eval_subset(
    a: &[f64],
    b: &[f64],
    yty: f64,
    dim: usize,
    indices: &[usize],
    lambda: f64,
    n: usize,
    folds: &[GramFold],
) -> (DVector<f64>, f64, f64) {
    let (a_sub, b_sub) = build_subsystem(a, b, dim, indices);
    let w = solve_subsystem(&a_sub, &b_sub, indices.len(), lambda);
    let in_rmse = (sse_of(&w, &a_sub, &b_sub, yty) / n as f64).sqrt();
    if folds.len() < 2 {
        return (w, in_rmse, in_rmse);
    }
    let mut sse_cv = 0.0f64;
    for f in folds {
        let (ak, bk) = build_subsystem(&f.a, &f.b, dim, indices);
        let Some(wk) =
            try_solve_subsystem(&(&a_sub - &ak), &(&b_sub - &bk), indices.len(), lambda)
        else {
            return (w, in_rmse, f64::INFINITY);
        };
        sse_cv += sse_of(&wk, &ak, &bk, f.yty);
    }
    (w, in_rmse, (sse_cv / n as f64).sqrt())
}

/// Greedy forward selection over the Gram. The criterion is the K-fold CV probe
/// RMSE when `folds` holds more than one fold, and the in-sample probe RMSE
/// otherwise; the two coincide at K = 1, so the single-fold path is exactly the
/// pre-CV behaviour. The weights returned per step are always the fit over all
/// the rows: cross-validation decides *which* columns to take, never the
/// coefficients that are finally used.
///
/// `fixed` is pre-selected and never dropped. Returns one prefix fit per step
/// (for downstream clipped probe/quiz evaluation) plus per-step metadata.
/// Candidate evaluation within each step runs in parallel over Rayon.
#[allow(clippy::too_many_arguments)]
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
    folds: &[GramFold],
    cv_patience: Option<usize>,
) -> (Vec<(String, Vec<usize>, DVector<f64>)>, Vec<ForwardStep>) {
    let cv = folds.len() > 1;
    let mut in_fixed = vec![false; m];
    for &i in fixed {
        in_fixed[i] = true;
    }
    let mut selected: Vec<usize> = fixed.to_vec();
    let mut remaining: Vec<usize> = (0..m).filter(|&i| !in_fixed[i]).collect();

    let mut fits: Vec<(String, Vec<usize>, DVector<f64>)> = Vec::new();
    let mut steps: Vec<ForwardStep> = Vec::new();

    let target = max_features.unwrap_or(m).min(m);

    // The score a step is judged by, and how long it has been since the best.
    let mut best_cv = f64::INFINITY;
    let mut since_best = 0usize;
    let mut note_step = |cv_rmse: f64| {
        if cv_rmse < best_cv {
            best_cv = cv_rmse;
            since_best = 0;
        } else {
            since_best += 1;
        }
        matches!(cv_patience, Some(p) if since_best >= p)
    };

    // Baseline row for the pre-selected (fixed) set, if any.
    if !selected.is_empty() {
        let (w, rmse, cv_rmse) = eval_subset(a, b, yty, dim, &selected, lambda, n, folds);
        if cv {
            eprintln!(
                "  baseline ({} fixed): in-sample {:.6}  cv {:.6}",
                selected.len(), rmse, cv_rmse,
            );
        } else {
            eprintln!("  baseline ({} fixed): in-sample {:.6}", selected.len(), rmse);
        }
        steps.push(ForwardStep {
            added: format!("<baseline: {} fixed>", selected.len()),
            in_sample_rmse: rmse,
            cv_rmse,
        });
        fits.push(("base".to_string(), selected.clone(), w));
        note_step(cv_rmse);
    }

    while !remaining.is_empty() && selected.len() < target {
        let (pos, cand, rmse, cv_rmse, w) = remaining
            .par_iter()
            .enumerate()
            .map(|(pos, &c)| {
                let mut trial = selected.clone();
                trial.push(c);
                let (w, rmse, cv_rmse) = eval_subset(a, b, yty, dim, &trial, lambda, n, folds);
                (pos, c, rmse, cv_rmse, w)
            })
            .min_by(|x, y| {
                let (sx, sy) = if cv { (x.3, y.3) } else { (x.2, y.2) };
                sx.partial_cmp(&sy).expect("NaN RMSE in candidate eval")
            })
            .expect("non-empty remaining");

        remaining.remove(pos);
        selected.push(cand);
        if cv {
            eprintln!(
                "  step {}/{}: + {} (in-sample {:.6}  cv {:.6})",
                selected.len(), target, unique[cand], rmse, cv_rmse,
            );
        } else {
            eprintln!(
                "  step {}/{}: + {} (in-sample {:.6})",
                selected.len(), target, unique[cand], rmse,
            );
        }
        steps.push(ForwardStep {
            added: unique[cand].clone(),
            in_sample_rmse: rmse,
            cv_rmse,
        });
        fits.push((format!("k={}", selected.len()), selected.clone(), w));
        if note_step(cv_rmse) {
            eprintln!(
                "  stopping: {} steps without a CV improvement",
                cv_patience.expect("patience fired"),
            );
            break;
        }
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
fn recover_quiz_b(
    a: &[f64],
    b_true: &[f64],
    yty: f64,
    n: usize,
    m: usize,
    decimals: i32,
) -> (Vec<f64>, f64) {
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
    (b, yty_rec)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let args = parse_args();

    let (mut unique, mut clip, mut preds_dirs, mut labels, mut group_indices) =
        build_registry(&args);
    let dropped_noclip = if args.quiz_blend && !args.raw_probes {
        drop_noclip(&mut unique, &mut clip, &mut preds_dirs, &mut labels, &mut group_indices)
    } else {
        Vec::new()
    };
    // Columns measured on top of the ensemble. They are appended after the stored
    // registry (and after the no-clip drop, which concerns stored columns only),
    // so the reference fit stays the prefix `0..ref_m` whatever is added, and a
    // bare name resolves in the last source's preds dir while `dir/name` names
    // its own directory.
    let ref_m = unique.len();
    for spec in &args.extras {
        let no_clip = spec.starts_with('>');
        if no_clip && args.quiz_blend {
            eprintln!("error: '{}' is a no-clip column, and a quiz blend fits only on", spec);
            eprintln!("       columns that could have been submitted as ratings");
            return ExitCode::from(2);
        }
        preds_dirs.push(preds_dirs.last().cloned().unwrap_or_default());
        labels.push(labels.last().cloned().unwrap_or_default());
        unique.push(spec.trim_start_matches('>').to_string());
        clip.push(!no_clip);
    }

    let m = unique.len();
    if m == 0 {
        eprintln!("error: no models left after exclusion");
        return ExitCode::from(2);
    }

    // Column names as reported. A cross-split registry can hold the same model
    // name once per split (dedup is per source), so qualify it there; a column
    // that names its own directory is already unambiguous.
    let display: Vec<String> = if args.cross_split {
        unique.iter().zip(labels.iter())
            .map(|(n, l)| if n.contains('/') { n.clone() } else { format!("{}/{}", l, n) })
            .collect()
    } else {
        unique.clone()
    };


    // The qual dataset name is shared across splits (both pipelines set
    // fulltrain_pr = "qual"); read it from the first source's pipeline.
    let split0 = load_pipeline_split(&args.sources[0].pipeline);
    let qual = split0.get("fulltrain_pr").expect("pipeline [split].fulltrain_pr missing").clone();

    // --- Header ---
    if args.cross_split {
        println!("Mode:      cross-split quiz blend ({} sources)", args.sources.len());
        for src in &args.sources {
            let mt = src.models_toml.clone().unwrap_or_else(|| match src.label.as_str() {
                "new" => MODELS_NEW.to_string(),
                _ => MODELS_OLD.to_string(),
            });
            let g = if src.groups.is_empty() { "all".to_string() } else { src.groups.join(",") };
            let man = if src.models_manual.is_empty() { String::new() }
                      else { format!(" +{} manual", src.models_manual.len()) };
            let exc = if src.models_exclude.is_empty() { String::new() }
                      else { format!(", excl {}", src.models_exclude.len()) };
            println!("  from {:<3}  {} (groups: {}{}{})", src.label, mt, g, man, exc);
        }
        println!("Models:    {} unique columns across {} group(s)", m, group_indices.len());
    } else {
        let src = &args.sources[0];
        let split_name = split0.get("name").cloned().unwrap_or_else(|| "?".to_string());
        println!("Pipeline:  {} (split = {})", src.pipeline, split_name);
        match (&src.models_toml, src.models_manual.len()) {
            (Some(t), 0) => println!("Models:    {} ({} unique across {} group(s))",
                t, m, group_indices.len()),
            (Some(t), k) => println!("Models:    {} + {} manual ({} unique across {} group(s))",
                t, k, m, group_indices.len()),
            (None, k) => println!("Models:    {} manual model(s)", k),
        }
        if src.models_toml.is_some() {
            let g = if src.groups.is_empty() { "all".to_string() } else { src.groups.join(",") };
            println!("Groups:    {}", g);
        }
        if !src.models_exclude.is_empty() {
            println!("Excluded:  {} name(s): {}",
                src.models_exclude.len(), src.models_exclude.join(", "));
        }
    }
    if let Some(def) = &args.ensemble {
        println!("Ensemble:  {} ({}){}", def.name, ENSEMBLES_TOML,
            if def.description.is_empty() { String::new() } else { format!(" - {}", def.description) });
        if m > ref_m {
            println!("Added:     {}", display[ref_m..].join(", "));
        }
    }
    println!("Lambda λ:  {}", args.lambda);
    if args.quiz_blend && !args.raw_probes {
        println!("In-clip:   [{}, {}] (applied to every column)",
            args.in_clip_min, args.in_clip_max);
        if !dropped_noclip.is_empty() {
            println!("No-clip:   dropped {} '>' column(s) — not submittable as ratings:",
                dropped_noclip.len());
            for chunk in dropped_noclip.chunks(3) {
                println!("           {}", chunk.join(", "));
            }
        }
    } else {
        println!("In-clip:   [{}, {}] (skips '>' columns)", args.in_clip_min, args.in_clip_max);
    }
    println!("Out-clip:  [{}, {}]", args.out_clip_min, args.out_clip_max);
    print_blas_info();
    println!();

    // Forward selection on a quiz-blend Gram: the whole run is qual-only, so the
    // probe set is neither fitted nor scored and stays unopened.
    let fwd_quiz = args.forward && args.quiz_blend;

    // Load probe ratings + open probe readers — legacy single-split only; a
    // cross-split blend has no shared probe set, so it fits/evaluates on qual.
    let mut readers: Vec<NpyF32Reader> = Vec::new();
    let mut probe_y_i8: Option<Array1<i8>> = None;
    let mut probe_y: Vec<f64> = Vec::new();
    let mut n_probe = 0usize;
    let need_probe = !args.cross_split && !fwd_quiz;
    let pr_name: Option<String> = if need_probe {
        Some(load_pipeline_split(&args.sources[0].pipeline)
            .get("pr").expect("pipeline [split].pr missing").clone())
    } else {
        None
    };

    // Preflight. A column that was never downloaded is a fetch problem, so say
    // which ones and how to get them rather than failing on the first open. A
    // missing qual column is fatal only where the fit itself lives on qual; a
    // plain probe run can still report everything but the quiz number, which is
    // what an experiment that has not run its `--final` phase looks like.
    let added_only = |missing: &[(String, String)]| {
        missing.iter().all(|(name, _)| display[ref_m..].contains(name))
    };
    if let Some(pr) = &pr_name {
        let missing = missing_columns(&display, &unique, &preds_dirs, &[pr]);
        if !missing.is_empty() {
            abort_on_missing(&missing, m, added_only(&missing));
        }
    }
    let missing_qual = missing_columns(&display, &unique, &preds_dirs, &[&qual]);
    let have_qual = missing_qual.is_empty();
    if !have_qual && (args.quiz_blend || args.forward) {
        abort_on_missing(&missing_qual, m, added_only(&missing_qual));
    }

    if let Some(pr) = &pr_name {
        let y_path = format!("data/{}/ratings.npy", pr);
        let y_i8: Array1<i8> = read_npy(&y_path).unwrap_or_else(|e| panic!("read {}: {}", y_path, e));
        probe_y = y_i8.iter().map(|&r| r as f64).collect();
        n_probe = y_i8.len();
        println!("Probe set: {} ratings ({})", n_probe, y_path);
        for (name, dir) in unique.iter().zip(preds_dirs.iter()) {
            let path = preds_path(dir, name, pr);
            let r = NpyF32Reader::open(&path);
            assert_eq!(r.len, n_probe, "{}: length {} != probe {}", path, r.len, n_probe);
            readers.push(r);
        }
        probe_y_i8 = Some(y_i8);
    }

    // Load qual ratings + is_test (for the quiz evaluation pass; in quiz-blend
    // mode the Gram is also built here, over the full qual set).
    let (y_q_i8, is_test_q) = if have_qual {
        load_qual_labels(&qual)
    } else {
        (Array1::<i8>::zeros(0), Array1::<i8>::zeros(0))
    };
    let n_q = y_q_i8.len();
    let quiz_n_expected = is_test_q.iter().filter(|&&t| t == 0).count();
    if have_qual {
        println!("Quiz set:  {} of {} qual ratings", quiz_n_expected, n_q);
    } else {
        println!("Quiz set:  skipped, {} column(s) have no {} predictions:",
            missing_qual.len(), qual);
        for (name, _) in missing_qual.iter().take(3) {
            println!("           {}", name);
        }
        println!("           re-run that model with --final to get its quiz number");
    }

    // Open one partial qual reader per unique model, from its own preds dir.
    let mut qual_readers: Vec<NpyF32Reader> = Vec::with_capacity(m);
    if have_qual {
        for (name, dir) in unique.iter().zip(preds_dirs.iter()) {
            let path = preds_path(dir, name, &qual);
            let r = NpyF32Reader::open(&path);
            assert_eq!(r.len, n_q, "{}: length {} != qual {}", path, r.len, n_q);
            qual_readers.push(r);
        }
    }

    // Build the shared Gram (A = ZᵀZ), its right-hand side (b = Zᵀy) and yty.
    // Normal mode fits on the probe labels; quiz-blend mode fits on the full
    // qual labels recovered from rounded per-model + constant RMSE probes.
    //
    // With --cv-folds K the probe rows are split into K folds and one system is
    // accumulated per fold; `folds` then feeds the CV criterion and their sum is
    // the same pooled system a non-CV run would have built. Quiz-blend never
    // splits: its `b` is recovered from RMSEs published over the whole quiz set,
    // which cannot be reproduced fold by fold. That is moot in practice, since
    // --quiz-blend and --from are both rejected alongside --forward.
    let dim = m + 1;
    let k_folds = if args.forward && !args.quiz_blend { args.cv_folds } else { 1 };
    // The exact (Zᵀy, yᵀy) the recovery approximates. Diagnostics only: it feeds
    // the control column of the forward table, never the selection itself.
    let mut quiz_truth: Option<(Vec<f64>, f64)> = None;
    let (folds, a, b, yty, n): (Vec<GramFold>, Vec<f64>, Vec<f64>, f64, usize) = if args.quiz_blend {
        let y_q: Vec<f64> = y_q_i8.iter().map(|&r| r as f64).collect();
        println!(
            "Quiz-blend: building Gram over full qual ({} ratings), decimals = {}",
            n_q, args.decimals,
        );
        let one = vec![0u32; n_q];
        let g = build_grams(&y_q, &mut qual_readers, &clip,
            args.in_clip_min as f32, args.in_clip_max as f32, &one, 1);
        let a = g.into_iter().next().expect("one fold");
        let yty: f64 = y_q.iter().map(|v| v * v).sum();
        // `yty_rec` (not the true yᵀy) is what goes on: every number the forward
        // criterion sees then comes from the rounded RMSE feedback alone.
        let (b, yty_rec) = recover_quiz_b(&a.a, &a.b, yty, n_q, m, args.decimals);
        quiz_truth = Some((a.b, yty));
        (Vec::new(), a.a, b, yty_rec, n_q)
    } else {
        println!("Building Gram matrix over {} model(s) × {} ratings...", m, n_probe);
        let fold_of = assign_folds(n_probe, k_folds, args.cv_seed);
        if k_folds > 1 {
            println!("CV:        {} folds, seed {}", k_folds, args.cv_seed);
        }
        let mut folds = build_grams(&probe_y, &mut readers, &clip,
            args.in_clip_min as f32, args.in_clip_max as f32, &fold_of, k_folds);
        // Without CV the single fold *is* the pooled system, so move it out
        // rather than summing a one-element list back into a fresh allocation.
        let total = if folds.len() == 1 {
            folds.pop().expect("one fold")
        } else {
            GramFold::sum(&folds, dim)
        };
        (folds, total.a, total.b, total.yty, total.n)
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
                "Forward:   λ={} candidates={} fixed={} max_features={}{}",
                args.lambda,
                m - fixed.len(),
                fixed.len(),
                args.max_features.map(|k| k.to_string()).unwrap_or_else(|| "all".to_string()),
                args.cv_patience.map(|p| format!(" patience={p}")).unwrap_or_default(),
            );
            if fwd_quiz {
                println!(
                    "Probes:    {} RMSE submissions (one per model + 2 constants); the search itself",
                    m + 2,
                );
                println!("           needs none — every step is Gram algebra over the recovered system.");
            }
            println!();
            println!("Phase 1/2: forward selection — at each step greedily add the predictor");
            if k_folds > 1 {
                println!("           that most lowers the {}-fold CV (Gram-only, unclipped) probe RMSE.", k_folds);
            } else if fwd_quiz {
                println!("           that most lowers the in-sample (Gram-only, unclipped) qual RMSE,");
                println!("           taken over the recovered Zᵀy / yᵀy alone.");
            } else {
                println!("           that most lowers the in-sample (Gram-only, unclipped) probe RMSE.");
            }
            let (fits, steps) = forward_select(
                &a, &b, yty, dim, n, m, &display, &fixed, args.max_features, args.lambda,
                &folds, args.cv_patience,
            );
            (fits, Some(steps))
        } else {
            // Solve per group, plus 'all' if more than one group
            let mut fits: Vec<(String, Vec<usize>, DVector<f64>)> = Vec::new();
            for (gname, gidxs) in &group_indices {
                let w = solve_group(&a, &b, dim, gidxs, args.lambda);
                fits.push((gname.clone(), gidxs.clone(), w));
            }
            // Everything in the registry: the selected groups plus any -m
            // predictors. Named `all*` so it is not read as an `all` TOML group —
            // a helper group left out of that meta is absent here too.
            if args.ensemble.is_some() {
                // The reference fit, and the same fit with the added columns.
                // Both are slices of the one Gram, so measuring an addition costs
                // one extra column in the streaming pass, not a second run.
                let ref_idxs: Vec<usize> = (0..ref_m).collect();
                let w = solve_group(&a, &b, dim, &ref_idxs, args.lambda);
                fits.push((ENSEMBLE_ROW.to_string(), ref_idxs, w));
                if m > ref_m {
                    let all_idxs: Vec<usize> = (0..m).collect();
                    let w = solve_group(&a, &b, dim, &all_idxs, args.lambda);
                    fits.push((format!("{} + {}", ENSEMBLE_ROW, display[ref_m..].join(", ")),
                               all_idxs, w));
                }
            } else if group_indices.len() > 1 {
                let all_idxs: Vec<usize> = (0..m).collect();
                let w = solve_group(&a, &b, dim, &all_idxs, args.lambda);
                fits.push(("all*".to_string(), all_idxs, w));
            }
            (fits, None)
        };

    // Control column: the selected weights re-scored against the exact Zᵀy / yᵀy.
    // The gap to the criterion is what the rounded-RMSE recovery costs per step.
    let insample_true: Option<Vec<f64>> = match (args.forward, &quiz_truth) {
        (true, Some((b_true, yty_true))) => Some(
            fits.iter()
                .map(|(_, gidxs, w)| {
                    let (a_sub, b_sub) = build_subsystem(&a, b_true, dim, gidxs);
                    (sse_of(w, &a_sub, &b_sub, *yty_true) / n as f64).sqrt()
                })
                .collect(),
        ),
        _ => None,
    };

    // Which fits reach the streaming clipped pass. A quiz-blend forward run
    // scores only its final prefix: the per-step curve is the Gram criterion,
    // and the clipped number is an after-the-fact diagnostic, not a selector.
    let scored: &[(String, Vec<usize>, DVector<f64>)] = if fwd_quiz {
        &fits[fits.len().saturating_sub(1)..]
    } else {
        &fits
    };

    if fwd_steps.is_some() {
        println!();
        if fwd_quiz {
            println!("Phase 2/2: one streaming qual pass for the final prefix only ({} columns).",
                scored.last().map(|f| f.1.len()).unwrap_or(0));
        } else {
            println!("Phase 2/2: streaming probe + qual passes to score the actual clipped");
            println!("           probe & quiz RMSE for every selected prefix ({} fits).", fits.len());
        }
    }

    // Second pass on probe: compute clipped RMSE per fit (legacy single-split
    // only; a cross-split blend has no shared probe, so this is skipped).
    let (probe_sse, probe_n, _, _) = if !need_probe {
        (vec![0.0f64; scored.len()], 0usize, Vec::new(), 0usize)
    } else {
        compute_clipped_sse(
            &mut readers, &clip,
            args.in_clip_min as f32, args.in_clip_max as f32, args.out_clip_min, args.out_clip_max,
            probe_y_i8.as_ref().expect("probe labels loaded in legacy mode"),
            None, false, scored, m, "probe",
        )
    };

    // Third pass on qual: compute clipped quiz RMSE per fit (mask via is_test).
    // y_q_i8 / is_test_q / qual_readers were loaded above; reuse them (the
    // readers re-seek on every block, so quiz-blend's Gram pass left them usable).
    // The ensemble report also splits off the qual rows the quiz mask excludes:
    // the weights are fitted against quiz feedback, so that half is held out.
    let (quiz_sse, quiz_n, test_sse, test_n) = if have_qual {
        compute_clipped_sse(
            &mut qual_readers, &clip,
            args.in_clip_min as f32, args.in_clip_max as f32, args.out_clip_min, args.out_clip_max,
            &y_q_i8, Some(&is_test_q), args.ensemble.is_some(), scored, m, "quiz",
        )
    } else {
        (vec![0.0f64; scored.len()], 0usize, vec![0.0f64; scored.len()], 0usize)
    };

    // Residual correlation of each added column with the reference blend, over
    // the rows the delta was measured on. This is the triage number: a column
    // pays when it is wrong in different places, not when it is merely accurate.
    let extra_corr: Vec<f64> = match (&args.ensemble, fits.iter().position(|(n, _, _)| n == ENSEMBLE_ROW)) {
        (Some(_), Some(ens_i)) if m > ref_m => {
            let fit = &fits[ens_i];
            (ref_m..m).map(|j| {
                if need_probe {
                    residual_correlation(
                        &mut readers, &clip, args.in_clip_min as f32, args.in_clip_max as f32,
                        args.out_clip_min, args.out_clip_max,
                        probe_y_i8.as_ref().expect("probe labels"), None, fit, j, m,
                    )
                } else {
                    residual_correlation(
                        &mut qual_readers, &clip, args.in_clip_min as f32, args.in_clip_max as f32,
                        args.out_clip_min, args.out_clip_max,
                        &y_q_i8, Some(&is_test_q), fit, j, m,
                    )
                }
            }).collect()
        }
        _ => Vec::new(),
    };

    let mut check_failed = false;
    println!();
    match &fwd_steps {
        Some(steps) if fwd_quiz => {
            // Qual-only run: no probe column exists, and the clipped quiz number
            // is a single after-the-fact line rather than a per-step column.
            // `insample_true` re-scores the same weights against the exact Zᵀy /
            // yᵀy, so `delta_true` shows which steps only fitted recovery noise.
            let truth = insample_true.as_ref().expect("quiz-blend keeps the exact system");
            println!(
                "{:>4}  {:<44} {:>13} {:>13} {:>10} {:>10}",
                "step", "model added", "insample_qual", "insample_true", "delta", "delta_true",
            );
            let (mut prev, mut prev_t) = (f64::INFINITY, f64::INFINITY);
            let mut best_true = (0usize, f64::INFINITY);
            let mut noise_steps = 0usize;
            for (i, (step, &t_rmse)) in steps.iter().zip(truth.iter()).enumerate() {
                let delta = if prev.is_finite() { step.in_sample_rmse - prev } else { 0.0 };
                let delta_t = if prev_t.is_finite() { t_rmse - prev_t } else { 0.0 };
                prev = step.in_sample_rmse;
                prev_t = t_rmse;
                if t_rmse < best_true.1 {
                    best_true = (i + 1, t_rmse);
                }
                if delta < 0.0 && delta_t > 0.0 {
                    noise_steps += 1;
                }
                println!(
                    "{:>4}  {:<44} {:>13.6} {:>13.6} {:>+10.6} {:>+10.6}",
                    i + 1, step.added, step.in_sample_rmse, t_rmse, delta, delta_t,
                );
            }
            println!();
            println!(
                "Recovery noise: {} of {} steps lowered the criterion while raising the true fit.",
                noise_steps, steps.len(),
            );
            println!(
                "Best true in-sample: step {} ({:.6}) — diagnostic, not available to the search.",
                best_true.0, best_true.1,
            );
            if let (Some((_, gidxs, _)), Some(q_sse)) = (scored.last(), quiz_sse.last()) {
                println!(
                    "Final prefix ({} columns): clipped quiz RMSE {:.6} (scored on the true labels).",
                    gidxs.len(), (q_sse / quiz_n as f64).sqrt(),
                );
            }
        }
        Some(steps) => {
            // With CV the extra column is the criterion the selection actually
            // used; `delta` tracks whichever of the two that was.
            let cv = k_folds > 1;
            let score = |s: &ForwardStep| if cv { s.cv_rmse } else { s.in_sample_rmse };
            if cv {
                println!(
                    "{:>4}  {:<40} {:>13} {:>11} {:>11} {:>11}  {:>9}",
                    "step", "model added", "insample_pr", "cv_pr", "clip_probe", "quiz", "delta",
                );
            } else {
                println!(
                    "{:>4}  {:<40} {:>13} {:>11} {:>11}  {:>9}",
                    "step", "model added", "insample_pr", "clip_probe", "quiz", "delta",
                );
            }
            let mut prev = f64::INFINITY;
            let mut best = (0usize, f64::INFINITY);
            for (i, (step, (p_sse, q_sse))) in
                steps.iter().zip(probe_sse.iter().zip(quiz_sse.iter())).enumerate()
            {
                let p_rmse = (p_sse / probe_n as f64).sqrt();
                let q_rmse = (q_sse / quiz_n as f64).sqrt();
                let delta = if prev.is_finite() { score(step) - prev } else { 0.0 };
                prev = score(step);
                if score(step) < best.1 {
                    best = (i + 1, score(step));
                }
                if cv {
                    println!(
                        "{:>4}  {:<40} {:>13.6} {:>11.6} {:>11.6} {:>11.6}  {:>+9.6}",
                        i + 1, step.added, step.in_sample_rmse, step.cv_rmse, p_rmse, q_rmse, delta,
                    );
                } else {
                    println!(
                        "{:>4}  {:<40} {:>13.6} {:>11.6} {:>11.6}  {:>+9.6}",
                        i + 1, step.added, step.in_sample_rmse, p_rmse, q_rmse, delta,
                    );
                }
            }
            if cv {
                println!();
                println!("Best by {}-fold CV: step {} ({:.6})", k_folds, best.0, best.1);
            }
        }
        None if args.ensemble.is_some() => {
            let def = args.ensemble.as_ref().expect("ensemble mode");
            let val = |sse: &[f64], n: usize, i: usize| -> Option<f64> {
                if n == 0 { None } else { Some((sse[i] / n as f64).sqrt()) }
            };
            let fmt = |v: Option<f64>| -> String {
                v.map_or_else(|| "-".to_string(), |x| format!("{:.6}", x))
            };
            let w = fits.iter().map(|(n, _, _)| n.len()).max().unwrap_or(20).clamp(20, 56);

            if need_probe {
                println!("{:<w$} {:>7} {:>10} {:>10} {:>10} {:>10} {:>11}",
                    "row", "models", "probe", "quiz", "test", "expected", "delta", w = w);
            } else {
                println!("{:<w$} {:>7} {:>10} {:>10} {:>10} {:>11}",
                    "row", "models", "quiz", "test", "expected", "delta", w = w);
            }

            let mut checked = 0usize;
            let mut details: Vec<String> = Vec::new();
            let mut measured: Vec<(String, Option<f64>, Option<f64>)> = Vec::new();
            let ens_primary = fits.iter().position(|(n, _, _)| n == ENSEMBLE_ROW)
                .and_then(|i| if need_probe { val(&probe_sse, probe_n, i) } else { val(&quiz_sse, quiz_n, i) });

            for (i, (name, gidxs, _w)) in fits.iter().enumerate() {
                let p = val(&probe_sse, probe_n, i);
                let q = val(&quiz_sse, quiz_n, i);
                let t = val(&test_sse, test_n, i);
                let primary = if need_probe { p } else { q };
                measured.push((name.clone(), p, q));

                // Every metric the file pins down is checked; the column shows
                // the one the row is fitted on, which is the one that moves.
                let exp = def.expected.get(name);
                let mut status = String::new();
                let mut delta: Option<f64> = None;
                match exp {
                    Some(e) => {
                        let pairs = [("probe", p, e.probe), ("quiz", q, e.quiz)];
                        let mut row_ok = true;
                        let mut any = false;
                        for (label, got, want) in pairs {
                            if let (Some(g), Some(wv)) = (got, want) {
                                any = true;
                                if (g - wv).abs() > def.tolerance {
                                    row_ok = false;
                                    details.push(format!(
                                        "  {}: {} {:.6} vs expected {:.6} ({:+.1e})",
                                        name, label, g, wv, g - wv));
                                }
                            }
                        }
                        if any {
                            checked += 1;
                            status = if row_ok { "OK".to_string() } else { "FAIL".to_string() };
                            if !row_ok { check_failed = true; }
                        }
                        let want_primary = if need_probe { e.probe } else { e.quiz };
                        delta = match (primary, want_primary) {
                            (Some(g), Some(wv)) => Some(g - wv),
                            _ => None,
                        };
                    }
                    // The added-column row has nothing to reproduce: its number
                    // is the gain over the reference in the same run.
                    None if name.starts_with(ENSEMBLE_ROW) => {
                        delta = match (primary, ens_primary) {
                            (Some(g), Some(r)) => Some(g - r),
                            _ => None,
                        };
                    }
                    None => {}
                }
                let d = delta.map_or_else(|| "-".to_string(), |x| format!("{:+.2e}", x));
                let e_str = exp.and_then(|e| if need_probe { e.probe } else { e.quiz });
                if need_probe {
                    println!("{:<w$} {:>7} {:>10} {:>10} {:>10} {:>10} {:>11}  {}",
                        name, gidxs.len(), fmt(p), fmt(q), fmt(t), fmt(e_str), d, status, w = w);
                } else {
                    println!("{:<w$} {:>7} {:>10} {:>10} {:>10} {:>11}  {}",
                        name, gidxs.len(), fmt(q), fmt(t), fmt(e_str), d, status, w = w);
                }
            }

            if !details.is_empty() {
                println!();
                println!("Rows outside the {:.0e} tolerance:", def.tolerance);
                for d in &details { println!("{}", d); }
            }

            if m > ref_m {
                println!();
                println!("Added column(s), against the same reference in the same fit:");
                for (k, j) in (ref_m..m).enumerate() {
                    let corr = extra_corr.get(k).copied();
                    println!("  {:<40} residual corr {}", display[j],
                        corr.map_or_else(|| "-".to_string(), |c| format!("{:.4}", c)));
                }
                for line in &args.scale {
                    println!("  scale: {}", line);
                }
                if !have_qual {
                    println!("  (probe only: no qual predictions, so the quiz number is unmeasured)");
                }
            }

            println!();
            if checked == 0 {
                println!("Reference check: no expected values in {} for '{}'", ENSEMBLES_TOML, def.name);
            } else if check_failed {
                println!("Reference check: FAIL ({} of {} rows outside {:.0e})",
                    details.len(), checked, def.tolerance);
                println!("  A single group means those columns changed; many means a stale");
                println!("  download. ./target/release/preds pull --dry-run re-checks every md5.");
            } else {
                println!("Reference check: PASS ({} rows within {:.0e})", checked, def.tolerance);
            }

            if args.update_expected {
                update_expected(&def.name, &measured, need_probe, have_qual);
            }
        }
        None if args.cross_split => {
            // Cross-split: only the quiz RMSE is meaningful (no shared probe).
            println!("{:<24} {:>8} {:>14}", "source/group", "models", "quiz_rmse");
            for ((name, gidxs, _w), q_sse) in fits.iter().zip(quiz_sse.iter()) {
                let q_rmse = (q_sse / quiz_n as f64).sqrt();
                println!("{:<24} {:>8} {:>14.6}", name, gidxs.len(), q_rmse);
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

    if check_failed { return ExitCode::from(1); }
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
    split_mask: bool,
    fits: &[(String, Vec<usize>, DVector<f64>)],
    m: usize,
    label: &str,
) -> (Vec<f64>, usize, Vec<f64>, usize) {
    let n = y.len();
    let n_fits = fits.len();
    let dim = m + 1;
    let mut sse = vec![0.0f64; n_fits];
    let mut count = 0usize;
    // The masked-out rows, accumulated separately when `split_mask` is set and
    // skipped outright otherwise (the extra arithmetic is wasted on a run that
    // never reports them).
    let mut sse_held = vec![0.0f64; n_fits];
    let mut count_held = 0usize;

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
            let mut acc_held = 0.0f64;
            for k in 0..blen {
                let held = mask.is_some_and(|mk| mk[start + k] != 0);
                if held && !split_mask { continue; }
                let mut yhat = bias;
                for (jj, &gi) in gidxs.iter().enumerate() {
                    yhat += w[jj] * zt_f32[gi + k * dim] as f64;
                }
                let yh = yhat.clamp(out_clip_min, out_clip_max);
                let err = yh - y[start + k] as f64;
                if held { acc_held += err * err; } else { acc += err * err; }
            }
            sse[fi] += acc;
            sse_held[fi] += acc_held;
        }

        for k in 0..blen {
            if mask.is_some_and(|mk| mk[start + k] != 0) { count_held += 1; } else { count += 1; }
        }

        start += blen;
        bidx += 1;
        eprint!("\r  {} block {}/{}", label, bidx, n_blocks);
    }
    eprintln!();

    (sse, count, sse_held, if split_mask { count_held } else { 0 })
}

/// Pearson correlation between an added column's residuals and the reference
/// blend's, over the scored rows. The project's rule of thumb is that a column
/// pays when it is wrong in different places at comparable accuracy, so this
/// number triages a new model long before a full blend does.
#[allow(clippy::too_many_arguments)]
fn residual_correlation(
    readers: &mut [NpyF32Reader],
    clip: &[bool],
    in_clip_min: f32,
    in_clip_max: f32,
    out_clip_min: f64,
    out_clip_max: f64,
    y: &Array1<i8>,
    mask: Option<&Array1<i8>>,
    fit: &(String, Vec<usize>, DVector<f64>),
    col: usize,
    m: usize,
) -> f64 {
    let n = y.len();
    let dim = m + 1;
    let (_, gidxs, w) = fit;
    let bias = w[gidxs.len()];
    let (mut cnt, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

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
        for k in 0..blen {
            if mask.is_some_and(|mk| mk[start + k] != 0) { continue; }
            let mut yhat = bias;
            for (jj, &gi) in gidxs.iter().enumerate() {
                yhat += w[jj] * zt_f32[gi + k * dim] as f64;
            }
            let yv = y[start + k] as f64;
            let x = yv - yhat.clamp(out_clip_min, out_clip_max);
            let z = yv - zt_f32[col + k * dim] as f64;
            cnt += 1.0;
            sx += x;
            sy += z;
            sxx += x * x;
            syy += z * z;
            sxy += x * z;
        }
        start += blen;
        bidx += 1;
        eprint!("\r  corr block {}/{}", bidx, n_blocks);
    }
    eprintln!();

    let cov = sxy - sx * sy / cnt;
    let vx = sxx - sx * sx / cnt;
    let vy = syy - sy * sy / cnt;
    if vx <= 0.0 || vy <= 0.0 { return f64::NAN; }
    cov / (vx * vy).sqrt()
}

/// Rewrite one ensemble's `expected` table with the numbers this run measured.
/// Comments and every other table are preserved, so the commit diff shows
/// exactly how far the reference moved and nothing else.
fn update_expected(
    name: &str,
    rows: &[(String, Option<f64>, Option<f64>)],
    keep_probe: bool,
    keep_quiz: bool,
) {
    let text = std::fs::read_to_string(ENSEMBLES_TOML)
        .unwrap_or_else(|e| panic!("read {}: {}", ENSEMBLES_TOML, e));
    let mut doc: toml_edit::DocumentMut = text
        .parse()
        .unwrap_or_else(|e| panic!("parse {}: {}", ENSEMBLES_TOML, e));
    let arr = doc
        .get_mut("ensemble")
        .and_then(|i| i.as_array_of_tables_mut())
        .unwrap_or_else(|| panic!("{}: no [[ensemble]] tables", ENSEMBLES_TOML));
    let Some(t) = arr.iter_mut().find(|t| t.get("name").and_then(|v| v.as_str()) == Some(name))
    else {
        panic!("{}: no ensemble '{}'", ENSEMBLES_TOML, name);
    };

    let mut tbl = toml_edit::Table::new();
    let mut n = 0usize;
    for (row, p, q) in rows {
        // The added-column row is a measurement of something that is not in the
        // ensemble, so there is nothing to pin down.
        if row.contains(" + ") { continue; }
        // Written as text and parsed back, so the numbers keep the six decimals
        // the hand-written rows use and an update diff shows only what moved.
        let mut parts: Vec<String> = Vec::new();
        if let (true, Some(v)) = (keep_probe, p) { parts.push(format!("probe = {:.6}", v)); }
        if let (true, Some(v)) = (keep_quiz, q) { parts.push(format!("quiz = {:.6}", v)); }
        if parts.is_empty() { continue; }
        let value: toml_edit::Value = format!("{{ {} }}", parts.join(", "))
            .parse()
            .expect("inline table built here is valid TOML");
        tbl.insert(row, toml_edit::Item::Value(value));
        n += 1;
    }
    t.insert("expected", toml_edit::Item::Table(tbl));
    std::fs::write(ENSEMBLES_TOML, doc.to_string())
        .unwrap_or_else(|e| panic!("write {}: {}", ENSEMBLES_TOML, e));
    println!("Updated {} rows of [expected] for '{}' in {}", n, name, ENSEMBLES_TOML);
}
