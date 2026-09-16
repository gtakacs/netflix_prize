//! Support for one-file experiments in `src/bin/lab-*.rs`: a sandbox `Split`
//! that writes into `preds_lab/`, env-overridable knobs, a small argument
//! parser and user subsampling for smoke runs. See docs/EXPERIMENTS.md.

use crate::{Dataset, Split};
use ndarray::Array1;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};

/// Preds directory every lab experiment writes into. No manifest references
/// it, so `preds push` never uploads it and `run -c -f` never prunes it, and
/// dropping an experiment is `rm src/bin/lab-foo.rs preds_lab/lab-foo.*`.
pub const LAB_PREDS_DIR: &str = "preds_lab";

/// Manifest a lab run takes its datasets from when `-p` is absent.
pub const DEFAULT_MANIFEST: &str = "pipeline-new.toml";

/// Fraction of users a `--smoke` run keeps.
const SMOKE_SAMPLE: f64 = 0.02;

/// Users kept per mille; 1000 (the default) means every dataset loads whole.
static KEEP_PERMILLE: AtomicU32 = AtomicU32::new(1000);

/// Tuning knob from the environment, so a knob can be re-tried without editing
/// the source: `LR=0.01 cargo run --release --bin lab-foo`.
pub fn ev<T: FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Command line of a lab binary.
#[derive(Debug, Clone)]
pub struct LabArgs {
    /// Model name, and with it the name of every file written: the binary's own
    /// name, or the positional argument. A subsampled run gets a `-smoke`
    /// suffix so it cannot overwrite the column of a real run.
    pub name: String,
    /// Pipeline manifest the datasets come from.
    pub manifest: String,
    /// Target spec for `Dataset::load`: "rtg", or "0.5*dnn-24" to train on what
    /// another model left over.
    pub target: String,
    /// `--final`: also run the fulltrain -> qual phase, needed only once the
    /// column has earned a place in the blend.
    pub final_run: bool,
    /// Fraction of users kept; 1.0 = all of them.
    pub sample: f64,
}

impl LabArgs {
    pub fn parse() -> Self {
        let argv: Vec<String> = std::env::args().collect();
        let mut name = exe_name();
        let mut manifest = DEFAULT_MANIFEST.to_string();
        let mut target = "rtg".to_string();
        let mut final_run = false;
        let mut sample = 1.0;

        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "-h" | "--help" => { print_help(&name); std::process::exit(0); }
                "-p" | "--pipeline" => { manifest = need(&argv, i); i += 2; }
                "--target" => { target = need(&argv, i); i += 2; }
                "--final" => { final_run = true; i += 1; }
                "--smoke" => { sample = SMOKE_SAMPLE; i += 1; }
                "--sample" => {
                    sample = need(&argv, i).parse().expect("bad --sample value");
                    assert!(sample > 0.0 && sample <= 1.0, "--sample must be in (0, 1]");
                    i += 2;
                }
                a if !a.starts_with('-') => { name = a.to_string(); i += 1; }
                a => {
                    eprintln!("error: unknown argument '{a}'");
                    print_help(&name);
                    std::process::exit(2);
                }
            }
        }
        if sample < 1.0 { name.push_str("-smoke"); }

        Self { name, manifest, target, final_run, sample }
    }

    /// True if the datasets are subsampled, i.e. the run is a sanity check and
    /// its RMSE is not comparable with anything.
    pub fn sampled(&self) -> bool { self.sample < 1.0 }

    /// The sandbox split: datasets from the manifest, predictions into
    /// `preds_lab/`, and reads that miss there served from the manifest's own
    /// preds dir. Also installs the subsample, so call it before `fit2!`.
    pub fn split(&self) -> Split {
        let base = Split::from_pipeline(&self.manifest);
        crate::set_preds_fallback(LAB_PREDS_DIR, base.preds_dir);
        set_sample(self.sample);
        std::fs::create_dir_all(LAB_PREDS_DIR).unwrap();
        Split { preds_dir: LAB_PREDS_DIR, ..base }
    }
}

fn need(argv: &[String], i: usize) -> String {
    argv.get(i + 1).unwrap_or_else(|| panic!("'{}' needs a value", argv[i])).clone()
}

/// Binary name, which is also the default model name.
fn exe_name() -> String {
    std::env::args().next()
        .and_then(|p| std::path::Path::new(&p).file_stem().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "lab".to_string())
}

fn print_help(name: &str) {
    println!("Usage: {name} [NAME] [OPTIONS]");
    println!();
    println!("  NAME                 model name (default: the binary's own name)");
    println!("  -p, --pipeline FILE  manifest the datasets come from (default: {DEFAULT_MANIFEST})");
    println!("  --target SPEC        training target: 'rtg' or 'w*model + ...' (default: rtg)");
    println!("  --final              also run the fulltrain -> qual phase");
    println!("  --smoke              sanity run on {}% of the users", (SMOKE_SAMPLE * 100.0) as u32);
    println!("  --sample FRAC        subsample the users by FRAC (0 < FRAC <= 1)");
    println!("  -h, --help           show this help");
    println!();
    println!("Predictions go to {LAB_PREDS_DIR}/; a subsampled run appends '-smoke' to NAME.");
}

/// Keep `frac` of the users in every dataset loaded from here on.
pub fn set_sample(frac: f64) {
    let permille = (frac * 1000.0).round().clamp(1.0, 1000.0) as u32;
    KEEP_PERMILLE.store(permille, Ordering::Relaxed);
}

/// Drop all but the sampled users. The decision is a hash of the user index, so
/// train, probe and qual keep the same users without having to agree on
/// anything, and row order (user-sorted or item-sorted) survives the filter.
pub(crate) fn maybe_subsample(ds: Dataset) -> Dataset {
    let keep = KEEP_PERMILLE.load(Ordering::Relaxed) as u64;
    if keep >= 1000 { return ds; }

    let idx: Vec<usize> = (0..ds.n_ratings)
        .filter(|&k| hash_user(ds.user_idxs[k]) < keep)
        .collect();
    assert!(!idx.is_empty(), "subsample kept no ratings of '{}'", ds.name);

    let user_idxs = gather(&ds.user_idxs, &idx);
    let item_idxs = gather(&ds.item_idxs, &idx);
    let mut user_cnts = Array1::<i32>::zeros(ds.n_users);
    let mut item_cnts = Array1::<i32>::zeros(ds.n_items);
    for k in 0..idx.len() {
        user_cnts[user_idxs[k] as usize] += 1;
        item_cnts[item_idxs[k] as usize] += 1;
    }

    crate::teeln!("lab: '{}' subsampled to {} of {} ratings", ds.name, idx.len(), ds.n_ratings);

    Dataset {
        raw_ratings: gather(&ds.raw_ratings, &idx),
        residuals:   gather(&ds.residuals, &idx),
        dates:       gather(&ds.dates, &idx),
        is_test:     gather(&ds.is_test, &idx),
        n_ratings:   idx.len(),
        user_idxs, item_idxs, user_cnts, item_cnts,
        item_years: ds.item_years,
        name: ds.name,
        n_users: ds.n_users,
        n_items: ds.n_items,
        transposed: ds.transposed,
    }
}

/// Uniform hash of a user index into 0..1000.
fn hash_user(u: i32) -> u64 {
    ((u as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33) % 1000
}

fn gather<T: Copy>(a: &Array1<T>, idx: &[usize]) -> Array1<T> {
    Array1::from_iter(idx.iter().map(|&k| a[k]))
}
