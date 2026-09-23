//! Pipeline runner. Reads a manifest like `pipeline-old.toml`, lists every
//! job with its current status, and runs a single job on demand. Status
//! is computed from input/output file existence; staleness checks and
//! transitive runs are intentionally out of scope for this initial version.
//! Manifest parsing and job resolution live in `netflix_prize::pipeline`.

use indexmap::IndexMap;
use netflix_prize::pipeline::{
    Pipeline, ResolvedJob, Status, build_producers, referenced_files, resolve_pipeline, status_of,
};
use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, ExitCode};

const DEFAULT_PIPELINE: &str = "pipeline-old.toml";

/// Split missing inputs into the producer jobs (deduplicated, in first-seen
/// order) and orphan paths (no job produces them — typically source files or
/// raw data).
fn diagnose_blocked(missing: &[String], producers: &HashMap<String, String>) -> (Vec<String>, Vec<String>) {
    let mut producer_jobs: Vec<String> = Vec::new();
    let mut orphans: Vec<String> = Vec::new();
    for m in missing {
        match producers.get(m) {
            Some(p) => {
                if !producer_jobs.contains(p) {
                    producer_jobs.push(p.clone());
                }
            }
            None => orphans.push(m.clone()),
        }
    }
    (producer_jobs, orphans)
}

fn format_blocked_detail(missing: &[String], producers: &HashMap<String, String>) -> String {
    let (jobs, orphans) = diagnose_blocked(missing, producers);
    if jobs.is_empty() {
        return if orphans.len() == 1 {
            format!("  (missing: {})", orphans[0])
        } else {
            format!("  (missing {}: {}, ...)", orphans.len(), orphans[0])
        };
    }
    let mut s = format!("  (blocked by: {}", jobs.join(", "));
    if !orphans.is_empty() {
        s.push_str(&format!(", +{} file{}", orphans.len(), if orphans.len() == 1 { "" } else { "s" }));
    }
    s.push(')');
    s
}

fn list_jobs(pipeline_path: &str, p: &Pipeline, resolved: &IndexMap<String, ResolvedJob>) {
    let split_name = p.split.get("name").cloned().unwrap_or_default();
    println!("Pipeline: {} (split = {})", pipeline_path, split_name);
    println!();
    let producers = build_producers(resolved);
    let mut n_done = 0;
    let mut n_runnable = 0;
    let mut n_blocked = 0;
    for (name, job) in resolved {
        let s = status_of(job, resolved);
        let label = match &s {
            Status::Done => { n_done += 1; "DONE" }
            Status::Runnable => { n_runnable += 1; "RUNNABLE" }
            Status::Blocked(_) => { n_blocked += 1; "BLOCKED" }
        };
        let detail = match &s {
            Status::Blocked(missing) => format_blocked_detail(missing, &producers),
            _ => String::new(),
        };
        let jobtype = p.jobs.get(name)
            .and_then(|s| s.jobtype.as_deref())
            .unwrap_or("-");
        println!("  {:30} {:15} {}{}", name, jobtype, label, detail);
    }
    println!();
    println!(
        "{} jobs total: {} DONE, {} RUNNABLE, {} BLOCKED",
        resolved.len(), n_done, n_runnable, n_blocked,
    );
}

fn run_job(job_name: &str, resolved: &IndexMap<String, ResolvedJob>, force: bool) -> ExitCode {
    ExitCode::from(run_one(job_name, resolved, force))
}

/// Run one job by name. Returns 0 when it ran or was skipped as DONE, and the
/// job's own exit code otherwise, so callers can chain several jobs.
fn run_one(job_name: &str, resolved: &IndexMap<String, ResolvedJob>, force: bool) -> u8 {
    let job = match resolved.get(job_name) {
        Some(s) => s,
        None => {
            eprintln!("error: unknown job '{}'", job_name);
            let names: Vec<&str> = resolved.keys().map(|s| s.as_str()).collect();
            eprintln!("available: {}", names.join(", "));
            return 2;
        }
    };
    match status_of(job, resolved) {
        Status::Blocked(missing) => {
            eprintln!("error: job '{}' is BLOCKED", job_name);
            let producers = build_producers(resolved);
            let (jobs, orphans) = diagnose_blocked(&missing, &producers);
            if !jobs.is_empty() {
                eprintln!("  blocked by jobs:");
                for j in &jobs {
                    eprintln!("    - {}", j);
                }
            }
            if !orphans.is_empty() {
                eprintln!("  missing files:");
                for f in &orphans {
                    eprintln!("    - {}", f);
                }
            }
            return 1;
        }
        Status::Done if !force => {
            println!("Job '{}' is DONE — skipping. Use -f to force re-run.", job_name);
            return 0;
        }
        _ => {}
    }
    println!("Running '{}': {}", job_name, job.cmd);
    match Command::new("sh").arg("-c").arg(&job.cmd).status() {
        Ok(es) if es.success() => 0,
        Ok(es) => {
            eprintln!("job exited with {}", es);
            es.code().unwrap_or(1) as u8
        }
        Err(e) => {
            eprintln!("failed to spawn shell: {}", e);
            127
        }
    }
}

const NEW_PIPELINE: &str = "pipeline-new.toml";

/// Total size of a directory tree, for the closing summary.
fn dir_size(path: &str) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else { return 0 };
    entries.flatten().map(|e| match e.file_type() {
        Ok(t) if t.is_dir() => dir_size(&e.path().to_string_lossy()),
        _ => e.metadata().map(|m| m.len()).unwrap_or(0),
    }).sum()
}

fn human(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{} MB", b >> 20),
        b => format!("{} B", b),
    }
}

/// Everything a fresh clone needs before any model can run: fetch the archive,
/// parse it into the npy datasets, then derive the second split. The three
/// jobs live in two manifests, so they are run by name rather than by walking
/// one graph. Each is skipped when its outputs are already there.
fn cmd_setup(pipeline_path: &str, force: bool) -> ExitCode {
    let build = "cargo build --release --bin download --bin ingest --bin newsplit";

    println!("Setup: dataset -> npy arrays -> both splits");
    println!();
    println!("  build      {}", build);
    println!("  download   data/raw/  (697 MB archive, md5 verified, resumable)");
    println!("  ingest     data/{{train,probe,fulltrain,qual}}/   [{}]", pipeline_path);
    println!("  newsplit   data/{{trainx,probex}}/                [{}]", NEW_PIPELINE);
    println!();
    println!("About 3.3 GB on disk when finished. Steps whose outputs exist are skipped.");
    println!();

    println!("Running: {}", build);
    match Command::new("sh").arg("-c").arg(build).status() {
        Ok(es) if es.success() => {}
        Ok(es) => {
            eprintln!("build exited with {}", es);
            return ExitCode::from(es.code().unwrap_or(1) as u8);
        }
        Err(e) => {
            eprintln!("failed to spawn shell: {}", e);
            return ExitCode::from(127);
        }
    }

    // download and ingest come from the selected manifest, newsplit only exists
    // in the new-split one.
    for (path, jobs) in [(pipeline_path, &["download", "ingest"][..]), (NEW_PIPELINE, &["newsplit"][..])] {
        let pipeline = match Pipeline::load(path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}", e);
                return ExitCode::from(2);
            }
        };
        let resolved = resolve_pipeline(&pipeline);
        for job in jobs {
            println!();
            let code = run_one(job, &resolved, force);
            if code != 0 {
                return ExitCode::from(code);
            }
        }
    }

    println!();
    println!("Datasets ready:");
    for d in ["train", "probe", "fulltrain", "qual", "trainx", "probex"] {
        let path = format!("data/{}", d);
        let size = dir_size(&path);
        if size > 0 {
            println!("  {:15} {:>9}", path, human(size));
        }
    }
    println!();
    println!("Next: ./target/release/run -n            # list the jobs and their status");
    println!("      ./target/release/preds pull        # the published predictions, no training needed");
    ExitCode::SUCCESS
}

fn print_help() {
    println!("Usage: run [-p FILE | -n] [-l] [-c] [-f] [--setup] [JOB]");
    println!();
    println!("  -p FILE, --pipeline FILE   pipeline manifest (default: {})", DEFAULT_PIPELINE);
    println!("  -n, --new                  shortcut for -p pipeline-new.toml");
    println!("  -l, --list                 list jobs with status (default if no JOB)");
    println!("  -c, --clean                list (or delete with -f) files in preds dir");
    println!("                             not referenced by any active job");
    println!("  -f, --force                re-run JOB even if DONE; or actually delete with --clean");
    println!("      --setup                fetch the dataset, parse it into npy arrays and");
    println!("                             derive both splits (download, ingest, newsplit)");
    println!("  -h, --help                 show this help");
    println!("  JOB                        run the named job");
}

fn cmd_clean(p: &Pipeline, resolved: &IndexMap<String, ResolvedJob>, force: bool) -> ExitCode {
    let protected = referenced_files(resolved);

    // Directories to scan: those declared in [split] under "preds".
    let mut dirs: Vec<String> = Vec::new();
    if let Some(d) = p.split.get("preds") { dirs.push(d.clone()); }

    let mut candidates: Vec<String> = Vec::new();
    for dir in &dirs {
        if !Path::new(dir).is_dir() { continue; }
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() { continue; }
            let path_str = path.to_string_lossy().to_string();
            if !protected.contains(&path_str) {
                candidates.push(path_str);
            }
        }
    }
    candidates.sort();

    if candidates.is_empty() {
        println!("No deletion candidates.");
        return ExitCode::SUCCESS;
    }

    let n = candidates.len();
    println!("{} candidate{} for deletion:", n, if n == 1 { "" } else { "s" });
    for c in &candidates {
        println!("  {}", c);
    }

    if !force {
        println!();
        println!("Dry run. Use --clean -f to delete.");
        return ExitCode::SUCCESS;
    }

    println!();
    println!("Deleting...");
    let mut errs = 0;
    for c in &candidates {
        if let Err(e) = std::fs::remove_file(c) {
            eprintln!("  error deleting {}: {}", c, e);
            errs += 1;
        }
    }
    if errs > 0 {
        eprintln!("{} deletion(s) failed.", errs);
        return ExitCode::from(1);
    }
    println!("Deleted {} file{}.", n, if n == 1 { "" } else { "s" });
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let mut pipeline_path = DEFAULT_PIPELINE.to_string();
    let mut job_arg: Option<String> = None;
    let mut force_list = false;
    let mut force = false;
    let mut clean_mode = false;
    let mut setup_mode = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => { print_help(); return ExitCode::SUCCESS; }
            "-l" | "--list" => { force_list = true; i += 1; }
            "-f" | "--force" => { force = true; i += 1; }
            "-c" | "--clean" => { clean_mode = true; i += 1; }
            "--setup" => { setup_mode = true; i += 1; }
            "-n" | "--new" => { pipeline_path = "pipeline-new.toml".to_string(); i += 1; }
            "-p" | "--pipeline" => {
                if i + 1 >= args.len() {
                    eprintln!("error: {} requires an argument", args[i]);
                    return ExitCode::from(2);
                }
                pipeline_path = args[i + 1].clone();
                i += 2;
            }
            s if s.starts_with('-') => {
                eprintln!("error: unknown flag '{}'", s);
                print_help();
                return ExitCode::from(2);
            }
            s => {
                if job_arg.is_some() {
                    eprintln!("error: only one job argument allowed");
                    return ExitCode::from(2);
                }
                job_arg = Some(s.to_string());
                i += 1;
            }
        }
    }

    if setup_mode {
        if job_arg.is_some() {
            eprintln!("warning: JOB argument ignored with --setup");
        }
        return cmd_setup(&pipeline_path, force);
    }

    let pipeline = match Pipeline::load(&pipeline_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::from(2);
        }
    };
    let resolved = resolve_pipeline(&pipeline);

    if clean_mode {
        if job_arg.is_some() {
            eprintln!("warning: JOB argument ignored with --clean");
        }
        return cmd_clean(&pipeline, &resolved, force);
    }

    match job_arg {
        Some(name) => {
            if force_list {
                eprintln!("warning: -l ignored when a JOB is specified");
            }
            run_job(&name, &resolved, force)
        }
        None => {
            if force {
                eprintln!("warning: -f ignored without a JOB");
            }
            list_jobs(&pipeline_path, &pipeline, &resolved);
            ExitCode::SUCCESS
        }
    }
}
