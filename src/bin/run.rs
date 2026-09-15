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
    let job = match resolved.get(job_name) {
        Some(s) => s,
        None => {
            eprintln!("error: unknown job '{}'", job_name);
            let names: Vec<&str> = resolved.keys().map(|s| s.as_str()).collect();
            eprintln!("available: {}", names.join(", "));
            return ExitCode::from(2);
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
            return ExitCode::from(1);
        }
        Status::Done if !force => {
            println!("Job '{}' is DONE — skipping. Use -f to force re-run.", job_name);
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    println!("Running '{}': {}", job_name, job.cmd);
    match Command::new("sh").arg("-c").arg(&job.cmd).status() {
        Ok(es) if es.success() => ExitCode::SUCCESS,
        Ok(es) => {
            eprintln!("job exited with {}", es);
            ExitCode::from(es.code().unwrap_or(1) as u8)
        }
        Err(e) => {
            eprintln!("failed to spawn shell: {}", e);
            ExitCode::from(127)
        }
    }
}

fn print_help() {
    println!("Usage: run [-p FILE | -n] [-l] [-c] [-f] [JOB]");
    println!();
    println!("  -p FILE, --pipeline FILE   pipeline manifest (default: {})", DEFAULT_PIPELINE);
    println!("  -n, --new                  shortcut for -p pipeline-new.toml");
    println!("  -l, --list                 list jobs with status (default if no JOB)");
    println!("  -c, --clean                list (or delete with -f) files in preds dir");
    println!("                             not referenced by any active job");
    println!("  -f, --force                re-run JOB even if DONE; or actually delete with --clean");
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

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => { print_help(); return ExitCode::SUCCESS; }
            "-l" | "--list" => { force_list = true; i += 1; }
            "-f" | "--force" => { force = true; i += 1; }
            "-c" | "--clean" => { clean_mode = true; i += 1; }
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
