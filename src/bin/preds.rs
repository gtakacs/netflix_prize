//! Remote prediction store. The blend-relevant predictions live in a public
//! bucket that needs no account to read; this tool keeps the local `preds_*`
//! directories and that bucket in sync. `index` writes the md5 index that
//! `pull` verifies every downloaded file against.

use netflix_prize::pipeline::{Pipeline, referenced_files, resolve_pipeline};
use netflix_prize::remote::{self, BUCKET, FileEntry, INDEX_FILE, LocalState};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::process::ExitCode;

const DEFAULT_JOBS: usize = 8;
/// Both manifests, because the store spans both splits.
const DEFAULT_MANIFESTS: [&str; 2] = ["pipeline-old.toml", "pipeline-new.toml"];
/// Uploads go through one `hf` process each, so stay gentler than with GETs.
const DEFAULT_UPLOAD_JOBS: usize = 4;

fn print_help() {
    println!("Usage: preds [OPTIONS] index");
    println!("       preds [OPTIONS] pull [PATTERN...]");
    println!("       preds [OPTIONS] push");
    println!("       preds [OPTIONS] prune --yes");
    println!();
    println!("  index                      write the md5 index of the bucket's contents");
    println!("  pull [PATTERN...]          download the files an index lists, verifying each");
    println!("                             md5; PATTERN globs the path ('*' matches anything)");
    println!("  push                       upload what the manifests reference, then the index");
    println!("                             (needs 'hf auth login'; never deletes)");
    println!("  prune                      delete bucket objects outside that set, but only");
    println!("                             ones with an intact local copy; irreversible");
    println!();
    println!("  --bucket ID                bucket as <owner>/<name> (default: {})", BUCKET);
    println!("  -o FILE, --output FILE     index: where to write it (default: {})", INDEX_FILE);
    println!("  --index FILE               pull: read this index instead of the bucket's");
    println!("  -n, --dry-run              pull: report what is missing, download nothing");
    println!("  --quick                    pull: judge local files by size, skipping the md5 pass");
    println!("  -p FILE, --pipeline FILE   push: manifest to take the upload set from,");
    println!("                             repeatable (default: {})", DEFAULT_MANIFESTS.join(", "));
    println!("  -j N, --jobs N             parallel transfers (default: {} pulling, {} pushing)", DEFAULT_JOBS, DEFAULT_UPLOAD_JOBS);
    println!("  -y, --yes                  prune: actually delete (without it, a dry run)");
    println!("  -h, --help                 show this help");
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else {
        format!("{:.1} kB", b / 1e3)
    }
}

/// `*` matches any run of characters, `/` included; everything else is literal.
fn glob_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    if !text.starts_with(parts[0]) {
        return false;
    }
    let mut pos = parts[0].len();
    for (i, part) in parts.iter().enumerate().skip(1) {
        if i == parts.len() - 1 {
            // The tail must match at the end, and must not reuse consumed bytes.
            return text.len() - pos >= part.len() && text.ends_with(part);
        }
        if part.is_empty() {
            continue;
        }
        match text[pos..].find(part) {
            Some(idx) => pos += idx + part.len(),
            None => return false,
        }
    }
    true
}

fn walk_files(dir: &str, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.to_string_lossy().to_string();
        if path.is_dir() {
            walk_files(&name, out);
        } else if path.is_file() {
            out.push(name);
        }
    }
}

/// What the store should hold: every path under a preds directory that some job
/// references, plus the `.out` logs and `.cfg` configs sitting beside them.
///
/// Predictions on the *training* sets (`{tr}`, `{fulltrain_tr}`) are left out.
/// They are 21 GB, and they only matter for retraining residual models, not for
/// blending. The dataset names come from each manifest's `[split]` table, so
/// nothing here is hardcoded per split.
fn upload_set(manifests: &[String]) -> Result<BTreeSet<String>, String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for manifest in manifests {
        let p = Pipeline::load(manifest)?;
        let preds_dir = p
            .split
            .get("preds")
            .ok_or_else(|| format!("{manifest}: [split].preds is missing"))?
            .clone();
        let training: Vec<String> = ["tr", "fulltrain_tr"]
            .iter()
            .filter_map(|k| p.split.get(*k))
            .map(|ds| format!(".{ds}.npy"))
            .collect();

        let resolved = resolve_pipeline(&p);
        for f in referenced_files(&resolved) {
            if !f.starts_with(&format!("{preds_dir}/")) {
                continue;
            }
            if training.iter().any(|suffix| f.ends_with(suffix)) {
                continue;
            }
            // A referenced file that does not exist is simply a job not yet run.
            if Path::new(&f).is_file() {
                set.insert(f);
            }
        }

        let mut found = Vec::new();
        walk_files(&preds_dir, &mut found);
        for f in found {
            if f.ends_with(".out") || f.ends_with(".cfg") {
                set.insert(f);
            }
        }
    }
    Ok(set)
}

/// List the bucket, then md5 the local copy of every object it holds. The
/// bucket is what decides which files belong in the index; the local tree only
/// supplies the digests.
fn cmd_index(bucket: &str, out: &str) -> Result<(), String> {
    println!("Listing {} ...", bucket);
    let mut remote_files = remote::list_bucket(bucket)?;
    let n_objects = remote_files.len();
    remote_files.retain(|f| !remote::is_bucket_meta(&f.path));
    let total: u64 = remote_files.iter().map(|f| f.size).sum();
    println!(
        "  {} object(s), {} to index, {}; hashing local copies ...",
        n_objects,
        remote_files.len(),
        human(total),
    );

    let index = remote::build_index(bucket, &remote_files)?;
    remote::write_index(&index, out)?;

    // Read it straight back: the index is published for other people's tools,
    // so it must parse with the same reader they will use.
    let reread = remote::read_index(out)?;
    if reread.file.len() != index.file.len() {
        return Err(format!(
            "{}: wrote {} entries but read back {}",
            out,
            index.file.len(),
            reread.file.len(),
        ));
    }
    println!("Wrote {} ({} entries, generated {})", out, index.file.len(), index.meta.generated);
    Ok(())
}

fn cmd_pull(
    bucket: &str,
    index_path: Option<&str>,
    patterns: &[String],
    dry_run: bool,
    quick: bool,
    jobs: usize,
) -> Result<(), String> {
    let index = match index_path {
        Some(p) => {
            println!("Reading index {} ...", p);
            remote::read_index(p)?
        }
        None => {
            println!("Fetching index from {} ...", bucket);
            remote::fetch_index(bucket)?
        }
    };
    let indexed: u64 = index.file.iter().map(|e| e.size).sum();
    println!(
        "  {} entries, {}, generated {}",
        index.file.len(),
        human(indexed),
        index.meta.generated,
    );

    let selected: Vec<FileEntry> = if patterns.is_empty() {
        index.file.clone()
    } else {
        index
            .file
            .iter()
            .filter(|e| patterns.iter().any(|p| glob_match(p, &e.path)))
            .cloned()
            .collect()
    };
    if selected.is_empty() {
        return Err("no index entry matches the given pattern(s)".to_string());
    }
    let selected_bytes: u64 = selected.iter().map(|e| e.size).sum();
    println!(
        "Selected {} file(s), {}; checking local copies ...",
        selected.len(),
        human(selected_bytes),
    );

    let states = remote::classify_all(&selected, quick);
    let mut todo: Vec<FileEntry> = Vec::new();
    let (mut n_current, mut n_missing, mut n_stale) = (0, 0, 0);
    for (entry, state) in selected.iter().zip(&states) {
        match state {
            LocalState::Current => n_current += 1,
            LocalState::Missing => {
                n_missing += 1;
                todo.push(entry.clone());
            }
            LocalState::Stale => {
                n_stale += 1;
                todo.push(entry.clone());
            }
        }
    }
    println!("  {} up to date, {} missing, {} stale", n_current, n_missing, n_stale);
    if todo.is_empty() {
        println!("Nothing to download.");
        print_next_steps();
        return Ok(());
    }

    let todo_bytes: u64 = todo.iter().map(|e| e.size).sum();
    if dry_run {
        println!("Dry run: would download {} file(s), {}.", todo.len(), human(todo_bytes));
        for e in todo.iter().take(20) {
            println!("  {}", e.path);
        }
        if todo.len() > 20 {
            println!("  ... and {} more", todo.len() - 20);
        }
        return Ok(());
    }

    println!("Downloading {} file(s), {} ...", todo.len(), human(todo_bytes));
    let errs = remote::fetch_all(&index.meta.base_url, &todo, jobs);
    if !errs.is_empty() {
        let shown = errs.iter().take(10).cloned().collect::<Vec<_>>().join("\n  ");
        let more = if errs.len() > 10 {
            format!("\n  ... and {} more", errs.len() - 10)
        } else {
            String::new()
        };
        return Err(format!("{} download(s) failed:\n  {}{}", errs.len(), shown, more));
    }
    println!("Done: {} file(s) downloaded.", todo.len());
    print_next_steps();
    Ok(())
}

/// What to run once the predictions are on disk. The store exists so that
/// blending can be tried without training anything, and this is the one command
/// that shows whether the download is complete and correct.
fn print_next_steps() {
    println!();
    println!("Next:");
    println!("  cargo build --release --features blas --bin ridge");
    println!("  ./target/release/ridge --ensemble          # does the stored blend reproduce?");
    println!("  ./target/release/ridge --ensemble -m NAME  # what does a column of yours add?");
    println!();
    println!("See README.md and docs/EXPERIMENTS.md for the rest of the loop.");
}

/// Bring the bucket up to date with the local tree, then refresh the index.
/// The index is the record of what the bucket holds, so it is also what decides
/// whether a local file has changed: a retrained model keeps its name and size,
/// and only the md5 gives it away.
fn cmd_push(bucket: &str, manifests: &[String], dry_run: bool, jobs: usize) -> Result<(), String> {
    let who = remote::require_auth()?;
    println!("Authenticated ({})", who);

    let set = upload_set(manifests)?;
    let set_bytes: u64 = set.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();
    println!(
        "Upload set from {}: {} file(s), {}",
        manifests.join(" + "),
        set.len(),
        human(set_bytes),
    );

    println!("Fetching index from {} ...", bucket);
    let index = remote::fetch_index(bucket)?;
    let known: HashMap<&str, &FileEntry> =
        index.file.iter().map(|e| (e.path.as_str(), e)).collect();

    let indexed: Vec<FileEntry> = set
        .iter()
        .filter_map(|p| known.get(p.as_str()).map(|e| (*e).clone()))
        .collect();
    let mut todo: Vec<String> =
        set.iter().filter(|p| !known.contains_key(p.as_str())).cloned().collect();
    let n_new = todo.len();

    println!("  {} already in the index; checking them for changes ...", indexed.len());
    let states = remote::classify_all(&indexed, false);
    let mut n_same = 0;
    for (entry, state) in indexed.iter().zip(&states) {
        if *state == LocalState::Current {
            n_same += 1;
        } else {
            todo.push(entry.path.clone());
        }
    }
    todo.sort();
    println!("  {} unchanged, {} changed, {} new", n_same, todo.len() - n_new, n_new);

    let orphans: Vec<&FileEntry> =
        index.file.iter().filter(|e| !set.contains(&e.path)).collect();
    if !orphans.is_empty() {
        let bytes: u64 = orphans.iter().map(|e| e.size).sum();
        println!(
            "  {} object(s) in the bucket sit outside the upload set ({}); push never deletes",
            orphans.len(),
            human(bytes),
        );
        // Grouped by extension, so it is obvious what a later prune would drop.
        let mut by_ext: Vec<(&str, usize, u64)> = Vec::new();
        for e in &orphans {
            let ext = e.path.rsplit('.').next().unwrap_or("");
            match by_ext.iter_mut().find(|(x, _, _)| *x == ext) {
                Some(row) => {
                    row.1 += 1;
                    row.2 += e.size;
                }
                None => by_ext.push((ext, 1, e.size)),
            }
        }
        by_ext.sort_by(|a, b| b.2.cmp(&a.2));
        for (ext, n, bytes) in by_ext {
            println!("      {:5} .{:<4} {}", n, ext, human(bytes));
        }
    }

    if todo.is_empty() {
        println!("Bucket is up to date.");
        return Ok(());
    }
    let bytes: u64 = todo.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();
    if dry_run {
        println!("Dry run: would upload {} file(s), {}.", todo.len(), human(bytes));
        for p in todo.iter().take(20) {
            println!("  {}", p);
        }
        if todo.len() > 20 {
            println!("  ... and {} more", todo.len() - 20);
        }
        return Ok(());
    }

    println!("Uploading {} file(s), {} ...", todo.len(), human(bytes));
    let errs = remote::upload_all(bucket, &todo, jobs);
    if !errs.is_empty() {
        let shown = errs.iter().take(10).cloned().collect::<Vec<_>>().join("\n  ");
        let more = if errs.len() > 10 {
            format!("\n  ... and {} more", errs.len() - 10)
        } else {
            String::new()
        };
        return Err(format!("{} upload(s) failed:\n  {}{}", errs.len(), shown, more));
    }

    // The index goes up last, so the bucket never advertises a file it lacks.
    println!("Refreshing the index ...");
    cmd_index(bucket, INDEX_FILE)?;
    remote::upload_file(bucket, INDEX_FILE, INDEX_FILE)?;
    println!("Done: {} file(s) pushed, index updated.", todo.len());
    Ok(())
}

/// Delete what the bucket holds beyond the upload set. Buckets are unversioned,
/// so this cannot be undone; the safeguard is that every object is verified
/// against an intact local copy first, and nothing is deleted unless all of them
/// pass. The local files are never touched.
fn cmd_prune(
    bucket: &str,
    manifests: &[String],
    assume_yes: bool,
    jobs: usize,
) -> Result<(), String> {
    let who = remote::require_auth()?;
    println!("Authenticated ({})", who);

    let set = upload_set(manifests)?;
    println!("Fetching index from {} ...", bucket);
    let index = remote::fetch_index(bucket)?;
    let orphans: Vec<FileEntry> =
        index.file.iter().filter(|e| !set.contains(&e.path)).cloned().collect();
    if orphans.is_empty() {
        println!("Nothing in the bucket sits outside the upload set.");
        return Ok(());
    }
    let bytes: u64 = orphans.iter().map(|e| e.size).sum();
    println!("  {} object(s) outside the upload set, {}", orphans.len(), human(bytes));

    println!("Verifying that every one of them survives locally ...");
    let states = remote::classify_all(&orphans, false);
    let unverified: Vec<&FileEntry> = orphans
        .iter()
        .zip(&states)
        .filter(|(_, s)| **s != LocalState::Current)
        .map(|(e, _)| e)
        .collect();
    if !unverified.is_empty() {
        let shown = unverified
            .iter()
            .take(10)
            .map(|e| e.path.clone())
            .collect::<Vec<_>>()
            .join("\n  ");
        return Err(format!(
            "{} object(s) have no intact local copy, so deleting them would lose data. \
             Nothing was deleted:\n  {}",
            unverified.len(),
            shown,
        ));
    }
    println!("  all {} verified byte for byte against the local tree", orphans.len());

    let paths: Vec<String> = orphans.iter().map(|e| e.path.clone()).collect();
    if !assume_yes {
        println!("Dry run: would delete {} object(s), {}.", paths.len(), human(bytes));
        for p in paths.iter().take(20) {
            println!("  {}", p);
        }
        if paths.len() > 20 {
            println!("  ... and {} more", paths.len() - 20);
        }
        println!("Pass --yes to delete. This cannot be undone.");
        return Ok(());
    }

    println!("Deleting {} object(s) ...", paths.len());
    let errs = remote::delete_all(bucket, &paths, jobs);
    if !errs.is_empty() {
        let shown = errs.iter().take(10).cloned().collect::<Vec<_>>().join("\n  ");
        return Err(format!("{} deletion(s) failed:\n  {}", errs.len(), shown));
    }

    println!("Refreshing the index ...");
    cmd_index(bucket, INDEX_FILE)?;
    remote::upload_file(bucket, INDEX_FILE, INDEX_FILE)?;
    println!("Done: {} object(s) deleted, index updated.", paths.len());
    Ok(())
}

fn main() -> ExitCode {
    let mut bucket = BUCKET.to_string();
    let mut out = INDEX_FILE.to_string();
    let mut index_path: Option<String> = None;
    let mut dry_run = false;
    let mut quick = false;
    let mut assume_yes = false;
    let mut jobs: Option<usize> = None;
    let mut manifests: Vec<String> = Vec::new();
    let mut positional: Vec<String> = Vec::new();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "-h" | "--help" => { print_help(); return ExitCode::SUCCESS; }
            "-n" | "--dry-run" => { dry_run = true; i += 1; }
            "--quick" => { quick = true; i += 1; }
            "-y" | "--yes" => { assume_yes = true; i += 1; }
            "--bucket" | "-o" | "--output" | "--index" | "-j" | "--jobs" | "-p" | "--pipeline" => {
                let Some(val) = args.get(i + 1).cloned() else {
                    eprintln!("error: {} requires an argument", flag);
                    return ExitCode::from(2);
                };
                match flag {
                    "--bucket" => bucket = val,
                    "-o" | "--output" => out = val,
                    "--index" => index_path = Some(val),
                    "-p" | "--pipeline" => manifests.push(val),
                    _ => match val.parse::<usize>() {
                        Ok(n) if n > 0 => jobs = Some(n),
                        _ => {
                            eprintln!("error: {} needs a positive number, got '{}'", flag, val);
                            return ExitCode::from(2);
                        }
                    },
                }
                i += 2;
            }
            s if s.starts_with('-') => {
                eprintln!("error: unknown flag '{}'", s);
                print_help();
                return ExitCode::from(2);
            }
            s => { positional.push(s.to_string()); i += 1; }
        }
    }

    let (subcommand, rest) = match positional.split_first() {
        Some((first, rest)) => (first.as_str(), rest),
        None => { print_help(); return ExitCode::SUCCESS; }
    };
    let result = match subcommand {
        "index" => {
            if !rest.is_empty() {
                eprintln!("error: 'index' takes no further arguments");
                return ExitCode::from(2);
            }
            cmd_index(&bucket, &out)
        }
        "pull" => cmd_pull(
            &bucket,
            index_path.as_deref(),
            rest,
            dry_run,
            quick,
            jobs.unwrap_or(DEFAULT_JOBS),
        ),
        "push" => {
            if !rest.is_empty() {
                eprintln!("error: 'push' takes no further arguments");
                return ExitCode::from(2);
            }
            if manifests.is_empty() {
                manifests = DEFAULT_MANIFESTS.iter().map(|s| s.to_string()).collect();
            }
            cmd_push(&bucket, &manifests, dry_run, jobs.unwrap_or(DEFAULT_UPLOAD_JOBS))
        }
        "prune" => {
            if !rest.is_empty() {
                eprintln!("error: 'prune' takes no further arguments");
                return ExitCode::from(2);
            }
            if manifests.is_empty() {
                manifests = DEFAULT_MANIFESTS.iter().map(|s| s.to_string()).collect();
            }
            cmd_prune(
                &bucket,
                &manifests,
                assume_yes && !dry_run,
                jobs.unwrap_or(DEFAULT_UPLOAD_JOBS),
            )
        }
        other => {
            eprintln!("error: unknown subcommand '{}'", other);
            print_help();
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::from(1)
        }
    }
}
