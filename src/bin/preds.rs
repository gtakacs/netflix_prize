//! Remote prediction store. The blend-relevant predictions live in a public
//! bucket that needs no account to read; this tool keeps the local `preds_*`
//! directories and that bucket in sync. `index` writes the md5 index that
//! `pull` verifies every downloaded file against.

use netflix_prize::remote::{self, BUCKET, FileEntry, INDEX_FILE, LocalState};
use std::process::ExitCode;

const DEFAULT_JOBS: usize = 8;

fn print_help() {
    println!("Usage: preds [OPTIONS] index");
    println!("       preds [OPTIONS] pull [PATTERN...]");
    println!();
    println!("  index                      write the md5 index of the bucket's contents");
    println!("  pull [PATTERN...]          download the files an index lists, verifying each");
    println!("                             md5; PATTERN globs the path ('*' matches anything)");
    println!();
    println!("  --bucket ID                bucket as <owner>/<name> (default: {})", BUCKET);
    println!("  -o FILE, --output FILE     index: where to write it (default: {})", INDEX_FILE);
    println!("  --index FILE               pull: read this index instead of the bucket's");
    println!("  -n, --dry-run              pull: report what is missing, download nothing");
    println!("  --quick                    pull: judge local files by size, skipping the md5 pass");
    println!("  -j N, --jobs N             pull: parallel downloads (default: {})", DEFAULT_JOBS);
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
    Ok(())
}

fn main() -> ExitCode {
    let mut bucket = BUCKET.to_string();
    let mut out = INDEX_FILE.to_string();
    let mut index_path: Option<String> = None;
    let mut dry_run = false;
    let mut quick = false;
    let mut jobs = DEFAULT_JOBS;
    let mut positional: Vec<String> = Vec::new();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "-h" | "--help" => { print_help(); return ExitCode::SUCCESS; }
            "-n" | "--dry-run" => { dry_run = true; i += 1; }
            "--quick" => { quick = true; i += 1; }
            "--bucket" | "-o" | "--output" | "--index" | "-j" | "--jobs" => {
                let Some(val) = args.get(i + 1).cloned() else {
                    eprintln!("error: {} requires an argument", flag);
                    return ExitCode::from(2);
                };
                match flag {
                    "--bucket" => bucket = val,
                    "-o" | "--output" => out = val,
                    "--index" => index_path = Some(val),
                    _ => match val.parse::<usize>() {
                        Ok(n) if n > 0 => jobs = n,
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
        "pull" => cmd_pull(&bucket, index_path.as_deref(), rest, dry_run, quick, jobs),
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
