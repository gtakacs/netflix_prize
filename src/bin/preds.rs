//! Remote prediction store. The blend-relevant predictions live in a public
//! bucket that needs no account to read; this tool keeps the local `preds_*`
//! directories and that bucket in sync. `index` writes the md5 index every
//! other subcommand verifies against.

use netflix_prize::remote::{self, BUCKET, INDEX_FILE};
use std::process::ExitCode;

fn print_help() {
    println!("Usage: preds [--bucket ID] [-o FILE] index");
    println!();
    println!("  index                      write the md5 index of the bucket's contents");
    println!();
    println!("  --bucket ID                bucket as <owner>/<name> (default: {})", BUCKET);
    println!("  -o FILE, --output FILE     index path (default: {})", INDEX_FILE);
    println!("  -h, --help                 show this help");
}

/// List the bucket, then md5 the local copy of every object it holds. The
/// bucket is what decides which files belong in the index; the local tree only
/// supplies the digests.
fn cmd_index(bucket: &str, out: &str) -> Result<(), String> {
    println!("Listing {} ...", bucket);
    let remote_files = remote::list_bucket(bucket)?;
    let total: u64 = remote_files.iter().map(|f| f.size).sum();
    println!(
        "  {} file(s), {:.2} GB; hashing local copies ...",
        remote_files.len(),
        total as f64 / 1e9,
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

fn main() -> ExitCode {
    let mut bucket = BUCKET.to_string();
    let mut out = INDEX_FILE.to_string();
    let mut subcommand: Option<String> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => { print_help(); return ExitCode::SUCCESS; }
            "--bucket" | "-o" | "--output" => {
                let Some(val) = args.get(i + 1) else {
                    eprintln!("error: {} requires an argument", args[i]);
                    return ExitCode::from(2);
                };
                if args[i] == "--bucket" { bucket = val.clone(); } else { out = val.clone(); }
                i += 2;
            }
            s if s.starts_with('-') => {
                eprintln!("error: unknown flag '{}'", s);
                print_help();
                return ExitCode::from(2);
            }
            s => {
                if subcommand.is_some() {
                    eprintln!("error: only one subcommand allowed");
                    return ExitCode::from(2);
                }
                subcommand = Some(s.to_string());
                i += 1;
            }
        }
    }

    let result = match subcommand.as_deref() {
        Some("index") => cmd_index(&bucket, &out),
        Some(other) => {
            eprintln!("error: unknown subcommand '{}'", other);
            print_help();
            return ExitCode::from(2);
        }
        None => { print_help(); return ExitCode::SUCCESS; }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::from(1)
        }
    }
}
