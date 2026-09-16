//! Remote prediction store: bucket listing and the md5 index that describes it.
//!
//! The store is a public Hugging Face bucket served over plain HTTPS, so a
//! consumer needs no account and no extra CLI. The index carries the md5 of
//! every object, which the bucket itself cannot provide (its ETag is a Xet
//! hash, not computable from a local file).

use crate::make_pb;
use indicatif::ProgressStyle;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default bucket, as `<owner>/<name>`.
pub const BUCKET: &str = "gtakacs/netflix_prize";
/// Index file name, both locally and at the bucket root.
pub const INDEX_FILE: &str = "index.toml";
const INDEX_VERSION: u32 = 1;
const HTTP_TIMEOUT_SECS: u64 = 120;

/// Prefix every object URL shares. `{base_url}/{path}` answers with a 302 to a
/// signed CDN URL; the signature expires, so re-resolve rather than caching it.
pub fn base_url(bucket: &str) -> String {
    format!("https://huggingface.co/buckets/{bucket}/resolve")
}

fn tree_url(bucket: &str) -> String {
    format!("https://huggingface.co/api/buckets/{bucket}/tree?recursive=true&limit=1000&sort=path")
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Index {
    pub meta: Meta,
    #[serde(default)]
    pub file: Vec<FileEntry>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Meta {
    pub version: u32,
    pub generated: String,
    pub bucket: String,
    pub base_url: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
    pub md5: String,
}

/// One object in the bucket, as the Hub's tree API reports it.
#[derive(Debug)]
pub struct RemoteFile {
    pub path: String,
    pub size: u64,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
}

/// Every object in the bucket. The tree API pages at 1000 entries and points at
/// the next page with a `Link: <url>; rel="next"` header.
pub fn list_bucket(bucket: &str) -> Result<Vec<RemoteFile>, String> {
    let mut url = tree_url(bucket);
    let mut out: Vec<RemoteFile> = Vec::new();
    loop {
        let resp = minreq::get(&url)
            .with_timeout(HTTP_TIMEOUT_SECS)
            .send()
            .map_err(|e| format!("GET {url}: {e}"))?;
        if resp.status_code != 200 {
            return Err(format!("GET {url}: HTTP {}", resp.status_code));
        }
        let body = resp.as_str().map_err(|e| format!("GET {url}: {e}"))?;
        let page: Vec<TreeEntry> =
            serde_json::from_str(body).map_err(|e| format!("parse {url}: {e}"))?;
        for e in page {
            if e.kind == "file" {
                out.push(RemoteFile { path: e.path, size: e.size });
            }
        }
        match resp.headers.get("link").and_then(|h| next_link(h)) {
            Some(next) => url = next,
            None => break,
        }
    }
    Ok(out)
}

/// The `rel="next"` target of a `Link` header, if it has one.
fn next_link(header: &str) -> Option<String> {
    for part in header.split(',') {
        if !part.contains("rel=\"next\"") {
            continue;
        }
        let start = part.find('<')? + 1;
        let end = part.find('>')?;
        return Some(part[start..end].to_string());
    }
    None
}

/// Bucket-level documentation and bookkeeping, as opposed to a prediction file.
/// The index covers the predictions only: an entry for the index itself would
/// describe the previous generation the moment the index is rewritten, and the
/// bucket's README has no counterpart in the local tree (the repo's own
/// README.md is a different file that happens to share the name).
pub fn is_bucket_meta(path: &str) -> bool {
    path == INDEX_FILE || path.starts_with("index/") || path == "README.md"
}

pub fn md5_file(path: &str) -> Result<String, String> {
    let mut f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut ctx = md5::Context::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("read {path}: {e}"))?;
        if n == 0 {
            break;
        }
        ctx.consume(&buf[..n]);
    }
    Ok(format!("{:x}", ctx.compute()))
}

/// Index every object the bucket holds, taking each md5 from the local copy.
/// A remote object with no local counterpart, or one whose local size differs,
/// is an error: the index must never claim an md5 that was not computed over
/// the bytes the bucket serves.
pub fn build_index(bucket: &str, remote: &[RemoteFile]) -> Result<Index, String> {
    let total: u64 = remote.iter().map(|f| f.size).sum();
    let pb = make_pb(total);
    pb.set_style(
        ProgressStyle::with_template("  {bytes}/{total_bytes} [{bar:30}] {bytes_per_sec}, ETA {eta}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let results: Vec<Result<FileEntry, String>> = remote
        .par_iter()
        .map(|rf| {
            let meta = std::fs::metadata(&rf.path)
                .map_err(|e| format!("{}: no local copy ({e})", rf.path))?;
            if meta.len() != rf.size {
                return Err(format!(
                    "{}: size differs (local {}, bucket {})",
                    rf.path,
                    meta.len(),
                    rf.size
                ));
            }
            let md5 = md5_file(&rf.path)?;
            pb.inc(rf.size);
            Ok(FileEntry { path: rf.path.clone(), size: rf.size, md5 })
        })
        .collect();
    pb.finish_and_clear();

    let mut files: Vec<FileEntry> = Vec::new();
    let mut errs: Vec<String> = Vec::new();
    for r in results {
        match r {
            Ok(f) => files.push(f),
            Err(e) => errs.push(e),
        }
    }
    if !errs.is_empty() {
        errs.sort();
        let shown = errs.iter().take(10).cloned().collect::<Vec<_>>().join("\n  ");
        let more = if errs.len() > 10 {
            format!("\n  ... and {} more", errs.len() - 10)
        } else {
            String::new()
        };
        return Err(format!("{} file(s) unusable:\n  {}{}", errs.len(), shown, more));
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(Index {
        meta: Meta {
            version: INDEX_VERSION,
            generated: utc_now(),
            bucket: bucket.to_string(),
            base_url: base_url(bucket),
        },
        file: files,
    })
}

pub fn write_index(index: &Index, path: &str) -> Result<(), String> {
    let body = toml::to_string_pretty(index).map_err(|e| format!("serialize index: {e}"))?;
    std::fs::write(path, body).map_err(|e| format!("write {path}: {e}"))
}

pub fn read_index(path: &str) -> Result<Index, String> {
    let body = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    toml::from_str(&body).map_err(|e| format!("parse {path}: {e}"))
}

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_utc(secs)
}

/// Unix seconds to `YYYY-MM-DDTHH:MM:SSZ`, via the usual civil-from-days
/// algorithm (year starts in March, so the leap day lands at the end).
fn format_utc(secs: u64) -> String {
    let (days, tod) = ((secs / 86_400) as i64, secs % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year,
        month,
        day,
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

const MAX_ATTEMPTS: u32 = 5;

/// Download the index straight from the bucket. This is the bootstrap: a fresh
/// clone has no local index, and the bucket's copy is the authority anyway.
pub fn fetch_index(bucket: &str) -> Result<Index, String> {
    let url = format!("{}/{}", base_url(bucket), INDEX_FILE);
    let resp = minreq::get(&url)
        .with_max_redirects(5)
        .with_timeout(HTTP_TIMEOUT_SECS)
        .send()
        .map_err(|e| format!("GET {url}: {e}"))?;
    if resp.status_code != 200 {
        return Err(format!("GET {url}: HTTP {}", resp.status_code));
    }
    let body = resp.as_str().map_err(|e| format!("GET {url}: {e}"))?;
    toml::from_str(body).map_err(|e| format!("parse {url}: {e}"))
}

/// What the local tree holds for an index entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalState {
    /// Present with the expected content (or the expected size under `--quick`).
    Current,
    /// Not there at all.
    Missing,
    /// There, but not what the index describes: a retrained model keeps both the
    /// file name and the array size, so only the md5 tells the two apart.
    Stale,
}

pub fn classify(entry: &FileEntry, quick: bool) -> LocalState {
    let Ok(meta) = std::fs::metadata(&entry.path) else {
        return LocalState::Missing;
    };
    if meta.len() != entry.size {
        return LocalState::Stale;
    }
    if quick {
        return LocalState::Current;
    }
    match md5_file(&entry.path) {
        Ok(md5) if md5 == entry.md5 => LocalState::Current,
        _ => LocalState::Stale,
    }
}

/// Classify every entry, in parallel, with a progress bar over the bytes read.
pub fn classify_all(entries: &[FileEntry], quick: bool) -> Vec<LocalState> {
    let total: u64 = entries.iter().map(|e| e.size).sum();
    let pb = make_pb(if quick { 0 } else { total });
    pb.set_style(
        ProgressStyle::with_template("  {bytes}/{total_bytes} [{bar:30}] {bytes_per_sec}, ETA {eta}")
            .unwrap()
            .progress_chars("=>-"),
    );
    let out = entries
        .par_iter()
        .map(|e| {
            let s = classify(e, quick);
            pb.inc(e.size);
            s
        })
        .collect();
    pb.finish_and_clear();
    out
}

/// Fetch one entry into place: resume into `<path>.part`, verify the md5, then
/// rename. The file the caller sees is therefore either absent or complete.
pub fn fetch_file(base: &str, entry: &FileEntry) -> Result<(), String> {
    let url = format!("{}/{}", base, entry.path);
    let tmp = format!("{}.part", entry.path);
    if let Some(parent) = Path::new(&entry.path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }

    let mut attempt = 0;
    loop {
        attempt += 1;
        let outcome = fetch_once(&url, &tmp).and_then(|()| {
            let md5 = md5_file(&tmp)?;
            if md5 == entry.md5 {
                Ok(())
            } else {
                // A bad resume or a truncated transfer: drop the partial file so
                // the next attempt starts clean rather than appending to it.
                let _ = std::fs::remove_file(&tmp);
                Err(format!("md5 mismatch (got {md5}, expected {})", entry.md5))
            }
        });
        match outcome {
            Ok(()) => {
                return std::fs::rename(&tmp, &entry.path)
                    .map_err(|e| format!("{}: rename: {e}", entry.path));
            }
            Err(_) if attempt < MAX_ATTEMPTS => {
                std::thread::sleep(Duration::from_secs(2 * attempt as u64));
            }
            Err(e) => return Err(format!("{}: {e} (after {attempt} attempts)", entry.path)),
        }
    }
}

/// One transfer attempt, resuming from whatever `<path>.part` already holds.
fn fetch_once(url: &str, tmp: &str) -> Result<(), String> {
    let offset = std::fs::metadata(tmp).map(|m| m.len()).unwrap_or(0);
    let mut req = minreq::get(url)
        .with_max_redirects(5) // the bucket answers with a 302 to a signed CDN URL
        .with_timeout(HTTP_TIMEOUT_SECS);
    if offset > 0 {
        req = req.with_header("Range", format!("bytes={}-", offset));
    }

    let resp = req.send_lazy().map_err(|e| e.to_string())?;
    let status = resp.status_code;
    // 416 on a resume means the partial file is already the whole object; let
    // the md5 check judge it.
    if offset > 0 && status == 416 {
        return Ok(());
    }
    if status != 200 && status != 206 {
        return Err(format!("HTTP {status}"));
    }

    // A ranged request answered with 200 means the range was ignored (a
    // redirect may drop the header); rewrite from scratch instead of appending.
    let truncate = offset > 0 && status != 206;
    let mut oo = OpenOptions::new();
    oo.create(true).write(true);
    if truncate {
        oo.truncate(true);
    } else {
        oo.append(true);
    }
    let mut file = oo.open(tmp).map_err(|e| e.to_string())?;
    let mut reader = resp;
    io::copy(&mut reader, &mut file).map_err(|e| e.to_string())?;
    Ok(())
}

/// Fetch every entry, `jobs` at a time. Returns the failures; the caller
/// decides how loud to be about them.
pub fn fetch_all(base: &str, entries: &[FileEntry], jobs: usize) -> Vec<String> {
    let total: u64 = entries.iter().map(|e| e.size).sum();
    let pb = make_pb(total);
    pb.set_style(
        ProgressStyle::with_template("  {bytes}/{total_bytes} [{bar:30}] {bytes_per_sec}, ETA {eta}")
            .unwrap()
            .progress_chars("=>-"),
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("build thread pool");
    let errs: Vec<String> = pool.install(|| {
        entries
            .par_iter()
            .filter_map(|e| {
                let r = fetch_file(base, e);
                pb.inc(e.size);
                r.err()
            })
            .collect()
    });
    pb.finish_and_clear();
    errs
}

// ---------------------------------------------------------------------------
// Uploading
// ---------------------------------------------------------------------------

/// Uploads go through the `hf` CLI, which holds the credentials. Nothing in
/// this crate ever sees a token, and a reader needs none: the bucket is public
/// to read and authenticated to write.
pub fn require_auth() -> Result<String, String> {
    let out = Command::new("hf")
        .args(["auth", "whoami"])
        .output()
        .map_err(|e| format!("cannot run the 'hf' CLI ({e}); see https://hf.co/cli"))?;
    if !out.status.success() {
        return Err("not logged in to Hugging Face; run 'hf auth login'".to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Copy one local file into the bucket, at `remote` relative to its root.
pub fn upload_file(bucket: &str, local: &str, remote: &str) -> Result<(), String> {
    let dst = format!("hf://buckets/{bucket}/{remote}");
    let out = Command::new("hf")
        .args(["buckets", "cp", local, &dst])
        .output()
        .map_err(|e| format!("{local}: cannot run the 'hf' CLI ({e})"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail = err.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
        return Err(format!("{local}: upload failed: {tail}"));
    }
    Ok(())
}

/// Upload every path, `jobs` at a time. Returns the failures.
pub fn upload_all(bucket: &str, paths: &[String], jobs: usize) -> Vec<String> {
    let pb = make_pb(paths.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("  {pos}/{len} [{bar:30}] ETA {eta}")
            .unwrap()
            .progress_chars("=>-"),
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("build thread pool");
    let errs = pool.install(|| {
        paths
            .par_iter()
            .filter_map(|p| {
                let r = upload_file(bucket, p, p);
                pb.inc(1);
                r.err()
            })
            .collect()
    });
    pb.finish_and_clear();
    errs
}

/// Delete one object. Always one explicit path: the CLI's recursive mode takes
/// a prefix, and on an unversioned bucket a mistyped prefix is unrecoverable.
pub fn delete_file(bucket: &str, path: &str) -> Result<(), String> {
    let target = format!("hf://buckets/{bucket}/{path}");
    let out = Command::new("hf")
        .args(["buckets", "rm", &target, "--yes"])
        .output()
        .map_err(|e| format!("{path}: cannot run the 'hf' CLI ({e})"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail = err.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
        return Err(format!("{path}: delete failed: {tail}"));
    }
    Ok(())
}

/// Delete every path, `jobs` at a time. Returns the failures.
pub fn delete_all(bucket: &str, paths: &[String], jobs: usize) -> Vec<String> {
    let pb = make_pb(paths.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("  {pos}/{len} [{bar:30}] ETA {eta}")
            .unwrap()
            .progress_chars("=>-"),
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("build thread pool");
    let errs = pool.install(|| {
        paths
            .par_iter()
            .filter_map(|p| {
                let r = delete_file(bucket, p);
                pb.inc(1);
                r.err()
            })
            .collect()
    });
    pb.finish_and_clear();
    errs
}
