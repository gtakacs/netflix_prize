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
use std::fs::File;
use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

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

/// The index covers the prediction files, not itself: an entry for the index
/// would describe the previous generation the moment the index is rewritten.
/// Snapshots under `index/` are left out for the same reason.
pub fn is_index_path(path: &str) -> bool {
    path == INDEX_FILE || path.starts_with("index/")
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
