//! Fetch the original Netflix Prize dataset archive. Runs as the very first
//! pipeline job (before `ingest`).
//!
//! The dataset is publicly available — no Kaggle account required — at
//! <https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz>.
//! It is downloaded with `minreq` (pure-Rust HTTP + rustls TLS, no system
//! dependencies) using a retry loop with HTTP-range resume, so a dropped
//! connection on the 665 MB file costs only the un-transferred tail rather than
//! a full restart. The result is md5-verified against the known digest before it
//! is published, so a truncated or wrong file is caught before `ingest` reads it.

use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::Path;
use std::thread;
use std::time::Duration;

const RAW_DIR: &str = "data/raw";
const ARCHIVE: &str = "data/raw/nf_prize_dataset.tar.gz";
const ARCHIVE_URL: &str =
    "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz";
const EXPECTED_MD5: &str = "a8f23d2d76461211c6b4c0ca6df2547d";
const MAX_ATTEMPTS: u32 = 8;
// Per-attempt overall timeout. minreq only offers a whole-request timeout (not a
// per-read one), so this is a generous backstop against a hung connection: if it
// trips mid-download, the retry loop simply resumes from the bytes already on
// disk, so even slow links converge across attempts.
const TIMEOUT_SECS: u64 = 1200;

fn main() {
    if Path::new(ARCHIVE).exists() {
        println!("Archive already present: {}", ARCHIVE);
        // Validate the local copy in place; it's the canonical file, so on a
        // mismatch we tell the user to remove it rather than deleting it for them.
        if let Err(got) = check_md5(ARCHIVE) {
            panic!(
                "md5 mismatch for {ARCHIVE}: got {got}, expected {EXPECTED_MD5}. \
                 Delete the file and re-run to re-download."
            );
        }
        println!("md5 ok: {EXPECTED_MD5}");
    } else {
        download();
    }
}

fn download() {
    fs::create_dir_all(RAW_DIR).unwrap();
    let tmp = format!("{}.part", ARCHIVE);
    println!("Downloading {} ...", ARCHIVE_URL);

    // Retry loop: each attempt resumes from whatever bytes the `.part` already
    // holds, so interruptions only cost the remaining tail. The partial file is
    // deliberately kept between attempts (and between job runs) to enable that.
    let mut attempt = 0;
    loop {
        attempt += 1;
        match fetch_once(&tmp) {
            Ok(()) => break,
            Err(e) if attempt < MAX_ATTEMPTS => {
                let have = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
                eprintln!(
                    "  attempt {attempt}/{MAX_ATTEMPTS} failed: {e}; \
                     have {have} bytes, retrying (resume) ..."
                );
                thread::sleep(Duration::from_secs(2 * attempt as u64));
            }
            Err(e) => panic!(
                "download failed after {MAX_ATTEMPTS} attempts: {e}. \
                 Partial file kept at {tmp} — re-run to resume, or delete it to start over."
            ),
        }
    }

    // Validate the assembled file before publishing it. A bad resume (e.g. a
    // server that ignored the range request) is caught here; drop the partial so
    // the next run starts clean rather than resuming onto corrupt bytes.
    if let Err(got) = check_md5(&tmp) {
        let _ = fs::remove_file(&tmp);
        panic!(
            "md5 mismatch for downloaded archive: got {got}, expected {EXPECTED_MD5}. \
             Removed the partial file; re-run to download again."
        );
    }
    fs::rename(&tmp, ARCHIVE).unwrap();
    println!("Saved {} (md5 ok)", ARCHIVE);
}

/// One download attempt: ask the server (via a `Range` header) to resume from
/// the current `.part` size and append the rest. Returns an error string on a
/// failed or interrupted transfer so the caller can retry.
fn fetch_once(tmp: &str) -> Result<(), String> {
    let offset = fs::metadata(tmp).map(|m| m.len()).unwrap_or(0);
    if offset > 0 {
        println!("  resuming from {} bytes", offset);
    }

    let mut req = minreq::get(ARCHIVE_URL)
        .with_max_redirects(5) // archive.org answers with a 302 to a storage node
        .with_timeout(TIMEOUT_SECS);
    if offset > 0 {
        req = req.with_header("Range", format!("bytes={}-", offset));
    }

    let resp = req.send_lazy().map_err(|e| e.to_string())?;
    let status = resp.status_code;

    // 416 Range Not Satisfiable with a resume offset means the `.part` is already
    // complete — treat as success and let the md5 check judge.
    if offset > 0 && status == 416 {
        return Ok(());
    }
    if status != 200 && status != 206 {
        return Err(format!("unexpected HTTP status {}", status));
    }

    // If we requested a range but got a full 200, the server ignored it; discard
    // the partial and rewrite from scratch instead of appending onto it.
    let truncate = offset > 0 && status != 206;
    let total = full_size(&resp);

    let mut oo = OpenOptions::new();
    oo.create(true).write(true);
    if truncate {
        oo.truncate(true);
    } else {
        oo.append(true);
    }
    let mut file = oo.open(tmp).map_err(|e| e.to_string())?;

    let pb = ProgressBar::new(total.unwrap_or(0));
    pb.set_style(
        ProgressStyle::with_template("  {bytes}/{total_bytes} [{bar:30}] {bytes_per_sec}, ETA {eta}")
            .unwrap()
            .progress_chars("=>-"),
    );
    if !truncate {
        pb.set_position(offset); // these bytes are already on disk from a prior attempt
    }

    let mut reader = pb.wrap_read(resp);
    let res = io::copy(&mut reader, &mut file);
    pb.finish_and_clear();
    res.map_err(|e| e.to_string())?;
    Ok(())
}

/// Full size of the file for the progress bar: from `Content-Range` (`.../TOTAL`)
/// on a `206` resume, or `Content-Length` on a full `200`. `None` if absent.
fn full_size(resp: &minreq::ResponseLazy) -> Option<u64> {
    if resp.status_code == 206 {
        resp.headers
            .get("content-range")
            .and_then(|v| v.rsplit('/').next())
            .and_then(|s| s.trim().parse::<u64>().ok())
    } else {
        resp.headers.get("content-length").and_then(|s| s.parse::<u64>().ok())
    }
}

/// Compute the md5 of `path` and compare to the expected digest. Returns the
/// computed hex digest as `Err` on mismatch.
fn check_md5(path: &str) -> Result<(), String> {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mut ctx = md5::Context::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        ctx.consume(&buf[..n]);
    }
    let got = format!("{:x}", ctx.compute());
    if got == EXPECTED_MD5 { Ok(()) } else { Err(got) }
}
