//! Convert the original Netflix Prize dataset into the four numpy datasets the
//! rest of the pipeline uses: `train`, `probe`, `fulltrain`, `qual`. Runs after
//! the `download` job, reading `data/raw/nf_prize_dataset.tar.gz` directly.
//!
//! The small, human-readable members (`README`, `movie_titles.csv`, `probe.txt`,
//! `qualifying.txt`) are first extracted to `data/raw/`; the bulky training
//! ratings stay in the archive and are streamed out of the inner
//! `training_set.tar` in Phase 1. probe/qualifying/movie_titles are then parsed
//! from the extracted files. Qual ratings and is_test (in neither the archive
//! nor the Kaggle release) are read from `data/qual_ratings/qual_ratings.csv.gz`.

use flate2::read::GzDecoder;
use indicatif::ProgressIterator;
use ndarray::Array1;
use ndarray_npy::write_npy;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use tar::Archive;

const N_USERS: usize = 480_189;
const N_ITEMS: usize = 17_770;
const N_TOTAL: usize = 100_480_507;
const N_PROBE: usize = 1_408_395;
const N_QUAL: usize = 2_817_131;
const ARCHIVE: &str = "data/raw/nf_prize_dataset.tar.gz";
const RAW_DIR: &str = "data/raw";
const QUAL_RATINGS_CSV_GZ: &str = "data/qual_ratings/qual_ratings.csv.gz";
const TRAIN_OUT: &str = "data/train";
const PROBE_OUT: &str = "data/probe";
const FULLTRAIN_OUT: &str = "data/fulltrain";
const QUAL_OUT: &str = "data/qual";

const YEAR_OFFSETS: [i32; 8] = [0, 365, 731, 1096, 1461, 1826, 2192, 2557];
const MONTH_OFFSETS_NORMAL: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
const MONTH_OFFSETS_LEAP: [i32; 12] = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
const ORIGIN_OFFSET: i32 = 304 + 10;

#[inline]
fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[inline]
fn parse_date_bytes(b: &[u8]) -> i16 {
    let y = (b[0] - b'0') as i32 * 1000
        + (b[1] - b'0') as i32 * 100
        + (b[2] - b'0') as i32 * 10
        + (b[3] - b'0') as i32;
    let m = (b[5] - b'0') as i32 * 10 + (b[6] - b'0') as i32;
    let d = (b[8] - b'0') as i32 * 10 + (b[9] - b'0') as i32;
    let year_off = YEAR_OFFSETS[(y - 1999) as usize];
    let month_off = if is_leap(y) {
        MONTH_OFFSETS_LEAP[(m - 1) as usize]
    } else {
        MONTH_OFFSETS_NORMAL[(m - 1) as usize]
    };
    (year_off + month_off + d - 1 - ORIGIN_OFFSET) as i16
}

#[inline]
fn parse_u32(b: &[u8]) -> u32 {
    let mut v = 0u32;
    for &c in b {
        v = v * 10 + (c - b'0') as u32;
    }
    v
}

fn rstrip(line: &[u8]) -> &[u8] {
    let mut len = line.len();
    while len > 0 && (line[len - 1] == b'\n' || line[len - 1] == b'\r') {
        len -= 1;
    }
    &line[..len]
}

#[inline]
fn pack_key(uidx: i32, iidx: i16) -> u64 {
    ((uidx as u64) << 16) | (iidx as u64 & 0xFFFF)
}

fn read_lines_from<R: BufRead, F: FnMut(&[u8])>(reader: &mut R, mut handler: F) {
    let mut buf = Vec::with_capacity(64);
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf).unwrap();
        if n == 0 {
            break;
        }
        handler(rstrip(&buf));
    }
}

fn read_lines_gz<F: FnMut(&[u8])>(path: &str, handler: F) {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let gz: Box<dyn Read> = Box::new(GzDecoder::new(f));
    let mut reader = BufReader::with_capacity(1 << 20, gz);
    read_lines_from(&mut reader, handler);
}

/// Open the dataset archive (gzipped tar) and return an iterator over its
/// top-level members.
fn open_archive() -> Archive<GzDecoder<File>> {
    let f = File::open(ARCHIVE).unwrap_or_else(|e| panic!("open {}: {}", ARCHIVE, e));
    Archive::new(GzDecoder::new(f))
}

fn read_lines<F: FnMut(&[u8])>(path: &str, handler: F) {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mut reader = BufReader::with_capacity(1 << 20, f);
    read_lines_from(&mut reader, handler);
}

/// Extract the small, human-readable members of the archive to `data/raw/`:
/// `README`, `movie_titles.csv` (renamed from `.txt`), `probe.txt`,
/// `qualifying.txt`. They sit before `training_set.tar` in the archive, so this
/// is a cheap partial pass; the bulky training ratings are never materialised.
fn extract_small_files() {
    fs::create_dir_all(RAW_DIR).unwrap();
    let mut ar = open_archive();
    let mut n = 0;
    for entry in ar.entries().unwrap() {
        let mut entry = entry.unwrap();
        let name = entry
            .path()
            .unwrap()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let dst = match name.as_str() {
            "README" => format!("{}/README", RAW_DIR),
            "movie_titles.txt" => format!("{}/movie_titles.csv", RAW_DIR),
            "probe.txt" => format!("{}/probe.txt", RAW_DIR),
            "qualifying.txt" => format!("{}/qualifying.txt", RAW_DIR),
            _ => continue,
        };
        println!("  writing {}", dst);
        let mut out = File::create(&dst).unwrap_or_else(|e| panic!("create {}: {}", dst, e));
        io::copy(&mut entry, &mut out).unwrap_or_else(|e| panic!("write {}: {}", dst, e));
        n += 1;
    }
    assert_eq!(n, 4, "expected 4 small members in {}, extracted {}", ARCHIVE, n);
}

/// Feed each line of every per-movie file inside the inner `training_set.tar` to
/// `handler`. The blocks are `<id>:`-headed, so a single closure carrying the
/// current movie across files reconstructs the combined-data stream in order.
fn read_training<F: FnMut(&[u8])>(mut handler: F) {
    let mut outer = open_archive();
    for entry in outer.entries().unwrap() {
        let entry = entry.unwrap();
        let name = entry
            .path()
            .unwrap()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if name != "training_set.tar" {
            continue;
        }
        let mut inner = Archive::new(entry);
        for ie in inner.entries().unwrap() {
            let ie = ie.unwrap();
            let iname = ie
                .path()
                .unwrap()
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if !iname.starts_with("mv_") || !iname.ends_with(".txt") {
                continue; // directory entry or anything unexpected
            }
            let mut reader = BufReader::with_capacity(1 << 20, ie);
            read_lines_from(&mut reader, &mut handler);
        }
        return;
    }
    panic!("training_set.tar not found in {}", ARCHIVE);
}

fn main() {
    // ---- Phase 0: extract small members to data/raw/ ----
    println!("Phase 0/10: Extract raw files to {}/", RAW_DIR);
    extract_small_files();

    // ---- Phase 1: parse training_set.tar (combined-data blocks) ----
    println!("Phase 1/10: Parse combined_data");
    let mut user_ids: Vec<u32> = Vec::with_capacity(N_TOTAL);
    let mut item_ids: Vec<u16> = Vec::with_capacity(N_TOTAL);
    let mut ratings: Vec<i8> = Vec::with_capacity(N_TOTAL);
    let mut dates: Vec<i16> = Vec::with_capacity(N_TOTAL);

    let mut current_movie: u16 = 0;
    read_training(|line| {
        if line.last() == Some(&b':') {
            current_movie = parse_u32(&line[..line.len() - 1]) as u16;
        } else {
            let comma1 = line.iter().position(|&c| c == b',').unwrap();
            let comma2 = comma1
                + 1
                + line[comma1 + 1..]
                    .iter()
                    .position(|&c| c == b',')
                    .unwrap();
            let uid = parse_u32(&line[..comma1]);
            let r = (line[comma1 + 1] - b'0') as i8;
            let date = parse_date_bytes(&line[comma2 + 1..]);
            user_ids.push(uid);
            item_ids.push(current_movie);
            ratings.push(r);
            dates.push(date);
        }
    });
    assert_eq!(user_ids.len(), N_TOTAL);

    // ---- Phase 2: user remapping + index conversion ----
    println!("Phase 2/10: Build user remapping & remap IDs");
    let mut sorted_uids: Vec<u32> = user_ids.iter().copied().collect();
    sorted_uids.sort_unstable();
    sorted_uids.dedup();
    assert_eq!(sorted_uids.len(), N_USERS);

    let uid_to_uidx: HashMap<u32, i32> = sorted_uids
        .iter()
        .enumerate()
        .map(|(idx, &uid)| (uid, idx as i32))
        .collect();

    let mut user_idxs: Vec<i32> = Vec::with_capacity(N_TOTAL);
    let mut item_idxs: Vec<i16> = Vec::with_capacity(N_TOTAL);
    for k in (0..N_TOTAL).progress() {
        user_idxs.push(uid_to_uidx[&user_ids[k]]);
        item_idxs.push(item_ids[k] as i16 - 1);
    }
    drop(user_ids);
    drop(item_ids);

    // ---- Phase 3: parse probe.txt ----
    println!("Phase 3/10: Parse probe.txt");
    let mut probe_set: HashSet<u64> = HashSet::with_capacity(N_PROBE);
    let mut probe_groups: Vec<Vec<i32>> = (0..N_ITEMS).map(|_| Vec::new()).collect();
    let mut current_iidx: i16 = 0;
    read_lines(&format!("{}/probe.txt", RAW_DIR), |line| {
        if line.last() == Some(&b':') {
            current_iidx = (parse_u32(&line[..line.len() - 1]) as i16) - 1;
        } else {
            let uid = parse_u32(line);
            let uidx = uid_to_uidx[&uid];
            probe_set.insert(pack_key(uidx, current_iidx));
            probe_groups[current_iidx as usize].push(uidx);
        }
    });
    assert_eq!(probe_set.len(), N_PROBE);

    // ---- Phase 4: parse qualifying.txt (push directly in parse order) ----
    println!("Phase 4/10: Parse qualifying.txt");
    let mut qual_user_idxs: Vec<i32> = Vec::with_capacity(N_QUAL);
    let mut qual_item_idxs: Vec<i16> = Vec::with_capacity(N_QUAL);
    let mut qual_dates: Vec<i16> = Vec::with_capacity(N_QUAL);
    let mut current_iidx: i16 = 0;
    read_lines(&format!("{}/qualifying.txt", RAW_DIR), |line| {
        if line.last() == Some(&b':') {
            current_iidx = (parse_u32(&line[..line.len() - 1]) as i16) - 1;
        } else {
            let comma = line.iter().position(|&c| c == b',').unwrap();
            let uid = parse_u32(&line[..comma]);
            let uidx = uid_to_uidx[&uid];
            let date = parse_date_bytes(&line[comma + 1..]);
            qual_user_idxs.push(uidx);
            qual_item_idxs.push(current_iidx);
            qual_dates.push(date);
        }
    });
    assert_eq!(qual_user_idxs.len(), N_QUAL);
    drop(uid_to_uidx);

    // ---- Phase 5: split mask + probe lookup ----
    println!("Phase 5/10: Split train/probe");
    let mut is_probe: Vec<bool> = Vec::with_capacity(N_TOTAL);
    let mut probe_lookup: HashMap<u64, (i8, i16)> = HashMap::with_capacity(N_PROBE);
    for k in (0..N_TOTAL).progress() {
        let key = pack_key(user_idxs[k], item_idxs[k]);
        let in_probe = probe_set.contains(&key);
        is_probe.push(in_probe);
        if in_probe {
            probe_lookup.insert(key, (ratings[k], dates[k]));
        }
    }
    drop(probe_set);

    let n_train = N_TOTAL - N_PROBE;

    // ---- Phase 6: sort all + build train + fulltrain ----
    println!("Phase 6/10: Sort all + build train + fulltrain");
    let mut all_idx: Vec<usize> = (0..N_TOTAL).collect();
    all_idx.sort_unstable_by_key(|&k| (user_idxs[k], dates[k], item_idxs[k]));

    let mut fulltrain_user_idxs: Vec<i32> = Vec::with_capacity(N_TOTAL);
    let mut fulltrain_item_idxs: Vec<i16> = Vec::with_capacity(N_TOTAL);
    let mut fulltrain_ratings: Vec<i8> = Vec::with_capacity(N_TOTAL);
    let mut fulltrain_dates: Vec<i16> = Vec::with_capacity(N_TOTAL);
    let mut train_user_idxs: Vec<i32> = Vec::with_capacity(n_train);
    let mut train_item_idxs: Vec<i16> = Vec::with_capacity(n_train);
    let mut train_ratings: Vec<i8> = Vec::with_capacity(n_train);
    let mut train_dates: Vec<i16> = Vec::with_capacity(n_train);

    for &k in all_idx.iter().progress() {
        let u = user_idxs[k];
        let i = item_idxs[k];
        let r = ratings[k];
        let d = dates[k];
        fulltrain_user_idxs.push(u);
        fulltrain_item_idxs.push(i);
        fulltrain_ratings.push(r);
        fulltrain_dates.push(d);
        if !is_probe[k] {
            train_user_idxs.push(u);
            train_item_idxs.push(i);
            train_ratings.push(r);
            train_dates.push(d);
        }
    }
    drop(all_idx);

    // ---- Phase 7: build probe arrays ----
    println!("Phase 7/10: Build probe arrays");
    let mut probe_user_idxs: Vec<i32> = Vec::with_capacity(N_PROBE);
    let mut probe_item_idxs: Vec<i16> = Vec::with_capacity(N_PROBE);
    let mut probe_ratings: Vec<i8> = Vec::with_capacity(N_PROBE);
    let mut probe_dates: Vec<i16> = Vec::with_capacity(N_PROBE);
    for iidx in (0..(N_ITEMS as i16)).progress() {
        for &uidx in &probe_groups[iidx as usize] {
            let (r, d) = probe_lookup[&pack_key(uidx, iidx)];
            probe_user_idxs.push(uidx);
            probe_item_idxs.push(iidx);
            probe_ratings.push(r);
            probe_dates.push(d);
        }
    }
    drop(probe_lookup);
    drop(probe_groups);
    drop(user_idxs);
    drop(item_idxs);
    drop(ratings);
    drop(dates);
    drop(is_probe);

    // ---- Phase 8: load qual_ratings.csv.gz ----
    println!("Phase 8/10: Load qual_ratings.csv.gz");
    let mut qual_ratings: Vec<i8> = Vec::with_capacity(N_QUAL);
    let mut qual_is_test: Vec<i8> = Vec::with_capacity(N_QUAL);
    let mut header_seen = false;
    read_lines_gz(QUAL_RATINGS_CSV_GZ, |line| {
        if !header_seen {
            header_seen = true;
            return;
        }
        let comma = line.iter().position(|&c| c == b',').unwrap();
        let r = parse_u32(&line[..comma]) as i8;
        let t = parse_u32(&line[comma + 1..]) as i8;
        qual_ratings.push(r);
        qual_is_test.push(t);
    });
    assert_eq!(qual_ratings.len(), N_QUAL);
    assert_eq!(qual_is_test.len(), N_QUAL);

    // ---- Phase 9: counts + item_years ----
    println!("Phase 9/10: Counts & item_years");
    let mut train_user_cnts = vec![0i32; N_USERS];
    let mut train_item_cnts = vec![0i32; N_ITEMS];
    for k in (0..n_train).progress() {
        train_user_cnts[train_user_idxs[k] as usize] += 1;
        train_item_cnts[train_item_idxs[k] as usize] += 1;
    }
    let mut probe_user_cnts = vec![0i32; N_USERS];
    let mut probe_item_cnts = vec![0i32; N_ITEMS];
    for k in (0..N_PROBE).progress() {
        probe_user_cnts[probe_user_idxs[k] as usize] += 1;
        probe_item_cnts[probe_item_idxs[k] as usize] += 1;
    }
    let mut fulltrain_user_cnts = vec![0i32; N_USERS];
    let mut fulltrain_item_cnts = vec![0i32; N_ITEMS];
    for k in (0..N_TOTAL).progress() {
        fulltrain_user_cnts[fulltrain_user_idxs[k] as usize] += 1;
        fulltrain_item_cnts[fulltrain_item_idxs[k] as usize] += 1;
    }
    let mut qual_user_cnts = vec![0i32; N_USERS];
    let mut qual_item_cnts = vec![0i32; N_ITEMS];
    for k in (0..N_QUAL).progress() {
        qual_user_cnts[qual_user_idxs[k] as usize] += 1;
        qual_item_cnts[qual_item_idxs[k] as usize] += 1;
    }

    let mut item_years = vec![0i32; N_ITEMS];
    read_lines(&format!("{}/movie_titles.csv", RAW_DIR), |line| {
        let c1 = line.iter().position(|&c| c == b',').unwrap();
        let rest = &line[c1 + 1..];
        let c2 = rest.iter().position(|&c| c == b',').unwrap();
        let id = parse_u32(&line[..c1]) as usize;
        let year_str = &rest[..c2];
        let yr: i32 = if year_str == b"NULL" {
            0
        } else {
            std::str::from_utf8(year_str).unwrap().parse().unwrap()
        };
        item_years[id - 1] = yr;
    });
    item_years[4387] = 2001;
    item_years[4793] = 2001;
    item_years[7240] = 2001;
    item_years[10781] = 1974;
    item_years[15917] = 1999;
    item_years[16677] = 2004;
    item_years[17666] = 1999;

    // ---- Phase 10: save ----
    println!("Phase 10/10: Save");
    fs::create_dir_all(TRAIN_OUT).unwrap();
    fs::create_dir_all(PROBE_OUT).unwrap();
    fs::create_dir_all(FULLTRAIN_OUT).unwrap();
    fs::create_dir_all(QUAL_OUT).unwrap();

    let uidx_to_uid: Vec<i32> = sorted_uids.iter().map(|&x| x as i32).collect();
    let iidx_to_iid: Vec<i16> = (1..=N_ITEMS as i16).collect();
    let train_is_test = vec![0i8; n_train];
    let probe_is_test = vec![0i8; N_PROBE];
    let fulltrain_is_test = vec![0i8; N_TOTAL];

    macro_rules! save {
        ($dir:expr, $name:expr, $vec:expr) => {
            write_npy(format!("{}/{}.npy", $dir, $name), &Array1::from($vec)).unwrap();
        };
    }

    save!(TRAIN_OUT, "user_idxs", train_user_idxs);
    save!(TRAIN_OUT, "item_idxs", train_item_idxs);
    save!(TRAIN_OUT, "ratings", train_ratings);
    save!(TRAIN_OUT, "dates", train_dates);
    save!(TRAIN_OUT, "is_test", train_is_test);
    save!(TRAIN_OUT, "user_cnts", train_user_cnts);
    save!(TRAIN_OUT, "item_cnts", train_item_cnts);
    save!(TRAIN_OUT, "item_years", item_years.clone());
    save!(TRAIN_OUT, "uidx_to_uid", uidx_to_uid.clone());
    save!(TRAIN_OUT, "iidx_to_iid", iidx_to_iid.clone());

    save!(PROBE_OUT, "user_idxs", probe_user_idxs);
    save!(PROBE_OUT, "item_idxs", probe_item_idxs);
    save!(PROBE_OUT, "ratings", probe_ratings);
    save!(PROBE_OUT, "dates", probe_dates);
    save!(PROBE_OUT, "is_test", probe_is_test);
    save!(PROBE_OUT, "user_cnts", probe_user_cnts);
    save!(PROBE_OUT, "item_cnts", probe_item_cnts);
    save!(PROBE_OUT, "item_years", item_years.clone());

    save!(FULLTRAIN_OUT, "user_idxs", fulltrain_user_idxs);
    save!(FULLTRAIN_OUT, "item_idxs", fulltrain_item_idxs);
    save!(FULLTRAIN_OUT, "ratings", fulltrain_ratings);
    save!(FULLTRAIN_OUT, "dates", fulltrain_dates);
    save!(FULLTRAIN_OUT, "is_test", fulltrain_is_test);
    save!(FULLTRAIN_OUT, "user_cnts", fulltrain_user_cnts);
    save!(FULLTRAIN_OUT, "item_cnts", fulltrain_item_cnts);
    save!(FULLTRAIN_OUT, "item_years", item_years.clone());
    save!(FULLTRAIN_OUT, "uidx_to_uid", uidx_to_uid);
    save!(FULLTRAIN_OUT, "iidx_to_iid", iidx_to_iid);

    save!(QUAL_OUT, "user_idxs", qual_user_idxs);
    save!(QUAL_OUT, "item_idxs", qual_item_idxs);
    save!(QUAL_OUT, "ratings", qual_ratings);
    save!(QUAL_OUT, "dates", qual_dates);
    save!(QUAL_OUT, "is_test", qual_is_test);
    save!(QUAL_OUT, "user_cnts", qual_user_cnts);
    save!(QUAL_OUT, "item_cnts", qual_item_cnts);
    save!(QUAL_OUT, "item_years", item_years);

    println!(
        "Done. {}: {}, {}: {}, {}: {}, {}: {} ratings",
        TRAIN_OUT, n_train,
        PROBE_OUT, N_PROBE,
        FULLTRAIN_OUT, N_TOTAL,
        QUAL_OUT, N_QUAL,
    );
}
