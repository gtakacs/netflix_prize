//! The mean of several prediction columns as one column (e.g. the seed runs of a model). For every set
//! file `NAME1.<set>.npy` in DIR (train / probe / fulltrain / qual and the split_2 / split_3 sets; not
//! the ifeat / ufeat artifacts or the epoch snapshots) that every NAME has, writes `OUT.<set>.npy`:
//! the mean computed in f64, stored as f32.
//!
//!   ./target/release/predavg preds_1 s1-txseq-zbq9 s1-txseq-zbq{1..9}

use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 || args[0] == "-h" || args[0] == "--help" {
        eprintln!("usage: predavg DIR OUT NAME...");
        std::process::exit(2);
    }
    let (dir, out, names) = (&args[0], &args[1], &args[2..]);
    let mut n_sets = 0;
    for set in ["train", "probe", "fulltrain", "qual", "train_2", "probe_2", "train_3", "probe_3"] {
        let paths: Vec<String> = names.iter().map(|n| format!("{dir}/{n}.{set}.npy")).collect();
        if !paths.iter().all(|p| std::path::Path::new(p).exists()) { continue; }
        let mut acc: Option<Array1<f64>> = None;
        for p in &paths {
            let a: Array1<f32> = read_npy(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
            let a = a.mapv(|v| v as f64);
            acc = Some(match acc { None => a, Some(s) => { assert_eq!(s.len(), a.len(), "{p}: length"); s + a } });
        }
        let mean = acc.unwrap().mapv(|v| (v / names.len() as f64) as f32);
        let path = format!("{dir}/{out}.{set}.npy");
        write_npy(&path, &mean).unwrap_or_else(|e| panic!("write {path}: {e}"));
        println!("{path}: mean of {} columns, {} rows", names.len(), mean.len());
        n_sets += 1;
    }
    if n_sets == 0 { eprintln!("no set file is present for every NAME"); std::process::exit(1); }
}
