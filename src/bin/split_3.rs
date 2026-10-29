//! Build split_3: probe *extended* by train ratings, so that the probe set is as
//! hard as qual in terms of same-day evidence. The result is written as
//! data/train_3 and data/probe_3.
//!
//! A qual rating (u, i, d) has another rating of u on day d in fulltrain in 58.8%
//! of the cases, a probe rating in train only in 37.0%. The gap comes entirely
//! from probe ratings landing on qual days (qual vs train alone is also 37.0%),
//! and it sits mostly at 1-2 same-day partners. So the added ratings are chosen
//! to always keep a same-day partner in train, preferring thin days:
//!
//! * only users with a probe rating, only their last K train days;
//! * on each such day one random rating always stays in train, so no day (and
//!   no probe rating on it) loses its same-day evidence;
//! * the rest are candidates with weight 1/(c-1), c = train ratings that day;
//! * `ADD_FRAC` * |probe| of them are drawn without replacement
//!   (Efraimidis-Spirakis keys) and moved to probe.
//!
//! +25% lifts the same-day coverage from 37.0% to 49.6%. Matching qual's 58.8%
//! would need about +53%, because the probe ratings without a partner alone
//! exceed qual's 41.2% share at any smaller size.
//!
//! K sets the difficulty: older days have more train ratings after them, so
//! their ratings are easier. With a 20-factor MF (`lab-mf3`), probe_3 minus quiz
//! RMSE at epoch 20 is -0.0031 (K=10), -0.0014 (K=5), -0.0001 (K=2) and
//! -0.0006 (K=1, all from the last day, often the first probe day); seed noise
//! is about 0.0003. K=2 matches quiz best.
//!
//! `split_3 [SEED] [K]`: a seed other than the default 42, or a K other than
//! the default 2, writes to data/{train,probe}_3[s<SEED>][k<K>] instead, for
//! checking sensitivity to them.

use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::fs;

const N_USERS: usize = 480_189;
const N_ITEMS: usize = 17_770;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_K_DAYS: usize = 2;
const ADD_FRAC: f64 = 0.25;
const DATE_MUL: u32 = 2500;

const TRAIN_DIR: &str = "data/train";
const PROBE_DIR: &str = "data/probe";

fn load_i32(path: &str) -> Vec<i32> {
    let a: Array1<i32> = read_npy(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    a.to_vec()
}
fn load_i16(path: &str) -> Vec<i16> {
    let a: Array1<i16> = read_npy(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    a.to_vec()
}
fn load_i8(path: &str) -> Vec<i8> {
    let a: Array1<i8> = read_npy(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    a.to_vec()
}

/// Share of `(users, dates)` with another rating of the same user on the same
/// day among the `keys` (sorted (user, date) keys of the training side).
fn same_day_coverage(users: &[i32], dates: &[i16], keys: &[u32]) -> f64 {
    let hit = users.iter().zip(dates)
        .filter(|&(&u, &d)| keys.binary_search(&((u as u32) * DATE_MUL + d as u32)).is_ok())
        .count();
    hit as f64 / users.len() as f64
}

fn ud_keys(users: &[i32], dates: &[i16], keep: impl Fn(usize) -> bool) -> Vec<u32> {
    let mut k: Vec<u32> = (0..users.len())
        .filter(|&k| keep(k))
        .map(|k| (users[k] as u32) * DATE_MUL + dates[k] as u32)
        .collect();
    k.sort_unstable();
    k.dedup();
    k
}

fn main() {
    let seed: u64 = std::env::args().nth(1)
        .map(|a| a.parse().expect("SEED must be an unsigned integer"))
        .unwrap_or(DEFAULT_SEED);
    let k_days: usize = std::env::args().nth(2)
        .map(|a| a.parse().expect("K must be a positive integer"))
        .unwrap_or(DEFAULT_K_DAYS);
    assert!(k_days > 0, "K must be a positive integer");
    let mut suffix = String::new();
    if seed != DEFAULT_SEED { suffix.push_str(&format!("s{seed}")); }
    if k_days != DEFAULT_K_DAYS { suffix.push_str(&format!("k{k_days}")); }
    let train3_out = format!("data/train_3{suffix}");
    let probe3_out = format!("data/probe_3{suffix}");
    println!("Seed {seed}, K {k_days} -> {train3_out}, {probe3_out}");

    println!("Phase 1/5: Load probe & train");
    let pu = load_i32(&format!("{}/user_idxs.npy", PROBE_DIR));
    let pi = load_i16(&format!("{}/item_idxs.npy", PROBE_DIR));
    let pr = load_i8(&format!("{}/ratings.npy", PROBE_DIR));
    let pd = load_i16(&format!("{}/dates.npy", PROBE_DIR));
    let n_probe = pu.len();

    let tu = load_i32(&format!("{}/user_idxs.npy", TRAIN_DIR));
    let ti = load_i16(&format!("{}/item_idxs.npy", TRAIN_DIR));
    let tr = load_i8(&format!("{}/ratings.npy", TRAIN_DIR));
    let td = load_i16(&format!("{}/dates.npy", TRAIN_DIR));
    let n_train = tu.len();
    println!("  probe: {}, train: {}", n_probe, n_train);

    println!("Phase 2/5: Order train ratings of probe users, newest day first");
    let mut in_probe = vec![false; N_USERS];
    for &u in &pu {
        in_probe[u as usize] = true;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    // (user, newest day first, random within the day): the first rating of
    // each day is the one that stays in train.
    let mut order: Vec<(i32, i16, u32, u32)> = (0..n_train)
        .filter(|&k| in_probe[tu[k] as usize])
        .map(|k| (tu[k], -td[k], rng.random::<u32>(), k as u32))
        .collect();
    order.sort_unstable();

    println!("Phase 3/5: Draw candidates from each user's last {} train days", k_days);
    // (Efraimidis-Spirakis key, train row); the largest keys are moved.
    let mut cands: Vec<(f64, u32)> = Vec::new();
    let mut g = 0;
    while g < order.len() {
        let u = order[g].0;
        let mut day_rank = 0;
        while g < order.len() && order[g].0 == u {
            let d = order[g].1;
            let mut e = g;
            while e < order.len() && order[e].0 == u && order[e].1 == d {
                e += 1;
            }
            let c = e - g;
            if day_rank < k_days {
                let w = 1.0 / (c - 1).max(1) as f64;
                for o in &order[g + 1..e] {
                    let x: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                    cands.push((x.ln() / w, o.3));
                }
            }
            day_rank += 1;
            g = e;
        }
    }
    drop(order);
    let n_add = (ADD_FRAC * n_probe as f64).round() as usize;
    assert!(cands.len() >= n_add, "only {} candidates for {} moves", cands.len(), n_add);
    cands.select_nth_unstable_by(n_add, |a, b| b.0.total_cmp(&a.0));
    let mut moved = vec![false; n_train];
    for &(_, k) in &cands[..n_add] {
        moved[k as usize] = true;
    }
    println!("  candidates: {}, moved: {} ({:.2}% of probe)",
             cands.len(), n_add, n_add as f64 / n_probe as f64 * 100.0);
    drop(cands);

    println!("Phase 4/5: Build train_3 & probe_3");
    // train_3 = train \ moved, in train's order
    let keep: Vec<usize> = (0..n_train).filter(|&k| !moved[k]).collect();
    let train3_user: Vec<i32> = keep.iter().map(|&k| tu[k]).collect();
    let train3_item: Vec<i16> = keep.iter().map(|&k| ti[k]).collect();
    let train3_rate: Vec<i8> = keep.iter().map(|&k| tr[k]).collect();
    let train3_date: Vec<i16> = keep.iter().map(|&k| td[k]).collect();
    drop(keep);
    let n_train3 = train3_user.len();

    // probe_3 = probe ∪ moved, sorted by (item, user)
    let mut px_user = pu.clone();
    let mut px_item = pi.clone();
    let mut px_rate = pr.clone();
    let mut px_date = pd.clone();
    for k in (0..n_train).filter(|&k| moved[k]) {
        px_user.push(tu[k]);
        px_item.push(ti[k]);
        px_rate.push(tr[k]);
        px_date.push(td[k]);
    }
    let n_probe3 = px_user.len();
    let mut perm: Vec<u32> = (0..n_probe3 as u32).collect();
    perm.sort_unstable_by_key(|&k| (px_item[k as usize], px_user[k as usize]));
    let probe3_user: Vec<i32> = perm.iter().map(|&k| px_user[k as usize]).collect();
    let probe3_item: Vec<i16> = perm.iter().map(|&k| px_item[k as usize]).collect();
    let probe3_rate: Vec<i8> = perm.iter().map(|&k| px_rate[k as usize]).collect();
    let probe3_date: Vec<i16> = perm.iter().map(|&k| px_date[k as usize]).collect();
    drop(perm);

    let train_keys = ud_keys(&tu, &td, |_| true);
    let train3_keys = ud_keys(&train3_user, &train3_date, |_| true);
    println!("  same-day coverage: probe|train {:.4}, probe|train_3 {:.4}, probe_3|train_3 {:.4}",
             same_day_coverage(&pu, &pd, &train_keys),
             same_day_coverage(&pu, &pd, &train3_keys),
             same_day_coverage(&probe3_user, &probe3_date, &train3_keys));

    let mut t3_user_cnts = vec![0i32; N_USERS];
    let mut t3_item_cnts = vec![0i32; N_ITEMS];
    for k in 0..n_train3 {
        t3_user_cnts[train3_user[k] as usize] += 1;
        t3_item_cnts[train3_item[k] as usize] += 1;
    }
    let mut p3_user_cnts = vec![0i32; N_USERS];
    let mut p3_item_cnts = vec![0i32; N_ITEMS];
    for k in 0..n_probe3 {
        p3_user_cnts[probe3_user[k] as usize] += 1;
        p3_item_cnts[probe3_item[k] as usize] += 1;
    }

    // shared arrays from train
    let item_years: Vec<i32> = load_i32(&format!("{}/item_years.npy", TRAIN_DIR));
    let uidx_to_uid: Vec<i32> = load_i32(&format!("{}/uidx_to_uid.npy", TRAIN_DIR));
    let iidx_to_iid: Vec<i16> = load_i16(&format!("{}/iidx_to_iid.npy", TRAIN_DIR));
    let t3_is_test = vec![0i8; n_train3];
    let p3_is_test = vec![0i8; n_probe3];

    println!("Phase 5/5: Save");
    fs::create_dir_all(&train3_out).unwrap();
    fs::create_dir_all(&probe3_out).unwrap();

    macro_rules! save {
        ($dir:expr, $name:expr, $vec:expr) => {
            write_npy(format!("{}/{}.npy", $dir, $name), &Array1::from($vec)).unwrap();
        };
    }

    save!(&train3_out, "user_idxs", train3_user);
    save!(&train3_out, "item_idxs", train3_item);
    save!(&train3_out, "ratings", train3_rate);
    save!(&train3_out, "dates", train3_date);
    save!(&train3_out, "is_test", t3_is_test);
    save!(&train3_out, "user_cnts", t3_user_cnts);
    save!(&train3_out, "item_cnts", t3_item_cnts);
    save!(&train3_out, "item_years", item_years.clone());
    save!(&train3_out, "uidx_to_uid", uidx_to_uid);
    save!(&train3_out, "iidx_to_iid", iidx_to_iid);

    save!(&probe3_out, "user_idxs", probe3_user);
    save!(&probe3_out, "item_idxs", probe3_item);
    save!(&probe3_out, "ratings", probe3_rate);
    save!(&probe3_out, "dates", probe3_date);
    save!(&probe3_out, "is_test", p3_is_test);
    save!(&probe3_out, "user_cnts", p3_user_cnts);
    save!(&probe3_out, "item_cnts", p3_item_cnts);
    save!(&probe3_out, "item_years", item_years);

    println!("Done. train_3: {} ratings, probe_3: {} ratings", n_train3, n_probe3);
}
