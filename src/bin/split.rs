//! Build the alternative train/probe splits from the original data/train and
//! data/probe (written by `ingest`):
//!
//!   split 2               data/train_2, data/probe_2
//!   split 3 [SEED] [K]    data/train_3, data/probe_3
//!   split 4               data/train_4, data/probe_4
//!   ... --out DIR         write under DIR instead of data/ (for checking)
//!
//! Every probe set is sorted by (item, user), as the original probe.
//!
//! **split_2** (ex `newsplit`): for each user with >1 probe rating, move one
//! randomly chosen probe rating into train, provided the chosen rating's date is
//! not already represented for that user in train and a 90% probability check
//! passes. Train is sorted by (user, item). The selection was first implemented
//! in Python and must stay bit-identical to it, so this file includes a port of
//! `np.random.default_rng(seed)`: numpy's SeedSequence and the PCG64
//! BitGenerator, plus the `random()` and `choice()` methods, verified to match
//! numpy output for the same seed.
//!
//! **split_4**: the same ratings as split_2 (the same selection), but train is
//! sorted by (user, date, item), like train, train_3 and fulltrain, so a model
//! visits a user's train ratings in time order on both of its phases.
//!
//! **split_3**: probe *extended* by train ratings, so that the probe set is as
//! hard as qual in terms of same-day evidence. Train keeps train's order.
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
//! A seed other than the default 42, or a K other than the default 2, writes
//! to data/{train,probe}_3[s<SEED>][k<K>] instead, for checking sensitivity to
//! them.

use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::fs;

const N_USERS: usize = 480_189;
const N_ITEMS: usize = 17_770;
const DATE_MUL: u32 = 2500;

const TRAIN_DIR: &str = "data/train";
const PROBE_DIR: &str = "data/probe";

// split_2 / split_4
const SEED_2: u64 = 42;
const MOVE_PROB: f64 = 0.9;

// split_3
const DEFAULT_SEED_3: u64 = 42;
const DEFAULT_K_DAYS: usize = 2;
const ADD_FRAC: f64 = 0.25;

// ---- numpy RNG port ----

const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;
const PCG_MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

#[inline]
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

#[inline]
fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    fn new(seed: u64) -> Self {
        let mut entropy: Vec<u32> = Vec::new();
        let mut n = seed;
        while n > 0 {
            entropy.push((n & 0xFFFF_FFFF) as u32);
            n >>= 32;
        }
        if entropy.is_empty() {
            entropy.push(0);
        }
        let mut pool = [0u32; POOL_SIZE];
        let mut hc = INIT_A;
        for i in 0..POOL_SIZE {
            let val = if i < entropy.len() { entropy[i] } else { 0 };
            pool[i] = hashmix(val, &mut hc);
        }
        for s in 0..POOL_SIZE {
            for d in 0..POOL_SIZE {
                if s != d {
                    let h = hashmix(pool[s], &mut hc);
                    pool[d] = mix(pool[d], h);
                }
            }
        }
        for s in POOL_SIZE..entropy.len() {
            for d in 0..POOL_SIZE {
                let h = hashmix(entropy[s], &mut hc);
                pool[d] = mix(pool[d], h);
            }
        }
        SeedSequence { pool }
    }

    fn generate_state_u64(&self, n: usize) -> Vec<u64> {
        let mut s = Vec::with_capacity(2 * n);
        let mut hc = INIT_B;
        for i in 0..(2 * n) {
            let mut d = self.pool[i % POOL_SIZE];
            d ^= hc;
            hc = hc.wrapping_mul(MULT_B);
            d = d.wrapping_mul(hc);
            d ^= d >> XSHIFT;
            s.push(d);
        }
        (0..n)
            .map(|i| ((s[2 * i + 1] as u64) << 32) | (s[2 * i] as u64))
            .collect()
    }
}

struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    fn new(seed: u64) -> Self {
        let s = SeedSequence::new(seed).generate_state_u64(4);
        let initstate = ((s[0] as u128) << 64) | (s[1] as u128);
        let initseq = ((s[2] as u128) << 64) | (s[3] as u128);
        let inc = (initseq << 1) | 1;
        let mut state = 0u128;
        state = state.wrapping_mul(PCG_MULT).wrapping_add(inc);
        state = state.wrapping_add(initstate);
        state = state.wrapping_mul(PCG_MULT).wrapping_add(inc);
        Pcg64 { state, inc, has_uint32: false, uinteger: 0 }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        let val = hi ^ lo;
        let rot = (self.state >> 122) as u32 & 63;
        val.rotate_right(rot)
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let v = self.next_u64();
        self.uinteger = (v >> 32) as u32;
        self.has_uint32 = true;
        v as u32
    }

    fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    fn choice(&mut self, n: u32) -> u32 {
        // numpy short-circuits choice(n=1) without consuming RNG state.
        if n == 1 {
            return 0;
        }
        let mut rnd = self.next_u32() as u64;
        let mut m = rnd * (n as u64);
        let mut leftover = (m & 0xFFFF_FFFF) as u32;
        if leftover < n {
            let threshold = n.wrapping_neg() % n;
            while leftover < threshold {
                rnd = self.next_u32() as u64;
                m = rnd * (n as u64);
                leftover = (m & 0xFFFF_FFFF) as u32;
            }
        }
        (m >> 32) as u32
    }
}

// ---- shared data handling ----

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

/// A rating table: parallel user / item / rating / date columns.
#[derive(Default)]
struct Ratings {
    user: Vec<i32>,
    item: Vec<i16>,
    rate: Vec<i8>,
    date: Vec<i16>,
}

impl Ratings {
    fn load(dir: &str) -> Ratings {
        Ratings {
            user: load_i32(&format!("{}/user_idxs.npy", dir)),
            item: load_i16(&format!("{}/item_idxs.npy", dir)),
            rate: load_i8(&format!("{}/ratings.npy", dir)),
            date: load_i16(&format!("{}/dates.npy", dir)),
        }
    }

    fn len(&self) -> usize { self.user.len() }

    fn push_from(&mut self, src: &Ratings, k: usize) {
        self.user.push(src.user[k]);
        self.item.push(src.item[k]);
        self.rate.push(src.rate[k]);
        self.date.push(src.date[k]);
    }

    /// The rows of `self` whose `keep(k)` holds, in order.
    fn filter(&self, keep: impl Fn(usize) -> bool) -> Ratings {
        let mut out = Ratings::default();
        for k in (0..self.len()).filter(|&k| keep(k)) {
            out.push_from(self, k);
        }
        out
    }

    /// The rows reordered by `key` (unique per row in every use here).
    fn sorted_by<K: Ord>(&self, key: impl Fn(usize) -> K) -> Ratings {
        let mut perm: Vec<u32> = (0..self.len() as u32).collect();
        perm.sort_unstable_by_key(|&k| key(k as usize));
        Ratings {
            user: perm.iter().map(|&k| self.user[k as usize]).collect(),
            item: perm.iter().map(|&k| self.item[k as usize]).collect(),
            rate: perm.iter().map(|&k| self.rate[k as usize]).collect(),
            date: perm.iter().map(|&k| self.date[k as usize]).collect(),
        }
    }

    fn by_user_item(&self) -> Ratings { self.sorted_by(|k| (self.user[k], self.item[k])) }
    fn by_user_date_item(&self) -> Ratings { self.sorted_by(|k| (self.user[k], self.date[k], self.item[k])) }
    fn by_item_user(&self) -> Ratings { self.sorted_by(|k| (self.item[k], self.user[k])) }

    /// Writes the dataset dir: the rating columns, is_test, the counts and
    /// item_years; with `ids` also the index maps (train sets only).
    fn save(self, dir: &str, item_years: &[i32], ids: Option<(&[i32], &[i16])>) {
        let n = self.len();
        let mut user_cnts = vec![0i32; N_USERS];
        let mut item_cnts = vec![0i32; N_ITEMS];
        for k in 0..n {
            user_cnts[self.user[k] as usize] += 1;
            item_cnts[self.item[k] as usize] += 1;
        }
        fs::create_dir_all(dir).unwrap_or_else(|e| panic!("mkdir {}: {}", dir, e));
        macro_rules! save {
            ($name:expr, $vec:expr) => {
                write_npy(format!("{}/{}.npy", dir, $name), &Array1::from($vec)).unwrap();
            };
        }
        save!("user_idxs", self.user);
        save!("item_idxs", self.item);
        save!("ratings", self.rate);
        save!("dates", self.date);
        save!("is_test", vec![0i8; n]);
        save!("user_cnts", user_cnts);
        save!("item_cnts", item_cnts);
        save!("item_years", item_years.to_vec());
        if let Some((uidx_to_uid, iidx_to_iid)) = ids {
            save!("uidx_to_uid", uidx_to_uid.to_vec());
            save!("iidx_to_iid", iidx_to_iid.to_vec());
        }
    }
}

/// Saves a new train / probe pair; item_years and the index maps come from train.
fn save_pair(train: Ratings, probe: Ratings, train_out: &str, probe_out: &str) {
    let item_years = load_i32(&format!("{}/item_years.npy", TRAIN_DIR));
    let uidx_to_uid = load_i32(&format!("{}/uidx_to_uid.npy", TRAIN_DIR));
    let iidx_to_iid = load_i16(&format!("{}/iidx_to_iid.npy", TRAIN_DIR));
    let (n_train, n_probe) = (train.len(), probe.len());
    train.save(train_out, &item_years, Some((&uidx_to_uid, &iidx_to_iid)));
    probe.save(probe_out, &item_years, None);
    println!("Done. {}: {} ratings, {}: {} ratings", train_out, n_train, probe_out, n_probe);
}

fn ud_key(u: i32, d: i16) -> u32 { (u as u32) * DATE_MUL + d as u32 }

/// Sorted, deduplicated (user, date) keys of `r`.
fn ud_keys(r: &Ratings) -> Vec<u32> {
    let mut k: Vec<u32> = (0..r.len()).map(|k| ud_key(r.user[k], r.date[k])).collect();
    k.sort_unstable();
    k.dedup();
    k
}

/// Share of the ratings of `r` with another rating of the same user on the same
/// day among the `keys` (sorted (user, date) keys of the training side).
fn same_day_coverage(r: &Ratings, keys: &[u32]) -> f64 {
    let hit = (0..r.len())
        .filter(|&k| keys.binary_search(&ud_key(r.user[k], r.date[k])).is_ok())
        .count();
    hit as f64 / r.len() as f64
}

// ---- split_2 / split_4 ----

/// The probe ratings that split_2 (and split_4) move into train.
fn split2_moves(probe: &Ratings, train: &Ratings) -> Vec<bool> {
    // RNG sanity check (will panic if the port regresses)
    {
        let mut t = Pcg64::new(42);
        assert_eq!(t.next_u64(), 0xc621fbcd16d92688);
    }
    let n_probe = probe.len();
    let train_keys = ud_keys(train);
    println!("  unique (user, date) pairs in train: {}", train_keys.len());
    let ud_in_train = |u: i32, d: i16| train_keys.binary_search(&ud_key(u, d)).is_ok();

    let mut probe_cnt = vec![0u32; N_USERS];
    for &u in &probe.user {
        probe_cnt[u as usize] += 1;
    }
    let multi: Vec<bool> = probe_cnt.iter().map(|&c| c > 1).collect();
    let n_multi = multi.iter().filter(|&&b| b).count();
    let n_unique = probe_cnt.iter().filter(|&&c| c > 0).count();
    println!("  users with >1 probe rating: {} / {}", n_multi, n_unique);

    let mut users_in_order: Vec<i32> = Vec::new();
    let mut user_indices: Vec<Vec<u32>> = vec![Vec::new(); N_USERS];
    let mut seen = vec![false; N_USERS];
    for i in 0..n_probe {
        let u = probe.user[i] as usize;
        if multi[u] {
            if !seen[u] {
                seen[u] = true;
                users_in_order.push(u as i32);
            }
            user_indices[u].push(i as u32);
        }
    }

    let mut rng = Pcg64::new(SEED_2);
    let mut to_train = vec![false; n_probe];
    let (mut n_moved, mut n_skipped) = (0usize, 0usize);
    for &u in users_in_order.iter() {
        let new_date_indices: Vec<u32> = user_indices[u as usize]
            .iter()
            .copied()
            .filter(|&i| !ud_in_train(u, probe.date[i as usize]))
            .collect();
        if !new_date_indices.is_empty() {
            let pick = rng.choice(new_date_indices.len() as u32) as usize;
            let chosen = new_date_indices[pick] as usize;
            if rng.random() < MOVE_PROB {
                to_train[chosen] = true;
                n_moved += 1;
            }
        } else {
            n_skipped += 1;
        }
    }
    println!(
        "  moved: {} ({:.2}%), skipped (all dates in train): {}",
        n_moved, n_moved as f64 / n_probe as f64 * 100.0, n_skipped
    );
    to_train
}

/// split_2 (`time_order` false: train by (user, item)) or split_4 (true: train
/// by (user, date, item)).
fn split2(n: u32, time_order: bool, root: &str) {
    println!("Phase 1/3: Load probe & train");
    let probe = Ratings::load(PROBE_DIR);
    let train = Ratings::load(TRAIN_DIR);
    println!("  probe: {}, train: {}", probe.len(), train.len());

    println!("Phase 2/3: Pick & move probe ratings");
    let to_train = split2_moves(&probe, &train);

    println!("Phase 3/3: Build & save train_{n} & probe_{n}");
    let mut new_train = train;
    for i in (0..probe.len()).filter(|&i| to_train[i]) {
        new_train.push_from(&probe, i);
    }
    let new_train = if time_order { new_train.by_user_date_item() } else { new_train.by_user_item() };
    let new_probe = probe.filter(|i| !to_train[i]).by_item_user();
    save_pair(new_train, new_probe, &format!("{root}/train_{n}"), &format!("{root}/probe_{n}"));
}

// ---- split_3 ----

fn split3(seed: u64, k_days: usize, root: &str) {
    assert!(k_days > 0, "K must be a positive integer");
    let mut suffix = String::new();
    if seed != DEFAULT_SEED_3 { suffix.push_str(&format!("s{seed}")); }
    if k_days != DEFAULT_K_DAYS { suffix.push_str(&format!("k{k_days}")); }
    let train3_out = format!("{root}/train_3{suffix}");
    let probe3_out = format!("{root}/probe_3{suffix}");
    println!("Seed {seed}, K {k_days} -> {train3_out}, {probe3_out}");

    println!("Phase 1/4: Load probe & train");
    let probe = Ratings::load(PROBE_DIR);
    let train = Ratings::load(TRAIN_DIR);
    let (n_probe, n_train) = (probe.len(), train.len());
    println!("  probe: {}, train: {}", n_probe, n_train);

    println!("Phase 2/4: Order train ratings of probe users, newest day first");
    let mut in_probe = vec![false; N_USERS];
    for &u in &probe.user {
        in_probe[u as usize] = true;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    // (user, newest day first, random within the day): the first rating of
    // each day is the one that stays in train.
    let mut order: Vec<(i32, i16, u32, u32)> = (0..n_train)
        .filter(|&k| in_probe[train.user[k] as usize])
        .map(|k| (train.user[k], -train.date[k], rng.random::<u32>(), k as u32))
        .collect();
    order.sort_unstable();

    println!("Phase 3/4: Draw candidates from each user's last {} train days", k_days);
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

    println!("Phase 4/4: Build & save train_3 & probe_3");
    // train_3 = train \ moved, in train's order; probe_3 = probe ∪ moved
    let train3 = train.filter(|k| !moved[k]);
    let mut probe3 = probe.filter(|_| true);
    for k in (0..n_train).filter(|&k| moved[k]) {
        probe3.push_from(&train, k);
    }
    let probe3 = probe3.by_item_user();

    let train3_keys = ud_keys(&train3);
    println!("  same-day coverage: probe|train {:.4}, probe|train_3 {:.4}, probe_3|train_3 {:.4}",
             same_day_coverage(&probe, &ud_keys(&train)),
             same_day_coverage(&probe, &train3_keys),
             same_day_coverage(&probe3, &train3_keys));
    save_pair(train3, probe3, &train3_out, &probe3_out);
}

fn usage() -> ! {
    eprintln!("usage: split 2|3|4 [SEED K (split 3 only)] [--out DIR]");
    std::process::exit(2);
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut root = "data".to_string();
    if let Some(p) = args.iter().position(|a| a == "--out") {
        if p + 1 >= args.len() { usage(); }
        root = args.remove(p + 1);
        args.remove(p);
    }
    match args.first().map(String::as_str) {
        Some("2") if args.len() == 1 => split2(2, false, &root),
        Some("4") if args.len() == 1 => split2(4, true, &root),
        Some("3") if args.len() <= 3 => {
            let seed = args.get(1)
                .map(|a| a.parse().expect("SEED must be an unsigned integer"))
                .unwrap_or(DEFAULT_SEED_3);
            let k_days = args.get(2)
                .map(|a| a.parse().expect("K must be a positive integer"))
                .unwrap_or(DEFAULT_K_DAYS);
            split3(seed, k_days, &root);
        }
        _ => usage(),
    }
}
