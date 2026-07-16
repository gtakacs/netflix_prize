//! Build an alternative train/probe split: for each user with >1 probe rating,
//! move one randomly chosen probe rating into train (provided the chosen rating's
//! date is not already represented for that user in train, and a 90% probability
//! check passes). The result is written as data/trainx and data/probex.
//!
//! Originally implemented in Python; ported here to keep the data-prep pipeline
//! pure Rust. The randomization must be bit-identical to the Python original,
//! so this file includes a port of `np.random.default_rng(seed)` — both
//! numpy's SeedSequence and the PCG64 BitGenerator, plus the `random()` and
//! `choice()` methods — verified to match numpy output for the same seed.

use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use std::fs;

const N_USERS: usize = 480_189;
const N_ITEMS: usize = 17_770;
const SEED: u64 = 42;
const MOVE_PROB: f64 = 0.9;
const DATE_MUL: u32 = 2500;

const TRAIN_DIR: &str = "data/train";
const PROBE_DIR: &str = "data/probe";
const TRAINX_OUT: &str = "data/trainx";
const PROBEX_OUT: &str = "data/probex";

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

// ---- main split logic ----

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

fn main() {
    // RNG sanity check (will panic if the port regresses)
    {
        let mut t = Pcg64::new(42);
        assert_eq!(t.next_u64(), 0xc621fbcd16d92688);
    }

    println!("Phase 1/6: Load probe & train");
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

    println!("Phase 2/6: Build train (user, date) keys");
    let mut train_keys: Vec<u32> = (0..n_train)
        .map(|k| (tu[k] as u32) * DATE_MUL + (td[k] as u32))
        .collect();
    train_keys.sort_unstable();
    train_keys.dedup();
    println!("  unique (user, date) pairs in train: {}", train_keys.len());

    let ud_in_train = |u: i32, d: i16| -> bool {
        let key = (u as u32) * DATE_MUL + (d as u32);
        train_keys.binary_search(&key).is_ok()
    };

    println!("Phase 3/6: Find multi-rating users & group probe by user");
    let mut probe_cnt = vec![0u32; N_USERS];
    for &u in &pu {
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
        let u = pu[i] as usize;
        if multi[u] {
            if !seen[u] {
                seen[u] = true;
                users_in_order.push(u as i32);
            }
            user_indices[u].push(i as u32);
        }
    }

    println!("Phase 4/6: Pick & move probe ratings");
    let mut rng = Pcg64::new(SEED);
    let mut to_train_mask = vec![false; n_probe];
    let mut n_moved: usize = 0;
    let mut n_skipped: usize = 0;

    for &u in users_in_order.iter() {
        let indices = &user_indices[u as usize];
        let new_date_indices: Vec<u32> = indices
            .iter()
            .copied()
            .filter(|&i| !ud_in_train(u, pd[i as usize]))
            .collect();
        if !new_date_indices.is_empty() {
            let pick = rng.choice(new_date_indices.len() as u32) as usize;
            let chosen = new_date_indices[pick] as usize;
            if rng.random() < MOVE_PROB {
                to_train_mask[chosen] = true;
                n_moved += 1;
            }
        } else {
            n_skipped += 1;
        }
    }
    let pct = n_moved as f64 / n_probe as f64 * 100.0;
    println!(
        "  moved: {} ({:.2}%), skipped (all dates in train): {}",
        n_moved, pct, n_skipped
    );

    println!("Phase 5/6: Build trainx & probex (sorted)");
    // trainx = train ∪ moved probe, sorted by (user, item)
    let n_trainx = n_train + n_moved;
    let mut tx_user: Vec<i32> = Vec::with_capacity(n_trainx);
    let mut tx_item: Vec<i16> = Vec::with_capacity(n_trainx);
    let mut tx_rate: Vec<i8> = Vec::with_capacity(n_trainx);
    let mut tx_date: Vec<i16> = Vec::with_capacity(n_trainx);
    tx_user.extend_from_slice(&tu);
    tx_item.extend_from_slice(&ti);
    tx_rate.extend_from_slice(&tr);
    tx_date.extend_from_slice(&td);
    for i in 0..n_probe {
        if to_train_mask[i] {
            tx_user.push(pu[i]);
            tx_item.push(pi[i]);
            tx_rate.push(pr[i]);
            tx_date.push(pd[i]);
        }
    }
    assert_eq!(tx_user.len(), n_trainx);

    let mut perm: Vec<u32> = (0..n_trainx as u32).collect();
    perm.sort_unstable_by_key(|&k| (tx_user[k as usize], tx_item[k as usize]));
    let trainx_user: Vec<i32> = perm.iter().map(|&k| tx_user[k as usize]).collect();
    let trainx_item: Vec<i16> = perm.iter().map(|&k| tx_item[k as usize]).collect();
    let trainx_rate: Vec<i8> = perm.iter().map(|&k| tx_rate[k as usize]).collect();
    let trainx_date: Vec<i16> = perm.iter().map(|&k| tx_date[k as usize]).collect();
    drop(perm);
    drop(tx_user);
    drop(tx_item);
    drop(tx_rate);
    drop(tx_date);

    // probex = probe \ moved, sorted by (item, user)
    let n_probex = n_probe - n_moved;
    let mut px_user: Vec<i32> = Vec::with_capacity(n_probex);
    let mut px_item: Vec<i16> = Vec::with_capacity(n_probex);
    let mut px_rate: Vec<i8> = Vec::with_capacity(n_probex);
    let mut px_date: Vec<i16> = Vec::with_capacity(n_probex);
    for i in 0..n_probe {
        if !to_train_mask[i] {
            px_user.push(pu[i]);
            px_item.push(pi[i]);
            px_rate.push(pr[i]);
            px_date.push(pd[i]);
        }
    }
    assert_eq!(px_user.len(), n_probex);

    let mut perm: Vec<u32> = (0..n_probex as u32).collect();
    perm.sort_unstable_by_key(|&k| (px_item[k as usize], px_user[k as usize]));
    let probex_user: Vec<i32> = perm.iter().map(|&k| px_user[k as usize]).collect();
    let probex_item: Vec<i16> = perm.iter().map(|&k| px_item[k as usize]).collect();
    let probex_rate: Vec<i8> = perm.iter().map(|&k| px_rate[k as usize]).collect();
    let probex_date: Vec<i16> = perm.iter().map(|&k| px_date[k as usize]).collect();
    drop(perm);
    drop(px_user);
    drop(px_item);
    drop(px_rate);
    drop(px_date);

    // counts
    let mut tx_user_cnts = vec![0i32; N_USERS];
    let mut tx_item_cnts = vec![0i32; N_ITEMS];
    for k in 0..n_trainx {
        tx_user_cnts[trainx_user[k] as usize] += 1;
        tx_item_cnts[trainx_item[k] as usize] += 1;
    }
    let mut px_user_cnts = vec![0i32; N_USERS];
    let mut px_item_cnts = vec![0i32; N_ITEMS];
    for k in 0..n_probex {
        px_user_cnts[probex_user[k] as usize] += 1;
        px_item_cnts[probex_item[k] as usize] += 1;
    }

    // shared arrays from train
    let item_years: Vec<i32> = load_i32(&format!("{}/item_years.npy", TRAIN_DIR));
    let uidx_to_uid: Vec<i32> = load_i32(&format!("{}/uidx_to_uid.npy", TRAIN_DIR));
    let iidx_to_iid: Vec<i16> = load_i16(&format!("{}/iidx_to_iid.npy", TRAIN_DIR));
    let tx_is_test = vec![0i8; n_trainx];
    let px_is_test = vec![0i8; n_probex];

    println!("Phase 6/6: Save");
    fs::create_dir_all(TRAINX_OUT).unwrap();
    fs::create_dir_all(PROBEX_OUT).unwrap();

    macro_rules! save {
        ($dir:expr, $name:expr, $vec:expr) => {
            write_npy(format!("{}/{}.npy", $dir, $name), &Array1::from($vec)).unwrap();
        };
    }

    save!(TRAINX_OUT, "user_idxs", trainx_user);
    save!(TRAINX_OUT, "item_idxs", trainx_item);
    save!(TRAINX_OUT, "ratings", trainx_rate);
    save!(TRAINX_OUT, "dates", trainx_date);
    save!(TRAINX_OUT, "is_test", tx_is_test);
    save!(TRAINX_OUT, "user_cnts", tx_user_cnts);
    save!(TRAINX_OUT, "item_cnts", tx_item_cnts);
    save!(TRAINX_OUT, "item_years", item_years.clone());
    save!(TRAINX_OUT, "uidx_to_uid", uidx_to_uid);
    save!(TRAINX_OUT, "iidx_to_iid", iidx_to_iid);

    save!(PROBEX_OUT, "user_idxs", probex_user);
    save!(PROBEX_OUT, "item_idxs", probex_item);
    save!(PROBEX_OUT, "ratings", probex_rate);
    save!(PROBEX_OUT, "dates", probex_date);
    save!(PROBEX_OUT, "is_test", px_is_test);
    save!(PROBEX_OUT, "user_cnts", px_user_cnts);
    save!(PROBEX_OUT, "item_cnts", px_item_cnts);
    save!(PROBEX_OUT, "item_years", item_years);

    println!(
        "Done. trainx: {} ratings, probex: {} ratings",
        n_trainx, n_probex
    );
}
