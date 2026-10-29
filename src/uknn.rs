// User-user kNN as a chain stage on another column's residual (split_3; the
// `uknn[-tag]` stage of `s3 chain`, `<base>__uknn-<tag>`, target `W*base`, which
// reads the knobs named below from the environment).
//
// A user-user similarity over common ratings is out of reach (480k^2 pairs, few
// common items), so users are compared by vectors built from item factors:
//
//   x_u = |N(u)|^-1/2 sum_{j in N(u)} f_j,   then centred over users and scaled to unit length
//
// or, with UK_VEC=res, x_u = |R(u)|^-1/2 sum_{j in R(u)} e_uj f_j (e = the base's training
// residual): users are close when the base misses on the same kind of items the same way.
//
// f_j = rows of an item-factor file `<preds>/<UK_FACT>.<ds>.npy` (e.g. s3-tx50n.ifeat; an
// RBM's W matrix also works), randomly projected to UK_DIM dims if wider. N(u) = u's
// training items (IMPL=1: plus u's prediction-set items). sim(u, v) = x_u . x_v (cosine).
// UK_VEC=coday compares users by co-rating instead: bucket b = (item, day / UK_WIN) holds the
// users that rated the item in that window (training ratings plus prediction-set presence),
// buckets with more than UK_CAP users are skipped, w_b = 1 / ln(1 + n_b), and
// sim(u, v) = sum_{b shared} w_b^2 / (|u| |v|) with |u|^2 = sum_{b of u} w_b^2: users that
// rated the same items on the same days. Neighbours come from the sparse co-occurrence.
// Every user's UK_K most similar users are found once, exactly, by blocked x X^T products.
//
// Prediction for (u, i): over the neighbours v of u (most similar first) that rated i in
// training, the first UK_KP of them,
//
//   w_uv = max(sim, 0)^UK_SCALE / (1 + UK_TAU |day - day_vi|),   r = sum(w e_vi) / sum(w) * n / (n + UK_SHR)
//
// with e_vi = v's training residual of the base and n the number of neighbours used. u itself is never its own neighbour, so a
// training rating never sees its own residual.

use ndarray::{Array2, s};
use ndarray_npy::read_npy;
use crate::{Dataset, MaskedDataset, Regressor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct UknnConfig {
    pub fact: &'static str,
    pub res_vec: bool,
    pub coday: bool,
    pub win: usize,
    pub cap: usize,
    pub dim: usize,
    pub k: usize,
    pub kp: usize,
    pub scaling: f32,
    pub shrinkage: f32,
    pub tau: f32,
    pub implicit: bool,
    pub preds_dir: &'static str,
}

pub struct UknnModel {
    cfg: UknnConfig,
    nbr: Vec<u32>,    // n_users x K, most similar first (u32::MAX pads)
    sim: Vec<f32>,    // matching similarities
    off: Vec<usize>,  // user -> rows of (item, residual, day), sorted by item
    item: Vec<u32>,
    res: Vec<f32>,
    day: Vec<i16>,
}

/// Unit-length, centred user vectors from item factors.
fn user_vectors(tr: &Dataset, pr: &MaskedDataset, cfg: &UknnConfig) -> Array2<f32> {
    let mut f: Array2<f32> = read_npy(format!("{}/{}.{}.npy", cfg.preds_dir, cfg.fact, tr.name)).unwrap();
    assert_eq!(f.nrows(), tr.n_items, "factor rows != items");
    if f.ncols() > cfg.dim {
        // Gaussian random projection to `dim` columns
        let mut rng = StdRng::seed_from_u64(7);
        let nrm = Normal::new(0.0, 1.0 / (cfg.dim as f32).sqrt()).unwrap();
        let p = Array2::from_shape_fn((f.ncols(), cfg.dim), |_| nrm.sample(&mut rng));
        f = f.dot(&p);
    }
    let d = f.ncols();
    // (item, weight): weight 1 (presence) or the base's training residual (UK_VEC=res)
    let mut lists: Vec<Vec<(u32, f32)>> = vec![Vec::new(); tr.n_users];
    for t in 0..tr.n_ratings {
        let wt = if cfg.res_vec { tr.residuals[t] } else { 1.0 };
        lists[tr.user_idxs[t] as usize].push((tr.item_idxs[t] as u32, wt));
    }
    if cfg.implicit && !cfg.res_vec {
        for t in 0..pr.n_ratings { lists[pr.user_idxs[t] as usize].push((pr.item_idxs[t] as u32, 1.0)); }
    }
    let mut x = Array2::<f32>::zeros((tr.n_users, d));
    x.as_slice_mut().unwrap().par_chunks_mut(d).zip(lists.par_iter()).for_each(|(row, l)| {
        for &(j, wt) in l { for (a, b) in row.iter_mut().zip(f.row(j as usize)) { *a += wt * b; } }
        if !l.is_empty() { let c = (l.len() as f32).sqrt(); row.iter_mut().for_each(|a| *a /= c); }
    });
    // centre over the users that have items; a user without any keeps a zero vector
    // (sim 0 to everyone), so it never fills anyone's neighbour list
    let has: Vec<bool> = lists.iter().map(|l| !l.is_empty()).collect();
    let n_has = has.iter().filter(|&&h| h).count().max(1) as f32;
    let mut mean = ndarray::Array1::<f32>::zeros(d);
    for (row, &h) in x.axis_iter(ndarray::Axis(0)).zip(&has) { if h { mean += &row; } }
    mean /= n_has;
    x.as_slice_mut().unwrap().par_chunks_mut(d).zip(has.par_iter()).for_each(|(row, &h)| {
        if !h { return; }
        for (a, m) in row.iter_mut().zip(&mean) { *a -= m; }
        let n = row.iter().map(|a| a * a).sum::<f32>().sqrt();
        if n > 0.0 { row.iter_mut().for_each(|a| *a /= n); }
    });
    x
}

/// Exact top-K neighbours of every user by cosine, in blocks of users.
fn neighbours(x: &Array2<f32>, k: usize) -> (Vec<u32>, Vec<f32>) {
    let n = x.nrows();
    let xt = x.t();
    const B: usize = 256;
    let blocks: Vec<(Vec<u32>, Vec<f32>)> = (0..n.div_ceil(B)).into_par_iter().map(|b| {
        let (a, e) = (b * B, ((b + 1) * B).min(n));
        let sims = x.slice(s![a..e, ..]).dot(&xt); // (e-a) x n
        let mut nb = vec![u32::MAX; (e - a) * k];
        let mut sv = vec![0f32; (e - a) * k];
        let mut cand: Vec<(f32, u32)> = Vec::with_capacity(n);
        for r in 0..e - a {
            let u = a + r;
            cand.clear();
            cand.extend(sims.row(r).iter().enumerate().filter(|&(v, _)| v != u).map(|(v, &s)| (s, v as u32)));
            let take = k.min(cand.len());
            if cand.len() > take { cand.select_nth_unstable_by(take, |p, q| q.0.total_cmp(&p.0)); cand.truncate(take); }
            cand.sort_unstable_by(|p, q| q.0.total_cmp(&p.0));
            for (m, &(s, v)) in cand.iter().enumerate() { nb[r * k + m] = v; sv[r * k + m] = s; }
        }
        (nb, sv)
    }).collect();
    let mut nbr = Vec::with_capacity(n * k);
    let mut sim = Vec::with_capacity(n * k);
    for (a, b) in blocks { nbr.extend(a); sim.extend(b); }
    (nbr, sim)
}

/// Top-K neighbours by same-window co-rating (UK_VEC=coday).
fn coday_neighbours(tr: &Dataset, pr: &MaskedDataset, cfg: &UknnConfig) -> (Vec<u32>, Vec<f32>) {
    let (n, w, k) = (tr.n_users, cfg.win.max(1) as u32, cfg.k);
    let n_win = 2300 / w + 1;
    let key = |i: i32, d: i16| i as u32 * n_win + (d.max(0) as u32) / w;
    let mut pairs: Vec<(u32, u32)> = (0..tr.n_ratings).map(|t| (key(tr.item_idxs[t], tr.dates[t]), tr.user_idxs[t] as u32))
        .chain((0..pr.n_ratings).map(|t| (key(pr.item_idxs[t], pr.dates[t]), pr.user_idxs[t] as u32))).collect();
    pairs.par_sort_unstable();
    pairs.dedup();
    // buckets -> users (CSR), bucket weights
    let mut boff = vec![0usize];
    let mut bkey = Vec::new();
    for (t, &(b, _)) in pairs.iter().enumerate() {
        if t == 0 || b != pairs[t - 1].0 { if t > 0 { boff.push(t); } bkey.push(b); }
    }
    boff.push(pairs.len());
    let nb = bkey.len();
    let bw: Vec<f32> = (0..nb).map(|b| {
        let c = boff[b + 1] - boff[b];
        if c < 2 || c > cfg.cap { 0.0 } else { 1.0 / (1.0 + c as f32).ln() }
    }).collect();
    let busers: Vec<u32> = pairs.iter().map(|x| x.1).collect();
    // user -> buckets
    let mut ub: Vec<Vec<u32>> = vec![Vec::new(); n];
    for b in 0..nb { if bw[b] > 0.0 { for t in boff[b]..boff[b + 1] { ub[busers[t] as usize].push(b as u32); } } }
    drop(pairs);
    let norm: Vec<f32> = ub.par_iter().map(|l| l.iter().map(|&b| bw[b as usize] * bw[b as usize]).sum::<f32>().sqrt()).collect();
    let kept = ub.iter().map(|l| l.len()).sum::<usize>();
    crate::teeln!("uknn coday: {nb} buckets (window {w} d, cap {}), {kept} user-bucket entries kept", cfg.cap);
    let rows: Vec<(Vec<u32>, Vec<f32>)> = (0..n).into_par_iter().map_init(
        || (vec![0f32; n], Vec::<u32>::new()),
        |(acc, touched), u| {
            for &b in &ub[u] {
                let ww = bw[b as usize] * bw[b as usize];
                for &v in &busers[boff[b as usize]..boff[b as usize + 1]] {
                    if v as usize == u { continue; }
                    if acc[v as usize] == 0.0 { touched.push(v); }
                    acc[v as usize] += ww;
                }
            }
            let mut cand: Vec<(f32, u32)> = touched.iter().map(|&v| (acc[v as usize] / (norm[u] * norm[v as usize]), v)).collect();
            for &v in touched.iter() { acc[v as usize] = 0.0; }
            touched.clear();
            let take = k.min(cand.len());
            if cand.len() > take { cand.select_nth_unstable_by(take, |p, q| q.0.total_cmp(&p.0)); cand.truncate(take); }
            cand.sort_unstable_by(|p, q| q.0.total_cmp(&p.0));
            let mut nb = vec![u32::MAX; k];
            let mut sv = vec![0f32; k];
            for (m, &(s, v)) in cand.iter().enumerate() { nb[m] = v; sv[m] = s; }
            (nb, sv)
        }).collect();
    let mut nbr = Vec::with_capacity(n * k);
    let mut sim = Vec::with_capacity(n * k);
    for (a, b) in rows { nbr.extend(a); sim.extend(b); }
    (nbr, sim)
}

impl Regressor for UknnModel {
    type Config = UknnConfig;

    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {
        let t0 = std::time::Instant::now();
        let (nbr, sim) = if cfg.coday {
            coday_neighbours(tr, pr, &cfg)
        } else {
            let x = user_vectors(tr, pr, &cfg);
            crate::teeln!("uknn: user vectors {} x {} ({:.0} s)", x.nrows(), x.ncols(), t0.elapsed().as_secs_f32());
            neighbours(&x, cfg.k)
        };
        let k = cfg.k;
        // over users that have neighbours with sim > 0 (smoke runs leave most users empty)
        let act: Vec<usize> = (0..tr.n_users).filter(|&u| sim[u * k] > 0.0).collect();
        let na = act.len().max(1) as f32;
        let mean_top = act.iter().map(|&u| sim[u * k]).sum::<f32>() / na;
        let mean_last = act.iter().map(|&u| sim[u * k + k - 1]).sum::<f32>() / na;
        crate::teeln!("uknn: top-{k} neighbours ({:.0} s); mean sim of nearest {mean_top:.3}, of K-th {mean_last:.3}",
               t0.elapsed().as_secs_f32());
        // training ratings per user, sorted by item
        let mut rows: Vec<Vec<(u32, f32, i16)>> = vec![Vec::new(); tr.n_users];
        for t in 0..tr.n_ratings {
            rows[tr.user_idxs[t] as usize].push((tr.item_idxs[t] as u32, tr.residuals[t], tr.dates[t]));
        }
        rows.par_iter_mut().for_each(|r| r.sort_unstable_by_key(|x| x.0));
        let mut off = vec![0usize; tr.n_users + 1];
        for u in 0..tr.n_users { off[u + 1] = off[u] + rows[u].len(); }
        let (mut item, mut res, mut day) = (Vec::with_capacity(tr.n_ratings), Vec::with_capacity(tr.n_ratings), Vec::with_capacity(tr.n_ratings));
        for r in rows { for (i, e, d) in r { item.push(i); res.push(e); day.push(d); } }
        Self { cfg, nbr, sim, off, item, res, day }
    }

    fn fit_epoch(&mut self, _tr: &Dataset, _pr: &MaskedDataset, _epoch: usize) {}

    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        let (k, c) = (self.cfg.k, &self.cfg);
        let (mut num, mut den, mut used) = (0f32, 0f32, 0usize);
        for m in 0..k {
            let v = self.nbr[u * k + m];
            if v == u32::MAX || used >= c.kp { break; }
            let s = self.sim[u * k + m];
            if s <= 0.0 { break; }
            let (a, b) = (self.off[v as usize], self.off[v as usize + 1]);
            if let Ok(p) = self.item[a..b].binary_search(&(i as u32)) {
                let t = a + p;
                let w = s.powf(c.scaling) / (1.0 + c.tau * (day - self.day[t] as i32).abs() as f32);
                num += w * self.res[t];
                den += w;
                used += 1;
            }
        }
        if used == 0 { 0.0 } else { num / den * used as f32 / (used as f32 + c.shrinkage) }
    }
}
