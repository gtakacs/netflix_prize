// Compute item-item co-rating statistics over the dataset.
// Used by k-NN models and as input for similarity matrices.
//
// Two-tiered API:
// - `build_*` returns the in-memory matrix (pure compute, no I/O)
// - `save_*` calls the corresponding `build_*` and writes the result to `sim/`
//
// In-memory callers (tests, chained workflows) use `build_*`; binaries use
// `save_*` for the standard "compute + dump to .npy" CLI flow.

use crate::{Dataset, MaskedDataset, Split, SPLIT_NEW, SPLIT_OLD};
use indicatif::{ProgressIterator, ParallelProgressIterator};
use ndarray::{Array1, Array2, s};
use ndarray_npy::write_npy;
use parking_lot::Mutex;
use rayon::prelude::*;

/// Build a per-user index over a dataset that isn't user-sorted (e.g. probe).
///
/// Returns `(index, offsets)` where for user `u`:
/// - `offsets[u]..offsets[u+1]` is u's slice in `index`
/// - `index[k]` is a rating row index in the original dataset
///
/// Useful when the underlying dataset stores ratings in arbitrary order
/// (probe is item-sorted, not user-sorted).
pub fn build_user_index(ds: &Dataset) -> (Array1<usize>, Array1<usize>) {
    build_user_index_inner(&ds.user_idxs, ds.n_users, ds.n_ratings)
}

/// `MaskedDataset` variant of `build_user_index`. Callable from training code
/// without exposing residuals/raw_ratings.
pub fn build_user_index_masked(ds: &MaskedDataset) -> (Array1<usize>, Array1<usize>) {
    build_user_index_inner(ds.user_idxs, ds.n_users, ds.n_ratings)
}

fn build_user_index_inner(
    user_idxs: &Array1<i32>,
    n_users: usize,
    n_ratings: usize,
) -> (Array1<usize>, Array1<usize>) {
    // offsets
    let mut counts = Array1::<usize>::zeros(n_users);
    for u in user_idxs { counts[*u as usize] += 1; }
    let mut offsets = Array1::<usize>::zeros(n_users + 1);
    for u in 0..n_users {
        offsets[u + 1] = offsets[u] + counts[u];
    }

    // index
    let mut index = Array1::<usize>::zeros(n_ratings);
    let mut cursor = offsets.clone();
    for idx in 0..n_ratings {
        let u = user_idxs[idx] as usize;
        let pos = cursor[u];
        index[pos] = idx;
        cursor[u] += 1;
    }
    (index, offsets)
}

/// Build a binary item-item co-occurrence matrix from union of `tr_set` + `pr_set`.
///
/// Returns a (n_items × n_items) f32 matrix where entry [i,j] counts how many
/// users rated both i and j (across train + probe).
///
/// `stat` selects the per-pair contribution:
/// - `"supp"`  — each user contributes 1.0 to every (i,j) pair
/// - `"wsupp"` — each user contributes 1/n to every pair, where n = total ratings
///               of that user (downweights heavy users)
///
/// # Panics
/// Panics on any other `stat` value.
pub fn build_bin_nbstats(tr: &Dataset, pr: &MaskedDataset, stat: &str) -> Array2<f32> {
    let (pr_index, pr_offsets) = build_user_index_masked(pr);

    let mut bin_sum2 = Array2::<f32>::zeros((tr.n_items, tr.n_items));
    let mut idx = 0;
    for u in progress!(0..tr.n_users) {
        let cnt = tr.user_cnts[u] as usize;
        if cnt == 0 { continue; }
        let start = idx;
        let end = idx + cnt;
        idx = end;

        let tr_len = end - start;
        let pr_start = pr_offsets[u];
        let pr_end = pr_offsets[u + 1];
        let pr_len = pr_end - pr_start;

        let w = match stat {
            "supp" => 1.0,
            "wsupp" => 1.0 / (tr_len + pr_len) as f32,
            _ => panic!("Invalid stat name")
        };

        // Concatenate u's items from train + probe into one flat array,
        // then accumulate the outer product.
        let mut user_items = Array1::<usize>::zeros(tr_len + pr_len);
        // train part
        user_items
            .slice_mut(s![0..tr_len])
            .assign(&tr.item_idxs.slice(s![start..end]).mapv(|x| x as usize));
        // probe part (probe is item-sorted, so we go through pr_index)
        for (k, &pidx) in pr_index.slice(s![pr_start..pr_end]).iter().enumerate() {
            user_items[tr_len + k] = pr.item_idxs[pidx] as usize;
        }

        for idx_i in 0..user_items.len() {
            let i = user_items[idx_i] as usize;
            for idx_j in 0..user_items.len() {
                let j = user_items[idx_j] as usize;
                bin_sum2[[i, j]] += w;
            }
        }
    }
    bin_sum2
}

/// Compute the binary co-occurrence matrix and save to
/// `<split.sim_dir>/bin_<stat>.<tr_set>.npy`.
pub fn save_bin_nbstats(tr_set: &str, pr_set: &str, split: Split, stat: &str) {
    let tr = Dataset::load(tr_set, "rtg", split.preds_dir);
    let pr = Dataset::load(pr_set, "rtg", split.preds_dir);
    let pr_masked = MaskedDataset::from(&pr);
    let mat = build_bin_nbstats(&tr, &pr_masked, stat);
    write_npy(format!("{}/bin_{}.{}.npy", split.sim_dir, stat, tr_set), &mat).unwrap();
}

/// Build a residual-based item-item statistics matrix from `tr_set`.
///
/// Loads `tr_set` with target spec `rtg` (so residuals = rating - rtg-prediction),
/// then for every (item i, item j) pair co-rated by some user, accumulates
/// `op(residual_i, residual_j)`.
///
/// `stat` selects the operator:
/// - `"supp"`  — count of co-ratings (residuals ignored)
/// - `"prod"`  — Σ r_i * r_j  (covariance-like)
/// - `"diff1"` — Σ (r_i - r_j)
/// - `"diff2"` — Σ (r_i - r_j)²  (squared difference)
///
/// # Panics
/// Panics on any other `stat` value.
pub fn build_rtg_nbstats(tr: &Dataset, stat: &str) -> Array2<f32> {
    let residuals = tr.residuals.clone();

    let op: fn(f32, f32) -> f32 = match stat {
        "supp" => |_x, _y| 1.0,
        "prod" => |x, y| x * y,
        "diff1" => |x, y| x - y,
        "diff2" => |x, y| (x - y).powi(2),
        _ => panic!("Invalid stat name")
    };

    // Per-item rows of the output matrix, each behind its own Mutex so users
    // can be processed in parallel without contention on the same row.
    let stat_rows: Vec<Mutex<Vec<f32>>> =
        (0..tr.n_items).map(|_| Mutex::new(vec![0.0; tr.n_items])).collect();

    let mut starts = vec![0; tr.n_users + 1];
    for u in 0..tr.n_users {
        starts[u + 1] = starts[u] + tr.user_cnts[u] as usize;
    }

    progress_count!((0..tr.n_users).into_par_iter(), tr.n_users as u64)
        .for_each(|u| {
        let start = starts[u];
        let end = starts[u + 1];
        if start == end { return; }

        for idx_i in start..end {
            let i = tr.item_idxs[idx_i] as usize;
            let ri = residuals[idx_i] as f32;

            let mut row = stat_rows[i].lock();
            for idx_j in start..end {
                let j = tr.item_idxs[idx_j] as usize;
                let rj = residuals[idx_j] as f32;
                row[j] += op(ri, rj);
            }
        }
    });

    // Convert per-row Vecs into the dense Array2 output.
    let mut stat_mx = Array2::<f32>::zeros((tr.n_items, tr.n_items));
    for i in 0..tr.n_items {
        let row = stat_rows[i].lock();
        stat_mx.row_mut(i).assign(&Array1::from(row.clone()));
    }

    stat_mx
}

/// Compute the residual-based statistics matrix and save to
/// `<split.sim_dir>/<rtg>_<stat>.<tr_set>.npy`.
pub fn save_rtg_nbstats(tr_set: &str, rtg: &str, split: Split, stat: &str) {
    crate::teeln!("{} {} {} {}", tr_set, rtg, split.preds_dir, stat);
    let tr = Dataset::load(tr_set, rtg, split.preds_dir);
    let mat = build_rtg_nbstats(&tr, stat);
    write_npy(format!("{}/{}_{}.{}.npy", split.sim_dir, rtg, stat, tr_set), &mat).unwrap();
}

/// Save nbstats for the standard pipeline (`SPLIT_OLD`): `preds` predictions
/// over `train`+`probe` and over `fulltrain`+`qual`.
pub fn save_nbstats(rtg: &str, stat: &str) {
    save_nbstats_split(rtg, stat, SPLIT_OLD);
}

/// Save nbstats for the alternative pipeline (`SPLIT_NEW`): `preds_new`
/// predictions over `trainx`+`probex` and over `fulltrain`+`qual`.
pub fn save_nbstatsx(rtg: &str, stat: &str) {
    save_nbstats_split(rtg, stat, SPLIT_NEW);
}

/// Save nbstats for a (target spec, stat name, split) triple — covers both
/// `split.tr`+`split.pr` and `split.fulltrain_tr`+`split.fulltrain_pr` in one call.
///
/// Creates `split.sim_dir` if missing. The fulltrain output is skipped if its
/// `.npy` already exists (those files are expensive to compute and never need
/// to be regenerated as long as `data/fulltrain/` is unchanged); the trainx
/// output is always (re)written.
///
/// `rtg == "bin"` triggers the binary co-occurrence path (`save_bin_nbstats`);
/// any other value is treated as a target spec for residuals (`save_rtg_nbstats`).
pub fn save_nbstats_split(rtg: &str, stat: &str, split: Split) {
    std::fs::create_dir_all(split.sim_dir).unwrap();

    if rtg == "bin" {
        save_bin_nbstats(split.tr, split.pr, split, stat);
        let ft = format!("{}/bin_{}.{}.npy", split.sim_dir, stat, split.fulltrain_tr);
        if std::path::Path::new(&ft).exists() {
            crate::teeln!("nbstats: skipping {} (exists)", ft);
        } else {
            save_bin_nbstats(split.fulltrain_tr, split.fulltrain_pr, split, stat);
        }
    } else {
        save_rtg_nbstats(split.tr, rtg, split, stat);
        let ft = format!("{}/{}_{}.{}.npy", split.sim_dir, rtg, stat, split.fulltrain_tr);
        if std::path::Path::new(&ft).exists() {
            crate::teeln!("nbstats: skipping {} (exists)", ft);
        } else {
            save_rtg_nbstats(split.fulltrain_tr, rtg, split, stat);
        }
    }
}
