mod helpers;
use helpers::{make_tiny_train, make_tiny_probe};

use netflix_prize::MaskedDataset;
use netflix_prize::nbstats::{build_user_index, build_bin_nbstats, build_rtg_nbstats};

/// Sum of all entries (used as a compact regression check).
fn sum(mat: &ndarray::Array2<f32>) -> f32 { mat.iter().sum() }

#[test]
fn build_user_index_regression() {
    // tiny_probe is item-sorted (not user-sorted), so build_user_index
    // produces a non-trivial reordering.
    let pr = make_tiny_probe();
    let (index, offsets) = build_user_index(&pr);

    // Shape checks: 20 users → offsets length 21; index length = n_ratings = 20.
    assert_eq!(offsets.len(), 21);
    assert_eq!(index.len(), 20);
    assert_eq!(offsets[20], 20);

    // Verify that indexing yields user-sorted records: for each user u, every
    // index in offsets[u]..offsets[u+1] should map to a row whose user_idx is u.
    for u in 0..20 {
        for k in offsets[u]..offsets[u + 1] {
            let row = index[k];
            assert_eq!(pr.user_idxs[row] as usize, u);
        }
    }
}

#[test]
fn build_bin_nbstats_regression() {
    let tr = make_tiny_train();
    let pr = make_tiny_probe();
    let cases = [
        ("supp",  302.0),
        ("wsupp", 75.999985),
    ];
    let pr_masked = MaskedDataset::from(&pr);
    for (stat, expected) in cases {
        let mat = build_bin_nbstats(&tr, &pr_masked, stat);
        assert_eq!(mat.dim(), (10, 10));
        let s = sum(&mat);
        println!("[bin] stat={} sum={}", stat, s);
        assert_eq!(s, expected, "sum mismatch for stat={}", stat);
    }
}

#[test]
fn build_rtg_nbstats_regression() {
    let tr = make_tiny_train();
    let cases = [
        ("supp",  170.0),
        ("prod",  2146.0),
        ("diff1", 0.0),
        ("diff2", 342.0),
    ];
    for (stat, expected) in cases {
        let mat = build_rtg_nbstats(&tr, stat);
        assert_eq!(mat.dim(), (10, 10));
        let s = sum(&mat);
        println!("[rtg] stat={} sum={}", stat, s);
        assert_eq!(s, expected, "sum mismatch for stat={}", stat);
    }
}
