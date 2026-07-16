use netflix_prize::Dataset;
use ndarray::Array1;

#[ctor::ctor]
fn test_setup() {
    netflix_prize::suppress_progress();
}

fn build_dataset(ratings: &[(i32, i32, i8, i16)], n_users: usize, n_items: usize, name: &str) -> Dataset {
    let n_ratings = ratings.len();
    let user_idxs  = Array1::from_iter(ratings.iter().map(|&(u, _, _, _)| u));
    let item_idxs  = Array1::from_iter(ratings.iter().map(|&(_, i, _, _)| i));
    let raw_ratings = Array1::from_iter(ratings.iter().map(|&(_, _, r, _)| r));
    let dates      = Array1::from_iter(ratings.iter().map(|&(_, _, _, d)| d));
    let residuals  = raw_ratings.mapv(|r| r as f32);
    let is_test    = Array1::zeros(n_ratings);

    let mut user_cnts = Array1::<i32>::zeros(n_users);
    for &(u, _, _, _) in ratings { user_cnts[u as usize] += 1; }

    let mut item_cnts = Array1::<i32>::zeros(n_items);
    for &(_, i, _, _) in ratings { item_cnts[i as usize] += 1; }

    let item_years = Array1::from_iter((0..n_items as i32).map(|i| 2000 + i));

    Dataset {
        n_ratings, n_users, n_items,
        user_idxs, user_cnts, item_idxs, item_cnts, item_years,
        raw_ratings, residuals, dates, is_test,
        name: name.to_string(),
        transposed: false,
    }
}

/// 20 users, 10 items, 56 ratings; (user, date)-sorted.
/// Each user has 2–4 ratings spread over 1–2 days.
pub fn make_tiny_train() -> Dataset {
    #[rustfmt::skip]
    let data: &[(i32, i32, i8, i16)] = &[
        // (user, item, rtg, date)
        (0,  3, 3, 10), (0,  7, 4, 10), (0,  9, 2, 20),
        (1,  0, 2, 15), (1,  5, 5, 15),
        (2,  1, 4, 12), (2,  8, 3, 12), (2,  4, 5, 22), (2,  6, 2, 22),
        (3,  2, 5, 18), (3,  7, 3, 28),
        (4,  4, 3, 11), (4,  9, 4, 11), (4,  1, 2, 11),
        (5,  0, 5, 14), (5,  3, 2, 14), (5,  6, 4, 24), (5,  8, 3, 24),
        (6,  1, 4, 16), (6,  7, 3, 26),
        (7,  2, 2, 19), (7,  5, 5, 19), (7,  0, 4, 29),
        (8,  6, 3, 13), (8,  9, 4, 13), (8,  3, 3, 13), (8,  2, 5, 13),
        (9,  0, 4,  9), (9,  4, 3,  9),
        (10, 1, 5, 27), (10, 8, 2, 27), (10, 5, 4, 37),
        (11, 3, 3, 11), (11, 6, 4, 21),
        (12, 2, 2, 13), (12, 7, 5, 13), (12, 0, 3, 23), (12, 9, 4, 23),
        (13, 5, 4, 15), (13, 9, 3, 15),
        (14, 0, 3, 17), (14, 4, 2, 17), (14, 8, 5, 27),
        (15, 1, 5, 19), (15, 8, 4, 29),
        (16, 3, 2, 10), (16, 6, 3, 10), (16, 2, 4, 20), (16, 7, 5, 20),
        (17, 2, 4, 12), (17, 7, 5, 12),
        (18, 5, 3, 14), (18, 9, 2, 24), (18, 4, 4, 24),
        (19, 0, 4, 16), (19, 4, 5, 26),
    ];
    build_dataset(data, 20, 10, "tiny_train")
}

/// 20 users, 10 items, 1 rating per user; (user, date)-sorted.
/// Even users: rating on their last tiny_train day.
/// Odd users:  rating 10 days after their last tiny_train day.
/// Each rating is for an item the user has not rated in tiny_train.
pub fn make_tiny_probe() -> Dataset {
    #[rustfmt::skip]
    let data: &[(i32, i32, i8, i16)] = &[
        // (user, item, rtg, date)
        (8,  0, 4, 13),
        (10, 0, 3, 37),
        (11, 0, 4, 31),
        (13, 0, 3, 25),
        (15, 0, 2, 39),
        (16, 0, 3, 20),
        (17, 0, 5, 22),
        (18, 0, 4, 24),
        (7,  1, 2, 39),
        (12, 1, 5, 23),
        (14, 1, 4, 27),
        (19, 1, 3, 36),
        (6,  2, 5, 26),
        (3,  3, 5, 38),
        (1,  4, 4, 25),
        (4,  5, 4, 11),
        (9,  6, 5, 19),
        (5,  7, 3, 34),
        (0,  8, 3, 20),
        (2,  9, 2, 22),
    ];
    build_dataset(data, 20, 10, "tiny_probe")
}
