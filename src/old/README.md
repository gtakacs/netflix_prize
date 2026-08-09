# `src/old`: frozen legacy model sources

An archive, not part of the build. These are the original sources of base
predictors whose `.npy` files still sit in `preds_old/` and which are still
referenced from `models-old.toml` and `voting-old.toml`. They are kept only so
it is recorded where those predictions came from, and so ideas can be lifted
out of them. Nothing here is maintained, and the directory goes away once no
blend needs its predictors any more.

None of these files compile against the current crate. They `use gravity::…`
(the crate is now `netflix_prize`), and the FWLS generators additionally import
`gravity::fwls_common`, a module that no longer exists.

Only comments and indentation have been touched during archival; no executable
code was changed.

---

## Which file produced which predictor

`preds_old/<name>.cfg` records the config struct name and its field values for
every prediction, so the tables below are read off those files rather than
guessed from predictor names. Every predictor in the families listed here has a
`.cfg`, so these lists are complete.

### Models

| File | What it is | Config struct | Predictors |
| --- | --- | --- | --- |
| `tsvd.rs` | timeSVD++ trained with SGD: `μ + b_u + b_{u,bin} + α_u·dev_u(t) + b_i + b_{i,bin} + (p_u + s_u)·q_i`, where `s_u` is the NSVD1 implicit-feedback term | `TimeSvdConfig` | `tsvd-256`, `tsvd-2048` |
| `tsvdx.rs` | `tsvd.rs` plus a per-day NSVD1 term `s_{u,day}`, built only from the items the user rated on that same day | `TimeSvdConfig` (+ `lr_yd`, `store_ycache_day_tr`) | `tsvdx-20`, `tsvdx-256`, `knn-25w__tsvdx-16` |
| `tsvdx2.rs` | `tsvdx.rs` with the item factor `q_i` replaced by a per-(item, time-bin) factor `q_{i,Bin(t)}` (an `Array3`); no plain `q_i` remains | `TimeSvdConfig` (+ `lr_ifb`) | `tsvdx2-16` |
| `tsvdx3.rs` | `tsvdx.rs` plus a rating-weighted NSVD1 term `z_u = Σ (r_ui − v_shift)·v_scale·z_i / √\|R(u)\|` | `TimeSvdConfig` (+ `lr_z`, `v_shift`, `v_scale`) | `tsvdx3-16` |
| `tsvdxx.rs` | `tsvdx.rs` without materializing the full train `(user, day) → vector` map: day-runs are located by binary search over a compact copy of the train stream and held in a one-slot rolling `Mutex<PredDayCache>` | `TimeSvdConfig` (no `store_ycache_day_tr`) | `tsvdxx-2048` |
| `rbm.rs` | conditional RBM, softmax visible units over the 5 rating categories, per-user SGD with no mini-batching and no momentum, CD-1 switching to CD-3 after `gibbs3_after_epoch` | `RbmConfig` (with `n_levels`, `use_logprobs`) | `rbm-32`, `rbm-64` |
| `rbmb.rs` | conditional RBM, mini-batched over users with momentum and weight decay, a CD-k ramp, a conditional `D` matrix over the rated/unrated vector, per-user visible bias `bu[u,k]` and per-user-day visible bias `but[u,day,k]`, log-prior visible-bias init | `RbmConfig` (with `batch_size`, `momentum`, `lr_bu`, `lr_but`) | `rbmb-256`, `rbmb-256_ep01`, `rbmb-256_ep11`, `rbmb-512x`, `rbmb-2048x`, `rbmb-4096x` |
| `rbmr.rs` | a near-verbatim copy of `rbmb.rs` with Bernoulli hidden units replaced by noisy ReLU (Nair & Hinton 2010): `h_j = max(0, x_j + N(0, √σ(x_j)))`, mean `x·Φ(x/σ) + σ·φ(x/σ)` | same as `rbmb.rs` | `rbmr-512`, `rbmr-2048`, `rbmr-4096` |
| `bkbias.rs` | BellKor baseline predictor, equation (10) of Koren 2009: `b_ui = μ + b_u + α_u·dev_u(t) + b_{u,t} + (b_i + b_{i,Bin(t)})·(c_u + c_{u,t})` | `BkBiasConfig` | `bkbias` |
| `fbias.rs` | bias model with support-bucketed interaction terms: on top of `μ + b_u + b_i` it learns `b_{i,bucket(u)}` and `b_{u,bucket(i)}`, with `bucket(s) = ⌊log2 s⌋` of the rating count | `BiasConfig` | `fbias` |
| `catmf.rs` | 5-class multinomial-softmax MF (not ordinal): each category has its own `(gbias, ubias, ibias, p_u, q_i)` head, the scalar prediction is `E[r] = Σ c·p_c`, cross-entropy with label smoothing `eps` | `ClfMfConfig` | `catmf-16` |

`rbmb.rs` and `rbmr.rs` share the same config struct, so the `.cfg` files cannot
tell them apart; the split above follows the `rbmb-` / `rbmr-` name prefixes.
Apart from the hidden-unit activation the two files are the same ~950 lines of
code, which is the largest duplication in this directory.

`rbm_opt_v1` also carries an `RbmConfig`, but its field set matches neither
`rbm.rs` nor `rbmb.rs`/`rbmr.rs`, so it came from a source not archived here.

### Neighborhood models

| File | What it is | Config struct | Predictors |
| --- | --- | --- | --- |
| `knn.rs` | item-item kNN over precomputed `sim/*.npy` stat matrices. Seven similarity types behind one `SimType` enum: `Support`, `Cosine`, `Mse`, plus four over externally supplied factor vectors. Optional temporal decay `sim /= 1 + τ·\|Δday\|`, similarity threshold, power scaling, and an optional ridge-regularized Cholesky regression over the k neighbors that falls back to a weighted average. | `ItemItemConfig` | 30, listed below |
| `knn3x.rs` | kNN on the residual with the interaction matrices precomputed offline: reads `sim/{target}_prod` and `sim/{baseline}_prod` and mixes them as `x·err + (1−x)·bias`. Adaptive neighborhood size between `k_min` and `k_max`, then either a hand-written row-major Cholesky or an active-set NNLS for non-negative weights. | `Knn3xConfig` | `tsvdx4-180f__knn3x`, `tsvdx4-180f__nlpp__knn3x`, `tsvdx5-200__nlpp__knn3x`, `tsvdx5-400__nlpp__knn3x` |

The 30 `ItemItemConfig` predictors:

```
knn-25w                     bkbias__knn-20f            rbmb-256__knn-25s
als8-8__knn-25c             bkbias__knn-25f            rbmb-256__knn-r
als8-8__knn-25f             catmf-16__knn-25s          rbmb-256_ep11__knn-25s
als8-8__knn-25m             fbias__knn-25f             rbmb-512x__knn-25s
mf-24__knn-25m              ge14__rbmg-64__knn-c       rbmb-512x__knn-r
mf-64u__knn-25f             rbm-32__knn-15r            rbmx-500mf__knn-25f
tsvd-2048__knn-25c          rbm-32__knn-25d            rbmx-700__knn-25f
tsvd-2048__knn-25f          rbm-32__knn-25f            tsvdxx-2048__knn-25f
tsvd-2048__knn-50m          rbm-64__knn-c              tsvdxx-2048__knn-25s
tsvd-2048__knn-c            tsvdx4-160_ep19__knn-25s   tsvdx4-200f__knn-25f
```

Note that `__knn3` (without the `x`) is a different model: those predictors
carry a `Knn3Config` and come from `src/knn3.rs`, not from `knn3x.rs`.

### FWLS / voting feature generators

Standalone binaries, not `Regressor`s. They compute per-rating meta-features
that the FWLS and GBM blenders consume as voting features.

| File | Features | Writes |
| --- | --- | --- |
| `fwls_featuresA.rs` | 10: counts, log-counts, rating stddevs, user tenure, and one log-count product | `features/fwls_A{1..10}.{set}.npy` |
| `fwls_featuresB.rs` | 2: Bayesian-shrunk user and movie means | `features/fwls_B{1,2}.{set}.npy` |
| `fwls_featuresC.rs` | 5: rater count, movie rating-date stddev, user date-bias stddev, single-day share, avg movie popularity | `features/fwls_C{1..5}.{set}.npy` |
| `fwls_featuresD.rs` | 3: item-item similarity aggregates (sum, top-20% concentration, max) | `features/fwls_D{1..3}.{set}.npy` |
| `fwls_featuresE.rs` | 5: SVD factor norms, ordinal-SVD stddev, user overlap, same-day correlation | `features/fwls_E{1..5}.{set}.npy` |
| `fwls_featuresF.rs` | 20: "user gave rating *r* on this / on another day", as binary indicators and as log(1+count) | `features/fwls_F{1..20}.{set}.npy` |
| `fwls_features_claude105.rs` | 105: user and movie temporal windows, kNN context, rating-distribution shape, user×movie interactions, 5 ordinal-SVD class probabilities | `features/claude105_features.{set}.npy`, or one file per column with `--separate` |

The six phase files contribute 45 features in total, and `preds_old/fwls/` holds
90 `.npy` files, i.e. 45 features × {probe, qual}. Likewise `preds_old/claude105/`
holds 210 files for the 105 features.

**The output paths differ from what the blenders read today.** These binaries
write into `features/` with the generator name in the filename; the blenders
load voting features as `{preds_dir}/{name}.{dataset}.npy`, where `name` is the
spec from `voting-old.toml` (`fwls/A1`, `claude105/035`). The shipped files were
renamed into `preds_old/fwls/` and `preds_old/claude105/` accordingly; that
rename is not scripted anywhere in the repo.

In `preds_old/claude105/` the filename is the zero-padded feature index. This
matches the `F_*` index constants at the top of the generator: for example
`claude105/035` is `F_KNN_SAME_DAY = 35`, and `claude105/050` is
`F_USER_RATING_ENTROPY = 50`.

---

## How these were run

There was no manifest. Each file was its own `cargo` binary, and the
hyperparameters and output model name were edited by hand in `main()` before
each run; earlier configs were often left behind as commented-out blocks (see
the `let base = …` stack in `knn.rs::main`, or the alternative `reg_w` values in
`rbm.rs`).

The `main()` shipped in each file is therefore a snapshot of one run, not
necessarily one that appears in the tables above. All five `tsvd*` files end
with `model_name = "tmp"`; `rbm.rs` ends with `"rbm-120"`, which is not among
its predictors; `rbmr.rs` ends with a low-level `fit` call naming datasets that
no longer exist, so as written it produces no qual prediction at all.
