# Notes

Measured dead ends and the reasoning behind them. Everything here was run
against a control column and scored with
`./target/release/ridge -N --lambda 1000 -m <name>`, reading the `all*` row.
The entries are kept out of the source files so the model code stays readable,
but each source file points here.

Read this before starting a new direction. Several of these were rediscovered
and re-measured more than once.

## The rule that decides everything

A column pays only when it is **both** accurate enough to matter and genuinely
new. Neither alone is sufficient, and both failure modes have been measured:

* **Novel but weak.** Bregman co-clustering reaches residual correlation 0.9293
  with the ensemble, squarely in the range that usually pays, at probe RMSE
  0.92511. It is worth zero, because the columns that do pay sit at 0.88 to 0.91.
* **Accurate but redundant.** A small tsvdx on `0.5*dnn-24`, chained with kNN3,
  reaches probe RMSE 0.88240, the most accurate single model in the project. It
  is worth 3e-6 on probe and nothing on quiz.

There is a third trap. Decorrelation produced by a **linear** transformation of
an existing column is worthless, because the blender is linear and simply
rescales it. Training on an over-corrected target (residual weight above 1)
drives the correlation down to 0.8187, the lowest ever measured here, and adds
nothing. The same explains why a Huber loss decorrelates and still does not pay.

## dnn (`src/dnn.rs`)

Removed rather than left as dormant config, each measured against a control
column and worth at most 2e-6:

* an SVD++ implicit y/z branch. The shared embeddings drift coherently and the
  sqrt(|N(u)|) normaliser amplifies it. Centring y/z, mean-normalising the
  profile and shrinking the rate all failed;
* a frozen random sketch of N(u), and plain per-user noise. Both made sibling
  models *more* correlated, not less;
* a same-day implicit term;
* a 5-way softmax head;
* BellKor's frequency-binned item bias;
* up-weighting sparse users;
* a FiLM context gate;
* a linear input-to-output skip;
* seven extra context features;
* extra seeds;
* an item-based transpose;
* residual training on another model's saved train-set predictions. Those are
  in-sample: `tsvdx5-120o` has a train residual of 0.694 against a probe
  residual of 0.873, so there is nothing left to learn.

### Quantile factor drift

This one deserves its own entry, because it **worked** on the axis it was aimed
at. Each user's history was cut into equal-count, day-disjoint quantile slices
and the interaction factor became `p_u + drift[u, slice]`. Standalone probe RMSE
went from 0.90792 to 0.90614. The step form really was new: the tsvdx models
drift their factors only linearly (`src/tx.rs`, `pu + dev * pu2`).

It added nothing to the blend, and one number says why. Its residuals correlate
0.9961 with those of the identical model without slices. Accuracy that leaves
the residuals where they were is invisible to a blend this saturated, and any
chain built on such a base inherits the same redundancy.

The `n_user_slices`, `lr_drift` and `reg_drift` knobs were removed rather than
left dormant at 1. A backup of the working implementation is under
`/tmp/netflix_prize_src_backup_20260814_054748` if it is ever wanted again.

### Two constants that look derivable and are not

`BIAS_BIN_SPAN` (`src/lib.rs`) is `N_DAYS + 1`, that is 2244, while the true span
is 2243: day 0 is 1999-11-11 and day 2242 is 2005-12-31, and every dataset has
2242 as its largest date. The extra one is an inherited off-by-one, not a
statistic. Correcting it moves twelve day boundaries at `n_bins = 30` and would
invalidate every fitted model, so it is kept and documented instead. Note that
`tx::TxModel` bins on a `day_range` it derives from the data, so it does not
carry the +1: the two must stay separate.

`day_scale` (default 1128.0) is an arbitrary normalising scale for the context
vector, not a property of the data. It is half of 2256, which is neither the true
span nor the mean rating date (1790) nor the median (1896). Any nearby value
would serve, since the first layer rescales whatever it is handed.

## Co-clustering (removed, was `src/cocluster.rs`)

Bregman co-clustering: hard user and item partitions with a per-block constant
on top of the shrunk baseline. It was written precisely because the ensemble has
no such function class. Everything else in it is bilinear, neighborhood-based
or an RBM, whereas this one is piecewise constant on a user by item
checkerboard, so it cannot express a smooth interaction at all.

Cheap to run: about 36 seconds an epoch on 99M ratings single-threaded, so a
full 24-epoch fit takes 14 minutes. That is roughly asym speed and an order of
magnitude below the dnn. The cluster count is nearly free, since the assignment
search is O(n_users * K * L) and the three O(nnz) passes dominate.

Measured, all worth **zero**:

| variant | probe RMSE | correlation |
|---|---|---|
| K=24 on `0.5*dnn-24` | 0.92511 | 0.9293 |
| K=128 on `0.5*dnn-24` | 0.93120 | |
| kNN3 chain on the K=24 base | 0.90815 | |
| K=24 on `rtg` | 0.95414 | 0.9084 |
| K=24 on `rtg` with a rank-8 block-weighted factorisation | 0.91943 | 0.9350 |

Two things are worth carrying forward. First, a correlation of 0.9293 sits
squarely in the range that normally pays, so **correlation in that range is
necessary but not sufficient**: at 0.925 the model is simply too weak next to the
0.88 to 0.91 of the columns that do pay. Second, the learning curve reverses
direction depending on the target. On raw ratings it improves monotonically
(0.98474 down to 0.95414) because the blocks carry the whole signal, while on
the dnn residual it degrades (0.92003 up to 0.92511) because 576 constants find
almost nothing left and end up fitting training noise.

The final extension answers a natural follow-up. Instead of a constant per
block, a shared rank-r factorisation modulated per block by a diagonal weight,
`p[u,k] * w[a,b,k] * q[i,k]`, which costs only K*L*r extra parameters. Local
factors *inside* each block were rejected on arithmetic: a user meets every item
cluster, so it would need one vector per cluster, leaving the median user about
four ratings to fit each. The weighted version does learn, and the curve turns
upward after epoch 5 instead of degrading. Accuracy improved from 0.92511 to
0.91943 and the correlation rose from 0.9293 to 0.9350 at the same time. The
factorisation moves the model toward the region where the ensemble is dense,
which is exactly the trade that never pays here.

## Slice specialists (removed, was `src/expert.rs`)

Every predictor in the project fits all training rows with equal weight. This
wrapped `DnnModel` and filtered the training rows to one region, so that the
errors would concentrate somewhere different rather than being uniformly a
little worse. Two slices were run: users with at most 50 ratings (4.2% of rows)
and ratings on or before day 1500 (16.7%).

Standalone probe RMSE 0.95567 and 0.98879, and worth zero under ridge. That much
was expected, since ridge gives every column one global weight and cannot use a
specialist at all.

The real test was **fwls**, whose weights vary with the voting features, and
where the right gates already exist (`vf004_log_user_cnt`, `vf050_log_day`). In
an 18-model mini-blend the probe RMSE was 0.86493 with the expert against
0.86494 without. Nothing there either.

So the idea fails on its own terms, not merely on the measurement instrument. A
specialist has to be found by the blender, and even a blender built to vary its
weights by context did not find one.
