# Trying a new predictor

The models in the manifests are a stable, cross-checked set, and adding one to
it is deliberately a bit of work: a module in `src/`, a dispatcher in
`src/bin/`, a job in the pipeline TOML, an entry in the models TOML. For an
*experiment* that is far too much ceremony, so there is a lighter path.

A lab experiment is **one file** in `src/bin/`, referenced by no manifest, whose
predictions go to a sandbox directory. Nothing else in the repo knows it exists.
If it does not pay, you delete two paths and no trace is left.

## 0. Get the existing predictions

Most experiments are worth judging by what they add to the ensemble, not by
their own RMSE, and for that you need the columns that are already in it. You
do not have to train them:

```
cargo build --release --bin preds
./target/release/preds pull 'preds_new/*'
```

You also need `data/` (`run -n download`, `run -n ingest`, `run -n newsplit`).

## 1. Copy the template

```
cp src/bin/lab-example.rs src/bin/lab-foo.rs
```

`src/bin/lab-example.rs` is a complete, running experiment: a global mean plus a
user and an item bias, fitted by SGD. It is dull on purpose, so that what you
read is the wiring and not someone else's idea. Replace the `LabConfig` /
`LabModel` pair with yours; the four methods of the `Regressor` trait (`new`,
`n_epochs`, `fit_epoch`, `predict`) are the whole contract.

Cargo picks up `src/bin/*.rs` automatically, so there is nothing to register.

## 2. Smoke run

```
cargo run --release --bin lab-foo -- --smoke
```

`--smoke` keeps 2% of the users (`--sample FRAC` for another share) and the
template drops to a single epoch, so this takes seconds and answers the only
question worth asking first: does it run, terminate, and produce a finite RMSE.

A subsampled run writes under a `-smoke` name and its `.npy` covers only the
sampled rows, so it can neither overwrite nor be mistaken for a real column.

## 3. Real run

```
cargo run --release --bin lab-foo
```

This is the `trainx -> probex` phase only, and it prints the probe RMSE per
epoch. The `fulltrain -> qual` phase costs as much again and only matters once
the column has earned a place, so it is off by default; `--final` turns it on.

Knobs marked with `ev(...)` in the template can be overridden from the
environment, so a retry needs no recompile:

```
EPOCHS=20 LR=0.01 cargo run --release --bin lab-foo
```

Useful flags: `--target "0.5*dnn-24"` trains on what another model left over
(the residual trick most of the project's chains are built on), `-p
pipeline-old.toml` runs against the other split, and a positional argument
overrides the model name.

## 4. Judge it

The standalone RMSE is printed by the run itself. What actually matters is the
marginal value in the blend, and the cheapest early proxy for it is the
correlation of the new column's probe residuals with the ensemble's: what pays
is low correlation *at comparable accuracy*, not accuracy on its own. The best
standalone model in this project adds nothing to the blend for exactly that
reason.

Blending a `preds_lab/` column directly is phase 2 of this work and not
implemented yet. Until then, to measure a column against the ensemble, copy it
next to the others and use the manual-model flag:

```
cp preds_lab/lab-foo.probex.npy preds_new/
./target/release/ridge -N --lambda 1000 -m lab-foo      # read the 'all*' row
```

(This is the one step that touches `preds_new/`. Remove the copy afterwards if
the experiment does not pay; `preds push` would otherwise upload it.)

## 5a. Throw it away

```
rm src/bin/lab-foo.rs preds_lab/lab-foo.*
```

That is all: no manifest entry, no models TOML line, no library module.

## 5b. Or promote it

Once a column is worth keeping:

1. Move the model into `src/<name>.rs` and declare it in `src/lib.rs` if anything
   else will reuse it. A one-off can stay in its binary.
2. Add a job to `pipeline-new.toml`. `jobtype = "anysplit_model"` lets one
   binary serve both splits (it takes its `Split` from `-p`).
3. Run it with `--final` for the qual column, and move the files from
   `preds_lab/` to `preds_new/`.
4. Add the name to the right group in `models-new.toml`.
5. `./target/release/preds push`.

## What to know before writing the model

- **Sort order differs by set.** Training sets (`train`, `trainx`, `fulltrain`)
  are user-sorted, so `calc_user_offsets` gives each user's contiguous block.
  Probe and qual are item-sorted; user offsets are meaningless there, scan
  instead.
- **`MaskedDataset` hides the targets.** The probe view handed to `new` and
  `fit_epoch` deliberately omits `raw_ratings` and `residuals`, so the type
  system prevents training on what you are predicting.
- **You predict `residuals`, not ratings.** Under the default `"rtg"` target
  those are the ratings; under `"w*model + ..."` they are what the weighted
  combination of earlier models left over.
- **Determinism and memory.** Multi-threaded fits over a shared dense matrix are
  not reproducible; `n_threads: 1` is what the stable jobs use when it matters.
  About 15 GB of RAM caps concurrent model runs at four.
- **Build one binary at a time.** `cargo build --release` compiles every file in
  `src/bin/`, so a half-finished experiment would break the whole build. Use
  `cargo build --release --bin lab-foo` (or `cargo run ... --bin lab-foo`) while
  the file is in flux.
- `preds_lab/` is gitignored, and no manifest references it, so `preds push`
  never uploads it and `run -c -f` never prunes it. Reads that miss there fall
  back to the manifest's own preds dir, which is how a lab model trains on a
  base model it never produced.
