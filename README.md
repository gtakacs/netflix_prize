
The goal of this project is to revisit the Netflix Prize problem, solve it with modern tools, and surpass the previous best result achieved by the Grand Prize winners back in 2009.

## Demo video

[![Watch the video](assets/video-preview.jpg)](https://drive.google.com/file/d/1v7Nvz7EGfXDIBADgBS-sqfcoZ2ksCmEn/view?usp=sharing)

## Quickstart

### 1. Install Rust

Install the toolchain via [rustup](https://rustup.rs/):

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installation, restart the shell (or `source "$HOME/.cargo/env"`) so that
`cargo` and `rustc` are on the `PATH`. The crate uses Rust edition 2024, so a
reasonably recent stable toolchain is required.

On Linux the BLAS backend is OpenBLAS — install the dev package, e.g.
`sudo apt install libopenblas-dev` on Debian/Ubuntu or
`sudo pacman -S openblas` on Arch/Manjaro. On macOS the Accelerate framework
is used and no extra setup is needed.

### 2. Build with cargo

All model and pipeline binaries live in `src/bin/`. Always build with
`--release` — debug builds are typically 50–100× slower and unusable for
training. The first build downloads dependencies and takes a few minutes;
subsequent builds are incremental.

```
cargo build --release                 # build the whole workspace
cargo build --release --bin run       # build a single binary
cargo build --release --bin tsvdx4-new
```

The resulting executables land in `target/release/`.

### 3. Run computations

Computations are orchestrated by the `run` binary, which reads a pipeline
manifest and runs the requested job. From a fresh clone, one command prepares
all the data (see [Data](#data)): it builds the three data binaries, fetches
the archive, parses it into the `.npy` arrays the rest of the pipeline
consumes, and derives the second split. Steps whose outputs already exist are
skipped, so it is safe to re-run:

```
./target/release/run --setup         # data/raw/ -> data/{train,probe,fulltrain,qual}/
                                     #           -> data/{trainx,probex}/  (~3.3 GB)
./target/release/run -n              # list all jobs and their status
./target/release/run -n tsvdx4-64    # train a single model
```

The three steps are ordinary jobs and can also be run one at a time:
`download` fetches the archive into `data/raw/`, `ingest` parses it, and
`newsplit` (in `pipeline-new.toml` only) derives `trainx`/`probex` from
`train`/`probe`.

See [Pipeline](#pipeline) below for the available flags, the two manifests,
and how job dependencies are resolved.

## Predictions without training

Training the ~300 base predictors takes weeks of CPU time. They are published as
a public store, so the interesting part, blending, can be tried straight away:

```
cargo build --release --bin preds
./target/release/preds pull 'preds_*/*.qual.npy'   # 6.2 GB, the qual columns only
./target/release/preds pull                        # everything, 8.9 GB
```

The qual columns alone are enough for the reference check below; the full pull
adds the probe-set predictions, which is what measuring a new column needs.

### Check that it reproduces

`ensembles.toml` records the blends this project reports and the numbers they
produce. One command runs a stored blend and compares it with its record:

```
cargo build --release --features blas --bin ridge
./target/release/ridge --ensemble
```

```
row                   models       quiz       test   expected       delta
old/integrated            48   0.865765   0.866673   0.865765    +2.71e-7  OK
...
ensemble                 302   0.856716   0.857684   0.856716    -6.74e-8  OK

Reference check: PASS (13 rows within 1e-5)
```

It takes about half a minute, exits non-zero if the numbers have drifted, and
needs no dataset download: the qualifying-set labels ship with the repo
(`data/qual_ratings/qual_ratings.csv.gz`), and the blend reads them directly
when `data/qual/` has not been ingested. `--ensemble new` runs the single-split
blend instead, which is fitted on the probe set and reports a probe number too.

### Measure a column of your own

The same command, with the column added:

```
./target/release/ridge --ensemble new -m preds_lab/lab-foo
```

It fits the reference and the reference-plus-your-column from one shared Gram
matrix, so the measurement costs about as much as the check alone, and the
check still runs: a gain is never read off a baseline that has silently moved.
The output gives the delta and the residual correlation with the ensemble,
alongside what deltas in this project are normally worth.

## Data

You do not have to download anything by hand — the `download` and `ingest` jobs
handle it.

The original Netflix Prize dataset is publicly available (no Kaggle account
required) at the Internet Archive:
https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz
(md5 `a8f23d2d76461211c6b4c0ca6df2547d`). The `download` job fetches it to
`data/raw/nf_prize_dataset.tar.gz` over HTTPS (with retry and
resume-on-interruption; skipped if already present) and verifies that md5. The
`ingest` job then reads that archive directly:

```
data/raw/
  nf_prize_dataset.tar.gz   # downloaded
  README                    # extracted by ingest
  movie_titles.csv          # extracted by ingest
  probe.txt                 # extracted by ingest
  qualifying.txt            # extracted by ingest
```

`ingest` extracts the small, human-readable members (`README`,
`movie_titles.csv` — renamed from `.txt` —, `probe.txt`, `qualifying.txt`) to
`data/raw/`, then parses them from there. The bulky training ratings are *not*
materialised: they ship as one `<id>:`-headed file per movie inside an inner
`training_set.tar`, and `ingest` streams those blocks straight out of the tar
(each block is self-describing, so order doesn't matter). If you already have
the archive, drop it into `data/raw/` and `download` becomes a no-op.

The qualifying-set ratings and the Quiz/Test split (in neither the Internet
Archive nor the Kaggle release) are bundled with the repo as
`data/qual_ratings/qual_ratings.csv.gz` — one row per qualifying entry in
`qualifying.txt` parse order, with columns `rating` and `is_test`
(0 = Quiz, 1 = Test). The `ingest` job reads this file directly.

## Pipeline

The pipeline is described declaratively in a TOML manifest. Each job has a
`jobtype` (`model`, `paropt`, `legacy_eblend`, ...), explicit `inputs` and
`outputs`, and a build/run command. The repo holds two manifests, one per
train/probe split:

- `pipeline-old.toml` — `data/{train,probe,fulltrain,qual}` →
  `preds/`, `features/`
- `pipeline-new.toml` — `data/{trainx,probex,fulltrain,qual}` →
  `preds_new/`, `features_new/`. Each job `<name>` has a dedicated binary
  at `src/bin/<name>-new.rs` with hardcoded config.

The orchestrator `src/bin/run.rs` reads a manifest, computes each job's
status (`READY` / `BLOCKED` / `DONE`) from file existence, and runs the
requested job:

```
./target/release/run -n              # list jobs (pipeline-new.toml)
./target/release/run -n JOB          # run JOB
./target/release/run -n -f JOB       # force re-run
./target/release/run --setup         # download + ingest + newsplit in one go
./target/release/run -n -c -f        # delete preds/features files no
                                     # active job references
```

`-p FILE` selects a manifest explicitly; `-n` is a shorthand for
`-p pipeline-new.toml` (default is `pipeline-old.toml`).

## Trying your own predictor

Adding a model to a manifest takes a library module, a dispatcher, a job and a
models-TOML entry. An *experiment* needs none of that: copy
`src/bin/lab-example.rs` to `src/bin/lab-<yours>.rs`, replace the model, and
run it. Predictions go to the gitignored `preds_lab/`, no manifest refers to
the file, and dropping the experiment is one `rm`.

```
cp src/bin/lab-example.rs src/bin/lab-foo.rs
cargo run --release --bin lab-foo -- --smoke   # seconds: does it run?
cargo run --release --bin lab-foo              # full trainx -> probex
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the whole loop, including how
to judge a new column and what to do when it pays.
