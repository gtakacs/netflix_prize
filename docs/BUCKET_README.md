# Netflix Prize prediction store

Per-rating predictions from a Rust reimplementation of a Netflix Prize
collaborative-filtering ensemble. Training the several hundred base predictors
of its three train/probe splits takes weeks of CPU time; this bucket exists so
that anyone can skip straight to the interesting part, which is blending them.

Source code, pipeline manifests and the tooling described below:
<https://github.com/gtakacs/netflix_prize>. This README lives there too, as
`docs/BUCKET_README.md`.

The bucket is public. Reading it needs no Hugging Face account, no token and no
extra CLI.

## What is here

Every file is a float32 `.npy` array of one prediction per rating, in the row
order of the dataset it is named after, plus the training logs and configs.

| Path | Files | Size | Contents |
| --- | ---: | ---: | --- |
| `preds_1/` | 1481 | 10.59 GB | split_1: `probe` and `qual` predictions, with `.out` logs |
| `preds_1/{lat,vf,vf0,vfx}/` | 772 | 6.52 GB | split_1 blend inputs: latent factors and voting (context) features |
| `preds_2/` | 649 | 3.42 GB | split_2: `probe_2` and `qual` predictions, with `.out` logs and `.cfg` configs |
| `preds_2/{lat,vf,vfx}/` | 460 | 3.63 GB | split_2 blend inputs |
| `preds_3/` | 779 | 5.27 GB | split_3: `probe_3` and `qual` predictions, with logs |
| `preds_3/{lat,vf,vfx}/` | 460 | 4.21 GB | split_3 blend inputs |
| total | 4601 | 33.64 GB | |

The store holds exactly what the pipeline manifests reference, plus the logs and
configs beside them. Columns that no current job consumes are not kept here,
even when they exist in someone's working tree.

The three splits are parallel views of the same data. Each holds out a
different probe set from the training ratings, and all of them share the final
`fulltrain` → `qual` step:

- **split_1** (`preds_1`, `pipeline-1.toml`): the original Netflix
  `train` / `probe`.
- **split_2** (`preds_2`, `pipeline-2.toml`): `train_2` / `probe_2`. Here, for most
  users with more than one probe rating, one of them moves back into train.
- **split_3** (`preds_3`, `pipeline-3.toml`): `train_3` / `probe_3`. Here the
  probe grows by a quarter, with train ratings from each user's last days. This
  makes it as hard as qual in same-day evidence.

The `qual` predictions are the Netflix qualifying set, and they are what a
submission would be built from. The headline blend, `cross1`, combines all
three splits on the quiz half of qual (quiz RMSE 0.855674).

Older paths are gone. `preds_old/`, the project's original split, was retired on
2026-11-20, when split_1 replaced it with the same train / probe sets.
`preds_new/` was renamed `preds_2/` the same day, and its `probex` / `trainx`
files became `probe_2` / `train_2`.

## What is not here

- **Predictions on the training sets** (`*.fulltrain.npy`, `*.train.npy`,
  `*.train_2.npy`, `*.train_3.npy`). They are only needed to retrain residual
  models, not to blend.
- **The ratings themselves.** The single-split blends and the probe side of
  your own measurements read `data/<set>/ratings.npy` and `is_test.npy`, which
  come from the original dataset. To get them:
  - in the repo, `run --setup` downloads the public archive from archive.org and
    builds `train` / `probe` / `fulltrain` / `qual` and split_2;
  - `run -p pipeline-3.toml split_3` adds split_3.

  The qualifying-set labels ship with the repo, so the `cross1` check needs
  none of this.

## Getting the files

The repo's `preds` tool is the intended way. It reads `index.toml` from this
bucket, skips what you already have, verifies every download against its md5,
and resumes interrupted transfers.

```bash
git clone https://github.com/gtakacs/netflix_prize
cd netflix_prize
cargo build --release --bin preds

./target/release/preds pull 'preds_2/aex-*'      # 63 MB smoke test first
./target/release/preds pull --ensemble cross1    # 6.6 GB, what the reference check reads
./target/release/preds pull 'preds_*/*.qual.npy' # 22 GB, every qual column
./target/release/preds pull -j 16                # everything, 33 GB
./target/release/preds pull --dry-run            # verify what you have
```

Run it from the repo root: paths are written relative to the working directory.
Expect about an hour for the full set on a 100 Mbit line.

Without the repo, `hf sync` works too, and so does plain HTTP:

```bash
hf sync hf://buckets/gtakacs/netflix_prize/preds_2 ./preds_2
curl -sL -O "https://huggingface.co/buckets/gtakacs/netflix_prize/resolve/index.toml"
curl -sL -o aex-16.probe_2.npy \
  "https://huggingface.co/buckets/gtakacs/netflix_prize/resolve/preds_2/aex-16.probe_2.npy"
```

## What to run next

`ensembles.toml` in the repo records the blends the project reports and the
numbers they produce, and one command checks the download against them:

```bash
cargo build --release --features blas --bin ridge
./target/release/ridge --ensemble              # does cross1 reproduce?
./target/release/ridge --ensemble s2 -m NAME   # what does a column of yours add?
```

The check runs the default blend, `cross1`, in about half a minute, and exits
non-zero if the numbers have drifted. It needs no dataset download: the
qualifying-set labels ship with the repo and are read directly when `data/qual/`
has not been ingested.

The single-split blends `s1`, `s2` and `s3` are fitted on their probe set, so
they need that split's data (see above) and its probe-side predictions. A plain
`preds pull` brings those, and so does `--ensemble s2`. The repo's
`docs/EXPERIMENTS.md` walks through producing a column of your own.

## index.toml

`index.toml` at the bucket root lists every prediction file with its size and
md5:

```toml
[meta]
version = 1
generated = "2026-11-25T18:47:17Z"
bucket = "gtakacs/netflix_prize"
base_url = "https://huggingface.co/buckets/gtakacs/netflix_prize/resolve"

[[file]]
path = "preds_1/fwls15k4-s1.probe.npy"
size = 5633708
md5 = "cc2aff52bb940472284c7d83fbedca5b"
```

It matters because buckets are mutable and unversioned: a retrained model keeps
its file name and its exact byte size, so only the md5 tells two generations
apart. Trust `index.toml` over file sizes and timestamps, and re-read it before
deciding that a local copy is current. The index describes the prediction files
only, not itself or this README.

## Provenance

The predictions are derived from the Netflix Prize dataset, which Netflix
released publicly in 2006 for the competition of the same name and which is
archived at <https://archive.org/details/nf_prize_dataset.tar>. No raw rating
data is redistributed here.
