//! I/O for the cross-fit blending framework.
//!
//! Loads model predictions, features, ratings and quiz masks into in-memory
//! matrices shaped for the `Blender` models (row-major `Array2<f32>`).

use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use ndarray_npy::{read_npy, write_npy};
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};

pub const DATA_DIR: &str = "data";
pub const NOCLIP_OP: char = '>';
pub const CLIP_MIN: f32 = 1.00;
pub const CLIP_MAX: f32 = 4.95;

// ---------------------------------------------------------------------------
// Blend run logging (tee to `{preds_dir}/{name}.out`)
// ---------------------------------------------------------------------------

/// Open `{preds_dir}/{name}.out` as the crate-wide tee log (same mechanism as
/// `fit2_inner`), so subsequent `teeln!` calls write to stdout *and* the file.
/// Overwrites any prior log; pair with [`close_log`].
pub fn open_log(preds_dir: &str, name: &str) {
    std::fs::create_dir_all(preds_dir).ok();
    let path = format!("{preds_dir}/{name}.out");
    *crate::LOG_FILE.lock().unwrap() =
        Some(BufWriter::new(File::create(&path).unwrap_or_else(|e| panic!("create {path}: {e}"))));
}

/// Flush and close the tee log opened by [`open_log`].
pub fn close_log() {
    if let Some(mut f) = crate::LOG_FILE.lock().unwrap().take() {
        let _ = f.flush();
    }
}

/// Tee the exact blend input column order (stdout + `.out`): base predictors
/// first in `models` order, then `voting` features — matching `build_xy`'s
/// column layout. A leading `>` on a model marks a no-clip column.
pub fn log_columns(models: &[String], voting: &[String]) {
    let n = models.len();
    crate::teeln!(
        "Columns:   {} ({} base + {} voting), in build order:",
        n + voting.len(), n, voting.len(),
    );
    for (j, m) in models.iter().enumerate() {
        crate::teeln!("  {:>4}  {}", j, m);
    }
    for (k, f) in voting.iter().enumerate() {
        crate::teeln!("  {:>4}  {}  (voting)", n + k, f);
    }
}

/// Expand a model spec into individual names. Two forms:
///
/// - Braced (clearer in TOML files): `prefix{a,b,c}suffix` or `prefix{1..3}suffix`.
///   `{...}` is a comma list whose items may be integer `..` ranges; the prefix
///   and suffix wrap each result. `{}` is a shell metacharacter, so quote it on
///   the command line — it is meant for files. An empty *first* item is the bare
///   `prefix+suffix` (shell-style `tsvdx4-60{,__knns,__epochs}`); an empty item
///   anywhere else (a trailing/middle stray comma) is a hard error.
/// - Raw (convenient in the shell, no quoting): `prefix:item,item,...`. A single
///   `:` marks the shared prefix; each comma item is appended to it and may be
///   text (`tsvdx4-:60,70b`) or an integer `..` range (`gbr1_s:1..3`). The `:`
///   is required in raw mode — without it (and without `{}`) the spec is a
///   single literal name (so a bare `gbr1_s1..3` is NOT expanded).
///
/// A leading `>` (no-clip) rides along in the prefix. Range numbers are
/// zero-padded bash-style when a bound has a leading zero (`03..05`).
pub fn expand_specs(spec: &str) -> Vec<String> {
    if let (Some(open), Some(close)) = (spec.find('{'), spec.rfind('}')) {
        if open < close {
            return expand_list(&spec[open + 1..close], &spec[..open], &spec[close + 1..]);
        }
    }
    if let Some(pos) = spec.find(':') {
        return expand_list(&spec[pos + 1..], &spec[..pos], "");
    }
    vec![spec.to_string()]
}

/// Split `list` on commas, expand each item's integer `..` range, and wrap each
/// result with `prefix`/`suffix`.
///
/// An empty alternative = the bare `prefix+suffix`, allowed ONLY as the first
/// item (shell-style `a{,b,c}`). A trailing or middle empty item is a malformed
/// spec (usually a stray comma) and panics, to force a fix.
fn expand_list(list: &str, prefix: &str, suffix: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (idx, item) in list.split(',').enumerate() {
        let item = item.trim();
        if item.is_empty() {
            if idx == 0 {
                out.push(format!("{prefix}{suffix}"));
                continue;
            }
            panic!(
                "malformed brace spec '{prefix}{{{list}}}{suffix}': an empty \
                 alternative is allowed only as the first item (leading comma)"
            );
        }
        let mut expanded = Vec::new();
        expand_range_into(item, &mut expanded);
        for e in expanded {
            out.push(format!("{prefix}{e}{suffix}"));
        }
    }
    out
}

fn expand_range_into(tok: &str, out: &mut Vec<String>) {
    if let Some(dpos) = tok.find("..") {
        let before = &tok[..dpos];
        let after = &tok[dpos + 2..];
        let start: String = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        let end: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let (Ok(n), Ok(m)) = (start.parse::<u64>(), end.parse::<u64>()) {
            let prefix = &before[..before.len() - start.len()];
            let suffix = &after[end.len()..];
            let width = if start.starts_with('0') || end.starts_with('0') {
                start.len().max(end.len())
            } else {
                0
            };
            let nums: Vec<u64> = if n <= m { (n..=m).collect() } else { (m..=n).rev().collect() };
            for i in nums {
                out.push(format!("{prefix}{i:0width$}{suffix}"));
            }
            return;
        }
    }
    out.push(tok.to_string());
}

/// Parse `name`, `name[:stop]`, `name[start:]` or `name[start:stop]` (Python-like
/// slice; bounds optional) into (name, start, stop). Shared by the `pairs.toml`
/// loader and by `-g/-G` group selection.
fn parse_slice(spec: &str) -> (String, Option<usize>, Option<usize>) {
    if let Some(open) = spec.rfind('[') {
        if spec.ends_with(']') {
            let name = spec[..open].to_string();
            let inner = &spec[open + 1..spec.len() - 1];
            let (a, b) = inner.split_once(':')
                .unwrap_or_else(|| panic!("bad slice '{spec}' (use NAME[start:stop])"));
            let parse = |s: &str, w: &str| -> Option<usize> {
                let s = s.trim();
                if s.is_empty() { None }
                else { Some(s.parse().unwrap_or_else(|_| panic!("bad slice {w} in '{spec}'"))) }
            };
            return (name, parse(a, "start"), parse(b, "stop"));
        }
    }
    (spec.to_string(), None, None)
}

// ---------------------------------------------------------------------------
// Models TOML: groups + meta-groups
// ---------------------------------------------------------------------------

/// A parsed models TOML. `groups` maps a group name to its model specs (in
/// document order). `meta` maps a meta-group name to a list of group (or other
/// meta) names; meta-groups are shortcuts usable only via `-g/--groups` and are
/// never iterated as real groups.
pub struct ModelGroups {
    pub groups: IndexMap<String, Vec<String>>,
    pub meta: IndexMap<String, Vec<String>>,
}

/// Load a models TOML. Every top-level array key is a model group; an optional
/// `[meta]` table holds meta-groups (unions of group names). Document order is
/// preserved for both.
pub fn load_models_toml(path: &str) -> ModelGroups {
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let table: toml::Table = toml::from_str(&s).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let mut groups: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut meta: IndexMap<String, Vec<String>> = IndexMap::new();
    for (k, v) in table {
        if k == "meta" {
            meta = v.try_into().unwrap_or_else(|e| panic!("parse {path} [meta]: {e}"));
        } else {
            let specs: Vec<String> =
                v.try_into().unwrap_or_else(|e| panic!("parse {path} group '{k}': {e}"));
            groups.insert(k, specs);
        }
    }
    ModelGroups { groups, meta }
}

/// Resolve `-g/--groups` names into a filtered, ordered group map. Empty names
/// (or the single builtin token `all`, unless a meta group shadows it) selects
/// every real group in TOML order. A name matching a `[meta]` entry expands
/// recursively to its constituent groups; otherwise it must be a real group.
/// Unknown names and meta cycles are hard errors. Duplicates are dropped,
/// keeping first-seen order.
///
/// A name may carry a Python-like slice — `integrated[:5]`, `integrated[5:]`,
/// `integrated[3:7]` — which keeps only that window of the group's *expanded*
/// model list (brace expansion applied, so the bounds count models, not TOML
/// lines). On a meta group or `all` the window runs over the concatenation of
/// its groups in order. Handy for cutting a small U down for a test run.
pub fn select_groups(mg: &ModelGroups, names: &[String]) -> IndexMap<String, Vec<String>> {
    if names.is_empty() || (names.len() == 1 && names[0] == "all" && !mg.meta.contains_key("all")) {
        return mg.groups.clone();
    }
    let mut out: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut visiting: HashSet<String> = HashSet::new();
    for n in names {
        let (name, start, stop) = parse_slice(n);
        if start.is_none() && stop.is_none() {
            resolve_group(mg, &name, &mut out, &mut visiting);
            continue;
        }
        let mut sub: IndexMap<String, Vec<String>> = IndexMap::new();
        resolve_group(mg, &name, &mut sub, &mut visiting);
        let sliced = slice_groups(sub, start, stop);
        if sliced.is_empty() {
            eprintln!("warning: '{n}' selected no models");
        }
        for (g, specs) in sliced {
            out.entry(g).or_insert(specs);
        }
    }
    out
}

/// Keep only the `[start, stop)` window of the expanded model list of `groups`,
/// counted across the whole map in order. Groups left empty are dropped; the
/// kept entries are already-expanded literal specs (`>` prefix preserved), which
/// [`expand_specs`] passes through unchanged downstream.
fn slice_groups(
    groups: IndexMap<String, Vec<String>>,
    start: Option<usize>,
    stop: Option<usize>,
) -> IndexMap<String, Vec<String>> {
    let start = start.unwrap_or(0);
    let stop = stop.unwrap_or(usize::MAX);
    let mut seen = 0usize;
    let mut out: IndexMap<String, Vec<String>> = IndexMap::new();
    for (gname, specs) in groups {
        let mut kept: Vec<String> = Vec::new();
        for raw in &specs {
            for spec in expand_specs(raw) {
                if seen >= start && seen < stop {
                    kept.push(spec);
                }
                seen += 1;
            }
        }
        if !kept.is_empty() {
            out.insert(gname, kept);
        }
    }
    out
}

/// Recursively resolve one `-g` name into `out` (meta → groups; cycle-guarded).
fn resolve_group(
    mg: &ModelGroups,
    name: &str,
    out: &mut IndexMap<String, Vec<String>>,
    visiting: &mut HashSet<String>,
) {
    if let Some(members) = mg.meta.get(name) {
        if !visiting.insert(name.to_string()) {
            eprintln!("error: meta group '{name}' is cyclic");
            std::process::exit(2);
        }
        for m in members {
            resolve_group(mg, m, out, visiting);
        }
        visiting.remove(name);
        return;
    }
    if name == "all" {
        for (g, specs) in &mg.groups {
            out.entry(g.clone()).or_insert_with(|| specs.clone());
        }
        return;
    }
    match mg.groups.get(name) {
        Some(specs) => {
            out.entry(name.to_string()).or_insert_with(|| specs.clone());
        }
        None => {
            eprintln!(
                "error: group '{}' not in models TOML (groups: {}; meta: {})",
                name,
                mg.groups.keys().cloned().collect::<Vec<_>>().join(", "),
                mg.meta.keys().cloned().collect::<Vec<_>>().join(", "),
            );
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Model registry: flatten groups + apply exclusions
// ---------------------------------------------------------------------------

/// Flattened, exclusion-filtered model registry produced by [`flatten_groups`].
pub struct FlatModels {
    /// Deduplicated model names (clip prefix stripped), in first-seen order.
    pub names: Vec<String>,
    /// Parallel clip flags (`false` = no-clip, i.e. a `>` appeared for the name).
    pub clip: Vec<bool>,
    /// Group name → indices into `names`. Groups left empty by exclusion are dropped.
    pub group_indices: IndexMap<String, Vec<usize>>,
}

impl FlatModels {
    /// Reconstruct clip-prefixed spec strings (`>name` when no-clip) in `names`
    /// order — the shape consumed by `load_preds`/`build_xy`.
    pub fn specs(&self) -> Vec<String> {
        self.names
            .iter()
            .zip(&self.clip)
            .map(|(n, &c)| if c { n.clone() } else { format!("{NOCLIP_OP}{n}") })
            .collect()
    }
}

/// Flatten the selected `groups` into a deduplicated `(name, clip)` registry with
/// group membership, dropping any model whose name matches an entry in `exclude`.
/// Each `exclude` raw spec is brace-expanded via [`expand_specs`] and `>`-stripped
/// before matching. Deduplication is by clip-stripped name in first-seen order,
/// with the no-clip (`>`) variant winning on conflict. Emits a warning for any
/// `--exclude` that matched no model and for any group left empty by exclusion.
pub fn flatten_groups(groups: &IndexMap<String, Vec<String>>, exclude: &[String]) -> FlatModels {
    let mut excluded: HashSet<String> = HashSet::new();
    for raw in exclude {
        for spec in expand_specs(raw) {
            let name = spec.strip_prefix(NOCLIP_OP).unwrap_or(&spec).to_string();
            excluded.insert(name);
        }
    }
    let mut exclude_hit: HashSet<String> = HashSet::new();

    let mut names: Vec<String> = Vec::new();
    let mut clip: Vec<bool> = Vec::new();
    let mut idx_of: HashMap<String, usize> = HashMap::new();
    let mut group_indices: IndexMap<String, Vec<usize>> = IndexMap::new();

    for (gname, specs) in groups {
        let mut idxs = Vec::new();
        for raw in specs {
            for spec in expand_specs(raw) {
                let (name, c) = match spec.strip_prefix(NOCLIP_OP) {
                    Some(rest) => (rest.to_string(), false),
                    None => (spec.clone(), true),
                };
                if excluded.contains(&name) {
                    exclude_hit.insert(name);
                    continue;
                }
                let i = *idx_of.entry(name.clone()).or_insert_with(|| {
                    names.push(name.clone());
                    clip.push(c);
                    names.len() - 1
                });
                // Any '>' wins: a no-clip appearance disables clipping for the name.
                if !c {
                    clip[i] = false;
                }
                idxs.push(i);
            }
        }
        // Drop groups emptied by exclusion rather than emit a degenerate row.
        if idxs.is_empty() {
            eprintln!("warning: group '{gname}' is empty after exclusion — skipping");
            continue;
        }
        group_indices.insert(gname.clone(), idxs);
    }

    for name in excluded.difference(&exclude_hit) {
        eprintln!("warning: --exclude '{name}' matched no model");
    }

    FlatModels { names, clip, group_indices }
}

/// Resolve the voting-feature specs selected from `voting_toml` by `groups`
/// (empty = all groups), as names relative to the preds dir — e.g.
/// `vf/vf000_constant` for a computed voting feature, or a bare predictor name
/// to use a normal prediction as a context feature. Reuses the model-group
/// machinery: `{a,b}` / `..` expansion, dedup, group order. Voting features
/// carry no clip flag, so only the flattened names are returned.
pub fn resolve_voting(voting_toml: &str, groups: &[String]) -> Vec<String> {
    let mg = load_models_toml(voting_toml);
    let selected = select_groups(&mg, groups);
    flatten_groups(&selected, &[]).names
}

// ---------------------------------------------------------------------------
// Pair (product) features — shared across blenders
// ---------------------------------------------------------------------------

/// Load a `pairs.toml` spec (`pairs = ["A × B", ...]`, the fwls-fs interaction
/// format), optionally sliced (`path[:10]`), into `(left, right)` atom-name
/// tuples. Each entry must contain exactly one `" × "` separator.
pub fn load_pairs(spec: &str) -> Vec<(String, String)> {
    let (path, start, stop) = parse_slice(spec);
    #[derive(serde::Deserialize)]
    struct P { #[serde(default)] pairs: Vec<String> }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let p: P = toml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let mut out: Vec<(String, String)> = Vec::with_capacity(p.pairs.len());
    for s in &p.pairs {
        let parts: Vec<&str> = s.split(" × ").collect();
        if parts.len() != 2 {
            panic!("pairs entry '{s}' is not of the form 'A × B'");
        }
        out.push((parts[0].trim().to_string(), parts[1].trim().to_string()));
    }
    let start = start.unwrap_or(0).min(out.len());
    let stop = stop.unwrap_or(out.len()).min(out.len());
    if start >= stop { return Vec::new(); }
    out[start..stop].to_vec()
}

/// Deduplicate the atom names referenced by `pairs` (first-seen order) and map
/// each pair to its (left, right) atom indices — the shared building block for
/// materializing product columns in any blender.
pub fn pair_atoms(pairs: &[(String, String)]) -> (Vec<String>, Vec<(usize, usize)>) {
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut atoms: Vec<String> = Vec::new();
    let mut pidx: Vec<(usize, usize)> = Vec::with_capacity(pairs.len());
    for (a, b) in pairs {
        let ia = *index.entry(a.clone()).or_insert_with(|| { atoms.push(a.clone()); atoms.len() - 1 });
        let ib = *index.entry(b.clone()).or_insert_with(|| { atoms.push(b.clone()); atoms.len() - 1 });
        pidx.push((ia, ib));
    }
    (atoms, pidx)
}

/// Load one model's predictions for `dataset` from
/// `{preds_dir}/{model}.{dataset}.npy`. A leading `>` in `spec` disables
/// clipping; otherwise values are clamped to [CLIP_MIN, CLIP_MAX]. A missing
/// file yields a zero vector of length `n` (matching the Python `load_preds`).
pub fn load_preds(spec: &str, preds_dir: &str, dataset: &str, n: usize) -> Array1<f32> {
    let clip = !spec.starts_with(NOCLIP_OP);
    let model = spec.trim_start_matches(NOCLIP_OP);
    let path = format!("{preds_dir}/{model}.{dataset}.npy");
    let mut arr: Array1<f32> = match read_npy(&path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("warning: {path}: {e}; using zeros");
            return Array1::zeros(n);
        }
    };
    if clip {
        arr.mapv_inplace(|v| v.clamp(CLIP_MIN, CLIP_MAX));
    }
    arr
}

/// Load one feature column from `{features_dir}/{name}.{dataset}.npy` (f32).
pub fn load_feature(name: &str, features_dir: &str, dataset: &str) -> Array1<f32> {
    let path = format!("{features_dir}/{name}.{dataset}.npy");
    read_npy(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Load ratings for `dataset` as f32 (stored on disk as int8).
pub fn load_ratings(dataset: &str) -> Array1<f32> {
    let path = format!("{DATA_DIR}/{dataset}/ratings.npy");
    let r: Array1<i8> = read_npy(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    r.mapv(|v| v as f32)
}

/// Quiz mask for `dataset`: `true` where the rating belongs to the quiz subset
/// (`is_test == 0`), matching the Python `quiz_mask`.
pub fn load_quiz_mask(dataset: &str) -> Array1<bool> {
    let path = format!("{DATA_DIR}/{dataset}/is_test.npy");
    let t: Array1<i8> = read_npy(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    t.mapv(|v| v == 0)
}

/// Assemble the design matrix for `dataset`: one column per model prediction
/// (from `models`, `>`-prefix to skip clipping) followed by one column per
/// feature (from `features`). The result is row-major, so `x.as_slice()` yields
/// the flat row-major buffer the `Blender` models expect. Columns are loaded in
/// parallel. Also returns the f32 ratings vector for `dataset`.
pub fn build_xy(
    models: &[String],
    features: &[String],
    preds_dir: &str,
    features_dir: &str,
    dataset: &str,
) -> (Array2<f32>, Array1<f32>) {
    let y = load_ratings(dataset);
    let n = y.len();

    let model_cols: Vec<Array1<f32>> = models
        .par_iter()
        .map(|m| load_preds(m, preds_dir, dataset, n))
        .collect();
    let feat_cols: Vec<Array1<f32>> = features
        .par_iter()
        .map(|f| load_feature(f, features_dir, dataset))
        .collect();

    let d = model_cols.len() + feat_cols.len();
    let mut x = Array2::<f32>::zeros((n, d)); // C-order = row-major
    for (j, col) in model_cols.iter().chain(feat_cols.iter()).enumerate() {
        assert_eq!(col.len(), n, "column {j} length {} != {n}", col.len());
        x.column_mut(j).assign(col);
    }
    (x, y)
}

/// Save a prediction vector to `path` as a float32 `.npy`.
pub fn save_preds(path: &str, preds: &Array1<f32>) {
    write_npy(path, preds).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Root mean squared error over all rows.
pub fn rmse(yhat: &[f32], y: &[f32]) -> f64 {
    assert_eq!(yhat.len(), y.len());
    let sse: f64 = yhat
        .iter()
        .zip(y)
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    (sse / yhat.len() as f64).sqrt()
}

/// RMSE over only the rows where `mask` is `true` (e.g. the quiz subset).
pub fn rmse_masked(yhat: &[f32], y: &[f32], mask: &[bool]) -> f64 {
    let mut sse = 0.0f64;
    let mut cnt = 0usize;
    for ((&a, &b), &m) in yhat.iter().zip(y).zip(mask) {
        if m {
            let d = a as f64 - b as f64;
            sse += d * d;
            cnt += 1;
        }
    }
    (sse / cnt as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Cross-fit utilities
// ---------------------------------------------------------------------------

/// Shuffle `0..n` with a seeded RNG and split into `k` index folds, following
/// `numpy.array_split`: the first `n % k` folds get one extra element. Bitwise
/// reproducibility with numpy is not a goal; only Rust-side determinism is.
pub fn permuted_folds(n: usize, k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.shuffle(&mut StdRng::seed_from_u64(seed));
    let base = n / k;
    let rem = n % k;
    let mut folds = Vec::with_capacity(k);
    let mut start = 0;
    for f in 0..k {
        let sz = base + if f < rem { 1 } else { 0 };
        folds.push(idxs[start..start + sz].to_vec());
        start += sz;
    }
    folds
}

/// Gather the rows of `x` listed in `idxs` into a fresh row-major matrix.
pub fn gather_rows(x: &Array2<f32>, idxs: &[usize]) -> Array2<f32> {
    let d = x.ncols();
    let mut out = Array2::<f32>::zeros((idxs.len(), d));
    for (oi, &i) in idxs.iter().enumerate() {
        out.row_mut(oi).assign(&x.row(i));
    }
    out
}

/// Gather the entries of `v` listed in `idxs`.
pub fn gather(v: &[f32], idxs: &[usize]) -> Vec<f32> {
    idxs.iter().map(|&i| v[i]).collect()
}

/// Scatter-add: `dst[idxs[i]] += src[i]`.
pub fn scatter_add(dst: &mut [f32], idxs: &[usize], src: &[f32]) {
    for (&i, &s) in idxs.iter().zip(src) {
        dst[i] += s;
    }
}

// ---------------------------------------------------------------------------
// Blender trait + cross-fit harness
// ---------------------------------------------------------------------------

/// A blending regressor: `fit` builds a fresh model from a row-major feature
/// matrix, `predict` returns one score per row. The features are always passed
/// as a flat row-major `&[f32]` of `n_rows * n_features`. Models that work in a
/// different space internally (e.g. an LGBM multiclass classifier collapsed to
/// the rating expectation Σ p_k·k) still expose a single scalar per row here, so
/// they remain interchangeable under this trait.
pub trait Blender: Sized {
    type Cfg;
    fn fit(x: &[f32], y: &[f32], n_features: usize, cfg: &Self::Cfg) -> Self;
    fn predict(&self, x: &[f32], n_features: usize) -> Vec<f32>;
}

/// K-fold cross-fit blending, mirroring the Python `cvk_blend`: each fold `k`
/// trains on shard `k` (1/K of probe) and predicts the remaining (K-1)/K. Probe
/// out-of-fold predictions are averaged by how many times each row was
/// predicted; qual predictions are averaged across all K folds. Prints per-fold,
/// probe, and quiz RMSE; returns `(probe_oof_preds, qual_avg_preds)`.
pub fn cvk_blend<B: Blender>(
    x_pr: &Array2<f32>,
    y_pr: &[f32],
    x_ql: &Array2<f32>,
    y_ql: &[f32],
    qz: &[bool],
    n_folds: usize,
    seed: u64,
    cfg: &B::Cfg,
) -> (Array1<f32>, Array1<f32>) {
    let n = y_pr.len();
    let d = x_pr.ncols();
    assert_eq!(x_ql.ncols(), d, "probe/qual feature dim mismatch");

    let folds = permuted_folds(n, n_folds, seed);
    let mut yhat_pr = vec![0.0f32; n];
    let mut yhat_ql = vec![0.0f32; y_ql.len()];
    let mut counted = vec![0u32; n];

    for k in 0..n_folds {
        let tr = &folds[k];
        let te: Vec<usize> = (0..n_folds)
            .filter(|&j| j != k)
            .flat_map(|j| folds[j].iter().copied())
            .collect();

        let x_tr = gather_rows(x_pr, tr);
        let y_tr = gather(y_pr, tr);
        let model = B::fit(x_tr.as_slice().unwrap(), &y_tr, d, cfg);

        let x_te = gather_rows(x_pr, &te);
        let p_te = model.predict(x_te.as_slice().unwrap(), d);
        scatter_add(&mut yhat_pr, &te, &p_te);
        for &i in &te {
            counted[i] += 1;
        }

        let y_te = gather(y_pr, &te);
        crate::teeln!(
            "  fold {}/{}: train {} predict {} RMSE {:.5}",
            k + 1, n_folds, tr.len(), te.len(), rmse(&p_te, &y_te),
        );

        let p_ql = model.predict(x_ql.as_slice().unwrap(), d);
        for (acc, &p) in yhat_ql.iter_mut().zip(&p_ql) {
            *acc += p;
        }
    }

    for (v, &c) in yhat_pr.iter_mut().zip(&counted) {
        if c > 0 {
            *v /= c as f32;
        }
    }
    let inv = 1.0 / n_folds as f32;
    for v in yhat_ql.iter_mut() {
        *v *= inv;
    }

    crate::teeln!(" ProbeRMSE: {:.5}", rmse(&yhat_pr, y_pr));
    crate::teeln!("  QuizRMSE: {:.5}", rmse_masked(&yhat_ql, y_ql, qz));

    (Array1::from(yhat_pr), Array1::from(yhat_ql))
}

// ---------------------------------------------------------------------------
// LightGBM blender (feature = "lgbm")
// ---------------------------------------------------------------------------

#[cfg(feature = "lgbm")]
mod lgbm {
    use super::Blender;
    use lightgbm3::{Booster, Dataset};
    use serde_json::json;

    /// LightGBM blending mode.
    #[derive(Clone, Debug)]
    pub enum LgbmMode {
        /// `objective = "regression"`; `predict` returns the raw GBM output.
        Regression,
        /// `objective = "multiclass"`; `predict` collapses the per-class
        /// probabilities into the rating expectation Σ_k p_k · values[k].
        /// Assumes integer ratings 1..=values.len(), mapped to 0-based classes.
        Multiclass { values: Vec<f32> },
    }

    /// Hyperparameters for `LgbmBlender`. `Default` mirrors the Python
    /// `LGBMRegressor(num_leaves=63, n_estimators=200, random_state=42)`.
    #[derive(Clone, Debug)]
    pub struct LgbmCfg {
        pub mode: LgbmMode,
        pub num_iterations: usize,
        pub num_leaves: usize,
        pub learning_rate: f64,
        pub min_data_in_leaf: usize,
        pub feature_fraction: f64,
        pub bagging_fraction: f64,
        pub bagging_freq: usize,
        pub lambda_l1: f64,
        pub lambda_l2: f64,
        pub max_bin: usize,
        pub num_threads: usize,
        pub seed: u64,
        pub verbosity: i32,
    }

    impl Default for LgbmCfg {
        fn default() -> Self {
            Self {
                mode: LgbmMode::Regression,
                num_iterations: 200,
                num_leaves: 63,
                learning_rate: 0.1,
                min_data_in_leaf: 20,
                feature_fraction: 1.0,
                bagging_fraction: 1.0,
                bagging_freq: 0,
                lambda_l1: 0.0,
                lambda_l2: 0.0,
                max_bin: 255,
                num_threads: 0, // 0 = all cores
                seed: 42,
                verbosity: -1, // quiet
            }
        }
    }

    /// A trained LightGBM model behind the `Blender` interface. In multiclass
    /// mode `values` holds the rating values used to collapse the predicted
    /// probabilities into one scalar per row.
    pub struct LgbmBlender {
        booster: Booster,
        values: Option<Vec<f32>>,
    }

    impl Blender for LgbmBlender {
        type Cfg = LgbmCfg;

        fn fit(x: &[f32], y: &[f32], n_features: usize, cfg: &LgbmCfg) -> Self {
            let mut params = json!({
                "num_iterations": cfg.num_iterations,
                "num_leaves": cfg.num_leaves,
                "learning_rate": cfg.learning_rate,
                "min_data_in_leaf": cfg.min_data_in_leaf,
                "feature_fraction": cfg.feature_fraction,
                "bagging_fraction": cfg.bagging_fraction,
                "bagging_freq": cfg.bagging_freq,
                "lambda_l1": cfg.lambda_l1,
                "lambda_l2": cfg.lambda_l2,
                "max_bin": cfg.max_bin,
                "num_threads": cfg.num_threads,
                "seed": cfg.seed,
                "verbosity": cfg.verbosity,
            });

            let (label, values): (Vec<f32>, Option<Vec<f32>>) = match &cfg.mode {
                LgbmMode::Regression => {
                    params["objective"] = json!("regression");
                    params["metric"] = json!("l2");
                    (y.to_vec(), None)
                }
                LgbmMode::Multiclass { values } => {
                    params["objective"] = json!("multiclass");
                    params["num_class"] = json!(values.len());
                    params["metric"] = json!("multi_logloss");
                    let label = y.iter().map(|&r| r - 1.0).collect();
                    (label, Some(values.clone()))
                }
            };

            let dataset = Dataset::from_slice(x, &label, n_features as i32, true)
                .expect("lightgbm Dataset::from_slice");
            let booster = Booster::train(dataset, &params).expect("lightgbm Booster::train");
            Self { booster, values }
        }

        fn predict(&self, x: &[f32], n_features: usize) -> Vec<f32> {
            let raw = self
                .booster
                .predict(x, n_features as i32, true)
                .expect("lightgbm predict");
            match &self.values {
                None => raw.iter().map(|&v| v as f32).collect(),
                Some(values) => {
                    let k = values.len();
                    raw.chunks_exact(k)
                        .map(|p| p.iter().zip(values).map(|(&pi, &v)| pi as f32 * v).sum())
                        .collect()
                }
            }
        }
    }
}

#[cfg(feature = "lgbm")]
pub use lgbm::{LgbmBlender, LgbmCfg, LgbmMode};
