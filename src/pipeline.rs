//! Pipeline manifest parsing and job resolution.
//!
//! Reads a manifest like `pipeline-old.toml` and expands each job's
//! `inputs`/`outputs`/`cmd` templates into concrete paths. Shared by the `run`
//! runner and the `preds` remote-store tool.

use crate::blend::{expand_specs, load_models_toml, resolve_voting, select_groups};
use indexmap::IndexMap;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Deserialize, Debug, Default)]
pub struct Pipeline {
    #[serde(default)]
    pub split: HashMap<String, String>,
    #[serde(default)]
    pub defaults: HashMap<String, JobConfig>,
    #[serde(default)]
    pub jobs: IndexMap<String, JobConfig>,
}

#[derive(Deserialize, Debug, Default, Clone)]
pub struct JobConfig {
    pub jobtype: Option<String>,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub inputs_from: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub cmd: String,
    pub runner: Option<String>,
    pub config: Option<String>,
    pub target: Option<String>,
    pub keep_epochs: Option<Vec<u32>>,
    #[serde(default)]
    pub extras: Vec<String>,
    pub n_epochs: Option<u32>,
    #[serde(default)]
    pub non_epoch: Vec<String>,
    /// List of feature names that expand `{feature}` placeholders in outputs.
    /// Each output template containing `{feature}` is replicated once per
    /// feature name (with `{feature}` replaced); other templates are kept as-is.
    #[serde(default)]
    pub features: Vec<String>,
    /// Optional override for the `{binary}` substitution variable. When unset,
    /// `{binary}` resolves to the job name. Used to share a dispatcher
    /// binary across multiple jobs (the job name is passed to the binary
    /// as a CLI arg by the default `cmd`).
    pub binary: Option<String>,
    /// Path to a models TOML (groups of base-predictor names) for blend jobs.
    /// When set, the runner expands it into per-model prediction inputs (so the
    /// job gates on those predictions existing) and exposes the path to the
    /// `cmd` as `{models}`.
    pub models: Option<String>,
    /// Fold seeds for a blend job. When non-empty, each output template's
    /// `{name}.` is expanded to `{name}-s<seed>.` (one prediction pair per seed)
    /// and the comma-joined list is exposed to the `cmd` as `{seeds}`.
    #[serde(default)]
    pub seeds: Vec<u64>,
    /// Model-group names a blend job consumes from its `models` TOML. When
    /// non-empty the runner gates only on those groups' predictions and exposes
    /// the comma-joined list to the `cmd` as `{groups}`; empty resolves to the
    /// TOML's `all` group (or every group when it defines none).
    #[serde(default)]
    pub groups: Vec<String>,
    /// Individual base predictors for a blend job, named directly instead of (or
    /// alongside) `groups`. Rendered into the `cmd` as `-m '<name>'` flags, and
    /// gated on like any group member. With `groups` empty these are the whole
    /// model set, so the job never pulls in the models TOML's groups.
    #[serde(default)]
    pub models_manual: Vec<String>,
    /// Base-predictor names to drop from a blend job's model set. Each entry is
    /// brace-expanded (like a model spec) and `>`-stripped before matching. The
    /// runner both skips them when gating on `models` predictions and renders
    /// them to the `cmd` as `{exclude}` = `-x '<name>' ...` (empty when unset).
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Path to a voting-feature groups TOML. For `vfeat` jobs the runner resolves
    /// the selected `voting` groups into the job's `{feature}` outputs (so the
    /// feature list lives once, in the TOML); for blend jobs it gates on those
    /// feature files and exposes the path to the `cmd` as `{voting_models}`.
    pub voting_models: Option<String>,
    /// Voting-feature group names selected from `voting_models`; must be given
    /// explicitly — an empty list is a hard error for any job that sets
    /// `voting_models`. Every name resolves through the voting TOML (there is no
    /// builtin `all`), so a TOML that wants an everything-group declares it.
    /// Exposed to the `cmd` as `{voting}` (comma-joined).
    #[serde(default)]
    pub voting: Vec<String>,
    /// Extra CLI arguments appended verbatim to the job's `cmd` via `{extra}`
    /// (empty string when unset, like `{exclude}`). For one-off blend settings a
    /// jobtype template cannot express — an ad-hoc `--lambda`, an extra `-m`.
    /// Unlike `groups`/`voting` these are opaque to the runner, so anything that
    /// should gate the job must still come from the models/voting TOMLs.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug)]
pub struct ResolvedJob {
    pub inputs: Vec<String>,
    pub inputs_from: Vec<String>,
    pub outputs: Vec<String>,
    pub cmd: String,
}

#[derive(Debug)]
pub enum Status {
    Done,
    Runnable,
    Blocked(Vec<String>),
}

impl Pipeline {
    /// Read and parse a manifest. The error string is ready to print as-is.
    pub fn load(path: &str) -> Result<Pipeline, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {}", path, e))?;
        toml::from_str(&content).map_err(|e| format!("failed to parse {}: {}", path, e))
    }
}

fn merge_with_defaults(job: &JobConfig, defaults: &HashMap<String, JobConfig>) -> JobConfig {
    let mut merged = job.clone();
    if let Some(jobtype) = &job.jobtype {
        // Look up exact jobtype, falling back to "model" defaults for jobtypes
        // that share the model template (binary at src/bin/{name}-{split.name}.rs,
        // standard outputs).
        let d = defaults.get(jobtype).or_else(|| {
            match jobtype.as_str() {
                "epoch_blend" | "legacy_eblend" | "legacy_model" => defaults.get("model"),
                _ => None,
            }
        });
        if let Some(d) = d {
            if merged.inputs.is_empty() { merged.inputs = d.inputs.clone(); }
            if merged.inputs_from.is_empty() { merged.inputs_from = d.inputs_from.clone(); }
            if merged.outputs.is_empty() { merged.outputs = d.outputs.clone(); }
            if merged.cmd.is_empty() { merged.cmd = d.cmd.clone(); }
            if merged.models.is_none() { merged.models = d.models.clone(); }
            if merged.groups.is_empty() { merged.groups = d.groups.clone(); }
            if merged.voting_models.is_none() { merged.voting_models = d.voting_models.clone(); }
            if merged.voting.is_empty() { merged.voting = d.voting.clone(); }
        }
    }
    merged
}

/// For `jobtype = "epoch_blend"` jobs: auto-add per-epoch input paths and
/// pull in the base + non_epoch chain jobs via inputs_from. Job name
/// must end with `__epochs`; `n_epochs` is required.
fn expand_eblend_inputs(job_name: &str, job: &mut JobConfig) {
    let jobtype = job.jobtype.as_deref();
    if jobtype != Some("epoch_blend") && jobtype != Some("legacy_eblend") { return; }
    let base = match job_name.strip_suffix("__epochs") {
        Some(b) if !b.is_empty() => b.to_string(),
        _ => panic!("{} job '{}' name must end with '__epochs'", jobtype.unwrap(), job_name),
    };
    let n_epochs = job.n_epochs.unwrap_or_else(|| {
        panic!("{} job '{}' requires 'n_epochs' field", jobtype.unwrap(), job_name)
    });
    if !job.inputs_from.contains(&base) {
        job.inputs_from.push(base.clone());
    }
    for ne in job.non_epoch.clone() {
        if !job.inputs_from.contains(&ne) {
            job.inputs_from.push(ne);
        }
    }
    for ep in 1..=n_epochs {
        for ds_var in &["{pr}", "{fulltrain_pr}"] {
            let inp = format!("{{preds}}/{}_ep{:02}.{}.npy", base, ep, ds_var);
            if !job.inputs.contains(&inp) {
                job.inputs.push(inp);
            }
        }
    }
}

/// Expand `extras = ["@train_preds", "@ifeat", ...]` into auto-inputs.
/// `@train_preds` adds `<base>.{tr}.npy` and `<base>.{fulltrain_tr}.npy`;
/// `@ifeat` adds `<base>.ifeat.{tr}.npy` and `<base>.ifeat.{fulltrain_tr}.npy`.
/// Base is the job name's prefix before `__`. When `extras` is non-empty,
/// `inputs_from = [base]` is also auto-added (if not already present).
fn expand_extras(job_name: &str, job: &mut JobConfig) {
    if job.extras.is_empty() { return; }
    // Split on the LAST `__` so deep chains like `mf-61__asym-16__knn3` resolve
    // base = `mf-61__asym-16` (not `mf-61`).
    let base = match job_name.rsplit_once("__") {
        Some((b, _)) if !b.is_empty() => b.to_string(),
        _ => panic!("'extras' on job '{}' but no '__'-base found", job_name),
    };
    if !job.inputs_from.contains(&base) {
        job.inputs_from.push(base.clone());
    }
    for ex in job.extras.clone() {
        let auto: Vec<String> = match ex.as_str() {
            "@train_preds" => vec![
                format!("{{preds}}/{}.{{tr}}.npy", base),
                format!("{{preds}}/{}.{{fulltrain_tr}}.npy", base),
            ],
            "@ifeat" => vec![
                format!("{{preds}}/{}.ifeat.{{tr}}.npy", base),
                format!("{{preds}}/{}.ifeat.{{fulltrain_tr}}.npy", base),
            ],
            other => panic!(
                "unknown extras keyword '{}' on job '{}' (expected one of: @train_preds, @ifeat)",
                other, job_name,
            ),
        };
        for inp in auto {
            if !job.inputs.contains(&inp) { job.inputs.push(inp); }
        }
    }
}

/// For jobs with a `models` TOML: resolve the selected groups plus any
/// `models_manual` names, and add each base predictor's `{pr}` and
/// `{fulltrain_pr}` prediction files as inputs so the runner gates on them. The
/// `>` no-clip prefix is stripped. The paths are kept as templates (`{preds}`,
/// `{pr}`, `{fulltrain_pr}` substituted later).
///
/// `groups` left empty means "every group in the TOML" as before — except when
/// `models_manual` is given, which then *is* the whole model set. The resolved
/// group names are written back into `job.groups` so the rendered `--groups`
/// lists them explicitly (there is no builtin `all` to fall back on).
fn expand_blend_models(job: &mut JobConfig) {
    let Some(path) = job.models.clone() else { return; };
    let manual_only = job.groups.is_empty() && !job.models_manual.is_empty();
    let groups = if manual_only {
        IndexMap::new()
    } else {
        let mg = load_models_toml(&path);
        let sel = select_groups(&mg, &job.groups);
        if job.groups.is_empty() {
            job.groups = sel.keys().cloned().collect();
        }
        sel
    };
    // Names dropped via `exclude` must not gate the job (mirrors the binaries'
    // `-x` filtering); brace-expand and `>`-strip each entry to match resolved names.
    let excluded: HashSet<String> = job.exclude.iter()
        .flat_map(|raw| expand_specs(raw))
        .map(|spec| spec.trim_start_matches('>').to_string())
        .collect();
    let from_groups = groups.values().flat_map(|specs| specs.iter());
    for raw in from_groups.chain(job.models_manual.iter()).cloned().collect::<Vec<_>>() {
        for spec in expand_specs(&raw) {
            let name = spec.trim_start_matches('>');
            if excluded.contains(name) {
                continue;
            }
            for ds in ["{pr}", "{fulltrain_pr}"] {
                let inp = format!("{{preds}}/{}.{}.npy", name, ds);
                if !job.inputs.contains(&inp) {
                    job.inputs.push(inp);
                }
            }
        }
    }
}

/// Voting-feature wiring for jobs with a `voting_models` TOML. For `vfeat`
/// producer jobs, resolve the selected voting groups into the job's `{feature}`
/// list (so the names live once, in the voting TOML, not re-listed here). For
/// blend consumer jobs, add each voting feature's `{pr}`/`{fulltrain_pr}` file as
/// a gating input (covers both `vf/…` features and predictors used as voting).
fn expand_voting(name: &str, job: &mut JobConfig) {
    let Some(path) = job.voting_models.clone() else { return; };
    if job.voting.is_empty() {
        panic!(
            "job '{name}' (voting_models = \"{path}\") must specify voting explicitly \
             (e.g. voting = [\"all\"]); implicit all is not allowed"
        );
    }
    let specs = resolve_voting(&path, &job.voting);
    if job.jobtype.as_deref() == Some("vfeat") {
        if job.features.is_empty() {
            job.features = specs;
        }
        return;
    }
    for spec in &specs {
        for ds in ["{pr}", "{fulltrain_pr}"] {
            let inp = format!("{{preds}}/{}.{}.npy", spec, ds);
            if !job.inputs.contains(&inp) {
                job.inputs.push(inp);
            }
        }
    }
}

fn build_subst_vars(job_name: &str, job: &JobConfig, pipeline: &Pipeline) -> HashMap<String, String> {
    let mut vars: HashMap<String, String> = HashMap::new();
    for (k, v) in &pipeline.split {
        vars.insert(k.clone(), v.clone());
        vars.insert(format!("split.{}", k), v.clone());
    }
    vars.insert("name".to_string(), job_name.to_string());
    let binary = job.binary.clone().unwrap_or_else(|| job_name.to_string());
    vars.insert("binary".to_string(), binary);
    if let Some(r) = &job.runner { vars.insert("runner".to_string(), r.clone()); }
    if let Some(c) = &job.config { vars.insert("config".to_string(), c.clone()); }
    if let Some(t) = &job.target { vars.insert("target".to_string(), t.clone()); }
    if let Some(m) = &job.models { vars.insert("models".to_string(), m.clone()); }
    if !job.seeds.is_empty() {
        let csv = job.seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
        vars.insert("seeds".to_string(), csv);
    }
    // `{model_sel}` is the whole model-selection argument: the `--groups` flag
    // (omitted when the job names its predictors directly) plus one `-m` per
    // manual name. By this point expand_blend_models has filled in `groups` for
    // a job that left it empty, so there is never a bare `--groups all` to emit.
    let mut model_sel: Vec<String> = Vec::new();
    if !job.groups.is_empty() {
        model_sel.push(format!("--groups {}", job.groups.join(",")));
    }
    model_sel.extend(job.models_manual.iter().map(|m| format!("-m '{}'", m)));
    vars.insert("model_sel".to_string(), model_sel.join(" "));
    vars.insert("groups".to_string(), job.groups.join(","));
    if let Some(vm) = &job.voting_models { vars.insert("voting_models".to_string(), vm.clone()); }
    vars.insert("voting".to_string(), job.voting.join(","));
    // Render exclusions as repeated `-x '<name>'` flags (empty string when unset,
    // so a trailing `{exclude}` in the cmd template simply disappears).
    let exclude_flags = job.exclude.iter()
        .map(|e| format!("-x '{}'", e))
        .collect::<Vec<_>>()
        .join(" ");
    vars.insert("exclude".to_string(), exclude_flags);
    vars.insert("extra".to_string(), job.extra_args.join(" "));
    vars
}

fn substitute(s: &str, vars: &HashMap<String, String>) -> String {
    let mut out = s.to_string();
    loop {
        let mut changed = false;
        for (k, v) in vars {
            let pat = format!("{{{}}}", k);
            if out.contains(&pat) {
                out = out.replace(&pat, v);
                changed = true;
            }
        }
        if !changed { break; }
    }
    out
}

/// Expand `{feature}` placeholders in output templates. For each output
/// containing `{feature}`, emit one copy per feature name; templates without
/// `{feature}` are passed through unchanged. If `features` is empty, returns
/// `outputs` unchanged.
fn expand_features(outputs: &[String], features: &[String]) -> Vec<String> {
    if features.is_empty() {
        return outputs.to_vec();
    }
    let mut out = Vec::new();
    for tpl in outputs {
        if tpl.contains("{feature}") {
            for f in features {
                out.push(tpl.replace("{feature}", f));
            }
        } else {
            out.push(tpl.clone());
        }
    }
    out
}

/// If `keep_epochs` is set, append per-epoch variants of any output template
/// containing `{name}.{pr}.npy` or `{name}.{fulltrain_pr}.npy`. The original
/// outputs (final-epoch predictions) stay in the list — keep_epochs is additive.
fn expand_keep_epochs(outputs: &[String], keep: &[u32]) -> Vec<String> {
    let mut out = outputs.to_vec();
    for tpl in outputs {
        if tpl.contains("{name}.{pr}.npy") || tpl.contains("{name}.{fulltrain_pr}.npy") {
            for &e in keep {
                let s = tpl
                    .replace("{name}.{pr}.npy", &format!("{{name}}_ep{:02}.{{pr}}.npy", e))
                    .replace(
                        "{name}.{fulltrain_pr}.npy",
                        &format!("{{name}}_ep{:02}.{{fulltrain_pr}}.npy", e),
                    );
                out.push(s);
            }
        }
    }
    out
}

/// For blend jobs with `seeds`: expand each output template containing
/// `{name}.` into one per seed (`{name}-s<seed>.`). Templates without `{name}.`
/// pass through unchanged.
fn expand_seeds(outputs: &[String], seeds: &[u64]) -> Vec<String> {
    if seeds.is_empty() {
        return outputs.to_vec();
    }
    let mut out = Vec::new();
    for tpl in outputs {
        // Only per-seed prediction files (`.npy`) are seed-expanded; a single
        // `{name}.out` log covers the whole job, so it passes through unchanged.
        if tpl.contains("{name}.") && tpl.ends_with(".npy") {
            for s in seeds {
                out.push(tpl.replace("{name}.", &format!("{{name}}-s{s}.")));
            }
        } else {
            out.push(tpl.clone());
        }
    }
    out
}

pub fn resolve_pipeline(p: &Pipeline) -> IndexMap<String, ResolvedJob> {
    let mut out: IndexMap<String, ResolvedJob> = IndexMap::new();
    for (name, job) in &p.jobs {
        let mut merged = merge_with_defaults(job, &p.defaults);
        expand_extras(name, &mut merged);
        expand_eblend_inputs(name, &mut merged);
        expand_blend_models(&mut merged);
        expand_voting(name, &mut merged);
        let subst = build_subst_vars(name, &merged, p);
        let outputs_tpl = expand_features(&merged.outputs, &merged.features);
        let outputs_tpl = match &merged.keep_epochs {
            Some(epochs) if !epochs.is_empty() => expand_keep_epochs(&outputs_tpl, epochs),
            _ => outputs_tpl,
        };
        let outputs_tpl = expand_seeds(&outputs_tpl, &merged.seeds);
        out.insert(name.clone(), ResolvedJob {
            inputs: merged.inputs.iter().map(|s| substitute(s, &subst)).collect(),
            inputs_from: merged.inputs_from.clone(),
            outputs: outputs_tpl.iter().map(|s| substitute(s, &subst)).collect(),
            cmd: substitute(&merged.cmd, &subst),
        });
    }
    out
}

pub fn path_exists(p: &str) -> bool {
    if p.ends_with('/') {
        Path::new(p).is_dir()
    } else {
        Path::new(p).exists()
    }
}

pub fn collect_all_inputs(job: &ResolvedJob, resolved: &IndexMap<String, ResolvedJob>) -> Vec<String> {
    let mut all: Vec<String> = job.inputs.clone();
    for up in &job.inputs_from {
        if let Some(up_job) = resolved.get(up) {
            all.extend(up_job.outputs.iter().cloned());
        }
    }
    all
}

pub fn status_of(job: &ResolvedJob, resolved: &IndexMap<String, ResolvedJob>) -> Status {
    let inputs = collect_all_inputs(job, resolved);
    let missing: Vec<String> = inputs.iter().filter(|p| !path_exists(p)).cloned().collect();
    if !missing.is_empty() {
        return Status::Blocked(missing);
    }
    if job.outputs.iter().all(|p| path_exists(p)) {
        Status::Done
    } else {
        Status::Runnable
    }
}

/// Map each declared output path to the job that produces it. If two jobs
/// declare the same output (shouldn't happen in practice), the first one wins.
pub fn build_producers(resolved: &IndexMap<String, ResolvedJob>) -> HashMap<String, String> {
    let mut producers: HashMap<String, String> = HashMap::new();
    for (name, job) in resolved {
        for o in &job.outputs {
            producers.entry(o.clone()).or_insert_with(|| name.clone());
        }
    }
    producers
}

/// Every path any job references: the union of all outputs and all inputs
/// (including the ones inherited through `inputs_from`). `run --clean` protects
/// exactly this set; the remote store uploads exactly this set.
pub fn referenced_files(resolved: &IndexMap<String, ResolvedJob>) -> HashSet<String> {
    let mut referenced: HashSet<String> = HashSet::new();
    for (_, s) in resolved {
        for o in &s.outputs { referenced.insert(o.clone()); }
        for i in &s.inputs { referenced.insert(i.clone()); }
        for up in &s.inputs_from {
            if let Some(us) = resolved.get(up) {
                for o in &us.outputs { referenced.insert(o.clone()); }
            }
        }
    }
    referenced
}
