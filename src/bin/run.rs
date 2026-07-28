//! Pipeline runner. Reads a manifest like `pipeline-old.toml`, lists every
//! job with its current status, and runs a single job on demand. Status
//! is computed from input/output file existence; staleness checks and
//! transitive runs are intentionally out of scope for this initial version.

use indexmap::IndexMap;
use netflix_prize::blend::{expand_specs, load_models_toml, select_groups};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, ExitCode};

const DEFAULT_PIPELINE: &str = "pipeline-old.toml";

#[derive(Deserialize, Debug, Default)]
struct Pipeline {
    #[serde(default)]
    split: HashMap<String, String>,
    #[serde(default)]
    defaults: HashMap<String, JobConfig>,
    #[serde(default)]
    jobs: IndexMap<String, JobConfig>,
}

#[derive(Deserialize, Debug, Default, Clone)]
struct JobConfig {
    jobtype: Option<String>,
    #[serde(default)]
    inputs: Vec<String>,
    #[serde(default)]
    inputs_from: Vec<String>,
    #[serde(default)]
    outputs: Vec<String>,
    #[serde(default)]
    cmd: String,
    runner: Option<String>,
    config: Option<String>,
    target: Option<String>,
    keep_epochs: Option<Vec<u32>>,
    #[serde(default)]
    extras: Vec<String>,
    n_epochs: Option<u32>,
    #[serde(default)]
    non_epoch: Vec<String>,
    /// List of feature names that expand `{feature}` placeholders in outputs.
    /// Each output template containing `{feature}` is replicated once per
    /// feature name (with `{feature}` replaced); other templates are kept as-is.
    #[serde(default)]
    features: Vec<String>,
    /// Optional override for the `{binary}` substitution variable. When unset,
    /// `{binary}` resolves to the job name. Used to share a dispatcher
    /// binary across multiple jobs (the job name is passed to the binary
    /// as a CLI arg by the default `cmd`).
    binary: Option<String>,
    /// Path to a models TOML (groups of base-predictor names) for blend jobs.
    /// When set, the runner expands it into per-model prediction inputs (so the
    /// job gates on those predictions existing) and exposes the path to the
    /// `cmd` as `{models}`.
    models: Option<String>,
    /// Fold seeds for a blend job. When non-empty, each output template's
    /// `{name}.` is expanded to `{name}-s<seed>.` (one prediction pair per seed)
    /// and the comma-joined list is exposed to the `cmd` as `{seeds}`.
    #[serde(default)]
    seeds: Vec<u64>,
    /// Model-group names a blend job consumes from its `models` TOML. When
    /// non-empty the runner gates only on those groups' predictions and exposes
    /// the comma-joined list to the `cmd` as `{groups}`; empty resolves to `all`.
    #[serde(default)]
    groups: Vec<String>,
    /// Base-predictor names to drop from a blend job's model set. Each entry is
    /// brace-expanded (like a model spec) and `>`-stripped before matching. The
    /// runner both skips them when gating on `models` predictions and renders
    /// them to the `cmd` as `{exclude}` = `-x '<name>' ...` (empty when unset).
    #[serde(default)]
    exclude: Vec<String>,
}

#[derive(Debug)]
struct ResolvedJob {
    inputs: Vec<String>,
    inputs_from: Vec<String>,
    outputs: Vec<String>,
    cmd: String,
}

#[derive(Debug)]
enum Status {
    Done,
    Runnable,
    Blocked(Vec<String>),
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

/// For jobs with a `models` TOML: read it, flatten all groups, and add each
/// base predictor's `{pr}` and `{fulltrain_pr}` prediction files as inputs so
/// the runner gates on them. The `>` no-clip prefix is stripped. The paths are
/// kept as templates (`{preds}`, `{pr}`, `{fulltrain_pr}` substituted later).
fn expand_blend_models(job: &mut JobConfig) {
    let Some(path) = job.models.clone() else { return; };
    let mg = load_models_toml(&path);
    let groups = select_groups(&mg, &job.groups);
    // Names dropped via `exclude` must not gate the job (mirrors the binaries'
    // `-x` filtering); brace-expand and `>`-strip each entry to match resolved names.
    let excluded: std::collections::HashSet<String> = job.exclude.iter()
        .flat_map(|raw| expand_specs(raw))
        .map(|spec| spec.trim_start_matches('>').to_string())
        .collect();
    for specs in groups.values() {
        for raw in specs {
            for spec in expand_specs(raw) {
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
    let groups_csv = if job.groups.is_empty() { "all".to_string() } else { job.groups.join(",") };
    vars.insert("groups".to_string(), groups_csv);
    // Render exclusions as repeated `-x '<name>'` flags (empty string when unset,
    // so a trailing `{exclude}` in the cmd template simply disappears).
    let exclude_flags = job.exclude.iter()
        .map(|e| format!("-x '{}'", e))
        .collect::<Vec<_>>()
        .join(" ");
    vars.insert("exclude".to_string(), exclude_flags);
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

fn resolve_pipeline(p: &Pipeline) -> IndexMap<String, ResolvedJob> {
    let mut out: IndexMap<String, ResolvedJob> = IndexMap::new();
    for (name, job) in &p.jobs {
        let mut merged = merge_with_defaults(job, &p.defaults);
        expand_extras(name, &mut merged);
        expand_eblend_inputs(name, &mut merged);
        expand_blend_models(&mut merged);
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

fn path_exists(p: &str) -> bool {
    if p.ends_with('/') {
        Path::new(p).is_dir()
    } else {
        Path::new(p).exists()
    }
}

fn collect_all_inputs(job: &ResolvedJob, resolved: &IndexMap<String, ResolvedJob>) -> Vec<String> {
    let mut all: Vec<String> = job.inputs.clone();
    for up in &job.inputs_from {
        if let Some(up_job) = resolved.get(up) {
            all.extend(up_job.outputs.iter().cloned());
        }
    }
    all
}

fn status_of(job: &ResolvedJob, resolved: &IndexMap<String, ResolvedJob>) -> Status {
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
fn build_producers(resolved: &IndexMap<String, ResolvedJob>) -> HashMap<String, String> {
    let mut producers: HashMap<String, String> = HashMap::new();
    for (name, job) in resolved {
        for o in &job.outputs {
            producers.entry(o.clone()).or_insert_with(|| name.clone());
        }
    }
    producers
}

/// Split missing inputs into the producer jobs (deduplicated, in first-seen
/// order) and orphan paths (no job produces them — typically source files or
/// raw data).
fn diagnose_blocked(missing: &[String], producers: &HashMap<String, String>) -> (Vec<String>, Vec<String>) {
    let mut producer_jobs: Vec<String> = Vec::new();
    let mut orphans: Vec<String> = Vec::new();
    for m in missing {
        match producers.get(m) {
            Some(p) => {
                if !producer_jobs.contains(p) {
                    producer_jobs.push(p.clone());
                }
            }
            None => orphans.push(m.clone()),
        }
    }
    (producer_jobs, orphans)
}

fn format_blocked_detail(missing: &[String], producers: &HashMap<String, String>) -> String {
    let (jobs, orphans) = diagnose_blocked(missing, producers);
    if jobs.is_empty() {
        return if orphans.len() == 1 {
            format!("  (missing: {})", orphans[0])
        } else {
            format!("  (missing {}: {}, ...)", orphans.len(), orphans[0])
        };
    }
    let mut s = format!("  (blocked by: {}", jobs.join(", "));
    if !orphans.is_empty() {
        s.push_str(&format!(", +{} file{}", orphans.len(), if orphans.len() == 1 { "" } else { "s" }));
    }
    s.push(')');
    s
}

fn list_jobs(pipeline_path: &str, p: &Pipeline, resolved: &IndexMap<String, ResolvedJob>) {
    let split_name = p.split.get("name").cloned().unwrap_or_default();
    println!("Pipeline: {} (split = {})", pipeline_path, split_name);
    println!();
    let producers = build_producers(resolved);
    let mut n_done = 0;
    let mut n_runnable = 0;
    let mut n_blocked = 0;
    for (name, job) in resolved {
        let s = status_of(job, resolved);
        let label = match &s {
            Status::Done => { n_done += 1; "DONE" }
            Status::Runnable => { n_runnable += 1; "RUNNABLE" }
            Status::Blocked(_) => { n_blocked += 1; "BLOCKED" }
        };
        let detail = match &s {
            Status::Blocked(missing) => format_blocked_detail(missing, &producers),
            _ => String::new(),
        };
        let jobtype = p.jobs.get(name)
            .and_then(|s| s.jobtype.as_deref())
            .unwrap_or("-");
        println!("  {:30} {:15} {}{}", name, jobtype, label, detail);
    }
    println!();
    println!(
        "{} jobs total: {} DONE, {} RUNNABLE, {} BLOCKED",
        resolved.len(), n_done, n_runnable, n_blocked,
    );
}

fn run_job(job_name: &str, resolved: &IndexMap<String, ResolvedJob>, force: bool) -> ExitCode {
    let job = match resolved.get(job_name) {
        Some(s) => s,
        None => {
            eprintln!("error: unknown job '{}'", job_name);
            let names: Vec<&str> = resolved.keys().map(|s| s.as_str()).collect();
            eprintln!("available: {}", names.join(", "));
            return ExitCode::from(2);
        }
    };
    match status_of(job, resolved) {
        Status::Blocked(missing) => {
            eprintln!("error: job '{}' is BLOCKED", job_name);
            let producers = build_producers(resolved);
            let (jobs, orphans) = diagnose_blocked(&missing, &producers);
            if !jobs.is_empty() {
                eprintln!("  blocked by jobs:");
                for j in &jobs {
                    eprintln!("    - {}", j);
                }
            }
            if !orphans.is_empty() {
                eprintln!("  missing files:");
                for f in &orphans {
                    eprintln!("    - {}", f);
                }
            }
            return ExitCode::from(1);
        }
        Status::Done if !force => {
            println!("Job '{}' is DONE — skipping. Use -f to force re-run.", job_name);
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    println!("Running '{}': {}", job_name, job.cmd);
    match Command::new("sh").arg("-c").arg(&job.cmd).status() {
        Ok(es) if es.success() => ExitCode::SUCCESS,
        Ok(es) => {
            eprintln!("job exited with {}", es);
            ExitCode::from(es.code().unwrap_or(1) as u8)
        }
        Err(e) => {
            eprintln!("failed to spawn shell: {}", e);
            ExitCode::from(127)
        }
    }
}

fn print_help() {
    println!("Usage: run [-p FILE | -n] [-l] [-c] [-f] [JOB]");
    println!();
    println!("  -p FILE, --pipeline FILE   pipeline manifest (default: {})", DEFAULT_PIPELINE);
    println!("  -n, --new                  shortcut for -p pipeline-new.toml");
    println!("  -l, --list                 list jobs with status (default if no JOB)");
    println!("  -c, --clean                list (or delete with -f) files in preds dir");
    println!("                             not referenced by any active job");
    println!("  -f, --force                re-run JOB even if DONE; or actually delete with --clean");
    println!("  -h, --help                 show this help");
    println!("  JOB                        run the named job");
}

fn cmd_clean(p: &Pipeline, resolved: &IndexMap<String, ResolvedJob>, force: bool) -> ExitCode {
    use std::collections::HashSet;

    // Protected paths: union of all outputs and all inputs (including inputs_from-derived).
    let mut protected: HashSet<String> = HashSet::new();
    for (_, s) in resolved {
        for o in &s.outputs { protected.insert(o.clone()); }
        for i in &s.inputs { protected.insert(i.clone()); }
        for up in &s.inputs_from {
            if let Some(us) = resolved.get(up) {
                for o in &us.outputs { protected.insert(o.clone()); }
            }
        }
    }

    // Directories to scan: those declared in [split] under "preds".
    let mut dirs: Vec<String> = Vec::new();
    if let Some(d) = p.split.get("preds") { dirs.push(d.clone()); }

    let mut candidates: Vec<String> = Vec::new();
    for dir in &dirs {
        if !Path::new(dir).is_dir() { continue; }
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() { continue; }
            let path_str = path.to_string_lossy().to_string();
            if !protected.contains(&path_str) {
                candidates.push(path_str);
            }
        }
    }
    candidates.sort();

    if candidates.is_empty() {
        println!("No deletion candidates.");
        return ExitCode::SUCCESS;
    }

    let n = candidates.len();
    println!("{} candidate{} for deletion:", n, if n == 1 { "" } else { "s" });
    for c in &candidates {
        println!("  {}", c);
    }

    if !force {
        println!();
        println!("Dry run. Use --clean -f to delete.");
        return ExitCode::SUCCESS;
    }

    println!();
    println!("Deleting...");
    let mut errs = 0;
    for c in &candidates {
        if let Err(e) = std::fs::remove_file(c) {
            eprintln!("  error deleting {}: {}", c, e);
            errs += 1;
        }
    }
    if errs > 0 {
        eprintln!("{} deletion(s) failed.", errs);
        return ExitCode::from(1);
    }
    println!("Deleted {} file{}.", n, if n == 1 { "" } else { "s" });
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let mut pipeline_path = DEFAULT_PIPELINE.to_string();
    let mut job_arg: Option<String> = None;
    let mut force_list = false;
    let mut force = false;
    let mut clean_mode = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => { print_help(); return ExitCode::SUCCESS; }
            "-l" | "--list" => { force_list = true; i += 1; }
            "-f" | "--force" => { force = true; i += 1; }
            "-c" | "--clean" => { clean_mode = true; i += 1; }
            "-n" | "--new" => { pipeline_path = "pipeline-new.toml".to_string(); i += 1; }
            "-p" | "--pipeline" => {
                if i + 1 >= args.len() {
                    eprintln!("error: {} requires an argument", args[i]);
                    return ExitCode::from(2);
                }
                pipeline_path = args[i + 1].clone();
                i += 2;
            }
            s if s.starts_with('-') => {
                eprintln!("error: unknown flag '{}'", s);
                print_help();
                return ExitCode::from(2);
            }
            s => {
                if job_arg.is_some() {
                    eprintln!("error: only one job argument allowed");
                    return ExitCode::from(2);
                }
                job_arg = Some(s.to_string());
                i += 1;
            }
        }
    }

    let content = match std::fs::read_to_string(&pipeline_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to read {}: {}", pipeline_path, e);
            return ExitCode::from(2);
        }
    };
    let pipeline: Pipeline = match toml::from_str(&content) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to parse {}: {}", pipeline_path, e);
            return ExitCode::from(2);
        }
    };
    let resolved = resolve_pipeline(&pipeline);

    if clean_mode {
        if job_arg.is_some() {
            eprintln!("warning: JOB argument ignored with --clean");
        }
        return cmd_clean(&pipeline, &resolved, force);
    }

    match job_arg {
        Some(name) => {
            if force_list {
                eprintln!("warning: -l ignored when a JOB is specified");
            }
            run_job(&name, &resolved, force)
        }
        None => {
            if force {
                eprintln!("warning: -f ignored without a JOB");
            }
            list_jobs(&pipeline_path, &pipeline, &resolved);
            ExitCode::SUCCESS
        }
    }
}
