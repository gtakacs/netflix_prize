//! Context features derived from the base predictors, for the blends (fwls / mlp
//! voting features): per-rating statistics over the predictions of the selected
//! model groups, each group a model family. Every prediction is clipped to
//! [CLIP_MIN, CLIP_MAX] first, as the blends clip their model columns. Written for
//! the split's probe and qual sets as `<preds>/<out>/<feature>.<ds>.npy` (f32):
//!
//!   std_all, range_all   population std and max − min over all selected columns
//!   std_<g>              std within group g
//!   d_<g0>_<g>           mean of the first group minus the mean of group g (g ≠ g0)
//!   mean_c               mean over all columns − 3.6
//!
//!   predfeat -p pipeline-3.toml -t models-3.toml --groups tx,rbm,nn,other

use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use netflix_prize::blend::{flatten_groups, load_models_toml, select_groups, CLIP_MAX, CLIP_MIN};
use rayon::prelude::*;
use std::collections::HashMap;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut pipeline, mut models, mut groups, mut out) =
        (String::new(), String::new(), Vec::<String>::new(), "vfx".to_string());
    let mut i = 0;
    while i < argv.len() {
        let val = || argv.get(i + 1).cloned().unwrap_or_else(|| panic!("'{}' needs a value", argv[i]));
        match argv[i].as_str() {
            "-p" | "--pipeline" => pipeline = val(),
            "-t" | "--models" => models = val(),
            "--groups" => groups = val().split(',').map(|s| s.trim().to_string()).collect(),
            "--out" => out = val(),
            "-h" | "--help" => {
                println!("Usage: predfeat -p PIPELINE -t MODELS --groups G1,G2,... [--out DIR (default vfx)]");
                println!("  DIR is relative to the split's preds dir; G1 is the reference of the d_<G1>_<G> features.");
                return;
            }
            s => panic!("unknown argument '{s}' (see --help)"),
        }
        i += 2;
    }
    assert!(!pipeline.is_empty() && !models.is_empty() && groups.len() >= 2, "need -p, -t and at least two --groups");

    #[derive(serde::Deserialize)]
    struct P { split: HashMap<String, String> }
    let p: P = toml::from_str(&std::fs::read_to_string(&pipeline).unwrap()).unwrap();
    let preds = p.split["preds"].clone();
    let out_dir = format!("{preds}/{out}");
    std::fs::create_dir_all(&out_dir).unwrap();

    // Columns per group (a column in two groups counts in both).
    let mg = load_models_toml(&models);
    let cols: Vec<Vec<String>> = groups.iter().map(|g| {
        let sel = select_groups(&mg, std::slice::from_ref(g));
        flatten_groups(&sel, &[]).names
    }).collect();
    for (g, c) in groups.iter().zip(&cols) { println!("group {g}: {} columns", c.len()); }

    for ds in [p.split["pr"].clone(), p.split["fulltrain_pr"].clone()] {
        // x[g][j] = clipped predictions of column j of group g
        let x: Vec<Vec<Array1<f32>>> = cols.iter().map(|cs| cs.par_iter().map(|c| {
            let path = format!("{preds}/{c}.{ds}.npy");
            let a: Array1<f32> = read_npy(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            a.mapv(|v| v.clamp(CLIP_MIN, CLIP_MAX))
        }).collect()).collect();
        let n = x[0][0].len();
        let ng = groups.len();
        // per row: [std_all, range_all, mean_c, std_g..., d_g0_g...]
        let nf = 3 + ng + (ng - 1);
        let rows: Vec<Vec<f32>> = (0..n).into_par_iter().map(|r| {
            let (mut s_all, mut q_all, mut c_all) = (0.0f64, 0.0f64, 0usize);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            let mut mean = vec![0.0f64; ng];
            let mut std = vec![0.0f64; ng];
            for (g, xs) in x.iter().enumerate() {
                let (mut s, mut q) = (0.0f64, 0.0f64);
                for col in xs {
                    let v = col[r] as f64;
                    s += v; q += v * v;
                    lo = lo.min(v); hi = hi.max(v);
                }
                let k = xs.len() as f64;
                mean[g] = s / k;
                std[g] = (q / k - mean[g] * mean[g]).max(0.0).sqrt();
                s_all += s; q_all += q; c_all += xs.len();
            }
            let m_all = s_all / c_all as f64;
            let mut f = Vec::with_capacity(nf);
            f.push((q_all / c_all as f64 - m_all * m_all).max(0.0).sqrt() as f32);
            f.push((hi - lo) as f32);
            f.push((m_all - 3.6) as f32);
            f.extend(std.iter().map(|&v| v as f32));
            f.extend((1..ng).map(|g| (mean[0] - mean[g]) as f32));
            f
        }).collect();
        let mut names = vec!["std_all".to_string(), "range_all".to_string(), "mean_c".to_string()];
        names.extend(groups.iter().map(|g| format!("std_{g}")));
        names.extend(groups[1..].iter().map(|g| format!("d_{}_{g}", groups[0])));
        for (j, name) in names.iter().enumerate() {
            let col = Array1::from_iter(rows.iter().map(|f| f[j]));
            let path = format!("{out_dir}/{name}.{ds}.npy");
            write_npy(&path, &col).unwrap_or_else(|e| panic!("write {path}: {e}"));
            println!("{ds}: wrote {path} (mean {:.4})", col.mean().unwrap());
        }
    }
}
