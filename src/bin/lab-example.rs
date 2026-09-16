// One-file experiment template: copy to src/bin/lab-<slug>.rs, replace the
// model, run it. No library module, no manifest job, no models TOML entry;
// predictions land in preds_lab/, so nothing here can disturb the stable set.
//
//   cargo run --release --bin lab-example -- --smoke   # minutes: does it run?
//   cargo run --release --bin lab-example              # full trainx -> probex
//   cargo run --release --bin lab-example -- --final   # + fulltrain -> qual
//
// See docs/EXPERIMENTS.md.

use netflix_prize::{
    Dataset, MaskedDataset, Regressor, calc_gbias, calc_user_offsets, fit2, get_users,
    lab::{LabArgs, ev},
};
use ndarray::Array1;

/// Deliberately the dullest model that fits the trait: a global mean plus a
/// user and an item bias, fitted by SGD. Replace it with the actual idea.
#[derive(Clone, Copy, Debug)]
struct LabConfig {
    n_epochs: usize,
    lr: f32,             // learning rate of both bias vectors
    reg: f32,            // L2 pull towards zero
    seed: u64,
    shuffle_users: bool, // user order within an epoch
}

struct LabModel {
    cfg: LabConfig,
    gbias: f32,          // mean residual over the training set
    ubias: Array1<f32>,  // per-user bias [n_users]
    ibias: Array1<f32>,  // per-item bias [n_items]
}

impl Regressor for LabModel {
    type Config = LabConfig;

    // `pr` hides the held-out ratings on purpose, so no amount of typing here
    // can peek at the targets being predicted.
    fn new(tr: &Dataset, _pr: &MaskedDataset, cfg: Self::Config) -> Self {
        Self {
            cfg,
            gbias: calc_gbias(tr),
            ubias: Array1::zeros(tr.n_users),
            ibias: Array1::zeros(tr.n_items),
        }
    }

    fn n_epochs(&self) -> usize { self.cfg.n_epochs }

    // Predicts the *residual* the dataset was loaded with: the rating itself
    // under the default "rtg" target, what the base model left over under a
    // "0.5*dnn-24" one.
    fn predict(&self, u: usize, i: usize, _day: i32) -> f32 {
        self.gbias + self.ubias[u] + self.ibias[i]
    }

    fn fit_epoch(&mut self, tr: &Dataset, _pr: &MaskedDataset, epoch: usize) {
        let cfg = self.cfg;
        // Training sets are user-sorted, so each user owns one contiguous block.
        // (Probe and qual are item-sorted; user offsets are meaningless there.)
        let user_offsets = calc_user_offsets(tr);
        let users = get_users(tr.n_users, cfg.shuffle_users, cfg.seed, epoch);

        for &u in users.iter() {
            for t in user_offsets[u]..user_offsets[u + 1] {
                let i = tr.item_idxs[t] as usize;
                let err = self.predict(u, i, 0) - tr.residuals[t];
                self.ubias[u] -= cfg.lr * (err + cfg.reg * self.ubias[u]);
                self.ibias[i] -= cfg.lr * (err + cfg.reg * self.ibias[i]);
            }
        }
    }
}

fn main() {
    let args = LabArgs::parse();

    // `ev` reads a knob from the environment, so a retry needs no recompile:
    // `LR=0.01 cargo run --release --bin lab-example`.
    let cfg = LabConfig {
        n_epochs: if args.sampled() { 1 } else { ev("EPOCHS", 10) },
        lr: ev("LR", 0.005),
        reg: ev("REG", 0.02),
        seed: 42,
        shuffle_users: true,
    };

    // Phase 1 (train -> probe) only; the qual phase costs as much again and is
    // worth running once the column has proved itself. `--final` adds it.
    fit2!(LabModel, cfg, &args.target, &args.name, args.split(),
          skip_fulltrain: !args.final_run);
}
