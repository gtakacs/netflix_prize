#![deprecated(note = "Use `crate::tx::{TxConfig, TxModel}` directly")]

use crate::OrdinalHeadConfig;
use crate::tx::TxConfig;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Tsvdx4Config {
    pub n_feat: usize,
    pub n_epochs: usize,
    pub seed: u64,
    pub shuffle_users: bool,

    pub n_time_bins: usize,
    pub beta: f32,

    pub n_freq_bins: usize,

    pub lr_u: f32,
    pub lr_ud: f32,
    pub lr_u2: f32,
    pub lr_ub: f32,
    pub lr_ubd: f32,
    pub lr_i: f32,
    pub lr_ib: f32,
    pub lr_y: f32,
    pub lr_yb: f32,
    pub lr_yd: f32,
    pub lr_tu: f32,
    pub lr_ti: f32,
    pub lr_ta: f32,

    pub lr_ibf: f32,
    pub lr_iqf: f32,
    pub reg_iqf: f32,
    pub sigma_iqf: f32,

    pub lr_cu: f32,
    pub reg_cu: f32,

    pub reg_u: f32,
    pub reg_u2: f32,
    pub reg_ud: f32,
    pub reg_i: f32,
    pub reg_y: f32,
    pub reg_yd: f32,

    pub sigma_u: f32,
    pub sigma_i: f32,
    pub sigma_y: f32,
    pub sigma_yd: f32,

    pub reset_u_epoch: usize,

    pub ordinal_head: Option<OrdinalHeadConfig>,
    pub sum_err_bug: bool,
    pub save_ifeat: bool,
    pub low_memory: bool,

    /// When true, include probe/qual items in su (NSVD1) vectors during training.
    pub full_su: bool,
}

// ---------------------------------------------------------------------------
// Config conversion
// ---------------------------------------------------------------------------

pub fn into_tx_config(cfg: Tsvdx4Config) -> TxConfig {
    TxConfig {
        n_feat:        cfg.n_feat,
        n_epochs:      cfg.n_epochs,
        seed:          cfg.seed,
        shuffle_users: cfg.shuffle_users,

        n_time_bins:   cfg.n_time_bins,
        beta:          cfg.beta,
        n_freq_bins:   cfg.n_freq_bins,

        lr_u:          cfg.lr_u,
        lr_ud:         cfg.lr_ud,
        lr_u2:         cfg.lr_u2,
        lr_ub:         cfg.lr_ub,
        lr_ubd:        cfg.lr_ubd,
        lr_i:          cfg.lr_i,
        lr_ib:         cfg.lr_ib,
        lr_y:          cfg.lr_y,
        lr_yb:         cfg.lr_yb,
        lr_yd:         cfg.lr_yd,
        lr_tu:         cfg.lr_tu,
        lr_ti:         cfg.lr_ti,
        lr_ta:         cfg.lr_ta,

        lr_ibf:        cfg.lr_ibf,
        lr_iqf:        cfg.lr_iqf,
        reg_iqf:       cfg.reg_iqf,
        sigma_iqf:     cfg.sigma_iqf,

        lr_cu:         cfg.lr_cu,
        reg_cu:        cfg.reg_cu,

        reg_u:         cfg.reg_u,
        reg_u2:        cfg.reg_u2,
        reg_ud:        cfg.reg_ud,
        reg_i:         cfg.reg_i,
        reg_y:         cfg.reg_y,
        reg_yd:        cfg.reg_yd,

        sigma_u:       cfg.sigma_u,
        sigma_i:       cfg.sigma_i,
        sigma_y:       cfg.sigma_y,
        sigma_yd:      cfg.sigma_yd,

        reset_u_epoch: cfg.reset_u_epoch,

        // Neighborhood disabled
        max_neighbors: 0,
        lr_w:          0.0,
        lr_c:          0.0,
        reg_w:         0.0,
        reg_c:         0.0,

        // Same-day neighborhood disabled
        lr_w_day:      0.0,
        lr_c_day:      0.0,
        reg_w_day:     0.0,
        reg_c_day:     0.0,

        // No sub-model lr multipliers in tsvdx4
        w_bias:        1.0,
        w_factor:      1.0,
        w_nbr:         1.0,

        sum_err_bug:   cfg.sum_err_bug,

        lambda1:       0.0,
        lambda2:       0.0,

        ordinal_head:  cfg.ordinal_head,
        save_ifeat:    cfg.save_ifeat,
        low_memory:    cfg.low_memory,
        full_su:       cfg.full_su,
    }
}
