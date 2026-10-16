"""
Section 6.3.2 — Safe batch update: audit incoming batches before updating a standard NLL model.

Fits all configured Hölder proposals (default: DPD, ExpSatDPD, RatSatDPD when holder_phi is omitted).
No NLL/Gamma/Student-t score baselines.

Default ``--preprocessing kanamori`` aligns batch contamination with Sec.~6.2: z-score **x** and **y**
from clean audit data, fixed base/batch/test split per replicate, then contaminate the batch slice
in standardized-``y`` space with :math:`\\mathcal{N}(\\mu_{\\mathrm{out}}, 1)` at rate ``eps_batch_design``.
``legacy`` restores shift/heavy-tail batch contamination.

Compares BaseOnly | NaiveUpdate | RandomUpdate | ProposedUpdate{DPD,ExpSat,RatSat} | OracleUpdate.
Random removal uses k_est from the reference proposal (default DPD) on the batch only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

if "MALLOC_TRIM_THRESHOLD_" not in os.environ:
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.exp1_regression.cleaning_utils import (
    contaminate_batch_response,
    evaluate_batch_filtering,
    sample_base_batch_test,
)
from experiments.exp1_regression.run_exp1_regression import (  # noqa: E402
    NumpyEncoder,
    get_primary_proposal_method,
)
from experiments.exp1_regression.run_exp1_realdata_benchmark import (  # noqa: E402
    load_benchmark_dataset,
    parse_csv_list,
    _sanitize_dataset_name,
)
from experiments.exp1_regression.utils_exp1_regression import (  # noqa: E402
    PROPOSAL_METHOD_ORDER,
    compute_dpd_outlier_prob,
    contaminate_response,
    evaluate_detection_metrics,
    fit_downstream_nll_and_evaluate,
    fit_regression_methods,
    kanamori_standardize_regression_data,
    make_realdata_contamination_seed,
    make_realdata_fit_seed,
    resolve_realdata_split_seed,
    standardize_regression_data,
    summarize_dpd_posterior,
)

PROPOSED_UPDATE_LABEL = {
    "DPD": "ProposedUpdateDPD",
    "ExpSatDPD": "ProposedUpdateExpSatDPD",
    "RatSatDPD": "ProposedUpdateRatSatDPD",
}


def _proposed_update_strategy_name(pname: str) -> str:
    return PROPOSED_UPDATE_LABEL.get(pname, f"ProposedUpdate{pname}")


def _preferred_update_bar_order(have: set) -> List[str]:
    preferred = [
        "BaseOnly",
        "NaiveUpdate",
        "RandomUpdate",
        "ProposedUpdateDPD",
        "ProposedUpdateExpSatDPD",
        "ProposedUpdateRatSatDPD",
        "OracleUpdate",
    ]
    return [s for s in preferred if s in have]


def _preferred_pr_updates(have: set) -> List[str]:
    preferred = [
        "ProposedUpdateDPD",
        "ProposedUpdateExpSatDPD",
        "ProposedUpdateRatSatDPD",
        "RandomUpdate",
        "OracleUpdate",
    ]
    return [s for s in preferred if s in have]


def _sanitize_mode(mode: str) -> str:
    m = str(mode).strip().lower().replace("-", "_")
    if m in {"heavytail"}:
        return "heavy_tail"
    return m


def _mode_dir_name(mode: str) -> str:
    return f"mode_{_sanitize_mode(mode)}"


def _make_fit_seed(
    cfg: "SafeBatchConfig", rep: int, eps: float, mu_out: float, gamma: float, mode: str
) -> int:
    mode_hash = sum(ord(c) for c in _sanitize_mode(mode))
    return make_realdata_fit_seed(cfg.base_seed, rep, eps, mu_out, gamma) + mode_hash % 997


@dataclass
class SafeBatchConfig:
    n_base: int = 500
    n_batch: int = 500
    n_test: int = 1000
    datasets: List[str] | None = None
    gammas: List[float] | None = None
    eps_batch_grid: List[float] | None = None
    mu_out_grid: List[float] | None = None
    batch_contam_modes: List[str] | None = None
    replicates: int = 10
    student_t_df: float = 4.0
    batch_t_df: float = 1.5
    base_seed: int = 42
    output_dir: str = "outputs/exp1_realdata_safe_batch_update"
    num_warmup: int = 1000
    num_samples: int = 1000
    chains: int = 4
    downstream_num_warmup: int | None = None
    downstream_num_samples: int | None = None
    downstream_chains: int | None = None
    eta_mode: str = "covdet"
    eta_fixed: float = 1.0
    holder_phi: str | None = None
    holder_kappa: float | None = None
    target_accept: float = 0.8
    init_step: float = 1.0
    local_data_dir: str | None = None
    preprocessing: str = "kanamori"
    split_mode: str = "fixed"
    contam_unit: str = "std"
    audit_scope: str = "combined"
    save_raw: bool = False

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["cal_housing", "pumadyn32fh", "ccpp"]
        if self.gammas is None:
            self.gammas = [0.1, 0.5, 1.0]
        if self.eps_batch_grid is None:
            self.eps_batch_grid = [0.05, 0.2]
        if self.mu_out_grid is None:
            self.mu_out_grid = [6.0]
        if self.batch_contam_modes is None:
            self.batch_contam_modes = ["shift", "heavy_tail"]
        prep = str(self.preprocessing).strip().lower()
        if prep == "kanamori":
            # Align with Sec.~6.2: additive N(mu_out,1) on Kanamori-standardized y only.
            self.batch_contam_modes = ["shift"]


def run_single_safe_batch(
    cfg: SafeBatchConfig,
    dataset_name: str,
    gamma: float,
    eps_design: float,
    mu_out: float,
    batch_contam_mode: str,
    rep_id: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    mode_s = _sanitize_mode(batch_contam_mode)
    fit_seed = _make_fit_seed(cfg, rep_id, eps_design, mu_out, gamma, mode_s)
    split_seed = resolve_realdata_split_seed(
        cfg.base_seed, rep_id, eps_design, mu_out, split_mode=cfg.split_mode
    )
    contam_seed = make_realdata_contamination_seed(cfg.base_seed, rep_id, eps_design, mu_out)

    x_base_raw, y_base_raw, x_batch_raw, y_batch_clean, x_test_raw, y_test_raw = sample_base_batch_test(
        x_all, y_all, cfg.n_base, cfg.n_batch, cfg.n_test, seed=split_seed
    )

    prep = str(cfg.preprocessing).strip().lower()
    if prep == "kanamori":
        if cfg.audit_scope == "combined":
            x_audit_raw = np.vstack([x_base_raw, x_batch_raw])
            y_audit_clean_raw = np.concatenate([y_base_raw, y_batch_clean])
            n_audit = cfg.n_base + cfg.n_batch
            audit_is_batch = np.zeros(n_audit, dtype=bool)
            audit_is_batch[cfg.n_base :] = True
            audit_batch_indices = np.where(audit_is_batch)[0]
            x_audit_d, y_clean_sc, _, scaler = kanamori_standardize_regression_data(
                x_audit_raw,
                y_audit_clean_raw,
                x_audit_raw,
            )
            y_sc = np.asarray(y_clean_sc, dtype=float).copy()
            batch_seg = y_sc[cfg.n_base :]
            if eps_design > 0:
                batch_cont_sc, mask_batch_outlier = contaminate_response(
                    batch_seg, eps_design, mu_out, seed=contam_seed
                )
            else:
                batch_cont_sc = batch_seg.copy()
                mask_batch_outlier = np.zeros(cfg.n_batch, dtype=bool)
            y_sc[cfg.n_base :] = batch_cont_sc
            y_audit = jnp.asarray(y_sc, dtype=jnp.float32)
            x_audit = x_audit_d
            y_audit_raw_contam = scaler["y_mean"] + scaler["y_std"] * y_sc
            y_batch_contam = y_audit_raw_contam[cfg.n_base :]
            mask_audit_outlier = np.zeros(n_audit, dtype=bool)
            mask_audit_outlier[cfg.n_base :] = mask_batch_outlier
        elif cfg.audit_scope == "batch":
            x_audit_raw = x_batch_raw
            y_audit_clean_raw = y_batch_clean
            n_audit = cfg.n_batch
            audit_batch_indices = np.arange(cfg.n_batch)
            x_audit_d, y_clean_sc, _, scaler = kanamori_standardize_regression_data(
                x_audit_raw,
                y_audit_clean_raw,
                x_audit_raw,
            )
            y_clean_np = np.asarray(y_clean_sc, dtype=float)
            if eps_design > 0:
                y_cont_sc, mask_batch_outlier = contaminate_response(
                    y_clean_np, eps_design, mu_out, seed=contam_seed
                )
            else:
                y_cont_sc = y_clean_np.copy()
                mask_batch_outlier = np.zeros(cfg.n_batch, dtype=bool)
            y_audit = jnp.asarray(y_cont_sc, dtype=jnp.float32)
            x_audit = x_audit_d
            y_batch_contam = scaler["y_mean"] + scaler["y_std"] * y_cont_sc
            mask_audit_outlier = mask_batch_outlier.copy()
        else:
            raise ValueError(f"Unknown audit_scope: {cfg.audit_scope}")
        eps_batch_true = float(np.mean(mask_batch_outlier.astype(float)))
        eps_audit_true = float(np.mean(mask_audit_outlier.astype(float)))

    elif prep == "legacy":
        y_batch_contam, mask_batch_outlier = contaminate_batch_response(
            y_batch_clean,
            eps_design,
            mu_out,
            seed=contam_seed,
            mode=mode_s,
            unit=cfg.contam_unit,
            t_df=cfg.batch_t_df,
        )
        eps_batch_true = float(np.mean(mask_batch_outlier.astype(float)))
        if cfg.audit_scope == "combined":
            x_audit_raw = np.vstack([x_base_raw, x_batch_raw])
            y_audit_raw = np.concatenate([y_base_raw, y_batch_contam])
            n_audit = cfg.n_base + cfg.n_batch
            audit_is_batch = np.zeros(n_audit, dtype=bool)
            audit_is_batch[cfg.n_base :] = True
            audit_batch_indices = np.where(audit_is_batch)[0]
            mask_audit_outlier = np.zeros(n_audit, dtype=bool)
            mask_audit_outlier[cfg.n_base :] = mask_batch_outlier
        elif cfg.audit_scope == "batch":
            x_audit_raw = x_batch_raw
            y_audit_raw = y_batch_contam
            n_audit = cfg.n_batch
            audit_batch_indices = np.arange(cfg.n_batch)
            mask_audit_outlier = mask_batch_outlier.copy()
        else:
            raise ValueError(f"Unknown audit_scope: {cfg.audit_scope}")
        eps_audit_true = float(np.mean(mask_audit_outlier.astype(float)))
        x_audit, y_audit, _, _ = standardize_regression_data(x_audit_raw, y_audit_raw, x_audit_raw)
    else:
        raise ValueError(f"Unknown preprocessing: {cfg.preprocessing}")

    fit_res = fit_regression_methods(
        x_audit,
        y_audit,
        seed=fit_seed,
        chains=cfg.chains,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        gamma=gamma,
        eta_mode=cfg.eta_mode,
        eta_fixed=cfg.eta_fixed,
        holder_phi=cfg.holder_phi,
        holder_kappa=cfg.holder_kappa,
        student_t_df=cfg.student_t_df,
        target_accept=cfg.target_accept,
        init_step=cfg.init_step,
        fit_score_baselines=False,
    )

    fitted_proposals = [name for name in PROPOSAL_METHOD_ORDER if name in fit_res]
    if not fitted_proposals:
        fitted_proposals = list(fit_res.keys())

    extras: Dict[str, Dict[str, Any]] = {}
    for pname in fitted_proposals:
        extras[pname] = compute_dpd_outlier_prob(
            fit_res[pname]["samples"],
            x_audit,
            y_audit,
            seed=fit_seed,
        )

    ref_name = "DPD" if "DPD" in extras else fitted_proposals[0]

    p_out_batch_by_prop: Dict[str, np.ndarray] = {}
    k_est_batch_by_prop: Dict[str, int] = {}
    k_est_audit_total_by_prop: Dict[str, int] = {}
    proposed_removed_by_prop: Dict[str, np.ndarray] = {}

    for pname in fitted_proposals:
        ex = extras[pname]
        p_out_audit = ex["p_out"]
        k_est_audit_total_by_prop[pname] = int(round(ex["k_mean"]))
        p_out_batch_by_prop[pname] = p_out_audit[audit_batch_indices]
        kb = int(np.clip(k_est_audit_total_by_prop[pname], 0, cfg.n_batch))
        kb = int(np.clip(kb, 0, int(0.5 * cfg.n_batch)))
        k_est_batch_by_prop[pname] = kb
        pb = p_out_batch_by_prop[pname]
        proposed_removed_by_prop[pname] = np.argsort(pb)[::-1][:kb] if kb > 0 else np.array([], dtype=int)

    k_ref = k_est_batch_by_prop[ref_name]
    k_oracle_batch = int(np.sum(mask_batch_outlier))

    rng = np.random.default_rng(fit_seed + 12345)
    random_removed = rng.choice(cfg.n_batch, size=k_ref, replace=False) if k_ref > 0 else np.array([], dtype=int)

    def make_batch_keep(removed_local: np.ndarray) -> np.ndarray:
        km = np.ones(cfg.n_batch, dtype=bool)
        if removed_local.size > 0:
            km[removed_local] = False
        return km

    random_keep = make_batch_keep(random_removed)
    oracle_keep = ~mask_batch_outlier

    train_sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "BaseOnly": (x_base_raw, y_base_raw),
        "NaiveUpdate": (
            np.vstack([x_base_raw, x_batch_raw]),
            np.concatenate([y_base_raw, y_batch_contam]),
        ),
        "OracleUpdate": (
            np.vstack([x_base_raw, x_batch_raw[oracle_keep]]),
            np.concatenate([y_base_raw, y_batch_contam[oracle_keep]]),
        ),
        "RandomUpdate": (
            np.vstack([x_base_raw, x_batch_raw[random_keep]]),
            np.concatenate([y_base_raw, y_batch_contam[random_keep]]),
        ),
    }
    proposed_removed_by_strategy: Dict[str, np.ndarray] = {}
    for pname in fitted_proposals:
        strat = _proposed_update_strategy_name(pname)
        pk = make_batch_keep(proposed_removed_by_prop[pname])
        train_sets[strat] = (
            np.vstack([x_base_raw, x_batch_raw[pk]]),
            np.concatenate([y_base_raw, y_batch_contam[pk]]),
        )
        proposed_removed_by_strategy[strat] = proposed_removed_by_prop[pname]

    update_order = (
        ["BaseOnly", "NaiveUpdate", "RandomUpdate"]
        + [_proposed_update_strategy_name(p) for p in fitted_proposals]
        + ["OracleUpdate"]
    )

    d_chains = cfg.downstream_chains if cfg.downstream_chains is not None else cfg.chains
    d_warm = cfg.downstream_num_warmup if cfg.downstream_num_warmup is not None else cfg.num_warmup
    d_samp = cfg.downstream_num_samples if cfg.downstream_num_samples is not None else cfg.num_samples

    updates_out: Dict[str, Any] = {}
    for name in update_order:
        x_tr, y_tr = train_sets[name]
        seed_d = fit_seed + 8000 + sum(ord(c) for c in name) % 9000
        m = fit_downstream_nll_and_evaluate(
            x_tr,
            y_tr,
            x_test_raw,
            y_test_raw,
            seed=seed_d,
            chains=d_chains,
            num_warmup=d_warm,
            num_samples=d_samp,
            target_accept=cfg.target_accept,
            init_step=cfg.init_step,
        )
        m.pop("n_train_clean", None)
        n_up = int(len(y_tr))
        if name == "BaseOnly":
            n_br = 0
            brf = 0.0
        else:
            n_br = int(n_up - cfg.n_base)
            brf = float(n_br / cfg.n_batch) if cfg.n_batch > 0 else 0.0
        m["n_update"] = n_up
        m["n_batch_retained"] = n_br
        m["batch_retained_fraction"] = brf
        updates_out[name] = m

    rmse_base = updates_out["BaseOnly"]["rmse_test"]
    rmse_naive = updates_out["NaiveUpdate"]["rmse_test"]
    nlpd_base = updates_out["BaseOnly"]["nlpd_test"]
    nlpd_naive = updates_out["NaiveUpdate"]["nlpd_test"]

    for name in update_order:
        u = updates_out[name]
        u["rmse_delta_from_base"] = float(u["rmse_test"] - rmse_base)
        u["rmse_delta_from_naive"] = float(u["rmse_test"] - rmse_naive)
        rd = (rmse_naive - u["rmse_test"]) / max(rmse_naive - rmse_base, 1e-12)
        u["rmse_degradation_prevented"] = float(rd)
        u["rmse_degradation_prevented_clipped"] = float(np.clip(rd, -1.0, 1.5))

        u["nlpd_delta_from_base"] = float(u["nlpd_test"] - nlpd_base)
        u["nlpd_delta_from_naive"] = float(u["nlpd_test"] - nlpd_naive)
        nd = (nlpd_naive - u["nlpd_test"]) / max(nlpd_naive - nlpd_base, 1e-12)
        u["nlpd_degradation_prevented"] = float(nd)
        u["nlpd_degradation_prevented_clipped"] = float(np.clip(nd, -1.0, 1.5))

        if name == "RandomUpdate":
            u.update(evaluate_batch_filtering(random_removed, mask_batch_outlier, cfg.n_batch))
        elif name == "OracleUpdate":
            oracle_removed = np.where(mask_batch_outlier)[0]
            u.update(evaluate_batch_filtering(oracle_removed, mask_batch_outlier, cfg.n_batch))
        elif name in proposed_removed_by_strategy:
            u.update(
                evaluate_batch_filtering(
                    proposed_removed_by_strategy[name], mask_batch_outlier, cfg.n_batch
                )
            )

    audits: Dict[str, Dict[str, Any]] = {}
    for pname in fitted_proposals:
        det_b = evaluate_detection_metrics(
            p_out_batch_by_prop[pname],
            mask_batch_outlier.astype(int),
            k_oracle=k_oracle_batch,
            k_est=k_est_batch_by_prop[pname],
            is_prob=True,
        )
        audit_eps = summarize_dpd_posterior(fit_res[pname]["samples"], eps_true=eps_audit_true)
        audits[pname] = {
            **audit_eps,
            "auprc_batch": float(det_b["auprc"]),
            "auroc_batch": float(det_b["auroc"]),
            "brier_batch": float(det_b.get("brier", np.nan)),
            "recall_k_oracle_batch": float(det_b["recall_k_oracle"]),
            "recall_k_est_batch": float(det_b["recall_k_est"]),
            "k_est_audit_total": k_est_audit_total_by_prop[pname],
            "k_est_batch": k_est_batch_by_prop[pname],
        }

    primary = get_primary_proposal_method(cfg)

    payload: Dict[str, Any] = {
        "dataset": dataset_name,
        "gamma": gamma,
        "preprocessing": prep,
        "split_mode": str(cfg.split_mode),
        "batch_contam_mode": mode_s,
        "eps_batch_design": eps_design,
        "eps_batch_true": eps_batch_true,
        "eps_audit_true": eps_audit_true,
        "mu_out": mu_out,
        "replicate": rep_id,
        "n_base": cfg.n_base,
        "n_batch": cfg.n_batch,
        "n_test": cfg.n_test,
        "audit_scope": cfg.audit_scope,
        "contam_unit": (cfg.contam_unit if prep == "legacy" else None),
        "proposal_method_primary": primary,
        "proposal_methods_fitted": fitted_proposals,
        "k_oracle_batch": k_oracle_batch,
        "k_est_batch_reference": k_ref,
        "k_est_batch_by_proposal": k_est_batch_by_prop,
        "k_est_audit_total_by_proposal": k_est_audit_total_by_prop,
        "audits": audits,
        "updates": updates_out,
    }

    raw_bundle = None
    if cfg.save_raw:
        raw_bundle = {
            "x_base_raw": x_base_raw,
            "y_base_raw": y_base_raw,
            "x_batch_raw": x_batch_raw,
            "y_batch_clean": y_batch_clean,
            "y_batch_contam": y_batch_contam,
            "mask_batch_outlier": mask_batch_outlier,
            "x_test_raw": x_test_raw,
            "y_test_raw": y_test_raw,
        }

    return payload, raw_bundle


def _save_metrics(
    cfg: SafeBatchConfig,
    payload: Dict[str, Any],
    dataset_name: str,
    gamma: float,
    eps_design: float,
    mu_out: float,
    mode: str,
    rep_id: int,
    raw_bundle: Dict[str, Any] | None,
) -> None:
    run_dir = os.path.join(
        cfg.output_dir,
        f"dataset_{dataset_name}",
        f"gamma_{gamma:.3f}",
        _mode_dir_name(mode),
        f"epsbatch_{eps_design:.2f}",
        f"muout_{mu_out:.1f}",
        f"rep_{rep_id:03d}",
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    if raw_bundle is not None:
        np.savez_compressed(
            os.path.join(run_dir, "raw.npz"),
            **{k: np.asarray(v) for k, v in raw_bundle.items()},
        )


def _rows_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    base = {k: payload[k] for k in payload if k not in {"updates", "audits"}}
    for strat, d in payload["updates"].items():
        r = {**base, "update_strategy": strat, **d}
        rows.append(r)
    return rows


def _plot_faceted_bar(
    df: pd.DataFrame,
    value_col: str,
    out_path: str,
    ylabel: str,
    title_prefix: str = "",
) -> None:
    if df.empty:
        return
    datasets = sorted(df["dataset"].unique())
    modes = sorted(df["batch_contam_mode"].unique())
    nrows, ncols = len(datasets), len(modes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, mo in enumerate(modes):
            ax = axes[i][j]
            sub = df[(df["dataset"] == ds) & (df["batch_contam_mode"] == mo)]
            if sub.empty:
                ax.set_visible(False)
                continue
            means = []
            order = _preferred_update_bar_order(set(sub["update_strategy"].unique()))
            for s in order:
                sl = sub[sub["update_strategy"] == s][value_col]
                means.append(sl.mean() if len(sl) else np.nan)
            x = np.arange(len(order))
            ax.bar(x, means, color="steelblue", alpha=0.88)
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title_prefix}{ds} | {mo}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_degradation_prevented(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        return
    datasets = sorted(df["dataset"].unique())
    modes = sorted(df["batch_contam_mode"].unique())
    nrows, ncols = len(datasets), len(modes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, mo in enumerate(modes):
            ax = axes[i][j]
            sub = df[(df["dataset"] == ds) & (df["batch_contam_mode"] == mo)]
            if sub.empty:
                ax.set_visible(False)
                continue
            means = []
            order = [
                s
                for s in _preferred_update_bar_order(set(sub["update_strategy"].unique()))
                if s not in {"BaseOnly", "NaiveUpdate"}
            ]
            for s in order:
                sl = sub[sub["update_strategy"] == s]["rmse_degradation_prevented_clipped"]
                means.append(sl.mean() if len(sl) else np.nan)
            x = np.arange(len(order))
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.bar(x, means, color="darkorange", alpha=0.88)
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("RMSE degradation prevented (clipped)")
            ax.set_title(f"{ds} | {mo}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_precision_recall(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        return
    datasets = sorted(df["dataset"].unique())
    modes = sorted(df["batch_contam_mode"].unique())
    nrows, ncols = len(datasets), len(modes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, mo in enumerate(modes):
            ax = axes[i][j]
            sub = df[(df["dataset"] == ds) & (df["batch_contam_mode"] == mo)]
            if sub.empty:
                ax.set_visible(False)
                continue
            labels = [s for s in _preferred_pr_updates(set(sub["update_strategy"].unique()))]
            if not labels:
                ax.set_visible(False)
                continue
            prec = [sub[sub["update_strategy"] == s]["removed_precision"].mean() for s in labels]
            rec = [sub[sub["update_strategy"] == s]["removed_recall"].mean() for s in labels]
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w / 2, prec, width=w, label="precision", color="#4c78a8")
            ax.bar(x + w / 2, rec, width=w, label="recall", color="#f58518")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7)
            ax.set_title(f"{ds} | {mo}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gamma_sensitivity(df: pd.DataFrame, out_path: str) -> None:
    sub = df[np.isclose(df["eps_batch_design"], 0.2) & np.isclose(df["mu_out"], 6.0)]
    if sub.empty:
        return
    datasets = sorted(sub["dataset"].unique())
    modes = sorted(sub["batch_contam_mode"].unique())
    nrows, ncols = len(datasets), len(modes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, mo in enumerate(modes):
            ax = axes[i][j]
            sdf = sub[(sub["dataset"] == ds) & (sub["batch_contam_mode"] == mo)]
            if sdf.empty:
                ax.set_visible(False)
                continue
            gammas = sorted(sdf["gamma"].unique())
            for strat in _preferred_update_bar_order(set(sdf["update_strategy"].unique())):
                ys = []
                for g in gammas:
                    sl = sdf[np.isclose(sdf["gamma"], g) & (sdf["update_strategy"] == strat)]["rmse_test"]
                    ys.append(sl.mean() if len(sl) else np.nan)
                ax.plot(gammas, ys, marker="o", label=strat, linewidth=1.2)
            ax.set_xlabel("gamma")
            ax.set_ylabel("mean RMSE")
            ax.legend(fontsize=5, loc="best")
            ax.set_title(f"{ds} | {mo}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Section 6.3.2 safe batch update experiment")
    p.add_argument("--n_base", type=int, default=500)
    p.add_argument("--n_batch", type=int, default=500)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument(
        "--datasets",
        type=str,
        default="cal_housing,pumadyn32fh,ccpp",
        help="e.g. cal_housing,concrete,kin8nm (see run_exp1_realdata_benchmark.DATASET_ALIASES)",
    )
    p.add_argument("--gammas", type=str, default="0.1,0.5,1.0")
    p.add_argument("--eps_batch_grid", type=str, default="0.05,0.2")
    p.add_argument("--mu_out_grid", type=str, default="6")
    p.add_argument("--batch_contam_modes", type=str, default="shift,heavy_tail")
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--student_t_df", type=float, default=4.0)
    p.add_argument("--batch_t_df", type=float, default=1.5, help="df for heavy_tail batch noise (legacy only)")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--num_warmup", type=int, default=1000)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--downstream_chains", type=int, default=None)
    p.add_argument("--downstream_num_warmup", type=int, default=None)
    p.add_argument("--downstream_num_samples", type=int, default=None)
    p.add_argument("--eta_mode", type=str, default="covdet")
    p.add_argument("--eta_fixed", type=float, default=1.0)
    p.add_argument(
        "--holder_phi",
        type=str,
        default=None,
        help="Single Hölder proposal (e.g. dpd). Omit for all three: DPD, ExpSatDPD, RatSatDPD.",
    )
    p.add_argument("--holder_kappa", type=float, default=None)
    p.add_argument("--target_accept", type=float, default=0.8)
    p.add_argument("--init_step", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, default="outputs/exp1_realdata_safe_batch_update")
    p.add_argument("--local_data_dir", type=str, default=None)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument(
        "--preprocessing",
        type=str,
        default="kanamori",
        choices=["kanamori", "legacy"],
        help="kanamori: z-score x and y from clean audit data, batch contamination in scaled-y space. "
        "legacy: shift/heavy-tail batch contamination.",
    )
    p.add_argument(
        "--split_mode",
        type=str,
        default="fixed",
        choices=["fixed", "per_setting"],
        help="fixed: same base/batch/test split per replicate across eps/mu.",
    )
    p.add_argument(
        "--contam_unit",
        type=str,
        default="std",
        choices=["std", "raw"],
        help="Only used when --preprocessing legacy.",
    )
    p.add_argument("--audit_scope", type=str, default="combined", choices=["combined", "batch"])
    p.add_argument("--save_raw", action="store_true")

    args = p.parse_args()

    modes_list = [_sanitize_mode(s) for s in parse_csv_list(args.batch_contam_modes, str)]

    cfg = SafeBatchConfig(
        n_base=args.n_base,
        n_batch=args.n_batch,
        n_test=args.n_test,
        datasets=[_sanitize_dataset_name(v) for v in parse_csv_list(args.datasets, str)],
        gammas=parse_csv_list(args.gammas, float),
        eps_batch_grid=parse_csv_list(args.eps_batch_grid, float),
        mu_out_grid=parse_csv_list(args.mu_out_grid, float),
        batch_contam_modes=modes_list,
        replicates=args.replicates,
        student_t_df=args.student_t_df,
        batch_t_df=args.batch_t_df,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        chains=args.chains,
        downstream_chains=args.downstream_chains,
        downstream_num_warmup=args.downstream_num_warmup,
        downstream_num_samples=args.downstream_num_samples,
        eta_mode=args.eta_mode,
        eta_fixed=args.eta_fixed,
        holder_phi=(None if args.holder_phi is None else str(args.holder_phi).strip() or None),
        holder_kappa=args.holder_kappa,
        target_accept=args.target_accept,
        init_step=args.init_step,
        local_data_dir=args.local_data_dir,
        preprocessing=args.preprocessing,
        split_mode=args.split_mode,
        contam_unit=args.contam_unit,
        audit_scope=args.audit_scope,
        save_raw=args.save_raw,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    all_rows: List[Dict[str, Any]] = []

    for ds_name in cfg.datasets:
        x_all, y_all = load_benchmark_dataset(ds_name, local_data_dir=cfg.local_data_dir)
        for gamma in cfg.gammas:
            for eps_d in cfg.eps_batch_grid:
                for mu_out in cfg.mu_out_grid:
                    for mode in cfg.batch_contam_modes:
                        for rep_id in range(cfg.replicates):
                            payload, raw_b = run_single_safe_batch(
                                cfg,
                                ds_name,
                                gamma,
                                eps_d,
                                mu_out,
                                mode,
                                rep_id,
                                x_all,
                                y_all,
                            )
                            _save_metrics(
                                cfg,
                                payload,
                                ds_name,
                                gamma,
                                eps_d,
                                mu_out,
                                mode,
                                rep_id,
                                raw_b if cfg.save_raw else None,
                            )
                            all_rows.extend(_rows_from_payload(payload))
                            gc.collect()
                            if hasattr(jax, "clear_caches"):
                                jax.clear_caches()

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(cfg.output_dir, "summary_runs.csv"), index=False)

    num_cols = [
        "rmse_test",
        "mae_test",
        "nlpd_test",
        "n_update",
        "n_batch_retained",
        "batch_retained_fraction",
        "removed_precision",
        "removed_recall",
        "removed_f1",
        "rmse_degradation_prevented",
        "nlpd_degradation_prevented",
        "rmse_degradation_prevented_clipped",
        "nlpd_degradation_prevented_clipped",
    ]
    present = [c for c in num_cols if c in df.columns]
    gcols = [
        "dataset",
        "gamma",
        "batch_contam_mode",
        "eps_batch_design",
        "mu_out",
        "update_strategy",
    ]
    gb = df.groupby(gcols, dropna=False)[present].agg(["mean", "std"])
    gb.columns = [f"{a}_{b}" for a, b in gb.columns]
    gb.reset_index().to_csv(os.path.join(cfg.output_dir, "summary_agg.csv"), index=False)

    gamma_tar, eps_tar, mu_tar = 0.5, 0.2, 6.0
    filt = (
        np.isclose(df["gamma"], gamma_tar)
        & np.isclose(df["eps_batch_design"], eps_tar)
        & np.isclose(df["mu_out"], mu_tar)
    )
    plot_df = df[filt]
    if not plot_df.empty:
        _plot_faceted_bar(
            plot_df,
            "rmse_test",
            os.path.join(cfg.output_dir, "fig_safe_batch_rmse_gamma0p5_eps0p20_muout6.png"),
            "clean test RMSE",
        )
        _plot_faceted_bar(
            plot_df,
            "nlpd_test",
            os.path.join(cfg.output_dir, "fig_safe_batch_nlpd_gamma0p5_eps0p20_muout6.png"),
            "clean test NLPD",
        )
        _plot_degradation_prevented(
            plot_df,
            os.path.join(cfg.output_dir, "fig_safe_batch_degradation_prevented_gamma0p5_eps0p20_muout6.png"),
        )
        _plot_precision_recall(
            plot_df,
            os.path.join(cfg.output_dir, "fig_safe_batch_filtering_precision_recall_gamma0p5_eps0p20_muout6.png"),
        )

    _plot_gamma_sensitivity(df, os.path.join(cfg.output_dir, "fig_safe_batch_rmse_by_gamma_eps0p20_muout6.png"))

    print(f"Done. Output: {cfg.output_dir}")


if __name__ == "__main__":
    main()
