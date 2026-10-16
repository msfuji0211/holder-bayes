"""
Section 6.3.1 — Automated training-data cleaning with Hölder-Bayes auditing.

Fits all configured Hölder-Bayes proposals (default: DPD, ExpSatDPD, RatSatDPD when
``holder_phi`` is omitted).

Default ``--preprocessing kanamori`` matches Sec.~6.2: z-score **x** and **y** from clean train,
fixed train/test split per replicate, then homogeneous contamination in standardized-``y`` space.

Each fitted proposal runs three cleaning budgets (central / optimistic / pessimistic K from the
posterior of epsilon) ranked by log-likelihood evaluated at posterior mean parameters.

Anomaly-detection baselines (IsolationForest, LOF, One-Class SVM) remove the same central ``k``
as the reference DPD proposal. Random / Oracle cleaning run only when ``eps_true > 0``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
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
from hbayes.experiment_temperature import (TEMPERATURE_VERSION, covariance_temperature,
    response_temperature, positive_temperature, fit_metadata, nuts_summary, convergence_summary,
    configuration, begin_run, record_completion)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.exp1_regression.run_exp1_regression import NumpyEncoder  # noqa: E402
from experiments.exp1_regression.run_exp1_realdata_benchmark import (  # noqa: E402
    load_benchmark_dataset,
    parse_csv_list,
    _sanitize_dataset_name,
)
from experiments.exp1_regression.utils_exp1_regression import (  # noqa: E402
    PROPOSAL_METHOD_ORDER,
    build_realdata_contamination_settings,
    compute_dpd_outlier_prob,
    compute_loglik_posterior_mean_regression,
    contaminate_response,
    contaminate_response_scaled,
    detection_metrics_nan,
    evaluate_cleaning_set,
    evaluate_detection_metrics,
    fit_downstream_nll_and_evaluate,
    fit_regression_methods,
    kanamori_standardize_regression_data,
    make_realdata_contamination_seed,
    make_realdata_fit_seed,
    normalize_train_test_ratios,
    resolve_realdata_split_seed,
    sample_train_test_with_split_cfg,
    standardize_regression_data,
    summarize_dpd_posterior,
)

# Cleaning row labels per fitted proposal method (must stay in sync with PROPOSAL_METHOD_ORDER).
PROPOSED_CLEANING_LABEL = {
    "DPD": "ProposedDPD",
    "ExpSatDPD": "ProposedExpSatDPD",
    "RatSatDPD": "ProposedRatSatDPD",
}
K_BUDGET_SCHEMES = ("central", "optimistic", "pessimistic")
K_BUDGET_KEYS = {
    "central": "k_central",
    "optimistic": "k_optimistic",
    "pessimistic": "k_pessimistic",
}
AD_CLEANING_METHODS = (
    "IFCleaning",
    "LOFCleaning",
    "OneClassSVMCleaning",
)
_SYNTHETIC_ONLY_METHODS = ("RandomCleaning", "OracleCleaning")
_CLEANING_DIAG_NA_KEYS = (
    "removed_precision",
    "removed_recall",
    "removed_f1",
    "inlier_false_removal_rate",
)


def _proposed_cleaning_label(pname: str, scheme: str) -> str:
    base = PROPOSED_CLEANING_LABEL.get(pname, f"Proposed{pname}")
    if scheme == "central":
        return base
    return f"{base}_{scheme}"


def _indices_to_remove_loglik(log_lik: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        return np.array([], dtype=int)
    return np.argsort(np.asarray(log_lik, dtype=float))[:k]


def _indices_to_remove_lowest_scores(scores: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        return np.array([], dtype=int)
    return np.argsort(np.asarray(scores, dtype=float))[:k]


def _build_ad_feature_matrix(x_train_raw: np.ndarray, y_train_raw: np.ndarray) -> np.ndarray:
    x_np = np.asarray(x_train_raw, dtype=float)
    y_np = np.asarray(y_train_raw, dtype=float).reshape(-1, 1)
    return np.hstack([x_np, y_np])


def _standardize_ad_features(z: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(z, dtype=float))


def _ad_removed_indices(
    method: str,
    z_scaled: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    k = int(k)
    n = z_scaled.shape[0]
    if k <= 0 or n == 0:
        return np.array([], dtype=int)

    if method == "IFCleaning":
        model = IsolationForest(
            n_estimators=200,
            random_state=int(seed % (2**31)),
            n_jobs=1,
        )
        model.fit(z_scaled)
        scores = model.decision_function(z_scaled)
        return _indices_to_remove_lowest_scores(scores, k)

    if method == "LOFCleaning":
        n_neighbors = int(min(20, max(2, n - 1)))
        model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, n_jobs=1)
        model.fit(z_scaled)
        scores = model.negative_outlier_factor_
        return _indices_to_remove_lowest_scores(scores, k)

    if method == "OneClassSVMCleaning":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=min(0.5, max(k / n, 0.01)))
        model.fit(z_scaled)
        scores = model.decision_function(z_scaled)
        return _indices_to_remove_lowest_scores(scores, k)

    raise ValueError(f"Unknown AD cleaning method: {method}")


def _cleaning_plot_order(have: set, *, include_synthetic: bool) -> List[str]:
    preferred = ["NoCleaning", *AD_CLEANING_METHODS]
    for pname in PROPOSAL_METHOD_ORDER:
        for scheme in K_BUDGET_SCHEMES:
            label = _proposed_cleaning_label(pname, scheme)
            if label in have:
                preferred.append(label)
    if include_synthetic:
        preferred.extend(_SYNTHETIC_ONLY_METHODS)
    return [m for m in preferred if m in have]


def _fig_cleaning_bar_order(have: set) -> List[str]:
    include_synthetic = any(m in have for m in _SYNTHETIC_ONLY_METHODS)
    return _cleaning_plot_order(have, include_synthetic=include_synthetic)


def _pr_cleaning_order(have: set) -> List[str]:
    order = [m for m in AD_CLEANING_METHODS if m in have]
    for pname in PROPOSAL_METHOD_ORDER:
        for scheme in K_BUDGET_SCHEMES:
            label = _proposed_cleaning_label(pname, scheme)
            if label in have:
                order.append(label)
    if "RandomCleaning" in have:
        order.append("RandomCleaning")
    if "OracleCleaning" in have:
        order.append("OracleCleaning")
    return order


@dataclass
class CleaningConfig:
    n_train: int = 500
    n_test: int = 1000
    split_sizes: str = "absolute"
    train_ratio: float | None = None
    test_ratio: float | None = None
    datasets: List[str] | None = None
    gammas: List[float] | None = None
    eps_grid: List[float] | None = None
    mu_out_grid: List[float] | None = None
    replicates: int = 10
    student_t_df: float = 4.0
    base_seed: int = 42
    output_dir: str = "outputs/exp1_realdata_cleaning"
    num_warmup: int = 1000
    num_samples: int = 1000
    chains: int = 4
    eta_mode: str = "response_var"
    eta_gamma_div: float = 1.0
    eta_fixed: float = 1.0
    holder_phi: str | None = None
    holder_kappa: float | None = None
    target_accept: float = 0.8
    init_step: float = 1.0
    local_data_dir: str | None = None
    preprocessing: str = "kanamori"
    split_mode: str = "fixed"
    contam_unit: str = "std"
    save_raw: bool = False

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["cal_housing", "pumadyn32fh", "ccpp"]
        if self.gammas is None:
            self.gammas = [0.1, 0.5, 1.0]
        if self.eps_grid is None:
            self.eps_grid = [0.05, 0.2]
        if self.mu_out_grid is None:
            self.mu_out_grid = [6.0]


def _make_fit_seed(cfg: CleaningConfig, rep_id: int, eps_true: float, mu_out: float, gamma: float) -> int:
    return make_realdata_fit_seed(cfg.base_seed, rep_id, eps_true, mu_out, gamma)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _total_cleaning_runs(cfg: CleaningConfig) -> int:
    n_settings = len(
        build_realdata_contamination_settings(cfg.eps_grid, cfg.mu_out_grid)
    )
    return len(cfg.datasets) * len(cfg.gammas) * n_settings * cfg.replicates


def _clip_k_est(k: int, n_train: int) -> int:
    return int(np.clip(k, 0, int(0.5 * n_train)))


def _derive_gap_metrics(
    rmse_no: float,
    rmse_method: float,
    rmse_oracle: float,
) -> Dict[str, float]:
    denom_o = max(rmse_no - rmse_oracle, 1e-12)
    ogc = (rmse_no - rmse_method) / denom_o
    return {
        "rmse_reduction": float(rmse_no - rmse_method),
        "relative_rmse_reduction": float((rmse_no - rmse_method) / max(rmse_no, 1e-12)),
        "oracle_gap_closed": float(ogc),
        "oracle_gap_closed_clipped": float(np.clip(ogc, -1.0, 1.5)),
    }


def run_single_cleaning_run(
    cfg: CleaningConfig,
    dataset_name: str,
    gamma: float,
    eps_true: float,
    mu_out: float,
    rep_id: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fit_seed = _make_fit_seed(cfg, rep_id, eps_true, mu_out, gamma)
    split_seed = resolve_realdata_split_seed(
        cfg.base_seed, rep_id, eps_true, mu_out, split_mode=cfg.split_mode
    )
    contam_seed = make_realdata_contamination_seed(cfg.base_seed, rep_id, eps_true, mu_out)

    x_train_raw, y_train_raw_clean, x_test_raw, y_test_raw = sample_train_test_with_split_cfg(
        x_all,
        y_all,
        seed=split_seed,
        split_sizes=cfg.split_sizes,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        train_ratio=cfg.train_ratio,
        test_ratio=cfg.test_ratio,
    )
    n_train_actual = int(len(y_train_raw_clean))
    n_test_actual = int(len(y_test_raw))

    prep = str(cfg.preprocessing).strip().lower()
    downstream_scaler = None
    if prep == "kanamori":
        x_train, y_train_clean_scaled, x_test, scaler = kanamori_standardize_regression_data(
            x_train_raw,
            y_train_raw_clean,
            x_test_raw,
        )
        y_clean_np = np.asarray(y_train_clean_scaled, dtype=float)
        if eps_true > 0:
            y_train_scaled_contam, mask_outlier = contaminate_response(
                y_clean_np, eps_true, mu_out, seed=contam_seed
            )
        else:
            y_train_scaled_contam = y_clean_np.copy()
            mask_outlier = np.zeros(len(y_clean_np), dtype=bool)
        y_train = jnp.asarray(y_train_scaled_contam, dtype=jnp.float32)
        y_train_raw_contam = scaler["y_mean"] + scaler["y_std"] * y_train_scaled_contam
        downstream_scaler = scaler
    elif prep == "legacy":
        y_train_raw_contam, mask_outlier = contaminate_response_scaled(
            y_train_raw_clean,
            eps_true,
            mu_out,
            seed=contam_seed,
            contam_unit=cfg.contam_unit,
        )
        x_train, y_train, x_test, _scaler_audit = standardize_regression_data(
            x_train_raw, y_train_raw_contam, x_test_raw
        )
    else:
        raise ValueError(f"Unknown preprocessing: {cfg.preprocessing}")

    fit_res = fit_regression_methods(
        x_train,
        y_train,
        seed=fit_seed,
        chains=cfg.chains,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        gamma=gamma,
        eta_mode=cfg.eta_mode,
        eta_gamma_div=cfg.eta_gamma_div,
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
            x_train,
            y_train,
            seed=fit_seed,
        )

    ref_name = "DPD" if "DPD" in extras else fitted_proposals[0]
    k_ref = _clip_k_est(int(extras[ref_name]["k_central"]), n_train_actual)
    k_oracle = int(np.sum(mask_outlier))

    def oracle_remove_k(k: int) -> np.ndarray:
        rng_o = np.random.default_rng(fit_seed + 7)
        true_out = np.where(mask_outlier)[0]
        true_in = np.where(~mask_outlier)[0]
        rng_o.shuffle(true_out)
        rng_o.shuffle(true_in)
        if k <= len(true_out):
            return true_out[:k]
        return np.concatenate([true_out, true_in[: k - len(true_out)]])

    rng = np.random.default_rng(fit_seed + 12345)
    has_synthetic_labels = eps_true > 0
    z_ad = _build_ad_feature_matrix(x_train_raw, y_train_raw_contam)
    z_ad_scaled = _standardize_ad_features(z_ad)

    removed_sets: Dict[str, np.ndarray] = {"NoCleaning": np.array([], dtype=int)}
    removal_meta: Dict[str, Dict[str, Any]] = {
        "NoCleaning": {"k_budget": 0, "k_budget_scheme": "none", "ranking": "none"},
    }
    for ad_method in AD_CLEANING_METHODS:
        removed_sets[ad_method] = _ad_removed_indices(
            ad_method,
            z_ad_scaled,
            k_ref,
            seed=fit_seed + (sum(ord(c) for c in ad_method) % 10_000),
        )
        removal_meta[ad_method] = {
            "k_budget": k_ref,
            "k_budget_scheme": "central",
            "ranking": ad_method.replace("Cleaning", "").lower(),
        }

    for pname in fitted_proposals:
        ex = extras[pname]
        log_lik = compute_loglik_posterior_mean_regression(
            fit_res[pname]["samples"], x_train, y_train
        )
        for scheme in K_BUDGET_SCHEMES:
            label = _proposed_cleaning_label(pname, scheme)
            k_budget_p = _clip_k_est(int(ex[K_BUDGET_KEYS[scheme]]), n_train_actual)
            removed_sets[label] = _indices_to_remove_loglik(log_lik, k_budget_p)
            removal_meta[label] = {
                "k_budget": k_budget_p,
                "k_budget_scheme": scheme,
                "ranking": "loglik_posterior_mean",
                "proposal_method": pname,
            }

    if has_synthetic_labels:
        removed_sets["RandomCleaning"] = (
            rng.choice(n_train_actual, size=k_ref, replace=False) if k_ref > 0 else np.array([], dtype=int)
        )
        removal_meta["RandomCleaning"] = {
            "k_budget": k_ref,
            "k_budget_scheme": "central",
            "ranking": "random",
        }
        removed_sets["OracleCleaning"] = oracle_remove_k(k_ref) if k_ref > 0 else np.array([], dtype=int)
        removal_meta["OracleCleaning"] = {
            "k_budget": k_ref,
            "k_budget_scheme": "central",
            "ranking": "oracle",
        }

    method_order = ["NoCleaning", *AD_CLEANING_METHODS]
    method_order.extend(
        _proposed_cleaning_label(p, scheme)
        for p in fitted_proposals
        for scheme in K_BUDGET_SCHEMES
    )
    if has_synthetic_labels:
        method_order.extend(_SYNTHETIC_ONLY_METHODS)

    downstream_seed = fit_seed + 9000
    cleaning_out: Dict[str, Any] = {}

    for cm in method_order:
        rid = removed_sets[cm]
        keep = np.ones(n_train_actual, dtype=bool)
        if rid.size > 0:
            keep[rid] = False
        x_tr = x_train_raw[keep]
        y_tr = y_train_raw_contam[keep]

        import hashlib
        # Identical retained datasets share a downstream fit, irrespective of
        # which ranking/budget label selected them. NoCleaning shares the
        # benchmark's NLL posterior with identical inputs, priors and settings.
        seed_cm = (fit_seed + 101 if cm == "NoCleaning" else
                   (downstream_seed + int.from_bytes(hashlib.sha256(
                       np.sort(rid).astype("<i8").tobytes()).digest()[:4], "little")) % (2**32))
        dmetrics = fit_downstream_nll_and_evaluate(
            x_tr,
            y_tr,
            x_test_raw,
            y_test_raw,
            seed=seed_cm,
            chains=cfg.chains,
            num_warmup=cfg.num_warmup,
            num_samples=cfg.num_samples,
            target_accept=cfg.target_accept,
            init_step=cfg.init_step,
            scaler=downstream_scaler,
            fit_role="NLL" if cm == "NoCleaning" else "DownstreamNLL",
        )
        diag = evaluate_cleaning_set(rid, mask_outlier)
        row = {**dmetrics, **diag, **removal_meta.get(cm, {}), "downstream_seed": seed_cm}
        if not has_synthetic_labels:
            for key in _CLEANING_DIAG_NA_KEYS:
                row[key] = np.nan
            row.update(
                {
                    "rmse_reduction": np.nan,
                    "relative_rmse_reduction": np.nan,
                    "oracle_gap_closed": np.nan,
                    "oracle_gap_closed_clipped": np.nan,
                }
            )
        cleaning_out[cm] = row

    rmse_no = cleaning_out["NoCleaning"]["rmse_test"]
    if has_synthetic_labels and "OracleCleaning" in cleaning_out:
        rmse_oracle = cleaning_out["OracleCleaning"]["rmse_test"]
        for cm in method_order:
            cleaning_out[cm].update(
                _derive_gap_metrics(rmse_no, cleaning_out[cm]["rmse_test"], rmse_oracle)
            )

    audits: Dict[str, Dict[str, Any]] = {}
    for pname in fitted_proposals:
        ex = extras[pname]
        log_lik = compute_loglik_posterior_mean_regression(
            fit_res[pname]["samples"], x_train, y_train
        )
        k_budgets = {
            "k_central": _clip_k_est(int(ex["k_central"]), n_train_actual),
            "k_optimistic": _clip_k_est(int(ex["k_optimistic"]), n_train_actual),
            "k_pessimistic": _clip_k_est(int(ex["k_pessimistic"]), n_train_actual),
        }
        kb = k_budgets["k_central"]
        if has_synthetic_labels:
            det = evaluate_detection_metrics(
                ex["p_out"],
                mask_outlier.astype(int),
                k_oracle=k_oracle,
                k_est=kb,
                is_prob=True,
                log_lik=log_lik,
                k_budgets=k_budgets,
            )
        else:
            det = detection_metrics_nan(is_prob=True)
        audits[pname] = {
            **fit_metadata(fit_res[pname]),
            **summarize_dpd_posterior(fit_res[pname]["samples"], eps_true=eps_true),
            **det,
            "k_est_cleaning": kb,
            "k_est_cleaning_optimistic": k_budgets["k_optimistic"],
            "k_est_cleaning_pessimistic": k_budgets["k_pessimistic"],
        }

    out: Dict[str, Any] = {
        "temperature_version": TEMPERATURE_VERSION,
        "run_config": configuration(cfg, __file__),
        "eta_mode": cfg.eta_mode,
        "eta_gamma_div": cfg.eta_gamma_div,
        "dataset": dataset_name,
        "gamma": gamma,
        "eps_true": eps_true,
        "mu_out": mu_out,
        "replicate": rep_id,
        "n_train": n_train_actual,
        "n_test": n_test_actual,
        "split_sizes": str(cfg.split_sizes),
        "train_ratio": cfg.train_ratio,
        "test_ratio": cfg.test_ratio,
        "preprocessing": prep,
        "split_mode": str(cfg.split_mode),
        "contam_unit": (cfg.contam_unit if prep == "legacy" else None),
        "k_oracle": k_oracle,
        "has_synthetic_labels": has_synthetic_labels,
        "k_est_cleaning_reference": k_ref,
        "proposal_method_primary": ref_name,
        "proposal_methods_fitted": fitted_proposals,
        "audits": audits,
        "cleaning": cleaning_out,
    }

    raw_arrays = {
        "x_train_raw": x_train_raw,
        "y_train_raw_clean": y_train_raw_clean,
        "y_train_raw_contam": y_train_raw_contam,
        "mask_outlier": mask_outlier,
        "x_test_raw": x_test_raw,
        "y_test_raw": y_test_raw,
    }

    for cm, rid in removed_sets.items():
        raw_arrays[f"removed_indices_{cm}"] = np.asarray(rid, dtype=np.int64)
    for method, result in fit_res.items():
        raw_arrays[f"audit_samples_{method}"] = np.asarray(result["samples"])
        raw_arrays[f"audit_p_out_{method}"] = np.asarray(extras[method]["p_out"])
    return out, raw_arrays


def _save_run_json(
    cfg: CleaningConfig,
    payload: Dict[str, Any],
    dataset_name: str,
    gamma: float,
    eps_true: float,
    mu_out: float,
    rep_id: int,
    raw_bundle: Dict[str, Any] | None = None,
) -> str:
    run_dir = os.path.join(
        cfg.output_dir,
        f"dataset_{dataset_name}",
        f"gamma_{gamma:.3f}",
        f"eps_{eps_true:.2f}",
        f"muout_{mu_out:.1f}",
        f"rep_{rep_id:03d}",
    )
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    if raw_bundle is not None and cfg.save_raw:
        np.savez_compressed(
            os.path.join(run_dir, "raw.npz"),
            **{k: np.asarray(v) for k, v in raw_bundle.items()},
        )
    return path


def _collect_rows(payload: Dict[str, Any], cfg: CleaningConfig) -> List[Dict[str, Any]]:
    rows = []
    base = {
        "dataset": payload["dataset"],
        "gamma": payload["gamma"],
        "eps_true": payload["eps_true"],
        "mu_out": payload["mu_out"],
        "replicate": payload["replicate"],
        "n_train": payload["n_train"],
        "n_test": payload["n_test"],
        "k_oracle": payload["k_oracle"],
        "has_synthetic_labels": payload.get("has_synthetic_labels", payload.get("eps_true", 0) > 0),
        "preprocessing": payload.get("preprocessing"),
        "contam_unit": payload.get("contam_unit"),
        "k_est_cleaning_reference": payload.get("k_est_cleaning_reference"),
        "proposal_method_primary": payload.get("proposal_method_primary"),
    }
    for cm, d in payload["cleaning"].items():
        r = {**base, "cleaning_method": cm}
        for k, v in d.items():
            r[k] = v
        rows.append(r)
    return rows


def _plot_main_figures(summary_runs: pd.DataFrame, out_dir: str) -> None:
    gamma_tar = 0.5
    eps_tar = 0.2
    mu_tar = 6.0
    sub = summary_runs[
        np.isclose(summary_runs["gamma"], gamma_tar)
        & np.isclose(summary_runs["eps_true"], eps_tar)
        & np.isclose(summary_runs["mu_out"], mu_tar)
    ]
    if sub.empty:
        return

    datasets = sorted(sub["dataset"].unique())
    n_ds = len(datasets)
    fig1, axes1 = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), squeeze=False)
    for ax, ds in zip(axes1[0], datasets):
        sdf = sub[sub["dataset"] == ds]
        means = []
        order = _fig_cleaning_bar_order(set(sdf["cleaning_method"].unique()))
        for m in order:
            sl = sdf[sdf["cleaning_method"] == m]["rmse_test"]
            means.append(sl.mean() if len(sl) else np.nan)
        x = np.arange(len(order))
        ax.bar(x, means, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylabel("clean test RMSE")
        ax.set_title(ds)
    fig1.tight_layout()
    fig1.savefig(
        os.path.join(out_dir, "fig_cleaning_rmse_gamma0p5_eps0p20_muout6.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), squeeze=False)
    for ax, ds in zip(axes2[0], datasets):
        sdf = sub[sub["dataset"] == ds]
        means = []
        order = _fig_cleaning_bar_order(set(sdf["cleaning_method"].unique()))
        for m in order:
            sl = sdf[sdf["cleaning_method"] == m]["oracle_gap_closed_clipped"]
            means.append(sl.mean() if len(sl) else np.nan)
        x = np.arange(len(order))
        ax.bar(x, means, color="darkorange", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylabel("oracle gap closed (clipped)")
        ax.set_title(ds)
    fig2.tight_layout()
    fig2.savefig(
        os.path.join(out_dir, "fig_cleaning_oracle_gap_closed_gamma0p5_eps0p20_muout6.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig2)

    fig3, axes3 = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4), squeeze=False)
    for ax, ds in zip(axes3[0], datasets):
        sdf = sub[sub["dataset"] == ds]
        pr_order = _pr_cleaning_order(set(sdf["cleaning_method"].unique()))
        prec = []
        rec = []
        labels = []
        for m in pr_order:
            if m not in sdf["cleaning_method"].values:
                continue
            sl = sdf[sdf["cleaning_method"] == m]
            prec.append(sl["removed_precision"].mean())
            rec.append(sl["removed_recall"].mean())
            labels.append(m)
        if not labels:
            continue
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, prec, width=w, label="precision", color="#4c78a8")
        ax.bar(x + w / 2, rec, width=w, label="recall", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title(ds)
    fig3.tight_layout()
    fig3.savefig(
        os.path.join(out_dir, "fig_cleaning_precision_recall_gamma0p5_eps0p20_muout6.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig3)


def _plot_gamma_sensitivity(summary_runs: pd.DataFrame, out_dir: str) -> None:
    eps_tar = 0.2
    mu_tar = 6.0
    sub = summary_runs[np.isclose(summary_runs["eps_true"], eps_tar) & np.isclose(summary_runs["mu_out"], mu_tar)]
    if sub.empty:
        return
    datasets = sorted(sub["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sdf = sub[sub["dataset"] == ds]
        gammas = sorted(sdf["gamma"].unique())
        line_order = _fig_cleaning_bar_order(set(sdf["cleaning_method"].unique()))
        for m in line_order:
            ys = []
            for g in gammas:
                sl = sdf[np.isclose(sdf["gamma"], g) & (sdf["cleaning_method"] == m)]["rmse_test"]
                ys.append(sl.mean() if len(sl) else np.nan)
            ax.plot(gammas, ys, marker="o", label=m, linewidth=1.5)
        ax.set_xlabel("gamma")
        ax.set_ylabel("mean clean test RMSE")
        ax.set_title(ds)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "fig_cleaning_rmse_by_gamma_eps0p20_muout6.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Section 6.3.1 real-data cleaning experiment")
    p.add_argument("--n_train", type=int, help="Train size when --split_sizes absolute (default 500)")
    p.add_argument("--n_test", type=int, help="Test size when --split_sizes absolute (default 1000)")
    p.add_argument(
        "--split_sizes",
        type=str,
        choices=["absolute", "ratio"],
        default="absolute",
        help="absolute: fixed n_train/n_test subsample; ratio: --train_ratio of full dataset",
    )
    p.add_argument(
        "--train_ratio",
        type=float,
        help="Train fraction in (0,1) for ratio split (e.g. 0.8; test gets the rest)",
    )
    p.add_argument(
        "--test_ratio",
        type=float,
        help="Optional test fraction; if omitted, 1 - train_ratio is used",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="cal_housing,pumadyn32fh,ccpp",
        help="e.g. cal_housing,ccpp,concrete,kin8nm (see run_exp1_realdata_benchmark.DATASET_ALIASES)",
    )
    p.add_argument("--gammas", type=str, default="0.1,0.5,1.0")
    p.add_argument("--eps_grid", type=str, default="0.05,0.2")
    p.add_argument("--mu_out_grid", type=str, default="6")
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--student_t_df", type=float, default=4.0)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--num_warmup", type=int, default=1000)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--eta_mode", type=str, default="response_var")
    p.add_argument("--eta_fixed", type=float, default=1.0)
    p.add_argument("--eta_gamma_div", type=float, default=1.0)
    p.add_argument(
        "--holder_phi",
        type=str,
        default=None,
        help="Single Hölder proposal (e.g. dpd). Omit for all three: DPD, ExpSatDPD, RatSatDPD.",
    )
    p.add_argument("--holder_kappa", type=float, default=None)
    p.add_argument("--target_accept", type=float, default=0.8)
    p.add_argument("--init_step", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, default="outputs/exp1_realdata_cleaning")
    p.add_argument("--local_data_dir", type=str, default=None)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument(
        "--preprocessing",
        type=str,
        default="kanamori",
        choices=["kanamori", "legacy"],
        help="kanamori: z-score x and y from clean train, then N(mu_out,1) on scaled y. "
        "legacy: contaminate raw y then train-set standardization.",
    )
    p.add_argument(
        "--split_mode",
        type=str,
        default="fixed",
        choices=["fixed", "per_setting"],
        help="fixed: same train/test split per replicate across eps/mu.",
    )
    p.add_argument(
        "--contam_unit",
        type=str,
        default="std",
        choices=["std", "raw"],
        help="Only used when --preprocessing legacy.",
    )
    p.add_argument("--save_raw", action="store_true")

    args = p.parse_args()

    cfg = CleaningConfig(
        datasets=[_sanitize_dataset_name(v) for v in parse_csv_list(args.datasets, str)],
        gammas=parse_csv_list(args.gammas, float),
        eps_grid=parse_csv_list(args.eps_grid, float),
        mu_out_grid=parse_csv_list(args.mu_out_grid, float),
        replicates=args.replicates,
        student_t_df=args.student_t_df,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        chains=args.chains,
        eta_mode=args.eta_mode,
        eta_fixed=args.eta_fixed,
        eta_gamma_div=args.eta_gamma_div,
        holder_phi=(None if args.holder_phi is None else str(args.holder_phi).strip() or None),
        holder_kappa=args.holder_kappa,
        target_accept=args.target_accept,
        init_step=args.init_step,
        local_data_dir=args.local_data_dir,
        preprocessing=args.preprocessing,
        split_mode=args.split_mode,
        contam_unit=args.contam_unit,
        save_raw=args.save_raw,
    )
    cfg.split_sizes = args.split_sizes
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_test is not None:
        cfg.n_test = args.n_test
    if args.train_ratio is not None:
        cfg.train_ratio = args.train_ratio
    if args.test_ratio is not None:
        cfg.test_ratio = args.test_ratio
    if args.train_ratio is not None or args.test_ratio is not None:
        cfg.split_sizes = "ratio"
        try:
            cfg.train_ratio, cfg.test_ratio = normalize_train_test_ratios(
                cfg.train_ratio, cfg.test_ratio
            )
        except ValueError as exc:
            p.error(str(exc))
    elif cfg.split_sizes == "ratio":
        p.error("--train_ratio is required when --split_sizes ratio")

    begin_run(cfg, __file__)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    total_runs = _total_cleaning_runs(cfg)
    _log("=== exp1 real-data cleaning (Sec. 6.3.1) ===")
    _log(f"output_dir={cfg.output_dir}")
    if cfg.split_sizes == "ratio":
        _log(
            f"split=ratio train_ratio={cfg.train_ratio} test_ratio={cfg.test_ratio} "
            "(actual n_train/n_test depend on dataset size)"
        )
    else:
        _log(f"split=absolute n_train={cfg.n_train} n_test={cfg.n_test}")
    _log(f"preprocessing={cfg.preprocessing} split_mode={cfg.split_mode}")
    _log(
        f"MCMC: chains={cfg.chains} warmup={cfg.num_warmup} samples={cfg.num_samples} "
        f"target_accept={cfg.target_accept} init_step={cfg.init_step}"
    )
    _log(
        f"grid: datasets={cfg.datasets} gammas={cfg.gammas} "
        f"eps={cfg.eps_grid} mu_out={cfg.mu_out_grid} replicates={cfg.replicates}"
    )
    _log(f"planned runs: {total_runs}")
    start_time = time.time()
    run_idx = 0
    n_ok = 0
    n_fail = 0
    failures = []

    all_rows: List[Dict[str, Any]] = []

    for ds_name in cfg.datasets:
        _log(f"=== Loading dataset: {ds_name} ===")
        x_all, y_all = load_benchmark_dataset(ds_name, local_data_dir=cfg.local_data_dir)
        _log(f"Loaded {ds_name}: n={x_all.shape[0]}, d={x_all.shape[1]}")
        for gamma in cfg.gammas:
            _log(f"=== Dataset={ds_name}, gamma={gamma:.3f} ===")
            setting_list = build_realdata_contamination_settings(
                cfg.eps_grid, cfg.mu_out_grid
            )
            for eps_true, mu_out in setting_list:
                _log(f"  > eps={eps_true:.2f}, mu_out={mu_out:.1f}")
                for rep_id in range(cfg.replicates):
                    run_idx += 1
                    _log(
                        f"    [{run_idx}/{total_runs}] rep={rep_id:03d} — running "
                        f"(dataset={ds_name}, gamma={gamma:.3f}, eps={eps_true:.2f}) ..."
                    )
                    t_run = time.time()
                    try:
                        payload, raw_bundle = run_single_cleaning_run(
                            cfg,
                            ds_name,
                            gamma,
                            eps_true,
                            mu_out,
                            rep_id,
                            x_all,
                            y_all,
                        )
                        metrics_path = _save_run_json(
                            cfg,
                            payload,
                            ds_name,
                            gamma,
                            eps_true,
                            mu_out,
                            rep_id,
                            raw_bundle=raw_bundle if cfg.save_raw else None,
                        )
                        all_rows.extend(_collect_rows(payload, cfg))
                        n_ok += 1
                        dt_run = time.time() - t_run
                        clean = payload.get("cleaning", {})
                        rmse_no = float(clean.get("NoCleaning", {}).get("rmse_test", float("nan")))
                        rmse_dpd = float(clean.get("ProposedDPD", {}).get("rmse_test", float("nan")))
                        rmse_dpd_pess = float(
                            clean.get("ProposedDPD_pessimistic", {}).get("rmse_test", float("nan"))
                        )
                        k_ref = payload.get("k_est_cleaning_reference", float("nan"))
                        _log(
                            f"    [{run_idx}/{total_runs}] done in {dt_run:.0f}s — "
                            f"NoClean RMSE={rmse_no:.4f} ProposedDPD={rmse_dpd:.4f} "
                            f"ProposedDPD_pess={rmse_dpd_pess:.4f} k_ref={k_ref} — saved {metrics_path}"
                        )
                    except Exception as exc:
                        n_fail += 1
                        failures.append({"dataset": ds_name, "gamma": gamma, "eps_true": eps_true, "mu_out": mu_out, "replicate": rep_id, "error": str(exc)})
                        dt_run = time.time() - t_run
                        _log(
                            f"    [{run_idx}/{total_runs}] FAILED after {dt_run:.0f}s — "
                            f"dataset={ds_name}, gamma={gamma:.3f}, eps={eps_true:.2f}, "
                            f"mu_out={mu_out:.1f}, rep={rep_id:03d}: {exc}"
                        )
                        import traceback

                        traceback.print_exc()
                    gc.collect()
                    if hasattr(jax, "clear_caches"):
                        jax.clear_caches()

    record_completion(cfg, total_runs, n_ok, failures)
    if not all_rows:
        raise RuntimeError(
            f"No successful runs ({n_fail} failed, {n_ok} ok). Check logs and hyperparameters."
        )

    df_runs = pd.DataFrame(all_rows)
    df_runs.to_csv(os.path.join(cfg.output_dir, "summary_runs.csv"), index=False)

    agg_cols = [
        "rmse_test",
        "mae_test",
        "nlpd_test",
        "nlpd_test_mc",
        "k_removed",
        "retained_fraction",
        "removed_precision",
        "removed_recall",
        "removed_f1",
        "inlier_false_removal_rate",
        "rmse_reduction",
        "relative_rmse_reduction",
        "oracle_gap_closed",
        "oracle_gap_closed_clipped",
    ]
    present = [c for c in agg_cols if c in df_runs.columns]
    gcols = ["dataset", "gamma", "eps_true", "mu_out", "cleaning_method"]
    gb = df_runs.groupby(gcols, dropna=False)[present].agg(["mean", "std"])
    gb.columns = [f"{a}_{b}" for a, b in gb.columns]
    df_agg = gb.reset_index()
    df_agg.to_csv(os.path.join(cfg.output_dir, "summary_agg.csv"), index=False)

    _plot_main_figures(df_runs, cfg.output_dir)
    _plot_gamma_sensitivity(df_runs, cfg.output_dir)

    elapsed = time.time() - start_time
    _log(f"Saved: {os.path.join(cfg.output_dir, 'summary_runs.csv')}")
    _log(f"Saved: {os.path.join(cfg.output_dir, 'summary_agg.csv')}")
    _log(f"Completed {n_ok}/{total_runs} runs ({n_fail} failed). Total time: {elapsed:.1f}s")
    _log(f"Wrote summaries and figures under {cfg.output_dir}")


if __name__ == "__main__":
    main()
