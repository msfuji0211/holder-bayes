import argparse
import gc
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

if "MALLOC_TRIM_THRESHOLD_" not in os.environ:
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
if "MALLOC_ARENA_MAX" not in os.environ:
    os.environ["MALLOC_ARENA_MAX"] = "2"
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from hbayes.experiment_temperature import (TEMPERATURE_VERSION, covariance_temperature,
    response_temperature, positive_temperature, fit_metadata, nuts_summary, convergence_summary,
    configuration, begin_run, record_completion)
from sklearn.datasets import fetch_california_housing, fetch_openml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.exp1_regression.run_exp1_regression import (  # noqa: E402
    BASELINE_METHODS,
    METHOD_JSON_KEYS,
    NumpyEncoder,
    get_all_methods,
    get_primary_proposal_method,
    get_proposal_methods,
)
from experiments.exp1_regression.utils_exp1_regression import (  # noqa: E402
    build_realdata_contamination_settings,
    compute_regression_mcmc_convergence_summaries,
    compute_dpd_outlier_prob,
    compute_loglik_posterior_mean_regression,
    contaminate_response,
    detection_metrics_nan,
    evaluate_detection_metrics,
    evaluate_prediction_metrics,
    fit_regression_methods,
    kanamori_standardize_regression_data,
    make_realdata_contamination_seed,
    make_realdata_fit_seed,
    normalize_train_test_ratios,
    resolve_realdata_split_seed,
    resolve_train_test_sizes,
    sample_train_test_with_split_cfg,
    score_baseline,
    standardize_regression_data,
    summarize_dpd_posterior,
    summarize_posterior_prediction,
)


DATASET_ALIASES = {
    "cal_housing": "cal_housing",
    "cal-housing": "cal_housing",
    "calhousing": "cal_housing",
    "pumadyn32fh": "pumadyn32fh",
    "pumadyn-32fh": "pumadyn32fh",
    "puma32h": "pumadyn32fh",
    "ccpp": "ccpp",
    "combined_cycle_power_plant": "ccpp",
    "combined-cycle-power-plant": "ccpp",
    "concrete": "concrete",
    "concrete_compressive_strength": "concrete",
    "concrete-compressive-strength": "concrete",
    "kin8nm": "kin8nm",
    "kin8-nm": "kin8nm",
    "kin8_nm": "kin8nm",
}

# UCI ML Repository (ID 294): Combined Cycle Power Plant
_CCPP_UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
_CCPP_XLSX_IN_ZIP = "CCPP/Folds5x2_pp.xlsx"

# UCI ML Repository (ID 165): Concrete Compressive Strength
_CONCRETE_UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
_CONCRETE_UCI_XLS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
)


@dataclass
class RealDataConfig:
    n_train: int = 500
    n_test: int = 1000
    # ``absolute``: use ``n_train`` / ``n_test``; ``ratio``: use ``train_ratio`` / ``test_ratio``.
    split_sizes: str = "absolute"
    train_ratio: float | None = None
    test_ratio: float | None = None
    datasets: List[str] = None
    gammas: List[float] = None
    eps_grid: List[float] = None
    mu_out_grid: List[float] = None
    replicates: int = 10
    student_t_df: float = 4.0
    base_seed: int = 42
    output_dir: str = "outputs/exp1_realdata_benchmark"
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
    # ``kanamori``: z-score x and y (clean train), then contaminate in standardized y-space.
    # ``legacy``: contaminate raw y, then train-set standardize.
    preprocessing: str = "kanamori"
    # ``fixed``: same train/test split per rep across eps/mu; ``per_setting``: legacy combined seed.
    split_mode: str = "fixed"
    save_samples: bool = False

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["cal_housing", "pumadyn32fh", "ccpp"]
        if self.gammas is None:
            self.gammas = [0.5]
        if self.eps_grid is None:
            self.eps_grid = [0.05, 0.2]
        if self.mu_out_grid is None:
            self.mu_out_grid = [3.0, 6.0]


def parse_csv_list(value: str, cast):
    tokens = value.replace(",", " ").split()
    return [cast(v) for v in tokens]


def _sanitize_dataset_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset alias: {name}")
    return DATASET_ALIASES[key]


def _drop_nan_rows(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_num = np.asarray(x, dtype=float)
    y_num = np.asarray(y, dtype=float)
    mask = np.isfinite(y_num) & np.all(np.isfinite(x_num), axis=1)
    return x_num[mask], y_num[mask]


def _load_dataset_local(dataset_name: str, local_data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(local_data_dir)
    candidate_paths = [
        data_dir / f"{dataset_name}.csv",
        data_dir / f"{dataset_name}.tsv",
    ]
    if dataset_name == "ccpp":
        candidate_paths.extend(
            [
                data_dir / "Folds5x2_pp.xlsx",
                data_dir / "folds5x2_pp.xlsx",
            ]
        )
    existing = [p for p in candidate_paths if p.exists()]
    if not existing:
        raise FileNotFoundError(f"Local file not found for {dataset_name} in {local_data_dir}")
    path = existing[0]
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    if "target" in df.columns:
        y = df["target"].to_numpy()
        x = df.drop(columns=["target"]).to_numpy()
    else:
        y = df.iloc[:, -1].to_numpy()
        x = df.iloc[:, :-1].to_numpy()
    x, y = _drop_nan_rows(x, y)
    return x, y


def _load_ccpp_from_uci(cache_dir: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load UCI Combined Cycle Power Plant (9568 × 4 → PE). Cached zip under ``~/.cache/hbayes_datasets``."""
    base = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "hbayes_datasets")
    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, "uci_294_combined_cycle_power_plant.zip")
    if not os.path.isfile(zip_path):
        urllib.request.urlretrieve(_CCPP_UCI_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(_CCPP_XLSX_IN_ZIP) as fp:
            df = pd.read_excel(fp)
    feature_cols = ["AT", "V", "AP", "RH"]
    if not all(c in df.columns for c in feature_cols) or "PE" not in df.columns:
        raise ValueError(f"Unexpected CCPP columns: {list(df.columns)}")
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["PE"].to_numpy(dtype=float)
    return _drop_nan_rows(x, y)


def _fetch_openml_regression(
    candidates: list[dict],
    *,
    min_features: int = 1,
    dataset_label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Try OpenML name/version or data_id specs until one loads."""
    last_exc: Exception | None = None
    for spec in candidates:
        try:
            if "data_id" in spec:
                ds = fetch_openml(data_id=int(spec["data_id"]), parser="auto", as_frame=False)
            else:
                ds = fetch_openml(
                    name=str(spec["name"]),
                    version=int(spec.get("version", 1)),
                    parser="auto",
                    as_frame=False,
                )
            x, y = _drop_nan_rows(ds.data, ds.target)
            if x.shape[1] >= min_features:
                return x, y
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"{dataset_label} could not be downloaded from OpenML. "
        "Use --local_data_dir with a prepared CSV/TSV (last column = target)."
    ) from last_exc


def _read_concrete_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if df.shape[1] < 9:
        raise ValueError(f"Expected >= 9 columns for concrete, got {df.shape[1]}")
    y = df.iloc[:, -1].to_numpy(dtype=float)
    x = df.iloc[:, :-1].to_numpy(dtype=float)
    return _drop_nan_rows(x, y)


def _download_with_retries(url: str, dst_path: str, *, attempts: int = 3, sleep_s: float = 2.0) -> None:
    """Download a URL to disk with small backoff (handles transient UCI 503s)."""
    last_exc: Exception | None = None
    for k in range(attempts):
        try:
            urllib.request.urlretrieve(url, dst_path)
            return
        except Exception as exc:
            last_exc = exc
            if k < attempts - 1:
                time.sleep(sleep_s * (k + 1))
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_exc


def _load_concrete_from_uci(cache_dir: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load UCI Concrete Compressive Strength (~1030 × 8). Cached under ``~/.cache/hbayes_datasets``."""
    base = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "hbayes_datasets")
    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, "uci_165_concrete_compressive_strength.zip")
    xls_path = os.path.join(base, "Concrete_Data.xls")

    if os.path.isfile(xls_path):
        df = pd.read_excel(xls_path)
        return _read_concrete_dataframe(df)

    if not os.path.isfile(zip_path):
        zip_exc: Exception | None = None
        try:
            _download_with_retries(_CONCRETE_UCI_ZIP_URL, zip_path, attempts=3)
        except Exception as exc:
            zip_exc = exc
            try:
                _download_with_retries(_CONCRETE_UCI_XLS_URL, xls_path, attempts=3)
                df = pd.read_excel(xls_path)
                return _read_concrete_dataframe(df)
            except Exception as exc2:
                raise RuntimeError(
                    "Concrete dataset download failed from UCI (both zip and xls). "
                    "Please use --local_data_dir with concrete.csv/tsv."
                ) from (exc2 or zip_exc)

    with zipfile.ZipFile(zip_path, "r") as zf:
        xls_names = [n for n in zf.namelist() if n.lower().endswith(".xls") or n.lower().endswith(".xlsx")]
        if not xls_names:
            raise ValueError(f"No spreadsheet found in concrete UCI zip: {zf.namelist()}")
        with zf.open(xls_names[0]) as fp:
            df = pd.read_excel(fp)
    return _read_concrete_dataframe(df)


def _load_concrete() -> Tuple[np.ndarray, np.ndarray]:
    """UCI / OpenML Concrete Compressive Strength (~1030 × 8 → strength)."""
    try:
        return _load_concrete_from_uci()
    except Exception:
        # OpenML fallback for environments where UCI is rate-limited / unavailable.
        return _fetch_openml_regression(
            [{"data_id": 4353}, {"name": "Concrete_Compressive_Strength", "version": 1}],
            min_features=8,
            dataset_label="concrete",
        )


def _load_kin8nm_from_openml() -> Tuple[np.ndarray, np.ndarray]:
    """OpenML kin8nm robot arm simulation (~8192 × 8)."""
    return _fetch_openml_regression(
        [
            {"data_id": 189},
            {"name": "kin8nm", "version": 1},
            {"name": "Kin8nm", "version": 1},
        ],
        min_features=8,
        dataset_label="kin8nm",
    )


def load_benchmark_dataset(dataset_name: str, local_data_dir: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    name = _sanitize_dataset_name(dataset_name)
    if local_data_dir is not None:
        return _load_dataset_local(name, local_data_dir)

    if name == "cal_housing":
        try:
            ds = fetch_openml(name="cal_housing", version=1, as_frame=False, parser="auto")
            x, y = _drop_nan_rows(ds.data, ds.target)
            return x, y
        except Exception:
            ds = fetch_california_housing()
            x, y = _drop_nan_rows(ds.data, ds.target)
            return x, y

    if name == "pumadyn32fh":
        candidates = [
            ("pumadyn32fh", 1),
            ("pumadyn-32fh", 1),
            ("puma32H", 1),
            ("puma8NH", 1),
        ]
        last_exc = None
        for cand_name, cand_ver in candidates:
            try:
                ds = fetch_openml(name=cand_name, version=cand_ver, as_frame=False, parser="auto")
                x, y = _drop_nan_rows(ds.data, ds.target)
                if x.shape[1] >= 8:
                    return x, y
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            "pumadyn32fh could not be downloaded from OpenML. "
            "Use --local_data_dir with a prepared CSV/TSV."
        ) from last_exc

    if name == "ccpp":
        return _load_ccpp_from_uci()

    if name == "concrete":
        return _load_concrete()

    if name == "kin8nm":
        return _load_kin8nm_from_openml()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def run_single_setting(
    cfg: RealDataConfig,
    dataset_name: str,
    gamma: float,
    eps_true: float,
    mu_out: float,
    rep_id: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
) -> Dict[str, Any]:
    split_seed = resolve_realdata_split_seed(
        cfg.base_seed, rep_id, eps_true, mu_out, split_mode=cfg.split_mode
    )
    contam_seed = make_realdata_contamination_seed(cfg.base_seed, rep_id, eps_true, mu_out)
    fit_seed = make_realdata_fit_seed(cfg.base_seed, rep_id, eps_true, mu_out, gamma)
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
    if prep == "kanamori":
        x_train, y_train_clean_scaled, x_test, scaler = kanamori_standardize_regression_data(
            x_train_raw,
            y_train_raw_clean,
            x_test_raw,
        )
        y_clean_np = np.asarray(y_train_clean_scaled, dtype=float)
        if eps_true > 0:
            y_train_np, mask_outlier = contaminate_response(
                y_clean_np, eps=eps_true, mu_out=mu_out, seed=contam_seed
            )
        else:
            y_train_np = y_clean_np.copy()
            mask_outlier = np.zeros(len(y_train_np), dtype=bool)
        y_train = jnp.asarray(y_train_np, dtype=jnp.float32)
    elif prep == "legacy":
        y_train_raw, mask_outlier = contaminate_response(
            y_train_raw_clean, eps=eps_true, mu_out=mu_out, seed=contam_seed
        )
        x_train, y_train, x_test, scaler = standardize_regression_data(x_train_raw, y_train_raw, x_test_raw)
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
    )

    proposal_methods = get_proposal_methods(cfg)
    primary_proposal = get_primary_proposal_method(cfg)
    all_methods = get_all_methods(cfg)
    k_oracle = int(mask_outlier.sum())

    proposal_extras = {}
    for method in proposal_methods:
        proposal_extras[method] = compute_dpd_outlier_prob(
            fit_res[method]["samples"], x_train, y_train, seed=fit_seed
        )
    k_est = int(proposal_extras[primary_proposal]["k_mean"])
    primary_k_budgets = {
        key: int(proposal_extras[primary_proposal][key])
        for key in ("k_central", "k_optimistic", "k_pessimistic")
    }

    results: Dict[str, Dict[str, Any]] = {}
    rank_arrays = {}
    diag_detail: Dict[str, Any] = {}
    for method in all_methods:
        pred = summarize_posterior_prediction(fit_res[method]["samples"], x_test, scaler)
        metrics = evaluate_prediction_metrics(
            pred["y_pred"],
            y_test_raw,
            sigma_hat=pred["sigma_hat"],
            samples=fit_res[method]["samples"],
            x_test_design=x_test,
            scaler=scaler,
            joint_with_rho=(method in proposal_methods),
        )

        conv_scalars, conv_detail = compute_regression_mcmc_convergence_summaries(
            fit_res[method]["samples"],
            n_chains=cfg.chains,
            joint_with_rho=(method in proposal_methods),
            n_design_cols=int(x_train.shape[1]),
        )
        metrics.update(conv_scalars)

        if method in proposal_methods:
            metrics.update(summarize_dpd_posterior(fit_res[method]["samples"], eps_true=eps_true))
            metrics["eta"] = fit_res[method]["eta"]
            metrics["cov_det"] = fit_res[method]["cov_det"]
            metrics["holder_phi"] = fit_res[method]["holder_phi"]
            metrics["holder_kappa"] = fit_res[method]["holder_kappa"]
            metrics["k_est"] = int(proposal_extras[method]["k_mean"])
            metrics["k_est_central"] = int(proposal_extras[method]["k_central"])
            metrics["k_est_optimistic"] = int(proposal_extras[method]["k_optimistic"])
            metrics["k_est_pessimistic"] = int(proposal_extras[method]["k_pessimistic"])
            method_k_budgets = {
                key: int(proposal_extras[method][key])
                for key in ("k_central", "k_optimistic", "k_pessimistic")
            }
            rank_arrays[f"p_out_{METHOD_JSON_KEYS[method]}"] = np.asarray(proposal_extras[method]["p_out"])
            log_lik_prop = compute_loglik_posterior_mean_regression(
                fit_res[method]["samples"], x_train, y_train
            )
            if eps_true > 0:
                metrics.update(
                    evaluate_detection_metrics(
                        proposal_extras[method]["p_out"],
                        mask_outlier,
                        k_oracle=k_oracle,
                        k_est=metrics["k_est_central"],
                        is_prob=True,
                        log_lik=log_lik_prop,
                        k_budgets=method_k_budgets,
                    )
                )
            else:
                metrics.update(detection_metrics_nan(is_prob=True))
        else:
            likelihood = "student_t" if method == "StudentT" else "gaussian"
            score = score_baseline(
                fit_res[method]["samples"],
                x_train,
                y_train,
                likelihood=likelihood,
                student_t_df=cfg.student_t_df,
                seed=fit_seed,
            )
            rank_arrays[f"score_{METHOD_JSON_KEYS[method]}"] = np.asarray(score)
            if eps_true > 0:
                log_lik_base = compute_loglik_posterior_mean_regression(
                    fit_res[method]["samples"], x_train, y_train
                )
                metrics.update(
                    evaluate_detection_metrics(
                        score,
                        mask_outlier,
                        k_oracle=k_oracle,
                        k_est=primary_k_budgets["k_central"],
                        is_prob=False,
                        log_lik=log_lik_base,
                        k_budgets=primary_k_budgets,
                    )
                )
            else:
                metrics.update(detection_metrics_nan(is_prob=False))
            if method == "StudentT":
                metrics["nu"] = cfg.student_t_df

        diag_detail[method] = conv_detail
        metrics.update(fit_metadata(fit_res[method]))
        results[method] = metrics

    results["_rank_arrays"] = rank_arrays
    results["_inputs_for_disk"] = {"X": np.asarray(x_train), "y": np.asarray(y_train),
        "X_test": np.asarray(x_test), "x_train_raw": x_train_raw,
        "y_train_raw_clean": y_train_raw_clean, "x_test_raw": x_test_raw,
        "y_test_raw": y_test_raw, "mask_outlier": mask_outlier}
    results["_mcmc_diag_detail"] = diag_detail
    if getattr(cfg, "save_samples", False):
        results["_samples_for_disk"] = {m: fit_res[m]["samples"] for m in all_methods}

    results["_meta"] = {
        "dataset": dataset_name,
        "gamma": gamma,
        "eps_true": eps_true,
        "mu_out": mu_out,
        "replicate": rep_id,
        "preprocessing": prep,
        "split_mode": str(cfg.split_mode),
        "split_sizes": str(cfg.split_sizes),
        "n_train": n_train_actual,
        "n_test": n_test_actual,
        "train_ratio": cfg.train_ratio,
        "test_ratio": cfg.test_ratio,
        "k_oracle": k_oracle,
        "k_est_primary": k_est,
        "scenario": "original" if eps_true == 0.0 else "synthetic_contamination",
    }
    return results


def save_single_run_results(cfg: RealDataConfig, result: Dict[str, Any]) -> None:
    diag_detail = result.pop("_mcmc_diag_detail", None) or {}
    samples_by_method = result.pop("_samples_for_disk", None)
    ranks = result.pop("_rank_arrays", None)
    inputs = result.pop("_inputs_for_disk", None)
    meta = result["_meta"]
    run_dir = os.path.join(
        cfg.output_dir,
        f"dataset_{meta['dataset']}",
        f"gamma_{meta['gamma']:.3f}",
        f"eps_{meta['eps_true']:.2f}",
        f"muout_{meta['mu_out']:.1f}",
        f"rep_{meta['replicate']:03d}",
    )
    os.makedirs(run_dir, exist_ok=True)
    out_json = {
        "temperature_version": TEMPERATURE_VERSION,
        "run_config": configuration(cfg, __file__),
        "eta_mode": cfg.eta_mode,
        "eta_gamma_div": cfg.eta_gamma_div,
        "split_mode": meta["split_mode"],
        "dataset": meta["dataset"],
        "gamma_hyperparam": meta["gamma"],
        "eps_true": meta["eps_true"],
        "mu_out": meta["mu_out"],
        "scenario": meta["scenario"],
        "n_train": meta.get("n_train", cfg.n_train),
        "n_test": meta.get("n_test", cfg.n_test),
        "split_sizes": meta.get("split_sizes", cfg.split_sizes),
        "train_ratio": meta.get("train_ratio", cfg.train_ratio),
        "test_ratio": meta.get("test_ratio", cfg.test_ratio),
        "replicate": meta["replicate"],
        "proposal_methods": get_proposal_methods(cfg),
        "holder_phi": cfg.holder_phi,
        "holder_kappa": cfg.holder_kappa,
        "k_oracle": meta["k_oracle"],
        "k_est_primary": meta["k_est_primary"],
        "preprocessing": meta.get("preprocessing", cfg.preprocessing),
        "mcmc_chains": cfg.chains,
        "mcmc_num_warmup": cfg.num_warmup,
        "mcmc_num_samples": cfg.num_samples,
        "mcmc_save_samples": bool(getattr(cfg, "save_samples", False)),
    }
    for method in get_all_methods(cfg):
        block = dict(result[method])
        dextra = diag_detail.get(method)
        if isinstance(dextra, dict) and "error" in dextra:
            block["mcmc_diagnostics_error"] = dextra["error"]
        elif isinstance(dextra, dict):
            block["mcmc_diagnostics_by_param"] = dextra
        out_json[METHOD_JSON_KEYS[method]] = block
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(out_json, f, indent=2, cls=NumpyEncoder)

    if ranks is not None:
        np.savez_compressed(os.path.join(run_dir, "detection_scores.npz"), **ranks)
    if inputs is not None:
        np.savez_compressed(os.path.join(run_dir, "likelihood_inputs.npz"), **inputs)
    if samples_by_method is not None:
        np.savez_compressed(
            os.path.join(run_dir, "posterior_samples.npz"),
            **{METHOD_JSON_KEYS[k]: np.asarray(v) for k, v in samples_by_method.items()},
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Real-data benchmark extension of exp1_regression. "
            "Runs original + synthetic contamination settings for benchmark regression datasets "
            "(cal_housing, pumadyn32fh, ccpp, concrete, kin8nm). "
            "Default: y-only standardization (x raw + intercept), fixed split per replicate, "
            "homogeneous N(mu_out,1) contamination on standardized y."
        )
    )
    parser.add_argument("--n_train", type=int, help="Train size when --split_sizes absolute (default 500)")
    parser.add_argument("--n_test", type=int, help="Test size when --split_sizes absolute (default 1000)")
    parser.add_argument(
        "--split_sizes",
        type=str,
        choices=["absolute", "ratio"],
        default="absolute",
        help="absolute: fixed n_train/n_test subsample; ratio: --train_ratio of full dataset",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        help="Train fraction in (0,1) for ratio split (e.g. 0.8; test gets the rest)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        help="Optional test fraction; if omitted, 1 - train_ratio is used",
    )
    parser.add_argument("--datasets", type=str, help="Comma/space-separated list")
    parser.add_argument("--gammas", type=str, help="Comma/space-separated gamma list")
    parser.add_argument("--eps_grid", type=str, help="Comma/space-separated epsilon list (for synthetic contamination)")
    parser.add_argument("--mu_out_grid", type=str, help="Comma/space-separated mu_out list (for synthetic contamination)")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--student_t_df", type=float)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--num_warmup", type=int)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--eta_mode", type=str)
    parser.add_argument("--eta_gamma_div", type=float, default=1.0)
    parser.add_argument("--eta_fixed", type=float)
    parser.add_argument("--holder_phi", type=str)
    parser.add_argument("--holder_kappa", type=float)
    parser.add_argument("--target_accept", type=float)
    parser.add_argument("--init_step", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--local_data_dir", type=str, help="Directory with CSV/TSV per dataset (e.g. cal_housing.csv) or CCPP xlsx")
    parser.add_argument(
        "--preprocessing",
        type=str,
        choices=["kanamori", "legacy"],
        default="kanamori",
        help=(
            "kanamori: z-score x and y from clean train, then N(mu_out,1) on standardized y. "
            "legacy: contaminate raw y then standardize using contaminated train statistics."
        ),
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["fixed", "per_setting"],
        default="fixed",
        help=(
            "fixed: train/test split depends on replicate only (same split across eps/mu). "
            "per_setting: legacy combined seed (split varies with eps and mu_out)."
        ),
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save flattened posterior draws to posterior_samples.npz in each run directory (large).",
    )
    args = parser.parse_args()

    cfg = RealDataConfig()
    cfg.eta_gamma_div = positive_temperature(args.eta_gamma_div)
    cfg.base_seed = args.base_seed
    cfg.preprocessing = args.preprocessing
    cfg.split_mode = args.split_mode
    cfg.save_samples = bool(args.save_samples)
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
            parser.error(str(exc))
    elif cfg.split_sizes == "ratio":
        parser.error("--train_ratio is required when --split_sizes ratio")
    if args.datasets is not None:
        cfg.datasets = [_sanitize_dataset_name(v) for v in parse_csv_list(args.datasets, str)]
    if args.gammas is not None:
        cfg.gammas = parse_csv_list(args.gammas, float)
    if args.eps_grid is not None:
        cfg.eps_grid = parse_csv_list(args.eps_grid, float)
    if args.mu_out_grid is not None:
        cfg.mu_out_grid = parse_csv_list(args.mu_out_grid, float)
    if args.replicates is not None:
        cfg.replicates = args.replicates
    if args.student_t_df is not None:
        cfg.student_t_df = args.student_t_df
    if args.chains is not None:
        cfg.chains = args.chains
    if args.num_warmup is not None:
        cfg.num_warmup = args.num_warmup
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.eta_mode is not None:
        cfg.eta_mode = args.eta_mode
    if args.eta_fixed is not None:
        cfg.eta_fixed = args.eta_fixed
    if args.holder_phi is not None:
        cfg.holder_phi = args.holder_phi
    if args.holder_kappa is not None:
        cfg.holder_kappa = args.holder_kappa
    if args.target_accept is not None:
        cfg.target_accept = args.target_accept
    if args.init_step is not None:
        cfg.init_step = args.init_step
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.local_data_dir is not None:
        cfg.local_data_dir = args.local_data_dir

    begin_run(cfg, __file__)
    os.makedirs(cfg.output_dir, exist_ok=True)

    start_time = time.time()
    all_rows = []
    failures = []
    dataset_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for dataset_name in cfg.datasets:
        print(f"=== Loading dataset: {dataset_name} ===")
        x_all, y_all = load_benchmark_dataset(dataset_name, local_data_dir=cfg.local_data_dir)
        dataset_cache[dataset_name] = (x_all, y_all)
        print(f"Loaded {dataset_name}: n={x_all.shape[0]}, d={x_all.shape[1]}")
        if cfg.split_sizes == "ratio":
            n_tr, n_te = resolve_train_test_sizes(
                x_all.shape[0],
                train_ratio=cfg.train_ratio,
                test_ratio=cfg.test_ratio,
            )
            print(f"  split ratio {cfg.train_ratio}/{cfg.test_ratio} -> n_train={n_tr}, n_test={n_te}")

    setting_list = build_realdata_contamination_settings(cfg.eps_grid, cfg.mu_out_grid)
    print(f"Contamination settings ({len(setting_list)}): {setting_list}")

    for dataset_name in cfg.datasets:
        x_all, y_all = dataset_cache[dataset_name]
        for gamma in cfg.gammas:
            print(f"=== Dataset={dataset_name}, gamma={gamma:.3f} ===")
            for eps_true, mu_out in setting_list:
                scenario = "original" if eps_true == 0.0 else "synthetic_contamination"
                print(f"  > scenario={scenario}, eps={eps_true:.2f}, mu_out={mu_out:.1f}")
                for rep in range(cfg.replicates):
                    try:
                        result = run_single_setting(
                            cfg=cfg,
                            dataset_name=dataset_name,
                            gamma=gamma,
                            eps_true=eps_true,
                            mu_out=mu_out,
                            rep_id=rep,
                            x_all=x_all,
                            y_all=y_all,
                        )
                        save_single_run_results(cfg, result)

                        for method in get_all_methods(cfg):
                            row = {
                                "dataset": dataset_name,
                                "gamma": gamma,
                                "scenario": result["_meta"]["scenario"],
                                "eps_true": eps_true,
                                "mu_out": mu_out,
                                "replicate": rep,
                                "method": method,
                                "preprocessing": result["_meta"].get("preprocessing", cfg.preprocessing),
                                "split_mode": result["_meta"].get("split_mode", cfg.split_mode),
                                "k_oracle": result["_meta"]["k_oracle"],
                                "k_est_primary": result["_meta"]["k_est_primary"],
                            }
                            row.update(result[method])
                            all_rows.append(row)

                        if hasattr(jax, "clear_caches"):
                            jax.clear_caches()
                        gc.collect()
                    except Exception as exc:
                        failures.append({"dataset": dataset_name, "gamma": gamma, "eps_true": eps_true, "mu_out": mu_out, "replicate": rep, "error": str(exc)})
                        print(
                            f"    ! Error on dataset={dataset_name}, gamma={gamma}, "
                            f"eps={eps_true}, mu_out={mu_out}, rep={rep}: {exc}"
                        )
                        import traceback

                        traceback.print_exc()

    record_completion(cfg, len(cfg.datasets) * len(cfg.gammas) * len(setting_list) * cfg.replicates,
        len(all_rows) // len(get_all_methods(cfg)), failures)
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No successful runs were completed. Check dataset availability and hyperparameters.")

    summary_path = os.path.join(cfg.output_dir, "summary_metrics.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_keys = ["dataset", "gamma", "scenario", "eps_true", "mu_out", "method"]
    numeric_cols = [c for c in numeric_cols if c not in group_keys and c != "replicate"]
    df_agg = df.groupby(group_keys)[numeric_cols].agg(["mean", "std"]).reset_index()
    df_agg.columns = ["_".join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]

    agg_path = os.path.join(cfg.output_dir, "summary_agg.csv")
    df_agg.to_csv(agg_path, index=False)
    print(f"Saved: {agg_path}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
