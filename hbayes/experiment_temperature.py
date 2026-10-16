"""Temperature and reproducibility helpers for the paper's rerun.

The density experiment uses det(sample covariance)**(gamma/2).
Conditional regression uses Var(observed training response)**(gamma/2)
in the response coordinates actually passed to the likelihood.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

TEMPERATURE_VERSION = "holder_affine_response_v1"


def positive_temperature(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Temperature must be positive and finite, got {value}")
    return value


def _gamma(gamma: float) -> float:
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be positive and finite")
    return gamma


def covariance_temperature(data, gamma: float) -> tuple[float, float]:
    """Return eta and det(Cov(data)), using unbiased covariance (ddof=1)."""
    gamma = _gamma(gamma)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError("Covariance temperature requires at least two observations")
    if not np.isfinite(data).all():
        raise ValueError("Non-finite training data in temperature calculation")
    centered = data - data.mean(axis=0)
    cov = centered.T @ centered / (data.shape[0] - 1)
    # A positive determinant alone does not establish positive definiteness.
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Training covariance is not positive definite") from exc
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("Invalid training covariance determinant")
    eta = positive_temperature(np.exp((gamma / 2.0) * logdet))
    determinant = float(np.exp(logdet))
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("Covariance determinant cannot be represented as a finite float")
    return eta, determinant


def response_temperature(y, gamma: float) -> tuple[float, float]:
    """Return Var(y, ddof=1)**(gamma/2) for conditional p(y|x).

    y must be the observed (possibly contaminated) training response in the
    coordinates supplied to the likelihood; covariates have no density Jacobian.
    """
    gamma = _gamma(gamma)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size < 2 or not np.isfinite(y).all():
        raise ValueError("Response temperature requires finite training responses, n >= 2")
    variance = float(np.var(y, ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Observed training response must have positive finite variance")
    return positive_temperature(np.exp((gamma / 2.0) * np.log(variance))), variance


def fit_metadata(fit: dict) -> dict:
    """Scalars safe to copy into a metrics block / CSV row."""
    keys = ("eta", "eta_mode", "cov_det", "response_variance", "temperature_version",
            "holder_phi", "holder_kappa", "rhat_max", "ess_bulk_min", "ess_tail_min",
            "mcmc_diagnostics_error", "nuts_divergences", "nuts_acceptance_mean",
            "inference_version", "sigma_min", "scale_coordinates", "target_support_changed",
            "sigma_near_floor_fraction", "sampler", "diagnostics_pass", "mh_acceptance_mean",
            "proposal_nuts_divergences", "importance_ess", "fit_contract_sha256", "fit_cache",
            "origin", "imported_samples_sha256", "gamma_posterior_definition", "mode_count")
    return {key: fit[key] for key in keys if key in fit}


def nuts_summary(chain_results: list[dict]) -> dict:
    infos = [info for result in chain_results for info in result.get("infos", [])]
    if not infos:
        return {}
    divergences = [bool(np.asarray(info.is_divergent)) for info in infos]
    acceptance = [float(np.asarray(info.acceptance_rate)) for info in infos]
    return {"nuts_divergences": int(sum(divergences)),
            "nuts_acceptance_mean": float(np.mean(acceptance))}


def convergence_summary(samples, chains: int, names: list[str]) -> dict:
    """Use the project's existing rank R-hat / bulk and tail ESS helpers."""
    try:
        from hbayes.diagnostics import arviz_rhat, arviz_ess, arviz_ess_tail
        arr = np.asarray(samples)
        if arr.ndim != 2 or arr.shape[1] != len(names) or arr.shape[0] % chains:
            raise ValueError("Invalid flattened chain shape")
        arr = arr.reshape(chains, -1, len(names))
        rhat = np.asarray(arviz_rhat(arr, names), dtype=float)
        bulk = np.asarray(arviz_ess(arr, names), dtype=float)
        tail = np.asarray(arviz_ess_tail(arr, names), dtype=float)
        if not all(np.isfinite(v).all() for v in (rhat, bulk, tail)):
            raise ValueError("Non-finite convergence diagnostics")
        return {"rhat_max": float(rhat.max()), "ess_bulk_min": float(bulk.min()),
                "ess_tail_min": float(tail.min())}
    except Exception as exc:
        return {"rhat_max": float("nan"), "ess_bulk_min": float("nan"),
                "ess_tail_min": float("nan"), "mcmc_diagnostics_error": str(exc)}


def configuration(cfg, runner_path: str) -> dict:
    config = asdict(cfg) if is_dataclass(cfg) else vars(cfg).copy()
    project = Path(__file__).resolve().parent.parent
    paths = [Path(runner_path).resolve(), Path(__file__).resolve(),
             *[project / "hbayes" / name for name in
               ("models.py", "dpd.py", "posteriors.py", "sampling.py", "diagnostics.py")],
             project / "experiments/exp1_regression/utils_exp1_regression.py",
             project / "experiments/exp1_simulation/utils_exp1.py"]
    from hbayes.paper_sampling import VERSION
    config["inference_version"] = VERSION
    config["regression_holder_sigma_min"] = 0.05
    config["density_holder_sigma_min"] = None
    from hbayes.gamma_posterior import specification
    config["gamma_div_protocol"] = specification()
    paths += [project / "hbayes/gamma_posterior.py", project / "hbayes/gamma_sampling.py"]
    config["temperature_version"] = TEMPERATURE_VERSION
    config["temperature_covariance_ddof"] = 1
    config["code_sha256"] = {
        (str(p.relative_to(project)) if p.is_relative_to(project) else p.name):
            hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths if p.is_file()
    }
    return config


def begin_run(cfg, runner_path: str) -> None:
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_config.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get("temperature_version") != TEMPERATURE_VERSION:
            raise ValueError("Output directory has a different temperature version; use a new output_dir")
    elif next(root.rglob("metrics.json"), None) is not None:
        raise ValueError("Output directory contains unversioned results; use a new output_dir")
    path.write_text(json.dumps(configuration(cfg, runner_path), indent=2) + "\n")


def record_completion(cfg, expected: int, succeeded: int, failures: list[dict]) -> None:
    report = {"temperature_version": TEMPERATURE_VERSION, "expected_runs": expected,
              "successful_runs": succeeded, "failed_runs": failures,
              "complete": succeeded == expected and not failures}
    Path(cfg.output_dir, "run_completion.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        raise RuntimeError(f"Incomplete experiment: {succeeded}/{expected} successful runs; see run_completion.json")
