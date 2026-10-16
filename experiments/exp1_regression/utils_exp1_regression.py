import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, expit
from scipy.stats import norm, t as student_t_distribution
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from hbayes.models import LinearRegressionParams, linear_regression_log_pdf
from hbayes.posteriors import (
    MVNGibbsConfig,
    MVNPriorsConfig,
    PriorsConfig,
    log_prior_theta_regression,
    logprob_joint_regression_from_vector,
    logprob_theta_only_nll_regression_from_vector,
)
from hbayes.sampling import run_nuts_adapted
from hbayes.paper_sampling import CONTEXT
from hbayes.gamma_sampling import fit as paper_fit
from hbayes.gamma_posterior import logprob_regression as logprob_theta_only_gamma_div_regression_from_vector
from hbayes.experiment_temperature import (TEMPERATURE_VERSION, covariance_temperature,
    response_temperature, positive_temperature, fit_metadata, nuts_summary, convergence_summary,
    configuration, begin_run, record_completion)


def simulate_regression_dataset(
    n,
    n_test,
    eps,
    mu_out,
    seed,
    x_dim=2,
    noise_scale=1.0,
    contamination_mode="response_homogeneous",
):
    """Simulate contaminated linear regression data.

    contamination_mode:
      - response_homogeneous: y is replaced with N(x^T beta + mu_out, 1) at constant rate eps
      - response_heterogeneous: y is replaced with x-dependent rate
        c0(x)=1 if x1>0, c0(x)=0.6 otherwise (expected contamination ~20%)
      - intercept_shift / leverage are kept for backward compatibility
    """
    key = jax.random.PRNGKey(seed)
    key_beta, key_x_tr, key_x_te, key_noise_tr, key_noise_te, key_mask, key_out, key_x_out = jax.random.split(key, 8)

    beta_true = jax.random.normal(key_beta, shape=(x_dim,))
    x_train = jax.random.normal(key_x_tr, shape=(n, x_dim))
    x_test = jax.random.normal(key_x_te, shape=(n_test, x_dim))

    noise_train = noise_scale * jax.random.normal(key_noise_tr, shape=(n,))
    noise_test = noise_scale * jax.random.normal(key_noise_te, shape=(n_test,))
    noise_out = noise_scale * jax.random.normal(key_out, shape=(n,))

    mean_train = x_train @ beta_true
    mean_test = x_test @ beta_true
    mask_outlier = jax.random.bernoulli(key_mask, p=eps, shape=(n,))
    y_clean = mean_train + noise_train

    mode = str(contamination_mode).lower()
    if mode == "leverage":
        # High-leverage outliers: shift x away from the clean design cloud,
        # and generate responses with an opposing slope to perturb beta estimates.
        shift = jnp.array([float(mu_out), -float(mu_out)], dtype=x_train.dtype)
        if x_dim > 2:
            shift = jnp.pad(shift, (0, x_dim - 2))
        x_out = jax.random.normal(key_x_out, shape=(n, x_dim)) + shift[None, :]
        beta_out = -beta_true
        y_out = x_out @ beta_out + noise_out
        x_train = jnp.where(mask_outlier[:, None], x_out, x_train)
    elif mode in {"response_heterogeneous", "heterogeneous"}:
        # Literature-style heterogeneous contamination:
        # c0(x)=1 if x1>0, c0(x)=0.6 otherwise => outlier prob in {0, 0.4}.
        p_outlier = jnp.where(x_train[:, 0] > 0.0, 0.0, 0.4)
        mask_outlier = jax.random.bernoulli(key_mask, p=p_outlier, shape=(n,))
        y_out = mean_train + mu_out + noise_out
    else:
        # Default: literature-style homogeneous response contamination.
        y_out = mean_train + mu_out + noise_out

    y_train = jnp.where(mask_outlier, y_out, y_clean)
    y_test = mean_test + noise_test

    beta_true_full = jnp.concatenate([jnp.zeros(1), beta_true])

    return {
        "x_train": np.asarray(x_train),
        "y_train": np.asarray(y_train),
        "x_test": np.asarray(x_test),
        "y_test": np.asarray(y_test),
        "mask_outlier": np.asarray(mask_outlier).astype(bool),
        "beta_true": np.asarray(beta_true_full),
        "sigma_true": float(noise_scale),
    }


def make_realdata_split_seed(base_seed: int, rep_id: int) -> int:
    """Seed for train/test (or base/batch/test) split; depends on replicate only."""
    return int(base_seed + 100_000 * rep_id)


def make_realdata_contamination_seed(
    base_seed: int, rep_id: int, eps_true: float, mu_out: float
) -> int:
    """Seed for Bernoulli contamination mask and additive shift on standardized ``y``."""
    return int(
        base_seed
        + 100_000 * rep_id
        + 1_000 * int(round(100 * eps_true))
        + 10 * int(round(mu_out))
        + 97
    )


def make_realdata_combined_data_seed(
    base_seed: int, rep_id: int, eps_true: float, mu_out: float
) -> int:
    """Legacy seed: split and contamination settings share one value (pre-2026 protocol)."""
    return int(
        base_seed
        + 100_000 * rep_id
        + 1_000 * int(round(100 * eps_true))
        + 10 * int(round(mu_out))
    )


def make_realdata_fit_seed(
    base_seed: int, rep_id: int, eps_true: float, mu_out: float, gamma: float
) -> int:
    return int(
        base_seed
        + 100_000 * rep_id
        + 1_000 * int(round(100 * eps_true))
        + 10 * int(round(mu_out))
        + int(round(gamma * 10_000))
    )


def resolve_realdata_split_seed(
    base_seed: int,
    rep_id: int,
    eps_true: float,
    mu_out: float,
    *,
    split_mode: str = "fixed",
) -> int:
    mode = str(split_mode).strip().lower()
    if mode == "fixed":
        return make_realdata_split_seed(base_seed, rep_id)
    if mode == "per_setting":
        return make_realdata_combined_data_seed(base_seed, rep_id, eps_true, mu_out)
    raise ValueError(f"Unknown split_mode: {split_mode}")


def normalize_train_test_ratios(
    train_ratio: float | None = None,
    test_ratio: float | None = None,
) -> Tuple[float, float]:
    """Return ``(train_ratio, test_ratio)`` with the missing side inferred as ``1 - other``.

    If both are given they must sum to 1 (within tolerance).
    """
    if train_ratio is None and test_ratio is None:
        raise ValueError("train_ratio or test_ratio must be set for ratio split.")
    if train_ratio is not None and test_ratio is not None:
        tr = float(train_ratio)
        te = float(test_ratio)
        if not (0.0 < tr < 1.0 and 0.0 < te < 1.0):
            raise ValueError(f"Ratios must be in (0, 1), got train_ratio={tr}, test_ratio={te}")
        if abs(tr + te - 1.0) > 1e-9:
            raise ValueError(f"train_ratio + test_ratio must equal 1, got {tr + te}")
        return tr, te
    if train_ratio is not None:
        tr = float(train_ratio)
        if not 0.0 < tr < 1.0:
            raise ValueError(f"train_ratio must be in (0, 1), got {tr}")
        return tr, 1.0 - tr
    te = float(test_ratio)
    if not 0.0 < te < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {te}")
    return 1.0 - te, te


def build_realdata_contamination_settings(
    eps_grid: Sequence[float],
    mu_out_grid: Sequence[float],
    *,
    include_clean_if_missing: bool = True,
) -> List[Tuple[float, float]]:
    """Build ``(eps_true, mu_out)`` pairs without redundant runs when ``eps=0``.

    - ``eps == 0``: one run with ``mu_out=0``; entries in ``mu_out_grid`` are ignored.
    - ``eps > 0``: one run per ``mu_out`` in ``mu_out_grid``.
    - If ``include_clean_if_missing`` and no ``eps=0`` appears in ``eps_grid``, prepend
      ``(0, 0)`` once (clean baseline).
    """
    eps_list = [float(e) for e in eps_grid]
    mu_list = [float(m) for m in mu_out_grid]
    settings: List[Tuple[float, float]] = []

    if include_clean_if_missing and not any(e == 0.0 for e in eps_list):
        settings.append((0.0, 0.0))

    for eps in eps_list:
        if eps == 0.0:
            if not any(s[0] == 0.0 for s in settings):
                settings.append((0.0, 0.0))
        else:
            if not mu_list:
                raise ValueError(
                    "mu_out_grid must be non-empty when eps_grid contains eps > 0"
                )
            for mu in mu_list:
                settings.append((eps, mu))
    return settings


def resolve_train_test_sizes(
    n_total: int,
    *,
    n_train: int | None = None,
    n_test: int | None = None,
    train_ratio: float | None = None,
    test_ratio: float | None = None,
) -> Tuple[int, int]:
    """Resolve train/test counts from absolute sizes or a train/test ratio.

    Ratio mode uses all rows: ``n_train = floor(n_total * train_ratio)``,
    ``n_test = n_total - n_train``. Only ``train_ratio`` is required; ``test_ratio``
    defaults to ``1 - train_ratio``.
    Absolute mode subsamples ``n_train + n_test`` rows without replacement.
    """
    n_total = int(n_total)
    if train_ratio is not None or test_ratio is not None:
        tr, te = normalize_train_test_ratios(train_ratio, test_ratio)
        n_tr = int(np.floor(n_total * tr))
        n_te = int(n_total - n_tr)
        if n_tr < 1 or n_te < 1:
            raise ValueError(
                f"Ratio split too small for n_total={n_total}: n_train={n_tr}, n_test={n_te}"
            )
        return n_tr, n_te

    if n_train is None or n_test is None:
        raise ValueError("Both n_train and n_test must be set for absolute split.")
    n_tr = int(n_train)
    n_te = int(n_test)
    if n_tr < 1 or n_te < 1:
        raise ValueError(f"n_train and n_test must be positive, got {n_tr}, {n_te}")
    if n_total < n_tr + n_te:
        raise ValueError(f"Not enough rows: required {n_tr + n_te}, got {n_total}")
    return n_tr, n_te


def sample_train_test(
    x_all: np.ndarray,
    y_all: np.ndarray,
    seed: int,
    *,
    n_train: int | None = None,
    n_test: int | None = None,
    train_ratio: float | None = None,
    test_ratio: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random train/test split with absolute counts or train/test ratios."""
    n_tr, n_te = resolve_train_test_sizes(
        x_all.shape[0],
        n_train=n_train,
        n_test=n_test,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
    )
    rng = np.random.default_rng(seed)
    use_all = train_ratio is not None or test_ratio is not None
    if use_all:
        idx = rng.permutation(x_all.shape[0])
    else:
        idx = rng.choice(x_all.shape[0], size=n_tr + n_te, replace=False)
    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr : n_tr + n_te]
    return x_all[tr_idx], y_all[tr_idx], x_all[te_idx], y_all[te_idx]


def sample_train_test_with_split_cfg(
    x_all: np.ndarray,
    y_all: np.ndarray,
    seed: int,
    *,
    split_sizes: str = "absolute",
    n_train: int = 500,
    n_test: int = 1000,
    train_ratio: float | None = None,
    test_ratio: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch train/test split according to ``split_sizes`` (``absolute`` or ``ratio``)."""
    mode = str(split_sizes).strip().lower()
    if mode == "ratio":
        tr, te = normalize_train_test_ratios(train_ratio, test_ratio)
        return sample_train_test(
            x_all,
            y_all,
            seed,
            train_ratio=tr,
            test_ratio=te,
        )
    if mode == "absolute":
        return sample_train_test(x_all, y_all, seed, n_train=n_train, n_test=n_test)
    raise ValueError(f"Unknown split_sizes: {split_sizes}")


def kanamori_standardize_regression_data(
    x_train_raw: np.ndarray,
    y_train_clean_raw: np.ndarray,
    x_test_raw: np.ndarray,
):
    """Semi-synthetic real-data preprocessing for Sec.~6.2 / 6.3.

    Fit mean/std on the **clean** training set only, then z-score ``x`` and ``y``.
    Contamination should be applied **after** this step in standardized ``y`` space as
    additive :math:`\\mathcal{N}(\\mu_{\\mathrm{out}}, 1)` noise. An intercept column is
    appended to ``x``.
    """
    x_train_raw = np.asarray(x_train_raw, dtype=float)
    y_train_clean_raw = np.asarray(y_train_clean_raw, dtype=float)
    x_test_raw = np.asarray(x_test_raw, dtype=float)

    y_mean = float(y_train_clean_raw.mean())
    y_std = float(y_train_clean_raw.std())
    if y_std < 1e-12:
        y_std = 1.0
    y_train_clean_scaled = (y_train_clean_raw - y_mean) / y_std

    x_mean = x_train_raw.mean(axis=0)
    x_std = x_train_raw.std(axis=0)
    x_std = np.where(x_std < 1e-9, 1.0, x_std)
    x_train_features = (x_train_raw - x_mean) / x_std
    x_test_features = (x_test_raw - x_mean) / x_std

    x_train_design = np.hstack([np.ones((x_train_raw.shape[0], 1)), x_train_features])
    x_test_design = np.hstack([np.ones((x_test_raw.shape[0], 1)), x_test_features])

    scaler = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return (
        jnp.asarray(x_train_design, dtype=jnp.float32),
        jnp.asarray(y_train_clean_scaled, dtype=jnp.float32),
        jnp.asarray(x_test_design, dtype=jnp.float32),
        scaler,
    )


def contaminate_response(
    y_clean: np.ndarray, eps: float, mu_out: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Homogeneous response contamination (Bernoulli mask, additive Gaussian shift).

    Intended for use **after** Kanamori-style standardization of ``y`` so that the
    shift is :math:`\\mathcal{N}(\\mu_{\\mathrm{out}}, 1)` in standardized response units.
    """
    rng = np.random.default_rng(seed)
    y_clean = np.asarray(y_clean, dtype=float)
    n = len(y_clean)
    mask = rng.random(n) < float(eps)
    y_out = y_clean.copy()
    if mask.any():
        y_out[mask] = y_clean[mask] + rng.normal(loc=float(mu_out), scale=1.0, size=int(mask.sum()))
    return y_out, mask


def standardize_regression_data(x_train, y_train, x_test):
    """Standardize features and response using training statistics."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-9, 1.0, x_std)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-9:
        y_std = 1.0

    x_train_scaled = (x_train - x_mean) / x_std
    x_test_scaled = (x_test - x_mean) / x_std
    y_train_scaled = (y_train - y_mean) / y_std

    x_train_design = np.hstack([np.ones((x_train.shape[0], 1)), x_train_scaled])
    x_test_design = np.hstack([np.ones((x_test.shape[0], 1)), x_test_scaled])

    scaler = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return (
        jnp.asarray(x_train_design, dtype=jnp.float32),
        jnp.asarray(y_train_scaled, dtype=jnp.float32),
        jnp.asarray(x_test_design, dtype=jnp.float32),
        scaler,
    )


def apply_fixed_regression_scaler(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    scaler: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Transform raw data with a precomputed Kanamori-style scaler (clean train statistics)."""
    x_train_raw = np.asarray(x_train_raw, dtype=float)
    y_train_raw = np.asarray(y_train_raw, dtype=float)
    x_test_raw = np.asarray(x_test_raw, dtype=float)

    x_mean = np.asarray(scaler["x_mean"], dtype=float)
    x_std = np.asarray(scaler["x_std"], dtype=float)
    y_mean = float(scaler["y_mean"])
    y_std = float(scaler["y_std"])
    if y_std < 1e-12:
        y_std = 1.0

    x_train_features = (x_train_raw - x_mean) / x_std
    x_test_features = (x_test_raw - x_mean) / x_std
    y_train_scaled = (y_train_raw - y_mean) / y_std

    x_train_design = np.hstack([np.ones((x_train_raw.shape[0], 1)), x_train_features])
    x_test_design = np.hstack([np.ones((x_test_raw.shape[0], 1)), x_test_features])

    return (
        jnp.asarray(x_train_design, dtype=jnp.float32),
        jnp.asarray(y_train_scaled, dtype=jnp.float32),
        jnp.asarray(x_test_design, dtype=jnp.float32),
        scaler,
    )


def _eta_covdet_regression(x, y, gamma):
    """Compatibility name: return the conditional response-scale temperature.

    The covariates are conditioned on; their covariance determinant is not part
    of the response-density Jacobian. New commands should use response_var.
    """
    return response_temperature(y, gamma)


def _student_t_logpdf(y, mu, log_scale, nu):
    scale = jnp.exp(log_scale)
    z = (y - mu) / scale
    c0 = jsp.gammaln((nu + 1.0) / 2.0) - jsp.gammaln(nu / 2.0)
    c1 = -0.5 * jnp.log(nu * jnp.pi) - jnp.log(scale)
    c2 = -((nu + 1.0) / 2.0) * jnp.log1p((z * z) / nu)
    return c0 + c1 + c2


def logprob_theta_only_student_t_regression_from_vector(x, y, priors, nu=4.0):
    d_in = x.shape[1]

    def _logprob(z):
        beta = z[:d_in]
        log_scale = z[d_in]
        params = LinearRegressionParams(beta=beta, log_scale=log_scale)
        mu = x @ beta
        ll = jnp.sum(_student_t_logpdf(y, mu, log_scale, nu))
        return log_prior_theta_regression(params, priors) + ll

    return _logprob


def reshape_flat_mcmc_draws_to_chains3d(samples_flat: Any, n_chains: int) -> jnp.ndarray:
    """Undo ``_run_chains`` flattening: (n_chains * n_draws, dim) -> (n_chains, n_draws, dim)."""
    a = np.asarray(samples_flat, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D samples, got shape {a.shape}")
    n_total, dim = int(a.shape[0]), int(a.shape[1])
    if n_chains < 2:
        raise ValueError(f"n_chains must be >= 2 for R-hat, got {n_chains}")
    if n_total % n_chains != 0:
        raise ValueError(
            f"Total posterior draws {n_total} not divisible by n_chains={n_chains} "
            "(check chains * num_samples matches stacked sampler output)."
        )
    n_draw = n_total // n_chains
    return a.reshape((n_chains, n_draw, dim))


def regression_mcmc_var_names(n_design_cols: int, joint_with_rho: bool) -> list[str]:
    names = [f"beta_{j}" for j in range(int(n_design_cols))] + ["log_scale"]
    if joint_with_rho:
        names.append("rho")
    return names


def compute_regression_mcmc_convergence_summaries(
    samples_flat: Any,
    *,
    n_chains: int,
    joint_with_rho: bool,
    n_design_cols: int,
) -> Tuple[Dict[str, float], Union[Dict[str, Dict[str, float]], Dict[str, str]]]:
    """Rank R-hat and bulk / tail ESS (ArviZ) for linear-regression MCMC stacks from ``_run_chains``."""
    nan_scalars = {
        "rhat_max": float("nan"),
        "ess_bulk_min": float("nan"),
        "ess_tail_min": float("nan"),
    }
    try:
        from hbayes.paper_sampling import diagnostics
        names = regression_mcmc_var_names(n_design_cols, joint_with_rho)
        chains3 = reshape_flat_mcmc_draws_to_chains3d(samples_flat, int(n_chains))
        result = diagnostics(chains3, names, int(n_design_cols), joint=joint_with_rho)
        by_param = {stat: {name: row[stat] for name, row in result["by_parameter"].items()}
                    for stat in ["rhat", "ess_bulk", "ess_tail"]}
        scalars = {name: result[name] for name in ["rhat_max", "ess_bulk_min", "ess_tail_min"]}
        return scalars, by_param
    except Exception as exc:
        return nan_scalars, {"error": str(exc)}


def _run_chains(logprob_fn, init_val, seed, chains, num_warmup, num_samples,
                target_accept, init_step, return_metadata=False):
    keys = jax.random.split(jax.random.PRNGKey(seed), chains)
    draws, stats = [], []
    for k in keys:
        res = run_nuts_adapted(k, logprob_fn, init_val, num_warmup=num_warmup,
            num_samples=num_samples, target_acceptance_rate=target_accept,
            initial_step_size=init_step, is_mass_matrix_diagonal=True)
        draws.append(res['samples'])
        stats.append(nuts_summary([res]))
    samples = jnp.stack(draws, axis=0).reshape((-1, init_val.shape[0]))
    if not return_metadata:
        return samples
    out = {'samples': samples}
    if all('nuts_divergences' in stat for stat in stats):
        out['nuts_divergences'] = sum(stat['nuts_divergences'] for stat in stats)
        out['nuts_acceptance_mean'] = float(np.mean([stat['nuts_acceptance_mean'] for stat in stats]))
    return out


def _canonical_holder_phi(holder_phi: str) -> str:
    return str(holder_phi).lower().replace("-", "_")


def _resolve_holder_kappa(holder_phi: str, holder_kappa):
    if holder_kappa is not None:
        return float(holder_kappa)
    phi = _canonical_holder_phi(holder_phi)
    if phi in {"exponential_saturated", "exp_saturated", "saturated_exponential", "exp_sat"}:
        return 0.5
    if phi in {"rational_saturated", "sat_rational", "rational_sat"}:
        return 0.25
    return 1.0


PROPOSAL_METHOD_SPECS = {
    "DPD": {"holder_phi": "dpd", "holder_kappa": 1.0},
    "ExpSatDPD": {"holder_phi": "exponential_saturated", "holder_kappa": 0.5},
    "RatSatDPD": {"holder_phi": "rational_saturated", "holder_kappa": 0.25},
}

# Stable iteration order for experiments that fit all three Hölder proposals.
PROPOSAL_METHOD_ORDER = tuple(PROPOSAL_METHOD_SPECS.keys())


def resolve_proposal_method_specs(holder_phi=None, holder_kappa=None):
    if holder_phi is None:
        return {
            name: {
                "holder_phi": spec["holder_phi"],
                "holder_kappa": float(spec["holder_kappa"]),
            }
            for name, spec in PROPOSAL_METHOD_SPECS.items()
        }

    phi = _canonical_holder_phi(holder_phi)
    for name, spec in PROPOSAL_METHOD_SPECS.items():
        if _canonical_holder_phi(spec["holder_phi"]) == phi:
            return {
                name: {
                    "holder_phi": spec["holder_phi"],
                    "holder_kappa": _resolve_holder_kappa(phi, holder_kappa),
                }
            }
    raise ValueError(f"Unknown holder_phi: {holder_phi}")


def fit_dpd_variant(
    x_train,
    y_train,
    seed,
    chains=4,
    num_warmup=1000,
    num_samples=1000,
    gamma=0.5,
    eta_mode="response_var",
    eta_fixed=1.0,
    holder_phi="dpd",
    holder_kappa=None,
    target_accept=0.8,
    init_step=1.0,
):
    """Fit the DPD joint regression model and return samples plus eta metadata."""
    d_in = x_train.shape[1]
    pri = MVNPriorsConfig()

    if eta_mode in {"covdet", "response_var"}:
        eta_val, cov_det = response_temperature(y_train, gamma)
        eta_mode = "response_var"
    elif eta_mode == "fixed":
        eta_val, cov_det = positive_temperature(eta_fixed), np.nan
    else:
        raise ValueError(f"Unknown eta_mode: {eta_mode}")

    holder_phi_name = _canonical_holder_phi(holder_phi)
    holder_kappa_value = _resolve_holder_kappa(holder_phi_name, holder_kappa)

    gcfg = MVNGibbsConfig(
        gamma=gamma,
        eta=float(eta_val),
        holder_phi=holder_phi_name,
        holder_kappa=float(holder_kappa_value),
        priors=pri,
    )

    init_joint = jnp.concatenate([jnp.zeros(d_in), jnp.zeros(1), jnp.zeros(1)])
    import jax.experimental
    with jax.experimental.enable_x64():
        X64 = jnp.asarray(x_train, dtype=jnp.float64)
        y64 = jnp.asarray(y_train, dtype=jnp.float64)
        method = next((name for name, spec in PROPOSAL_METHOD_SPECS.items()
                       if _canonical_holder_phi(spec["holder_phi"]) == holder_phi_name), holder_phi_name)
        sampled = paper_fit(
            logprob_joint_regression_from_vector(X64, y64, gcfg), [x_train, y_train],
            method=method, model="regression", seed=seed, chains=chains,
            num_warmup=num_warmup, num_samples=num_samples, target_accept=target_accept,
            init_step=init_step, priors=pri, eta=float(eta_val), gamma=gamma,
            phi=holder_phi_name, kappa=holder_kappa_value, floor=0.05,
            names=regression_mcmc_var_names(d_in, True))
    samples = sampled["samples"]
    return {
        **sampled,
        "temperature_version": TEMPERATURE_VERSION,
        "eta_mode": eta_mode,
        "response_variance": float(np.var(np.asarray(y_train), ddof=1)),
        "samples": samples,
        "eta": float(eta_val),
        "cov_det": float(cov_det),
        "holder_phi": str(holder_phi_name),
        "holder_kappa": float(holder_kappa_value),
    }


def fit_regression_methods(
    x_train,
    y_train,
    seed,
    chains=4,
    num_warmup=1000,
    num_samples=1000,
    gamma=0.5,
    eta_mode="response_var",
    eta_fixed=1.0,
    holder_phi="dpd",
    holder_kappa=None,
    student_t_df=4.0,
    target_accept=0.8,
    init_step=1.0,
    fit_score_baselines: bool = True,
    eta_gamma_div: float = 1.0,
):
    """Fit proposal families plus (optionally) Gaussian NLL, Gamma-divergence, and Student-t Bayes."""
    d_in = x_train.shape[1]
    pri = MVNPriorsConfig()
    theta_priors = PriorsConfig()

    proposal_specs = resolve_proposal_method_specs(holder_phi=holder_phi, holder_kappa=holder_kappa)
    proposal_results = {}
    for offset, (name, spec) in enumerate(proposal_specs.items()):
        proposal_results[name] = fit_dpd_variant(
            x_train,
            y_train,
            seed=seed + 1000 * offset,
            chains=chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            gamma=gamma,
            eta_mode=eta_mode,
            eta_fixed=eta_fixed,
            holder_phi=spec["holder_phi"],
            holder_kappa=spec["holder_kappa"],
            target_accept=target_accept,
            init_step=init_step,
        )

    first_proposal = next(iter(proposal_results.values()))
    gcfg = MVNGibbsConfig(
        gamma=gamma,
        eta=positive_temperature(eta_gamma_div),
        holder_phi="dpd",
        holder_kappa=1.0,
        priors=pri,
    )

    init_theta = jnp.concatenate([jnp.zeros(d_in), jnp.zeros(1)])

    method_specs = {
        "NLL": {
            "logprob": logprob_theta_only_nll_regression_from_vector(x_train, y_train, pri),
            "init": init_theta,
            "seed": seed + 101,
        },
        "GammaDiv": {
            "logprob": logprob_theta_only_gamma_div_regression_from_vector(x_train, y_train, gcfg),
            "init": init_theta,
            "seed": seed + 202,
        },
        "StudentT": {
            "logprob": logprob_theta_only_student_t_regression_from_vector(
                x_train, y_train, theta_priors, nu=student_t_df
            ),
            "init": init_theta,
            "seed": seed + 303,
        },
    }

    results: Dict[str, Any] = {}
    results.update(proposal_results)
    if not fit_score_baselines:
        return results

    with jax.experimental.enable_x64():
        X64, y64 = jnp.asarray(x_train, dtype=jnp.float64), jnp.asarray(y_train, dtype=jnp.float64)
        logprobs = {
            "NLL": logprob_theta_only_nll_regression_from_vector(X64, y64, pri),
            "GammaDiv": logprob_theta_only_gamma_div_regression_from_vector(X64, y64, gcfg),
            "StudentT": logprob_theta_only_student_t_regression_from_vector(X64, y64, theta_priors, nu=student_t_df),
        }
        for name, spec in method_specs.items():
            sampled = paper_fit(logprobs[name], [x_train, y_train], method=name,
                model="regression", seed=spec["seed"], chains=chains,
                num_warmup=num_warmup, num_samples=num_samples, target_accept=target_accept,
                init_step=init_step, priors=pri, student_t_df=student_t_df,
                eta=positive_temperature(eta_gamma_div) if name == "GammaDiv" else 1.,
                gamma=gamma, names=regression_mcmc_var_names(d_in, False))
            results[name] = {**sampled, "temperature_version": TEMPERATURE_VERSION,
                "eta": positive_temperature(eta_gamma_div) if name == "GammaDiv" else 1.,
                "eta_mode": "gamma_div_fixed" if name == "GammaDiv" else "likelihood"}

    results["StudentT"]["nu"] = float(student_t_df)
    return results


def compute_log_likelihoods_gaussian(x, y, samples):
    x, y, samples = (np.asarray(a, dtype=np.float64) for a in [x, y, samples])
    d = x.shape[1]
    sigma = np.exp(samples[:, d:d+1])
    residual = (y[None, :] - samples[:, :d] @ x.T) / sigma
    return -.5*(np.log(2*np.pi)+2*samples[:, d:d+1]+residual**2)


def compute_log_likelihoods_student_t(x, y, samples, nu):
    x, y, samples = (np.asarray(a, dtype=np.float64) for a in [x, y, samples])
    d = x.shape[1]
    return student_t_distribution.logpdf(y[None, :], df=nu,
        loc=samples[:, :d] @ x.T, scale=np.exp(samples[:, d:d+1]))


def score_baseline(samples, x, y, likelihood="gaussian", student_t_df=4.0, s_eval_max=400, seed=42):
    s_full = samples.shape[0]
    if s_full > s_eval_max:
        rng = np.random.default_rng(seed)
        eval_idx = np.sort(rng.choice(s_full, size=s_eval_max, replace=False))
        samples_eval = samples[eval_idx]
    else:
        samples_eval = samples

    if likelihood == "gaussian":
        ll_mat = compute_log_likelihoods_gaussian(x, y, samples_eval)
    elif likelihood == "student_t":
        ll_mat = compute_log_likelihoods_student_t(x, y, samples_eval, nu=student_t_df)
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")

    log_mean_exp = logsumexp(ll_mat, axis=0) - np.log(samples_eval.shape[0])
    return -np.asarray(log_mean_exp)


def compute_dpd_outlier_prob(samples_joint, x, y, s_eval_max=400, seed=42):
    """Estimate outlier probabilities from DPD joint posterior samples."""
    s_full = samples_joint.shape[0]
    n = x.shape[0]
    if s_full > s_eval_max:
        rng = np.random.default_rng(seed)
        eval_idx = np.sort(rng.choice(s_full, size=s_eval_max, replace=False))
        samples_eval = samples_joint[eval_idx]
    else:
        samples_eval = samples_joint

    log_liks = np.asarray(compute_log_likelihoods_gaussian(x, y, samples_eval))
    rho = samples_eval[:, -1]
    eps_s = expit(-np.asarray(rho, dtype=np.float64))
    k_s = np.clip(np.round(eps_s * float(n)).astype(int), 0, n)

    flags = np.zeros((samples_eval.shape[0], n), dtype=np.uint8)
    for i in range(samples_eval.shape[0]):
        k = int(k_s[i])
        if k > 0:
            idx = np.argsort(log_liks[i])[:k]
            flags[i, idx] = 1

    return {
        "p_out": flags.mean(axis=0),
        "eps_samples": eps_s,
        "k_samples": k_s,
        "k_mean": float(k_s.mean()),
        **summarize_k_budgets(k_s, n),
    }


def summarize_k_budgets(
    k_samples: np.ndarray,
    n: int,
    *,
    ci_low_pct: float = 2.5,
    ci_high_pct: float = 97.5,
) -> dict[str, float | int]:
    """Posterior budgets for K = round(eps * n) (simulation / paper suite convention).

    central: posterior mean of K (rounded).
    optimistic: lower credible quantile of K (fewer flagged points).
    pessimistic: upper credible quantile of K (more flagged points).
    """
    k_arr = np.asarray(k_samples, dtype=float).ravel()
    n_int = int(n)
    k_central = int(np.clip(np.round(k_arr.mean()), 0, n_int))
    k_optimistic = int(np.clip(np.floor(np.percentile(k_arr, ci_low_pct)), 0, n_int))
    k_pessimistic = int(np.clip(np.ceil(np.percentile(k_arr, ci_high_pct)), 0, n_int))
    return {
        "k_central": k_central,
        "k_optimistic": k_optimistic,
        "k_pessimistic": k_pessimistic,
        "k_ci_low_pct": float(ci_low_pct),
        "k_ci_high_pct": float(ci_high_pct),
    }


def compute_loglik_posterior_mean_regression(samples: np.ndarray, x, y) -> np.ndarray:
    """Pointwise Gaussian log-likelihood using the posterior mean of (beta, log_scale)."""
    samples = np.asarray(samples, dtype=float)
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    d_in = x_np.shape[1]
    theta_mean = samples.mean(axis=0)
    beta = theta_mean[:d_in]
    log_scale = float(theta_mean[d_in])
    sigma = max(float(np.exp(log_scale)), 1e-9)
    mu = x_np @ beta
    z = (y_np - mu) / sigma
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (z**2)


def _recall_topk_by_rank(y_true: np.ndarray, idx_rank: np.ndarray, k: int, n_outliers: int) -> float:
    if k is None or k <= 0 or n_outliers <= 0:
        return 0.0
    hits = int(np.sum(y_true[idx_rank[: int(k)]]))
    return float(hits / n_outliers)


def detection_metrics_nan(*, is_prob: bool = False) -> dict[str, float]:
    out = {
        "auprc": np.nan,
        "auroc": np.nan,
        "recall_k_oracle": np.nan,
        "recall_k_est": np.nan,
        "recall_k_lik_oracle": np.nan,
        "recall_k_lik_central": np.nan,
        "recall_k_lik_optimistic": np.nan,
        "recall_k_lik_pessimistic": np.nan,
    }
    if is_prob:
        out["brier"] = np.nan
    return out


def evaluate_detection_metrics(
    p_or_score,
    y_true,
    k_oracle,
    k_est=None,
    is_prob=False,
    *,
    log_lik=None,
    k_budgets: dict[str, int] | None = None,
):
    y_true = np.asarray(y_true).astype(int)
    n_outliers = int(y_true.sum())
    if n_outliers == 0:
        return detection_metrics_nan(is_prob=is_prob)

    res = {
        "auprc": average_precision_score(y_true, p_or_score),
    }
    try:
        res["auroc"] = roc_auc_score(y_true, p_or_score)
    except Exception:
        res["auroc"] = np.nan

    idx_desc = np.argsort(p_or_score)[::-1]

    def _recall_at_k(k):
        return _recall_topk_by_rank(y_true, idx_desc, k, n_outliers)

    res["recall_k_oracle"] = _recall_at_k(k_oracle)
    res["recall_k_est"] = _recall_at_k(k_est)

    if log_lik is not None and k_budgets is not None:
        idx_lik_asc = np.argsort(np.asarray(log_lik, dtype=float))
        k_central = int(k_budgets.get("k_central", k_est or 0))
        k_opt = int(k_budgets.get("k_optimistic", k_central))
        k_pess = int(k_budgets.get("k_pessimistic", k_central))
        res["recall_k_lik_oracle"] = _recall_topk_by_rank(y_true, idx_lik_asc, k_oracle, n_outliers)
        res["recall_k_lik_central"] = _recall_topk_by_rank(y_true, idx_lik_asc, k_central, n_outliers)
        res["recall_k_lik_optimistic"] = _recall_topk_by_rank(y_true, idx_lik_asc, k_opt, n_outliers)
        res["recall_k_lik_pessimistic"] = _recall_topk_by_rank(y_true, idx_lik_asc, k_pess, n_outliers)
    else:
        res["recall_k_lik_oracle"] = np.nan
        res["recall_k_lik_central"] = np.nan
        res["recall_k_lik_optimistic"] = np.nan
        res["recall_k_lik_pessimistic"] = np.nan

    if is_prob:
        res["brier"] = brier_score_loss(y_true, p_or_score)
    return res


def recover_beta_original_scale(beta_scaled, scaler):
    beta_scaled = np.asarray(beta_scaled, dtype=float)
    slopes = scaler["y_std"] * beta_scaled[1:] / scaler["x_std"]
    intercept = scaler["y_mean"] + scaler["y_std"] * beta_scaled[0] - np.sum(slopes * scaler["x_mean"])
    return np.concatenate([[intercept], slopes])


def recover_posterior_params_original_scale(samples, scaler):
    """Convert posterior samples from standardized regression scale back to original scale."""
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return {"beta": np.empty((0, 0)), "sigma": np.empty((0,))}

    d_in = len(scaler["x_mean"]) + 1
    beta_scaled = samples[:, :d_in]
    log_scale = samples[:, d_in]

    slopes = scaler["y_std"] * beta_scaled[:, 1:] / scaler["x_std"][None, :]
    intercept = scaler["y_mean"] + scaler["y_std"] * beta_scaled[:, 0] - np.sum(
        slopes * scaler["x_mean"][None, :], axis=1
    )
    beta_orig = np.column_stack([intercept, slopes])
    sigma_orig = scaler["y_std"] * np.exp(log_scale)
    return {"beta": beta_orig, "sigma": sigma_orig}


def summarize_posterior_prediction(samples, x_test_design, scaler):
    d_in = x_test_design.shape[1]
    samples_np = np.asarray(samples)
    beta_scaled_mean = samples_np[:, :d_in].mean(axis=0)
    log_scale_mean = float(samples_np[:, d_in].mean())

    y_pred_scaled = np.asarray(x_test_design) @ beta_scaled_mean
    y_pred = scaler["y_mean"] + scaler["y_std"] * y_pred_scaled

    beta_hat = recover_beta_original_scale(beta_scaled_mean, scaler)
    sigma_hat = scaler["y_std"] * np.exp(log_scale_mean)

    return {
        "y_pred": y_pred,
        "beta_hat": beta_hat,
        "sigma_hat": float(sigma_hat),
    }


def compute_nlpd_test_plugin(
    y_test_raw: np.ndarray,
    y_pred: np.ndarray,
    sigma_hat: float,
) -> float:
    """Plug-in Gaussian NLPD using posterior-mean location and scale."""
    sigma = max(float(sigma_hat), 1e-8)
    return float(
        -np.mean(norm.logpdf(np.asarray(y_test_raw, dtype=float), loc=np.asarray(y_pred, dtype=float), scale=sigma))
    )


def compute_nlpd_test_mc(
    samples: np.ndarray,
    x_test_design: np.ndarray,
    scaler: dict,
    y_test_raw: np.ndarray,
    *,
    joint_with_rho: bool = False,
) -> float:
    """Monte Carlo NLPD: average log posterior predictive density over test points."""
    samples = np.asarray(samples, dtype=float)
    x_test = np.asarray(x_test_design, dtype=float)
    y_test = np.asarray(y_test_raw, dtype=float).reshape(-1)
    d_in = int(x_test.shape[1])

    betas = samples[:, :d_in]
    log_scales = samples[:, d_in]
    y_mean = float(scaler["y_mean"])
    y_std = float(scaler["y_std"])

    mu_scaled = betas @ x_test.T
    mu_raw = y_mean + y_std * mu_scaled
    sigmas = np.maximum(y_std * np.exp(log_scales), 1e-8)[:, None]

    logpdfs = norm.logpdf(y_test[None, :], loc=mu_raw, scale=sigmas)
    log_pred = logsumexp(logpdfs, axis=0) - np.log(float(samples.shape[0]))
    return float(-np.mean(log_pred))


def evaluate_prediction_metrics(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    *,
    sigma_hat: float | None = None,
    samples: np.ndarray | None = None,
    x_test_design: np.ndarray | None = None,
    scaler: dict | None = None,
    joint_with_rho: bool = False,
) -> Dict[str, float]:
    """Point-prediction metrics plus optional plug-in / MC NLPD on raw test responses."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    err = y_pred - y_test
    metrics: Dict[str, float] = {
        "rmse_test": float(np.sqrt(np.mean(err**2))),
        "mae_test": float(np.mean(np.abs(err))),
    }
    if sigma_hat is not None:
        metrics["nlpd_test"] = compute_nlpd_test_plugin(y_test, y_pred, sigma_hat)
    if samples is not None and x_test_design is not None and scaler is not None:
        metrics["nlpd_test_mc"] = compute_nlpd_test_mc(
            samples,
            x_test_design,
            scaler,
            y_test,
            joint_with_rho=joint_with_rho,
        )
    return metrics


def evaluate_regression_metrics(y_pred, y_test, beta_hat, beta_true, sigma_hat, sigma_true):
    y_pred = np.asarray(y_pred)
    y_test = np.asarray(y_test)
    beta_hat = np.asarray(beta_hat)
    beta_true = np.asarray(beta_true)

    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    beta_l2 = float(np.linalg.norm(beta_hat - beta_true))
    beta_rmse = float(np.sqrt(np.mean((beta_hat - beta_true) ** 2)))
    sigma_abs_error = float(abs(float(sigma_hat) - float(sigma_true)))

    return {
        "rmse_test": rmse,
        "beta_l2_error": beta_l2,
        "beta_rmse": beta_rmse,
        "sigma_abs_error": sigma_abs_error,
    }


def summarize_dpd_posterior(samples, eps_true):
    rho = np.asarray(samples[:, -1])
    eps_s = expit(-rho)
    ci95 = np.percentile(eps_s, [2.5, 97.5])
    return {
        "eps_post_mean": float(eps_s.mean()),
        "eps_post_median": float(np.median(eps_s)),
        "eps_post_ci95_low": float(ci95[0]),
        "eps_post_ci95_high": float(ci95[1]),
        "eps_ci95_cover": int(ci95[0] <= eps_true <= ci95[1]),
    }


def contaminate_response_scaled(
    y_clean: np.ndarray,
    eps: float,
    mu_out: float,
    seed: int,
    contam_unit: str = "std",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject response contamination on ``y_clean`` at rate ``eps``.

    contam_unit:
      - "std": shift mean ``mu_out * std(y_clean)`` with Gaussian noise scale ``std(y_clean)``
      - "raw": same as legacy ``y_clean + N(mu_out, 1)`` on contaminated indices
    """
    rng = np.random.default_rng(seed)
    n = len(y_clean)
    mask = rng.random(n) < eps
    y_out = np.asarray(y_clean, dtype=float).copy()
    if not np.any(mask):
        return y_out, mask.astype(bool)

    if contam_unit == "std":
        y_scale = float(np.std(y_clean))
        if y_scale < 1e-12:
            y_scale = 1.0
        y_out[mask] = y_clean[mask] + rng.normal(
            loc=mu_out * y_scale,
            scale=1.0 * y_scale,
            size=int(np.sum(mask)),
        )
    elif contam_unit == "raw":
        y_out[mask] = y_clean[mask] + rng.normal(loc=mu_out, scale=1.0, size=int(np.sum(mask)))
    else:
        raise ValueError(f"Unknown contam_unit: {contam_unit}")

    return y_out, mask.astype(bool)


def evaluate_cleaning_set(removed_idx: np.ndarray, mask_outlier: np.ndarray) -> Dict[str, Any]:
    removed = np.zeros(len(mask_outlier), dtype=bool)
    removed[np.asarray(removed_idx, dtype=int)] = True
    mo = np.asarray(mask_outlier).astype(bool)

    tp = int(np.sum(removed & mo))
    fp = int(np.sum(removed & ~mo))
    fn = int(np.sum(~removed & mo))
    tn = int(np.sum(~removed & ~mo))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr_removed = fp / max(int(np.sum(~mo)), 1)

    return {
        "k_removed": int(np.sum(removed)),
        "retained_fraction": float(1.0 - np.mean(removed)),
        "removed_precision": float(precision),
        "removed_recall": float(recall),
        "removed_f1": float(f1),
        "inlier_false_removal_rate": float(fpr_removed),
        "tp_removed": tp,
        "fp_removed": fp,
        "fn_remaining_outliers": fn,
        "tn_retained_inliers": tn,
    }


def fit_downstream_nll_and_evaluate(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    seed: int,
    chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    target_accept: float = 0.8,
    init_step: float = 1.0,
    scaler: dict | None = None,
    fit_role: str = "DownstreamNLL",
) -> Dict[str, Any]:
    """
    Fit Gaussian NLL (theta-only) posterior and evaluate on held-out raw test responses.

    If ``scaler`` is provided, transform with the fixed Kanamori clean-train scaler
    (Sec.~6.2 alignment). Otherwise recompute train-set mean/std from the passed train data.
    """
    x_train_raw = np.asarray(x_train_raw, dtype=float)
    y_train_raw = np.asarray(y_train_raw, dtype=float)
    x_test_raw = np.asarray(x_test_raw, dtype=float)
    y_test_raw = np.asarray(y_test_raw, dtype=float)

    if scaler is not None:
        x_train, y_train, x_test, scaler_used = apply_fixed_regression_scaler(
            x_train_raw, y_train_raw, x_test_raw, scaler
        )
    else:
        x_train, y_train, x_test, scaler_used = standardize_regression_data(
            x_train_raw, y_train_raw, x_test_raw
        )

    pri = MVNPriorsConfig()
    logprob_fn = logprob_theta_only_nll_regression_from_vector(x_train, y_train, pri)
    d_in = int(x_train.shape[1])
    init_val = jnp.concatenate([jnp.zeros(d_in), jnp.zeros(1)])
    with jax.experimental.enable_x64():
        X64, y64 = jnp.asarray(x_train, dtype=jnp.float64), jnp.asarray(y_train, dtype=jnp.float64)
        sampled = paper_fit(logprob_theta_only_nll_regression_from_vector(X64, y64, pri),
            [x_train, y_train], method=fit_role, model="regression", seed=seed,
            chains=chains, num_warmup=num_warmup, num_samples=num_samples,
            target_accept=target_accept, init_step=init_step, priors=pri,
            names=regression_mcmc_var_names(d_in, False))
    samples = sampled["samples"]

    pred = summarize_posterior_prediction(samples, x_test, scaler_used)
    y_pred = np.asarray(pred["y_pred"])
    sigma = max(float(pred["sigma_hat"]), 1e-8)

    err = y_test_raw - y_pred
    rmse_test = float(np.sqrt(np.mean(err**2)))
    mae_test = float(np.mean(np.abs(err)))
    nlpd_test = compute_nlpd_test_plugin(y_test_raw, y_pred, sigma)
    nlpd_test_mc = compute_nlpd_test_mc(samples, x_test, scaler_used, y_test_raw, joint_with_rho=False)

    return {
        **fit_metadata(sampled),
        **convergence_summary(samples, chains, regression_mcmc_var_names(d_in, False)),
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "nlpd_test": nlpd_test,
        "nlpd_test_mc": nlpd_test_mc,
        "sigma_hat": float(sigma),
        "n_train_clean": int(len(y_train_raw)),
    }


def generate_representative_plots_from_raw(raw, condition, output_dir, student_t_df=4.0):
    """Generate representative regression plots from saved raw arrays."""
    x_train_raw = np.asarray(raw["x_train"])
    y_train_raw = np.asarray(raw["y_train"])
    mask_outlier = np.asarray(raw["mask_outlier"]).astype(bool)
    beta_true = np.asarray(raw["beta_true"], dtype=float)
    sigma_true = float(np.asarray(raw["sigma_true"]).item()) if "sigma_true" in raw else 1.0

    x_train, y_train, _, scaler = standardize_regression_data(x_train_raw, y_train_raw, x_train_raw)

    p_out_dpd = np.asarray(raw["p_out_dpd"])
    samples_dpd = np.asarray(raw["samples_dpd"])
    samples_nll = np.asarray(raw["samples_nll"]) if "samples_nll" in raw else np.empty((0, 0))
    samples_gamma = np.asarray(raw["samples_gamma"]) if "samples_gamma" in raw else np.empty((0, 0))
    samples_student_t = np.asarray(raw["samples_student_t"]) if "samples_student_t" in raw else np.empty((0, 0))
    rho = samples_dpd[:, -1]
    eps_s = expit(-rho)
    k_s = np.clip(np.round(eps_s * float(len(mask_outlier))).astype(int), 0, len(mask_outlier))
    k_est = int(raw["k_est_dpd"]) if "k_est_dpd" in raw else int(k_s.mean())

    method_styles = {
        "DPD": {"samples": samples_dpd, "color": "#4c78a8"},
        "NLL": {"samples": samples_nll, "color": "#2ca02c"},
        "GammaDiv": {"samples": samples_gamma, "color": "#f58518"},
        "StudentT": {"samples": samples_student_t, "color": "#b279a2"},
    }

    fig_xy, axes_xy = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for j, ax in enumerate(axes_xy):
        sc = ax.scatter(
            x_train_raw[:, j],
            y_train_raw,
            c=p_out_dpd,
            cmap="coolwarm",
            edgecolors=np.where(mask_outlier, "black", "none"),
            linewidths=np.where(mask_outlier, 0.8, 0.0),
            s=36,
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
        )
        ax.set_xlabel(f"x{j + 1}")
        ax.set_title(f"x{j + 1} vs y")
        ax.grid(True, alpha=0.25)
    axes_xy[0].set_ylabel("y")
    fig_xy.colorbar(sc, ax=axes_xy, label="DPD outlier probability", shrink=0.9)
    fig_xy.suptitle(f"Representative training data ({condition})")
    fig_xy.subplots_adjust(top=0.84, right=0.88, wspace=0.12)
    fig_xy.savefig(os.path.join(output_dir, f"fig_feature_scatter_prob_{condition}.png"), bbox_inches="tight")
    plt.close(fig_xy)

    pred_dpd = summarize_posterior_prediction(samples_dpd, x_train, scaler)
    fitted_dpd = pred_dpd["y_pred"]
    resid_dpd = y_train_raw - fitted_dpd

    fig_resid, ax_resid = plt.subplots(1, 1, figsize=(6.5, 5))
    sc = ax_resid.scatter(
        fitted_dpd,
        resid_dpd,
        c=p_out_dpd,
        cmap="coolwarm",
        edgecolors=np.where(mask_outlier, "black", "none"),
        linewidths=np.where(mask_outlier, 0.9, 0.0),
        s=42,
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
    )
    ax_resid.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax_resid.set_xlabel("Posterior mean fitted value")
    ax_resid.set_ylabel("Residual")
    ax_resid.set_title(f"DPD residual view ({condition})")
    ax_resid.grid(True, alpha=0.25)
    fig_resid.colorbar(sc, ax=ax_resid, label="DPD outlier probability")
    fig_resid.tight_layout()
    fig_resid.savefig(os.path.join(output_dir, f"fig_residual_prob_{condition}.png"), bbox_inches="tight")
    plt.close(fig_resid)

    flagged = np.zeros_like(mask_outlier, dtype=bool)
    if k_est > 0:
        idx_top = np.argsort(p_out_dpd)[::-1][:k_est]
        flagged[idx_top] = True

    fig_flag, ax_flag = plt.subplots(1, 1, figsize=(6.5, 5))
    ax_flag.scatter(fitted_dpd[~flagged], resid_dpd[~flagged], c="lightgray", s=28, alpha=0.75, label="not flagged")
    ax_flag.scatter(fitted_dpd[flagged], resid_dpd[flagged], c="#d62728", s=42, alpha=0.9, label="DPD flagged")
    ax_flag.scatter(
        fitted_dpd[mask_outlier],
        resid_dpd[mask_outlier],
        facecolors="none",
        edgecolors="black",
        s=64,
        linewidths=1.0,
        label="true outlier",
    )
    ax_flag.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax_flag.set_xlabel("Posterior mean fitted value")
    ax_flag.set_ylabel("Residual")
    ax_flag.set_title(f"DPD estimated outlier flags ({condition}, k_est={k_est})")
    ax_flag.grid(True, alpha=0.25)
    ax_flag.legend(loc="best", frameon=True)
    fig_flag.tight_layout()
    fig_flag.savefig(os.path.join(output_dir, f"fig_residual_dpd_flagged_{condition}.png"), bbox_inches="tight")
    plt.close(fig_flag)

    param_names = ["Intercept"] + [f"beta{j + 1}" for j in range(len(beta_true) - 1)] + ["sigma"]
    true_values = np.concatenate([beta_true, [sigma_true]])
    n_panels = len(param_names)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig_post, axes_post = plt.subplots(nrows, ncols, figsize=(11, 3.8 * nrows))
    axes_post = np.atleast_1d(axes_post).ravel()

    for method, spec in method_styles.items():
        transformed = recover_posterior_params_original_scale(spec["samples"], scaler)
        if transformed["beta"].size == 0:
            continue
        param_arrays = [transformed["beta"][:, j] for j in range(transformed["beta"].shape[1])] + [transformed["sigma"]]
        for ax, pname, true_val, arr in zip(axes_post, param_names, true_values, param_arrays):
            ax.hist(arr, bins=40, density=True, histtype="step", linewidth=1.7, color=spec["color"], label=method)
            ax.axvline(true_val, color="black", linestyle="--", linewidth=1.2)
            ax.set_title(pname)
            ax.grid(True, alpha=0.2)

    for ax in axes_post[n_panels:]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=spec["color"], lw=1.8, label=method) for method, spec in method_styles.items()
    ]
    handles.append(plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="True value"))
    fig_post.legend(handles=handles, loc="upper center", ncol=min(5, len(handles)), frameon=True)
    fig_post.suptitle(f"Posterior distributions of model parameters ({condition})")
    fig_post.tight_layout(rect=(0, 0, 1, 0.94))
    fig_post.savefig(os.path.join(output_dir, f"fig_posterior_params_{condition}.png"), bbox_inches="tight")
    plt.close(fig_post)

    fig_eps, ax_eps = plt.subplots(1, 1, figsize=(6.5, 5))
    ax_eps.hist(eps_s, bins=40, density=True, alpha=0.7, color="#4c78a8", edgecolor="white")
    eps_true = float(mask_outlier.mean())
    ax_eps.axvline(eps_true, color="black", linestyle="--", linewidth=1.2, label="True epsilon")
    ax_eps.axvline(float(np.mean(eps_s)), color="#d62728", linestyle="-", linewidth=1.4, label="Posterior mean")
    ax_eps.set_xlabel("epsilon")
    ax_eps.set_ylabel("Density")
    ax_eps.set_title(f"DPD posterior of epsilon ({condition})")
    ax_eps.grid(True, alpha=0.25)
    ax_eps.legend(loc="best", frameon=True)
    fig_eps.tight_layout()
    fig_eps.savefig(os.path.join(output_dir, f"fig_posterior_epsilon_{condition}.png"), bbox_inches="tight")
    plt.close(fig_eps)
