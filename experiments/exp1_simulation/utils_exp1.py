import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t_distribution
import jax.scipy as jsp
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, average_precision_score, brier_score_loss
)
from hbayes.posteriors import (
    MVNPriorsConfig,
    MVNGibbsConfig,
    MVNDiagParams,
    mvn_diag_log_pdf,
    log_prior_theta_mvn,
    logprob_joint_mvn_from_vector,
    logprob_theta_only_nll_mvn_from_vector,
)
from hbayes.sampling import run_nuts_adapted
from hbayes.gamma_sampling import fit as paper_fit
from hbayes.gamma_posterior import logprob_density as logprob_theta_only_gamma_div_mvn_from_vector
from hbayes.experiment_temperature import (TEMPERATURE_VERSION, covariance_temperature,
    response_temperature, positive_temperature, fit_metadata, nuts_summary, convergence_summary,
    configuration, begin_run, record_completion)


# (1) simulate_dataset
def simulate_dataset(n, eps, scenario, seed, base_mean=(0.0, 0.0), base_scale=(1.0, 1.0),
                     out_dist=None, out_mean=None, out_scale=None):
    """
    Simulate contaminated dataset based on scenario A (Gaussian shift) or B (Student-t).
    """
    from hbayes.simulate import ContamConfig, simulate_contaminated_gaussian
    key = jax.random.PRNGKey(seed)
    
    if scenario == "A":
        # Gross outliers: N((5,5), I) by default
        cfg = ContamConfig(
            n=n,
            mean=base_mean,
            scale=base_scale,
            contam_ratio=eps,
            out_dist=out_dist if out_dist is not None else "gaussian",
            out_mean=out_mean if out_mean is not None else (5.0, 5.0),
            out_scale=out_scale if out_scale is not None else (1.0, 1.0)
        )
    elif scenario == "B":
        # Heavy-tail: t(df=1.5, scale=5.0)
        # Note: scenario B typically ignores out_mean/out_scale in original code but we can override if out_dist is set
        cfg = ContamConfig(
            n=n,
            mean=base_mean,
            scale=base_scale,
            contam_ratio=eps,
            out_dist=out_dist if out_dist is not None else "t",
            t_df=1.5, # Fixed for now or could be parametrized
            t_scale=5.0 # Fixed for now
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    x, mask_outlier = simulate_contaminated_gaussian(key, cfg)
    # Convert to numpy for easier handling downstream except where JAX is needed
    x = np.asarray(x)
    y_outlier_mask = np.asarray(mask_outlier).astype(bool)
    
    return x, y_outlier_mask


# (2) fit_methods
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


def _student_t_logpdf(y, mu, log_scale, nu):
    scale = jnp.exp(log_scale)
    z = (y - mu) / scale
    c0 = jsp.special.gammaln((nu + 1.0) / 2.0) - jsp.special.gammaln(nu / 2.0)
    c1 = -0.5 * jnp.log(nu * jnp.pi) - jnp.log(scale)
    c2 = -((nu + 1.0) / 2.0) * jnp.log1p((z * z) / nu)
    return c0 + c1 + c2


def mvn_diag_student_t_log_pdf(params: MVNDiagParams, x: jnp.ndarray, nu: float) -> jnp.ndarray:
    mu = params.mean
    log_scales = params.log_scales
    return jnp.sum(_student_t_logpdf(x, mu[None, :], log_scales[None, :], nu), axis=-1)


def logprob_theta_only_student_t_mvn_from_vector(x: jnp.ndarray, priors: MVNPriorsConfig, nu: float = 4.0):
    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        params = MVNDiagParams(mean=z[:d], log_scales=z[d : 2 * d])
        ll = jnp.sum(mvn_diag_student_t_log_pdf(params, x, nu))
        return log_prior_theta_mvn(params, priors) + ll

    return _logprob


def fit_methods(x, seed, chains=4, num_warmup=1000, num_samples=1000, gamma=0.5,
                standardize=True, eta_mode="covdet", holder_phi="dpd", holder_kappa=None,
                student_t_df=4.0, target_accept=0.8, init_step=1.0, eta_gamma_div=1.0):
    """Fit proposals and baselines in the original data coordinates.

    standardize is retained for API compatibility; covariance temperature is
    always computed from the same raw observations used by the likelihood.
    """
    if eta_mode == "covdet":
        eta_val, cov_det = covariance_temperature(x, gamma)
    elif eta_mode == "fixed":
        eta_val, cov_det = 1.0, float("nan")
    else:
        raise ValueError(f"Unknown density eta_mode: {eta_mode}")
    eta_gamma_div = positive_temperature(eta_gamma_div)
    pri = MVNPriorsConfig()
    results = {}
    with jax.experimental.enable_x64():
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        d = x_jax.shape[1]
        names = [f"mu_{i+1}" for i in range(d)] + [f"log_scale_{i+1}" for i in range(d)]
        specs = resolve_proposal_method_specs(holder_phi=holder_phi, holder_kappa=holder_kappa)
        for offset, (name, spec) in enumerate(specs.items()):
            config = MVNGibbsConfig(gamma=gamma, eta=eta_val, holder_phi=spec["holder_phi"],
                                   holder_kappa=spec["holder_kappa"], priors=pri)
            result = paper_fit(logprob_joint_mvn_from_vector(x_jax, config), [x], method=name,
                model="density", seed=seed+1000*offset, chains=chains,
                num_warmup=num_warmup, num_samples=num_samples, target_accept=target_accept,
                init_step=init_step, priors=pri, eta=eta_val, gamma=gamma,
                phi=spec["holder_phi"], kappa=spec["holder_kappa"], floor=None,
                names=names+["rho"])
            result.update(eta=eta_val, eta_mode=eta_mode, cov_det=cov_det,
                          temperature_version=TEMPERATURE_VERSION, **spec)
            results[name] = result
        config_gamma = MVNGibbsConfig(gamma=gamma, eta=eta_gamma_div,
                                     holder_phi="dpd", holder_kappa=1., priors=pri)
        targets = {
            "standard": ("NLL", logprob_theta_only_nll_mvn_from_vector(x_jax, pri), seed+101, 1., "likelihood"),
            "gamma": ("GammaDiv", logprob_theta_only_gamma_div_mvn_from_vector(x_jax, config_gamma), seed+202, eta_gamma_div, "gamma_div_fixed"),
            "student_t": ("StudentT", logprob_theta_only_student_t_mvn_from_vector(x_jax, pri, nu=student_t_df), seed+303, 1., "likelihood"),
        }
        for key, (name, target, fit_seed, eta, mode) in targets.items():
            result = paper_fit(target, [x], method=name, model="density", seed=fit_seed,
                chains=chains, num_warmup=num_warmup, num_samples=num_samples,
                target_accept=target_accept, init_step=init_step, priors=pri,
                eta=eta, gamma=gamma, student_t_df=student_t_df, names=names)
            result.update(eta=eta, eta_mode=mode, temperature_version=TEMPERATURE_VERSION)
            results[key] = result
    return results


# (3) compute_dpd_outlier_prob
def compute_dpd_outlier_prob(samples_joint, x, S_eval_max=400, seed=42):
    """
    Compute outlier probability p_i and epsilon estimates from Joint DPD samples.
    """
    S_full = samples_joint.shape[0]
    n = x.shape[0]
    
    # Subsampling
    if S_full > S_eval_max:
        rng = np.random.default_rng(seed)
        eval_indices = np.sort(rng.choice(S_full, size=S_eval_max, replace=False))
    else:
        eval_indices = np.arange(S_full)
    
    samples_eval = samples_joint[eval_indices]
    S_eval = len(eval_indices)
    
    # Unpack parameters
    mu_x = samples_eval[:, 0]
    mu_y = samples_eval[:, 1]
    log_sig_x = samples_eval[:, 2]
    log_sig_y = samples_eval[:, 3]
    rho = samples_eval[:, 4]
    
    # Epsilon estimate r = 1 - c
    r_arr = expit(-np.asarray(rho, dtype=np.float64))
    
    # Calculate K (number of outliers per sample)
    K_arr = np.clip(np.round(r_arr * float(n)).astype(int), 0, n)
    
    points = np.asarray(x, dtype=np.float64)
    means = np.asarray(samples_eval[:, :2], dtype=np.float64)
    ls = np.asarray(samples_eval[:, 2:4], dtype=np.float64)
    residual = (points[None, :, :] - means[:, None, :]) / np.exp(ls[:, None, :])
    ll_matrix = -.5*np.sum(np.log(2*np.pi)+2*ls[:, None, :]+residual**2, axis=2)

    # Compute flags (1 if in bottom-K likelihoods)
    flags = np.zeros((S_eval, n), dtype=np.uint8)
    for i in range(S_eval):
        k = int(K_arr[i])
        if k > 0:
            # Rank by log-likelihood ascending (smallest ll => outlier)
            idx = np.argsort(ll_matrix[i])[:k] 
            flags[i, idx] = 1
            
    p_out = flags.mean(axis=0)
    
    return {
        "p_out": p_out,
        "eps_samples": r_arr,
        "k_samples": K_arr,
        "k_mean": float(K_arr.mean()),
        "k_p05": float(np.percentile(K_arr, 5)),
        "k_p95": float(np.percentile(K_arr, 95))
    }


# (4) score_baseline
def score_baseline(theta_samples, x, likelihood="gaussian", student_t_df=4.0, S_eval_max=400, seed=42):
    """
    Compute anomaly score for baseline methods using negative LPPD.
    score_i = - log( 1/S sum_s p(x_i | theta_s) )
    """
    S_full = theta_samples.shape[0]
    n = x.shape[0]
    
    if S_full > S_eval_max:
        rng = np.random.default_rng(seed)
        eval_indices = np.sort(rng.choice(S_full, size=S_eval_max, replace=False))
    else:
        eval_indices = np.arange(S_full)
        
    samples = np.asarray(theta_samples[eval_indices], dtype=np.float64)
    points = np.asarray(x, dtype=np.float64)
    d = points.shape[1]
    mu, ls = samples[:, :d], samples[:, d:2*d]
    residual = (points[None, :, :]-mu[:, None, :])/np.exp(ls[:, None, :])
    if likelihood == "gaussian":
        ll = -.5*np.sum(np.log(2*np.pi)+2*ls[:, None, :]+residual**2, axis=2)
    elif likelihood == "student_t":
        ll = np.sum(student_t_distribution.logpdf(residual, df=student_t_df)-ls[:, None, :], axis=2)
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    return -(logsumexp(ll, axis=0)-np.log(len(samples)))



# (5) evaluate_metrics
def evaluate_metrics(p_or_score, y_true, k_oracle, k_est=None, is_prob=False):
    """
    Compute detection metrics.
    is_prob: True if input is probability p_i (for DPD), False if score (for Baselines).
    """
    y_true = np.asarray(y_true).astype(int)
    n_outliers = np.sum(y_true)
    
    res = {}
    
    # Only if there are positive classes or we define behavior
    if n_outliers == 0:
        return {
            "auprc": np.nan, "auroc": np.nan,
            "recall_k_oracle": np.nan, "recall_k_est": np.nan,
            "brier": 0.0 if is_prob else np.nan,
            "ece": 0.0 if is_prob else np.nan
        }

    # AUPRC / AUROC
    res["auprc"] = average_precision_score(y_true, p_or_score)
    try:
        res["auroc"] = roc_auc_score(y_true, p_or_score)
    except:
        res["auroc"] = np.nan
        
    # Top-k Recall
    # Sort indices by score descending (higher p or score => more likely outlier)
    idx_desc = np.argsort(p_or_score)[::-1]
    
    def calc_recall(k):
        if k is None or k <= 0: return 0.0
        top_k_idx = idx_desc[:int(k)]
        hits = np.sum(y_true[top_k_idx])
        return float(hits / n_outliers)
    
    res["recall_k_oracle"] = calc_recall(k_oracle)
    if k_est is not None:
        res["recall_k_est"] = calc_recall(k_est)
        
    # Calibration (Only for DPD probability)
    if is_prob:
        # Brier
        res["brier"] = brier_score_loss(y_true, p_or_score)
        
        # ECE (dynamic bins)
        n = len(y_true)
        n_bins = int(np.floor(n ** (1.0/3.0)))
        n_bins = max(5, n_bins)
        
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(p_or_score, bins, right=True)
        
        ece = 0.0
        bin_centers = []
        bin_frac_pos = []
        bin_counts = []
        
        for b in range(1, n_bins + 1):
            mask = bin_indices == b
            count = np.sum(mask)
            if count > 0:
                avg_pred = np.mean(p_or_score[mask])
                frac_pos = np.mean(y_true[mask])
                ece += (count / n) * np.abs(avg_pred - frac_pos)
                bin_centers.append(avg_pred)
                bin_frac_pos.append(frac_pos)
                bin_counts.append(count)
        
        res["ece"] = ece
        
        # Include calibration details for plotting
        res["calib_data"] = {
            "bin_centers": bin_centers,
            "bin_frac_pos": bin_frac_pos,
            "bin_counts": bin_counts,
            "ece": ece,
            "brier": res["brier"]
        }
        
    return res


def plot_reliability_diagram(ax, calib_data, title="Reliability Diagram"):
    """Plot reliability diagram on given axis."""
    centers = calib_data["bin_centers"]
    fracs = calib_data["bin_frac_pos"]
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    
    # Plot bins
    ax.plot(centers, fracs, "s-", label="Model")
    
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    
    # Add metrics to plot
    textstr = f"ECE = {calib_data['ece']:.3f}\nBrier = {calib_data['brier']:.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

