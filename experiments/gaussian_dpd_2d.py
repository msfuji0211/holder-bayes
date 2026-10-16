from __future__ import annotations

import argparse
from dataclasses import dataclass
import os

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as ss
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, average_precision_score, brier_score_loss,
    recall_score, roc_curve
)
import pandas as pd
import json

from hbayes.simulate import ContamConfig, simulate_contaminated_gaussian
from hbayes.posteriors import (
    MVNPriorsConfig,
    MVNGibbsConfig,
    logprob_joint_mvn_from_vector,
    logprob_theta_only_nll_mvn_from_vector,
    logprob_theta_only_gamma_div_mvn_from_vector,
    log_prior_theta_mvn,
    log_prior_c_from_rho_mvn,
)
from hbayes.dpd import mvn_diag_A_gamma, mvn_diag_B_gamma
from hbayes.sampling import NutsConfig, run_nuts, run_nuts_adapted

# Helper for MMD computation
def compute_mmd_gaussian_kernel_loss(
    params: MVNDiagParams, c: jnp.ndarray, x: jnp.ndarray, length_scale: float
) -> jnp.ndarray:
    """
    MMD^2 loss between empirical distribution P_n and model Q = c*P_theta.
    
    L(theta, c) = || \mu(P_n) - \mu(Q) ||_H^2
                = E[k(x,x')] - 2 E[k(x, y)] + E[k(y, y')]
                = const - 2 c/n \sum_i E_{y~P_theta}[k(x_i, y)] + c^2 E_{y,y'~P_theta}[k(y, y')]
    
    Using Gaussian kernel k(u,v) = exp(-||u-v||^2 / (2l^2)).
    
    1) Cross term: E_{y~N(mu, Sig)}[k(x_i, y)]
       Convolution of N(x_i, 0) and N(mu, Sig) with kernel N(0, l^2 I).
       Result is N(x_i | mu, Sig + l^2 I) * (2*pi*l^2)^(d/2).
       Actually, standard formula for E[k(x,y)] where y~N(m,S) and k is Gaussian(l):
       int exp(-|x-y|^2/2l^2) N(y|m,S) dy
       = det(I + S/l^2)^{-1/2} * exp( -1/2 (x-m)^T (S+l^2 I)^-1 (x-m) )
    
    2) Model term: E_{y,y'~N(mu,Sig)}[k(y,y')]
       = det(2 I + 2 S/l^2)^{-1/2} ?? No, let's use closed form.
       Standard result for Gaussian kernel expectation over two Gaussians:
       E[k(y,y')] = det(I + 2S/l^2)^{-1/2}
    """
    d = params.mean.shape[0]
    n = x.shape[0]
    mu = params.mean
    sig2 = jnp.exp(2.0 * params.log_scales) # diagonal variances
    
    # 1. Model self-similarity term: E[k(y,y')]
    # det(I + 2*Sig/l^2) = prod(1 + 2*sig2_j / l^2)
    det_term_model = jnp.prod(1.0 + 2.0 * sig2 / (length_scale**2))
    term_yy = det_term_model ** (-0.5)
    
    # 2. Cross term: 1/n sum_i E[k(x_i, y)]
    # E[k(x,y)] = det(I + Sig/l^2)^(-1/2) * exp( -0.5 * (x-mu)^T (Sig + l^2 I)^-1 (x-mu) )
    # Diagonal covariance simplifies this.
    var_eff = sig2 + length_scale**2  # (d,)
    det_term_cross = jnp.prod(1.0 + sig2 / (length_scale**2)) # this is not quite right pre-factor
    # Actually, let's look at the density interpretation:
    # int exp(-|x-y|^2/2l^2) N(y|m,S) dy = (2*pi*l^2)^(d/2) * N(x | m, S + l^2 I)
    
    # log N(x|m, V_eff)
    diff = x - mu # (n, d)
    # Mahalanobis term per point: sum_j (x_ij - mu_j)^2 / var_eff_j
    mahal = jnp.sum((diff**2) / var_eff, axis=1) # (n,)
    log_det_eff = jnp.sum(jnp.log(var_eff))
    log_prob_cross = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det_eff + mahal)
    
    prob_cross = jnp.exp(log_prob_cross)
    # The kernel integral is (2*pi*l^2)^(d/2) * N(...)
    norm_factor = (2.0 * jnp.pi * length_scale**2) ** (0.5 * d)
    
    # Average over data
    term_xy = jnp.mean(prob_cross) * norm_factor
    
    # MMD^2 (ignoring constant data-data term)
    # L = c^2 * term_yy - 2 * c * term_xy
    return (c**2) * term_yy - 2.0 * c * term_xy


def logprob_joint_mmd_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig, length_scale: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Joint MMD-Bayes log posterior.
    
    Posterior \propto exp( - n * eta * MMD^2 ).
    """
    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
    n = float(x.shape[0])

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        # z: [mu... (d), log_scale... (d), rho]
        mean = z[:d]
        log_scales = z[d : 2 * d]
        rho = z[-1]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        
        c = jax.nn.sigmoid(rho)
        
        loss = compute_mmd_gaussian_kernel_loss(params, c, x, length_scale)
        
        lp_theta = log_prior_theta_mvn(params, config.priors)
        lp_c = log_prior_c_from_rho_mvn(rho, config.priors)
        
        return lp_theta + lp_c - n * config.eta * loss

    return _logprob


# Helper for KSD computation (IMQ kernel) with auto-diff
def compute_ksd_imq_loss(
    params: MVNDiagParams, x: jnp.ndarray, gamma_ker: float = 0.5, sigma_ker: float = 1.0
) -> jnp.ndarray:
    """
    Compute KSD^2 estimate (V-statistic) for MVN Diag model and IMQ kernel using auto-diff.
    Incorporates weight function w(x) to improve robustness.
    
    KSD^2 = 1/n^2 sum_{i,j} u_p(x_i, x_j)
    u_p(x, y) is the Stein kernel.
    
    IMQ Kernel: k_base(x, y) = (1 + ||x-y||^2 / (2*sigma_ker^2))^(-gamma_ker)
    Weight: w(x) = (1 + ||x - mu||^2 / (2*sigma_ker^2))^(-gamma_ker)  (using model mean)
    """
    n, d = x.shape
    mu = params.mean
    inv_sig = jnp.exp(-2.0 * params.log_scales) # (d,) vector
    
    # Define score function s(x) = grad_x log p(x)
    def score_fn(val):
        diff = val - mu
        return -diff * inv_sig
    
    # Define weighted kernel function k(x, y)
    def kernel_fn(x_i, y_i):
        # Base IMQ kernel
        diff = x_i - y_i
        sq_dist = jnp.sum(diff**2)
        base = 1.0 + sq_dist / (2.0 * sigma_ker**2)
        k_val = base ** (-gamma_ker)
        
        # Robust weights w(x), w(y)
        # Using similar decay as kernel based on distance from model mean
        # w(z) = (1 + ||z - mu||^2 / (2*sigma^2))^(-gamma)
        dist_x = jnp.sum((x_i - mu)**2)
        dist_y = jnp.sum((y_i - mu)**2)
        w_x = (1.0 + dist_x / (2.0 * sigma_ker**2)) ** (-5.0)
        w_y = (1.0 + dist_y / (2.0 * sigma_ker**2)) ** (-5.0)
        
        return k_val * w_x * w_y

    # Stein operator u_p(x, y) for a pair
    # u_p(x,y) = s(x)^T s(y) k(x,y) + s(x)^T grad_y k(x,y) + s(y)^T grad_x k(x,y) + tr(grad_x grad_y k(x,y))
    def u_statistic(x_i, y_i):
        s_x = score_fn(x_i)
        s_y = score_fn(y_i)
        
        k_val = kernel_fn(x_i, y_i)
        
        # Gradients
        # grad_x_k: (d,)
        grad_x_k = jax.grad(kernel_fn, argnums=0)(x_i, y_i)
        grad_y_k = jax.grad(kernel_fn, argnums=1)(x_i, y_i)
        
        # Trace term: tr(grad_x grad_y k(x,y))
        # Instead of full Jacobian, we compute only diagonal entries.
        # tr(H) = sum_j (d/dy_j) (d/dx_j) k(x,y)
        # We can use forward-mode AD (jvp) to compute directional derivatives efficiently
        # or just loop over d since d is small (2).
        
        def diag_hessian_term(idx):
            # Computes d/dy_idx ( d/dx_idx k(x,y) )
            # inner: d/dx_idx k(x,y) -> scalar
            def d_dx_i(y_val):
                # We need gradient w.r.t x, evaluated at fixed x_i, varying y_val?
                # No, we need d/dy [ d/dx k(x,y) ]
                # Let's define g(x, y) = d/dx k(x,y)
                # We want d/dy g(x,y)
                # g_i(x,y) is scalar.
                return jax.grad(kernel_fn, argnums=0)(x_i, y_val)[idx]
            
            return jax.grad(d_dx_i)(y_i)[idx]

        # For d=2, a simple loop or sum is fast. 
        # Using vmap over indices 0..d-1
        indices = jnp.arange(d)
        trace_term = jnp.sum(jax.vmap(diag_hessian_term)(indices))
        
        term1 = jnp.dot(s_x, s_y) * k_val
        term2 = jnp.dot(s_x, grad_y_k)
        term3 = jnp.dot(s_y, grad_x_k)
        
        return term1 + term2 + term3 + trace_term

    # Vectorize over all pairs
    # vmap over x (rows) and y (cols)
    # Using vmap directly might be memory intensive for large n, but n=500 is fine.
    u_mat = jax.vmap(jax.vmap(u_statistic, (None, 0)), (0, None))(x, x)
    
    return jnp.mean(u_mat)


def logprob_joint_ksd_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig, sigma_ker: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Joint KSD-Bayes log posterior.
    
    Posterior \propto exp( - n * eta * KSD^2 ).
    """
    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
    n = float(x.shape[0])

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        # z: [mu... (d), log_scale... (d), rho]
        mean = z[:d]
        log_scales = z[d : 2 * d]
        rho = z[-1]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        
        # c is unused in KSD score function!
        # c = jax.nn.sigmoid(rho)
        
        loss = compute_ksd_imq_loss(params, x, gamma_ker=0.5, sigma_ker=sigma_ker)
        
        lp_theta = log_prior_theta_mvn(params, config.priors)
        lp_c = log_prior_c_from_rho_mvn(rho, config.priors)
        
        return lp_theta + lp_c - n * config.eta * loss

    return _logprob
from hbayes.diagnostics import arviz_rhat, arviz_ess
from hbayes.models import MVNDiagParams, mvn_diag_log_pdf


from typing import Callable

def logprob_joint_nll_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Joint NLL (unnormalized KL) log posterior.
    
    Loss = c - log(c) - (1/n)*sum(log p(x)).
    Log Post = log_prior(theta) + log_prior(c) - n*eta*Loss.
    """
    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
    n = float(x.shape[0])

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        # z: [mu... (d), log_scale... (d), rho]
        mean = z[:d]
        log_scales = z[d : 2 * d]
        rho = z[-1]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        
        c = jax.nn.sigmoid(rho)
        log_c = jnp.log(jnp.clip(c, 1e-12, 1.0))
        
        # log likelihood per point
        ll_per_point = mvn_diag_log_pdf(params, x) # (n,)
        mean_ll = jnp.mean(ll_per_point)
        
        # Loss for unnormalized KL
        loss = c - log_c - mean_ll
        
        lp_theta = log_prior_theta_mvn(params, config.priors)
        lp_c = log_prior_c_from_rho_mvn(rho, config.priors)
        
        return lp_theta + lp_c - n * config.eta * loss

    return _logprob


def logprob_joint_gamma_div_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Joint Gamma-Divergence log posterior.
    
    Based on Jones et al. (2001):
      Loss = 1/(1+gamma) * log(A) - 1/gamma * log(B)
    
    Note: Under this definition, the scaling factor 'c' cancels out perfectly.
    Therefore, the posterior of 'c' will be identical to its prior.
    """
    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
    n = float(x.shape[0])
    gamma = config.gamma

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        # z: [mu... (d), log_scale... (d), rho]
        mean = z[:d]
        log_scales = z[d : 2 * d]
        rho = z[-1]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        
        # c is used for prior calculation but does not affect the likelihood (loss)
        # c = jax.nn.sigmoid(rho) 
        
        a_val = mvn_diag_A_gamma(params, gamma)
        b_val = mvn_diag_B_gamma(params, x, gamma)
        
        log_a = jnp.log(a_val + 1e-30)
        log_b = jnp.log(b_val + 1e-30)
        
        # Correct Gamma-Divergence loss (c cancels out)
        loss = (1.0 / (1.0 + gamma)) * log_a - (1.0 / gamma) * log_b
        
        lp_theta = log_prior_theta_mvn(params, config.priors)
        lp_c = log_prior_c_from_rho_mvn(rho, config.priors)
        
        return lp_theta + lp_c - n * config.eta * loss

    return _logprob


def compute_calibration_metrics(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """
    Compute ECE and Brier score, and return binning details for reliability diagram.
    """
    # Brier score
    brier = brier_score_loss(y, p)
    
    # ECE and Reliability Diagram
    # Define bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p, bins, right=True)
    
    bin_centers = []
    bin_frac_pos = []
    bin_counts = []
    ece = 0.0
    n = len(y)
    
    for b in range(1, n_bins + 1):
        # Mask for current bin (indices are 1-based from digitize)
        mask = bin_indices == b
        count = np.sum(mask)
        if count > 0:
            avg_pred = np.mean(p[mask])
            frac_pos = np.mean(y[mask])
            bin_centers.append(avg_pred)
            bin_frac_pos.append(frac_pos)
            bin_counts.append(count)
            ece += (count / n) * np.abs(avg_pred - frac_pos)
        else:
            # Empty bin (optional: add placeholders or skip)
            pass
            
    return {
        "brier": brier,
        "ece": ece,
        "bin_centers": np.array(bin_centers),
        "bin_frac_pos": np.array(bin_frac_pos),
        "bin_counts": np.array(bin_counts)
    }

def plot_reliability_diagram(ax, calib_metrics, title="Reliability Diagram"):
    """Plot reliability diagram on given axis."""
    centers = calib_metrics["bin_centers"]
    fracs = calib_metrics["bin_frac_pos"]
    counts = calib_metrics["bin_counts"]
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    
    # Plot bins
    ax.plot(centers, fracs, "s-", label="Model")
    
    # Optional: histogram of counts on secondary axis or just size?
    # Let's just stick to standard reliability diagram
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add metrics to plot
    textstr = f"ECE = {calib_metrics['ece']:.3f}\nBrier = {calib_metrics['brier']:.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

@dataclass
class Expt2DConfig:
    n: int = 500
    mean_true: tuple[float, float] = (0.0, 0.0)
    scales_true: tuple[float, float] = (1.0, 1.0)
    contam_ratio: float = 0.2
    gamma: float = 0.5
    eta: float = 1.0
    seed: int = 123


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--contam", type=float, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--standardize", action="store_true")
    # Outlier options
    parser.add_argument("--outlier", type=str, choices=["t", "gaussian"], default="gaussian")
    parser.add_argument("--t_df", type=float, default=None)
    parser.add_argument("--t_scale", type=float, default=None)
    parser.add_argument("--out_mean_x", type=float, default=None)
    parser.add_argument("--out_mean_y", type=float, default=None)
    parser.add_argument("--out_scale_x", type=float, default=None)
    parser.add_argument("--out_scale_y", type=float, default=None)
    # Eta
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--eta_mode", type=str, choices=["fixed", "covdet"], default="fixed")
    # Sampling controls
    parser.add_argument("--num_warmup", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--step_size", type=float, default=0.1, help="step size for non-adapted NUTS")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--adapted", action="store_true", help="use window adaptation for joint sampler")
    parser.add_argument("--target_accept", type=float, default=0.8)
    parser.add_argument("--init_step", type=float, default=1.0)
    parser.add_argument("--load_samples", type=str, default=None, help="Path to .npz file to load samples from (skips sampling)")
    parser.add_argument("--save_samples", type=str, default="outputs/samples_2d.npz", help="Path to save samples to")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--output_suffix", type=str, default=None, help="Suffix for output files (overrides automatic suffix)")

    args = parser.parse_args()

    cfg = Expt2DConfig()
    if args.gamma is not None:
        cfg.gamma = float(args.gamma)
    if args.contam is not None:
        cfg.contam_ratio = float(args.contam)
    if args.n is not None:
        cfg.n = int(args.n)

    key = jax.random.PRNGKey(cfg.seed)
    chains = int(args.chains)
    
    # Variables to be populated either by loading or sampling
    x = None
    mask_outlier = None # True if outlier
    samples_joint = None
    samples_nll = None
    samples_gam = None

    if args.load_samples is not None and os.path.exists(args.load_samples):
        print(f"Loading samples and data from {args.load_samples}...")
        data = np.load(args.load_samples)
        x = jnp.array(data["x"])
        if "mask" in data:
            mask_outlier = jnp.array(data["mask"], dtype=bool)
        else:
            # If mask not saved, assume all false or unknown
            mask_outlier = jnp.zeros(x.shape[0], dtype=bool)
            
        samples_joint = jnp.array(data["joint"])

        # Optional: load theta-only baselines if present (new format)
        if "nll" in data:
            samples_nll = jnp.array(data["nll"])
        elif "theta_nll" in data:
            samples_nll = jnp.array(data["theta_nll"])
        elif "jnll" in data:
            # Backward compatibility: old files stored "Joint NLL" (different objective, includes rho).
            # We do NOT auto-convert it to Gaussian NLL.
            print("Warning: loaded file contains 'jnll' (Joint NLL) but not 'nll' (Gaussian NLL). Will resample Gaussian NLL.")

        if "gam" in data:
            samples_gam = jnp.array(data["gam"])
        elif "gamma_div" in data:
            samples_gam = jnp.array(data["gamma_div"])
        
        # Recalculate basic stats needed for plotting if not in cfg
        # (Assuming cfg params like eta might be different, but we use the loaded x)
        pass
    else:
        # Simulate 2D contaminated data
        contam_cfg = ContamConfig(
            n=cfg.n,
            mean=cfg.mean_true,
            scale=cfg.scales_true,
            contam_ratio=cfg.contam_ratio,
            out_dist=args.outlier,
        )
        if args.t_df is not None:
            contam_cfg.t_df = float(args.t_df)
        if args.t_scale is not None:
            contam_cfg.t_scale = float(args.t_scale)
        if args.out_mean_x is not None and args.out_mean_y is not None:
            contam_cfg.out_mean = (float(args.out_mean_x), float(args.out_mean_y))
        if args.out_scale_x is not None and args.out_scale_y is not None:
            contam_cfg.out_scale = (float(args.out_scale_x), float(args.out_scale_y))

        x, mask_outlier = simulate_contaminated_gaussian(key, contam_cfg)  # (n,2), (n,)

    # Pre-standardization covariance determinant
    n_obs = float(x.shape[0])
    denom = jnp.maximum(n_obs - 1.0, 1.0)
    x_centered = x - jnp.mean(x, axis=0)
    cov = (x_centered.T @ x_centered) / denom
    cov_det = float(jnp.linalg.det(cov))

    # Optional standardization per-dimension (for eta calculation mostly)
    if args.standardize:
        std_vec = jnp.sqrt(jnp.maximum(jnp.var(x_centered, axis=0, ddof=1), 1e-12))
        x_std = x_centered / std_vec
    else:
        x_std = x

    # Eta setting
    if args.eta is not None:
        cfg.eta = float(args.eta)
    else:
        if args.standardize and args.eta_mode == "covdet":
            # Compute cov_det on standardized data for eta calibration
            x_centered_new = x_std - jnp.mean(x_std, axis=0)
            cov_new = (x_centered_new.T @ x_centered_new) / denom
            cov_det_new = float(jnp.linalg.det(cov_new))
            cfg.eta = (float(max(cov_det_new, 1e-12)) ** cfg.gamma)

    # Define output suffix based on contamination ratio or user argument
    if args.output_suffix is not None:
        out_suffix = args.output_suffix
    else:
        out_suffix = f"_c{cfg.contam_ratio:.2f}"

    # Visualize simulated 2D data
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    x_np = np.asarray(x)

    # Scatter of simulated data
    fig_d, ax_d = plt.subplots(1, 1, figsize=(5, 4))
    ax_d.scatter(
        x_np[:, 0],
        x_np[:, 1],
        s=12,
        c="#4c78a8",
        edgecolors="white",
        linewidths=0.2,
        alpha=0.5,
        label="Data",
    )
    ax_d.scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0, label="True mean")
    ax_d.set_xlabel("x1")
    ax_d.set_ylabel("x2")
    ax_d.set_title("Simulated data (2D)")
    ax_d.legend(loc="best", frameon=True, framealpha=0.9)
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(output_dir, f"data_scatter_2d{out_suffix}.png"), bbox_inches="tight")
    plt.close(fig_d)

    # Marginal histograms
    fig_m, ax_m = plt.subplots(1, 2, figsize=(10, 4))
    ax_m[0].hist(x_np[:, 0], bins=30, alpha=0.7, color="#4c78a8")
    ax_m[0].axvline(cfg.mean_true[0], color="k", linestyle="--", linewidth=1.0)
    ax_m[0].set_xlabel("x1")
    ax_m[0].set_ylabel("Count")
    ax_m[0].set_title("Marginal of x1")
    ax_m[1].hist(x_np[:, 1], bins=30, alpha=0.7, color="#f58518")
    ax_m[1].axvline(cfg.mean_true[1], color="k", linestyle="--", linewidth=1.0)
    ax_m[1].set_xlabel("x2")
    ax_m[1].set_ylabel("Count")
    ax_m[1].set_title("Marginal of x2")
    fig_m.tight_layout()
    fig_m.savefig(os.path.join(output_dir, f"data_marginals_2d{out_suffix}.png"), bbox_inches="tight")
    plt.close(fig_m)

    # Priors and configs
    pri = MVNPriorsConfig()
    gcfg = MVNGibbsConfig(gamma=cfg.gamma, eta=cfg.eta, priors=pri)
    
    # If samples are not loaded, run MCMC
    if samples_joint is None:
        # Joint sampling: vector [mu_x, mu_y, log_sig_x, log_sig_y, rho]
        logprob_joint = logprob_joint_mvn_from_vector(x, gcfg)
        init = jnp.array([0.0, 0.0, jnp.log(1.0), jnp.log(1.0), 0.0])
        joint_dim = 5

        chains = int(args.chains)
        keys = jax.random.split(key, chains)
        chains_joint = []
        if args.adapted:
            for k in keys:
                chains_joint.append(
                    run_nuts_adapted(
                        k,
                        logprob_joint,
                        init,
                        num_warmup=int(args.num_warmup),
                        num_samples=int(args.num_samples),
                        initial_step_size=float(args.init_step),
                        target_acceptance_rate=float(args.target_accept),
                        is_mass_matrix_diagonal=True,
                    )["samples"]
                )
        else:
            nuts_cfg = NutsConfig(step_size=float(args.step_size), num_warmup=int(args.num_warmup), num_samples=int(args.num_samples))
            for k in keys:
                chains_joint.append(run_nuts(k, logprob_joint, init, nuts_cfg)["samples"])  # (S,5)
        chains_joint = jnp.stack(chains_joint, axis=0)
        samples_joint = chains_joint.reshape((-1, joint_dim))
    
    # Extract GBI-DPD samples (loaded or sampled)
    mu_x_s, mu_y_s = samples_joint[:, 0], samples_joint[:, 1]
    log_sig_x_s, log_sig_y_s = samples_joint[:, 2], samples_joint[:, 3]
    rho_s = samples_joint[:, 4]
    c_s = jax.nn.sigmoid(rho_s)

    # ================= Probabilistic contamination identification (joint chain) =================
    # Compute per-point posterior contamination probability p_i using sample-wise ranking by log-likelihood
    n_points = int(x.shape[0])
    
    # Subsampling posterior draws for evaluation efficiency
    S_full = int(samples_joint.shape[0])
    S_eval_max = 400
    if S_full > S_eval_max:
        rng_eval = np.random.default_rng(cfg.seed)
        eval_indices = np.sort(rng_eval.choice(S_full, size=S_eval_max, replace=False))
    else:
        eval_indices = np.arange(S_full)
    
    S = len(eval_indices)
    
    # Use subsampled parameters for evaluation
    mu_x_eval = mu_x_s[eval_indices]
    mu_y_eval = mu_y_s[eval_indices]
    log_sig_x_eval = log_sig_x_s[eval_indices]
    log_sig_y_eval = log_sig_y_s[eval_indices]
    c_eval = c_s[eval_indices]

    # Pre-allocate log-likelihood matrix: (S, n)
    ll_matrix = np.empty((S, n_points), dtype=np.float64)

    for i, s_orig_idx in enumerate(eval_indices):
        params_s = MVNDiagParams(
            mean=jnp.array([mu_x_eval[i], mu_y_eval[i]]),
            log_scales=jnp.array([log_sig_x_eval[i], log_sig_y_eval[i]]),
        )
        ll_s = mvn_diag_log_pdf(params_s, x)  # (n,)
        ll_matrix[i] = np.asarray(ll_s)

    # Convert to likelihoods for downstream metrics (visualization)
    # Note: may underflow for high dimensions or far outliers, but ranking uses ll_matrix
    lik_matrix = np.exp(ll_matrix)

    # Sample-wise threshold K^(s) from r^(s) = max(0, 1-c^(s))
    r_arr = np.maximum(0.0, 1.0 - np.asarray(c_eval))
    K_arr = np.clip(np.round(r_arr * float(n_points)).astype(int), 0, n_points)

    # Flags[s, i] = 1 if point i is among smallest K^(s) log-likelihoods (more outlier-like)
    flags = np.zeros((S, n_points), dtype=np.uint8)
    for s in range(S):
        k = int(K_arr[s])
        if k <= 0:
            continue
        # rank by log-likelihood (smaller log-likelihood => more outlier-like)
        # using ll_matrix avoids underflow issues with exp()
        idx = np.argsort(ll_matrix[s])[:k]
        flags[s, idx] = 1

    # Posterior contamination probability per point
    p_contam = flags.mean(axis=0)  # (n,)

    # Expected K and percentile band
    K_mean = float(K_arr.mean())
    K_med = float(np.median(K_arr))
    K_p05, K_p95 = np.percentile(K_arr, [5, 95])
    K_star = int(np.round(K_mean))

    # Expected log-likelihood per point and rankings
    ll_mean = ll_matrix.mean(axis=0)  # (n,)
    rank_by_ll = np.argsort(ll_mean)  # ascending (smaller ll => more likely contaminated)
    
    # Keep lik_mean for backward compatibility in plots (though ll_mean is more robust)
    lik_mean = lik_matrix.mean(axis=0)
    
    # IMPORTANT: Use log-likelihood ranking as the primary ranking for visualization/output
    rank_by_lik = rank_by_ll
    lik_mean_sorted = lik_mean[rank_by_lik]

    rank_by_p_desc = np.argsort(-p_contam)
    
    # Use log-likelihood ranking for defining flagged_idx
    flagged_idx = rank_by_p_desc[:K_star] # top K_star by probability
    flagged_mask = np.zeros(n_points, dtype=bool)
    flagged_mask[flagged_idx] = True

    # ================= Baselines (3 methods only) =================
    # 1) Gaussian NLL (standard Bayes; theta only: mu, log_scales)
    theta_init = jnp.array([0.0, 0.0, jnp.log(1.0), jnp.log(1.0)])

    if samples_nll is None:
        logprob_nll = logprob_theta_only_nll_mvn_from_vector(x, pri)
        keys_nll = jax.random.split(key, chains)
        key = keys_nll[-1]
        chains_nll = []
        for k in keys_nll:
            chains_nll.append(
                run_nuts_adapted(
                    k,
                    logprob_nll,
                    theta_init,
                    num_warmup=int(args.num_warmup),
                    num_samples=int(args.num_samples),
                    initial_step_size=float(args.init_step),
                    target_acceptance_rate=float(args.target_accept),
                    is_mass_matrix_diagonal=True,
                )["samples"]
            )
        chains_nll = jnp.stack(chains_nll, axis=0)
        samples_nll = chains_nll.reshape((-1, 4))

    mu_x_nll_s, mu_y_nll_s = samples_nll[:, 0], samples_nll[:, 1]

    # 2) Gamma-divergence (theta only)
    if samples_gam is None:
        logprob_gam = logprob_theta_only_gamma_div_mvn_from_vector(x, gcfg)
        keys_gam = jax.random.split(key, chains)
        key = keys_gam[-1]
        chains_gam = []
        for k in keys_gam:
            chains_gam.append(
                run_nuts_adapted(
                    k,
                    logprob_gam,
                    theta_init,
                    num_warmup=int(args.num_warmup),
                    num_samples=int(args.num_samples),
                    initial_step_size=float(args.init_step),
                    target_acceptance_rate=float(args.target_accept),
                    is_mass_matrix_diagonal=True,
                )["samples"]
            )
        chains_gam = jnp.stack(chains_gam, axis=0)
        samples_gam = chains_gam.reshape((-1, 4))

    mu_x_gam_s, mu_y_gam_s = samples_gam[:, 0], samples_gam[:, 1]

    # Save samples if not loaded
    if args.load_samples is None:
        print(f"Saving samples to {args.save_samples}...")
        os.makedirs(os.path.dirname(args.save_samples), exist_ok=True)
        np.savez_compressed(
            args.save_samples,
            x=np.asarray(x),
            mask=np.asarray(mask_outlier),
            joint=np.asarray(samples_joint),
            nll=np.asarray(samples_nll),
            gam=np.asarray(samples_gam),
        )

    # Diagnostics (Only calculate if we ran chains, skip if loaded from flattened)
    if args.load_samples is None:
        rhat_joint = arviz_rhat(chains_joint, ["mu_x", "mu_y", "log_sig_x", "log_sig_y", "rho"]).tolist()
        ess_joint = arviz_ess(chains_joint, ["mu_x", "mu_y", "log_sig_x", "log_sig_y", "rho"]).tolist()

        rhat_nll = arviz_rhat(chains_nll, ["mu_x", "mu_y", "log_sig_x", "log_sig_y"]).tolist()
        ess_nll = arviz_ess(chains_nll, ["mu_x", "mu_y", "log_sig_x", "log_sig_y"]).tolist()

        rhat_gam = arviz_rhat(chains_gam, ["mu_x", "mu_y", "log_sig_x", "log_sig_y"]).tolist()
        ess_gam = arviz_ess(chains_gam, ["mu_x", "mu_y", "log_sig_x", "log_sig_y"]).tolist()

        # Save diagnostics
        os.makedirs(output_dir, exist_ok=True)
        import json as _json

        with open(os.path.join(output_dir, f"diagnostics_2d{out_suffix}.json"), "w") as f:
            _json.dump({
                "rhat": {"joint": rhat_joint, "gaussian_nll": rhat_nll, "gamma_div": rhat_gam},
                "ess": {"joint": ess_joint, "gaussian_nll": ess_nll, "gamma_div": ess_gam},
            }, f, indent=2)

    # ================= Visualization of probabilistic contamination =================
    x_np = np.asarray(x)

    # 1) Scatter with posterior P(contaminated)
    fig_p, ax_p = plt.subplots(1, 1, figsize=(6, 5))
    sc = ax_p.scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=p_contam,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=18,
        edgecolors="white",
        linewidths=0.2,
        alpha=0.9,
    )
    ax_p.scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0, label="True mean")
    cb = fig_p.colorbar(sc, ax=ax_p)
    cb.set_label("Posterior P(contaminated)")
    ax_p.set_xlabel("x1")
    ax_p.set_ylabel("x2")
    ax_p.set_title("Contamination probability (joint posterior)")
    ax_p.legend(loc="best", frameon=True, framealpha=0.9)
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(output_dir, f"data_scatter_contam_prob_2d{out_suffix}.png"), bbox_inches="tight")
    plt.close(fig_p)

    # ================= Evaluation with True Labels (if available) =================
    mask_np = np.asarray(mask_outlier) if mask_outlier is not None else None
    metrics_res = {}
    
    if mask_np is not None and mask_np.any():
        try:
            # --- DPD Joint Metrics ---
            y_true = mask_np.astype(int)
            
            # 1. Epsilon estimation metrics
            eps_true = cfg.contam_ratio
            eps_post_mean = float(np.mean(r_arr))
            eps_post_median = float(np.median(r_arr))
            eps_ci95 = np.percentile(r_arr, [2.5, 97.5])
            eps_ci95_cover = int((eps_true >= eps_ci95[0]) and (eps_true <= eps_ci95[1]))
            
            # 2. Outlier Probability Metrics (p_contam)
            # AUPRC (Average Precision) - "smaller contamination -> more important"
            auprc = average_precision_score(y_true, p_contam)
            try:
                auroc = roc_auc_score(y_true, p_contam)
            except ValueError:
                auroc = np.nan
            
            # Calibration
            # Use bin size heuristic: floor(n_samples^(-1/3)) is incorrect -> usually ceil(n^(1/3)) or similar rules.
            # User specified: floor(n^(-1/3)) ?? 
            # Wait, n^(-1/3) is < 1. 
            # Common rules: 
            # Square-root rule: sqrt(n)
            # Sturges' rule: log2(n) + 1
            # Rice rule: 2 * n^(1/3)
            # The user asked for "floor(n^(-1/3))" but likely meant "number of bins = floor(n^(1/3))" or something similar because n^(-1/3) is small.
            # Assuming the user meant "number of bins scales with n^(1/3)". Let's use standard Scott's rule-like n^(1/3).
            # OR did they mean bin_width ~ n^(-1/3) => n_bins ~ n^(1/3)?
            # Let's interpret "n^{-1/3}の床関数でビン数を設定" as "n_bins = floor(n^(1/3))".
            # Wait, actually commonly used is 10 or 15. n=500 -> 500^(1/3) ≈ 7.9 -> 7 bins. Reasonable.
            
            n_eval = len(y_true)
            n_bins_dynamic = int(np.floor(n_eval ** (1.0/3.0)))
            # Ensure at least 5 bins
            n_bins_dynamic = max(5, n_bins_dynamic)
            
            calib = compute_calibration_metrics(p_contam, y_true, n_bins=n_bins_dynamic)
            
            # Top-k Recall
            # k_oracle: based on true number of outliers
            n_outliers_true = np.sum(y_true)
            k_oracle = n_outliers_true
            # k_est: based on estimated epsilon
            k_est = int(np.round(eps_post_mean * n_points))
            
            # Sort indices by probability descending
            # We already have flagged_idx = rank_by_p_desc[:K_star] where K_star is round(mean(K))
            # K_star depends on r_arr which is 1-c. 
            # Note: k_est calculation above is consistent with K_star logic.
            
            idx_desc = rank_by_p_desc
            
            def calc_topk_recall(k_val):
                if k_val <= 0: return 0.0
                if n_outliers_true == 0: return 1.0 # trivial
                top_k_indices = idx_desc[:k_val]
                # recall = (true outliers in top k) / (total true outliers)
                hits = np.sum(y_true[top_k_indices])
                return float(hits / n_outliers_true)

            recall_k_oracle = calc_topk_recall(k_oracle)
            recall_k_est = calc_topk_recall(k_est)
            
            metrics_res["DPD"] = {
                "eps_true": eps_true,
                "eps_post_mean": eps_post_mean,
                "eps_post_median": eps_post_median,
                "eps_post_ci95_low": float(eps_ci95[0]),
                "eps_post_ci95_high": float(eps_ci95[1]),
                "eps_ci95_cover": eps_ci95_cover,
                "auprc": auprc,
                "auroc": auroc,
                "brier": calib["brier"],
                "ece": calib["ece"],
                "k_oracle": int(k_oracle),
                "k_est": int(k_est),
                "recall_k_oracle": recall_k_oracle,
                "recall_k_est": recall_k_est
            }
            
            # Plot Reliability Diagram (DPD only)
            fig_cal, ax_cal = plt.subplots(1, 1, figsize=(5, 5))
            plot_reliability_diagram(ax_cal, calib, title="Reliability Diagram (DPD)")
            fig_cal.savefig(f"outputs/calibration_dpd_2d{out_suffix}.png", bbox_inches="tight")
            plt.close(fig_cal)
            
            # --- Baseline Metrics (Ranking only) ---
            # Helper to compute ranking metrics for baselines
            def compute_baseline_metrics(samples_theta, name_key):
                # Calculate Log Posterior Predictive Density (LPPD) for each point
                # LPPD(x_i) = log ( 1/S sum_s p(x_i | theta_s) )
                #           = LogSumExp_s( log p(x_i | theta_s) ) - log S
                
                # Subsample if needed
                S_base = samples_theta.shape[0]
                if S_base > 400:
                    rng_b = np.random.default_rng(cfg.seed)
                    idx_b = rng_b.choice(S_base, size=400, replace=False)
                    s_theta_sub = samples_theta[idx_b]
                else:
                    s_theta_sub = samples_theta
                    
                S_sub = s_theta_sub.shape[0]
                ll_mat_b = np.zeros((S_sub, n_points))
                
                for i_s in range(S_sub):
                    # theta: [mu_x, mu_y, log_sig_x, log_sig_y]
                    p_b = MVNDiagParams(
                        mean=s_theta_sub[i_s, :2],
                        log_scales=s_theta_sub[i_s, 2:4]
                    )
                    ll_mat_b[i_s] = mvn_diag_log_pdf(p_b, x)
                
                # Compute LPPD per point
                lppd = jsp.special.logsumexp(ll_mat_b, axis=0) - np.log(S_sub)
                
                # Score: -LPPD (larger => more outlier)
                # Or simply ranking by LPPD ascending (smaller LPPD => more outlier)
                score = -np.asarray(lppd) # larger is more anomalous
                
                # AUPRC / AUROC
                auprc_b = average_precision_score(y_true, score)
                try:
                    auroc_b = roc_auc_score(y_true, score)
                except ValueError:
                    auroc_b = np.nan
                
                # Top-k Recall
                # Indices sorted by score descending (most anomalous first)
                idx_desc_b = np.argsort(score)[::-1]
                
                def calc_topk_recall_b(k_val):
                    if k_val <= 0: return 0.0
                    if n_outliers_true == 0: return 1.0
                    top_k_indices = idx_desc_b[:k_val]
                    hits = np.sum(y_true[top_k_indices])
                    return float(hits / n_outliers_true)
                
                # Using k_oracle (from truth) and k_est (borrowed from DPD)
                rec_oracle = calc_topk_recall_b(k_oracle)
                rec_est = calc_topk_recall_b(k_est) # Note: k_est comes from DPD!
                
                return {
                    "auprc": auprc_b,
                    "auroc": auroc_b,
                    "k_oracle": int(k_oracle),
                    "k_est_dpd": int(k_est), # Explicitly note this is DPD's k
                    "recall_k_oracle": rec_oracle,
                    "recall_k_est_dpd": rec_est
                }

            # NLL Baseline
            if samples_nll is not None:
                metrics_res["NLL"] = compute_baseline_metrics(samples_nll, "nll")
                
            # Gamma Baseline
            if samples_gam is not None:
                metrics_res["GammaDiv"] = compute_baseline_metrics(samples_gam, "gam")
            
            # Save metrics to JSON
            with open(f"outputs/metrics_2d{out_suffix}.json", "w") as f_met:
                json.dump(metrics_res, f_met, indent=2)
            print(f"Saved metrics to outputs/metrics_2d{out_suffix}.json")

            # 1. ROC Curve
            fpr, tpr, _ = roc_curve(mask_np.astype(int), p_contam)
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots(1, 1, figsize=(6, 5))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve (Outlier Detection)')
            ax_roc.legend(loc="lower right")
            fig_roc.savefig(f"outputs/roc_curve_2d{out_suffix}.png", bbox_inches="tight")
            plt.close(fig_roc)
            
            # 2. Scatter with True Outliers
            fig_ev, ax_ev = plt.subplots(1, 1, figsize=(6, 5))
            
            # Inliers (mask=False)
            inliers = x_np[~mask_np]
            p_in = p_contam[~mask_np]
            sc_in = ax_ev.scatter(inliers[:, 0], inliers[:, 1], c=p_in, cmap="viridis", vmin=0, vmax=1, 
                                  marker='o', s=20, alpha=0.8, label="True Inliers", edgecolors='none')
            
            # Outliers (mask=True)
            outliers = x_np[mask_np]
            p_out = p_contam[mask_np]
            # Use red edge for outliers
            sc_out = ax_ev.scatter(outliers[:, 0], outliers[:, 1], c=p_out, cmap="viridis", vmin=0, vmax=1,
                                   marker='^', s=60, edgecolors='red', linewidths=1.2, label="True Outliers")
            
            cb = fig_ev.colorbar(sc_in, ax=ax_ev)
            cb.set_label("Posterior P(contaminated)")
            
            ax_ev.set_title(f"Detection Performance (AUC={roc_auc:.3f})")
            ax_ev.legend()
            fig_ev.savefig(f"outputs/detection_scatter_2d{out_suffix}.png", bbox_inches="tight")
            plt.close(fig_ev)
            
            # 3. Uncertainty vs Probability (colored by Truth)
            p_std_local = np.sqrt(p_contam * (1.0 - p_contam))
            
            fig_unc, ax_unc = plt.subplots(1, 1, figsize=(6, 5))
            # Plot inliers
            ax_unc.scatter(p_contam[~mask_np], p_std_local[~mask_np], c='blue', alpha=0.3, label="True Inliers", s=15)
            # Plot outliers
            ax_unc.scatter(p_contam[mask_np], p_std_local[mask_np], c='red', alpha=0.6, marker='^', label="True Outliers", s=30)
            
            ax_unc.set_xlabel("Estimated P(outlier)")
            ax_unc.set_ylabel("Uncertainty (std)")
            ax_unc.set_title("Uncertainty Analysis")
            ax_unc.legend()
            ax_unc.grid(alpha=0.3)
            fig_unc.savefig(f"outputs/uncertainty_analysis_2d{out_suffix}.png", bbox_inches="tight")
            plt.close(fig_unc)
            
        except ImportError:
            print("sklearn not found, skipping ROC curve.")

    # 2) Rank curve of mean likelihood with K-band
    fig_rc, ax_rc = plt.subplots(1, 1, figsize=(7, 4))
    xs_rank = np.arange(1, n_points + 1)
    ax_rc.plot(xs_rank, lik_mean_sorted, color="#4c78a8", linewidth=1.5)
    ax_rc.axvspan(K_p05, K_p95, color="gray", alpha=0.15, label="K 5-95%")
    ax_rc.axvline(K_star, color="k", linestyle="--", linewidth=1.5, label=f"E[K]≈{K_star}")
    ax_rc.set_xlabel("Rank (ascending mean likelihood)")
    ax_rc.set_ylabel("Mean likelihood")
    ax_rc.set_title("Ranking by mean likelihood with K-band")
    ax_rc.legend(loc="best", frameon=True, framealpha=0.9)
    # Limit Y axis to [-0.02, min(max(mean), 0.02)] for better near-zero visibility
    _ymax_mean = float(np.max(lik_mean_sorted)) if lik_mean_sorted.size > 0 else 0.02
    ax_rc.set_ylim(-0.02, max(_ymax_mean, 0.02))
    fig_rc.tight_layout()
    fig_rc.savefig(f"outputs/data_rank_curve_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig_rc)

    # 2-b) Rank distribution as bar chart (likelihood)
    # For each MCMC sample, sort likelihoods ascending to get order statistics, then aggregate across samples
    sorted_lik_per_s = np.sort(lik_matrix, axis=1)  # (S, n)
    rank_mean_lik = sorted_lik_per_s.mean(axis=0)
    rank_std_lik = sorted_lik_per_s.std(axis=0)
    fig_rc2, ax_rc2 = plt.subplots(1, 1, figsize=(7, 4))
    xs_rank2 = np.arange(1, n_points + 1)
    # Draw bars for all ranks
    ax_rc2.bar(
        xs_rank2,
        rank_mean_lik,
        width=1.0,
        color="#4c78a8",
        alpha=0.35,
        edgecolor="#2f4b7c",
        linewidth=0.3,
        yerr=rank_std_lik,
        error_kw={"elinewidth": 0.6, "capsize": 1, "alpha": 0.6},
        align="center",
    )
    # Overlay mean line across all ranks
    ax_rc2.plot(xs_rank2, rank_mean_lik, color="#4c78a8", linewidth=1.5, label="mean over samples")
    # K band and reference lines
    ax_rc2.axvspan(K_p05, K_p95, color="gray", alpha=0.15, label="K 5-95%")
    ax_rc2.axvline(K_star, color="k", linestyle="--", linewidth=1.5, label=f"E[K]≈{K_star}")
    ax_rc2.set_xlim(1, n_points)
    ax_rc2.set_xlabel("Rank (ascending likelihood per sample)")
    ax_rc2.set_ylabel("Likelihood")
    ax_rc2.set_title("Ranking distribution (bars: mean ±1 std; line: mean)")
    ax_rc2.legend(loc="best", frameon=True, framealpha=0.9)
    # Limit Y axis to [-0.02, min(max(mean+std), 0.02)]
    _ymax_bar = float(np.max(rank_mean_lik + rank_std_lik)) if rank_mean_lik.size > 0 else 0.02
    ax_rc2.set_ylim(-0.02, max(_ymax_bar, 0.02))
    fig_rc2.tight_layout()
    fig_rc2.savefig(f"outputs/data_rank_curve_2d_band{out_suffix}.png", bbox_inches="tight")
    plt.close(fig_rc2)

    # 1-b) Bubble scatter: color by P(contam), size by std of likelihood across samples
    std_lik = lik_matrix.std(axis=0)
    # Robust scaling for sizes
    p10, p90 = np.percentile(std_lik, [10, 90])
    if p90 - p10 < 1e-12:
        sizes = np.full(n_points, 30.0)
    else:
        norm = np.clip((std_lik - p10) / (p90 - p10), 0.0, 1.0)
        sizes = 12.0 + norm * (72.0 - 12.0)  # points^2 scale in matplotlib

    fig_b, ax_b = plt.subplots(1, 1, figsize=(6, 5))
    sc_b = ax_b.scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=p_contam,
        s=sizes,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolors="white",
        linewidths=0.2,
        alpha=0.9,
    )
    ax_b.scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0, label="True mean")
    cb_b = fig_b.colorbar(sc_b, ax=ax_b)
    cb_b.set_label("Posterior P(contaminated)")
    # Size legend (use representative percentiles)
    rep_vals = np.array([p10, 0.5 * (p10 + p90), p90])
    rep_sizes = (np.full_like(rep_vals, 30.0) if p90 - p10 < 1e-12
                 else 12.0 + np.clip((rep_vals - p10) / (p90 - p10), 0.0, 1.0) * (72.0 - 12.0))
    for rv, rs in zip(rep_vals, rep_sizes):
        ax_b.scatter([], [], s=rs, c="#aaaaaa", label=f"std(lik)≈{rv:.2g}")
    # Avoid duplicate legend handles
    handles, labels = ax_b.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax_b.legend(uniq.values(), uniq.keys(), loc="best", frameon=True, framealpha=0.9)
    ax_b.set_xlabel("x1")
    ax_b.set_ylabel("x2")
    ax_b.set_title("Bubble: color=P(contam), size=std(lik) across samples")
    fig_b.tight_layout()
    fig_b.savefig(f"outputs/data_scatter_contam_prob_bubble_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig_b)
    # 3) Scatter highlighting top E[K] by contamination probability
    fig_h, ax_h = plt.subplots(1, 1, figsize=(6, 5))
    ax_h.scatter(
        x_np[:, 0], x_np[:, 1], s=14, c="#bbbbbb", alpha=0.5, edgecolors="white", linewidths=0.2
    )
    ax_h.scatter(
        x_np[flagged_mask, 0], x_np[flagged_mask, 1], s=40, facecolors="none", edgecolors="#d62728", linewidths=1.5, label=f"Top E[K]={K_star} by P"
    )
    ax_h.scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0, label="True mean")
    ax_h.set_xlabel("x1")
    ax_h.set_ylabel("x2")
    ax_h.set_title("Flagged points (by posterior P and E[K])")
    ax_h.legend(loc="best", frameon=True, framealpha=0.9)
    fig_h.tight_layout()
    fig_h.savefig(f"outputs/data_scatter_flagged_topK_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig_h)

    # 4) Save ranking/probability info as JSON
    import json as _json
    with open(f"outputs/contam_ranking_2d{out_suffix}.json", "w") as fjson:
        _json.dump(
            {
                "K": {
                    "mean": K_mean,
                    "median": K_med,
                    "p05": float(K_p05),
                    "p95": float(K_p95),
                    "rounded_mean": K_star,
                },
                "indices_sorted_by_lik": rank_by_lik.tolist(),
                "indices_sorted_by_p_desc": rank_by_p_desc.tolist(),
                "p_contam": p_contam.tolist(),
                "lik_mean": lik_mean.tolist(),
                "std_lik": std_lik.tolist(),
                "rank_curve": {
                    "mean_lik": rank_mean_lik.tolist(),
                    "std_lik": rank_std_lik.tolist(),
                },
            },
            fjson,
            indent=2,
        )

    # ================= Uncertainty decomposition =================
    # (A) Parameter-only classification uncertainty: vary theta per sample, fix K at E[K]
    param_flags = np.zeros((S, n_points), dtype=np.uint8)
    for s in range(S):
        k_fixed = int(K_star)
        if k_fixed <= 0:
            continue
        idx_param = np.argsort(ll_matrix[s])[:k_fixed]
        param_flags[s, idx_param] = 1
    param_std = param_flags.std(axis=0)  # std over samples of 0/1 classification

    # (B) Epsilon-only classification uncertainty: fix theta at posterior mean, vary K by r^(s)
    theta_mean = MVNDiagParams(
        mean=jnp.array([float(mu_x_s.mean()), float(mu_y_s.mean())]),
        log_scales=jnp.array([float(log_sig_x_s.mean()), float(log_sig_y_s.mean())]),
    )
    ll_ref = np.asarray(mvn_diag_log_pdf(theta_mean, x))  # (n,)
    order_ref = np.argsort(ll_ref)  # ascending
    eps_flags = np.zeros((S, n_points), dtype=np.uint8)
    for s in range(S):
        k_s = int(K_arr[s])
        if k_s <= 0:
            continue
        idx_eps = order_ref[:k_s]
        eps_flags[s, idx_eps] = 1
    eps_std = eps_flags.std(axis=0)

    # Two-panel bubble plot: Left color=P(contam), size=std P(contam); Right color=mean log-lik, size=std log-lik
    # Robust scaling for sizes
    def _scale_sizes(v: np.ndarray) -> np.ndarray:
        q10, q90 = np.percentile(v, [10, 90])
        if q90 - q10 < 1e-12:
            return np.full(v.shape[0], 30.0)
        norm = np.clip((v - q10) / (q90 - q10), 0.0, 1.0)
        return 12.0 + norm * (72.0 - 12.0)

    # Generalized size scaling with explicit min/max sizes (used by combined figure as well)
    def _scale_sizes_range(v: np.ndarray, smin: float, smax: float) -> np.ndarray:
        q10, q90 = np.percentile(v, [10, 90])
        if q90 - q10 < 1e-12:
            return np.full(v.shape[0], smin + 0.5 * (smax - smin))
        norm = np.clip((v - q10) / (q90 - q10), 0.0, 1.0)
        return smin + norm * (smax - smin)
    # std of P(contam) per point (Bernoulli variability across draws)
    p_std = np.sqrt(np.clip(p_contam * (1.0 - p_contam), 0.0, 0.25))
    sizes_left = _scale_sizes(p_std)
    # std of likelihood across samples per point
    sizes_right = _scale_sizes(std_lik)

    fig_ud, ax_ud = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    sc_l = ax_ud[0].scatter(
        x_np[:, 0], x_np[:, 1], c=p_contam, s=sizes_left, cmap="coolwarm", vmin=0.0, vmax=1.0,
        #x_np[:, 0], x_np[:, 1], c=p_contam, s=sizes_left, cmap="cividis", vmin=0.0, vmax=1.0,
        #x_np[:, 0], x_np[:, 1], c=p_contam, s=sizes_left, cmap="cividis", vmin=0.0, vmax=1.0,
        edgecolors="white", linewidths=0.2, alpha=0.9
    )
    ax_ud[0].scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0)
    ax_ud[0].set_title("Color=P(contam), Size=std P(contam)")
    ax_ud[0].set_xlabel("x1")
    ax_ud[0].set_ylabel("x2")

    sc_r = ax_ud[1].scatter(
        x_np[:, 0], x_np[:, 1], c=lik_mean, s=sizes_right, cmap="coolwarm",
        edgecolors="white", linewidths=0.2, alpha=0.9
    )
    ax_ud[1].scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0)
    ax_ud[1].set_title("Color=mean lik, Size=std lik")
    ax_ud[1].set_xlabel("x1")

    # Separate colorbars for each panel
    cb_l = fig_ud.colorbar(sc_l, ax=ax_ud[0])
    cb_l.set_label("Posterior P(contaminated)")
    cb_r = fig_ud.colorbar(sc_r, ax=ax_ud[1])
    cb_r.set_label("Mean likelihood")
    fig_ud.tight_layout()
    fig_ud.savefig(f"outputs/data_scatter_uncert_decomp_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig_ud)

    # Combined figure: [Ranking distribution (bars+line)] | [Color=mean lik, Size=std lik]
    eps_post_mean = float(np.mean(r_arr))  # posterior mean of epsilon
    # Make bubbles slightly larger for visibility
    sizes_std_lik_big = _scale_sizes_range(std_lik, 30.0, 220.0)
    fig_combo, ax_combo = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})

    # Left: Ranking distribution (same content as data_rank_curve_2d_band.png)
    xs_rank_combo = np.arange(1, n_points + 1)
    ax_combo[0].bar(
        xs_rank_combo,
        rank_mean_lik,
        width=1.0,
        color="#4c78a8",
        alpha=0.35,
        edgecolor="#2f4b7c",
        linewidth=0.3,
        yerr=rank_std_lik,
        error_kw={"elinewidth": 0.6, "capsize": 1, "alpha": 0.6},
        align="center",
    )
    ax_combo[0].plot(xs_rank_combo, rank_mean_lik, color="#4c78a8", linewidth=1.5, label="mean over samples")
    ax_combo[0].axvspan(K_p05, K_p95, color="gray", alpha=0.15, label="posterior ε 5–95%")
    ax_combo[0].axvline(K_star, color="k", linestyle="--", linewidth=1.5, label=f"expected outliers ≈ {K_star} (posterior mean ε = {eps_post_mean:.3f})")
    ax_combo[0].set_xlim(1, n_points)
    # Y-limits consistent with standalone plot
    _ymax_bar2 = float(np.max(rank_mean_lik + rank_std_lik)) if rank_mean_lik.size > 0 else 0.02
    ax_combo[0].set_ylim(-0.02, max(_ymax_bar2, 0.02))
    ax_combo[0].set_xlabel("Rank (ascending likelihood per sample)")
    ax_combo[0].set_ylabel("Likelihood")
    ax_combo[0].set_title("Likelihood ranking", fontsize=16)
    # Place legend inside the axes at best location to avoid overlap with title
    ax_combo[0].legend(loc="best", frameon=True, framealpha=0.95)

    # Right: Color=mean lik, Size=std lik (same as right in data_scatter_uncert_decomp_2d.png)
    # Changed to: Color=std(FoD), Size=FoD (based on user request)
    sizes_fod = 30.0 + p_contam * (220.0 - 30.0) # scaling for this plot
    
    sc_right = ax_combo[1].scatter(
        x_np[:, 0], x_np[:, 1], c=p_std, s=sizes_fod, cmap="viridis", vmin=0.0, vmax=0.5,
        edgecolors="white", linewidths=0.2, alpha=0.9
    )
    ax_combo[1].scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0)
    ax_combo[1].set_title("FoD (size) & Std. (Color)", fontsize=16)
    ax_combo[1].set_xlabel("")
    ax_combo[1].set_facecolor("#f7f7f7")
    
    # Define thresholds by posterior epsilon quantiles and mean
    k_opt = int(np.clip(np.round(K_p05), 0, n_points))
    k_mean_i = int(np.clip(K_star, 0, n_points))
    k_pess = int(np.clip(np.round(K_p95), 0, n_points))
    order_idx = rank_by_lik  # ascending by mean likelihood
    idx_opt = order_idx[:k_opt]
    idx_mean_only = order_idx[k_opt:k_mean_i] if k_mean_i > k_opt else np.array([], dtype=int)
    idx_pess_only = order_idx[k_mean_i:k_pess] if k_pess > k_mean_i else np.array([], dtype=int)

    # Base: light gray background points for context
    ax_combo[2].scatter(x_np[:, 0], x_np[:, 1], s=12, c="#cccccc", alpha=0.5, edgecolors="white", linewidths=0.2)
    # Optimistic: circle
    if idx_opt.size > 0:
        ax_combo[2].scatter(x_np[idx_opt, 0], x_np[idx_opt, 1], s=40, facecolors="none", edgecolors="#2ca02c", linewidths=1.2, marker="o", label="optimistic ε")
    # Mean-only: triangle
    if idx_mean_only.size > 0:
        ax_combo[2].scatter(x_np[idx_mean_only, 0], x_np[idx_mean_only, 1], s=40, facecolors="none", edgecolors="#ff7f0e", linewidths=1.2, marker="^", label="mean ε")
    # Pessimistic-only: square
    if idx_pess_only.size > 0:
        ax_combo[2].scatter(x_np[idx_pess_only, 0], x_np[idx_pess_only, 1], s=40, facecolors="none", edgecolors="#d62728", linewidths=1.2, marker="s", label="pessimistic ε")
    ax_combo[2].scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0)
    ax_combo[2].set_title("Outlier detection by ε scenarios", fontsize=16)
    ax_combo[2].set_xlabel("")
    ax_combo[2].set_facecolor("#f7f7f7")
    ax_combo[2].legend(loc="upper left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    
    # Add colorbars after tight_layout to avoid UserWarning
    # Right panel colorbar (inset)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax_combo[1],
                       width="5%",
                       height="40%",
                       loc='lower left',
                       bbox_to_anchor=(0.05, 0.05, 1, 1),
                       bbox_transform=ax_combo[1].transAxes,
                       borderpad=0,
                       )
    cb_right = fig_combo.colorbar(sc_right, cax=axins, ticks=[0, 0.25, 0.5])
    cb_right.ax.tick_params(labelsize=8)

    fig_combo.savefig(f"outputs/data_rank_vs_scatter_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig_combo)

    # Probability-only decomposition figure: Left color=P(contam), Right size=std P(contam)
    sizes_pstd = _scale_sizes_range(p_std, 24.0, 180.0)
    fig_pd, ax_pd = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    # Use single-hue truncated colormap to avoid overly white low values
    from matplotlib.colors import ListedColormap, PowerNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    def _truncate_cmap(cmap_name: str, minval: float = 0.20, maxval: float = 1.0, n: int = 256):
        cmap = plt.get_cmap(cmap_name)
        return ListedColormap(cmap(np.linspace(minval, maxval, n)))
    cmap_prob = _truncate_cmap("viridis", 0.12, 1.0)
    # ensure low values are still visible (not pure white)
    cmap_prob.set_under("#f0f0f0")
    sc_p_only = ax_pd[0].scatter(
        x_np[:, 0], x_np[:, 1], c=p_contam, s=18, cmap=cmap_prob,
        norm=PowerNorm(gamma=0.85, vmin=0.0, vmax=1.0),
        edgecolors="white", linewidths=0.2, alpha=0.9
    )
    ax_pd[0].scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0)
    ax_pd[0].set_title("FoD of outliers", fontsize=20)
    # light background for contrast with low probabilities
    ax_pd[0].set_facecolor("#f7f7f7")
    # Vertical colorbar on the left without altering panel size
    cb_p_only = fig_pd.colorbar(sc_p_only)
    cb_p_only.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb_p_only.ax.tick_params(labelsize=10)

    ax_pd[1].scatter(
        x_np[:, 0], x_np[:, 1], s=sizes_pstd, c="#4c78a8",
        edgecolors="white", linewidths=0.2, alpha=0.85
    )
    test = ax_pd[1].scatter(cfg.mean_true[0], cfg.mean_true[1], marker="x", s=100, c="red", linewidths=2.0)
    ax_pd[1].set_title("Uncertainty in outlier detection", fontsize=20)
    # Match background to left panel
    ax_pd[1].set_facecolor("#f7f7f7")
    cbl = fig_pd.colorbar(test)
    # (legend removed as requested)
    cbl.solids.set_alpha(0.0)
    cbl.outline.set_visible(False)
    cbl.set_ticks([])
    #cbl.patch.set_alpha(0.0)

    fig_pd.subplots_adjust(left=0.12, bottom=0.18, wspace=0.08)
    fig_pd.savefig(f"outputs/data_scatter_prob_decomp_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig_pd)
    
    # Extend JSON with uncertainty decomposition
    with open(f"outputs/contam_ranking_2d{out_suffix}.json", "r") as fjson_r:
        _content = _json.load(fjson_r)
    _content["uncertainty_decomp"] = {
        "param_only_std": param_std.tolist(),
        "epsilon_only_std": eps_std.tolist(),
    }
    with open(f"outputs/contam_ranking_2d{out_suffix}.json", "w") as fjson_w:
        _json.dump(_content, fjson_w, indent=2)

    # Scatter + density for mu=(mu_x, mu_y): Proposed vs NLL
    def plot_scatter_kde(ax, xs, ys, mu_true, title, fill_cmap, point_color, xlim, ylim):
        x_np = np.asarray(xs)
        y_np = np.asarray(ys)
        data = np.vstack([x_np, y_np])
        kde = ss.gaussian_kde(data)
        x_min, x_max = xlim
        y_min, y_max = ylim
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 160),
            np.linspace(y_min, y_max, 160),
        )
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        
        # Check if density is flat or zero (no contours to draw)
        if zz.max() - zz.min() > 1e-12:
            # light filled density (drop the lowest band to remove outermost contour)
            levels_fill = np.linspace(zz.min(), zz.max(), 18)[1:]
            cf = ax.contourf(xx, yy, zz, levels=levels_fill, cmap=fill_cmap, alpha=0.35)
            # black contour lines (also drop the lowest band)
            levels_line = np.linspace(zz.min(), zz.max(), 8)[1:]
            ax.contour(xx, yy, zz, levels=levels_line, colors="k", linewidths=1.0, alpha=0.9)
        
        # scatter samples
        ax.scatter(
            x_np,
            y_np,
            s=12,
            c=point_color,
            edgecolors="white",
            linewidths=0.2,
            alpha=0.35,
            label="Samples",
        )
        # true parameter marker
        ax.scatter(mu_true[0], mu_true[1], marker="x", s=100, c="red", linewidths=2.0, label="True")
        # crosshair at true values
        ax.axvline(mu_true[0], color="k", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.axhline(mu_true[1], color="k", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title)
        ax.set_xlabel("mu_x")
        ax.set_ylabel("mu_y")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    # set shared axis ranges across both methods for fair comparison
    # Use fixed range as requested to avoid KSD scale dominating
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    # x_all = np.concatenate([np.asarray(mu_x_s), np.asarray(mu_x_jnll_s), np.asarray(mu_x_gam_s), np.asarray(mu_x_mmd_s), np.asarray(mu_x_ksd_s)])
    # y_all = np.concatenate([np.asarray(mu_y_s), np.asarray(mu_y_jnll_s), np.asarray(mu_y_gam_s), np.asarray(mu_y_mmd_s), np.asarray(mu_y_ksd_s)])
    # x_min, x_max = np.percentile(x_all, [0.5, 99.5])
    # y_min, y_max = np.percentile(y_all, [0.5, 99.5])
    # # ensure (-0.5, -0.5) is visible
    # x_min = min(x_min, -0.5)
    # y_min = min(y_min, -0.5)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    plot_scatter_kde(
        ax[0], mu_x_s, mu_y_s, cfg.mean_true, "GBI-DPD (joint)",
        fill_cmap="Blues", point_color="#4c78a8", xlim=(x_min, x_max), ylim=(y_min, y_max)
    )
    plot_scatter_kde(
        ax[1], mu_x_nll_s, mu_y_nll_s, cfg.mean_true, "Gaussian NLL (theta-only)",
        fill_cmap="Greens", point_color="#2ca02c", xlim=(x_min, x_max), ylim=(y_min, y_max)
    )
    plot_scatter_kde(
        ax[2], mu_x_gam_s, mu_y_gam_s, cfg.mean_true, "Gamma-Div (theta-only)",
        fill_cmap="Oranges", point_color="#f58518", xlim=(x_min, x_max), ylim=(y_min, y_max)
    )
    fig.tight_layout()
    fig.savefig(f"outputs/post_mu_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.hist(c_s, bins=20, alpha=0.6, label="GBI-DPD", color="#4c78a8")
    ax2.axvline(1.0 - cfg.contam_ratio, color="k", linestyle="--", label="True (1-r)")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_title("Posterior of c (2D)")
    ax2.legend()
    fig2.savefig(f"outputs/post_c_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig2)

    # Posterior histogram for contamination ratio r = max(0, 1-c)
    r_s = jnp.maximum(0.0, 1.0 - c_s)
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))
    ax3.hist(r_s, bins=20, alpha=0.6, label="GBI-DPD", color="#4c78a8")
    ax3.axvline(cfg.contam_ratio, color="k", linestyle="--", label="True")
    ax3.set_xlim(0.0, 1.0)
    ax3.set_title("Posterior of r=max(0,1-c)")
    ax3.legend()
    fig3.savefig(f"outputs/post_contam_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig3)

    # Combined figure: 2 rows x 3 columns
    # Col 0: GBI-DPD (joint), Col 1: Gaussian NLL (theta-only), Col 2: Gamma-Div (theta-only)
    
    fig4, ax4 = plt.subplots(2, 3, figsize=(15, 8))

    # --- Column 0: GBI-DPD ---
    plot_scatter_kde(
        ax4[0, 0], mu_x_s, mu_y_s, cfg.mean_true, "GBI-DPD (joint)\nParameters",
        fill_cmap="Blues", point_color="#4c78a8", xlim=(x_min, x_max), ylim=(y_min, y_max)
    )
    ax4[1, 0].hist(r_s, bins=20, alpha=0.6, color="#4c78a8", label="GBI-DPD")
    ax4[1, 0].axvline(cfg.contam_ratio, color="k", linestyle="--", label="True")
    ax4[1, 0].set_xlim(0.0, 1.0)
    ax4[1, 0].set_xlabel("r = max(0, 1-c)")
    ax4[1, 0].set_ylabel("Count")
    ax4[1, 0].set_title("Contamination ratio")
    ax4[1, 0].legend(loc="upper right")

    # --- Column 1: Gaussian NLL ---
    plot_scatter_kde(
        ax4[0, 1], mu_x_nll_s, mu_y_nll_s, cfg.mean_true, "Gaussian NLL (theta-only)\nParameters",
        fill_cmap="Greens", point_color="#2ca02c", xlim=(x_min, x_max), ylim=(y_min, y_max)
    )
    ax4[1, 1].axis("off")
    ax4[1, 1].text(0.5, 0.5, "r is not defined\n(no c parameter)", ha="center", va="center", fontsize=12)

    # --- Column 2: Gamma-Div ---
    plot_scatter_kde(
        ax4[0, 2], mu_x_gam_s, mu_y_gam_s, cfg.mean_true, "Gamma-Div (theta-only)\nParameters",
        fill_cmap="Oranges", point_color="#f58518", xlim=(x_min, x_max), ylim=(y_min, y_max)
    )
    ax4[1, 2].axis("off")
    ax4[1, 2].text(0.5, 0.5, "r is not defined\n(no c parameter)", ha="center", va="center", fontsize=12)

    fig4.tight_layout()
    fig4.savefig(f"outputs/post_posterior_2d{out_suffix}.png", bbox_inches="tight")
    plt.close(fig4)


if __name__ == "__main__":
    main()


