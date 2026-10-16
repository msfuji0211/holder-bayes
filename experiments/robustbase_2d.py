import argparse
import os
import sys

# Add project root to path to allow importing hbayes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import csv

from hbayes.posteriors import (
    MVNPriorsConfig,
    MVNGibbsConfig,
    logprob_joint_mvn_from_vector,
    log_prior_theta_mvn,
    log_prior_c_from_rho_mvn,
)
from hbayes.models import MVNDiagParams, mvn_diag_log_pdf
from hbayes.dpd import mvn_diag_A_gamma, mvn_diag_B_gamma
from hbayes.sampling import run_nuts_adapted
from hbayes.diagnostics import arviz_rhat, arviz_ess
from typing import Callable, Optional, List

# Helper for MMD computation
def compute_mmd_gaussian_kernel_loss(
    params: MVNDiagParams, c: jnp.ndarray, x: jnp.ndarray, length_scale: float
) -> jnp.ndarray:
    """
    MMD^2 loss between empirical distribution P_n and model Q = c*P_theta.
    """
    d = params.mean.shape[0]
    n = x.shape[0]
    mu = params.mean
    sig2 = jnp.exp(2.0 * params.log_scales) # diagonal variances
    
    # 1. Model self-similarity term: E[k(y,y')]
    det_term_model = jnp.prod(1.0 + 2.0 * sig2 / (length_scale**2))
    term_yy = det_term_model ** (-0.5)
    
    # 2. Cross term: 1/n sum_i E[k(x_i, y)]
    var_eff = sig2 + length_scale**2  # (d,)
    
    # log N(x|m, V_eff)
    diff = x - mu # (n, d)
    mahal = jnp.sum((diff**2) / var_eff, axis=1) # (n,)
    log_det_eff = jnp.sum(jnp.log(var_eff))
    log_prob_cross = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det_eff + mahal)
    
    prob_cross = jnp.exp(log_prob_cross)
    norm_factor = (2.0 * jnp.pi * length_scale**2) ** (0.5 * d)
    
    # Average over data
    term_xy = jnp.mean(prob_cross) * norm_factor
    
    # MMD^2 (ignoring constant data-data term)
    return (c**2) * term_yy - 2.0 * c * term_xy


def logprob_joint_mmd_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig, length_scale: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Joint MMD-Bayes log posterior."""
    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
    n = float(x.shape[0])

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
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
        dist_x = jnp.sum((x_i - mu)**2)
        dist_y = jnp.sum((y_i - mu)**2)
        w_x = (1.0 + dist_x / (2.0 * sigma_ker**2)) ** (-5.0)
        w_y = (1.0 + dist_y / (2.0 * sigma_ker**2)) ** (-5.0)
        
        return k_val * w_x * w_y

    # Stein operator u_p(x, y) for a pair
    def u_statistic(x_i, y_i):
        s_x = score_fn(x_i)
        s_y = score_fn(y_i)
        
        k_val = kernel_fn(x_i, y_i)
        
        # Gradients
        grad_x_k = jax.grad(kernel_fn, argnums=0)(x_i, y_i)
        grad_y_k = jax.grad(kernel_fn, argnums=1)(x_i, y_i)
        
        def diag_hessian_term(idx):
            def d_dx_i(y_val):
                return jax.grad(kernel_fn, argnums=0)(x_i, y_val)[idx]
            return jax.grad(d_dx_i)(y_i)[idx]

        indices = jnp.arange(d)
        trace_term = jnp.sum(jax.vmap(diag_hessian_term)(indices))
        
        term1 = jnp.dot(s_x, s_y) * k_val
        term2 = jnp.dot(s_x, grad_y_k)
        term3 = jnp.dot(s_y, grad_x_k)
        
        return term1 + term2 + term3 + trace_term

    u_mat = jax.vmap(jax.vmap(u_statistic, (None, 0)), (0, None))(x, x)
    
    return jnp.mean(u_mat)


def logprob_joint_ksd_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig, sigma_ker: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Joint KSD-Bayes log posterior."""
    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
    n = float(x.shape[0])

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        mean = z[:d]
        log_scales = z[d : 2 * d]
        rho = z[-1]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        
        loss = compute_ksd_imq_loss(params, x, gamma_ker=0.5, sigma_ker=sigma_ker)
        
        lp_theta = log_prior_theta_mvn(params, config.priors)
        lp_c = log_prior_c_from_rho_mvn(rho, config.priors)
        
        return lp_theta + lp_c - n * config.eta * loss

    return _logprob

def logprob_joint_nll_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Joint NLL (unnormalized KL) log posterior."""
    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
    n = float(x.shape[0])

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        mean = z[:d]
        log_scales = z[d : 2 * d]
        rho = z[-1]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        
        c = jax.nn.sigmoid(rho)
        log_c = jnp.log(jnp.clip(c, 1e-12, 1.0))
        
        ll_per_point = mvn_diag_log_pdf(params, x) # (n,)
        mean_ll = jnp.mean(ll_per_point)
        
        loss = c - log_c - mean_ll
        
        lp_theta = log_prior_theta_mvn(params, config.priors)
        lp_c = log_prior_c_from_rho_mvn(rho, config.priors)
        
        return lp_theta + lp_c - n * config.eta * loss

    return _logprob


def logprob_joint_gamma_div_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Joint Gamma-Divergence log posterior."""
    n = float(x.shape[0])
    gamma = config.gamma

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
        mean = z[:d]
        log_scales = z[d : 2 * d]
        rho = z[-1]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        
        a_val = mvn_diag_A_gamma(params, gamma)
        b_val = mvn_diag_B_gamma(params, x, gamma)
        
        log_a = jnp.log(a_val + 1e-30)
        log_b = jnp.log(b_val + 1e-30)
        
        loss = (1.0 / (1.0 + gamma)) * log_a - (1.0 / gamma) * log_b
        
        lp_theta = log_prior_theta_mvn(params, config.priors)
        lp_c = log_prior_c_from_rho_mvn(rho, config.priors)
        
        return lp_theta + lp_c - n * config.eta * loss

    return _logprob


def load_dataset(path: str, columns: Optional[List[str]] = None):
    """
    Generic dataset loader using pandas.
    If columns is None, it tries to use all numeric columns.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        raise e

    if columns:
        # Check if columns exist
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in {path}. Available: {list(df.columns)}")
        numeric_df = df[columns]
    else:
        # Use all numeric columns, skipping typical index columns if they look like indices
        # Heuristic: drop columns named 'Unnamed: 0', 'rownames', 'index', 'id'
        drop_cols = [c for c in df.columns if c.lower() in ['unnamed: 0', 'rownames', 'index', 'id']]
        if drop_cols:
            print(f"Dropping index-like columns: {drop_cols}")
            df = df.drop(columns=drop_cols)
        
        # Select only numeric
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in dataset.")
        print(f"Using columns: {list(numeric_df.columns)}")
        
    return jnp.array(numeric_df.values, dtype=jnp.float32), list(numeric_df.columns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=None, help="Explicit eta value. If None, determined by eta_mode")
    parser.add_argument("--eta_mode", type=str, choices=["fixed", "covdet"], default="fixed", help="Method to determine eta")
    parser.add_argument("--num_warmup", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.8, help="Target acceptance rate for NUTS")
    parser.add_argument("--init_step", type=float, default=1.0, help="Initial step size for NUTS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="data/starsCYG.csv")
    parser.add_argument("--output_dir", type=str, default="outputs_robustbase")
    parser.add_argument("--save_samples", type=str, default=None, help="Path to save .npz samples file")
    parser.add_argument("--columns", type=str, nargs='+', default=None, help="Specific columns to use")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    print(f"Loading data from {args.data_path}...")
    x, col_names = load_dataset(args.data_path, args.columns)
    n = x.shape[0]
    d = x.shape[1]
    print(f"Data shape: {x.shape} (n={n}, d={d})")

    # Standardize data for numerical stability
    mean_x = jnp.mean(x, axis=0)
    std_x = jnp.std(x, axis=0)
    # Avoid division by zero for constant columns
    std_x = jnp.where(std_x < 1e-9, 1.0, std_x)
    x_std = (x - mean_x) / std_x
    
    # Determine eta
    eta_val = 1.0
    if args.eta is not None:
        eta_val = float(args.eta)
    elif args.eta_mode == "covdet":
        # Compute covariance determinant of raw data for covdet mode
        x_centered = x - jnp.mean(x, axis=0)
        # Use simple covariance estimator
        cov = (x_centered.T @ x_centered) / max(n - 1, 1)
        cov_det = float(jnp.linalg.det(cov))
        
        # Jones et al. (2001) or standard DPD heuristic: |Sigma|^(gamma)
        # Here we use the raw data determinant.
        eta_val = (max(cov_det, 1e-12)) ** args.gamma
        print(f"Using covdet eta: {eta_val:.4f} (cov_det={cov_det:.4e}, gamma={args.gamma})")
    else:
        # Default fixed eta=1.0
        eta_val = 1.0
    
    print(f"Final eta: {eta_val}")
    
    # Plot raw data
    try:
        plt.figure(figsize=(10, 10))
        if d == 2:
            plt.scatter(x[:, 0], x[:, 1], c='black', alpha=0.7)
            plt.xlabel(col_names[0])
            plt.ylabel(col_names[1])
            plt.title(f"Data Scatter ({os.path.basename(args.data_path)})")
        else:
            # Pairplot for higher dimensions (limit to first 5 dims to avoid huge plots)
            plot_cols = col_names[:5]
            plot_df = pd.DataFrame(np.array(x)[:, :5], columns=plot_cols)
            sns.pairplot(plot_df, corner=True, diag_kind="kde", plot_kws={'alpha': 0.6, 's': 20})
            plt.suptitle(f"Data Pairplot ({os.path.basename(args.data_path)})", y=1.02)
        
        plt.savefig(os.path.join(args.output_dir, "data_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    key = jax.random.PRNGKey(args.seed)

    # Configs
    # Prior: Standard Normal for standardized data is reasonable weakly informative prior
    pri = MVNPriorsConfig() 
    
    gcfg = MVNGibbsConfig(gamma=args.gamma, eta=eta_val, priors=pri)

    # Initial state (Generalized for d dimensions)
    # mu (d) + log_scale (d) + rho (1)
    init_mu = jnp.zeros(d)
    init_log_scale = jnp.zeros(d) # log(1.0) = 0.0
    init_rho = jnp.array([0.0])
    init = jnp.concatenate([init_mu, init_log_scale, init_rho]) 

    print(f"Parameter dimension: {init.shape[0]}")

    # Define models
    models = {
        "GBI-DPD": logprob_joint_mvn_from_vector(x_std, gcfg),
        "Joint NLL": logprob_joint_nll_mvn_from_vector(x_std, gcfg),
        # "Gamma-Div": logprob_joint_gamma_div_mvn_from_vector(x_std, gcfg), # Gamma-Div doesn't estimate c in the same way (c cancels out)
        "MMD-Bayes": logprob_joint_mmd_mvn_from_vector(x_std, gcfg, length_scale=1.0), # Heuristic length scale for std data
        "KSD-Bayes": logprob_joint_ksd_mvn_from_vector(x_std, gcfg, sigma_ker=1.0),
    }

    results = {}

    for name, logprob_fn in models.items():
        print(f"Running {name}...")
        keys = jax.random.split(key, args.chains)
        key = keys[-1] # update key
        
        chains = []
        for k in keys[:args.chains]:
            samples = run_nuts_adapted(
                k,
                logprob_fn,
                init,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                target_acceptance_rate=args.target_accept,
                initial_step_size=args.init_step,
            )["samples"]
            chains.append(samples)
        
        chains = jnp.stack(chains, axis=0) # (chains, samples, dim)
        results[name] = chains.reshape(-1, chains.shape[-1]) # (total_samples, dim)

    # Save samples if requested
    if args.save_samples:
        print(f"Saving samples to {args.save_samples}...")
        os.makedirs(os.path.dirname(args.save_samples) or ".", exist_ok=True)
        # Prepare dictionary for saving
        save_dict = {"x": np.array(x), "col_names": col_names}
        for name, samples in results.items():
            # Sanitize name for key
            key_name = name.lower().replace("-", "_").replace(" ", "_")
            save_dict[key_name] = np.array(samples)
        
        np.savez_compressed(args.save_samples, **save_dict)

    # Plot Posterior of Outlier Proportion r = 1 - c
    # Note: c = sigmoid(rho)
    # r = max(0, 1-c)
    
    plt.figure(figsize=(10, 6))
    
    for name, samples in results.items():
        rho = samples[:, -1]
        c = jax.nn.sigmoid(rho)
        r = jnp.maximum(0.0, 1.0 - c)
        
        # Kernel Density Estimate or Histogram
        plt.hist(r, bins=30, alpha=0.5, label=name, density=True)
        
        # Calculate summary stats
        r_mean = jnp.mean(r)
        r_cred = jnp.percentile(r, jnp.array([2.5, 97.5]))
        print(f"{name}: Mean r = {r_mean:.3f}, 95% CI = [{r_cred[0]:.3f}, {r_cred[1]:.3f}]")

    plt.xlabel("Outlier Proportion (r)")
    plt.ylabel("Density")
    plt.title(f"Posterior Distribution of Outlier Proportion\nGamma={args.gamma}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "posterior_outlier_proportion.png"))
    plt.close()

    # Plot Posterior of Means (on original scale)
    # Only meaningful if d is manageable, let's stick to scatter for d=2 or just first 2 dims
    plt.figure(figsize=(8, 6))
    
    # Always plot first 2 dimensions if available
    if d >= 2:
        plt.scatter(x[:, 0], x[:, 1], c='black', alpha=0.3, label=f"Data ({col_names[0]} vs {col_names[1]})")
        plot_dims = (0, 1)
        plt.xlabel(col_names[0])
        plt.ylabel(col_names[1])
    else:
        # 1D case
        plt.hist(x[:, 0], bins=30, alpha=0.3, density=True, label="Data")
        plot_dims = (0,)
        plt.xlabel(col_names[0])
    
    colors = {"GBI-DPD": "blue", "Joint NLL": "green", "MMD-Bayes": "purple", "KSD-Bayes": "red"}
    
    for name, samples in results.items():
        if name not in colors: continue
        
        # Transform back to original scale
        # samples structure: [mu_0...mu_d-1, log_scale_0...log_scale_d-1, rho]
        
        if d >= 2:
            mu_std_x = samples[:, plot_dims[0]]
            mu_std_y = samples[:, plot_dims[1]]
            
            mu_orig_x = mu_std_x * std_x[plot_dims[0]] + mean_x[plot_dims[0]]
            mu_orig_y = mu_std_y * std_x[plot_dims[1]] + mean_x[plot_dims[1]]
            
            # Plot mean estimate (posterior mean)
            pm_x = jnp.mean(mu_orig_x)
            pm_y = jnp.mean(mu_orig_y)
            
            plt.scatter(pm_x, pm_y, s=100, marker='x', color=colors[name], label=f"{name} Mean", linewidth=2)
        else:
            # 1D Mean
            mu_std = samples[:, 0]
            mu_orig = mu_std * std_x[0] + mean_x[0]
            pm = jnp.mean(mu_orig)
            plt.axvline(pm, color=colors[name], label=f"{name} Mean", linewidth=2, linestyle="--")

    plt.title("Estimated Means (First 2 Dimensions)" if d >= 2 else "Estimated Mean")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "estimated_means.png"))
    plt.close()
    
    print("Done. Results saved to", args.output_dir)

if __name__ == "__main__":
    main()
