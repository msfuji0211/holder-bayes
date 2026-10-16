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
import warnings

# Suppress seaborn/pandas future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from hbayes.posteriors import (
    MVNPriorsConfig,
    MVNGibbsConfig,
    logprob_joint_regression_from_vector,
    logprob_theta_only_gamma_div_regression_from_vector,
    logprob_theta_only_nll_regression_from_vector,
)
from hbayes.models import LinearRegressionParams
from hbayes.sampling import run_nuts_adapted
from hbayes.diagnostics import arviz_rhat, arviz_ess

def load_dataset(path: str, target_col: str, feature_cols: list[str] = None, log_columns: list[str] = None):
    """
    Load dataset and split into X (features) and y (target).
    Adds bias term to X.
    Optionally apply log transform to specified columns.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        raise e

    # Drop index columns
    drop_cols = [c for c in df.columns if c.lower() in ['unnamed: 0', 'rownames', 'index', 'id']]
    if drop_cols:
        print(f"Dropping index-like columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Apply log transformation if requested
    if log_columns:
        for col in log_columns:
            if col in df.columns:
                print(f"Applying log transform to '{col}'")
                # Add small epsilon if there are zeros? Or assume data is positive.
                # robustbase data usually positive.
                if (df[col] <= 0).any():
                    print(f"Warning: Column '{col}' contains non-positive values. Log transform may result in NaNs/Infs.")
                df[col] = np.log(df[col])
                # Update column name to reflect transform?
                # Maybe keep original name for simplicity in args, or rename?
                # Let's rename in DF but we need to track it.
                # Actually, keeping original name is simpler for downstream 'target_col' reference
                # unless we want to be explicit.
                # Let's keep the values in place.
            else:
                print(f"Warning: Column '{col}' not found for log transform")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    y = df[target_col].values
    
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns {missing} not found.")
        X_df = df[feature_cols]
    else:
        # Use all other numeric columns
        X_df = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    
    feature_names = list(X_df.columns)
    X = X_df.values
    
    # Add bias term (column of ones)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    feature_names = ["Intercept"] + feature_names
    
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32), feature_names

def main():
    parser = argparse.ArgumentParser(description="Robust Bayesian Linear Regression Experiment")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--features", type=str, nargs='+', default=None, help="Feature columns (default: all other numeric)")
    parser.add_argument("--log_transform", type=str, nargs='+', default=None, help="Columns to apply log transform (e.g. brain body)")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=None, help="Explicit eta value. If None, determined by eta_mode")
    parser.add_argument("--eta_mode", type=str, choices=["fixed", "covdet"], default="fixed", help="Method to determine eta")
    parser.add_argument("--num_warmup", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.8)
    parser.add_argument("--init_step", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_regression")
    parser.add_argument("--save_samples", type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    print(f"Loading data from {args.data_path}...")
    X, y, feature_names = load_dataset(args.data_path, args.target, args.features, args.log_transform)
    n, d_in = X.shape # d_in includes intercept
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_names}")

    # Standardize Features (except Intercept) for stability
    # But keep original scale for y to interpret errors? Or standardize both?
    # Standardizing y helps with prior scaling. Let's standardize both and rescale later if needed.
    
    # Standardize X (skip intercept at index 0)
    X_mean = jnp.mean(X[:, 1:], axis=0)
    X_std = jnp.std(X[:, 1:], axis=0)
    X_std = jnp.where(X_std < 1e-9, 1.0, X_std)
    
    X_scaled = X.copy()
    X_scaled = X_scaled.at[:, 1:].set((X[:, 1:] - X_mean) / X_std)
    
    # Standardize y
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    y_scaled = (y - y_mean) / y_std
    
    # Determine eta
    eta_val = 1.0
    if args.eta is not None:
        eta_val = float(args.eta)
    elif args.eta_mode == "covdet":
        # For regression, we are modeling p(y|x). The generalized posterior logic
        # with DPD score often scales with the variance of the generative model.
        # But covdet usually refers to the determinant of the covariance of the response variables
        # (or residuals) for scaling the learning rate.
        # For simple linear regression y ~ N(mu, sigma^2), the "determinant" is just sigma^2.
        # However, we don't know sigma^2 yet.
        # A robust proxy for the scale of y is the variance of y (or robust equivalent).
        # We can use Var(y)^gamma as a heuristic, similar to |Sigma|^gamma in multivariate case.
        # Since we use standardized y (y_scaled), variance is 1.0. So eta=1.0 is appropriate for standardized data.
        # However, if the user requested covdet, maybe they want to account for the original scale?
        # But the inference is on standardized data.
        # Let's keep eta=1.0 for standardized data as a robust default under "covdet" logic for 1D output.
        # Or, we can use the covariance of the full joint (y, X) if we view it generatively?
        # Typically, eta = 1 is fine for standardized DPD.
        # Let's check gaussian_dpd_2d.py logic: eta = |Sigma|^gamma.
        # For standardized data |Sigma| approx 1. So eta approx 1.
        eta_val = 1.0
        print(f"Using covdet eta: {eta_val:.4f} (standardized data used, effective variance approx 1)")
    
    print(f"Final eta: {eta_val}")
    
    key = jax.random.PRNGKey(args.seed)

    # Configs
    pri = MVNPriorsConfig() # Default priors (N(0,10^2) for beta is good for standardized data)
    gcfg = MVNGibbsConfig(gamma=args.gamma, eta=eta_val, priors=pri)

    # Initial state: [beta (d_in), log_scale (1), rho (1)]
    # d_in is number of features + intercept
    init_beta = jnp.zeros(d_in)
    init_log_scale = jnp.zeros(1)
    init_rho = jnp.array([0.0])
    
    init_joint = jnp.concatenate([init_beta, init_log_scale, init_rho])
    init_nll = jnp.concatenate([init_beta, init_log_scale])

    print(f"Parameter dimension: {init_joint.shape[0]}")

    # Define models
    models = {
        "GBI-DPD": (
            logprob_joint_regression_from_vector(X_scaled, y_scaled, gcfg),
            init_joint,
            ["beta"] * d_in + ["log_scale", "rho"]
        ),
        "Gaussian NLL": (
            logprob_theta_only_nll_regression_from_vector(X_scaled, y_scaled, pri),
            init_nll,
            ["beta"] * d_in + ["log_scale"]
        ),
        "Gamma-Div": (
            logprob_theta_only_gamma_div_regression_from_vector(X_scaled, y_scaled, gcfg),
            init_nll,
            ["beta"] * d_in + ["log_scale"]
        ),
    }

    results = {}

    for name, (logprob_fn, init_val, param_names) in models.items():
        print(f"Running {name}...")
        keys = jax.random.split(key, args.chains)
        key = keys[-1]
        
        chains = []
        for k in keys[:args.chains]:
            samples = run_nuts_adapted(
                k,
                logprob_fn,
                init_val,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                target_acceptance_rate=args.target_accept,
                initial_step_size=args.init_step,
            )["samples"]
            chains.append(samples)
        
        chains = jnp.stack(chains, axis=0) # (chains, samples, dim)
        results[name] = chains.reshape(-1, chains.shape[-1]) # (total_samples, dim)

    # --- Analysis & Plotting ---
    
    # 1. Posterior of Outlier Proportion (GBI-DPD only)
    if "GBI-DPD" in results:
        samples = results["GBI-DPD"]
        rho = samples[:, -1]
        c = jax.nn.sigmoid(rho)
        r = jnp.maximum(0.0, 1.0 - c)
        
        plt.figure(figsize=(8, 5))
        plt.hist(r, bins=30, alpha=0.7, color='blue', density=True)
        r_mean = jnp.mean(r)
        r_cred = jnp.percentile(r, jnp.array([2.5, 97.5]))
        plt.title(f"Posterior Outlier Proportion (GBI-DPD)\nMean: {r_mean:.3f}, 95% CI: [{r_cred[0]:.3f}, {r_cred[1]:.3f}]")
        plt.xlabel("Outlier Proportion r")
        plt.ylabel("Density")
        plt.savefig(os.path.join(args.output_dir, "posterior_outlier_proportion.png"))
        plt.close()
        print(f"GBI-DPD Mean r = {r_mean:.3f}")

    # 2. Coefficients Comparison (Standardized scale)
    plt.figure(figsize=(10, 6))
    data_for_plot = []
    
    for name, samples in results.items():
        betas = samples[:, :d_in] # First d_in columns are betas
        for i in range(d_in):
            col_name = feature_names[i]
            for val in betas[:, i]:
                data_for_plot.append({"Method": name, "Feature": col_name, "Value": float(val)})
    
    df_plot = pd.DataFrame(data_for_plot)
    sns.boxplot(data=df_plot, x="Feature", y="Value", hue="Method")
    plt.title("Posterior Coefficients (Standardized Scale)")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "coefficients_comparison.png"))
    plt.close()

    # 3. Residual Plot & Outlier Probability (for GBI-DPD)
    # We need to compute p(y|x, theta) for each sample to estimate outlier probability
    # Similar to robustbase_2d.py logic
    
    if "GBI-DPD" in results:
        samples = results["GBI-DPD"]
        # Use mean estimate for theta to compute residuals for plotting
        beta_mean = jnp.mean(samples[:, :d_in], axis=0)
        scale_mean = jnp.exp(jnp.mean(samples[:, d_in])) # log_scale is at d_in
        
        y_pred_scaled = X_scaled @ beta_mean
        residuals_scaled = y_scaled - y_pred_scaled
        
        # Calculate outlier probability p_i
        # p_i approx 1 if likelihood is small relative to others?
        # Use the same logic as 2D: rank by likelihood
        
        # Compute likelihoods for all samples
        # shape: (n_samples, n_points)
        # memory heavy? sub-sample if needed
        n_sub = min(1000, samples.shape[0])
        idx_sub = np.random.choice(samples.shape[0], n_sub, replace=False)
        sub_samples = samples[idx_sub]
        
        ll_matrix = []
        for i in range(n_sub):
            p = sub_samples[i]
            beta = p[:d_in]
            scale = jnp.exp(p[d_in])
            mu = X_scaled @ beta
            # normal log pdf manually
            z = (y_scaled - mu) / scale
            ll = -0.5 * (jnp.log(2.0 * jnp.pi) + 2.0 * p[d_in] + z * z)
            ll_matrix.append(ll)
        ll_matrix = jnp.stack(ll_matrix) # (S, n)
        
        lik_matrix = jnp.exp(ll_matrix)
        c_sub = jax.nn.sigmoid(sub_samples[:, -1])
        r_sub = jnp.maximum(0.0, 1.0 - c_sub)
        K_sub = jnp.round(r_sub * n).astype(int)
        
        flags = np.zeros((n_sub, n))
        for i in range(n_sub):
            k = int(K_sub[i])
            if k > 0:
                # lowest k likelihoods
                idx = np.argsort(lik_matrix[i])[:k]
                flags[i, idx] = 1.0
        
        p_outlier = np.mean(flags, axis=0)
        
        # Plot Standardized Residuals vs Leverage
        # Compute Leverage (Hat Matrix Diagonal)
        # H = X(X^TX)^{-1}X^T
        # Use X_scaled (including intercept) for calculation
        try:
            # Add regularization for stability if needed, but usually robustbase data is fine
            XtX_inv = np.linalg.inv(X_scaled.T @ X_scaled)
            H = X_scaled @ XtX_inv @ X_scaled.T
            leverage = np.diag(H)
            
            # Rescale predictions to original scale
            y_pred_orig = y_pred_scaled * y_std + y_mean
            
            # Compute standardized residuals
            scale_mean_orig = scale_mean * y_std
            resid_std = (y - y_pred_orig) / scale_mean_orig
            
            plt.figure(figsize=(10, 6))
            
            # Scatter plot: Leverage vs Std Residuals
            sc = plt.scatter(leverage, resid_std, c=p_outlier, cmap='coolwarm', edgecolors='k', s=60, vmin=0, vmax=1)
            cb = plt.colorbar(sc)
            cb.set_label("Estimated Outlier Probability")
            
            # Add reference lines
            plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
            plt.axhline(2.5, color='red', linestyle='--', alpha=0.5, label='Residual Threshold (±2.5)')
            plt.axhline(-2.5, color='red', linestyle='--', alpha=0.5)
            
            # Leverage threshold (rule of thumb: 2p/n or 3p/n)
            p_dim = d_in # number of params
            lev_thresh = 2 * p_dim / n
            plt.axvline(lev_thresh, color='orange', linestyle='--', alpha=0.5, label=f'Lev. Threshold (2p/n={lev_thresh:.2f})')
            
            plt.xlabel("Leverage (Hat Value)")
            plt.ylabel("Standardized Residuals")
            plt.title(f"Leverage vs Standardized Residuals\n(Colored by Outlier Probability)")
            plt.legend(loc='upper right')
            
            # Annotate points of interest
            for i in range(n):
                is_high_lev = leverage[i] > lev_thresh
                is_large_resid = abs(resid_std[i]) > 2.5
                is_outlier_prob = p_outlier[i] > 0.5
                
                if is_high_lev or is_large_resid or is_outlier_prob:
                    plt.text(leverage[i], resid_std[i], f"{i}", fontsize=9, ha='right', va='bottom')
            
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "leverage_vs_residuals.png"))
            plt.close()
            
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix, skipping leverage plot.")

    # 4. Regression Fit Visualization
    colors = {"GBI-DPD": "blue", "Gaussian NLL": "green", "Gamma-Div": "orange"}
    
    # Case A: 1D Feature (plus intercept) -> Plot Regression Line with Band
    if d_in == 2:
        print("Plotting 1D Regression Fit...")
        plt.figure(figsize=(8, 6))
        
        # Raw data scatter
        # Color outliers if GBI-DPD results available
        feature_idx = 1 # 0 is intercept
        x_raw = X[:, feature_idx]
        
        if "GBI-DPD" in results:
            # Re-use p_outlier computed above
            plt.scatter(x_raw, y, c=p_outlier, cmap='coolwarm', edgecolors='k', label='Data', s=50, vmin=0, vmax=1)
            plt.colorbar(label="Outlier Probability")
        else:
            plt.scatter(x_raw, y, color='black', alpha=0.6, label='Data')

        # Generate grid
        x_grid = np.linspace(x_raw.min(), x_raw.max(), 200)
        X_grid = np.column_stack([np.ones_like(x_grid), x_grid])
        
        # Standardize grid
        X_grid_scaled = X_grid.copy()
        # Scale feature part using training stats
        X_grid_scaled[:, 1] = (x_grid - X_mean[0]) / X_std[0]
        
        for name, samples in results.items():
            if name not in colors: continue
            
            # Extract betas (posterior samples)
            betas = samples[:, :d_in] # (S, 2)
            
            # Compute regression lines: y = X @ beta
            # (200, 2) @ (2, S) -> (200, S)
            y_pred_grid_scaled = X_grid_scaled @ betas.T
            
            # Unscale y
            y_pred_grid = y_pred_grid_scaled * y_std + y_mean
            
            # Compute stats
            mu_pred = np.mean(y_pred_grid, axis=1)
            ci_pred = np.percentile(y_pred_grid, [2.5, 97.5], axis=1)
            
            plt.plot(x_grid, mu_pred, color=colors[name], label=f"{name} Mean", linewidth=2)
            plt.fill_between(x_grid, ci_pred[0], ci_pred[1], color=colors[name], alpha=0.2, label=f"{name} 95% CI")
            
        plt.xlabel(feature_names[1])
        plt.ylabel(args.target)
        plt.title("Regression Fit with 95% Credible Interval")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, "regression_fit_1d.png"))
        plt.close()

    # Case B: Multi-dimensional -> Observed vs Predicted Plot
    else:
        print("Plotting Observed vs Predicted...")
        plt.figure(figsize=(8, 6))
        # Determine plot limits across observed and predicted (posterior mean)
        y_min = float(y.min())
        y_max = float(y.max())

        preds: dict[str, jnp.ndarray] = {}
        for name, samples in results.items():
            if name not in colors:
                continue

            betas = jnp.mean(samples[:, :d_in], axis=0)
            y_pred_scaled = X_scaled @ betas
            y_pred = y_pred_scaled * y_std + y_mean
            preds[name] = y_pred

            y_min = min(y_min, float(y_pred.min()))
            y_max = max(y_max, float(y_pred.max()))

        # Add some margin for readability
        if y_max > y_min:
            margin = (y_max - y_min) * 0.05
        else:
            margin = 1.0
        y_min -= margin
        y_max += margin

        # Plot y=x reference
        plt.plot([y_min, y_max], [y_min, y_max], "k--", alpha=0.5)

        for name, y_pred in preds.items():
            plt.scatter(y, y_pred, alpha=0.6, label=name)
            
        plt.xlabel("Observed")
        plt.ylabel("Predicted (Posterior Mean)")
        plt.title("Observed vs Predicted")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, "observed_vs_predicted.png"))
        plt.close()

    # Save samples
    if args.save_samples:
        print(f"Saving samples to {args.save_samples}...")
        # Ensure directory exists
        os.makedirs(os.path.dirname(args.save_samples) or ".", exist_ok=True)
        
        save_dict = {
            "X": np.array(X), 
            "y": np.array(y), 
            "feature_names": feature_names,
            "X_mean": np.array(X_mean), "X_std": np.array(X_std),
            "y_mean": np.array(y_mean), "y_std": np.array(y_std)
        }
        for name, samples in results.items():
            key_name = name.lower().replace("-", "_")
            save_dict[key_name] = np.array(samples)
        np.savez_compressed(args.save_samples, **save_dict)

    print("Done.")

if __name__ == "__main__":
    main()
