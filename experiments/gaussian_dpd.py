from __future__ import annotations

import os
from dataclasses import dataclass
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from hbayes.simulate import ContamConfig, simulate_contaminated_gaussian
from hbayes.posteriors import (
    PriorsConfig,
    GibbsConfig,
    logprob_joint_from_vector,
    logprob_joint_fixed_sigma_from_vector,
    logprob_joint_hier_from_vector,
    logprob_joint_hier_fixed_sigma_from_vector,
    logprob_theta_only_from_vector,
    logprob_theta_only_fixed_sigma_from_vector,
    logprob_theta_only_nll_from_vector,
    logprob_theta_only_nll_fixed_sigma_from_vector,
)
from hbayes.sampling import NutsConfig, run_nuts, run_nuts_adapted
from hbayes.diagnostics import arviz_rhat, arviz_ess


@dataclass
class ExptConfig:
    n: int = 200
    mu_true: float = 0.0
    sigma_true: float = 1.0
    contam_ratio: float = 0.2
    gamma: float = 0.5
    eta: float = 1.0
    seed: int = 123


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=None, help="DPD gamma")
    parser.add_argument("--contam", type=float, default=None, help="contamination ratio in simulated data")
    parser.add_argument("--n", type=int, default=None, help="number of samples in simulated data")
    parser.add_argument("--sigma", type=float, default=None, help="fix sigma to this value if provided")
    # outlier distribution options
    parser.add_argument("--outlier", type=str, choices=["t", "gaussian"], default="gaussian", help="outlier distribution: 't' (Student-t) or 'gaussian'")
    parser.add_argument("--t_df", type=float, default=None, help="df for Student-t outliers (default from ContamConfig)")
    parser.add_argument("--t_scale", type=float, default=None, help="scale for Student-t outliers (default from ContamConfig)")
    parser.add_argument("--out_mean", type=float, default=None, help="mean for Gaussian outliers; defaults to mu_true + 5*sigma_true if not provided")
    parser.add_argument("--out_scale", type=float, default=None, help="scale for Gaussian outliers (std dev)")
    parser.add_argument("--standardize", action="store_true", help="standardize data (zero mean, unit variance)")
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="generalized Bayes temperature; overrides eta_mode if provided",
    )
    parser.add_argument(
        "--eta_mode",
        type=str,
        choices=["fixed", "covdet"],
        default="fixed",
        help="how to set temperature when --eta is not given; 'covdet' uses (|det Sigma|^gamma)/n computed from pre-standardization data",
    )
    # prior for c: mixture Beta options
    parser.add_argument("--c_prior", type=str, choices=["beta", "mixbeta", "hierbeta", "logitnormal"], default="beta", help="prior type for c")
    parser.add_argument("--ln_mu0", type=float, default=0.8472978603872037, help="mean of rho prior (default logit(0.7))")
    parser.add_argument("--ln_sigma", type=float, default=1.5, help="std of rho prior")
    parser.add_argument("--ln_tau_scale", type=float, default=1.0, help="scale for Half-Cauchy prior on tau (not yet sampled)")
    parser.add_argument("--c_w", type=float, default=0.5, help="mixture weight w for Beta mixture (component1 weight)")
    parser.add_argument("--c_a1", type=float, default=10.0, help="alpha of Beta component 1")
    parser.add_argument("--c_b1", type=float, default=2.0, help="beta of Beta component 1")
    parser.add_argument("--c_a2", type=float, default=6.0, help="alpha of Beta component 2")
    parser.add_argument("--c_b2", type=float, default=4.0, help="beta of Beta component 2")
    parser.add_argument("--c_am", type=float, default=1.0, help="a_m for hierarchical Beta prior (m ~ Beta(a_m,b_m))")
    parser.add_argument("--c_bm", type=float, default=1.0, help="b_m for hierarchical Beta prior")
    parser.add_argument("--c_ak", type=float, default=2.0, help="a_kappa for hierarchical Beta prior (kappa ~ Gamma(a_kappa,b_kappa))")
    parser.add_argument("--c_bk", type=float, default=0.2, help="b_kappa for hierarchical Beta prior (rate)")
    args = parser.parse_args()

    cfg = ExptConfig()
    if args.gamma is not None:
        cfg.gamma = float(args.gamma)
    if args.contam is not None:
        cfg.contam_ratio = float(args.contam)
    if args.n is not None:
        cfg.n = int(args.n)
    key = jax.random.PRNGKey(cfg.seed)

    # Simulate contaminated data
    contam_cfg = ContamConfig(
        n=cfg.n,
        mean=cfg.mu_true,
        scale=cfg.sigma_true,
        contam_ratio=cfg.contam_ratio,
        out_dist=args.outlier,
    )
    if args.t_df is not None:
        contam_cfg.t_df = float(args.t_df)
    if args.t_scale is not None:
        contam_cfg.t_scale = float(args.t_scale)
    if args.out_mean is not None:
        contam_cfg.out_mean = float(args.out_mean)
    if args.out_scale is not None:
        contam_cfg.out_scale = float(args.out_scale)

    x = simulate_contaminated_gaussian(key, contam_cfg)

    # Optional data standardization and temperature (eta) selection
    # Compute sample covariance determinant on original (pre-standardization) data.
    # Use unbiased normalization (divide by n-1) consistently.
    n_obs = float(x.shape[0])
    denom = jnp.maximum(n_obs - 1.0, 1.0)
    ddof = 1
    if x.ndim == 1:
        cov_det = float(jnp.var(x, ddof=ddof))
    else:
        x_centered = x - jnp.mean(x, axis=0)
        cov = (x_centered.T @ x_centered) / denom
        cov_det = float(jnp.linalg.det(cov))

    # Standardize data if requested (univariate: (x-mean)/std; multivariate: per-dimension scaling)
    if args.standardize:
        if x.ndim == 1:
            mean_x = jnp.mean(x)
            std_x = jnp.sqrt(jnp.maximum(jnp.var(x, ddof=1), 1e-12))
            #x = (x - mean_x) / std_x
        else:
            x_centered = x - jnp.mean(x, axis=0)
            std_vec = jnp.sqrt(jnp.maximum(jnp.var(x_centered, axis=0, ddof=1), 1e-12))
            #x = x_centered / std_vec

    # Determine eta　
    if args.eta is not None:
        cfg.eta = float(args.eta)
    else:
        if args.standardize and args.eta_mode == "covdet":
            cfg.eta = (float(max(cov_det, 1e-12)) ** cfg.gamma)
        # else keep default cfg.eta

    # Posteriors
    pri = PriorsConfig(mu0=0.0, sigma0=10.0, m_log_sigma=0.0, s_log_sigma=1.0, a_c=1, b_c=1)
    if args.c_prior == "mixbeta":
        pri.use_c_mixture = True
        pri.w_c = float(args.c_w)
        pri.a_c1 = float(args.c_a1)
        pri.b_c1 = float(args.c_b1)
        pri.a_c2 = float(args.c_a2)
        pri.b_c2 = float(args.c_b2)
    elif args.c_prior == "hierbeta":
        pri.use_c_hierarchical = True
        pri.a_m = float(args.c_am)
        pri.b_m = float(args.c_bm)
        pri.a_kappa = float(args.c_ak)
        pri.b_kappa = float(args.c_bk)
    elif args.c_prior == "logitnormal":
        pri.use_c_logitnormal = True
        pri.ln_mu0 = float(args.ln_mu0)
        pri.ln_sigma_mu = float(args.ln_sigma)
        pri.ln_tau_scale = float(args.ln_tau_scale)
    gcfg = GibbsConfig(gamma=cfg.gamma, eta=cfg.eta, priors=pri)

    # Joint sampling (mu, log_sigma, rho) with multiple chains
    if args.sigma is None:
        if args.c_prior == "hierbeta":
            logprob_joint = logprob_joint_hier_from_vector(x, gcfg)
            init = jnp.array([0.0, jnp.log(1.0), 0.0, 0.0, jnp.log(10.0)])  # [mu, log_sigma, rho, mu_m, log_kappa]
            joint_dim = 5
        else:
            logprob_joint = logprob_joint_from_vector(x, gcfg)
            init = jnp.array([0.0, jnp.log(1.0), 0.0])
            joint_dim = 3
        log_sigma_fixed = None
    else:
        log_sigma_fixed = jnp.log(float(args.sigma))
        if args.c_prior == "hierbeta":
            logprob_joint = logprob_joint_hier_fixed_sigma_from_vector(x, gcfg, float(log_sigma_fixed))
            init = jnp.array([0.0, 0.0, 0.0, jnp.log(10.0)])  # [mu, rho, mu_m, log_kappa]
            joint_dim = 4
        else:
            logprob_joint = logprob_joint_fixed_sigma_from_vector(x, gcfg, float(log_sigma_fixed))
            init = jnp.array([0.0, 0.0])  # [mu, rho]
            joint_dim = 2
    nuts_cfg = NutsConfig(step_size=0.1, num_warmup=1000, num_samples=1000)
    keys = jax.random.split(key, 4)
    chains_joint = []
    for k in keys:
        chains_joint.append(run_nuts(k, logprob_joint, init, nuts_cfg)["samples"])  # (S,3)
    chains_joint = jnp.stack(chains_joint, axis=0)  # (C,S,D)
    samples_joint = chains_joint.reshape((-1, joint_dim))
    if log_sigma_fixed is None:
        if args.c_prior == "hierbeta":
            mu_s, logsig_s, rho_s, mu_m_s, log_kappa_s = (
                samples_joint[:, 0], samples_joint[:, 1], samples_joint[:, 2], samples_joint[:, 3], samples_joint[:, 4]
            )
        else:
            mu_s, logsig_s, rho_s = samples_joint[:, 0], samples_joint[:, 1], samples_joint[:, 2]
    else:
        if args.c_prior == "hierbeta":
            mu_s, rho_s, mu_m_s, log_kappa_s = samples_joint[:, 0], samples_joint[:, 1], samples_joint[:, 2], samples_joint[:, 3]
        else:
            mu_s, rho_s = samples_joint[:, 0], samples_joint[:, 1]
        logsig_s = jnp.full((samples_joint.shape[0],), log_sigma_fixed)
    c_s = jax.nn.sigmoid(rho_s)

    # Laplace/Profile は現在の実装では不安定となるケースがあるため一時的に無効化
    do_theta_only = False
    if do_theta_only:
        if log_sigma_fixed is None:
            logprob_lap = logprob_theta_only_from_vector(x, gcfg, method="laplace")
            init_theta = jnp.array([0.0, jnp.log(1.0)])
            theta_dim = 2
        else:
            logprob_lap = logprob_theta_only_fixed_sigma_from_vector(
                x, gcfg, method="laplace", log_sigma_fixed=float(log_sigma_fixed)
            )
            init_theta = jnp.array([0.0])
            theta_dim = 1
        chains_lap = []
        for k in keys:
            chains_lap.append(run_nuts(k, logprob_lap, init_theta, nuts_cfg)["samples"])  # (S,D)
        chains_lap = jnp.stack(chains_lap, axis=0)
        samples_lap = chains_lap.reshape((-1, theta_dim))
    else:
        chains_lap = None
        samples_lap = None

    # Profile baseline theta-only sampling (multi-chain)
    if do_theta_only:
        if log_sigma_fixed is None:
            logprob_prof = logprob_theta_only_from_vector(x, gcfg, method="profile")
        else:
            logprob_prof = logprob_theta_only_fixed_sigma_from_vector(
                x, gcfg, method="profile", log_sigma_fixed=float(log_sigma_fixed)
            )
        chains_prof = []
        for k in keys:
            chains_prof.append(run_nuts(k, logprob_prof, init_theta, nuts_cfg)["samples"])  # (S,D)
        chains_prof = jnp.stack(chains_prof, axis=0)
        samples_prof = chains_prof.reshape((-1, theta_dim))
    else:
        chains_prof = None
        samples_prof = None

    # NLL baseline (theta only) multi-chain, adapted
    if log_sigma_fixed is None:
        logprob_nll = logprob_theta_only_nll_from_vector(x, pri)
        nll_init = init_theta if do_theta_only else jnp.array([0.0, jnp.log(1.0)])
        nll_dim = 2
    else:
        logprob_nll = logprob_theta_only_nll_fixed_sigma_from_vector(
            x, pri, log_sigma_fixed=float(log_sigma_fixed)
        )
        nll_init = jnp.array([0.0])
        nll_dim = 1
    chains_nll = []
    for k in keys:
        chains_nll.append(
            run_nuts_adapted(
                k,
                logprob_nll,
                nll_init,
                num_warmup=1000,
                num_samples=1000,
                initial_step_size=1.0,
                target_acceptance_rate=0.8,
                is_mass_matrix_diagonal=True,
            )["samples"]
        )
    chains_nll = jnp.stack(chains_nll, axis=0)
    samples_nll = chains_nll.reshape((-1, nll_dim))

    # Diagnostics (R-hat, ESS) and save
    if log_sigma_fixed is None:
        rhat_joint = arviz_rhat(chains_joint, ["mu", "log_sigma", "rho"])  # rho=logit(c)
        rhat_lap = arviz_rhat(chains_lap, ["mu", "log_sigma"]) if do_theta_only else None
        rhat_prof = arviz_rhat(chains_prof, ["mu", "log_sigma"]) if do_theta_only else None
        rhat_nll = arviz_rhat(chains_nll, ["mu", "log_sigma"]) 
        ess_joint = arviz_ess(chains_joint, ["mu", "log_sigma", "rho"])
        ess_lap = arviz_ess(chains_lap, ["mu", "log_sigma"]) if do_theta_only else None
        ess_prof = arviz_ess(chains_prof, ["mu", "log_sigma"]) if do_theta_only else None
        ess_nll = arviz_ess(chains_nll, ["mu", "log_sigma"]) 
    else:
        rhat_joint = arviz_rhat(chains_joint, ["mu", "rho"])  # fixed sigma
        rhat_lap = arviz_rhat(chains_lap, ["mu"]) if do_theta_only else None
        rhat_prof = arviz_rhat(chains_prof, ["mu"]) if do_theta_only else None
        rhat_nll = arviz_rhat(chains_nll, ["mu"]) 
        ess_joint = arviz_ess(chains_joint, ["mu", "rho"]) 
        ess_lap = arviz_ess(chains_lap, ["mu"]) if do_theta_only else None
        ess_prof = arviz_ess(chains_prof, ["mu"]) if do_theta_only else None
        ess_nll = arviz_ess(chains_nll, ["mu"]) 

    import json, os
    os.makedirs("outputs", exist_ok=True)
    diag = {
        "rhat": {
            "joint": rhat_joint.tolist(),
            "laplace": (rhat_lap.tolist() if do_theta_only else None),
            "profile": (rhat_prof.tolist() if do_theta_only else None),
            "nll": rhat_nll.tolist(),
        },
        "ess": {
            "joint": ess_joint.tolist(),
            "laplace": (ess_lap.tolist() if do_theta_only else None),
            "profile": (ess_prof.tolist() if do_theta_only else None),
            "nll": ess_nll.tolist(),
        },
    }
    with open("outputs/diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    # Quick summaries
    def summ(name, arr):
        print(f"{name}: mean={arr.mean():.3f}, sd={arr.std():.3f}")

    print("Joint posterior (mu, sigma, c):")
    summ("mu", mu_s)
    summ("sigma", jnp.exp(logsig_s))
    summ("c", c_s)

    if do_theta_only:
        print("Laplace theta-only:")
        summ("mu", samples_lap[:, 0])
        summ("sigma", jnp.exp(samples_lap[:, 1]))

    if do_theta_only:
        print("Profile theta-only:")
        summ("mu", samples_prof[:, 0])
        summ("sigma", jnp.exp(samples_prof[:, 1]))

    print("NLL baseline theta-only:")
    summ("mu", samples_nll[:, 0])
    summ("sigma", jnp.exp(samples_nll[:, 1]))

    # Simple plot for mu posteriors
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(mu_s, bins=50, alpha=0.5, label="joint")
    if do_theta_only:
        ax.hist(samples_lap[:, 0], bins=50, alpha=0.5, label="laplace")
        ax.hist(samples_prof[:, 0], bins=50, alpha=0.5, label="profile")
    ax.hist(samples_nll[:, 0], bins=50, alpha=0.5, label="nll")
    ax.axvline(cfg.mu_true, color="k", linestyle="--", label="true mu")
    ax.legend()
    ax.set_title("Posterior of mu")
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/post_mu.png", dpi=150)
    plt.close(fig)

    # Posterior histogram for c (joint) with true c line
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.hist(c_s, bins=50, alpha=0.6, label="joint c")
    ax2.axvline(1.0 - cfg.contam_ratio, color="k", linestyle="--", label="true c")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_title("Posterior of c")
    ax2.legend()
    fig2.savefig("outputs/post_c.png", dpi=150)
    plt.close(fig2)

    # Posterior histogram for contamination ratio r = max(0, 1-c)
    r_s = jnp.maximum(0.0, 1.0 - c_s)
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))
    ax3.hist(r_s, bins=50, alpha=0.6, label="joint r")
    ax3.axvline(cfg.contam_ratio, color="k", linestyle="--", label="true r")
    ax3.set_xlim(0.0, 1.0)
    ax3.set_title("Posterior of contamination ratio r=max(0,1-c)")
    ax3.legend()
    fig3.savefig("outputs/post_contam.png", dpi=150)
    plt.close(fig3)


if __name__ == "__main__":
    main()


