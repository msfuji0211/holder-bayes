from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .models import NormalParams, MVNDiagParams, LinearRegressionParams, normal_log_pdf, mvn_diag_log_pdf, linear_regression_log_pdf
from .dpd import (
    S_gamma,
    c_star_and_curvature,
    S_gamma_mvn_diag,
    c_star_and_curvature_mvn_diag,
    mvn_diag_A_gamma,
    mvn_diag_B_gamma,
    linear_regression_A_gamma,
    linear_regression_B_gamma,
)


@dataclass
class PriorsConfig:
    # mean ~ Normal(mu0, sigma0^2)
    mu0: float = 0.0
    sigma0: float = 10.0
    # log_scale ~ Normal(m_log_sigma, s_log_sigma^2)
    m_log_sigma: float = 0.0
    s_log_sigma: float = 1.0
    # c prior: either
    #  - single Beta(a_c,b_c)
    #  - mixture: w_c*Beta(a_c1,b_c1)+(1-w_c)*Beta(a_c2,b_c2)
    #  - hierarchical: c|m,κ ~ Beta(mκ,(1-m)κ),  m~Beta(a_m,b_m), κ~Gamma(a_kappa,b_kappa)
    a_c: float = 1.0
    b_c: float = 1.0
    use_c_mixture: bool = False
    w_c: float = 0.5
    a_c1: float = 10.0
    b_c1: float = 2.0
    a_c2: float = 6.0
    b_c2: float = 4.0
    use_c_hierarchical: bool = False
    a_m: float = 1.0
    b_m: float = 1.0
    a_kappa: float = 2.0
    b_kappa: float = 0.2
    # logistic-normal prior for rho=logit(c):
    # rho ~ Normal(ln_mu0, ln_sigma_mu^2), ln_sigma_mu>0; tau prior Half-Cauchy(ln_tau_scale)
    use_c_logitnormal: bool = False
    ln_mu0: float = 0.8472978603872037  # logit(0.7)
    ln_sigma_mu: float = 1.5
    ln_tau_scale: float = 1.0


@dataclass
class GibbsConfig:
    gamma: float = 0.5
    eta: float = 1.0
    holder_phi: str = "dpd"
    holder_kappa: float = 1.0
    priors: PriorsConfig = field(default_factory=PriorsConfig)


def log_prior_theta(params: NormalParams, priors: PriorsConfig) -> jnp.ndarray:
    """Log prior for (mean, log_scale)."""

    mu_term = -0.5 * (
        jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(priors.sigma0)
    ) - 0.5 * ((params.mean - priors.mu0) / priors.sigma0) ** 2
    ls_term = -0.5 * (
        jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(priors.s_log_sigma)
    ) - 0.5 * ((params.log_scale - priors.m_log_sigma) / priors.s_log_sigma) ** 2
    return mu_term + ls_term


def _log_beta_density(c: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    c_clip = jnp.clip(c, 1e-12, 1.0 - 1e-12)
    return (a - 1.0) * jnp.log(c_clip) + (b - 1.0) * jnp.log(1.0 - c_clip) - jsp.special.betaln(a, b)


def log_prior_c_from_rho(rho: jnp.ndarray, priors: PriorsConfig) -> jnp.ndarray:
    """Log prior for c = sigmoid(rho) with Beta prior on c and Jacobian.

    If priors.use_c_mixture is True, use mixture:
        w_c * Beta(a_c1, b_c1) + (1 - w_c) * Beta(a_c2, b_c2)
    Else use single Beta(a_c, b_c).
    """

    # Evaluate in logit coordinates: sigmoid(rho) can round to 1 in
    # float32, and 1 - 1e-12 is also exactly 1 in that dtype.
    log_c = jax.nn.log_sigmoid(rho)
    log_one_minus_c = jax.nn.log_sigmoid(-rho)

    def beta_log_density(a, b):
        return (
            (a - 1.0) * log_c
            + (b - 1.0) * log_one_minus_c
            - jsp.special.betaln(a, b)
        )

    if priors.use_c_logitnormal:
        # rho ~ Normal(ln_mu0, ln_sigma_mu^2)
        mu = priors.ln_mu0
        sigma = jnp.maximum(priors.ln_sigma_mu, 1e-6)
        log_density = -0.5 * (
            jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(sigma) + ((rho - mu) / sigma) ** 2
        )
    elif priors.use_c_hierarchical:
        # In the hierarchical model, this function should not be used; use the joint variant
        raise ValueError("log_prior_c_from_rho should not be used with hierarchical c prior")
    elif priors.use_c_mixture:
        w = jnp.clip(priors.w_c, 1e-6, 1.0 - 1e-6)
        log_comp1 = beta_log_density(priors.a_c1, priors.b_c1)
        log_comp2 = beta_log_density(priors.a_c2, priors.b_c2)
        # log-sum-exp of mixture components
        m = jnp.maximum(log_comp1 + jnp.log(w), log_comp2 + jnp.log(1.0 - w))
        log_density = m + jnp.log(
            jnp.exp(log_comp1 + jnp.log(w) - m) + jnp.exp(log_comp2 + jnp.log(1.0 - w) - m)
        )
    else:
        log_density = beta_log_density(priors.a_c, priors.b_c)

    # add Jacobian for rho -> c
    log_jac = log_c + log_one_minus_c
    return log_density + log_jac


def log_gibbs_posterior_joint(
    x: jnp.ndarray,
    params: NormalParams,
    rho: jnp.ndarray,
    config: GibbsConfig,
) -> jnp.ndarray:
    """Log generalized Bayes posterior for joint (theta, c) with DPD score.

    c = sigmoid(rho).
    log posterior ∝ log prior(θ) + log prior(c) − n η S_γ(θ,c).
    """

    c = jax.nn.sigmoid(rho)
    n = x.shape[0]
    s_val = S_gamma(params, c, x, config.gamma)
    lp_c = log_prior_c_from_rho(rho, config.priors)
    # Half-Cauchy prior on tau is only meaningful for logit-normal; here we use fixed sigma (ln_sigma_mu) in priors
    return log_prior_theta(params, config.priors) + lp_c - n * config.eta * s_val


def log_gibbs_posterior_laplace(
    x: jnp.ndarray,
    params: NormalParams,
    config: GibbsConfig,
) -> jnp.ndarray:
    """Log post. with c marginalized by Laplace around c*(θ).

    log π(θ|x) ≈ log π(θ) + log π(c* ) + 1/2 log(2π/(nη S''(c*))) − nη S_γ(θ, c*).
    """

    n = x.shape[0]
    gamma = config.gamma
    eta = config.eta
    priors = config.priors

    c_star, _, _, s2 = c_star_and_curvature(params, x, gamma)
    # Evaluate score at c* via S_gamma definition
    s_val = S_gamma(params, c_star, x, gamma)

    # log prior for theta
    lp_theta = log_prior_theta(params, priors)

    # log prior for c evaluated at c* (no Jacobian here; it's direct c-density)
    c_clip = jnp.clip(c_star, 1e-12, 1.0 - 1e-12)
    if priors.use_c_hierarchical:
        # For Laplace theta-only, we do not model m, kappa. Fallback to Beta(a_c,b_c) if provided,
        # or to an improper weak default Beta(1,1). Users are encouraged to use joint sampling with hierarchical prior.
        lp_c_star = _log_beta_density(c_clip, priors.a_c, priors.b_c)
    elif priors.use_c_mixture:
        w = jnp.clip(priors.w_c, 1e-6, 1.0 - 1e-6)
        log_c1 = _log_beta_density(c_clip, priors.a_c1, priors.b_c1)
        log_c2 = _log_beta_density(c_clip, priors.a_c2, priors.b_c2)
        m2 = jnp.maximum(log_c1 + jnp.log(w), log_c2 + jnp.log(1.0 - w))
        lp_c_star = m2 + jnp.log(
            jnp.exp(log_c1 + jnp.log(w) - m2) + jnp.exp(log_c2 + jnp.log(1.0 - w) - m2)
        )
    else:
        lp_c_star = _log_beta_density(c_clip, priors.a_c, priors.b_c)


def log_prior_m_from_mu(mu: jnp.ndarray, priors: PriorsConfig) -> jnp.ndarray:
    """Log prior for m in (0,1) parameterized by mu with m=sigmoid(mu)."""
    m = jax.nn.sigmoid(mu)
    log_density = _log_beta_density(m, priors.a_m, priors.b_m)
    log_jac = jnp.log(jnp.clip(m, 1e-12, 1.0)) + jnp.log(jnp.clip(1.0 - m, 1e-12, 1.0))
    return log_density + log_jac


def log_prior_kappa_from_logk(log_kappa: jnp.ndarray, priors: PriorsConfig) -> jnp.ndarray:
    """Log prior for kappa>0 parameterized by log_kappa; Gamma(a_kappa, b_kappa) with rate b."""
    kappa = jnp.exp(log_kappa)
    a = priors.a_kappa
    b = priors.b_kappa
    log_density = (a - 1.0) * jnp.log(jnp.clip(kappa, 1e-30, jnp.inf)) - b * kappa + a * jnp.log(b) - jsp.special.gammaln(a)
    # Jacobian for log transform
    return log_density + log_kappa


def log_prior_c_from_rho_given_m_kappa(rho: jnp.ndarray, m: jnp.ndarray, kappa: jnp.ndarray) -> jnp.ndarray:
    c = jax.nn.sigmoid(rho)
    a = m * kappa
    b = (1.0 - m) * kappa
    log_density = _log_beta_density(c, a, b)
    log_jac = jnp.log(jnp.clip(c, 1e-12, 1.0)) + jnp.log(jnp.clip(1.0 - c, 1e-12, 1.0))
    return log_density + log_jac


def log_gibbs_posterior_joint_hier(
    x: jnp.ndarray,
    params: NormalParams,
    rho: jnp.ndarray,
    mu_m: jnp.ndarray,
    log_kappa: jnp.ndarray,
    config: GibbsConfig,
) -> jnp.ndarray:
    """Joint log posterior with hierarchical Beta prior for c.

    Variables: theta(params), c via rho, m via mu_m, kappa via log_kappa.
    """

    n = x.shape[0]
    m = jax.nn.sigmoid(mu_m)
    kappa = jnp.exp(log_kappa)
    s_val = S_gamma(params, jax.nn.sigmoid(rho), x, config.gamma)
    return (
        log_prior_theta(params, config.priors)
        + log_prior_m_from_mu(mu_m, config.priors)
        + log_prior_kappa_from_logk(log_kappa, config.priors)
        + log_prior_c_from_rho_given_m_kappa(rho, m, kappa)
        - n * config.eta * s_val
    )


def log_gibbs_posterior_profile(
    x: jnp.ndarray,
    params: NormalParams,
    config: GibbsConfig,
) -> jnp.ndarray:
    """Log post. with c profiled at c*(θ) (not Bayesian marginalization)."""

    n = x.shape[0]
    gamma = config.gamma
    eta = config.eta
    c_star, _, _, _ = c_star_and_curvature(params, x, gamma)
    s_val = S_gamma(params, c_star, x, gamma)
    return log_prior_theta(params, config.priors) - n * eta * s_val


# Log-prob functions compatible with BlackJAX (vector parameters)

def unpack_theta(theta_vec: jnp.ndarray) -> NormalParams:
    return NormalParams(mean=theta_vec[0], log_scale=theta_vec[1])


def logprob_joint_from_vector(
    x: jnp.ndarray, config: GibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function mapping concatenated [mean, log_scale, rho] to log posterior."""

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        params = NormalParams(mean=z[0], log_scale=z[1])
        rho = z[2]
        return log_gibbs_posterior_joint(x, params, rho, config)

    return _logprob


def logprob_joint_hier_from_vector(
    x: jnp.ndarray, config: GibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function mapping [mean, log_scale, rho, mu_m, log_kappa] to log posterior."""

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        params = NormalParams(mean=z[0], log_scale=z[1])
        rho = z[2]
        mu_m = z[3]
        log_kappa = z[4]
        return log_gibbs_posterior_joint_hier(x, params, rho, mu_m, log_kappa, config)

    return _logprob


def logprob_joint_fixed_sigma_from_vector(
    x: jnp.ndarray, config: GibbsConfig, log_sigma_fixed: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function mapping [mean, rho] to log posterior with fixed log_sigma."""

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        params = NormalParams(mean=z[0], log_scale=log_sigma_fixed)
        rho = z[1]
        return log_gibbs_posterior_joint(x, params, rho, config)

    return _logprob


def logprob_joint_hier_fixed_sigma_from_vector(
    x: jnp.ndarray, config: GibbsConfig, log_sigma_fixed: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function mapping [mean, rho, mu_m, log_kappa] to log posterior with fixed log_sigma."""

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        params = NormalParams(mean=z[0], log_scale=log_sigma_fixed)
        rho = z[1]
        mu_m = z[2]
        log_kappa = z[3]
        return log_gibbs_posterior_joint_hier(x, params, rho, mu_m, log_kappa, config)

    return _logprob


def logprob_theta_only_from_vector(
    x: jnp.ndarray, config: GibbsConfig, method: str
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function mapping [mean, log_scale] to log posterior for theta-only.

    method ∈ {"laplace", "profile"}.
    """

    if method not in {"laplace", "profile"}:
        raise ValueError("method must be 'laplace' or 'profile'")

    def _logprob(theta_vec: jnp.ndarray) -> jnp.ndarray:
        params = unpack_theta(theta_vec)
        if method == "laplace":
            return log_gibbs_posterior_laplace(x, params, config)
        return log_gibbs_posterior_profile(x, params, config)

    return _logprob


def logprob_theta_only_fixed_sigma_from_vector(
    x: jnp.ndarray, config: GibbsConfig, method: str, log_sigma_fixed: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function mapping [mean] to log posterior for theta-only with fixed sigma."""

    if method not in {"laplace", "profile"}:
        raise ValueError("method must be 'laplace' or 'profile'")

    def _logprob(theta_vec: jnp.ndarray) -> jnp.ndarray:
        params = NormalParams(mean=theta_vec[0], log_scale=log_sigma_fixed)
        if method == "laplace":
            return log_gibbs_posterior_laplace(x, params, config)
        return log_gibbs_posterior_profile(x, params, config)

    return _logprob


# ============== Standard Gaussian NLL baseline (no c) ==================

def log_posterior_nll(
    x: jnp.ndarray,
    params: NormalParams,
    priors: PriorsConfig,
) -> jnp.ndarray:
    """Conventional Bayes: log π(θ) + ∑ log p_θ(x_i)."""

    ll = jnp.sum(normal_log_pdf(params, x))
    return log_prior_theta(params, priors) + ll


def logprob_theta_only_nll_from_vector(
    x: jnp.ndarray, priors: PriorsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def _logprob(theta_vec: jnp.ndarray) -> jnp.ndarray:
        params = unpack_theta(theta_vec)
        return log_posterior_nll(x, params, priors)

    return _logprob


def logprob_theta_only_nll_fixed_sigma_from_vector(
    x: jnp.ndarray, priors: PriorsConfig, log_sigma_fixed: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def _logprob(theta_vec: jnp.ndarray) -> jnp.ndarray:
        params = unpack_theta(jnp.array([theta_vec[0], log_sigma_fixed]))
        return log_posterior_nll(x, params, priors)

    return _logprob


# ================= Multivariate (diagonal covariance) Gibbs posteriors =================

@dataclass
class MVNPriorsConfig(PriorsConfig):
    # Overload to allow vectorized priors if desired in the future.
    pass


@dataclass
class MVNGibbsConfig(GibbsConfig):
    priors: MVNPriorsConfig = field(default_factory=MVNPriorsConfig)


def log_prior_theta_mvn(params: MVNDiagParams, priors: MVNPriorsConfig) -> jnp.ndarray:
    """Independent Normal priors on each mean_k and log_scale_k.

    Uses the same scalar hyperparameters from PriorsConfig for simplicity.
    """

    mu_term = -0.5 * (
        params.mean.shape[0] * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.log(priors.sigma0)
        + jnp.sum(((params.mean - priors.mu0) / priors.sigma0) ** 2)
    )
    ls_term = -0.5 * (
        params.log_scales.shape[0] * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.log(priors.s_log_sigma)
        + jnp.sum(((params.log_scales - priors.m_log_sigma) / priors.s_log_sigma) ** 2)
    )
    return mu_term + ls_term


def log_prior_c_from_rho_mvn(rho: jnp.ndarray, priors: MVNPriorsConfig) -> jnp.ndarray:
    # reuse scalar Beta/logit-normal prior on c
    from .posteriors import log_prior_c_from_rho  # local reuse to avoid duplication

    return log_prior_c_from_rho(rho, priors)


def log_gibbs_posterior_joint_mvn(
    x: jnp.ndarray,
    params: MVNDiagParams,
    rho: jnp.ndarray,
    config: MVNGibbsConfig,
) -> jnp.ndarray:
    c = jax.nn.sigmoid(rho)
    n = x.shape[0] if x.ndim > 1 else 1
    s_val = S_holder_mvn_diag(
        params,
        c,
        x,
        config.gamma,
        holder_phi=config.holder_phi,
        holder_kappa=config.holder_kappa,
    )
    lp_c = log_prior_c_from_rho_mvn(rho, config.priors)
    return log_prior_theta_mvn(params, config.priors) + lp_c - n * config.eta * s_val


def logprob_joint_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vector order: [mu_1..mu_d, log_sigma_1..log_sigma_d, rho]."""

    d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        mean = z[:d]
        log_scales = z[d : 2 * d]
        rho = z[-1]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        return log_gibbs_posterior_joint_mvn(x, params, rho, config)

    return _logprob


def logprob_theta_only_nll_mvn_from_vector(
    x: jnp.ndarray, priors: MVNPriorsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Conventional Bayes baseline for MVN diag.

    Vector order: [mu_1..mu_d, log_sigma_1..log_sigma_d].
    """

    d = x.shape[1]

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        mean = z[:d]
        log_scales = z[d:]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)
        ll = jnp.sum(mvn_diag_log_pdf(params, x))
        return log_prior_theta_mvn(params, priors) + ll

    return _logprob


# ================= MVN (diag covariance) Gamma-divergence =================

def logprob_theta_only_gamma_div_mvn_from_vector(
    x: jnp.ndarray, config: MVNGibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Gamma-divergence generalized Bayes posterior over theta only (no c).

    Uses the standard gamma-cross-entropy objective (up to additive constants):
        L(θ) = 1/(1+γ) * log A_γ(θ) - 1/γ * log B_γ(θ)
    where
        A_γ(θ) = ∫ p_θ(x)^{1+γ} dx,
        B_γ(θ) = (1/n) ∑ p_θ(x_i)^γ.

    The generalized posterior is:
        log π(θ|x) ∝ log π(θ) − n η L(θ).
    """

    if config.gamma <= 0.0:
        raise ValueError("gamma-divergence requires gamma > 0")

    n = float(x.shape[0])
    gamma = float(config.gamma)

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[1] if x.ndim == 2 else int(jnp.squeeze(x).shape[0])
        mean = z[:d]
        log_scales = z[d:]
        params = MVNDiagParams(mean=mean, log_scales=log_scales)

        a_val = mvn_diag_A_gamma(params, gamma)
        b_val = mvn_diag_B_gamma(params, x, gamma)

        log_a = jnp.log(a_val + 1e-30)
        log_b = jnp.log(b_val + 1e-30)

        loss = (1.0 / (1.0 + gamma)) * log_a - (1.0 / gamma) * log_b

        lp_theta = log_prior_theta_mvn(params, config.priors)
        return lp_theta - n * config.eta * loss

    return _logprob


def logprob_theta_only_gamma_div_regression_from_vector(
    X: jnp.ndarray, y: jnp.ndarray, config: GibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Gamma-divergence generalized Bayes posterior for Linear Regression (theta only)."""

    if config.gamma <= 0.0:
        raise ValueError("gamma-divergence requires gamma > 0")

    n = float(X.shape[0])
    gamma = float(config.gamma)
    d_in = X.shape[1]

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        beta = z[:d_in]
        log_scale = z[d_in]
        params = LinearRegressionParams(beta=beta, log_scale=log_scale)

        a_val = linear_regression_A_gamma(params, gamma)
        b_val = linear_regression_B_gamma(params, X, y, gamma)

        log_a = jnp.log(a_val + 1e-30)
        log_b = jnp.log(b_val + 1e-30)

        loss = (1.0 / (1.0 + gamma)) * log_a - (1.0 / gamma) * log_b

        lp_theta = log_prior_theta_regression(params, config.priors)
        return lp_theta - n * config.eta * loss

    return _logprob


# ================= Linear Regression Gibbs posteriors =================

def log_prior_theta_regression(params: LinearRegressionParams, priors: PriorsConfig) -> jnp.ndarray:
    """Prior for Linear Regression parameters (beta, log_scale).
    
    beta ~ Normal(mu0, sigma0^2) (i.i.d for now)
    log_scale ~ Normal(m_log_sigma, s_log_sigma^2)
    """
    beta_dim = params.beta.shape[0]
    
    # beta prior
    beta_term = -0.5 * (
        beta_dim * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.log(priors.sigma0)
        + jnp.sum(((params.beta - priors.mu0) / priors.sigma0) ** 2)
    )
    
    # log_scale prior
    ls_term = -0.5 * (
        jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(priors.s_log_sigma)
    ) - 0.5 * ((params.log_scale - priors.m_log_sigma) / priors.s_log_sigma) ** 2
    
    return beta_term + ls_term


def _holder_phi_value(z: jnp.ndarray, gamma: float, holder_phi: str, holder_kappa: float) -> jnp.ndarray:
    zc = jnp.clip(z, 1e-30, 1e30)
    phi = holder_phi.lower().replace("-", "_")
    if phi in {"dpd", "beta_div", "beta-div"}:
        # Recovers the DPD-style loss when plugged into H_n
        return gamma - (1.0 + gamma) * zc
    if phi in {"powerlaw", "power_law", "polynomial", "tsallis"}:
        kappa = jnp.maximum(jnp.asarray(holder_kappa, dtype=zc.dtype), 1e-12)
        return (1.0 - jnp.power(zc, kappa)) / kappa
    if phi in {"exponential", "exp"}:
        kappa = jnp.maximum(jnp.asarray(holder_kappa, dtype=zc.dtype), 1e-12)
        return 1.0 - jnp.exp(-kappa * zc)
    if phi in {"exponential_saturated", "exp_saturated", "saturated_exponential", "exp_sat"}:
        kappa = jnp.maximum(jnp.asarray(holder_kappa, dtype=zc.dtype), 1e-12)
        return -1.0 - ((1.0 + gamma) / kappa) * (1.0 - jnp.exp(-kappa * (zc - 1.0)))
    if phi in {"rational_saturated", "sat_rational", "rational_sat"}:
        kappa = jnp.maximum(jnp.asarray(holder_kappa, dtype=zc.dtype), 1e-12)
        denom = jnp.maximum(1.0 + kappa * (zc - 1.0), 1e-12)
        return -1.0 - (1.0 + gamma) * ((zc - 1.0) / denom)
    raise ValueError(f"Unknown holder_phi: {holder_phi}")


def S_holder_mvn_diag(
    params: MVNDiagParams,
    c: jnp.ndarray,
    x: jnp.ndarray,
    gamma: float,
    holder_phi: str = "dpd",
    holder_kappa: float = 1.0,
) -> jnp.ndarray:
    """Holder score for diagonal MVN with selectable phi."""
    a = mvn_diag_A_gamma(params, gamma)
    b = mvn_diag_B_gamma(params, x, gamma)
    c_clip = jnp.clip(c, 1e-12, 1.0)
    ratio = (b / jnp.maximum(a, 1e-30)) / c_clip
    phi_val = _holder_phi_value(ratio, gamma=gamma, holder_phi=holder_phi, holder_kappa=holder_kappa)
    return phi_val * jnp.power(c_clip, 1.0 + gamma) * a


def S_gamma_regression(
    params: LinearRegressionParams,
    c: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    gamma: float,
    holder_phi: str = "dpd",
    holder_kappa: float = 1.0,
) -> jnp.ndarray:
    """Holder score for Linear Regression with selectable phi."""
    a = linear_regression_A_gamma(params, gamma)
    b = linear_regression_B_gamma(params, X, y, gamma)
    c_clip = jnp.clip(c, 1e-12, 1.0)
    ratio = (b / jnp.maximum(a, 1e-30)) / c_clip
    phi_val = _holder_phi_value(ratio, gamma=gamma, holder_phi=holder_phi, holder_kappa=holder_kappa)
    return phi_val * jnp.power(c_clip, 1.0 + gamma) * a


def log_gibbs_posterior_joint_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    params: LinearRegressionParams,
    rho: jnp.ndarray,
    config: GibbsConfig,
) -> jnp.ndarray:
    """Joint log posterior for Linear Regression with DPD score."""
    c = jax.nn.sigmoid(rho)
    n = X.shape[0]
    s_val = S_gamma_regression(
        params,
        c,
        X,
        y,
        config.gamma,
        holder_phi=config.holder_phi,
        holder_kappa=config.holder_kappa,
    )
    
    # We reuse PriorsConfig for regression as well
    lp_c = log_prior_c_from_rho(rho, config.priors)
    return log_prior_theta_regression(params, config.priors) + lp_c - n * config.eta * s_val


def logprob_joint_regression_from_vector(
    X: jnp.ndarray, y: jnp.ndarray, config: GibbsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Vector order: [beta_0..beta_d, log_scale, rho]."""
    d_in = X.shape[1]

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        # z has size d_in + 2
        beta = z[:d_in]
        log_scale = z[d_in]
        rho = z[d_in + 1]
        
        params = LinearRegressionParams(beta=beta, log_scale=log_scale)
        return log_gibbs_posterior_joint_regression(X, y, params, rho, config)

    return _logprob


def logprob_theta_only_nll_regression_from_vector(
    X: jnp.ndarray, y: jnp.ndarray, priors: PriorsConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Standard Bayesian Linear Regression (NLL baseline)."""
    d_in = X.shape[1]

    def _logprob(z: jnp.ndarray) -> jnp.ndarray:
        beta = z[:d_in]
        log_scale = z[d_in]
        params = LinearRegressionParams(beta=beta, log_scale=log_scale)
        
        ll = jnp.sum(linear_regression_log_pdf(params, X, y))
        return log_prior_theta_regression(params, priors) + ll

    return _logprob


def linear_regression_dpd_joint_log_posterior(
    params: LinearRegressionParams,
    c: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    gamma: float,
    prior_log_prob: Callable[[LinearRegressionParams], jnp.ndarray] | None = None
) -> jnp.ndarray:
    """
    Log-posterior for DPD joint model:
    log p(theta, c | X, y) \propto -n * S_gamma(theta, c) + log p(theta) + log p(c)
    
    Here we assume prior_log_prob handles p(theta).
    p(c) is typically Beta(1,1) or similar, handled by caller or implicit flat.
    For simplicity in this helper, we just compute the likelihood part (-n * S_gamma)
    and add prior_log_prob if provided.
    
    Note: S_gamma returns the DPD score. The generalized posterior uses -n * S_gamma.
    """
    n = X.shape[0]
    s_val = S_gamma_regression(params, c, X, y, gamma)
    
    lp = -n * s_val
    
    if prior_log_prob is not None:
        lp += prior_log_prob(params)
        
    # We assume p(c) is handled externally (e.g. by log_prior_c_from_rho) or is uniform.
    return lp
