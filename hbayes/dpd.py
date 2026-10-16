from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .models import NormalParams, MVNDiagParams, LinearRegressionParams, normal_log_pdf, mvn_diag_log_pdf, linear_regression_log_pdf


def _normal_constants(gamma: float) -> Tuple[float, float]:
    """Return constants used by A_gamma and B_gamma for a standard Normal kernel.

    Returns
    -------
    c1 : float
        (2*pi)^(-gamma/2)
    c2 : float
        (1+gamma)^(-1/2)
    """

    two_pi = 2.0 * jnp.pi
    c1 = two_pi ** (-0.5 * gamma)
    c2 = (1.0 + gamma) ** (-0.5)
    return c1, c2


def normal_A_gamma(params: NormalParams, gamma: float) -> jnp.ndarray:
    """A_gamma(theta) = ∫ p_theta(x)^{1+gamma} dx for univariate Normal.

    For p_theta = Normal(mean, scale),
    A_gamma = (2π)^(-gamma/2) * (1+gamma)^(-1/2) * scale^{-gamma}.
    """

    c1, c2 = _normal_constants(gamma)
    scale = jnp.exp(params.log_scale)
    return c1 * c2 * scale ** (-gamma)


def normal_B_gamma(params: NormalParams, x: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """B_gamma(θ) = (1/n) ∑ p_θ(x_i)^γ for univariate Normal.

    Implemented via log-sum-exp for numerical stability.
    """

    n = x.shape[0]
    log_pdf = normal_log_pdf(params, x)
    log_terms = gamma * log_pdf  # log p^gamma
    # compute mean of exp(log_terms)
    max_log = jnp.max(log_terms)
    b = jnp.exp(log_terms - max_log).mean() * jnp.exp(max_log)
    return b


def S_gamma(params: NormalParams, c: jnp.ndarray, x: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """DPD score S_gamma(\tilde p, c p_θ) for the univariate Normal.

    S_γ = γ c^{1+γ} A_γ(θ) - (1+γ) c^γ B_γ(θ).
    """

    a = normal_A_gamma(params, gamma)
    b = normal_B_gamma(params, x, gamma)
    c_pow_g = jnp.power(c, gamma)
    return gamma * c_pow_g * c * a - (1.0 + gamma) * c_pow_g * b


def c_star_and_curvature(
    params: NormalParams, x: jnp.ndarray, gamma: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute c*(θ) = min(1, B/A), together with A, B, and S'' at c*.

    For interior optimum (c* < 1): S''(c*) = γ(1+γ) A c*^{γ-1}.
    For boundary optimum (c* = 1): use the general second derivative
        S''(1) = γ(1+γ)[ (γ-1)(A - B) + A ].
    """

    a = normal_A_gamma(params, gamma)
    b = normal_B_gamma(params, x, gamma)
    c_unclipped = b / (a + 1e-30)
    c_star = jnp.minimum(1.0, c_unclipped)
    # curvature
    def interior_curv(c_val: jnp.ndarray) -> jnp.ndarray:
        return gamma * (1.0 + gamma) * a * jnp.power(c_val, gamma - 1.0)

    s2_interior = interior_curv(c_star)
    s2_boundary = gamma * (1.0 + gamma) * ((gamma - 1.0) * (a - b) + a)
    s2 = jnp.where(c_star < 1.0 - 1e-9, s2_interior, s2_boundary)
    return c_star, a, b, s2


# ================= Linear Regression DPD =================

def linear_regression_A_gamma(params: LinearRegressionParams, gamma: float) -> jnp.ndarray:
    """A_gamma(theta) for Linear Regression with Gaussian errors.
    
    The integral ∫ p(y|x)^{1+gamma} dy is constant w.r.t x for homoscedastic noise.
    It equals (2π)^(-gamma/2) * (1+gamma)^(-1/2) * scale^{-gamma}.
    """
    c1, c2 = _normal_constants(gamma)
    scale = jnp.exp(params.log_scale)
    return c1 * c2 * scale ** (-gamma)


def linear_regression_B_gamma(params: LinearRegressionParams, X: jnp.ndarray, y: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """B_gamma(theta) = (1/n) ∑ p(y_i|x_i)^gamma."""
    log_pdf = linear_regression_log_pdf(params, X, y)
    log_terms = gamma * log_pdf
    max_log = jnp.max(log_terms)
    return jnp.exp(log_terms - max_log).mean() * jnp.exp(max_log)



def S_gamma_at_c_star(
    params: NormalParams, x: jnp.ndarray, gamma: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (S_gamma(θ, c*), c*)."""

    c_star, a, b, _ = c_star_and_curvature(params, x, gamma)
    s = gamma * jnp.power(c_star, gamma) * c_star * a - (1.0 + gamma) * jnp.power(
        c_star, gamma
    ) * b
    return s, c_star


# ================= Multivariate Normal (Diagonal covariance) DPD =================

def _mvn_constants(gamma: float, d: int) -> Tuple[float, float]:
    """Return constants for MVN diag kernel: (2π)^(-γd/2), (1+γ)^(-d/2)."""

    two_pi = 2.0 * jnp.pi
    c1 = two_pi ** (-0.5 * gamma * d)
    c2 = (1.0 + gamma) ** (-0.5 * d)
    return c1, c2


def mvn_diag_A_gamma(params: MVNDiagParams, gamma: float) -> jnp.ndarray:
    """A_gamma(θ) = ∫ p_θ(x)^{1+γ} dx for MVN with diagonal covariance.

    With Σ = diag(σ^2), we have A_γ = (2π)^(-γd/2) (1+γ)^(-d/2) |Σ|^{-γ/2}.
    Here |Σ|^{-γ/2} = exp(-γ ∑ log σ).
    """

    d = params.mean.shape[0]
    c1, c2 = _mvn_constants(gamma, d)
    # sum log σ = sum log_scales
    log_det_sqrt = jnp.sum(params.log_scales)
    return c1 * c2 * jnp.exp(-gamma * log_det_sqrt)


def mvn_diag_B_gamma(params: MVNDiagParams, x: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """B_gamma(θ) = (1/n) ∑ p_θ(x_i)^γ for MVN diag."""

    if x.ndim == 1:
        x = x[None, :]
    log_pdf = mvn_diag_log_pdf(params, x)
    log_terms = gamma * log_pdf
    max_log = jnp.max(log_terms)
    return jnp.exp(log_terms - max_log).mean() * jnp.exp(max_log)


def S_gamma_mvn_diag(
    params: MVNDiagParams, c: jnp.ndarray, x: jnp.ndarray, gamma: float
) -> jnp.ndarray:
    """DPD score for MVN diag."""

    a = mvn_diag_A_gamma(params, gamma)
    b = mvn_diag_B_gamma(params, x, gamma)
    c_pow_g = jnp.power(c, gamma)
    return gamma * c_pow_g * c * a - (1.0 + gamma) * c_pow_g * b


def c_star_and_curvature_mvn_diag(
    params: MVNDiagParams, x: jnp.ndarray, gamma: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute c*, A, B, and S''(c*) for MVN diag."""

    a = mvn_diag_A_gamma(params, gamma)
    b = mvn_diag_B_gamma(params, x, gamma)
    c_unclipped = b / (a + 1e-30)
    c_star = jnp.minimum(1.0, c_unclipped)

    def interior_curv(c_val: jnp.ndarray) -> jnp.ndarray:
        return gamma * (1.0 + gamma) * a * jnp.power(c_val, gamma - 1.0)

    s2_interior = interior_curv(c_star)
    s2_boundary = gamma * (1.0 + gamma) * ((gamma - 1.0) * (a - b) + a)
    s2 = jnp.where(c_star < 1.0 - 1e-9, s2_interior, s2_boundary)
    return c_star, a, b, s2

