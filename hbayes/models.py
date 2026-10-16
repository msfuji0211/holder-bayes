from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class NormalParams:
    mean: jnp.ndarray  # shape: ()
    log_scale: jnp.ndarray  # shape: ()


def normal_log_pdf(params: NormalParams, x: jnp.ndarray) -> jnp.ndarray:
    """Log pdf of univariate Normal with (mean, log_scale)."""

    z = (x - params.mean) / jnp.exp(params.log_scale)
    return -0.5 * (
        jnp.log(2.0 * jnp.pi) + 2.0 * params.log_scale + z * z
    )


# ============== Diagonal-covariance Multivariate Normal (for d>=1) ==============

@dataclass
class MVNDiagParams:
    mean: jnp.ndarray  # shape: (d,)
    log_scales: jnp.ndarray  # shape: (d,) where scales = exp(log_scales)


def mvn_diag_log_pdf(params: MVNDiagParams, x: jnp.ndarray) -> jnp.ndarray:
    """Log pdf of diagonal-covariance MVN with parameters (mean, log_scales).

    Parameters
    ----------
    params : MVNDiagParams
        mean in R^d and per-dimension log scales.
    x : jnp.ndarray
        Observations of shape (n, d).

    Returns
    -------
    jnp.ndarray
        Log-density for each observation, shape (n,).
    """

    # Ensure two-dimensional input
    if x.ndim == 1:
        x = x[None, :]
    d = params.mean.shape[0]
    scales = jnp.exp(params.log_scales)
    z = (x - params.mean) / scales  # (n, d)
    quad = jnp.sum(z * z, axis=1)
    log_det_term = jnp.sum(2.0 * params.log_scales)
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det_term + quad)


# ============== Linear Regression (Gaussian errors) ==============

@dataclass
class LinearRegressionParams:
    beta: jnp.ndarray      # shape: (d_in,) where X has shape (n, d_in)
    log_scale: jnp.ndarray # shape: () scalar log standard deviation

def linear_regression_log_pdf(params: LinearRegressionParams, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Log pdf of Gaussian linear regression y ~ N(X @ beta, scale^2).
    
    Parameters
    ----------
    params : LinearRegressionParams
        Coefficients beta and scalar log_scale.
    X : jnp.ndarray
        Design matrix of shape (n, d_in). Should include bias column if needed.
    y : jnp.ndarray
        Target vector of shape (n,).
        
    Returns
    -------
    jnp.ndarray
        Log-density for each observation, shape (n,).
    """
    scale = jnp.exp(params.log_scale)
    mu = X @ params.beta
    z = (y - mu) / scale
    return -0.5 * (jnp.log(2.0 * jnp.pi) + 2.0 * params.log_scale + z * z)

