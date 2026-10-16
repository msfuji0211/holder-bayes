"""Nakagawa--Hashimoto gamma-posterior with an explicitly proper scale prior.

The old logarithmic gamma-cross-entropy target remains in posteriors.py for
provenance only. The paper GammaDiv baseline uses this module in both models.
All coordinates here are physical [locations, log(sigma)]; the prior is Normal
on log(sigma), truncated below log(0.05), not Normal on the sampler transform.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import jax.numpy as jnp
from jax.scipy.special import log_ndtr

DEFINITION = "nakagawa_hashimoto_monotone_gamma_v1"
SIGMA_MIN = 0.05
ETA = 1.0


def specification():
    return {"definition": DEFINITION, "eta": ETA, "sigma_min": SIGMA_MIN,
            "scale_prior": "Normal(log_sigma; 0, 1) truncated below log(0.05)",
            "location_prior": "independent Normal(0, 10^2)",
            "reference": "Nakagawa and Hashimoto (2020), DOI:10.1080/03610926.2018.1543765",
            "regression_coordinates": "standardized_response",
            "density_coordinates": "original_data"}


def _settings(config):
    gamma, eta = float(config.gamma), float(config.eta)
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("GammaDiv gamma must be finite and positive")
    if eta != ETA:
        raise ValueError("The paper GammaDiv protocol fixes eta=1 for every condition")
    if (config.priors.mu0, config.priors.sigma0, config.priors.m_log_sigma, config.priors.s_log_sigma) != (0., 10., 0., 1.):
        raise ValueError("The paper GammaDiv protocol fixes Normal(0,10^2) locations and truncated Normal(0,1) log-scales")
    return gamma, eta, config.priors


def _identify(target, gamma, eta, priors, model):
    target.gamma_protocol = {"definition": DEFINITION, "gamma": gamma, "eta": eta,
                             "priors": asdict(priors), "sigma_min": SIGMA_MIN, "model": model}
    return target


def log_prior(locations, log_scales, priors):
    """Normalized independent location and truncated log-scale priors."""
    d, s = locations.size, log_scales.size
    lp = -.5 * (d * jnp.log(2 * jnp.pi * priors.sigma0**2)
                + jnp.sum(((locations - priors.mu0) / priors.sigma0)**2))
    lp += -.5 * (s * jnp.log(2 * jnp.pi * priors.s_log_sigma**2)
                 + jnp.sum(((log_scales - priors.m_log_sigma) / priors.s_log_sigma)**2))
    lp -= s * log_ndtr((priors.m_log_sigma - math.log(SIGMA_MIN)) / priors.s_log_sigma)
    return jnp.where(jnp.all(log_scales >= math.log(SIGMA_MIN)), lp, -jnp.inf)


def _reward(log_pdf, log_a, gamma, eta):
    # -n eta tilde_d = eta/gamma sum_i [ f_i^gamma / A^(gamma/(1+gamma)) - 1 ].
    # expm1 preserves the gamma -> 0 limit without subtracting two large terms.
    log_ratio = gamma * log_pdf - (gamma / (1 + gamma)) * log_a
    return (eta / gamma) * jnp.sum(jnp.expm1(log_ratio))


def logprob_regression(X, y, config):
    """Homoscedastic conditional extension; integrate f(y|x)^1+gamma over y."""
    gamma, eta, priors = _settings(config)
    d = X.shape[1]

    def target(z):
        beta, ls = z[:d], z[d:d+1]
        residual = (y - X @ beta) * jnp.exp(-ls[0])
        log_pdf = -.5 * jnp.log(2 * jnp.pi) - ls[0] - .5 * residual**2
        log_a = -.5 * gamma * jnp.log(2 * jnp.pi) - .5 * jnp.log1p(gamma) - gamma * ls[0]
        return log_prior(beta, ls, priors) + _reward(log_pdf, log_a, gamma, eta)

    return _identify(target, gamma, eta, priors, "regression")


def logprob_density(x, config):
    """Diagonal Gaussian density; the scale support applies to every axis."""
    gamma, eta, priors = _settings(config)
    d = x.shape[1]

    def target(z):
        mu, ls = z[:d], z[d:2*d]
        standardized = (x - mu) * jnp.exp(-ls)
        log_pdf = -.5 * d * jnp.log(2 * jnp.pi) - jnp.sum(ls) - .5 * jnp.sum(standardized**2, axis=1)
        log_a = -.5 * d * gamma * jnp.log(2 * jnp.pi) - .5 * d * jnp.log1p(gamma) - gamma * jnp.sum(ls)
        return log_prior(mu, ls, priors) + _reward(log_pdf, log_a, gamma, eta)

    return _identify(target, gamma, eta, priors, "density")
