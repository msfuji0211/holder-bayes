from __future__ import annotations

from typing import Callable
import jax.numpy as jnp

from .models import NormalParams, normal_log_pdf


# Standard NLL (no c)

def loglik_gaussian(x: jnp.ndarray, params: NormalParams) -> jnp.ndarray:
    return jnp.sum(normal_log_pdf(params, x))


def logposterior_gaussian(
    x: jnp.ndarray, params: NormalParams, log_prior_theta: Callable[[NormalParams], jnp.ndarray]
) -> jnp.ndarray:
    return log_prior_theta(params) + loglik_gaussian(x, params)


