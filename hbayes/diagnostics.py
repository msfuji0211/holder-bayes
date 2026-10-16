from __future__ import annotations

import jax.numpy as jnp
import arviz as az


def to_inference_data(chains: jnp.ndarray, var_names: list[str]) -> az.InferenceData:
    """Convert (n_chains, n_draws, dim) to ArviZ InferenceData with given names."""

    data_dict = {name: chains[..., i] for i, name in enumerate(var_names)}
    return az.from_dict(posterior=data_dict)


def arviz_rhat(chains: jnp.ndarray, var_names: list[str]) -> jnp.ndarray:
    idata = to_inference_data(chains, var_names)
    r = az.rhat(idata, method="rank")
    return jnp.array([r[var].values for var in var_names])


def arviz_ess(chains: jnp.ndarray, var_names: list[str]) -> jnp.ndarray:
    idata = to_inference_data(chains, var_names)
    e = az.ess(idata, method="bulk")
    return jnp.array([e[var].values for var in var_names])


def arviz_ess_tail(chains: jnp.ndarray, var_names: list[str]) -> jnp.ndarray:
    idata = to_inference_data(chains, var_names)
    e = az.ess(idata, method="tail")
    return jnp.array([e[var].values for var in var_names])


