from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import blackjax

from .models import LinearRegressionParams, linear_regression_log_pdf, NormalParams, normal_log_pdf
from .posteriors import linear_regression_dpd_joint_log_posterior


@dataclass
class NutsConfig:
    step_size: float = 0.1
    inverse_mass: float | None = None
    num_warmup: int = 500
    num_samples: int = 1000


def run_nuts(
    key: jax.Array,
    logprob: Callable[[jnp.ndarray], jnp.ndarray],
    init_position: jnp.ndarray,
    config: NutsConfig,
) -> Dict[str, jnp.ndarray]:
    """Run BlackJAX NUTS and return dictionary of samples and info."""

    inverse_mass = (
        config.inverse_mass
        if config.inverse_mass is not None
        else jnp.ones(init_position.shape, dtype=jnp.result_type(init_position))
    )
    nuts = blackjax.nuts(
        logprob,
        config.step_size,
        inverse_mass,
    )

    # Keep initialization and transition randomness independent.
    init_key, transition_key = jax.random.split(key)
    # blackjax 1.2: init(position, rng_key=None)
    state = nuts.init(init_position, rng_key=init_key)

    @jax.jit
    def one_step(state, key):
        state, info = nuts.step(key, state)
        return state, info

    # Warmup (fixed-step; adapt step size externally if needed)
    keys = jax.random.split(transition_key, config.num_warmup + config.num_samples)
    for i in range(config.num_warmup):
        state, _ = one_step(state, keys[i])

    samples = []
    infos = []
    for i in range(config.num_warmup, config.num_warmup + config.num_samples):
        state, info = one_step(state, keys[i])
        samples.append(state.position)
        infos.append(info)

    return {
        "samples": jnp.stack(samples, axis=0),
        "infos": infos,
        "last_state": state,
    }


def run_nuts_adapted(
    key: jax.Array,
    logprob: Callable[[jnp.ndarray], jnp.ndarray],
    init_position: jnp.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.8,
    is_mass_matrix_diagonal: bool = True,
) -> Dict[str, jnp.ndarray]:
    """Run NUTS with window adaptation (step size + mass matrix)."""

    adapt = blackjax.window_adaptation(
        blackjax.nuts,
        logprob,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
        progress_bar=False,
    )

    # A JAX key must not be consumed again after adaptation.
    warmup_key, sampling_key = jax.random.split(key)
    (state, parameters), _ = adapt.run(
        warmup_key, position=init_position, num_steps=num_warmup
    )

    nuts = blackjax.nuts(logprob, **parameters)

    @jax.jit
    def one_step(state, key):
        state, info = nuts.step(key, state)
        return state, info

    keys = jax.random.split(sampling_key, num_samples)

    def scan_step(state, key):
        state, info = nuts.step(key, state)
        return state, (state.position, info)

    state, (samples, stacked_infos) = jax.jit(
        lambda initial, transition_keys: jax.lax.scan(scan_step, initial, transition_keys)
    )(state, keys)
    infos = [jax.tree_util.tree_map(lambda value: value[i], stacked_infos)
             for i in range(num_samples)]

    return {
        "samples": samples,
        "infos": infos,
        "last_state": state,
        "parameters": parameters,
    }


def run_mcmc_dpd_linear_regression(
    key: jax.Array,
    X: jnp.ndarray,
    y: jnp.ndarray,
    gamma: float,
    n_warmup: int = 1000,
    n_samples: int = 2000,
    beta_init: jnp.ndarray | None = None,
    log_scale_init: jnp.ndarray | None = None,
    c_init: float | None = None,
) -> Dict[str, jnp.ndarray]:
    """
    Runs MCMC for Linear Regression using DPD joint posterior.
    Parameter vector is flattened [beta, log_scale, logit_c] or similar.
    Here we use the `linear_regression_dpd_joint_log_posterior` from posteriors.py.
    """
    
    n, d = X.shape
    
    if beta_init is None:
        beta_init = jnp.zeros(d)
    if log_scale_init is None:
        log_scale_init = jnp.zeros(())
    if c_init is None:
        c_init = 0.5
        
    # We sample in unconstrained space. 
    # c is in [0, 1], so we use logit(c).
    logit_c_init = jax.scipy.special.logit(jnp.array(c_init))
    
    # Initial position array
    # Structure: [beta_0, ..., beta_{d-1}, log_scale, logit_c]
    init_position = jnp.concatenate([
        beta_init,
        log_scale_init[None],
        logit_c_init[None]
    ])
    
    def logprob_fn(position):
        # Unpack
        beta = position[:d]
        log_scale = position[d]
        logit_c = position[d+1]
        c = jax.nn.sigmoid(logit_c)
        
        # Log Jacobian for logit transform: log(c(1-c))
        log_det_jac = jnp.log(c) + jnp.log(1.0 - c)
        
        params = LinearRegressionParams(beta=beta, log_scale=log_scale)
        
        # Prior log probs (can be customized, here assuming flat or encoded in posterior)
        # Using the defined posterior function
        # Note: posteriors.py usually expects params and c separately
        
        lp = linear_regression_dpd_joint_log_posterior(
            params, c, X, y, gamma, prior_log_prob=None
        )
        return lp + log_det_jac

    results = run_nuts_adapted(
        key=key,
        logprob=logprob_fn,
        init_position=init_position,
        num_warmup=n_warmup,
        num_samples=n_samples,
    )
    
    samples_flat = results["samples"]
    
    # Unpack samples
    beta_samples = samples_flat[:, :d]
    log_scale_samples = samples_flat[:, d]
    logit_c_samples = samples_flat[:, d+1]
    c_star_samples = jax.nn.sigmoid(logit_c_samples)
    
    return {
        "beta": beta_samples,
        "log_scale": log_scale_samples,
        "c_star": c_star_samples,
        "samples_flat": samples_flat
    }
