"""Truncate the existing log-scale prior using a smooth sampling transform.

This changes the statistical target explicitly. The Gaussian PDF and its
analytic A_gamma remain unchanged; the parameter support is restricted.
Apply the wrapper to each comparison method when using a common scale prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class BoundedScaleTarget:
    """Wrap a log density whose contiguous scale coordinates are log(sigma).

    In sampling coordinates u, log(sigma) = log(sigma_min) + softplus(u).
    The existing prior on log(sigma) is truncated at log(sigma_min), rather
    than changed to a normal prior on u. Its normalizing constant is omitted
    because it is constant within a fixed target. Include that constant when
    comparing marginal likelihoods across different scale supports.

    `to_physical` restores the original position layout for diagnostics,
    posterior prediction and outlier scores. An adapted mass matrix and
    step size refer to sampling coordinates, not physical coordinates.
    """

    base_logprob: Callable
    log_scale_start: int
    sigma_min: float | tuple[float, ...]

    def __post_init__(self):
        floors = np.asarray(self.sigma_min, dtype=float)
        if floors.ndim == 0:
            floors = floors.reshape(1)
        if floors.ndim != 1 or floors.size == 0:
            raise ValueError("sigma_min must be a scalar or a nonempty vector")
        if not np.isfinite(floors).all() or np.any(floors <= 0):
            raise ValueError("All scale lower bounds must be finite and positive")
        if not isinstance(self.log_scale_start, int) or self.log_scale_start < 0:
            raise ValueError("log_scale_start must be a nonnegative integer")
        object.__setattr__(self, "sigma_min", tuple(float(value) for value in floors))

    def _range(self, position):
        lo = self.log_scale_start
        hi = lo + len(self.sigma_min)
        if position.ndim < 1 or hi > position.shape[-1]:
            raise ValueError("Scale coordinates are outside the position layout")
        return lo, hi

    @staticmethod
    def _as_float(position):
        z = jnp.asarray(position)
        return z.astype(jnp.result_type(z, 1.0))

    def to_physical(self, sampling_position):
        """Accept one position or arrays with position in the last dimension."""
        z = self._as_float(sampling_position)
        lo, hi = self._range(z)
        log_floors = jnp.asarray(np.log(self.sigma_min), dtype=z.dtype)
        scales = log_floors + jax.nn.softplus(z[..., lo:hi])
        return jnp.concatenate([z[..., :lo], scales, z[..., hi:]], axis=-1)

    def to_unconstrained(self, physical_position):
        """Convert host-side initial positions; require sigma > sigma_min."""
        z_np = np.asarray(physical_position)
        lo, hi = self._range(z_np)
        delta_np = z_np[..., lo:hi] - np.log(self.sigma_min)
        if not np.isfinite(z_np).all() or np.any(delta_np <= 0):
            raise ValueError("Initial positions must be finite and sigma > sigma_min")
        z = self._as_float(physical_position)
        delta = z[..., lo:hi] - jnp.asarray(np.log(self.sigma_min), dtype=z.dtype)
        # inverse softplus, stable for both small and large positive delta
        u = delta + jnp.log(-jnp.expm1(-delta))
        if not np.isfinite(np.asarray(u)).all():
            raise ValueError("Initial scales must be representably above sigma_min in the inference dtype")
        return jnp.concatenate([z[..., :lo], u, z[..., hi:]], axis=-1)

    def log_density(self, sampling_position):
        """Single-position scalar log density for NUTS, including Jacobian."""
        z = self._as_float(sampling_position)
        lo, hi = self._range(z)
        log_jacobian = jnp.sum(jax.nn.log_sigmoid(z[lo:hi]))
        return self.base_logprob(self.to_physical(z)) + log_jacobian

    def metadata(self):
        return {
            "scale_support_version": "truncated_log_scale_softplus_v1",
            "sigma_min": list(self.sigma_min),
            "log_scale_start": self.log_scale_start,
            "prior": "existing_log_scale_prior_truncated_below_log_sigma_min",
            "sample_layout": "convert_with_to_physical_before_existing_analysis",
        }
