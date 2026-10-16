from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp


@dataclass
class ContamConfig:
    n: int
    mean: float | Tuple[float, float] = 0.0
    scale: float | Tuple[float, float] = 1.0
    contam_ratio: float = 0.2  # 1 - c0 in the paper
    # contamination component options
    # outlier distribution: 't' (Student-t) or 'gaussian'
    out_dist: str = "gaussian"
    # for Student-t outliers: t ~ t_df, scale t_scale, centered at 0
    t_df: float = 3.0
    t_scale: float = 5.0
    # for Gaussian outliers: N(out_mean, out_scale^2)
    # if out_mean is None, we set it to mean + 5*scale at runtime
    out_mean: Optional[float | Tuple[float, float]] = None
    out_scale: float | Tuple[float, float] = 5.0


def sample_student_t(key: jax.Array, df: float, scale: float, size: Tuple[int, ...]) -> jnp.ndarray:
    # t = z / sqrt(u/df), z ~ N(0,1), u ~ chi2(df)
    key_z, key_u = jax.random.split(key)
    z = jax.random.normal(key_z, shape=size)
    u = jax.random.gamma(key_u, df / 2.0, shape=size) * 2.0
    t = z / jnp.sqrt(u / df)
    return scale * t


def simulate_contaminated_gaussian(key: jax.Array, cfg: ContamConfig) -> jnp.ndarray:
    """Mixture: (1-ε) N(μ,σ) + ε OutlierDist.

    OutlierDist is either Student-t(ν, 0, s_t) or Gaussian N(m_out, s_out^2).
    Uses a Bernoulli mask to avoid shape issues.
    """

    key_main, key_out, key_mask, key_perm = jax.random.split(key, 4)
    n = int(cfg.n)
    # Support 1D or 2D via tuple configuration
    if isinstance(cfg.mean, tuple):
        # 2D case (diagonal covariance)
        mean_vec = jnp.array(cfg.mean, dtype=jnp.float32)
        if isinstance(cfg.scale, tuple):
            scale_vec = jnp.array(cfg.scale, dtype=jnp.float32)
        else:
            scale_vec = jnp.array([cfg.scale, cfg.scale], dtype=jnp.float32)
        x_in = mean_vec + scale_vec * jax.random.normal(key_main, shape=(n, 2))
        if cfg.out_dist.lower() in ("t", "student-t", "studentt"):
            # iid across dims
            t_samp = sample_student_t(key_out, cfg.t_df, cfg.t_scale, size=(n, 2))
            x_out = t_samp
        elif cfg.out_dist.lower() in ("gaussian", "normal"):
            if cfg.out_mean is None:
                m_out_vec = mean_vec + 5.0 * scale_vec
            else:
                m_out_vec = jnp.array(cfg.out_mean, dtype=jnp.float32)
            if isinstance(cfg.out_scale, tuple):
                out_scale_vec = jnp.array(cfg.out_scale, dtype=jnp.float32)
            else:
                out_scale_vec = jnp.array([cfg.out_scale, cfg.out_scale], dtype=jnp.float32)
            x_out = m_out_vec + out_scale_vec * jax.random.normal(key_out, shape=(n, 2))
        else:
            raise ValueError("out_dist must be 't' or 'gaussian'")
    else:
        # 1D case (backward-compatible)
        x_in = cfg.mean + cfg.scale * jax.random.normal(key_main, shape=(n,))
        if cfg.out_dist.lower() in ("t", "student-t", "studentt"):
            x_out = sample_student_t(key_out, cfg.t_df, cfg.t_scale, size=(n,))
        elif cfg.out_dist.lower() in ("gaussian", "normal"):
            m_out = cfg.out_mean if cfg.out_mean is not None else (cfg.mean + 5.0 * cfg.scale)
            x_out = m_out + cfg.out_scale * jax.random.normal(key_out, shape=(n,))
        else:
            raise ValueError("out_dist must be 't' or 'gaussian'")
    # contamination mask
    p = jnp.clip(cfg.contam_ratio, 0.0, 1.0)
    mask = jax.random.bernoulli(key_mask, p=p, shape=(n,))
    
    # Broadcast mask for x selection if needed, but mask is per sample
    if x_in.ndim == 2:
        # mask is (n,), x_in is (n, 2)
        # expand mask to (n, 1) for broadcasting
        x = jnp.where(mask[:, None], x_out, x_in)
    else:
        x = jnp.where(mask, x_out, x_in)
        
    # shuffle
    idx = jax.random.permutation(key_perm, n)
    x_shuffled = x[idx]
    mask_shuffled = mask[idx]
    
    return x_shuffled, mask_shuffled


