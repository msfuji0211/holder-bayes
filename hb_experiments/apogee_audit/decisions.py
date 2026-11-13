"""Finite-draw capacities, joint audit sets and cleanup sensitivity."""
import numpy as np
from scipy.special import expit


def count_quantile(k, p):
    """Smallest integer b with empirical P(K <= b) at least p."""
    if not len(k) or not 0 < p <= 1:
        raise ValueError("Invalid count distribution or quantile")
    return int(np.sort(k)[max(0, int(np.ceil(p*len(k)-1e-12))-1)])


def severity(X, y, z):
    """Negative Gaussian log density up to a constant."""
    d = X.shape[1]
    residual = (y[None, :]-z[:, :d] @ X.T)*np.exp(-z[:, d, None])
    return z[:, d, None]+.5*residual**2


def selections(X, y, position):
    d, n = X.shape[1], len(y)
    position = np.asarray(position, dtype=float)
    if position.ndim not in (1, 2, 3) or position.shape[-1] != d+2:
        raise ValueError("Unexpected parameter layout")
    flat = position.reshape(-1, d+2)
    if not len(flat) or not np.isfinite(flat).all():
        raise ValueError("Invalid parameter draws")
    epsilon = 1-expit(flat[:, -1])
    k = np.rint(n*epsilon).astype(int)
    flags = np.zeros((len(flat), n), dtype=bool)
    ss = np.zeros(n)
    for start in range(0, len(flat), 256):
        stop = min(start+256, len(flat))
        a = severity(X, y, flat[start:stop])
        ss += a.sum(axis=0)
        for i, order in enumerate(np.argsort(-a, axis=1, kind="stable"), start=start):
            flags[i, order[:k[i]]] = True
    if not np.array_equal(k, flags.sum(axis=1)):
        raise AssertionError("Draw-specific counts do not match the selection sets")
    return k, epsilon, flags, ss/len(flat)


def rank(q, severity):
    return np.lexsort((np.arange(len(q)), -severity, -q))


def prefix(order, count):
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:count]] = True
    return mask


def joint_prefix_bracket(flags, order, level=.95):
    """Shortest FoD-prefix bracket L ⊆ O ⊆ U; break width ties by smaller U."""
    draws, n = flags.shape
    if draws < 1 or n < 1 or not 0 < level <= 1:
        raise ValueError("Invalid audit sets or probability level")
    ordered = flags[:, order]
    first_missing = np.argmax(~ordered, axis=1)
    first_missing[ordered.all(axis=1)] = n
    last_selected = n-np.argmax(ordered[:, ::-1], axis=1)
    last_selected[~ordered.any(axis=1)] = 0
    histogram = np.zeros((n+1, n+1), dtype=np.int64)
    np.add.at(histogram, (first_missing, last_selected), 1)
    contained = histogram[::-1].cumsum(axis=0)[::-1].cumsum(axis=1)
    required = int(np.ceil(level*draws-1e-12))
    best = None
    for l in range(n+1):
        u = max(l, int(np.searchsorted(contained[l], required)))
        if u <= n:
            candidate = (u-l, u, l)
            if best is None or candidate < best:
                best = candidate
    _, u, l = best
    return prefix(order, l), prefix(order, u), float(contained[l, u]/draws)


def bracket_content(flags, lower, upper):
    return float(np.mean(flags[:, lower].all(axis=1) & ~flags[:, ~upper].any(axis=1)))


def cleanup_range(flags, mh):
    # Compare within-sample descriptive means, conditional on the chosen cleanup.
    retained = len(mh)-flags.sum(axis=1)
    valid = retained > 0
    if not valid.any():
        return dict(dex_low=None, dex_median=None, dex_high=None, dex_width=None, empty_draws=len(flags))
    removed_sum = np.fromiter((mh[mask].sum() for mask in flags), float, count=len(flags))
    a, b, c = np.quantile((mh.sum()-removed_sum[valid])/retained[valid], [.025, .5, .975])
    return dict(dex_low=float(a), dex_median=float(b), dex_high=float(c),
                dex_width=float(c-a), empty_draws=int((~valid).sum()))


def evaluate(X, y, mh, draws, *, point=False, saved_flags=None):
    k, eps, flags, ss = selections(X, y, draws)
    if saved_flags is not None and not np.array_equal(flags, saved_flags):
        raise ValueError("Reconstructed bootstrap selections do not match the archive")
    q = flags.mean(axis=0)
    order = rank(q, ss)
    b = count_quantile(k, .95)
    if point:
        lower = upper = flags[0]
        mass = None
    else:
        lower, upper, mass = joint_prefix_bracket(flags, order)
    top_b = prefix(order, b)
    row = dict(draws=len(k), epsilon_mean=float(eps.mean()), K_mean=float(k.mean()),
               K_q025=count_quantile(k, .025), K_q975=count_quantile(k, .975), B95=b if not point else None,
               K_point=int(k[0]) if point else None, L=int(lower.sum()), U=int(upper.sum()),
               deferred=int((upper & ~lower).sum()), bracket_content=mass,
               top_B_content=None if point else bracket_content(flags, np.zeros(len(q), dtype=bool), top_b))
    flat = np.asarray(draws).reshape(-1, X.shape[1]+2)
    fixed_rank = np.argsort(np.argsort(-severity(X, y, flat.mean(axis=0)[None, :])[0], kind="stable"))
    fixed = fixed_rank[None, :] < k[:, None]
    return dict(row=row, flags=flags, q=q, rank=order, lower=lower, upper=upper, counts=k,
                metallicity=cleanup_range(flags, mh), count_only=cleanup_range(fixed, mh))


def overlap(a, b):
    both = int((a & b).sum())
    union = int((a | b).sum())
    return dict(shared=both, dpd_only=int((a & ~b).sum()), other_only=int((b & ~a).sum()),
                jaccard=both/union if union else 1.0)
