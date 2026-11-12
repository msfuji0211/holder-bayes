"""Regression version of the previous add-on's profiled K-F DPD fit."""
import numpy as np
from scipy.optimize import minimize

SIGMA_MIN, SIGMA_MAX, COEF_BOUND = .05, 10., 50.
GRAD_TOL = 1e-4
LOG2PI = np.log(2 * np.pi)


def profile(X, y, weights, gamma, theta):
    """Objective, envelope gradient and inlier mass, with weights summing to one."""
    ell = theta[-1]
    inv_sigma = np.exp(-ell)
    r = (y - X @ theta[:-1]) * inv_sigma
    v = weights * np.exp(-gamma * (ell + .5 * LOG2PI + .5 * r*r))
    S = float(v.sum())
    C = np.exp(-gamma * (ell + .5 * LOG2PI) - .5 * np.log1p(gamma))
    a = float(np.clip(S/C, 0., 1.))
    ag = a**gamma
    val = ag * (gamma*a*C - (1+gamma)*S)
    b_grad = -gamma*(1+gamma)*ag*inv_sigma * (X.T @ (v*r))
    s_grad = gamma*ag*((1+gamma)*(S - v @ (r*r)) - gamma*a*C)
    return float(val), np.r_[b_grad, s_grad], a


def fit(X, y, gamma, n_starts, seed, *, counts=None, anchor=None):
    X = np.ascontiguousarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    counts = np.ones(n, dtype=int) if counts is None else np.asarray(counts, dtype=int)
    ids = np.repeat(np.arange(n), counts)
    if len(ids) != n:
        raise ValueError("Bootstrap counts do not sum to n")
    weights = counts / n
    rng = np.random.default_rng(seed)
    xx, yy = X[ids], y[ids]
    b0 = np.linalg.lstsq(xx, yy, rcond=None)[0]
    r0 = yy - xx @ b0
    starts = [] if anchor is None else [np.asarray(anchor, dtype=float)]

    def add(beta, rms=False):
        r = yy - xx @ beta
        scale = np.sqrt(np.mean(r*r)) if rms else 1.4826*np.median(abs(r-np.median(r)))
        starts.append(np.r_[beta, np.log(max(float(scale), SIGMA_MIN*1.5))])

    add(b0, rms=True)
    if anchor is None:
        from sklearn.linear_model import HuberRegressor
        try:
            add(HuberRegressor(fit_intercept=False, max_iter=1000).fit(xx, yy).coef_)
        except ValueError:
            add(b0)
    # Residual starts help when two or more incompatible scales fit the data.
    close = np.argsort(abs(r0), kind="stable")[:max(d+1, int(.65*n))]
    add(np.linalg.lstsq(xx[close], yy[close], rcond=None)[0])
    rank = np.argsort(r0, kind="stable")
    for part in (rank[:n//2], rank[n//2:]):
        if len(part) > d:
            beta = np.linalg.lstsq(xx[part], yy[part], rcond=None)[0]
            scale = max(float(np.std(yy[part] - xx[part] @ beta)), SIGMA_MIN*1.5)
            starts.append(np.r_[beta, np.log(scale)])
    while len(starts) < n_starts:
        take = rng.choice(n, size=min(n, max(d+3, int(.4*n))), replace=False)
        add(np.linalg.lstsq(xx[take], yy[take], rcond=None)[0])
    lo = np.r_[np.full(d, -COEF_BOUND), np.log(SIGMA_MIN)]
    hi = np.r_[np.full(d, COEF_BOUND), np.log(SIGMA_MAX)]
    bounds = list(zip(lo, hi))
    fits = []
    for start in np.clip(starts[:n_starts], lo, hi):
        opt = minimize(lambda t: profile(X, y, weights, gamma, t)[:2], start,
                       method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": 1800, "maxls": 80, "ftol": 1e-12, "gtol": 1e-8})
        if np.isfinite(opt.fun) and np.isfinite(opt.x).all():
            fits.append(opt)
    if not fits:
        raise RuntimeError("No finite K-F minimisation result")
    best = min(fits, key=lambda x: x.fun)
    value, grad, a = profile(X, y, weights, gamma, best.x)
    grad[(best.x <= lo+1e-6) & (grad > 0)] = 0
    grad[(best.x >= hi-1e-6) & (grad < 0)] = 0
    err = float(np.max(abs(grad)))
    outer = bool(np.any(abs(best.x[:-1]) >= COEF_BOUND-1e-4) or best.x[-1] >= hi[-1]-1e-4)
    return dict(theta=best.x.tolist(), epsilon=1-a, objective=value, projected_gradient=err,
                solver_success=bool(best.success), valid=bool(np.isfinite(err) and err <= GRAD_TOL and not outer),
                scale_floor=bool(best.x[-1] <= lo[-1]+1e-4), outer_boundary=outer,
                objectives=[float(opt.fun) for opt in fits], n_starts=n_starts)
