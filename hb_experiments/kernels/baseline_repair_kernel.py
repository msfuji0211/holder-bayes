"""Frozen mode proposals and exact MH transitions for regression baselines.

Student-t uses its original unbounded log-scale. GammaDiv uses its original
softplus transform above log(0.05). Optimization constructs proposals only;
the supplied original density determines every MH acceptance probability.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, log_ndtr, logsumexp
from scipy.stats import truncnorm

from .repair_kernel import longest_identical_run

VERSION = "baseline_frozen_modes_plus_local_scale_mh_v1"
MODE_OPTIONS = {"maxiter": 1200, "gtol": 1e-7,
                "gradient_tolerance": .002, "merge_tolerance": .001}
PROPOSAL_OPTIONS = {"df": 7., "inflation": 1.5,
                    "prior_weight": .10, "mode_weight_floor_total": .02}


def to_physical(u, scale_start, floor):
    z = np.array(u, dtype=np.float64, copy=True)
    if floor is not None:
        z[..., scale_start] = np.log(floor)+np.logaddexp(0., z[..., scale_start])
    return z


def to_sampling(z, scale_start, floor):
    u = np.array(z, dtype=np.float64, copy=True)
    if floor is not None:
        delta = u[..., scale_start]-np.log(floor)
        if not np.isfinite(delta).all() or np.any(delta <= 0):
            raise ValueError("Physical scales must be strictly above their original floor")
        u[..., scale_start] = delta+np.log(-np.expm1(-delta))
    return u


def log_jacobian(u, scale_start, floor):
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    return np.zeros(len(u)) if floor is None else -np.logaddexp(0., -u[:, scale_start])


class StudentTarget:
    """NumPy oracle for original Student-t values and analytic derivatives.

    The original beta-prior normalization is preserved, including its additive
    constant. This oracle builds q and checks checkpoints; inference uses the
    original JAX constructor, not this reimplementation.
    """
    def __init__(self, X, y, priors, nu):
        self.X, self.y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.priors, self.nu = dict(priors), float(nu)
        self.n, self.d = self.X.shape
        if (self.y.shape != (self.n,) or self.nu <= 0
                or not np.isfinite(self.X).all() or not np.isfinite(self.y).all()):
            raise ValueError("Invalid Student-t observed inputs")
        p = self.priors
        self.constant = (self.n*(gammaln((self.nu+1)/2)-gammaln(self.nu/2)
                         -.5*np.log(self.nu*np.pi))
                         -.5*(self.d*np.log(2*np.pi)+2*np.log(p['sigma0']))
                         -.5*np.log(2*np.pi*p['s_log_sigma']**2))

    def value_gradient(self, z):
        z = np.asarray(z, dtype=float)
        beta, ls = z[:self.d], float(z[self.d])
        p, k = self.priors, self.nu+1
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = self.y-self.X@beta
            scaled = residual*np.exp(-ls)
            den = self.nu+scaled**2
            value = (self.constant-self.n*ls-.5*k*np.log1p(scaled**2/self.nu).sum()
                     -.5*np.sum(((beta-p['mu0'])/p['sigma0'])**2)
                     -.5*((ls-p['m_log_sigma'])/p['s_log_sigma'])**2)
            gb = k*np.exp(-ls)*(self.X.T@(scaled/den))-(beta-p['mu0'])/p['sigma0']**2
            gs = -self.n+k*np.sum(scaled**2/den)-(ls-p['m_log_sigma'])/p['s_log_sigma']**2
        return float(value), np.r_[gb, gs]

    def __call__(self, z):
        return self.value_gradient(z)[0]

    def negative_hessian(self, z):
        z = np.asarray(z, dtype=float)
        p, k, ls = self.priors, self.nu+1, z[self.d]
        scaled = (self.y-self.X@z[:self.d])*np.exp(-ls)
        den = self.nu+scaled**2
        h = np.empty((self.d+1, self.d+1))
        weights = k*np.exp(-2*ls)*(self.nu-scaled**2)/den**2
        h[:self.d, :self.d] = self.X.T@(self.X*weights[:, None])+np.eye(self.d)/p['sigma0']**2
        cross = 2*k*self.nu*np.exp(-ls)*(self.X.T@(scaled/den**2))
        h[:self.d, self.d] = cross
        h[self.d, :self.d] = cross
        h[self.d, self.d] = 2*k*self.nu*np.sum(scaled**2/den**2)+1/p['s_log_sigma']**2
        return (h+h.T)/2


def gamma_physical_logp(z, X, y, contract):
    """Independent NumPy check of the supplied GammaDiv physical log density."""
    z = np.asarray(z, dtype=float)
    d, p = X.shape[1], contract['priors']
    ls, gamma, eta, floor = z[d], contract['gamma'], contract['eta'], contract['sigma_min']
    if ls < np.log(floor):
        return -np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        residual = (y-X@z[:d])*np.exp(-ls)
        log_pdf = -.5*np.log(2*np.pi)-ls-.5*residual**2
        log_a = -.5*gamma*np.log(2*np.pi)-.5*np.log1p(gamma)-gamma*ls
        reward = eta/gamma*np.expm1(gamma*log_pdf-gamma/(1+gamma)*log_a).sum()
    prior = -.5*(d*np.log(2*np.pi*p['sigma0']**2)+np.sum(((z[:d]-p['mu0'])/p['sigma0'])**2))
    prior += -.5*(np.log(2*np.pi*p['s_log_sigma']**2)+((ls-p['m_log_sigma'])/p['s_log_sigma'])**2)
    prior -= log_ndtr((p['m_log_sigma']-np.log(floor))/p['s_log_sigma'])
    return float(prior+reward)


def student_starts(X, y, historical_draws):
    """Observed-data starts plus means and highest-density historical states.

    Historical LP extrema are supplied separately by the caller. No true
    coefficients, injected contamination labels, or convergence-based chain
    filtering enter the search.
    """
    d, n = X.shape[1], len(y)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    order = np.argsort(y-X@ols)
    subsets = [np.arange(n), order[:n//2], order[n//2:], order[:n//3], order[-n//3:]]
    out = []
    for rows in subsets:
        if len(rows) <= d:
            rows = np.arange(n)
        beta = np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
        residual = y[rows]-X[rows]@beta
        scale = max(float(1.4826*np.median(np.abs(residual-np.median(residual)))), 1e-4)
        for factor in [.5, 1., 2.]:
            out.append(np.r_[beta, np.log(scale*factor)])
    for draws, lp in historical_draws:
        for chain in range(draws.shape[0]):
            out += [draws[chain].mean(axis=0), draws[chain, np.argmax(lp[chain])], draws[chain, -1]]
    return np.stack(out)


def discover_student_modes(oracle, starts, progress=None):
    modes, attempts = [], []
    def objective(z):
        value, gradient = oracle.value_gradient(z)
        return -value, -gradient
    for i, start in enumerate(starts):
        result = minimize(objective, start, jac=True, method="BFGS",
                          options={k: MODE_OPTIONS[k] for k in ['maxiter', 'gtol']})
        gradient = float(np.linalg.norm(result.jac))
        row = {"start_index": i, "optimizer_success": bool(result.success),
               "message": str(result.message), "gradient_norm": gradient,
               "log_density_sampling": -float(result.fun), "used": False}
        attempts.append(row)
        if (np.isfinite(result.fun) and np.isfinite(result.x).all()
                and np.isfinite(gradient) and gradient <= MODE_OPTIONS['gradient_tolerance']):
            H = oracle.negative_hessian(result.x)
            if np.isfinite(H).all() and np.linalg.eigvalsh(H)[0] > 0:
                row['used'] = True
                if not any(np.linalg.norm(result.x-m[0]) < MODE_OPTIONS['merge_tolerance'] for m in modes):
                    modes.append((result.x.copy(), H, -float(result.fun)))
        if progress is not None:
            progress(i+1, len(starts))
    if not modes:
        raise RuntimeError("No finite positive-curvature Student-t proposal mode found")
    modes.sort(key=lambda m: -m[2])
    return (np.stack([m[0] for m in modes]), np.stack([m[1] for m in modes]),
            np.array([m[2] for m in modes]), attempts)


class ModeProposal:
    """A normalized multivariate-t mixture plus the proper original prior.

    For floor=0.05 and default options this equals the original GammaDiv q.
    For floor=None the prior component has an ordinary Normal log-scale.
    """
    def __init__(self, means, hessians, mode_lp, scale_start, priors, floor,
                 df=7., inflation=1.5, prior_weight=.10, mode_weight_floor_total=.02):
        self.means = np.asarray(means, dtype=float)
        self.hessians = np.asarray(hessians, dtype=float)
        self.mode_lp = np.asarray(mode_lp, dtype=float)
        if self.means.ndim != 2 or not np.isfinite(self.means).all():
            raise ValueError("Invalid frozen means")
        self.count, self.dimension = self.means.shape
        if (self.count < 1 or self.dimension != scale_start+1
                or self.hessians.shape != (self.count, self.dimension, self.dimension)
                or not np.isfinite(self.hessians).all() or not np.isfinite(self.mode_lp).all()
                or self.mode_lp.shape != (self.count,)):
            raise ValueError("Invalid baseline proposal layout")
        if (df <= 2 or inflation <= 0 or not 0 < prior_weight < 1
                or not 0 <= mode_weight_floor_total < 1
                or not np.isfinite([df, inflation, prior_weight, mode_weight_floor_total]).all()
                or np.any(np.linalg.eigvalsh(self.hessians) <= 0)):
            raise ValueError("Invalid frozen proposal curvature or settings")
        self.scale_start, self.priors, self.floor, self.df = scale_start, dict(priors), floor, df
        if floor is not None and (not np.isfinite(floor) or floor <= 0):
            raise ValueError("Invalid original scale floor")
        self.covariance = np.linalg.inv(self.hessians)
        self.scatter = self.covariance*inflation*(df-2)/df
        self.cholesky = np.linalg.cholesky(self.scatter)
        self.inverse = np.linalg.inv(self.scatter)
        self.logconst = (gammaln((df+self.dimension)/2)-gammaln(df/2)
                         -.5*(self.dimension*np.log(df*np.pi)+np.linalg.slogdet(self.scatter)[1]))
        log_mass = self.mode_lp+.5*self.dimension*np.log(2*np.pi)-.5*np.linalg.slogdet(self.hessians)[1]
        w = np.exp(log_mass-logsumexp(log_mass))
        w = (1-mode_weight_floor_total)*w+mode_weight_floor_total/self.count
        self.weights = np.r_[(1-prior_weight)*w, prior_weight]

    def draw(self, rng, n):
        component = rng.choice(self.count+1, size=n, p=self.weights)
        u = np.empty((n, self.dimension))
        for k in range(self.count):
            idx = np.flatnonzero(component == k)
            noise = rng.normal(size=(len(idx), self.dimension))@self.cholesky[k].T
            u[idx] = self.means[k]+noise/np.sqrt(rng.chisquare(self.df, len(idx))/self.df)[:, None]
        idx = np.flatnonzero(component == self.count)
        p = self.priors
        if len(idx):
            z = np.empty((len(idx), self.dimension))
            z[:, :self.scale_start] = rng.normal(p['mu0'], p['sigma0'], (len(idx), self.scale_start))
            if self.floor is None:
                z[:, self.scale_start] = rng.normal(p['m_log_sigma'], p['s_log_sigma'], len(idx))
            else:
                lower = (np.log(self.floor)-p['m_log_sigma'])/p['s_log_sigma']
                ls = truncnorm.rvs(lower, np.inf, loc=p['m_log_sigma'], scale=p['s_log_sigma'],
                                   size=len(idx), random_state=rng)
                z[:, self.scale_start] = np.maximum(ls, np.nextafter(np.log(self.floor), np.inf))
            u[idx] = to_sampling(z, self.scale_start, self.floor)
        return u, component

    def logq(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        terms = []
        for k in range(self.count):
            delta = u-self.means[k]
            distance = np.einsum('ni,ij,nj->n', delta, self.inverse[k], delta)
            terms.append(np.log(self.weights[k])+self.logconst[k]
                         -.5*(self.df+self.dimension)*np.log1p(distance/self.df))
        p = self.priors
        z = to_physical(u, self.scale_start, self.floor)
        loc, ls = z[:, :self.scale_start], z[:, self.scale_start]
        lp = -.5*(self.scale_start*np.log(2*np.pi*p['sigma0']**2)
                   +np.sum(((loc-p['mu0'])/p['sigma0'])**2, axis=1))
        lp += -.5*(np.log(2*np.pi*p['s_log_sigma']**2)+((ls-p['m_log_sigma'])/p['s_log_sigma'])**2)
        if self.floor is not None:
            lp -= log_ndtr((p['m_log_sigma']-np.log(self.floor))/p['s_log_sigma'])
        lp += log_jacobian(u, self.scale_start, self.floor)
        terms.append(np.log(self.weights[-1])+lp)
        return logsumexp(np.stack(terms), axis=0)


def local_factors(proposal, multiplier=2.38):
    """Use within-mode curvature, not covariance pooled across distinct modes."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Invalid local multiplier")
    factors = np.linalg.cholesky(proposal.covariance)*(multiplier/np.sqrt(proposal.dimension))
    scales = np.sqrt(proposal.covariance[:, proposal.scale_start, proposal.scale_start])*multiplier
    weights = proposal.weights[:-1]/proposal.weights[:-1].sum()
    return factors, scales, weights


def run_chain(logp, logp_batch, proposal, start, *, seed, warmup, draws,
              composed, progress=None):
    """Fixed independence MH, optionally followed by two symmetric MH moves.

    Each local covariance is selected with state-independent frozen weights.
    Their mixture is symmetric even with unequal mode weights. The scale-only
    move is symmetric in the sampling coordinate (u for GammaDiv, log-sigma
    for Student-t); the supplied logp must include the GammaDiv Jacobian.
    """
    if warmup < 0 or draws < 1:
        raise ValueError("Invalid fixed run length")
    rng = np.random.default_rng(seed)
    candidates, components = proposal.draw(rng, warmup+draws)
    candidate_lp = np.asarray(logp_batch(candidates), dtype=float).reshape(-1)
    candidate_lq = np.asarray(proposal.logq(candidates), dtype=float).reshape(-1)
    if len(candidate_lp) != len(candidates) or len(candidate_lq) != len(candidates):
        raise ValueError("Invalid batch density shape")
    state = np.asarray(start, dtype=float).copy()
    state_lp, state_lq = float(logp(state)), float(proposal.logq(state)[0])
    if not np.isfinite(state).all() or not np.isfinite([state_lp, state_lq]).all():
        raise ValueError("Non-finite initial target or proposal")
    local_chol, scale_steps, local_weights = local_factors(proposal)
    retained, retained_lp = np.empty((draws, proposal.dimension)), np.empty(draws)
    accepted = np.zeros(3, dtype=int)
    mode_counts = np.zeros(proposal.count+1, dtype=int)
    mode_accepted = np.zeros_like(mode_counts)
    warmup_last = None
    for i in range(warmup+draws):
        lp, lq = candidate_lp[i], candidate_lq[i]
        threshold = np.log(rng.uniform())
        if i >= warmup:
            mode_counts[components[i]] += 1
        if np.isfinite([lp, lq]).all() and threshold < min(0., lp-state_lp+state_lq-lq):
            state, state_lp, state_lq = candidates[i].copy(), float(lp), float(lq)
            if i >= warmup:
                accepted[0] += 1
                mode_accepted[components[i]] += 1
        if composed:
            k = rng.choice(proposal.count, p=local_weights)
            candidate = state+local_chol[k]@rng.normal(size=proposal.dimension)
            lp = float(logp(candidate))
            threshold = np.log(rng.uniform())
            if np.isfinite(candidate).all() and np.isfinite(lp) and threshold < min(0., lp-state_lp):
                state, state_lp = candidate, lp
                if i >= warmup:
                    accepted[1] += 1
            k = rng.choice(proposal.count, p=local_weights)
            candidate = state.copy()
            candidate[proposal.scale_start] += scale_steps[k]*rng.normal()
            lp = float(logp(candidate))
            threshold = np.log(rng.uniform())
            if np.isfinite(candidate).all() and np.isfinite(lp) and threshold < min(0., lp-state_lp):
                state, state_lp = candidate, lp
                if i >= warmup:
                    accepted[2] += 1
            state_lq = float(proposal.logq(state)[0])
            if not np.isfinite(state_lq):
                raise ValueError("Non-finite proposal density after local move")
        if i+1 == warmup:
            warmup_last = state.copy()
        if i >= warmup:
            retained[i-warmup], retained_lp[i-warmup] = state, state_lp
        if progress is not None and ((i+1) % 500 == 0 or i+1 == warmup+draws):
            progress(i+1, warmup+draws)
    return retained, retained_lp, {
        "independence_acceptance": float(accepted[0]/draws),
        "local_acceptance": float(accepted[1]/draws) if composed else None,
        "scale_acceptance": float(accepted[2]/draws) if composed else None,
        "independent_component_proposed": mode_counts.tolist(),
        "independent_component_accepted": mode_accepted.tolist(),
        "longest_identical_position_run": longest_identical_run(retained),
        "start_sampling": np.asarray(start).tolist(),
        "warmup_last_sampling": None if warmup_last is None else warmup_last.tolist(),
        "target_transitions_per_iteration": 3 if composed else 1,
        "warmup": warmup, "draws": draws,
    }
