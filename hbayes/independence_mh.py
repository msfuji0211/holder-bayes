"""Frozen Student-t mixture plus proper prior proposal, with exact MH correction.

The physical target is supplied by the existing posterior code. Only regression
H-B truncates log(sigma) below log(0.05). Density coordinates remain unbounded.
The proposal is fixed before any retained MH chain is generated.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, gammaln, log_ndtr, logsumexp, betaln
from scipy.stats import truncnorm


def to_physical(u, scale_start, floor, n_scales=1):
    z = np.array(u, dtype=np.float64, copy=True)
    if floor is not None:
        sl = slice(scale_start, scale_start + n_scales)
        z[..., sl] = np.log(floor) + np.logaddexp(0.0, z[..., sl])
    return z


def to_sampling(z, scale_start, floor, n_scales=1):
    u = np.array(z, dtype=np.float64, copy=True)
    if floor is not None:
        sl = slice(scale_start, scale_start + n_scales)
        delta = u[..., sl] - np.log(floor)
        if np.any(delta <= 0) or not np.isfinite(delta).all():
            raise ValueError("Physical scales must exceed their lower bound")
        u[..., sl] = delta + np.log(-np.expm1(-delta))
    return u


def log_jacobian(u, scale_start, floor, n_scales=1):
    u = np.atleast_2d(u)
    if floor is None:
        return np.zeros(len(u))
    return -np.logaddexp(0, -u[:, scale_start:scale_start+n_scales]).sum(axis=1)


class FrozenProposal:
    def __init__(self, pilot_u, scale_start, n_scales, floor, priors,
                 df=7.0, inflation=1.5, prior_weight=0.10):
        pilot_u = np.asarray(pilot_u, dtype=float)
        if pilot_u.ndim != 3 or not np.isfinite(pilot_u).all():
            raise ValueError("Expected finite (chain, draw, parameter) proposal draws")
        self.chains, self.draws, self.dimension = pilot_u.shape
        if self.chains < 2 or self.draws < 4:
            raise ValueError("At least two chains and four draws are required")
        self.scale_start, self.n_scales, self.floor = scale_start, n_scales, floor
        self.priors, self.df = dict(priors), float(df)
        if self.dimension != scale_start + n_scales + 1:
            raise ValueError("Proposal layout must be [locations, log-scales, rho]")
        if any(self.priors.get(k, False) for k in
               ["use_c_mixture", "use_c_hierarchical", "use_c_logitnormal"]):
            raise ValueError("The prior proposal supports the paper's single Beta prior only")
        self.means = pilot_u.mean(axis=1)
        covariance = np.stack([np.cov(a, rowvar=False) for a in pilot_u])
        self.scatter = covariance * inflation * (df - 2) / df
        for i in range(self.chains):
            self.scatter[i] += np.eye(self.dimension) * max(
                np.trace(self.scatter[i]) / self.dimension, 1e-8) * 1e-8
        self.cholesky = np.linalg.cholesky(self.scatter)
        self.inverse = np.linalg.inv(self.scatter)
        logdet = np.linalg.slogdet(self.scatter)[1]
        self.logconst = (gammaln((df+self.dimension)/2) - gammaln(df/2)
                         - .5*(self.dimension*np.log(df*np.pi)+logdet))
        self.weights = np.r_[np.repeat((1-prior_weight)/self.chains,
                                       self.chains), prior_weight]

    def draw(self, rng, n):
        component = rng.choice(self.chains+1, size=n, p=self.weights)
        u = np.empty((n, self.dimension))
        for i in range(self.chains):
            idx = np.flatnonzero(component == i)
            g = rng.normal(size=(len(idx), self.dimension)) @ self.cholesky[i].T
            u[idx] = self.means[i] + g / np.sqrt(
                rng.chisquare(self.df, size=len(idx))/self.df)[:, None]
        idx = np.flatnonzero(component == self.chains)
        if len(idx):
            p = self.priors
            u[idx, :self.scale_start] = rng.normal(
                loc=p["mu0"], scale=p["sigma0"], size=(len(idx), self.scale_start))
            shape = (len(idx), self.n_scales)
            if self.floor is None:
                ls = rng.normal(p["m_log_sigma"], p["s_log_sigma"], size=shape)
            else:
                lo = (np.log(self.floor)-p["m_log_sigma"])/p["s_log_sigma"]
                ls = truncnorm.rvs(lo, np.inf, loc=p["m_log_sigma"],
                                   scale=p["s_log_sigma"], size=shape, random_state=rng)
            sl = slice(self.scale_start, self.scale_start+self.n_scales)
            u[idx, sl] = ls
            c = rng.beta(p["a_c"], p["b_c"], size=len(idx))
            c = np.clip(c, np.nextafter(0., 1.), np.nextafter(1., 0.))
            u[idx, -1] = np.log(c)-np.log1p(-c)
            u[idx] = to_sampling(u[idx], self.scale_start, self.floor, self.n_scales)
        return u, component

    def logq(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        logs = []
        for i in range(self.chains):
            delta = u-self.means[i]
            distance = np.einsum("ni,ij,nj->n", delta, self.inverse[i], delta)
            logs.append(np.log(self.weights[i])+self.logconst[i]
                        -(self.df+self.dimension)/2*np.log1p(distance/self.df))
        p = self.priors
        z = to_physical(u, self.scale_start, self.floor, self.n_scales)
        beta = z[:, :self.scale_start]
        ls = z[:, self.scale_start:self.scale_start+self.n_scales]
        lp = -.5*(self.scale_start*np.log(2*np.pi*p["sigma0"]**2)
                    + np.sum(((beta-p["mu0"])/p["sigma0"])**2, axis=1))
        lp += -.5*(self.n_scales*np.log(2*np.pi*p["s_log_sigma"]**2)
                     + np.sum(((ls-p["m_log_sigma"])/p["s_log_sigma"])**2, axis=1))
        if self.floor is not None:
            lp -= self.n_scales*log_ndtr(
                (p["m_log_sigma"]-np.log(self.floor))/p["s_log_sigma"])
        logc, log1c = -np.logaddexp(0, -z[:, -1]), -np.logaddexp(0, z[:, -1])
        lp += p["a_c"]*logc+p["b_c"]*log1c-betaln(p["a_c"], p["b_c"])
        lp += log_jacobian(u, self.scale_start, self.floor, self.n_scales)
        logs.append(np.log(self.weights[-1])+lp)
        return logsumexp(np.stack(logs), axis=0)


def run_chain(logp, proposal, start, seed, warmup, draws):
    """Include rejected states; never thin or select chains using diagnostics."""
    rng = np.random.default_rng(seed)
    candidates, components = proposal.draw(rng, warmup+draws)
    lp, lq = np.atleast_1d(logp(candidates)), proposal.logq(candidates)
    with np.errstate(invalid="ignore"):
        ratios = np.where(np.isfinite(lp) & np.isfinite(lq), lp-lq, -np.inf)
    state = np.asarray(start, dtype=float).copy()
    state_lp = float(np.atleast_1d(logp(state[None]))[0])
    state_ratio = state_lp-float(proposal.logq(state)[0])
    if not np.isfinite(state_ratio):
        raise ValueError("The initial MH state has a non-finite density")
    thresholds = np.log(rng.uniform(size=len(candidates)))
    retained = np.empty((draws, proposal.dimension))
    retained_logp = np.empty(draws)
    accepted = np.zeros(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        if np.isfinite(ratios[i]) and thresholds[i] < min(0., ratios[i]-state_ratio):
            state, state_lp, state_ratio = candidates[i].copy(), lp[i], ratios[i]
            accepted[i] = True
        if i >= warmup:
            retained[i-warmup], retained_logp[i-warmup] = state, state_lp
    weights = ratios-logsumexp(ratios)
    summary = {"mh_acceptance_mean": float(accepted[warmup:].mean()),
               "importance_ess": float(np.exp(-logsumexp(2*weights))),
               "proposed": len(candidates)}
    return retained, retained_logp, summary
