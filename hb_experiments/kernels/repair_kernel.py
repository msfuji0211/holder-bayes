"""Fixed composition of three MH transitions for a limited diagnostic pilot.

Each transition preserves the same density in sampling coordinates. Proposal
covariance and step sizes are frozen before both warmup and retained sampling.
No rejected state, chain, or late segment is removed based on diagnostics.
"""
from __future__ import annotations

import numpy as np

VERSION = "independence_plus_local_plus_rho_mh_pilot_v1"


def local_cholesky(pilot, multiplier=2.38):
    pilot = np.asarray(pilot, dtype=np.float64)
    if pilot.ndim != 3 or min(pilot.shape[:2]) < 2 or not np.isfinite(pilot).all():
        raise ValueError("Expected finite chain/draw/parameter pilot draws")
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Local multiplier must be positive and finite")
    dimension = pilot.shape[-1]
    covariance = np.atleast_2d(np.cov(pilot.reshape(-1, dimension), rowvar=False))
    ridge = max(float(np.trace(covariance)) / dimension, 1e-8) * 1e-8
    covariance += np.eye(dimension) * ridge
    return np.linalg.cholesky(covariance) * (multiplier / np.sqrt(dimension))


def longest_identical_run(draws):
    draws = np.asarray(draws)
    if len(draws) == 0:
        return 0
    changes = np.flatnonzero(np.any(draws[1:] != draws[:-1], axis=1)) + 1
    return int(np.max(np.diff(np.r_[0, changes, len(draws)])))


def run_composed_chain(logprob, logprob_batch, proposal, start, *, seed,
                       warmup, draws, local_chol, rho_step=0.75, progress=None):
    """One saved draw follows independence MH, full local MH, and rho-only MH.

    logprob and proposal.logq must refer to the same unconstrained coordinates.
    The two local proposals are symmetric, so their MH corrections use only
    the target density ratio. All randomness belongs to this new fixed run.
    """
    if warmup < 0 or draws < 1 or not np.isfinite(rho_step) or rho_step <= 0:
        raise ValueError("Invalid fixed sampling schedule")
    rng = np.random.default_rng(seed)
    state = np.asarray(start, dtype=np.float64).copy()
    dimension = len(state)
    local_chol = np.asarray(local_chol, dtype=np.float64)
    if local_chol.shape != (dimension, dimension) or not np.isfinite(local_chol).all():
        raise ValueError("Invalid fixed local proposal factor")
    if not np.isfinite(state).all():
        raise ValueError("Non-finite initial state")
    candidates, _ = proposal.draw(rng, warmup + draws)
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.shape != (warmup + draws, dimension):
        raise ValueError("Incorrect independent proposal shape")
    candidate_lp = np.asarray(logprob_batch(candidates), dtype=np.float64).reshape(-1)
    candidate_lq = np.asarray(proposal.logq(candidates), dtype=np.float64).reshape(-1)
    if len(candidate_lp) != len(candidates) or len(candidate_lq) != len(candidates):
        raise ValueError("Incorrect density batch shape")
    state_lp = float(logprob(state))
    state_lq = float(np.asarray(proposal.logq(state)).reshape(-1)[0])
    if not np.isfinite(state_lp) or not np.isfinite(state_lq):
        raise ValueError("Non-finite initial target/proposal density")
    retained = np.empty((draws, dimension), dtype=np.float64)
    retained_lp = np.empty(draws, dtype=np.float64)
    accepted = np.zeros(3, dtype=np.int64)

    def try_symmetric(candidate, current_lp):
        if not np.isfinite(candidate).all():
            return False, current_lp
        candidate_value = float(logprob(candidate))
        # Consume the uniform even on a rejected non-finite candidate.
        threshold = np.log(rng.uniform())
        take = np.isfinite(candidate_value) and threshold < min(0., candidate_value-current_lp)
        return bool(take), candidate_value

    for i in range(warmup + draws):
        threshold = np.log(rng.uniform())
        lp, lq = candidate_lp[i], candidate_lq[i]
        if (np.isfinite(lp) and np.isfinite(lq)
                and threshold < min(0., lp-state_lp+state_lq-lq)):
            state, state_lp, state_lq = candidates[i].copy(), float(lp), float(lq)
            if i >= warmup:
                accepted[0] += 1

        candidate = state + local_chol @ rng.normal(size=dimension)
        take, lp = try_symmetric(candidate, state_lp)
        if take:
            state, state_lp = candidate, lp
            state_lq = float(np.asarray(proposal.logq(state)).reshape(-1)[0])
            if i >= warmup:
                accepted[1] += 1

        candidate = state.copy()
        candidate[-1] += rho_step * rng.normal()
        take, lp = try_symmetric(candidate, state_lp)
        if take:
            state, state_lp = candidate, lp
            state_lq = float(np.asarray(proposal.logq(state)).reshape(-1)[0])
            if i >= warmup:
                accepted[2] += 1

        if not np.isfinite(state_lq):
            raise ValueError("Proposal density became non-finite at a retained target state")
        if i >= warmup:
            retained[i-warmup], retained_lp[i-warmup] = state, state_lp
        if progress is not None and ((i+1) % 500 == 0 or i+1 == warmup+draws):
            progress(i+1, warmup+draws)

    return retained, retained_lp, {
        "independence_acceptance": float(accepted[0] / draws),
        "local_acceptance": float(accepted[1] / draws),
        "rho_acceptance": float(accepted[2] / draws),
        "longest_identical_position_run": longest_identical_run(retained),
        "target_transitions_per_retained_iteration": 3,
        "fixed_warmup": int(warmup), "fixed_retained_draws": int(draws),
    }
