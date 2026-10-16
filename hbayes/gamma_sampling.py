"""GammaDiv-only mode discovery and frozen independence MH.

Non-GammaDiv calls delegate verbatim to the previous paper_sampling.fit, whose
file and cache contracts remain untouched. Local optimization builds a proposal
only. Proposal weights never replace posterior mode probabilities. Rejections
are retained and all chains enter diagnostics and summaries.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, log_ndtr, logsumexp
from scipy.stats import truncnorm

from . import paper_sampling as old
from .bounded_scale import BoundedScaleTarget
from .gamma_posterior import DEFINITION, SIGMA_MIN, ETA
from .independence_mh import to_physical, to_sampling, log_jacobian, run_chain

VERSION = "gamma_nh_floor005_frozen_modes_mh_v1"
OPTIONS = {"student_t_df": 7., "covariance_inflation": 1.5, "prior_weight": .10,
           "mode_weight_floor_total": .02, "optimizer_maxiter": 1200,
           "optimizer_gtol": 1e-7, "optimizer_ftol": 1e-13,
           "mode_gradient_tolerance": 2e-3, "mode_merge_tolerance": 1e-3}


def starts(inputs, model, seed):
    """Use observed data only; no true coefficients, labels or contamination rate."""
    rng = np.random.default_rng(seed + 148361)
    if model == "regression":
        X, y = inputs
        d, n = X.shape[1], len(y)
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        order = np.argsort(y - X @ ols)
    else:
        x = inputs[0]
        d, n = x.shape[1], len(x)
        order = np.argsort(x[:, 0])
    subsets = [np.arange(n), order[:n//2], order[n//2:], order[:n//3], order[-n//3:]]
    out = []
    for rows in subsets:
        if len(rows) <= d:
            rows = np.arange(n)
        if model == "regression":
            location = np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
            residual = y[rows] - X[rows] @ location
            scale = np.atleast_1d(1.4826 * np.median(np.abs(residual - np.median(residual))))
        else:
            location = np.median(x[rows], axis=0)
            scale = 1.4826 * np.median(np.abs(x[rows] - location), axis=0)
        scale = np.maximum(scale, 2 * SIGMA_MIN)
        for factor in [.5, 1., 2.]:
            out.append(np.r_[location, np.log(np.maximum(factor * scale, 1.5 * SIGMA_MIN))])
    for anchor in out[:3]:
        z = anchor.copy()
        z[:d] += rng.normal(size=d) * .05
        out.append(z)
    return np.stack(out)


def discover(target, initials, scale_start, n_scales):
    import jax
    import jax.numpy as jnp
    wrapped = BoundedScaleTarget(target, scale_start, (SIGMA_MIN,) * n_scales)
    value_grad = jax.jit(jax.value_and_grad(lambda u: -wrapped.log_density(u)))
    hessian = jax.jit(jax.hessian(lambda u: -wrapped.log_density(u)))

    def objective(u):
        value, gradient = value_grad(jnp.asarray(u, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=float)

    modes, attempts = [], []
    for z in initials:
        u = to_sampling(z, scale_start, SIGMA_MIN, n_scales)
        result = minimize(objective, u, jac=True, method="L-BFGS-B",
                          options={"maxiter": OPTIONS["optimizer_maxiter"],
                                   "gtol": OPTIONS["optimizer_gtol"], "ftol": OPTIONS["optimizer_ftol"]})
        gradient = float(np.linalg.norm(result.jac))
        row = {"optimizer_success": bool(result.success), "message": str(result.message),
               "gradient_norm": gradient, "log_density_sampling": -float(result.fun)}
        attempts.append(row)
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or not np.isfinite(gradient) or gradient > OPTIONS["mode_gradient_tolerance"]:
            row["used"] = False
            continue
        H = np.asarray(hessian(jnp.asarray(result.x, dtype=jnp.float64)), dtype=float)
        H = (H + H.T) / 2
        eigenvalues = np.linalg.eigvalsh(H)
        if not np.isfinite(H).all() or eigenvalues[0] <= 0:
            row["used"] = False
            continue
        row["used"] = True
        if any(np.linalg.norm(result.x - m[0]) < OPTIONS["mode_merge_tolerance"] for m in modes):
            continue
        modes.append((result.x.copy(), H, -float(result.fun)))
    if not modes:
        raise RuntimeError("GammaDiv found no finite positive-curvature proposal mode; inspect the data and target")
    modes.sort(key=lambda m: -m[2])
    means = np.stack([m[0] for m in modes])
    H = np.stack([m[1] for m in modes])
    lp = np.array([m[2] for m in modes])
    return wrapped, means, H, lp, attempts


class Proposal:
    def __init__(self, means, hessians, mode_lp, scale_start, n_scales, priors):
        self.means = np.asarray(means, dtype=float)
        self.dimension = self.means.shape[1]
        self.count = len(means)
        self.scale_start, self.n_scales, self.priors = scale_start, n_scales, dict(priors)
        self.df = OPTIONS["student_t_df"]
        covariance = np.linalg.inv(hessians)
        self.scatter = covariance * OPTIONS["covariance_inflation"] * (self.df - 2) / self.df
        self.cholesky = np.linalg.cholesky(self.scatter)
        self.inverse = np.linalg.inv(self.scatter)
        self.logconst = (gammaln((self.df + self.dimension)/2) - gammaln(self.df/2)
                         - .5 * (self.dimension * np.log(self.df*np.pi) + np.linalg.slogdet(self.scatter)[1]))
        # Laplace evidence estimates affect proposal efficiency only. The MH
        # acceptance ratio always uses the exact chosen posterior and full q.
        log_mass = mode_lp + .5*self.dimension*np.log(2*np.pi) - .5*np.linalg.slogdet(hessians)[1]
        weights = np.exp(log_mass - logsumexp(log_mass))
        weights = (1 - OPTIONS["mode_weight_floor_total"]) * weights + OPTIONS["mode_weight_floor_total"] / self.count
        self.weights = np.r_[(1 - OPTIONS["prior_weight"]) * weights, OPTIONS["prior_weight"]]

    def draw(self, rng, n):
        components = rng.choice(self.count + 1, size=n, p=self.weights)
        u = np.empty((n, self.dimension))
        for k in range(self.count):
            idx = np.flatnonzero(components == k)
            noise = rng.normal(size=(len(idx), self.dimension)) @ self.cholesky[k].T
            u[idx] = self.means[k] + noise / np.sqrt(rng.chisquare(self.df, len(idx)) / self.df)[:, None]
        idx = np.flatnonzero(components == self.count)
        p = self.priors
        if len(idx):
            z = np.empty((len(idx), self.dimension))
            z[:, :self.scale_start] = rng.normal(p["mu0"], p["sigma0"], (len(idx), self.scale_start))
            lower = (np.log(SIGMA_MIN) - p["m_log_sigma"]) / p["s_log_sigma"]
            ls = truncnorm.rvs(lower, np.inf, loc=p["m_log_sigma"], scale=p["s_log_sigma"],
                               size=(len(idx), self.n_scales), random_state=rng)
            z[:, self.scale_start:] = np.maximum(ls, np.nextafter(np.log(SIGMA_MIN), np.inf))
            u[idx] = to_sampling(z, self.scale_start, SIGMA_MIN, self.n_scales)
        return u, components

    def logq(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        terms = []
        for k in range(self.count):
            delta = u - self.means[k]
            distance = np.einsum("ni,ij,nj->n", delta, self.inverse[k], delta)
            terms.append(np.log(self.weights[k]) + self.logconst[k]
                         - .5 * (self.df + self.dimension) * np.log1p(distance / self.df))
        p = self.priors
        z = to_physical(u, self.scale_start, SIGMA_MIN, self.n_scales)
        loc, ls = z[:, :self.scale_start], z[:, self.scale_start:]
        lp = -.5 * (self.scale_start * np.log(2*np.pi*p["sigma0"]**2)
                    + np.sum(((loc - p["mu0"]) / p["sigma0"])**2, axis=1))
        lp += -.5 * (self.n_scales * np.log(2*np.pi*p["s_log_sigma"]**2)
                     + np.sum(((ls - p["m_log_sigma"]) / p["s_log_sigma"])**2, axis=1))
        lp -= self.n_scales * log_ndtr((p["m_log_sigma"] - np.log(SIGMA_MIN)) / p["s_log_sigma"])
        lp += log_jacobian(u, self.scale_start, SIGMA_MIN, self.n_scales)
        terms.append(np.log(self.weights[-1]) + lp)
        return logsumexp(np.stack(terms), axis=0)


def fit(logprob, inputs, **kwargs):
    if kwargs["method"] != "GammaDiv":
        return old.fit(logprob, inputs, **kwargs)
    return fit_gamma(logprob, inputs, **kwargs)


def fit_gamma(logprob, inputs, *, method, model, seed, chains, num_warmup, num_samples,
              target_accept, init_step, priors, eta=1., gamma=1., phi=None, kappa=1.,
              student_t_df=4., floor=None, names=None):
    import jax
    if eta != ETA or floor is not None or phi is not None:
        raise ValueError("GammaDiv has a separate fixed eta=1, sigma_min=0.05 protocol")
    identity = {"definition": DEFINITION, "gamma": float(gamma), "eta": float(eta),
                "priors": asdict(priors), "sigma_min": SIGMA_MIN, "model": model}
    if getattr(logprob, "gamma_protocol", None) != identity:
        raise ValueError("GammaDiv target does not carry the chosen NH definition, prior and temperature contract")
    arrays = [np.asarray(a, dtype=np.float64) for a in inputs]
    d = arrays[0].shape[1]
    s = 1 if model == "regression" else d
    dimension = d + s
    names = names or [f"location_{i}" for i in range(d)] + [f"log_scale_{i}" for i in range(s)]
    if len(names) != dimension or chains < 2:
        raise ValueError("Invalid GammaDiv parameter names or chain count")
    sources = {n: old.file_hash(Path(__file__).parent/n) for n in
               ["gamma_posterior.py", "gamma_sampling.py", "paper_sampling.py", "bounded_scale.py", "independence_mh.py"]}
    contract = {"version": VERSION, "definition": DEFINITION, "method": method, "model": model,
                "seed": int(seed), "input_hashes": [old.array_hash(a) for a in arrays],
                "priors": asdict(priors), "sigma_min": SIGMA_MIN, "eta": float(eta), "gamma": float(gamma),
                "chains": int(chains), "warmup": int(num_warmup), "samples": int(num_samples),
                "parameter_names": list(names), "proposal_options": OPTIONS, "sources": sources, "runtime": old.runtime()}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get("HBAYES_CACHE_ROOT", "outputs/temperature_floor005_cache_v2"))
    folder = root/model/method/digest
    folder.mkdir(parents=True, exist_ok=True)
    contract_path = folder/"contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("GammaDiv cache contract collision")
    old.atomic_json(contract_path, contract)
    with jax.experimental.enable_x64():
        batch = jax.jit(jax.vmap(logprob))
        final_path = folder/"posterior.npz"
        saved = old._restore(final_path)
        if saved is not None:
            draws = saved["samples_physical"]
            if draws.shape != (chains, num_samples, dimension):
                raise ValueError("GammaDiv cached chain layout differs")
            lp = old._batch(batch, draws.reshape(-1, dimension)).reshape(draws.shape[:2])
            np.testing.assert_allclose(lp, saved["log_posterior_physical"], rtol=1e-10, atol=2e-8)
            meta = json.loads((folder/"summary.json").read_text())
            print(f"Reuse {model}/GammaDiv (NH): {digest[:12]}", flush=True)
            return {**meta, "samples": draws.reshape(-1, dimension), "samples_chains": draws, "fit_cache": str(folder)}
        wrapped = BoundedScaleTarget(logprob, d, (SIGMA_MIN,) * s)
        proposal_path = folder/"frozen_proposal.npz"
        proposal_saved = old._restore(proposal_path)
        if proposal_saved is None:
            print(f"GammaDiv (NH): observed-data mode search, seed={seed}", flush=True)
            initials = starts(arrays, model, seed)
            _, means, H, mode_lp, attempts = discover(logprob, initials, d, s)
            old.atomic_json(folder/"modes.json", {"attempts": attempts,
                "modes_physical": to_physical(means, d, SIGMA_MIN, s),
                "mode_log_density_sampling": mode_lp,
                "note": "Modes and Laplace weights construct q only; they are not posterior mass estimates or a proof that all modes were found."})
            old._checkpoint(proposal_path, {"means": means, "hessians": H, "mode_lp": mode_lp})
        else:
            means, H, mode_lp = [proposal_saved[k] for k in ["means", "hessians", "mode_lp"]]
        proposal = Proposal(means, H, mode_lp, d, s, asdict(priors))
        sampling_batch = jax.jit(jax.vmap(wrapped.log_density))
        dependencies = {p.name: old.file_hash(p) for p in [contract_path, proposal_path, folder/"modes.json"]}
        parts = []
        for c in range(chains):
            path = folder/f"mh_chain{c:02d}.npz"
            part = old._restore(path)
            if part is None:
                # Cycle through discovered modes, including low-probability ones,
                # so the gate tests escape from different starting regions.
                initial = means[c % len(means)]
                print(f"Frozen MH {model}/GammaDiv (NH), seed={seed}, chain={c}, start_mode={c % len(means)}", flush=True)
                u, lp_u, info = run_chain(lambda v: old._batch(sampling_batch, v), proposal, initial,
                    seed + 910231 + 100003*c, num_warmup, num_samples)
                z = to_physical(u, d, SIGMA_MIN, s)
                lp = old._batch(batch, z)
                np.testing.assert_allclose(lp + log_jacobian(u, d, SIGMA_MIN, s), lp_u, rtol=1e-10, atol=2e-8)
                part = {"samples_sampling": u, "samples_physical": z, "log_posterior_physical": lp,
                        "start_mode": np.asarray(c % len(means)), **{k: np.asarray(v) for k, v in info.items()}}
                old._checkpoint(path, part)
                receipt = json.loads(path.with_suffix(".json").read_text())
                old.atomic_json(path.with_suffix(".json"), {**receipt, "dependencies": dependencies})
            parts.append(part)
        draws = np.stack([p["samples_physical"] for p in parts])
        lp = np.stack([p["log_posterior_physical"] for p in parts])
        augmented = np.concatenate([draws, lp[..., None]], axis=-1)
        diag = old.diagnostics(augmented, names + ["log_density"], d, n_scales=s, joint=False)
        meta = {**diag, "sampler": "frozen_mixture_independence_mh", "nuts_divergences": None,
                "inference_version": VERSION, "gamma_posterior_definition": DEFINITION,
                "eta": float(eta), "sigma_min": SIGMA_MIN, "target_support_changed": True,
                "scale_coordinates": "standardized_response" if model == "regression" else "original_data",
                "mode_count": len(means), "proposal_component_weights": proposal.weights.tolist(),
                "mh_acceptance_by_chain": [float(p["mh_acceptance_mean"]) for p in parts],
                "mh_acceptance_mean": float(np.mean([p["mh_acceptance_mean"] for p in parts])),
                "importance_ess": float(sum(p["importance_ess"] for p in parts)),
                "sigma_near_floor_fraction": float(np.mean(np.exp(draws[..., d:d+s]) <= SIGMA_MIN*1.1)),
                "fit_contract_sha256": digest}
        old.atomic_json(folder/"summary.json", meta)
        old._checkpoint(final_path, {"samples_physical": draws, "log_posterior_physical": lp})
        receipt = json.loads(final_path.with_suffix(".json").read_text())
        old.atomic_json(final_path.with_suffix(".json"), {**receipt, "dependencies": {
            **dependencies, "summary.json": old.file_hash(folder/"summary.json")}})
        return {**meta, "samples": draws.reshape(-1, dimension), "samples_chains": draws, "fit_cache": str(folder)}
