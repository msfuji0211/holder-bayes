"""Fit each phi with the repository's bounded regression target and dense NUTS."""
from pathlib import Path

import numpy as np

from hbayes.paper_sampling import atomic_json, atomic_npz, diagnostics, regression_starts
from .inputs import VARIANTS

FLOOR = .05
NAMES = ["intercept", "colour", "colour_squared", "log_scale", "rho"]


def fit_once(X, y, variant, c, seed, warmup, draws):
    import blackjax
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from hbayes.bounded_scale import BoundedScaleTarget
    from hbayes.experiment_temperature import response_temperature
    from hbayes.posteriors import GibbsConfig, PriorsConfig, logprob_joint_regression_from_vector

    phi, kappa = VARIANTS[variant]
    d, chains = X.shape[1], c["chains"]
    eta, _ = response_temperature(y, c["gamma"])
    with jax.experimental.enable_x64():
        priors = PriorsConfig(mu0=0., sigma0=10., m_log_sigma=0., s_log_sigma=1., a_c=1., b_c=1.)
        cfg = GibbsConfig(gamma=c["gamma"], eta=eta, holder_phi=phi, holder_kappa=kappa, priors=priors)
        base = logprob_joint_regression_from_vector(jnp.asarray(X), jnp.asarray(y), cfg)
        bounded = BoundedScaleTarget(base, d, FLOOR)
        target = jax.jit(bounded.log_density)
        vg = jax.jit(jax.value_and_grad(lambda u: -target(u)))

        def objective(u):
            value, grad = vg(jnp.asarray(u))
            return float(value), np.asarray(grad)

        initial = regression_starts(X, y, FLOOR, seed, chains, joint=True)
        starts = np.asarray(bounded.to_unconstrained(initial))
        fits = []
        for p in starts:
            fit = minimize(objective, p, method="L-BFGS-B", jac=True,
                           options={"maxiter": 1000, "gtol": 1e-7, "ftol": 1e-12})
            if np.isfinite(fit.fun) and np.isfinite(fit.x).all():
                fits.append((float(fit.fun), fit.x))
        if not fits:
            raise RuntimeError("Could not find a finite NUTS starting point")
        best_value, best = min(fits, key=lambda pair: pair[0])
        h = np.asarray(jax.hessian(lambda u: -target(u))(jnp.asarray(best)))
        if np.isfinite(h).all():
            eig, Q = np.linalg.eigh((h+h.T)/2)
            cov = (Q/np.maximum(eig, 1e-3)) @ Q.T
            jitter = np.random.default_rng(seed).multivariate_normal(np.zeros(d+2), .2*cov, size=chains)
            starts = best+jitter
        else:
            starts = np.repeat(best[None, :], chains, axis=0)
        for k in range(chains):
            if not np.isfinite(float(target(jnp.asarray(starts[k])))):
                starts[k] = best

        adapt = blackjax.window_adaptation(
            blackjax.nuts, target, is_mass_matrix_diagonal=False, initial_step_size=c["initial_step"],
            target_acceptance_rate=c["target_accept"], progress_bar=False, max_num_doublings=9,
        )
        warm_key, draw_key = jax.random.split(jax.random.PRNGKey(seed))
        warm_keys = jax.random.split(warm_key, chains)
        (states, params), _ = jax.jit(jax.vmap(
            lambda key, pos: adapt.run(key, pos, num_steps=warmup)
        ))(warm_keys, jnp.asarray(starts))
        kernel = blackjax.nuts.build_kernel()

        def transition(state, keys):
            def one(key, s, step, mass):
                return kernel(key, s, target, step, mass, max_num_doublings=9)
            nxt, info = jax.vmap(one)(keys, state, params["step_size"], params["inverse_mass_matrix"])
            return nxt, (nxt.position, info.is_divergent, info.acceptance_rate, info.num_integration_steps)

        keys = jax.random.split(draw_key, (draws, chains))
        _, (positions, divergent, acceptance, steps) = jax.jit(
            lambda state, ks: jax.lax.scan(transition, state, ks)
        )(states, keys)
        position = np.asarray(bounded.to_physical(positions)).transpose(1, 0, 2)
        divergent, acceptance, steps = map(np.asarray, (divergent, acceptance, steps))

    if not np.isfinite(position).all() or not np.isfinite(acceptance).all():
        raise ValueError("The NUTS fit returned non-finite values")
    diag = diagnostics(position, NAMES, d, joint=True)
    count_divergences = int(divergent.sum())
    diag["diagnostics_pass"] = bool(diag["diagnostics_pass"] and count_divergences == 0)
    return position, dict(
        **diag, phi=phi, kappa=kappa, eta=eta, seed=int(seed), chains=chains,
        warmup_per_chain=warmup, draws_per_chain=draws, divergences=count_divergences,
        acceptance_mean=float(acceptance.mean()), max_integration_steps=int(steps.max()),
        tree_limit_fraction=float((steps >= 511).mean()), map_objective=best_value, sigma_min=FLOOR,
    )


def run_fit(X, y, variant, c, seed, folder):
    """Save each completed attempt; repeat only if its diagnostics fail."""
    import json

    folder = Path(folder)
    selected = None
    for attempt, (warmup, draws) in enumerate((
        (c["warmup"], c["draws"]), (c["repair_warmup"], c["repair_draws"]),
    )):
        stage = folder / f"attempt_{attempt:02d}"
        result, info = stage/"samples.npz", stage/"diagnostics.json"
        if result.is_file() and info.is_file():
            with np.load(result, allow_pickle=False) as z:
                position = z["position"]
            diag = json.loads(info.read_text(encoding="utf-8"))
        else:
            print(f"    NUTS attempt {attempt+1}: {c['chains']} chains, {warmup} warmup, {draws} draws", flush=True)
            position, diag = fit_once(X, y, variant, c, seed+100_000*attempt, warmup, draws)
            atomic_npz(result, position=position)
            atomic_json(info, diag)
        if position.shape != (c["chains"], draws, X.shape[1]+2) or not np.isfinite(position).all():
            raise ValueError(f"Invalid retained draws in {result}")
        selected = dict(attempt=attempt, passed=diag["diagnostics_pass"], diagnostics=diag)
        if selected["passed"]:
            break
    atomic_json(folder/"fit.json", selected)
    return selected
