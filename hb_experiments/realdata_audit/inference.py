"""Use the paper's posterior target/sampler on the already-standardised observations."""
from contextlib import contextmanager
import json
import os

import numpy as np

from . import kf
from .design import VARIANTS, read_json, read_npz, write_json, write_npz


@contextmanager
def sampler_environment(folder, warmup, draws):
    updates = {"HBAYES_CACHE_ROOT": str((folder/"native_cache").resolve()),
               "HBAYES_MH_WARMUP": str(warmup), "HBAYES_MH_SAMPLES": str(draws),
               "HBAYES_MH30_ROOT": None}
    saved = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def gate(meta, s):
    rhat, bulk, tail = (meta.get(k) for k in ("rhat_max", "ess_bulk_min", "ess_tail_min"))
    vals = (rhat, bulk, tail)
    return bool(all(v is not None and np.isfinite(v) for v in vals)
                and rhat <= s["rhat_max"] and min(bulk, tail) >= s["ess_min"])


def posterior(X, y, variant, c, seed, folder):
    from experiments.exp1_regression.utils_exp1_regression import fit_dpd_variant
    from hbayes.paper_sampling import job_context

    phi, kappa = VARIANTS[variant]
    s = c["sampler"]
    trials = [s["draws"]] + ([s["repair_draws"]] if s["repair_draws"] else [])
    history = []
    for i, draws in enumerate(trials):
        tag = folder / f"attempt_{i:02d}"
        sample_path, info_path = tag/"samples.npz", tag/"diagnostics.json"
        if sample_path.exists() and info_path.exists():
            position, info = read_npz(sample_path)["position"], read_json(info_path)
        else:
            with sampler_environment(folder, s["warmup"], draws), job_context():
                v = fit_dpd_variant(
                    X, y, seed=int(seed), chains=s["chains"], num_warmup=s["warmup"],
                    num_samples=draws, gamma=c["gamma"], eta_mode="response_var",
                    holder_phi=phi, holder_kappa=kappa, target_accept=s["target_accept"],
                    init_step=s["initial_step"],
                )
            position = np.asarray(v.pop("samples_chains"), dtype=float)
            v.pop("samples", None)
            info = v
        if position.shape != (s["chains"], draws, X.shape[1]+2) or not np.isfinite(position).all():
            raise ValueError(f"{variant} returned invalid retained draws")
        ok = gate(info, s)
        info.update(audit_diagnostics_pass=ok, draws_per_chain=draws)
        if not sample_path.exists() or not info_path.exists():
            write_npz(sample_path, position=position)
            write_json(info_path, info)
        history.append(dict(attempt=i, draws=draws, valid=ok, rhat_max=info.get("rhat_max"),
                            ess_bulk_min=info.get("ess_bulk_min"), ess_tail_min=info.get("ess_tail_min")))
        write_json(folder/"attempts.json", history)
        if ok:
            break
    return position, {**info, "attempt": history[-1]["attempt"]}


def point(X, y, c, seed, path):
    if path.exists():
        return read_json(path)
    f = kf.fit(X, y, c["gamma"], c["kf_starts"], seed)
    write_json(path, f)
    return f


def bootstrap(X, y, c, anchor, seed, folder):
    """Resample paired rows, re-evaluate the fitted rule on the original rows."""
    n, d = X.shape
    B = c["bootstrap_B"]
    path = folder/"draws.npz"
    theta, eps, diagnostics = [], [], []
    if path.exists():
        z = read_npz(path)
        theta, eps = list(z["theta"]), list(z["epsilon"])
        diagnostics = json.loads(str(z["diagnostics"]))
        if not len(theta) == len(eps) == len(diagnostics) <= B:
            raise ValueError("Invalid bootstrap checkpoint")
    for j in range(len(theta), B):
        rng = np.random.default_rng(np.random.SeedSequence([seed, j]))
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        try:
            f = kf.fit(X, y, c["gamma"], c["bootstrap_starts"], int(rng.integers(2**31)),
                       counts=counts, anchor=anchor)
            theta.append(f.pop("theta"))
            eps.append(f.pop("epsilon"))
            diagnostics.append(dict(replicate=j, **f))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            theta.append(np.full(d+1, np.nan))
            eps.append(np.nan)
            diagnostics.append(dict(replicate=j, valid=False, error=f"{type(exc).__name__}: {exc}"))
        if (j+1) % 25 == 0 or j+1 == B:
            write_npz(path, theta=np.asarray(theta), epsilon=np.asarray(eps),
                      diagnostics=np.array(json.dumps(diagnostics, allow_nan=False)))
            write_json(folder/"diagnostics.json", diagnostics)
            print(f"  K-F bootstrap {j+1}/{B}", flush=True)
    return np.asarray(theta), np.asarray(eps), diagnostics
