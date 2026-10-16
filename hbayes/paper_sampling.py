"""Resumable paper inference shared by regression, benchmark and cleaning.

Dataset generation remains in float32; inference is promoted locally to float64.
H-B uses a frozen proposal with an exact MH correction. Other methods retain
their original target and use dense-mass NUTS. Cache identity includes actual
likelihood inputs, priors, target source and every sampler setting.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import expit

from .independence_mh import FrozenProposal, to_physical, to_sampling, run_chain

VERSION = "holder_regression_floor005_mh_v1"
CONTEXT = {}


def array_hash(a):
    a = np.ascontiguousarray(a)
    header = json.dumps({"shape": list(a.shape), "dtype": str(a.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header+a.tobytes()).hexdigest()


def file_hash(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(x):
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **values)
    temp.replace(path)


@contextmanager
def job_context(**values):
    old = dict(CONTEXT)
    CONTEXT.clear()
    CONTEXT.update(values)
    try:
        yield
    finally:
        CONTEXT.clear()
        CONTEXT.update(old)


def diagnostics(draws, names, scale_start, n_scales=1, joint=False):
    import arviz as az
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[-1] != len(names):
        raise ValueError("Diagnostic chain layout does not match variable names")
    posterior = {name: draws[..., i] for i, name in enumerate(names)}
    with np.errstate(over="ignore"):
        for i in range(n_scales):
            posterior[f"sigma_{i}"] = np.exp(draws[..., scale_start+i])
        if joint:
            posterior["epsilon"] = expit(-draws[..., -1])
    data = az.from_dict(posterior=posterior)
    arrays = {"rhat": az.rhat(data, method="rank"),
              "ess_bulk": az.ess(data, method="bulk"),
              "ess_tail": az.ess(data, method="tail")}
    by = {name: {stat: float(values[name].values) for stat, values in arrays.items()}
          for name in posterior}
    values = np.array([list(row.values()) for row in by.values()])
    good = np.isfinite(values).all()
    result = {"by_parameter": by, "diagnostic_backend": f"ArviZ {az.__version__}",
              "rhat_max": float(values[:, 0].max()),
              "ess_bulk_min": float(values[:, 1].min()),
              "ess_tail_min": float(values[:, 2].min())}
    result["diagnostics_pass"] = bool(good and result["rhat_max"] <= 1.01
                                     and result["ess_bulk_min"] >= 400
                                     and result["ess_tail_min"] >= 400)
    return result


def regression_starts(X, y, floor, seed, chains, joint):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    order = np.argsort(y-X@ols)
    half = len(y)//2
    subsets = [order[:half], np.arange(len(y)), order[half:], order[:half]]
    rng = np.random.default_rng(seed+314159)
    starts = []
    for i in range(chains):
        idx = subsets[i % 4] if joint else np.arange(len(y))
        # Underdetermined halves do not provide a useful dispersed start.
        if len(idx) <= X.shape[1]:
            idx = np.arange(len(y))
        beta = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        residual = y[idx]-X[idx]@beta
        sigma = max(float(1.4826*np.median(np.abs(residual-np.median(residual)))),
                    (floor or 1e-4)*1.5)
        if i % 4 == 3 or not joint:
            beta = beta+rng.normal(size=X.shape[1])*sigma/np.sqrt(len(idx))
            sigma *= 1.1
        out = np.r_[beta, np.log(sigma)]
        if joint:
            c = float(np.clip(np.sqrt(2)*np.mean(np.exp(-.5*((y-X@beta)/sigma)**2)), .05, .95))
            out = np.r_[out, np.log(c)-np.log1p(-c)]
        starts.append(out)
    return np.stack(starts)


def density_starts(x, seed, chains, joint):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x[:, 0])
    subsets = [order[:len(x)//2], np.arange(len(x)), order[len(x)//2:], order[:len(x)//2]]
    rng = np.random.default_rng(seed+314159)
    out = []
    for i in range(chains):
        part = x[subsets[i % 4]] if joint else x
        mu = np.median(part, axis=0)
        scale = np.maximum(1.4826*np.median(np.abs(part-mu), axis=0), 1e-3)
        if i % 4 == 3 or not joint:
            mu += rng.normal(size=x.shape[1])*scale/np.sqrt(len(part))
        z = np.r_[mu, np.log(scale)]
        if joint:
            z = np.r_[z, 0.]
        out.append(z)
    return np.stack(out)


def runtime():
    names = ["numpy", "scipy", "jax", "jaxlib", "blackjax", "arviz"]
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in names}}


def target_sources():
    root = Path(__file__).parent
    return {name: file_hash(root/name) for name in
            ["models.py", "dpd.py", "posteriors.py", "bounded_scale.py"]}


def _batch(logprob, z, batch=64):
    """Pad the final batch to avoid compiling a different shape for every tail."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    values = []
    for i in range(0, len(z), batch):
        part = z[i:i+batch]
        padded = np.pad(part, ((0, batch-len(part)), (0, 0)), mode="edge")
        values.append(np.asarray(logprob(padded), dtype=float)[:len(part)])
    return np.concatenate(values)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _checkpoint(path, payload):
    atomic_npz(path, **payload)
    atomic_json(path.with_suffix(".json"), {"sha256": file_hash(path)})


def _restore(path):
    receipt = path.with_suffix(".json")
    if not path.exists() or not receipt.exists():
        return None
    record = json.loads(receipt.read_text())
    if record["sha256"] != file_hash(path):
        raise ValueError(f"Checkpoint hash differs: {path}")
    for name, digest in record.get("dependencies", {}).items():
        if file_hash(path.parent/name) != digest:
            raise ValueError(f"Checkpoint metadata hash differs: {path.parent/name}")
    return _load_npz(path)


def _import_all30(contract, base_batch, names, scale_start, arrays):
    root = os.environ.get("HBAYES_MH30_ROOT")
    c = CONTEXT
    if not root or c.get("kind") != "regression" or c.get("eps") != .4:
        return None
    if (contract["method"] != "DPD" or contract["gamma"] != 1.
            or contract["floor"] != .05 or contract["kappa"] != 1.
            or contract["chains"] != 4):
        return None
    folder = Path(root)/"posterior"/f"rep{c['rep']:03d}"
    if not folder.exists():
        folder = Path(root)/f"rep{c['rep']:03d}"
    if not folder.exists():
        raise FileNotFoundError(f"Requested completed MH result is missing: {folder}")
    summary = json.loads((folder/"summary.json").read_text())
    receipt = json.loads((folder/"completion.json").read_text())
    for name, digest in receipt["output_sha256"].items():
        if file_hash(folder/name) != digest:
            raise ValueError(f"Imported MH result hash differs: {folder/name}")
    for name, digest in contract["target_sources"].items():
        recorded = summary.get("original_source_sha256", {}).get("hbayes."+name[:-3])
        if recorded != digest:
            raise ValueError(f"Completed MH target source differs: {name}")
    if (summary["gamma"] != 1. or summary["sigma_min"] != .05
            or summary["data_seed"] != c.get("data_seed")
            or not np.isclose(summary["eta"], contract["eta"], rtol=1e-12, atol=1e-12)):
        raise ValueError("Completed MH target does not match the requested condition")
    for name, arr in zip(["X", "y"], arrays):
        if array_hash(np.asarray(arr)) != summary["input_array_hashes"][name]:
            raise ValueError(f"Completed MH likelihood input differs: {name}")
    payload = _load_npz(folder/"samples.npz")
    draws = payload["samples_physical"]
    if draws.shape != (4, contract["samples"], scale_start+2):
        raise ValueError("Completed MH draw count differs from the requested experiment")
    if not np.isfinite(draws).all() or np.any(np.exp(draws[..., scale_start]) <= .05):
        raise ValueError("Invalid completed MH positions or scale support")
    actual = _batch(base_batch, draws.reshape(-1, draws.shape[-1]))
    saved = payload.get("log_posterior_physical")
    if saved is not None:
        np.testing.assert_allclose(actual, saved.reshape(-1), rtol=1e-10, atol=2e-8)
    diag = diagnostics(draws, names, scale_start, joint=True)
    # Saved names use beta0/log_sigma; compare aggregate checks and all values
    # by the physical ordering, avoiding accidental label-based selection.
    for name in ["rhat_max", "ess_bulk_min", "ess_tail_min"]:
        np.testing.assert_allclose(diag[name], summary[name], rtol=1e-8, atol=1e-8)
    if not diag["diagnostics_pass"] or not summary["diagnostics_pass"]:
        raise ValueError("The completed MH run has failed diagnostics")
    return draws, actual.reshape(draws.shape[:2]), {"origin": "completed_mh30",
        "imported_from": str(folder), "imported_samples_sha256": file_hash(folder/"samples.npz"),
        "max_saved_physical_logp_error": None if saved is None else float(np.max(np.abs(actual-saved.ravel()))),
        "sampler": "frozen_mixture_independence_mh", "nuts_divergences": None}


def _import_legacy(contract, arrays, base_batch, names, d, n_scales, joint):
    """Reuse old draws only with identical observed inputs and passing diagnostics.

    Metrics alone do not recover changed shared-budget recalls or ranks. Missing
    draws therefore trigger an explicit rank-recovery fit, rather than pretending
    an old budget-dependent score remains valid.
    """
    c = CONTEXT
    legacy = c.get("legacy_folder")
    if not legacy or contract["method"] in c.get("force_methods", []):
        return None
    if contract["floor"] is not None:
        return None  # Old regression H-B has a different support.
    path = Path(legacy)/"samples.npz"
    if not path.exists():
        return None
    raw = _load_npz(path)
    if contract["model"] == "regression":
        if not all(k in raw for k in ["x_train", "y_train", "x_test"]):
            return None
        from experiments.exp1_regression.utils_exp1_regression import standardize_regression_data
        old_X, old_y, _, _ = standardize_regression_data(raw["x_train"], raw["y_train"], raw["x_test"])
        observed = [old_X, old_y]
    else:
        if "x" not in raw:
            return None
        observed = [raw["x"]]
    if [array_hash(np.asarray(a, dtype=np.float64)) for a in observed] != contract["input_hashes"]:
        raise ValueError(f"Legacy data differ: {path}")
    method_keys = {"DPD": "dpd", "ExpSatDPD": "exp_sat_dpd", "RatSatDPD": "rat_sat_dpd",
                   "NLL": "nll", "GammaDiv": "gamma", "StudentT": "student_t"}
    if contract["method"] not in method_keys:
        return None
    suffix = method_keys[contract["method"]]
    key = "samples_"+suffix
    if key not in raw:
        return None
    draws = raw[key]
    if draws.ndim != 2 or draws.shape[0] % contract["chains"]:
        raise ValueError("Legacy chain layout cannot be reconstructed")
    draws = draws.reshape(contract["chains"], -1, len(names)).astype(float)
    old_cfg = c["legacy_payload"].get("run_config", {})
    for key in ["chains", "num_samples", "base_seed", "gamma", "student_t_df", "eta_gamma_div"]:
        current = {"chains": contract["chains"], "num_samples": draws.shape[1],
                   "base_seed": c.get("base_seed", 42), "gamma": c.get("gamma", contract["gamma"]),
                   "student_t_df": contract["student_t_df"], "eta_gamma_div": c.get("eta_gamma_div", 1.)}[key]
        if old_cfg.get(key) != current:
            raise ValueError(f"Legacy settings differ: {key}")
    if draws.shape[1] != contract["samples"] or not np.isfinite(draws).all():
        return None
    diag = diagnostics(draws, names, d, n_scales=n_scales, joint=joint)
    block_key = "standard" if contract["method"] == "NLL" and contract["model"] == "density" else suffix
    block = c["legacy_payload"].get(block_key, {})
    if not diag["diagnostics_pass"] or block.get("nuts_divergences") != 0:
        return None
    lp = _batch(base_batch, draws.reshape(-1, len(names)))
    if not np.isfinite(lp).all():
        return None
    return draws, lp.reshape(draws.shape[:2]), {"origin": "verified_legacy_draws",
        "sampler": "legacy_nuts", "nuts_divergences": 0,
        "imported_from": str(path), "imported_samples_sha256": file_hash(path)}


def fit(logprob, inputs, *, method, model, seed, chains, num_warmup, num_samples,
        target_accept, init_step, priors, eta=1., gamma=1., phi=None, kappa=1.,
        student_t_df=4., floor=None, names=None):
    import jax
    import jax.numpy as jnp
    from .sampling import run_nuts_adapted
    from .bounded_scale import BoundedScaleTarget

    joint = phi is not None
    arrays = [np.asarray(a, dtype=np.float64) for a in inputs]
    d = arrays[0].shape[1]
    n_scales = 1 if model == "regression" else d
    if floor is not None and (not joint or model != "regression"):
        raise ValueError("The 0.05 support restriction applies only to regression H-B")
    names = names or ([f"beta{i}" for i in range(d)] if model == "regression"
                       else [f"mu{i}" for i in range(d)]) + [f"log_sigma{i}" for i in range(n_scales)] + (["rho"] if joint else [])
    mh_warmup = int(os.environ.get("HBAYES_MH_WARMUP", num_warmup))
    mh_draws = int(os.environ.get("HBAYES_MH_SAMPLES", num_samples))
    contract = {"version": VERSION, "method": method, "model": model, "seed": int(seed),
        "input_hashes": [array_hash(a) for a in arrays], "priors": asdict(priors),
        "eta": float(eta), "gamma": float(gamma) if joint or method == "GammaDiv" else 1., "phi": phi, "kappa": float(kappa),
        "student_t_df": float(student_t_df), "floor": floor, "chains": int(chains),
        "warmup": int(num_warmup), "samples": int(mh_draws if joint else num_samples),
        "proposal_samples": int(num_samples), "mh_warmup": mh_warmup,
        "target_accept": float(target_accept), "init_step": float(init_step),
        "dense_mass": True, "target_sources": target_sources(),
        "sampler_sources": {n: file_hash(Path(__file__).parent/n) for n in
                            ["paper_sampling.py", "independence_mh.py", "sampling.py"]},
        "proposal": {"df": 7., "inflation": 1.5, "prior_weight": .1} if joint else None,
        "runtime": runtime()}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get("HBAYES_CACHE_ROOT", "outputs/holder_floor005_cache"))
    folder = root/model/method/digest
    folder.mkdir(parents=True, exist_ok=True)
    cfg = folder/"contract.json"
    if cfg.exists() and json.loads(cfg.read_text()) != contract:
        raise ValueError("Cache identity collision")
    atomic_json(cfg, contract)
    # local x64 context keeps the existing float32 data generators unchanged.
    with jax.experimental.enable_x64():
        base_batch = jax.jit(jax.vmap(logprob))
        final_path = folder/"posterior.npz"
        restored = _restore(final_path)
        if restored is not None:
            meta = json.loads((folder/"summary.json").read_text())
            draws = restored["samples_physical"]
            if draws.shape != (chains, contract["samples"], len(names)):
                raise ValueError("Cached posterior has the wrong chain layout")
            checks = _batch(base_batch, draws.reshape(-1, len(names)))
            np.testing.assert_allclose(checks, restored["log_posterior_physical"].ravel(), rtol=1e-10, atol=2e-8)
            print(f"Reuse {model}/{method}: {digest[:12]}", flush=True)
            return {**meta, "samples": draws.reshape(-1, len(names)), "samples_chains": draws,
                    "fit_cache": str(folder), "fit_contract_sha256": digest}
        imported = _import_all30(contract, base_batch, names, d, [np.asarray(a) for a in inputs]) if joint else None
        if imported is None:
            imported = _import_legacy(contract, arrays, base_batch, names, d, n_scales, joint)
        if imported is not None:
            draws, logp_physical, stats = imported
        else:
            if CONTEXT.get("legacy_folder"):
                reason = ("changed regression H-B support" if floor is not None else
                          "failed condition or missing verified draws/ranks")
                print(f"Recompute {model}/{method}: {reason}", flush=True)
            initials = (regression_starts(*arrays, floor, seed, chains, joint)
                        if model == "regression" else density_starts(arrays[0], seed, chains, joint))
            bounded = BoundedScaleTarget(logprob, d, floor) if floor is not None else None
            sampling_logprob = bounded.log_density if bounded else logprob
            sampling_batch = jax.jit(jax.vmap(sampling_logprob))
            keys = jax.random.split(jax.random.PRNGKey(seed+271828 if joint else seed), chains)
            pilot, pilot_stats = [], []
            for c, key in enumerate(keys):
                path = folder/f"nuts_chain{c:02d}.npz"
                saved = _restore(path)
                if saved is None:
                    print(f"NUTS {model}/{method}, seed={seed}, chain={c}", flush=True)
                    u0 = to_sampling(initials[c], d, floor, n_scales)
                    if not np.isfinite(float(sampling_logprob(jnp.asarray(u0, dtype=jnp.float64)))):
                        raise ValueError("Non-finite NUTS initial density")
                    result = run_nuts_adapted(key, sampling_logprob, jnp.asarray(u0, dtype=jnp.float64),
                        num_warmup=num_warmup, num_samples=num_samples,
                        initial_step_size=init_step, target_acceptance_rate=target_accept,
                        is_mass_matrix_diagonal=False)
                    u = np.asarray(result["samples"], dtype=float)
                    z = to_physical(u, d, floor, n_scales)
                    saved = {"samples_sampling": u, "samples_physical": z,
                             "log_posterior_physical": _batch(base_batch, z),
                             "divergences": np.asarray(sum(bool(info.is_divergent) for info in result["infos"])),
                             "acceptance": np.asarray(np.mean([float(info.acceptance_rate) for info in result["infos"]]))}
                    if not np.isfinite(u).all() or not np.isfinite(saved["log_posterior_physical"]).all():
                        raise ValueError("NUTS retained a non-finite position/density")
                    _checkpoint(path, saved)
                pilot.append(saved["samples_sampling"])
                pilot_stats.append({k: float(saved[k]) for k in ["divergences", "acceptance"]})
            pilot = np.stack(pilot)
            if joint:
                proposal = FrozenProposal(pilot, d, n_scales, floor, asdict(priors))
                atomic_npz(folder/"frozen_proposal.npz", means=proposal.means,
                           scatter=proposal.scatter, weights=proposal.weights)
                parts = []
                for c in range(chains):
                    path = folder/f"mh_chain{c:02d}.npz"
                    saved = _restore(path)
                    if saved is None:
                        print(f"Frozen MH {model}/{method}, seed={seed}, chain={c}", flush=True)
                        u, lp, info = run_chain(lambda v: _batch(sampling_batch, v), proposal,
                            pilot[c, -1], seed+910231+100003*c, mh_warmup, mh_draws)
                        z = to_physical(u, d, floor, n_scales)
                        physical_lp = _batch(base_batch, z)
                        jac = (-np.logaddexp(0, -u[:, d:d+n_scales]).sum(axis=1)
                               if floor is not None else np.zeros(len(u)))
                        np.testing.assert_allclose(physical_lp+jac, lp, rtol=1e-10, atol=2e-8)
                        saved = {"samples_sampling": u, "samples_physical": z,
                                 "log_posterior_physical": physical_lp, **{k: np.asarray(v) for k, v in info.items()}}
                        _checkpoint(path, saved)
                    parts.append(saved)
                draws = np.stack([p["samples_physical"] for p in parts])
                logp_physical = np.stack([p["log_posterior_physical"] for p in parts])
                stats = {"sampler": "frozen_mixture_independence_mh", "nuts_divergences": None,
                         "proposal_nuts_divergences": int(sum(p["divergences"] for p in pilot_stats)),
                         "proposal_nuts_acceptance_mean": float(np.mean([p["acceptance"] for p in pilot_stats])),
                         "mh_acceptance_mean": float(np.mean([p["mh_acceptance_mean"] for p in parts])),
                         "importance_ess": float(sum(p["importance_ess"] for p in parts))}
            else:
                draws = pilot
                logp_physical = np.stack([_batch(base_batch, z) for z in draws])
                stats = {"sampler": "dense_mass_nuts", "nuts_divergences": int(sum(p["divergences"] for p in pilot_stats)),
                         "nuts_acceptance_mean": float(np.mean([p["acceptance"] for p in pilot_stats]))}
        diag = diagnostics(draws, names, d, n_scales=n_scales, joint=joint)
        meta = {**diag, **stats, "inference_version": VERSION, "sigma_min": floor,
                "scale_coordinates": "standardized_response" if model == "regression" else "original_data",
                "target_support_changed": bool(floor is not None), "fit_contract_sha256": digest}
        if floor is not None:
            sigma = np.exp(draws[..., d])
            meta["sigma_near_floor_fraction"] = float(np.mean(sigma <= floor*1.1))
        atomic_json(folder/"summary.json", meta)
        _checkpoint(final_path, {"samples_physical": draws, "log_posterior_physical": logp_physical})
        receipt = json.loads(final_path.with_suffix(".json").read_text())
        receipt["dependencies"] = {"summary.json": file_hash(folder/"summary.json"),
                                   "contract.json": file_hash(cfg)}
        atomic_json(final_path.with_suffix(".json"), receipt)
        return {**meta, "samples": draws.reshape(-1, len(names)), "samples_chains": draws,
                "fit_cache": str(folder)}
