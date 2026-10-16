"""Small, separately saved pilot of the scale-bounded H-B regression target."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import textwrap
import time
import zipfile

EXPECTED_GENERATORS = {
    "simulate_regression_dataset": "edd179ec942adaadd7ddbd6e589488ce0742366bb02a2e0a298fa1957f454800",
    "standardize_regression_data": "f0c2ad36eb7572249a5e4d18b294acd9ad51183314f1e4f2cf3333c906116819",
}
EXPECTED_CORE = {
    "hbayes.sampling": "2d2c020bd5c8eb2c5a54d191c1861a28e27893aaf337ebfb10de4e57726237eb",
    "hbayes.posteriors": "00d150c5d1cf8fdad442a5c3fc2f392249036f41427e1ac673ff5c3c6b91eebc",
    "hbayes.bounded_scale": "00fa11cc4a5e763a3c55b46ee15c316698c0e5edc863d6a8750cfdc481a19cce",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rep-ids", type=int, nargs="+", default=[0, 1, 4])
    p.add_argument("--sigma-mins", type=float, nargs="+", default=[0.05])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/scale_corrected_pilot_v1"))
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--num-warmup", type=int, default=4000)
    p.add_argument("--num-samples", type=int, default=4000)
    p.add_argument("--target-accept", type=float, default=0.99)
    p.add_argument("--init-step", type=float, default=0.1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if len(set(a.rep_ids)) != len(a.rep_ids) or any(r < 0 or r >= 30 for r in a.rep_ids):
        p.error("rep-ids must be distinct original IDs from 0 through 29")
    if len(set(a.sigma_mins)) != len(a.sigma_mins) or any(not 0 < s < float('inf') for s in a.sigma_mins):
        p.error("sigma-mins must be distinct, finite, positive values")
    if a.num_warmup < 100 or a.num_samples < 4 or not 0 < a.target_accept < 1 or not 0 < a.init_step < float('inf'):
        p.error("Invalid sampler settings")
    a.input_dir = a.input_dir or a.output_dir / "inputs"
    return a


def plan(a):
    return {"pilot_version": "scale_bounded_regression_pilot_v1", "method": "H-B / DPD",
            "gamma": 1.0, "eps_true": 0.4, "mu_out": 6.0, "n": 1000, "n_test": 1000,
            "rep_ids": a.rep_ids, "seeds": [400042 + 1000 * r for r in a.rep_ids],
            "sigma_mins": a.sigma_mins, "fits": len(a.rep_ids) * len(a.sigma_mins), "chains": 4,
            "num_warmup": a.num_warmup, "num_samples": a.num_samples,
            "target_accept": a.target_accept, "initial_step": a.init_step,
            "mass_matrix": "dense", "data_dtype": "float32", "inference_dtype": "float64",
            "output_dir": str(a.output_dir), "input_dir": str(a.input_dir),
            "initialization": "OLS, lower and upper residual halves; data-derived c; no oracle parameters",
            "scope": "diagnostic pilot; not the paper's full experiment suite"}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(partial, path)


def atomic_npz(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)


def array_hash(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    header = json.dumps({"shape": list(a.shape), "dtype": str(a.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def source_fingerprint(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()


def prepare_inputs(a, utils, runtime):
    import numpy as np
    fingerprints = {name: source_fingerprint(getattr(utils, name)) for name in EXPECTED_GENERATORS}
    if fingerprints != EXPECTED_GENERATORS:
        raise ValueError("Data generator/standardization differs from the recorded experiment; reconcile it before reproducing the pilot")
    data = {}
    for rep in a.rep_ids:
        folder = a.input_dir / f"rep{rep:03d}"
        npz_path, meta_path = folder / "data.npz", folder / "data.json"
        settings = {"n": 1000, "n_test": 1000, "x_dim": 2, "eps": 0.4, "mu_out": 6.0,
                    "contamination_mode": "intercept_shift", "seed": 400042 + 1000 * rep,
                    "generator_fingerprints": fingerprints}
        if npz_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["settings"] != settings:
                raise ValueError(f"Saved input settings differ: {folder}; choose a new input-dir")
            with np.load(npz_path, allow_pickle=False) as z:
                arrays = {name: z[name] for name in z.files}
            if {k: array_hash(v) for k, v in arrays.items()} != meta["array_hashes"]:
                raise ValueError(f"Saved input hashes differ: {folder}")
        else:
            raw = utils.simulate_regression_dataset(**{k: v for k, v in settings.items() if k != "generator_fingerprints"})
            X, y, X_test, scaler = utils.standardize_regression_data(raw["x_train"], raw["y_train"], raw["x_test"])
            arrays = {"X": np.asarray(X), "y": np.asarray(y), "X_test": np.asarray(X_test),
                      **{k: np.asarray(v) for k, v in raw.items()},
                      **{f"scaler_{k}": np.asarray(v) for k, v in scaler.items()}}
            meta = {"settings": settings, "generation_runtime": runtime,
                    "array_hashes": {k: array_hash(v) for k, v in arrays.items()}}
            atomic_npz(npz_path, **arrays)
            atomic_json(meta_path, meta)
        data[rep] = (arrays, meta)
    return data


def initial_positions(X, y, floor, seed):
    """Use observed data only. Separate modes are intentionally explored."""
    import numpy as np
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    order = np.argsort(y - X @ ols)
    half = len(y) // 2
    subsets = [order[:half], np.arange(len(y)), order[half:], order[:half]]
    labels = ["lower_residual_half", "OLS", "upper_residual_half", "lower_half_perturbed"]
    rng = np.random.default_rng(seed + 314159)
    starts = []
    for i, idx in enumerate(subsets):
        beta = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        residual = y[idx] - X[idx] @ beta
        sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        sigma = max(float(sigma), floor * 1.5)
        if i == 3:
            beta = beta + rng.normal(size=X.shape[1]) * sigma / np.sqrt(len(idx))
            sigma *= 1.1
        c = np.sqrt(2.0) * np.mean(np.exp(-0.5 * ((y - X @ beta) / sigma) ** 2))
        c = float(np.clip(c, 0.05, 0.95))
        starts.append(np.r_[beta, np.log(sigma), np.log(c) - np.log1p(-c)])
    return np.asarray(starts), labels


def claim_output_dir(root):
    """Require a dedicated pilot directory before any experiment writes."""
    marker = root / "pilot_directory.json"
    identity = {"pilot_version": "scale_bounded_regression_pilot_v1"}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"Output directory belongs to another experiment: {root}")
    else:
        if root.exists() and any(p.name != "inputs" for p in root.iterdir()):
            raise ValueError(f"Use a new, separate output directory: {root}")
        atomic_json(marker, identity)


def make_archive(root, input_root, rep_ids):
    path, partial = root / "pilot_results.zip", root / "pilot_results.partial.zip"
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p not in (path, partial) and ".partial" not in p.name:
                z.write(p, p.relative_to(root))
        if not input_root.resolve().is_relative_to(root.resolve()):
            for rep in rep_ids:
                for name in ["data.npz", "data.json"]:
                    p = input_root / f"rep{rep:03d}" / name
                    z.write(p, f"inputs/rep{rep:03d}/{name}")
    os.replace(partial, path)


def run(a):
    import arviz as az
    import blackjax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.special import expit
    import hbayes.bounded_scale as bounds
    import hbayes.diagnostics as diagnostics
    import hbayes.models as models
    import hbayes.dpd as dpd
    import hbayes.posteriors as posteriors
    import hbayes.sampling as sampling
    import hbayes.experiment_temperature as temperature
    from hbayes.experiment_temperature import response_temperature, TEMPERATURE_VERSION
    from experiments.exp1_regression import utils_exp1_regression as utils

    runtime = {"python": platform.python_version(), "jax": jax.__version__,
               "blackjax": blackjax.__version__, "numpy": np.__version__, "arviz": az.__version__,
               "backend": jax.default_backend()}
    source = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
              for m in [bounds, diagnostics, models, dpd, posteriors, sampling, temperature, utils]}
    for name, expected in EXPECTED_CORE.items():
        if source[name] != expected:
            raise ValueError(f"Patched module differs: {name}; install the supplied replacement_files before this pilot")
    source["pilot_runner"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    claim_output_dir(a.output_dir)
    # Generate all datasets before enabling x64 for inference. Reuse saved
    # arrays for every floor and subsequent invocation.
    jax.config.update("jax_enable_x64", False)
    inputs = prepare_inputs(a, utils, runtime)
    jax.config.update("jax_enable_x64", True)
    jobs, failures = [], []

    for floor in a.sigma_mins:
        tag = format(floor, ".12g").replace(".", "p")
        for rep in a.rep_ids:
            folder = a.output_dir / f"sigma_min{tag}" / f"rep{rep:03d}"
            arrays, input_meta = inputs[rep]
            seed = input_meta["settings"]["seed"]
            X = jnp.asarray(arrays["X"], dtype=jnp.float64)
            y = jnp.asarray(arrays["y"], dtype=jnp.float64)
            d = X.shape[1]
            eta, variance = response_temperature(arrays["y"], 1.0)
            gcfg = posteriors.MVNGibbsConfig(gamma=1.0, eta=eta, holder_phi="dpd",
                                            priors=posteriors.MVNPriorsConfig())
            base = posteriors.logprob_joint_regression_from_vector(X, y, gcfg)
            target = bounds.BoundedScaleTarget(base, log_scale_start=d, sigma_min=floor)
            initials, labels = initial_positions(arrays["X"], arrays["y"], floor, seed)
            config = {**plan(a), "rep": rep, "sigma_min": floor, "data_seed": seed,
                      "fit_seed": seed + 271828, "eta": eta, "response_variance": variance,
                      "temperature_version": TEMPERATURE_VERSION, **target.metadata(),
                      "runtime": runtime, "source_sha256": source,
                      "initial_positions_physical": initials.tolist(), "initial_labels": labels,
                      "input_array_hashes": input_meta["array_hashes"]}
            # Job-specific settings must not depend on which other jobs are
            # requested, so pilot -> sensitivity can safely resume checkpoints.
            for name in ["rep_ids", "seeds", "sigma_mins", "fits", "output_dir", "input_dir"]:
                config.pop(name, None)
            config_path = folder / "fit_config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError(f"Existing fit settings differ: {folder}; choose a new output-dir")
            atomic_json(config_path, config)
            try:
                keys = jax.random.split(jax.random.PRNGKey(seed + 271828), 4)
                chains, sampling_chains, all_stats, chain_summaries = [], [], [], []
                lp_eval = jax.jit(jax.vmap(base))
                for chain, key in enumerate(keys):
                    chain_path = folder / f"chain{chain:02d}.npz"
                    receipt_path = folder / f"chain{chain:02d}.json"
                    if chain_path.exists() and receipt_path.exists():
                        receipt = json.loads(receipt_path.read_text())
                        if hashlib.sha256(chain_path.read_bytes()).hexdigest() != receipt["sha256"]:
                            raise ValueError(f"Checkpoint hash differs: {chain_path}")
                        with np.load(chain_path, allow_pickle=False) as z:
                            payload = {k: z[k] for k in z.files}
                        print(f"Resume rep={rep}, floor={floor}, chain={chain}", flush=True)
                    else:
                        print(f"Run rep={rep}, floor={floor}, chain={chain}, start={labels[chain]}", flush=True)
                        init_u = target.to_unconstrained(jnp.asarray(initials[chain], dtype=jnp.float64))
                        if not np.isfinite(float(target.log_density(init_u))):
                            raise ValueError("Non-finite initial log density")
                        started = time.monotonic()
                        fit = sampling.run_nuts_adapted(
                            key, target.log_density, init_u, num_warmup=a.num_warmup,
                            num_samples=a.num_samples, initial_step_size=a.init_step,
                            target_acceptance_rate=a.target_accept, is_mass_matrix_diagonal=False)
                        u_draws = np.asarray(fit["samples"])
                        draws = np.asarray(target.to_physical(fit["samples"]))
                        if not np.isfinite(draws).all():
                            raise ValueError("Non-finite retained positions")
                        payload = {"samples_physical": draws, "samples_sampling": u_draws,
                                   "initial_physical": initials[chain], "initial_sampling": np.asarray(init_u),
                                   "step_size": np.asarray(fit["parameters"]["step_size"]),
                                   "inverse_mass_matrix": np.asarray(fit["parameters"]["inverse_mass_matrix"])}
                        for name in ["is_divergent", "acceptance_rate", "energy", "num_integration_steps", "num_trajectory_expansions"]:
                            if all(hasattr(info, name) for info in fit["infos"]):
                                payload[name] = np.asarray(jnp.stack([getattr(info, name) for info in fit["infos"]]))
                        # Keep evaluation batches small enough for a laptop.
                        payload["log_posterior_physical"] = np.concatenate([
                            np.asarray(lp_eval(jnp.asarray(draws[i:i+128])))
                            for i in range(0, len(draws), 128)])
                        atomic_npz(chain_path, **payload)
                        atomic_json(receipt_path, {"sha256": hashlib.sha256(chain_path.read_bytes()).hexdigest(),
                                                  "elapsed_seconds": time.monotonic() - started})
                    if payload["samples_physical"].shape != (a.num_samples, d + 2):
                        raise ValueError("Checkpoint sample shape differs")
                    chains.append(payload["samples_physical"])
                    sampling_chains.append(payload["samples_sampling"])
                    all_stats.append(payload)
                    eps = expit(-payload["samples_physical"][:, -1])
                    sigma = np.exp(payload["samples_physical"][:, d])
                    chain_summaries.append({"chain": chain, "initial_label": labels[chain],
                        "eps_mean": float(eps.mean()), "eps_ci95": np.quantile(eps, [.025, .975]).tolist(),
                        "sigma_mean": float(sigma.mean()), "sigma_ci95": np.quantile(sigma, [.025, .975]).tolist(),
                        "fraction_sigma_within_10pct_of_floor": float(np.mean(sigma <= 1.1 * floor)),
                        "divergences": int(payload["is_divergent"].sum()),
                        "acceptance_mean": float(payload["acceptance_rate"].mean()),
                        "step_size": float(payload["step_size"]),
                        "mean_log_posterior_physical": float(payload["log_posterior_physical"].mean())})
                draws3 = np.stack(chains)
                flat = draws3.reshape(-1, d + 2)
                diagnostic_draws = np.concatenate([draws3, np.exp(draws3[..., d:d+1]), expit(-draws3[..., -1:])], axis=-1)
                names = [f"beta{i}" for i in range(d)] + ["log_sigma", "rho", "sigma", "epsilon"]
                rhat = np.asarray(diagnostics.arviz_rhat(diagnostic_draws, names))
                bulk = np.asarray(diagnostics.arviz_ess(diagnostic_draws, names))
                tail = np.asarray(diagnostics.arviz_ess_tail(diagnostic_draws, names))
                finite = all(np.isfinite(v).all() for v in [rhat, bulk, tail])
                divergences = sum(s["divergences"] for s in chain_summaries)
                passed = bool(finite and rhat.max() <= 1.01 and bulk.min() >= 400 and tail.min() >= 400 and divergences == 0)
                by_param = {name: {"rhat": float(rhat[i]) if np.isfinite(rhat[i]) else None,
                                  "ess_bulk": float(bulk[i]) if np.isfinite(bulk[i]) else None,
                                  "ess_tail": float(tail[i]) if np.isfinite(tail[i]) else None}
                            for i, name in enumerate(names)}
                scaler = {k[len("scaler_"):]: v for k, v in arrays.items() if k.startswith("scaler_")}
                pred = utils.summarize_posterior_prediction(flat, np.asarray(arrays["X_test"], dtype=float), scaler)
                evaluation = utils.evaluate_regression_metrics(pred["y_pred"], arrays["y_test"], pred["beta_hat"],
                    arrays["beta_true"], pred["sigma_hat"], arrays["sigma_true"])
                evaluation["eps_post_mean"] = float(expit(-flat[:, -1]).mean())
                stats_payload = {name: np.stack([s[name] for s in all_stats]) for name in
                                 ["is_divergent", "acceptance_rate", "energy", "num_integration_steps", "num_trajectory_expansions"]
                                 if all(name in s for s in all_stats)}
                atomic_npz(folder / "samples.npz", samples_physical=draws3, samples_sampling=np.stack(sampling_chains),
                           parameter_names=np.asarray(names[:d+2]), **stats_payload)
                metrics = {"scope": "diagnostic_pilot", "rep": rep, "sigma_min": floor,
                    "diagnostics_pass": passed, "rhat_max": float(rhat.max()) if finite else None,
                    "ess_bulk_min": float(bulk.min()) if finite else None, "ess_tail_min": float(tail.min()) if finite else None,
                    "nuts_divergences": divergences, "by_parameter": by_param,
                    "chains": chain_summaries, "evaluation": evaluation,
                    "limitation": "Passing these diagnostics does not establish global exploration of all modes."}
                atomic_json(folder / "metrics.json", metrics)
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2 * len(names)), sharex=True)
                    for i, ax in enumerate(axes):
                        for c in range(4):
                            ax.plot(diagnostic_draws[c, :, i], linewidth=0.5, label=f"chain {c}")
                        ax.set_ylabel(names[i])
                    axes[0].legend(ncol=4)
                    axes[-1].set_xlabel("retained draw")
                    fig.tight_layout(); fig.savefig(folder / "trace.png", dpi=120); plt.close(fig)
                except Exception as exc:
                    atomic_json(folder / "trace_error.json", {"error": str(exc)})
                jobs.append({"rep": rep, "sigma_min": floor, "diagnostics_pass": passed,
                             "rhat_max": metrics["rhat_max"], "nuts_divergences": divergences})
                print(f"Finished rep={rep}, floor={floor}, diagnostics_pass={passed}", flush=True)
                if not passed:
                    failures.append({"rep": rep, "sigma_min": floor, "reason": "MCMC diagnostics failed"})
            except Exception as exc:
                error = {"rep": rep, "sigma_min": floor, "reason": str(exc)}
                atomic_json(folder / "failure.json", error)
                failures.append(error)
                print(f"Failed rep={rep}, floor={floor}: {exc}", flush=True)
    summary = {**plan(a), "runtime": runtime, "completed_fits": len(jobs), "jobs": jobs,
               "failures": failures, "all_requested_diagnostics_pass": not failures,
               "next_step": "Review chain traces and floor sensitivity before scheduling full experiments."}
    atomic_json(a.output_dir / "run_summary.json", summary)
    make_archive(a.output_dir, a.input_dir, a.rep_ids)
    print(f"Saved {a.output_dir / 'pilot_results.zip'}", flush=True)
    return 2 if failures else 0


def main():
    a = parse_args()
    if a.dry_run:
        print(json.dumps(plan(a), indent=2))
        return 0
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
