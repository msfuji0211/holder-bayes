"""Plan, resume, and summarise the benchmark audit campaign."""
import argparse
import importlib.metadata
import itertools
from pathlib import Path
import sys
import traceback

from . import design as d


def fitting_runtime():
    import jax
    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "jax", "jaxlib", "blackjax", "arviz"):
        packages[name] = importlib.metadata.version(name)
    return dict(python=sys.version.split()[0], packages=packages,
                jax_devices=[str(device) for device in jax.devices()])


def run_one(out, manifest, job, split, runtime):
    from . import decision, inference

    c = manifest["config"]
    folder = out/"jobs"/job["id"]
    result_path = folder/"result.json"
    with d.lock(folder/".job.lock"):
        if result_path.exists():
            v = d.read_json(result_path)
            if v.get("fingerprint") != manifest["fingerprint"] or v.get("job") != job:
                raise ValueError("Existing job result has the wrong identity")
            print(f"Skip {job['id']} (already evaluated)", flush=True)
            return
        data = d.with_injection(split, job["shift"])
        X, y, z = (data[key] for key in ("X", "y", "Z"))
        hash_path = folder/"input.json"
        identity = {key: d.array_hash(data[key]) for key in
                    ("X", "y", "Z", "train_indices", "test_indices", "X_test", "y_test")}
        if hash_path.exists() and d.read_json(hash_path) != identity:
            raise ValueError("The job's input arrays changed; use a new output directory")
        if not (folder/"data.npz").exists():
            d.write_npz(folder/"data.npz", **data)
        d.write_json(hash_path, identity)
        print(f"{job['id']}: n_train={len(y)}, p={X.shape[1]-1}", flush=True)

        dataset, rep = job["dataset"], job["replicate"]
        fit_point = inference.point(X, y, c, d.seed(c, dataset, rep, 2), folder/"kf.json")
        fit_boot = inference.bootstrap(X, y, c, fit_point["theta"],
                                       d.seed(c, dataset, rep, 3), folder/"bootstrap")
        positions, posterior_diagnostics = {}, {}
        for method in d.VARIANTS:
            positions[method], posterior_diagnostics[method] = inference.posterior(
                X, y, method, c, d.seed(c, dataset, rep, 4), folder/"posterior"/method)
        fit, triage = decision.score_models(X, y, fit_point, fit_boot, positions, c["sampler"])
        rows = decision.evaluate(data, fit, triage, c["costs"])

        b_valid = sum(v["valid"] for v in fit_boot[2])
        status = dict(n_train=len(y), n_test=len(data["y_test"]), p=X.shape[1]-1,
                      injected=int(z.sum()), injected_rate=float(z.mean()),
                      reference_residual_sd=float(data["reference_sd"]),
                      kf_valid=fit_point["valid"], kf_at_scale_floor=fit_point["scale_floor"],
                      kf_projected_gradient=fit_point["projected_gradient"],
                      bootstrap_valid=b_valid == c["bootstrap_B"], bootstrap_passed=b_valid,
                      bootstrap_requested=c["bootstrap_B"],
                      bootstrap_floor_fraction=sum(v.get("scale_floor", False) for v in fit_boot[2])/c["bootstrap_B"])
        for method, info in posterior_diagnostics.items():
            status.update({f"{method}_{key}": info.get(value) for key, value in {
                "valid": "audit_diagnostics_pass", "rhat": "rhat_max", "ess_bulk": "ess_bulk_min",
                "ess_tail": "ess_tail_min", "attempt": "attempt", "sigma_near_floor_fraction": "sigma_near_floor_fraction",
            }.items()})
        status["paired_valid"] = all(status[k] for k in
                                     ("kf_valid", "bootstrap_valid", *(f"{v}_valid" for v in d.VARIANTS)))
        d.write_json(result_path, dict(fingerprint=manifest["fingerprint"], job=job, status=status,
                                      runtime=runtime, **rows))
        (folder/"failure.json").unlink(missing_ok=True)
        print(f"  done; common diagnostics: {'PASS' if status['paired_valid'] else 'FAIL'}", flush=True)


def plan(config, filter_jobs):
    n = len(filter_jobs)
    return dict(config=config, datasets=sorted({x["dataset"] for x in filter_jobs}),
                planned_datasets_and_scenarios=n, holder_posteriors=3*n, kf_point_fits=n,
                bootstrap_fits=config["bootstrap_B"]*n, jobs=filter_jobs,
                note="Counts exclude diagnostic repair attempts.")


def select_jobs(config, args):
    selections = d.jobs(config)
    if args.dataset:
        names = [s.replace("-", "_") for s in args.dataset]
        names = ["pumadyn32fh" if n == "pumadyn_32fh" else n for n in names]
        if set(names) - set(config["datasets"]):
            raise ValueError("Requested dataset is not in the configuration")
        selections = [j for j in selections if j["dataset"] in names]
    if args.replicate:
        if not all(0 <= r < config["replicates"] for r in args.replicate):
            raise ValueError("Replicate number is outside the configuration")
        selections = [j for j in selections if j["replicate"] in args.replicate]
    if args.shift is not None:
        selections = [j for j in selections if j["shift"] in args.shift]
    if not selections:
        raise ValueError("No jobs match the filters")
    return selections


def cli(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("plan", "run"):
        p = sub.add_parser(cmd)
        p.add_argument("--config", type=Path)
        p.add_argument("--dataset", action="append", help="Repeat to select more datasets")
        p.add_argument("--replicate", type=int, action="append", help="Zero-based; repeat to select more splits")
        p.add_argument("--shift", type=float, action="append", help="Repeat to select more shifts (0 is no injection)")
        if cmd == "run":
            p.add_argument("--output-dir", type=Path, required=True)
            source = p.add_mutually_exclusive_group()
            source.add_argument("--local-data-dir", type=Path,
                                help="Existing directory containing the benchmark NPZ, CSV or TSV files")
            source.add_argument("--download-data", action="store_true",
                                help="Fetch missing inputs from the documented public sources")
            p.add_argument("--require-gpu", action="store_true")
            p.add_argument("--finalize", action="store_true", help="Summarise after the selected jobs finish")
    p = sub.add_parser("summarize")
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "summarize":
        from .report import collect
        print(collect(args.output_dir.resolve()))
        return 0
    if args.cmd == "plan":
        c = d.config(args.config)
        print(json.dumps(plan(c, select_jobs(c, args)), indent=2))
        return 0

    out = args.output_dir.resolve()
    existing = out/"manifest.json"
    c = d.read_json(existing)["config"] if args.config is None and existing.exists() else d.config(args.config)
    jobs = select_jobs(c, args)
    names = list(dict.fromkeys(j["dataset"] for j in jobs))
    local_dir = args.local_data_dir.expanduser() if args.local_data_dir is not None else None
    try:
        d.preflight_inputs(out, names, local_dir, args.download_data)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    state = fitting_runtime()
    if args.require_gpu:
        import jax
        if not any(x.platform in {"gpu", "cuda", "rocm"} for x in jax.devices()):
            raise RuntimeError("No GPU in jax.devices(); check your JAX installation and device visibility")
    print(f"JAX devices: {state['jax_devices']}; {len(jobs)} selected jobs", flush=True)
    try:
        manifest = d.open_run(out, c)
    except ValueError as exc:
        parser.error(str(exc))
    inputs = {}
    for name in names:
        print(f"Checking benchmark input: {name}", flush=True)
        try:
            inputs[name] = d.dataset(out, name, local_dir, args.download_data)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            hint = ("Check network access or use the benchmark cache with --local-data-dir."
                    if args.download_data else "Check the benchmark file name, contents and row order.")
            parser.exit(1, f"{parser.prog}: could not load {name}: {exc}. {hint}\n")
    failed = 0
    for name, group in itertools.groupby(jobs, key=lambda j: j["dataset"]):
        x, y = inputs[name]
        for rep, same in itertools.groupby(group, key=lambda j: j["replicate"]):
            split = None
            for job in same:
                try:
                    if split is None:
                        split = d.make_split(x, y, c, name, rep)
                    run_one(out, manifest, job, split, state)
                except Exception as exc:
                    failed += 1
                    p = out/"jobs"/job["id"]/"failure.json"
                    d.write_json(p, dict(job=job, type=type(exc).__name__, message=str(exc),
                                         traceback=traceback.format_exc()))
                    print(f"{job['id']}: {type(exc).__name__}: {exc} (see {p})", file=sys.stderr, flush=True)
    if args.finalize:
        from .report import collect
        print(collect(out))
    return int(bool(failed))
