"""Command-line driver; leaves the existing benchmark scripts untouched."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import traceback
from zipfile import BadZipFile

from hbayes.paper_sampling import atomic_json
from .inputs import ROOT, SAMPLES, VARIANTS, file_hash, load_inputs

DEFAULT = ROOT/"configs/apogee_audit.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config(path=None):
    c = read_json(DEFAULT)
    if path is not None:
        override = read_json(path)
        if not isinstance(override, dict) or override.keys()-c.keys():
            raise ValueError("Configuration must contain only recognised keys")
        c.update(override)
    integers = ("seed", "chains", "warmup", "draws", "repair_warmup", "repair_draws")
    if any(type(c[k]) is not int for k in integers):
        raise ValueError("Seed, chains, warmup and draws must be integers")
    if (not 0 <= c["seed"] < 2**32-200_000 or c["chains"] < 2 or
            min(c["warmup"], c["draws"]) < 20 or
            c["repair_warmup"] < c["warmup"] or c["repair_draws"] <= c["draws"]):
        raise ValueError("Invalid seed, chain length or diagnostic retry length")
    for k in ("gamma", "initial_step", "target_accept"):
        if type(c[k]) not in (float, int) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f"{k} must be finite and positive")
    if c["target_accept"] >= 1:
        raise ValueError("target_accept must be below 1")
    return c


@contextmanager
def lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/".apogee_audit.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def contract(c, provenance):
    local = sorted(Path(__file__).parent.glob("*.py"))
    repo = [ROOT/"hbayes"/name for name in (
        "posteriors.py", "bounded_scale.py", "paper_sampling.py", "experiment_temperature.py",
    )]
    payload = dict(version=1, config=c, variants={k:list(v) for k,v in VARIANTS.items()},
                   samples=provenance, source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in local+repo})
    return {**payload, "fingerprint": hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()}


def run(args, out, manifest, samples, provenance):
    from .report import collect
    from .inference import run_fit

    path = out/"manifest.json"
    if path.is_file() and read_json(path)["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Configuration or source changed. Use a new --output-dir.")
    if not path.exists():
        atomic_json(path, manifest)
    import jax

    devices = jax.devices()
    if args.require_gpu and not any(x.platform in {"gpu", "cuda", "rocm"} for x in devices):
        raise RuntimeError("No GPU is visible to JAX")
    atomic_json(out/"runtime.json", dict(
        python=sys.version.split()[0], platform=platform.platform(), devices=[str(x) for x in devices],
        packages={k: metadata.version(k) for k in ("numpy","scipy","pandas","scikit-learn","jax","jaxlib","blackjax","arviz")},
    ))
    c = manifest["config"]
    failure = False
    for i, (name, (_, X, y, _)) in enumerate(samples.items()):
        if args.sample != "all" and name != args.sample:
            continue
        for method in VARIANTS:
            folder = out/"fits"/name/method
            print(f"{name} ({len(y)} stars), {method}", flush=True)
            try:
                result = run_fit(X, y, method, c, c["seed"]+1000*i, folder)
                (folder/"failure.json").unlink(missing_ok=True)
                print(f"    Diagnostics: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
                failure |= not result["passed"]
            except Exception as exc:
                failure = True
                atomic_json(folder/"failure.json", dict(message=str(exc), traceback=traceback.format_exc()))
                print(f"    Error: {exc} (see {folder/'failure.json'})", file=sys.stderr, flush=True)
    summary, _ = collect(out, manifest, samples, provenance, read_json, reference=args.reference_zip)
    print(summary)
    return int(failure)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="Check the inputs and print the design without fitting")
    p.add_argument("--config", type=Path)
    for cmd in ("run", "summarize"):
        p = sub.add_parser(cmd)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--reference-zip", type=Path, help="Optional original feasibility ZIP, for archived comparisons")
        if cmd == "run":
            p.add_argument("--config", type=Path)
            p.add_argument("--sample", choices=["all", *SAMPLES], default="all")
            p.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        samples, provenance = load_inputs()
        if args.cmd == "plan":
            c = config(args.config)
            print(json.dumps(dict(config=c, samples=SAMPLES, variants=VARIANTS, planned_fits=6,
                                  artificial_injection=False), indent=2))
            return 0
        out = args.output_dir.expanduser().resolve()
        path = out/"manifest.json"
        saved = read_json(path) if path.is_file() else None
        if args.cmd == "summarize" and saved is None:
            raise FileNotFoundError(f"No APOGEE run found in {out}; run the experiment first")
        config_path = getattr(args, "config", None)
        c = saved["config"] if saved is not None and config_path is None else config(config_path)
        if args.reference_zip is not None:
            args.reference_zip = args.reference_zip.expanduser().resolve()
            if not args.reference_zip.is_file():
                raise FileNotFoundError(f"Reference ZIP does not exist: {args.reference_zip}")
            if c["gamma"] != .5:
                raise ValueError("Archived K-F and DPD comparisons are available only at gamma=0.5")
            from .inputs import archived_inputs
            for name, (_, X, y, _) in samples.items():
                archived_inputs(args.reference_zip, name, X, y, provenance)
        with lock(out):
            if args.cmd == "run":
                return run(args, out, contract(c, provenance), samples, provenance)
            from .report import collect
            if saved["samples"] != provenance:
                raise ValueError("The saved run used different inputs")
            result, _ = collect(out, saved, samples, provenance, read_json, reference=args.reference_zip)
            print(result)
            return 0
    except (OSError, RuntimeError, ValueError, KeyError, BadZipFile, AssertionError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
