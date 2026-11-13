"""Run contract and paired contamination of the repository's benchmark splits."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "configs/realdata_audit.json"
DATASETS = ("cal_housing", "kin8nm", "pumadyn32fh")
VARIANTS = {
    "DPD": ("dpd", 1.0),
    "ExpSat": ("exponential_saturated", 0.5),
    "RatSat": ("rational_saturated", 0.25),
}
VERSION = 1


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_json(value):
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(v) for v in value]
    return value


@contextmanager
def atomic_file(path, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"encoding": "utf-8"} if "b" not in mode else {}
    with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, **options) as fh:
        temporary = Path(fh.name)
        try:
            yield fh
            fh.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path, "w") as fh:
        json.dump(_safe_json(value), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def write_npz(path, **arrays):
    with atomic_file(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)


def read_npz(path):
    with np.load(path, allow_pickle=False) as src:
        return dict(src)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}:{a.shape}".encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _overlay(base, overrides):
    for k, v in overrides.items():
        if k not in base:
            raise ValueError(f"Unknown configuration key: {k}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f"{k} must be an object")
            _overlay(base[k], v)
        else:
            base[k] = v
    return base


def config(path=None):
    c = read_json(DEFAULT)
    if path:
        _overlay(c, read_json(path))
    ds = c["datasets"]
    if not isinstance(ds, list) or not ds or len(set(ds)) != len(ds) or set(ds) - set(DATASETS):
        raise ValueError(f"datasets must be a nonempty subset of {DATASETS}")
    s, t = c["sampler"], c["costs"]
    ints = [c[k] for k in ("seed", "replicates", "train_size", "bootstrap_B", "kf_starts", "bootstrap_starts")]
    ints += [s[k] for k in ("chains", "warmup", "draws", "repair_draws", "ess_min", "max_fod_draws")]
    if any(type(x) is not int or x < 0 for x in ints):
        raise ValueError("Integer settings must be nonnegative integers")
    if c["seed"] + (c["replicates"]-1)*100_000 >= 2**32 or not 1 <= c["replicates"] <= 1000:
        raise ValueError("Use 1–1000 replicates and uint32 split seeds")
    if c["train_size"] and c["train_size"] < 2 + (32 if "pumadyn32fh" in ds else 8) + 1:
        raise ValueError("The training subsample must exceed the regression dimension by at least two")
    if c["bootstrap_B"] < 2 or min(c["kf_starts"], c["bootstrap_starts"]) < 2:
        raise ValueError("Use at least two bootstrap fits and two starts")
    if (s["chains"] < 2 or min(s["warmup"], s["draws"]) < 20
        or s["repair_draws"] and s["repair_draws"] <= s["draws"]
        or not 0 < s["target_accept"] < 1 or not s["initial_step"] > 0
        or s["rhat_max"] < 1 or s["ess_min"] < 1 or s["max_fod_draws"] < s["chains"]):
        raise ValueError("Invalid sampler settings")
    if not 0 < c["gamma"] or not 0 < c["injection_probability"] < 1:
        raise ValueError("gamma must be positive and injection_probability must lie in (0,1)")
    shifts = c["shifts"]
    if not isinstance(shifts, list) or not shifts or any(not v > 0 for v in shifts) or len(set(shifts)) != len(shifts):
        raise ValueError("shifts must be distinct positive multipliers of the reference residual SD")
    if type(c["include_no_injection"]) is not bool or c["purpose"] not in {"main", "pilot"}:
        raise ValueError("Invalid purpose or no-injection switch")
    if not 0 < t["review"] < t["shortage"] or not t["false_remove"] > 0 or not t["miss"] > 0:
        raise ValueError("Require 0 < review < shortage and positive false-removal and miss costs")
    for k in ("false_remove", "review"):
        grid = t[f"{k}_grid"]
        if not isinstance(grid, list) or any(not v > 0 for v in grid):
            raise ValueError(f"{k}_grid must contain positive costs")
    digest(c)  # Reject non-finite JSON settings before starting a run.
    return c


def jobs(c):
    shifts = ([0.0] if c["include_no_injection"] else []) + c["shifts"]
    return [
        dict(id=f"{d}__r{r:03d}__shift{str(float(s)).replace('.', 'p')}",
             dataset=d, replicate=r, shift=float(s), gamma=c["gamma"])
        for d in c["datasets"] for r in range(c["replicates"]) for s in shifts
    ]


def seed(c, dataset, rep, stream):
    state = np.random.SeedSequence([c["seed"], DATASETS.index(dataset), rep, stream]).generate_state(1)
    return int(state[0] % np.uint32(2**31 - 1))


def manifest(c):
    local = sorted(Path(__file__).parent.glob("*.py"))
    repo = [ROOT/p for p in (
        "experiments/exp1_regression/run_exp1_realdata_benchmark.py",
        "experiments/exp1_regression/utils_exp1_regression.py",
        "hb_experiments/native_run.py", "hbayes/paper_sampling.py",
        "hbayes/independence_mh.py", "hbayes/sampling.py", "hbayes/posteriors.py",
        "hbayes/experiment_temperature.py", "hbayes/bounded_scale.py", "hbayes/dpd.py",
    )]
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in local + repo}
    v = dict(version=VERSION, config=c, variants=VARIANTS, jobs=jobs(c), source_sha256=code)
    return {**v, "fingerprint": digest(v)}


def open_run(out, c):
    path = out / "manifest.json"
    v = manifest(c)
    with lock(out / ".manifest.lock"):
        if path.exists():
            old = read_json(path)
            if old.get("fingerprint") != v["fingerprint"]:
                raise ValueError("Code/configuration changed. Use a new --output-dir.")
            return old
        write_json(path, v)
    return v


def frozen_dataset(out, name):
    path = out / "datasets" / f"{name}.npz"
    return path.is_file() and path.with_suffix(".json").is_file()


def local_candidates(directory, name):
    return tuple(directory / f"{name}.{ext}" for ext in ("npz", "csv", "tsv"))


def preflight_inputs(out, names, local_dir, allow_download):
    """Check all selected local sources before initialising the sampler or run."""
    needed = [name for name in names if not frozen_dataset(out, name)]
    if not needed or allow_download:
        return
    help_text = ("Point --local-data-dir to the directory containing the benchmark files, "
                 "or replace it with --download-data for public sources.")
    if local_dir is None:
        raise FileNotFoundError(f"No frozen inputs for {', '.join(needed)}. {help_text}")
    if not local_dir.is_dir():
        hint = (" /path/to/benchmark_inputs is an example, not an installed dataset directory."
                if local_dir == Path("/path/to/benchmark_inputs") else "")
        raise FileNotFoundError(f"Input directory does not exist: {local_dir}.{hint} {help_text}")
    missing = [name for name in needed if not any(p.is_file() for p in local_candidates(local_dir, name))]
    if missing:
        expected = "; ".join(" or ".join(p.name for p in local_candidates(local_dir, name)) for name in missing)
        raise FileNotFoundError(f"Missing benchmark files in {local_dir}: {expected}. {help_text}")


def dataset(out, name, local_dir=None, allow_download=False):
    """Freeze the input rows and their order on first use in this run."""
    from hb_experiments.native_run import validate_dataset
    path = out / "datasets" / f"{name}.npz"
    receipt = path.with_suffix(".json")
    with lock(path.with_suffix(".lock")):
        if path.exists() and receipt.exists():
            z = read_npz(path)
            x, y = validate_dataset(name, z["X"], z["y"])
            meta = read_json(receipt)
            if [array_hash(x), array_hash(y)] != meta["hashes"]:
                raise ValueError(f"Frozen {name} input differs from its recorded hashes")
            return x, y
        preflight_inputs(out, [name], local_dir, allow_download)
        if local_dir is not None:
            origin = next((p for p in local_candidates(local_dir, name) if p.is_file()), None)
            if origin is None:
                raise FileNotFoundError(f"No local file for {name} in {local_dir}")
            if origin.suffix == ".npz":
                z = read_npz(origin)
                if not {"X", "y"} <= z.keys():
                    raise ValueError(f"{origin} must contain the arrays X and y")
                x, y = z["X"], z["y"]
            else:
                from experiments.exp1_regression.run_exp1_realdata_benchmark import load_benchmark_dataset
                x, y = load_benchmark_dataset(name, str(local_dir))
            source = {"source": str(origin)}
        else:
            from .data_sources import download
            x, y, source = download(name)
        x, y = validate_dataset(name, np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        write_npz(path, X=x, y=y)
        write_json(receipt, {**source, "hashes": [array_hash(x), array_hash(y)],
                             "shape": list(x.shape)})
        return x, y


def make_split(x, y, c, name, rep):
    from experiments.exp1_regression.utils_exp1_regression import (
        kanamori_standardize_regression_data, make_realdata_split_seed, resolve_train_test_sizes,
    )
    ntr, _ = resolve_train_test_sizes(len(y), train_ratio=0.8, test_ratio=0.2)
    split_seed = make_realdata_split_seed(c["seed"], rep)
    perm = np.random.default_rng(split_seed).permutation(len(y))
    count = c["train_size"] or ntr
    if count > ntr:
        raise ValueError(f"train_size={count} exceeds the {name} training fold ({ntr})")
    tr, te = perm[:count], perm[ntr:]
    X, y0, X_test, scaler = kanamori_standardize_regression_data(x[tr], y[tr], x[te])
    X, y0, X_test = map(np.asarray, (X, y0, X_test))
    coef, _, rank, _ = np.linalg.lstsq(X.astype(float), y0.astype(float), rcond=None)
    if rank != X.shape[1]:
        raise ValueError("The training design is rank deficient")
    reference_sd = float(np.sqrt(np.mean((y0 - X @ coef)**2)))
    if not np.isfinite(reference_sd) or reference_sd <= 1e-8:
        raise ValueError("Cannot define residual-scale perturbations on this split")
    rng = np.random.default_rng(seed(c, name, rep, 1))
    potential = rng.random(count) < c["injection_probability"]
    sign = rng.choice([-1, 1], size=count)
    return dict(X=X, y0=y0, X_test=X_test, y_test=y[te], train_indices=tr, test_indices=te,
                reference_sd=np.array(reference_sd), injection_mask=potential, injection_sign=sign,
                split_seed=np.array(split_seed), **{f"scaler_{k}": np.asarray(v) for k, v in scaler.items()})


def with_injection(split, shift):
    """Paired signs and mask; the clean response determines only the generator and scaler."""
    z = split["injection_mask"] & (shift > 0)
    perturbation = (shift * float(split["reference_sd"])) * split["injection_sign"] * z
    y = (split["y0"] + perturbation).astype(np.float32)
    return {**split, "y": y, "Z": z}
