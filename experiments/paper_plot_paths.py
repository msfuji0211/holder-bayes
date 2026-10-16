"""Directory parsing shared by the supplied paper figure suite."""
from pathlib import Path
import math


def discover_gamma_dirs(outputs_root, prefix):
    results = []
    for path in Path(outputs_root).glob(prefix+"*"):
        if not path.is_dir():
            continue
        try:
            gamma = float(path.name[len(prefix):].replace("p", "."))
        except ValueError:
            continue
        if math.isfinite(gamma):
            results.append((gamma, path))
    return sorted(results, key=lambda row: row[0])


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v) for v in str(value).replace(",", " ").split()]


def select_gammas(requested, available, fallback=None):
    values = requested or (fallback if fallback is not None else available)
    return [g for g in values if any(math.isclose(g, a, abs_tol=1e-12, rel_tol=0.) for a in available)]
