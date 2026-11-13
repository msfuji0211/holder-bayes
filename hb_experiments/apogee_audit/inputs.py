"""Load frozen APOGEE catalogue columns and optionally verify archived fits."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/apogee_audit"
SAMPLES = {"exploratory": 1000, "replication": 460}
VARIANTS = {
    "DPD": ("dpd", 1.0),
    "ExpSat": ("exponential_saturated", 0.5),
    "RatSat": ("rational_saturated", 0.25),
}
ARCHIVE_PREFIX = "holder_bayes_realdata_feasibility_20261103/apogee_sensitivity/data/"


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    provenance = json.loads((DATA / "source.json").read_text(encoding="utf-8"))
    samples = {}
    seen = set()
    for name, n in SAMPLES.items():
        path = DATA / f"{name}.csv"
        if file_hash(path) != provenance["samples"][name]["csv_sha256"]:
            raise ValueError(f"Frozen input differs from the original sample: {path}")
        frame = pd.read_csv(path, dtype={"apogee_id": str})
        if list(frame.columns) != provenance["columns"] or len(frame) != n:
            raise ValueError(f"Unexpected schema or row count: {path}")
        ids = frame["apogee_id"]
        if ids.isna().any() or not ids.is_unique or seen.intersection(ids):
            raise ValueError(f"Duplicate or missing APOGEE IDs: {path}")
        seen.update(ids)
        c, t, mh = (frame[col].to_numpy(float) for col in ("bp_rp", "fparam_teff", "fparam_m_h"))
        if not (np.isfinite(frame.iloc[:, 1:].to_numpy(float)).all() and
                ((c >= .5) & (c <= 2.5)).all() and ((t >= 3500) & (t <= 7000)).all() and
                not np.isin(mh, [-999, -9999]).any()):
            raise ValueError(f"Invalid colour, temperature or metallicity: {path}")
        # Match the archived quadratic design: square the raw colour, then scale
        # each non-intercept column separately using the entire frozen sample.
        x = StandardScaler().fit_transform(np.column_stack((c, c**2)))
        y = StandardScaler().fit_transform(t[:, None]).ravel()
        samples[name] = (ids.to_numpy(str), np.column_stack((np.ones(n), x)), y, mh)
    return samples, provenance


def archived_inputs(path, sample, X, y, provenance):
    """Read just the requested members; never unpack external paths onto disk."""
    info = provenance["samples"][sample]
    with ZipFile(path) as z:
        def read(member, expected):
            data = z.read(ARCHIVE_PREFIX + member)
            if sha256(data).hexdigest() != expected:
                raise ValueError(f"The archived member differs from the original: {member}")
            return data

        raw = read(f"{sample}.csv", info["source_csv_sha256"])
        if len(raw) == 0:
            raise ValueError("Empty archived input")
        key = f"{sample}_g0p5"
        archive = {}
        for label in ("posterior", "bootstrap"):
            data = read(f"{key}_{label}.npz", info[f"{label}_sha256"])
            with np.load(BytesIO(data), allow_pickle=False) as arrays:
                archive[label] = dict(arrays)
            np.testing.assert_allclose(archive[label]["X"], X, rtol=0, atol=5e-13)
            np.testing.assert_allclose(archive[label]["y"], y, rtol=0, atol=5e-13)
    return archive
