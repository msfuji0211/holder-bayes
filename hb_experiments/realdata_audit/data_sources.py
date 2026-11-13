"""Explicit, checksum-verified public sources for --download-data."""
import gzip
import hashlib
from io import BytesIO
import tarfile
from urllib.request import urlopen

import numpy as np

DELVE = "https://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles"
SOURCES = {
    "cal_housing": (
        "https://ndownloader.figshare.com/files/5976036",
        "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
        "CaliforniaHousing/cal_housing.data",
    ),
    "kin8nm": (
        f"{DELVE}/kin-family/kin-8nm.tar.gz",
        "2b77a78a36ca2628e39165acdb2b043d475d86577f396f60a9aa50c863b86e01",
        "kin-8nm/Dataset.data.gz",
    ),
    "pumadyn32fh": (
        f"{DELVE}/pumadyn-family/pumadyn-32fh.tar.gz",
        "f4c315f27953ca02c26a4fcfdb7e537811791aa0bb22b7eae8b1efb16dc1c7d9",
        "pumadyn-32fh/Dataset.data.gz",
    ),
}


def download(name):
    url, expected, member = SOURCES[name]
    with urlopen(url, timeout=25) as response:
        blob = response.read(8 * 1024 * 1024 + 1)
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f"Archive checksum mismatch for {name} ({url})")
    with tarfile.open(fileobj=BytesIO(blob), mode="r:gz") as archive:
        # Read only the named data member, never extract archive paths to disk.
        with archive.extractfile(member) as f:
            if name == "cal_housing":
                raw = np.loadtxt(f, delimiter=",")
            else:
                with gzip.GzipFile(fileobj=f) as records:
                    raw = np.loadtxt(records)
    if name == "cal_housing":
        # Same column order, household averages and units as
        # sklearn.datasets.fetch_california_housing for this archive.
        x = raw[:, [7, 2, 3, 4, 5, 6, 1, 0]].copy()
        x[:, 2] /= x[:, 5]
        x[:, 3] /= x[:, 5]
        x[:, 5] = x[:, 4] / x[:, 5]
        y = raw[:, 8] / 100000.0
    else:
        x, y = raw[:, :-1], raw[:, -1]
    return x, y, {"source": url, "archive_sha256": expected}
