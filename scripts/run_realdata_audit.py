#!/usr/bin/env python3
"""Launch the added audit experiment from the holder-bayes checkout."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

from hb_experiments.realdata_audit.runner import cli

if __name__ == "__main__":
    raise SystemExit(cli())
