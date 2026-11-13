#!/usr/bin/env python3
"""Run the APOGEE audit sensitivity add-on from a holder-bayes checkout."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from hb_experiments.apogee_audit.runner import cli

if __name__ == "__main__":
    raise SystemExit(cli())
