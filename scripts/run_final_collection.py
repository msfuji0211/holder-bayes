#!/usr/bin/env python3
"""Collect saved Hölder–Bayes results; this script never starts MCMC."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hb_experiments.final_collection import main

if __name__ == '__main__':
    raise SystemExit(main())
