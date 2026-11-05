#!/usr/bin/env python3
"""Run from the source checkout; never changes the original experiment runner."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hb_experiments.additional.runner import main
if __name__=='__main__':
    raise SystemExit(main())
