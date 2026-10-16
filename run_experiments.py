"""Source-checkout entry point; equivalent to `hbayes paper ...`."""
import sys
from hbayes.cli import main

raise SystemExit(main(["paper",*sys.argv[1:]]))
