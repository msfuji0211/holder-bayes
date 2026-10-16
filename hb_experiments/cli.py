"""Source-only paper experiment commands."""
import argparse
import json
import sys
from pathlib import Path
from .common import ROOT


def main(argv=None, *, prog=None):
    p = argparse.ArgumentParser(prog=prog, description='Generate and analyse Hölder–Bayes paper experiments.')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['check', 'doctor', 'plan']:
        sub.add_parser(name)
    q = sub.add_parser('run')
    q.add_argument('--phase', choices=['all', 'gamma_pilot', 'pilot', 'pilot_realdata', 'synthetic',
                                     'benchmark', 'cleaning', 'review', 'collect', 'plot'], default='all')
    q.add_argument('--output', type=Path, default=ROOT/'results/paper')
    q.add_argument('--dry-run', action='store_true')
    q.add_argument('--download-data', action='store_true', help='Download raw real datasets through the native loaders.')
    q.add_argument('--local-data-dir', type=Path, help='Directory containing dataset CSV/TSV files.')
    q = sub.add_parser('tables')
    q.add_argument('--output', type=Path, default=ROOT/'results/paper', help='Existing output of this campaign.')
    q = sub.add_parser('diagnose')
    q.add_argument('--output', type=Path, default=ROOT/'results/paper')
    q.add_argument('--dataset', default='cal_housing')
    q.add_argument('--gamma', type=float, default=.5)
    q.add_argument('--eps', type=float, default=.2)
    q.add_argument('--exclude-split', type=int, help='Optional post-hoc common-split exclusion; primary results keep all splits.')
    if prog and argv and argv[0] in sub.choices:
        sub.choices[argv[0]].prog = prog
    a = p.parse_args(argv)
    try:
        from . import inference
        if a.command in ['check', 'doctor', 'plan']:
            result = getattr(inference, a.command)()
        elif a.command == 'run':
            result = inference.run(a.output, a.phase, dry_run=a.dry_run,
                                   allow_download=a.download_data, local_data_dir=a.local_data_dir)
        elif a.command == 'diagnose':
            from .predictive import run
            result = run(a.output, dataset=a.dataset, gamma=a.gamma, eps=a.eps, exclude_split=a.exclude_split)
        else:
            from .tables import build
            from .common import safe_output
            output = safe_output(a.output)
            result = build(output/'summaries', output/'tables')
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, AssertionError) as exc:
        print(f'Hölder–Bayes task stopped: {exc}', file=sys.stderr)
        return 2
