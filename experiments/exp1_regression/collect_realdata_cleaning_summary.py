"""Rebuild cleaning and audit CSVs across all per-run metrics.json files.

No MCMC imports are required. Separate dataset jobs may share output_dir;
the collector reads every saved run rather than the most recent job's CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_rows(output_dir: str):
    cleaning_rows, audit_rows, seen = [], [], set()
    for path in sorted(Path(output_dir).glob('dataset_*/gamma_*/eps_*/muout_*/rep_*/metrics.json')):
        payload = json.loads(path.read_text())
        if not isinstance(payload.get('cleaning'), dict):
            raise ValueError(f'Expected cleaning payload: {path}')
        key = tuple(payload[k] for k in ('dataset', 'gamma', 'eps_true', 'mu_out', 'replicate'))
        if key in seen:
            raise ValueError(f'Duplicate experiment identity: {key}')
        seen.add(key)
        names = ('dataset', 'gamma', 'eps_true', 'mu_out', 'replicate', 'n_train', 'n_test',
                 'preprocessing', 'split_mode', 'k_oracle', 'has_synthetic_labels',
                 'k_est_cleaning_reference', 'proposal_method_primary', 'eta_mode')
        base = {k: payload.get(k) for k in names}
        base['temperature_version'] = payload.get('temperature_version', 'legacy_unversioned')
        for name, block in payload['cleaning'].items():
            scalar = {k: v for k, v in block.items() if not isinstance(v, (dict, list))}
            cleaning_rows.append({**base, 'cleaning_method': name, **scalar})
        for name, block in payload.get('audits', {}).items():
            scalar = {k: v for k, v in block.items() if not isinstance(v, (dict, list))}
            audit_rows.append({**base, 'method': name, **scalar})
    return cleaning_rows, audit_rows, len(seen)


def write_summary(rows, root, filename, method_key):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df['temperature_version'].nunique() != 1:
        raise ValueError('Mixed temperature versions in output_dir')
    keys = ['dataset', 'gamma', 'eps_true', 'mu_out', method_key]
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in keys and c != 'replicate']
    df.to_csv(root / filename, index=False)
    if numeric:
        grouped = df.groupby(keys, dropna=False)[numeric].agg(['mean', 'std', 'count']).reset_index()
        grouped.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in grouped.columns]
        agg_name = 'summary_agg.csv' if method_key == 'cleaning_method' else 'summary_audits_agg.csv'
        grouped.to_csv(root / agg_name, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--expected_runs', type=int)
    args = p.parse_args()
    rows, audits, runs = collect_rows(args.output_dir)
    if not rows:
        raise SystemExit('No saved cleaning results found')
    if args.expected_runs is not None and runs != args.expected_runs:
        raise SystemExit(f'Incomplete results: {runs}/{args.expected_runs} run identities')
    root = Path(args.output_dir)
    write_summary(rows, root, 'summary_runs.csv', 'cleaning_method')
    write_summary(audits, root, 'summary_audits.csv', 'method')
    print(f'Collected {runs} runs, {len(rows)} cleaning rows, {len(audits)} audit rows')


if __name__ == '__main__':
    main()
