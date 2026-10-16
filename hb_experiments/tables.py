"""Export numerical tables from this campaign's aggregate CSVs.

No expected paper values or frozen result schema is used. All displayed ties
are bold; oracle cleaning is excluded from practical-method comparisons.
"""
from pathlib import Path
from decimal import Decimal
import numpy as np
import pandas as pd
from .common import safe_output, sha, write

DIRECTIONS = dict(mu_bias_norm='min', rmse_test='min', mae_test='min',
                  nlpd_test_mc='min', auprc='max', removed_f1='max', f1='max')
METRICS = ['mu_bias_norm', 'rmse_test', 'nlpd_test_mc', 'auprc', 'removed_f1', 'f1',
           'eps_post_mean', 'eps_ci95_cover', 'removed_fraction']


def _escape(text):
    return str(text).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def render(frame, *, places=3):
    method = 'cleaning_method' if 'cleaning_method' in frame else 'method'
    if method not in frame:
        raise ValueError('Aggregate CSV has no method column')
    keys = [k for k in ['dataset', 'gamma', 'eps_true', 'scenario'] if k in frame]
    metrics = [m for m in METRICS if m+'_mean' in frame]
    if not metrics:
        raise ValueError('Aggregate CSV has no supported reported metrics')
    rows, cells = [], []
    grouper = frame.groupby(keys, dropna=False, sort=True) if keys else [((), frame)]
    for identity, group in grouper:
        if not isinstance(identity, tuple):
            identity = (identity,)
        best = {}
        for metric in metrics:
            values = []
            for _, row in group.iterrows():
                # Undefined ranking at zero contamination stays undefined.
                if str(row[method]) == 'OracleCleaning':
                    continue
                value = row[metric+'_mean']
                if pd.notna(value) and np.isfinite(float(value)):
                    values.append(Decimal(f'{value:.{places}f}'))
            if metric in DIRECTIONS and values:
                best[metric] = (min if DIRECTIONS[metric] == 'min' else max)(values)
        for _, row in group.iterrows():
            entries = [_escape(v) for v in identity]+[_escape(row[method])]
            for metric in metrics:
                value = row[metric+'_mean']
                sd = row.get(metric+'_std', np.nan)
                bold = False
                if pd.isna(value) or not np.isfinite(float(value)):
                    display = '--'
                else:
                    mean = f'{value:.{places}f}'
                    bold = (str(row[method]) != 'OracleCleaning' and metric in best
                            and Decimal(mean) == best[metric])
                    display = (r'\mathbf{'+mean+'}') if bold else mean
                    if pd.notna(sd) and np.isfinite(float(sd)):
                        display += rf'\;({sd:.{places}f})'
                    display = '$'+display+'$'
                entries.append(display)
                cells.append(dict(condition=dict(zip(keys, map(str, identity))), method=str(row[method]),
                                  metric=metric, mean=None if pd.isna(value) else float(value),
                                  sd=None if pd.isna(sd) else float(sd), bold=bold))
            rows.append(' & '.join(entries)+r' \\')
    header = ' & '.join(_escape(x) for x in [*keys, method, *metrics])+r' \\'
    text = '% Values are mean (sample SD) across all collected replicates.\n'
    text += '% Best displayed practical-method means and ties are bold; oracle is a reference.\n'
    text += r'\begin{tabular}{'+'l'*(len(keys)+1)+'r'*len(metrics)+'}\n'+r'\hline'+'\n'
    text += header+'\n'+r'\hline'+'\n'+'\n'.join(rows)+'\n'+r'\hline'+'\n'+r'\end{tabular}'+'\n'
    return text, cells


def build(summaries, output):
    summaries, output = Path(summaries), safe_output(output)
    paths = sorted([*summaries.rglob('aggregate.csv'), *summaries.rglob('audits_aggregate.csv')])
    if not paths:
        raise FileNotFoundError('No new-run aggregate CSVs. Run hbayes paper run --phase collect first.')
    output.mkdir(parents=True, exist_ok=True)
    sources, cells = {}, {}
    for path in paths:
        relative = path.relative_to(summaries)
        name = '_'.join(relative.with_suffix('').parts)+'.tex'
        text, values = render(pd.read_csv(path))
        (output/name).write_text(text)
        sources[str(relative)] = sha(path)
        cells[name] = values
    report = dict(status='passed', tables=len(paths), input_csv_sha256=sources,
                  numeric_cells=sum(map(len, cells.values())), source='current_campaign')
    write(output/'table_values.json', dict(summary=report, cells=cells))
    return report
