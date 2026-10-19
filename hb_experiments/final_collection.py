"""Collect saved paper runs and redraw figures without starting a sampler.

Accept native result trees or a public campaign containing ``runs/``. Optional
synthetic and real-data roots select rebuilt metrics before the base results.
No historical manifest, reference CSV, or previously published result is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from .common import ROOT, read, safe_output, sha, write
from .grid import diagnostic_issues, jobs


def native_root(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Result directory is missing: {path}')
    return path/'runs' if (path/'runs').is_dir() else path


def grid_settings(output):
    config = read(ROOT/'configs/paper.json')
    return SimpleNamespace(output_dir=output, gammas=config['gammas'],
        datasets=config['datasets'], replicates=config['synthetic_replicates'],
        real_replicates=config['realdata_replicates'])


def check_identity(payload, job, path):
    expected = {'eps_true': job['eps']}
    actual = {**payload}
    actual['replicate'] = payload.get('replicate', payload.get('rep'))
    expected['replicate'] = job['rep']
    gamma_candidates = [payload.get('gamma_hyperparam'), payload.get('gamma'),
                        payload.get('run_config', {}).get('gamma')]
    # Synthetic files use "gamma" for the GammaDiv method's metric block.
    actual['gamma'] = next((v for v in gamma_candidates if isinstance(v, (int, float))), None)
    expected['gamma'] = job['gamma']
    if job['dataset'] is not None:
        expected['dataset'] = job['dataset']
    for name, value in expected.items():
        if actual.get(name) != value:
            raise ValueError(f'Run identity differs at {path}: {name}={actual.get(name)!r}, expected {value!r}')


def selected_inputs(a, roots):
    config = grid_settings(a.output_root)
    rows = []
    counts = Counter()
    for phase in ['synthetic', 'benchmark', 'cleaning']:
        for job in jobs(config, phase):
            relative = Path(job['relative_dir'])/'metrics.json'
            overlay = roots['synthetic'] if phase == 'synthetic' else roots['real']
            candidates = [p/relative for p in [overlay, roots['base']] if p is not None]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError('Missing planned run: '+str(relative)+
                    '\nSearched: '+', '.join(map(str, candidates)))
            payload = read(path)
            check_identity(payload, job, path)
            issues = diagnostic_issues(payload, job['kind'])
            if issues:
                raise ValueError(f'Unresolved diagnostics at {path}: {issues}')
            rows.append(dict(job=job, source_path=str(path), metrics_sha256=sha(path)))
            counts[job['kind']] += 1
    return rows, dict(counts)


def representative_inputs(rows, roots, *, required):
    from .figures import DENSITY, REGRESSION
    needed = {DENSITY, REGRESSION} if required else set()
    result = []
    # A sample file is accepted from another root only if its neighbouring
    # metrics are byte-identical to the selected metrics. Never pair rebuilt
    # metrics with an unrelated, older posterior silently.
    for row in rows:
        job = row['job']
        if job['kind'] not in ['density', 'regression'] or job['eps'] != .2 or job['rep'] != 0:
            continue
        relative = Path(job['relative_dir'])
        candidates = [Path(row['source_path']).parent]
        candidates += [r/relative for r in [roots['synthetic'], roots['base']] if r is not None]
        folder = next((p for p in candidates if (p/'samples.npz').is_file()
            and (p/'metrics.json').is_file() and sha(p/'metrics.json') == row['metrics_sha256']), None)
        if folder is None:
            if str(relative) in needed:
                raise FileNotFoundError(f'Matching representative samples are missing: {relative}/samples.npz. '
                    'Use the original result tree containing observations and matching posterior draws, '
                    'or pass --no-plots to collect CSVs and tables only.')
            continue
        result.append(dict(relative_dir=str(relative), source_path=str(folder/'samples.npz'),
                           samples_sha256=sha(folder/'samples.npz')))
    return result


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'.collection.lock'
    try:
        handle = lock.open('x')
    except FileExistsError as exc:
        raise RuntimeError(f'Another collection may be active: {lock}') from exc
    try:
        handle.close()
        yield
    finally:
        lock.unlink(missing_ok=True)


def copy_unchanged(source, destination, digest):
    if sha(source) != digest:
        raise ValueError('Input changed during collection: '+str(source))
    if destination.exists():
        if sha(destination) != digest:
            raise FileExistsError('A different output already exists; choose a new --output-root: '+str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.tmp')
    try:
        shutil.copyfile(source, temporary)
        if sha(temporary) != digest:
            raise ValueError('Input changed while copying: '+str(source))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def collect(a):
    output = safe_output(a.output_root)
    roots = dict(base=native_root(a.results_root),
                 synthetic=native_root(a.synthetic_root) if a.synthetic_root else None,
                 real=native_root(a.real_root) if a.real_root else None)
    for root in [r for r in roots.values() if r is not None]:
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('Use an output directory separate from every input result tree')
        # Protect a public campaign as a whole, not only its runs/ subtree.
        if root.name == 'runs' and output.is_relative_to(root.parent):
            raise ValueError('Use an output directory outside the input public campaign')
    a.output_root = output
    rows, counts = selected_inputs(a, roots)
    samples = representative_inputs(rows, roots, required=not a.no_plots)
    signature = dict(version='saved_paper_collection_v1', roots={k:str(v) if v else None for k,v in roots.items()},
                     sources=rows, representatives=samples)
    report = dict(output=str(output), planned_runs=counts, total_runs=len(rows),
                  all_saved_diagnostics_pass=True, diagnostics_recomputed=False,
                  inference_run=False, original_results_modified=False,
                  representative_sample_files=len(samples), plots_requested=not a.no_plots)
    if a.dry_run:
        return dict(**report, dry_run=True)
    with output_lock(output):
        inputs = output/'collection_inputs.json'
        if inputs.exists() and read(inputs) != signature:
            raise ValueError('Collection inputs changed; choose a new --output-root')
        if not inputs.exists() and any(p.name != '.collection.lock' for p in output.iterdir()):
            raise FileExistsError('Output is not an empty collection directory; choose a new --output-root')
        write(inputs, signature)
        write(output/'final_collection.json', dict(**report, execution_complete=False, plots_complete=False))
        for row in rows:
            copy_unchanged(Path(row['source_path']), output/row['job']['relative_dir']/'metrics.json', row['metrics_sha256'])
        from .collection import collect as aggregate
        files = aggregate(grid_settings(output))
        import numpy as np
        import pandas as pd
        gamma = read(ROOT/'configs/paper.json')['realdata_gamma']
        gamma_tag = str(float(gamma)).replace('.', 'p')
        for family in ['realdata_benchmark', 'realdata_cleaning']:
            frame = pd.read_csv(output/family/'summary_agg.csv')
            destination = output/f'{family}_main_gamma{gamma_tag}.csv'
            frame[np.isclose(frame.gamma, gamma) & np.isclose(frame.eps_true, .2)].to_csv(destination, index=False)
            files[str(destination.relative_to(output))] = destination
        write(output/'workflow/collect.json', dict(complete=True, inference_run=False,
              diagnostic_source='saved_per_run_metrics', csv_sha256={name:sha(p) for name,p in files.items()}))
        from experiments.compact_cache import load as load_arrays
        for row in samples:
            source = Path(row['source_path'])
            if sha(source) != row['samples_sha256']:
                raise ValueError('Representative samples changed: '+str(source))
            destination = output/row['relative_dir']/'samples.npz'
            arrays = load_arrays(source)
            if sha(source) != row['samples_sha256']:
                raise ValueError('Representative samples changed while reading: '+str(source))
            temporary = destination.with_suffix('.tmp.npz')
            try:
                np.savez_compressed(temporary, **arrays)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        from .artifacts import _summaries
        public_csvs = _summaries(output, output/'summaries')
        from .tables import build
        tables = build(output/'summaries', output/'tables')
        plot_report = None
        if not a.no_plots:
            from .fresh_figures import finish
            plot_report = finish(output)
            target = output/'paper_exp1_suite_section6_revised'
            for native, public in [('main_text', 'main'), ('appendix', 'appendix')]:
                shutil.copytree(target/native, output/'figures'/public, dirs_exist_ok=True)
        report.update(execution_complete=True, aggregate_csvs=len(files), public_csvs=len(public_csvs),
                      tables=tables, plot_report=plot_report, plots_complete=plot_report is not None)
        write(output/'final_collection.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True,
                        help='Base saved results, or a public campaign with a runs/ directory.')
    parser.add_argument('--synthetic-root', type=Path,
                        help='Optional rebuilt synthetic metrics; existing files take priority over the base.')
    parser.add_argument('--real-root', type=Path,
                        help='Optional rebuilt benchmark/cleaning metrics; existing files take priority.')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/collection')
    parser.add_argument('--no-plots', action='store_true', help='Collect metrics, CSVs and tables without requiring representative samples.')
    parser.add_argument('--dry-run', action='store_true', help='Check saved metrics and locate samples without writing output or running MCMC.')
    a = parser.parse_args(argv)
    try:
        report = collect(a)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f'Final collection stopped: {exc}\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
