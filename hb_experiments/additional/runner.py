"""CLI for the two additive experiments; no stored paper results are required."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from . import common as C


def write_csv(path, rows):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


@contextmanager
def job_lock(path, *, blocking=False):
    """OS releases the lock after interruption; lock files carry no state."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def contract(cfg):
    return dict(version=C.VERSION, config=cfg, sources=C.source_hashes(), runtime=C.environment())


def initialize(root, cfg):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    value = contract(cfg)
    with job_lock(root / '.manifest.lock',blocking=True) as acquired:
        if not acquired:
            raise RuntimeError('Another process is creating the manifest. Retry this command.')
        path = root / 'manifest.json'
        if path.exists() and C.read(path) != value:
            raise ValueError('Configuration, source or dependency versions changed. Use a new --output-dir.')
        C.dump(path, value)
    return C.digest(value)


def load_output_config(root, config=None):
    root = Path(root)
    if (root / 'manifest.json').exists():
        saved = C.read(root / 'manifest.json')['config']
        if config and saved != C.load_config(config):
            raise ValueError('Given config differs from the output manifest')
        return saved
    return C.load_config(config)


def verify_complete(folder, fingerprint):
    path = folder / 'result.json'
    if not path.exists():
        return False
    receipt = C.read(path)
    if receipt['fingerprint'] != fingerprint:
        raise ValueError(f'Result contract changed: {folder}')
    for name, expected in receipt['artifacts'].items():
        artifact = folder / name
        if not artifact.is_file() or C.file_hash(artifact) != expected:
            raise ValueError(f'Incomplete or altered result artifact: {artifact}')
    return True


def run_job(job, cfg, root, fingerprint, *, local_data_dir=None, download_data=False, paper_cache=None):
    from .datasets import prepare, observed_hash
    from .frequentist import fit as minimize, Estimate
    from .posterior import fit as sample
    from .bootstrap import run as bootstrap
    from .auditing import score_fit
    from . import evaluation
    folder = Path(root) / job['study'] / 'jobs' / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with job_lock(folder / '.job.lock') as acquired:
        if not acquired:
            print(f'BUSY {job["id"]}', flush=True)
            return 'busy'
        if verify_complete(folder, fingerprint):
            print(f'REUSE {job["id"]}', flush=True)
            return 'reuse'
        C.dump(folder / 'state.json', dict(job=job, status='running', fingerprint=fingerprint))
        try:
            data_path = folder / 'data.npz'
            if data_path.exists():
                with np.load(data_path, allow_pickle=False) as z:
                    data = dict(z)
            else:
                data = prepare(job, cfg, local_data_dir=local_data_dir, download_data=download_data,
                               dataset_cache=Path(root) / 'datasets')
                C.save_npz(data_path, **data)
            observed = observed_hash(data)
            input_path = folder / 'inputs.json'
            inputs = dict(job=job, fingerprint=fingerprint, observed=observed,
                          data_sha256=C.file_hash(data_path))
            if input_path.exists() and C.read(input_path) != inputs:
                raise ValueError('Saved input arrays were changed')
            C.dump(input_path, inputs)
            X = data['X']
            model = 'density' if job['family'] == 'density' else 'regression'
            y = None if model == 'density' else data['y']
            seed = int(data['seed'])
            # Only observed arrays enter the optimisers and sampler.
            freq_path = folder / 'frequentist.json'
            if freq_path.exists():
                obj = C.read(freq_path)
                for key in ['theta', 'starts', 'values']:
                    obj[key] = np.array(obj[key])
                frequency = Estimate(**obj)
            else:
                frequency = minimize(X, y, model, job['gamma'], cfg['optimizer'], seed=seed)
                C.dump(freq_path, asdict(frequency))
            position, diag = sample(X, y, model, job['gamma'], seed, cfg, folder / 'posterior', paper_cache=paper_cache)
            B = cfg[job['study']]['bootstrap_B']
            bs = bootstrap(X, y, model, job['gamma'], cfg['optimizer'], frequency.theta,
                           B, seed, folder / 'bootstrap') if B else None
            audit = score_fit(position, frequency, bs, X, y, model, cfg['scoring']['max_draws'])
            hb_ok = bool(diag['additional_diagnostics_pass'])
            freq_ok = bool(frequency.valid)
            bs_ok = bs is None or all(r['valid'] for r in bs['diagnostics'])
            base = dict(job)
            base.update(n=len(X), p=X.shape[1]-1 if model == 'regression' else X.shape[1],
                        observed_contamination=float(data['Z'].mean()), hb_valid=hb_ok,
                        frequentist_valid=freq_ok, bootstrap_valid=bs_ok,
                        paired_valid=hb_ok and freq_ok and bs_ok,
                        scale_floor=frequency.lower_scale_boundary,
                        bootstrap_floor_fraction=None if bs is None else float(np.mean([r['scale_floor'] for r in bs['diagnostics']])))
            # job already has n,p for the cost DGP; dict assignment avoids duplicate keywords.
            base.update(n=len(X), p=X.shape[1]-1 if model == 'regression' else X.shape[1])
            score_rows = []
            for method, scores in audit['scores'].items():
                inverse = np.argsort(audit['orders'][method])
                for i in range(len(X)):
                    score_rows.append(dict(method=method, observation=i, score=scores[i], rank=int(inverse[i]+1),
                                           central_selected=bool(inverse[i] < audit['counts'][method]), truth=int(data['Z'][i])))
            write_csv(folder / 'scores.csv', score_rows)
            C.save_npz(folder / 'audit_draws.npz', selected=audit['posterior_selected'].astype(np.uint8),
                       K=audit['posterior_K'], posterior_indices=audit['posterior_indices'])
            if job['study'] == 'direct':
                rows, other = evaluation.direct(data, model, frequency, position, diag, audit, cfg, folder,
                                                 base, paper_cache=paper_cache)
                write_csv(folder / 'metrics.csv', rows)
                write_csv(folder / 'cleaning.csv', other)
                files = ['metrics.csv', 'cleaning.csv']
            else:
                rows, other = evaluation.costs(data, audit, cfg, base)
                write_csv(folder / 'policies.csv', rows)
                write_csv(folder / 'downstream.csv', other)
                files = ['policies.csv', 'downstream.csv']
            diagnostics = dict(hb=diag, frequentist=asdict(frequency),
                               bootstrap_requested=B, bootstrap_completed=0 if bs is None else len(bs['diagnostics']),
                               bootstrap_invalid=0 if bs is None else sum(not r['valid'] for r in bs['diagnostics']))
            C.dump(folder / 'diagnostics.json', diagnostics)
            files += ['data.npz', 'inputs.json', 'frequentist.json', 'diagnostics.json', 'scores.csv', 'audit_draws.npz']
            C.dump(folder / 'result.json', dict(job=job, status='complete', fingerprint=fingerprint, base=base,
                                               artifacts={f:C.file_hash(folder/f) for f in files},
                                               seconds=time.time()-started))
            C.dump(folder / 'state.json', dict(job=job, status='complete', paired_valid=base['paired_valid']))
            print(f'DONE {job["id"]}: paired_valid={base["paired_valid"]}', flush=True)
            # Avoid retaining compiled targets from hundreds of distinct native fits.
            import jax
            jax.clear_caches()
            return 'complete'
        except Exception as exc:
            C.dump(folder / 'state.json', dict(job=job, status='error', error=str(exc), traceback=traceback.format_exc()))
            print(f'ERROR {job["id"]}: {exc}', file=sys.stderr, flush=True)
            return 'error'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', 'collect', 'plot', 'doctor'])
    parser.add_argument('--study', choices=['direct', 'costs', 'both'], default='both')
    parser.add_argument('--config')
    parser.add_argument('--output-dir', default='outputs/additional_experiments')
    parser.add_argument('--local-data-dir')
    parser.add_argument('--download-data', action='store_true')
    parser.add_argument('--paper-cache', help='Optional native cache root; contract-compatible fits only')
    parser.add_argument('--workers', type=int, default=1, help='External process shard count')
    parser.add_argument('--worker-index', type=int, default=0, help='Zero-based external shard index')
    parser.add_argument('--limit', type=int, help='Maximum jobs per requested study for this invocation')
    parser.add_argument('--job-id', help='Run one exact ID from plan')
    parser.add_argument('--finalize', action='store_true', help='Collect and plot after run (single worker only)')
    args = parser.parse_args(argv)
    root = Path(args.output_dir).resolve()
    studies = ['direct','costs'] if args.study == 'both' else [args.study]
    cfg = load_output_config(root, args.config) if args.command in ['collect','plot'] else C.load_config(args.config)
    if args.command == 'doctor':
        print(json.dumps(C.environment(), indent=2))
        missing = [k for k,v in C.environment()['packages'].items() if v is None]
        if missing:
            print('Install from repository root: python -m pip install -e ".[experiments]"')
        return int(bool(missing))
    if args.command == 'plan':
        plan = {study:C.jobs(cfg, study) for study in studies}
        print(json.dumps(dict(counts={s:len(v) for s,v in plan.items()}, jobs=plan), indent=2))
        return 0
    if args.command in ['collect','plot']:
        if not (root/'manifest.json').exists():
            parser.error('No manifest found in --output-dir')
        from .summary import collect
        for study in studies:
            collect(root, cfg, study)
            if args.command == 'plot':
                from .plotting import plot
                plot(root, cfg, study)
        return 0
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error('Need 0 <= --worker-index < --workers')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.finalize and args.workers != 1:
        parser.error('Finalize separately after all worker processes have finished')
    # Native inference itself enables x64 locally. Preserve original float32 DGPs.
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    fingerprint = initialize(root, cfg)
    errors = 0
    found = False
    for study in studies:
        selected = C.jobs(cfg, study)[args.worker_index::args.workers]
        if args.job_id:
            selected = [j for j in selected if j['id']==args.job_id]
        if args.limit:
            selected = selected[:args.limit]
        found |= bool(selected)
        for i, job in enumerate(selected):
            print(f'[{i+1}/{len(selected)}] {job["id"]}', flush=True)
            result = run_job(job, cfg, root, fingerprint, local_data_dir=args.local_data_dir,
                             download_data=args.download_data, paper_cache=args.paper_cache)
            errors += result=='error'
        if args.finalize:
            from .summary import collect
            from .plotting import plot
            collect(root, cfg, study)
            plot(root, cfg, study)
    if args.job_id and not found:
        parser.error('Requested job ID is absent from this config/shard')
    return 1 if errors else 0
