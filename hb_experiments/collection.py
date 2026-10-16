"""Aggregate this campaign's per-run metrics, with no saved reference data."""
import json
from pathlib import Path
from .grid import jobs, relative, blocks, diagnostic_issues


def collect(a):
    # Check the whole planned grid before creating any summary. A missing or
    # failed fit is an error, never an instruction to filter a replicate.
    for phase in ['synthetic', 'benchmark', 'cleaning']:
        for job in jobs(a, phase):
            path = a.output_dir/job['relative_dir']/'metrics.json'
            payload = json.loads(path.read_text())
            issues = diagnostic_issues(payload, job['kind'])
            if issues:
                raise ValueError(f'Unresolved metric diagnostics: {path}: {issues}')
    import numpy as np
    import pandas as pd
    for kind in ["regression", "density"]:
        for g in a.gammas:
            rows = []
            for job in jobs(a, "synthetic"):
                if job["kind"] != kind or job["gamma"] != g:
                    continue
                payload = json.loads((a.output_dir/job["relative_dir"]/"metrics.json").read_text())
                for method, block in blocks(payload, kind):
                    scalars = {k: v for k, v in block.items() if isinstance(v, (int, float)) or v is None}
                    rows.append({"scenario": "A", "eps_true": job["eps"], "gamma": g, "rep": job["rep"], "method": method, **scalars})
            frame = pd.DataFrame(rows)
            group_dir = a.output_dir/Path(relative(kind, g, 0., 0)).parts[0]
            frame.to_csv(group_dir/"summary_metrics.csv", index=False)
            group_cols = ["scenario", "eps_true", "method"]
            numeric = [c for c in frame.select_dtypes(include=np.number).columns if c not in ["eps_true", "gamma", "rep"]]
            agg = frame.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
            agg.columns = ["_".join(str(x) for x in c if x) if isinstance(c, tuple) else c for c in agg.columns]
            agg.to_csv(group_dir/"summary_agg.csv", index=False)
    from experiments.exp1_regression.collect_realdata_benchmark_summary import collect_rows
    frame = pd.DataFrame(collect_rows(str(a.output_dir/"realdata_benchmark")))
    frame.to_csv(a.output_dir/"realdata_benchmark/summary_metrics.csv", index=False)
    group = ["dataset", "gamma", "eps_true", "mu_out", "method"]
    numeric = [c for c in frame.select_dtypes(include=np.number).columns if c not in group+["replicate"]]
    agg = frame.groupby(group)[numeric].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(str(x) for x in c if x) if isinstance(c, tuple) else c for c in agg.columns]
    agg.to_csv(a.output_dir/"realdata_benchmark/summary_agg.csv", index=False)
    from experiments.exp1_regression.collect_realdata_cleaning_summary import collect_rows as cleaning_rows, write_summary
    clean, audit, seen = cleaning_rows(str(a.output_dir/"realdata_cleaning"))
    write_summary(clean, a.output_dir/"realdata_cleaning", "summary_runs.csv", "cleaning_method")
    write_summary(audit, a.output_dir/"realdata_cleaning", "summary_audits.csv", "method")
    expected = len(a.datasets)*len(a.gammas)*2*a.real_replicates
    if seen != expected:
        raise ValueError("Missing cleaning runs")
    return {str(p.relative_to(a.output_dir)): p for p in a.output_dir.rglob('summary_*.csv')}
