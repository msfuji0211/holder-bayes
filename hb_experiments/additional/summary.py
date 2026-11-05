"""Complete-grid accounting and paired, replicate-blocked summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
from . import common as C
from .runner import write_csv, verify_complete


def read_table(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def grouped(frame, keys, metrics):
    if frame.empty:
        return []
    keys = [k for k in keys if k in frame.columns]
    metrics = [m for m in metrics if m in frame.columns]
    rows = []
    for labels, group in frame.groupby(keys, dropna=False, sort=True):
        labels = labels if isinstance(labels, tuple) else (labels,)
        for metric in metrics:
            v = pd.to_numeric(group[metric], errors='coerce').dropna().to_numpy()
            rows.append(dict(zip(keys, labels), metric=metric, mean=float(v.mean()) if len(v) else None,
                             sd=float(v.std(ddof=1)) if len(v)>1 else None, n=len(v),
                             rows=len(group)))
    return rows


def paired_costs(frame, cfg):
    """FoD - comparator; same dataset, costs, policy and replicate are paired."""
    if frame.empty:
        return [], [], []
    keys = ['id','rule','false_remove_cost','miss_cost','review_cost','review_error_rate']
    metrics = ['loss','false_removed','missed','reviewed']
    fod = frame[frame.method=='FoD']
    differences = []
    for method in frame.method.unique():
        if method == 'FoD':
            continue
        other = frame[frame.method==method]
        merged = fod.merge(other, on=keys, suffixes=('', '_other'), validate='one_to_one')
        for _, r in merged.iterrows():
            for metric in metrics:
                differences.append(dict(**{k:r[k] for k in keys}, comparator=method, metric=metric,
                                        gamma=r.gamma, n=r.n, p=r.p, shift=r['shift'], epsilon=r.epsilon, rep=r.rep,
                                        paired_valid=bool(r.paired_valid and r.paired_valid_other),
                                        difference=float(r[metric]-r[metric+'_other'])))
    if not differences:
        return [], [], []
    diff = pd.DataFrame(differences)
    group_keys = ['comparator','metric','rule','gamma','false_remove_cost','miss_cost','review_cost','review_error_rate']
    # Each replicate shares random streams across the whole condition grid.
    # First average a COMPLETE grid within a replicate; then resample replicate blocks.
    contaminated = diff[(diff.epsilon>0)&(diff['shift']>0)&diff.paired_valid]
    expected = len(cfg['costs']['sample_sizes'])*len(cfg['costs']['dimensions'])*len(cfg['costs']['shifts'])
    block_rows = []
    for labels, group in contaminated.groupby(group_keys+['rep'], dropna=False):
        count = len(group[['n','p','shift','epsilon']].drop_duplicates())
        if count == expected:
            block_rows.append(dict(zip(group_keys+['rep'],labels), difference=float(group.difference.mean()), conditions=count))
    blocks = pd.DataFrame(block_rows)
    macro = []
    if not blocks.empty:
        for labels, group in blocks.groupby(group_keys, dropna=False):
            v=group.difference.to_numpy();rng=np.random.default_rng(90261104)
            means=v[rng.integers(0,len(v),size=(4000,len(v)))].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975]) if len(v)>1 else (np.nan,np.nan)
            macro.append(dict(zip(group_keys,labels), difference=float(v.mean()), ci_low=lo,ci_high=hi,
                              n_complete_replicates=len(v), expected_replicates=cfg['costs']['replicates'],
                              expected_conditions=expected, interval='percentile bootstrap of complete replicate blocks'))
    return differences, block_rows, macro


def collect(root, cfg, study):
    root=Path(root);folder=root/study;summary=folder/'summary';summary.mkdir(parents=True,exist_ok=True)
    manifest=C.read(root/'manifest.json');fingerprint=C.digest(manifest)
    status=[];tables={name:[] for name in (['metrics','cleaning'] if study=='direct' else ['policies','downstream'])}
    for job in C.jobs(cfg,study):
        place=folder/'jobs'/job['id'];row=dict(job,status='missing',paired_valid=False)
        try:
            if verify_complete(place,fingerprint):
                receipt=C.read(place/'result.json');row.update(receipt['base']);row['status']='complete'
                for name in tables:
                    part=read_table(place/(name+'.csv'))
                    if not part.empty:tables[name].append(part)
            elif (place/'state.json').exists():
                state=C.read(place/'state.json');row.update(status=state['status'],error=state.get('error'))
        except Exception as exc:
            row.update(status='corrupt',error=str(exc))
        status.append(row)
    write_csv(summary/'completion.csv',status)
    frame_status=pd.DataFrame(status,columns=list(dict.fromkeys([k for r in status for k in r]+['status','paired_valid'])))
    report=dict(study=study,expected=len(status),complete=int((frame_status.status=='complete').sum()),
                paired_valid=int(frame_status.paired_valid.sum()),
                status_counts=frame_status.status.value_counts().to_dict(),
                note='Raw tables retain diagnostic failures. Summaries use common paired-valid fits; see completion.csv for exclusions.')
    C.dump(summary/'completion.json',report)
    metrics_direct=['epsilon_estimate','epsilon_abs_error','AP','AP_rank','AUC','F1','precision','recall',
                    'selected','false_positive','false_negative','location_error','scale_error',
                    'beta_error','sigma_error','test_RMSE','test_MAE']
    for name,parts in tables.items():
        data=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        write_csv(summary/(name+'_all.csv'),data.to_dict('records'))
        if data.empty:
            write_csv(summary/(name+'_summary.csv'),[])
            if name=='policies':
                for suffix in ['paired_differences','paired_blocks','macro_paired']:
                    write_csv(summary/(suffix+'.csv'),[])
            continue
        valid=data[data.paired_valid.astype(bool)].copy()
        if name in ['cleaning','downstream']:
            # Refit failures exclude the WHOLE replicate from a downstream paired comparison.
            bad=set(data.loc[data.refit_failed.astype(bool),'id'])
            valid=valid[~valid.id.isin(bad)]
        if study=='direct':
            keys=['family','dataset','gamma','epsilon','method']
            if name=='metrics':keys+=['metric_group','budget_fraction'] if 'budget_fraction' in valid.columns else ['metric_group']
            else:keys+=['refit']
            measures=metrics_direct
        else:
            keys=['n','p','shift','epsilon','gamma','method','rule']
            if name=='policies':
                keys+=['false_remove_cost','miss_cost','review_cost','review_error_rate']
                measures=['loss','removed','reviewed','retained','false_removed','missed']
            else:measures=['clean_population_RMSE','beta_error','effective_n']
        write_csv(summary/(name+'_summary.csv'),grouped(valid,keys,measures))
        if name=='policies':
            diffs,blocks,macro=paired_costs(data,cfg)
            write_csv(summary/'paired_differences.csv',diffs)
            write_csv(summary/'paired_blocks.csv',blocks)
            write_csv(summary/'macro_paired.csv',macro)
    print(f'{study}: {report["complete"]}/{report["expected"]} complete; {report["paired_valid"]} paired-valid',flush=True)
    return report
