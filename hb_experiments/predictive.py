"""Generate paired predictive diagnostics from a new campaign's fit cache."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .common import ROOT, read, write, sha, safe_output
from .grid import relative
from . import predictive_diagnostics as core
from hbayes.paper_sampling import array_hash, _restore

CLEANING = {'NoCleaning':'NoCleaning', 'OracleCleaning':'OracleCleaning',
            'HBlower':'ProposedDPD_optimistic', 'HBcentral':'ProposedDPD',
            'HBupper':'ProposedDPD_pessimistic'}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def cache_samples(output, block, method, inputs):
    digest=block['fit_contract_sha256']
    folder=output/'cache/regression'/method/digest
    contract=read(folder/'contract.json')
    actual=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if actual!=digest or contract['input_hashes']!=[array_hash(np.asarray(v,dtype=np.float64)) for v in inputs]:
        raise ValueError('Predictive cache target or training observations differ: '+str(folder))
    if not block.get('diagnostics_pass'):
        raise ValueError('Resolve fit diagnostics before predictive analysis: '+str(folder))
    restored=_restore(folder/'posterior.npz')
    if restored is None:
        raise FileNotFoundError('Missing generated posterior cache: '+str(folder))
    draws=restored['samples_physical']
    if draws.shape[:2]!=(contract['chains'],contract['samples']):
        raise ValueError('Saved posterior shape differs from its run contract')
    return draws.reshape(-1,draws.shape[-1]),folder/'posterior.npz'


def aggregate(frame):
    numeric=[k for k in frame.select_dtypes(include=np.number) if k!='split']
    result=frame.groupby('method',sort=False)[numeric].agg(['mean','std','count']).reset_index()
    result.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in result.columns]
    return result


def run(output, *, dataset='cal_housing', gamma=.5, eps=.2, exclude_split=None):
    """Primary summary keeps every split. An explicit exclusion adds sensitivity only."""
    output=safe_output(output);config=read(ROOT/'configs/paper.json')
    if dataset not in config['datasets'] or gamma not in config['gammas'] or eps not in config['realdata_eps']:
        raise ValueError('Choose a dataset/gamma/epsilon in the paper grid')
    n=config['realdata_replicates']
    if exclude_split is not None and not 0<=exclude_split<n:
        raise ValueError('Excluded split must be a valid zero-based split index')
    target=output/'diagnostics/predictive'/dataset/f'gamma_{gamma:.3f}'/f'eps_{eps:.2f}'
    rows=[];comparisons=[];source_hashes={}
    def track(path):
        source_hashes[str(path.relative_to(output))]=sha(path)
    for rep in range(n):
        bench=output/'runs'/relative('benchmark',gamma,eps,rep,dataset)
        clean=output/'runs'/relative('cleaning',gamma,eps,rep,dataset)
        for folder,name in [(bench,'likelihood_inputs.npz'),(clean,'raw.npz')]:
            track(folder/name);track(folder/'metrics.json')
        raw=load_arrays(clean/'raw.npz');observed=load_arrays(bench/'likelihood_inputs.npz')
        metrics=read(clean/'metrics.json');initial=read(bench/'metrics.json')['dpd']
        if metrics.get('preprocessing')!='kanamori':
            raise ValueError('Predictive diagnosis expects the fixed clean-training scaler')
        for key in ['x_train_raw','y_train_raw_clean','x_test_raw','y_test_raw']:
            np.testing.assert_array_equal(raw[key],observed[key],err_msg='Benchmark and cleaning splits are not paired')
        X,Xt,scaler=core.fixed_scaler(raw)
        np.testing.assert_array_equal(observed['X'],X)
        cases=[('HBinitial',initial,'DPD',observed['X'],observed['y'])]
        for label,name in CLEANING.items():
            if name not in metrics['cleaning']:
                if label=='OracleCleaning' and eps==0:continue
                raise ValueError('Missing cleaning fit: '+name)
            removal=np.asarray(raw['removed_indices_'+name])
            if removal.ndim!=1 or removal.dtype.kind not in 'iu' or len(np.unique(removal))!=len(removal):
                raise ValueError('Invalid saved removal indices')
            if np.any(removal<0) or np.any(removal>=len(X)):
                raise ValueError('Removal index outside training data')
            keep=np.ones(len(X),dtype=bool);keep[removal]=False
            y=((raw['y_train_raw_contam']-scaler['y_mean'])/scaler['y_std']).astype(np.float32)
            cases.append((label,metrics['cleaning'][name],'NLL' if name=='NoCleaning' else 'DownstreamNLL',X[keep],y[keep]))
        pointwise={}
        for label,block,method,train,response in cases:
            draws,path=cache_samples(output,block,method,[train,response]);track(path)
            arrays,summary=core.diagnose(draws,Xt,raw['y_test_raw'],scaler)
            design=core.add_design_diagnostics(arrays,train,Xt)
            target.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(arrays).to_csv(target/f'split_{rep+1}_{label}_pointwise.csv',index=False)
            write(target/f'split_{rep+1}_{label}_summary.json',dict(**summary,**design))
            rows.append(dict(split=rep,method=label,**{k:v for k,v in summary.items() if isinstance(v,(int,float))},**design))
            pointwise[label]=arrays
        for label,arrays in pointwise.items():
            comparisons.append(dict(split=rep,method=label,baseline='NoCleaning',
                                    **core.pairwise_excess(arrays,pointwise['NoCleaning'])))
    # Inputs/cache contents must stay fixed throughout the analysis.
    for rel,digest in source_hashes.items():
        if sha(output/rel)!=digest:raise ValueError('Input changed during predictive analysis: '+rel)
    frame=pd.DataFrame(rows);frame.to_csv(target/'split_metrics.csv',index=False)
    aggregate(frame).to_csv(target/'all_splits_aggregate.csv',index=False)
    if exclude_split is not None:
        aggregate(frame[frame.split!=exclude_split]).to_csv(target/'sensitivity_aggregate.csv',index=False)
    report=dict(status='passed',dataset=dataset,gamma=gamma,eps=eps,fits=len(frame),
                primary_splits=n,primary_retains_all_splits=True,
                sensitivity_excluded_split=exclude_split,sensitivity_is_post_hoc=exclude_split is not None,
                all_retained_draws_and_test_observations=True,input_sha256=source_hashes,
                inference_run=False,paired_comparisons=comparisons)
    write(target/'report.json',report)
    return dict(status='passed',fits=len(frame),output=str(target),primary_splits=n,inference_run=False)
