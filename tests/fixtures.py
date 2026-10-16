"""Generate small numerical fixtures in a temporary directory; never paper results."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.special import expit
from hb_experiments.common import ROOT, read, write, sha
from hb_experiments.grid import jobs, METHOD_KEYS
from hb_experiments.figures import DENSITY, REGRESSION


def campaign(root):
    config = read(ROOT/'configs/paper.json')
    a = SimpleNamespace(output_dir=Path(root), gammas=config['gammas'], datasets=config['datasets'],
                        replicates=config['synthetic_replicates'], real_replicates=config['realdata_replicates'])
    for phase in ['synthetic', 'benchmark', 'cleaning']:
        for job in jobs(a, phase):
            e, g, rep = job['eps'], job['gamma'], job['rep']
            payload = dict(dataset=job['dataset'], gamma=g, eps_true=e, replicate=rep, rep=rep,
                           mu_out=0. if e == 0 else 6., n_train=100, n_test=40,
                           temperature_version='generated_test_fixture', contamination_mode='intercept_shift')
            def block(offset):
                return dict(rhat_max=1., ess_bulk_min=1000., ess_tail_min=1000., nuts_divergences=0,
                            diagnostics_pass=True, mu_bias_norm=.05+e+rep/10000+offset,
                            rmse_test=.9+e+rep/10000+offset, nlpd_test_mc=1.5+e+rep/10000+offset,
                            auprc=None if e == 0 else .98-offset,
                            eps_post_mean=e+.005+rep/100000, eps_ci95_cover=1.,
                            eps_ci95_low=max(0.,e-.03), eps_ci95_high=min(1.,e+.03),
                            removed_f1=.8-offset, removed_fraction=e)
            if phase == 'cleaning':
                payload['audits'] = {m:block(i/100) for i,m in enumerate(list(METHOD_KEYS)[:3])}
                methods = ['NoCleaning', 'IFCleaning', 'LOFCleaning', 'OneClassSVMCleaning']
                methods += [f'Proposed{m}{s}' for m in list(METHOD_KEYS)[:3] for s in ['', '_optimistic', '_pessimistic']]
                if e > 0: methods += ['RandomCleaning', 'OracleCleaning']
                payload['cleaning'] = {m:block(i/100) for i,m in enumerate(methods)}
            else:
                for i,(method,key) in enumerate(METHOD_KEYS.items()):
                    b=block(i/100)
                    if method == 'GammaDiv':
                        b.update(gamma_posterior_definition='nakagawa_hashimoto_monotone_gamma_v1', sigma_min=.05, eta=1.)
                    payload['standard' if method=='NLL' and job['kind']=='density' else key]=b
            write(a.output_dir/job['relative_dir']/'metrics.json', payload)
    rng=np.random.default_rng(551)
    for kind,rel in [('density',DENSITY),('regression',REGRESSION)]:
        n=80; x=rng.normal(size=(n,2)); mask=np.arange(n)>=64
        z=rng.normal(size=(120,5))*.12; z[:,-1]=np.log(4.)+rng.normal(size=120)*.15
        baseline=rng.normal(size=(120,4))*.12
        raw=dict(samples_dpd=z, samples_nll=baseline, p_out_dpd=np.where(mask,.9,.02), eps_samples_dpd=expit(-z[:,-1]))
        if kind=='density':
            x[mask]+=5;raw.update(x=x,y_true=mask)
        else:
            y=1.+x@np.array([1.,-1.5])+rng.normal(size=n); y[mask]+=6
            raw.update(x_train=x,y_train=y,beta_true=np.array([1.,1.,-1.5]),mask_outlier=mask)
        np.savez_compressed(a.output_dir/rel/'samples.npz', **raw)
    return a


def collect_fixture(root):
    from hb_experiments.collection import collect
    a=campaign(root);files=collect(a)
    write(a.output_dir/'workflow/collect.json',dict(complete=True,csv_sha256={name:sha(p) for name,p in files.items()}))
    return a


def predictive_campaign(output):
    """Create paired raw inputs and cache contracts for 5 splits x 6 fits."""
    import hashlib
    from hb_experiments.grid import relative
    from hb_experiments import predictive_diagnostics as core
    from hb_experiments.predictive import CLEANING
    from hbayes.paper_sampling import array_hash, _checkpoint
    output=Path(output);rng=np.random.default_rng(331)
    def fit(method,X,y):
        contract=dict(method=method,chains=4,samples=20,input_hashes=[array_hash(np.asarray(v,dtype=np.float64)) for v in [X,y]])
        digest=hashlib.sha256(__import__('json').dumps(contract,sort_keys=True).encode()).hexdigest()
        folder=output/'cache/regression'/method/digest;write(folder/'contract.json',contract)
        z=rng.normal(size=(4,20,5 if method=='DPD' else 4))*.1
        _checkpoint(folder/'posterior.npz',dict(samples_physical=z))
        return dict(fit_contract_sha256=digest,diagnostics_pass=True)
    for rep in range(5):
        x=rng.normal(size=(30,2));y=rng.normal(size=30);xt=rng.normal(size=(12,2));yt=rng.normal(size=12)
        raw=dict(x_train_raw=x,y_train_raw_clean=y,x_test_raw=xt,y_test_raw=yt,y_train_raw_contam=y.copy())
        X,Xt,scaler=core.fixed_scaler(raw);scaled=((y-scaler['y_mean'])/scaler['y_std']).astype(np.float32)
        benchmark=output/'runs'/relative('benchmark',.5,.2,rep,'cal_housing')
        clean=output/'runs'/relative('cleaning',.5,.2,rep,'cal_housing')
        initial=fit('DPD',X,scaled)
        write(benchmark/'metrics.json',dict(dpd=initial))
        np.savez_compressed(benchmark/'likelihood_inputs.npz',**{**raw,'X':X,'y':scaled,'X_test':Xt})
        blocks={}
        for i,(label,name) in enumerate(CLEANING.items()):
            removed=np.arange(i,dtype=np.int64);raw['removed_indices_'+name]=removed
            keep=np.ones(len(X),dtype=bool);keep[removed]=False
            blocks[name]=fit('NLL' if name=='NoCleaning' else 'DownstreamNLL',X[keep],scaled[keep])
        write(clean/'metrics.json',dict(preprocessing='kanamori',cleaning=blocks))
        np.savez_compressed(clean/'raw.npz',**raw)
