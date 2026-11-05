"""Call the uploaded repository's target and exact-MH backend unchanged."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import os
from pathlib import Path
import numpy as np
from .common import dump,save_npz

@contextmanager
def _environment(values):
 old={k:os.environ.get(k) for k in values}
 try:
  for k,v in values.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=str(v)
  yield
 finally:
  for k,v in old.items():
   if v is None:os.environ.pop(k,None)
   else:os.environ[k]=v

def fit(X,y,model,gamma,seed,cfg,output,*,paper_cache=None,likelihood_only=False):
 import jax
 import jax.numpy as jnp
 from hbayes.posteriors import (PriorsConfig,MVNPriorsConfig,GibbsConfig,MVNGibbsConfig,
  logprob_joint_regression_from_vector,logprob_joint_mvn_from_vector,logprob_theta_only_nll_regression_from_vector)
 from hbayes.experiment_temperature import covariance_temperature,response_temperature
 from hbayes.paper_sampling import fit as native_fit,job_context
 output=Path(output);s=cfg['sampler'];d=X.shape[1]
 pri=(MVNPriorsConfig if model=='density' else PriorsConfig)(sigma0=s['beta_prior_sd'],s_log_sigma=s['log_scale_prior_sd'])
 eta0,_=covariance_temperature(X,gamma) if model=='density' else response_temperature(y,gamma)
 eta=1. if likelihood_only else eta0*s['eta_multiplier']
 attempts=[];last=None
 counts=[s['draws'],*s['repair_draws']]
 for stage,draws in enumerate(dict.fromkeys(counts)):
  stage_dir=output/f'attempt_{stage:02d}';stage_dir.mkdir(parents=True,exist_ok=True)
  cache=Path(paper_cache).resolve() if paper_cache else output/'native_cache'
  env={'HBAYES_CACHE_ROOT':str(cache.resolve()),'HBAYES_MH_WARMUP':s['warmup'],'HBAYES_MH_SAMPLES':draws,'HBAYES_MH30_ROOT':None}
  with _environment(env),job_context(),jax.experimental.enable_x64():
   inputs=[np.asarray(X)] if model=='density' else [np.asarray(X),np.asarray(y)]
   arr=[jnp.asarray(a,dtype=jnp.float64) for a in inputs]
   if likelihood_only:
    if model!='regression':raise ValueError('NLL refit supports regression only')
    target=logprob_theta_only_nll_regression_from_vector(*arr,pri);method='NLL';floor=None;phi=None
    names=[f'beta_{i}' for i in range(d)]+['log_scale']
   else:
    gcfg=(MVNGibbsConfig if model=='density' else GibbsConfig)(gamma=gamma,eta=eta,holder_phi='dpd',holder_kappa=1.,priors=pri)
    target=logprob_joint_mvn_from_vector(arr[0],gcfg) if model=='density' else logprob_joint_regression_from_vector(*arr,gcfg)
    method='DPD';floor=None if model=='density' else .05;phi='dpd'
    names=([f'mu_{i+1}' for i in range(d)]+[f'log_scale_{i+1}' for i in range(d)] if model=='density' else [f'beta_{i}' for i in range(d)]+['log_scale'])+['rho']
   raw=native_fit(target,inputs,method=method,model=model,seed=int(seed),chains=s['chains'],num_warmup=s['warmup'],num_samples=draws,
       target_accept=s['target_accept'],init_step=s['initial_step'],priors=pri,eta=eta,gamma=gamma,phi=phi,kappa=1.,floor=floor,names=names)
  position=np.asarray(raw['samples_chains'],float)
  diag={k:v for k,v in raw.items() if k not in ['samples','samples_chains']}
  finite=all(isinstance(diag.get(k),(float,int,np.number)) and np.isfinite(diag[k]) for k in ['rhat_max','ess_bulk_min','ess_tail_min'])
  gate=bool(finite and diag['rhat_max']<=cfg['diagnostics']['rhat_max'] and min(diag['ess_bulk_min'],diag['ess_tail_min'])>=cfg['diagnostics']['ess_min'])
  if likelihood_only:gate=gate and diag.get('nuts_divergences')==0
  diag.update(additional_diagnostics_pass=gate,eta=float(eta),eta_default=float(eta0),priors=asdict(pri),draws_per_chain=draws,
              probability_interpretation='generalised posterior audit membership; not latent contamination probability')
  save_npz(stage_dir/'samples.npz',position=position);dump(stage_dir/'diagnostics.json',diag)
  attempts.append({'stage':stage,'draws':draws,'diagnostics_pass':gate,'rhat_max':diag['rhat_max'],'ess_bulk_min':diag['ess_bulk_min'],'ess_tail_min':diag['ess_tail_min']})
  last=(position,diag)
  if gate:break
 dump(output/'attempts.json',attempts)
 return last
