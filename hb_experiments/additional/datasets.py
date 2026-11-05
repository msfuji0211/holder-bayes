"""Original paper generators/splits, plus the explicitly separate cost DGP."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import save_npz,array_hash

def _scaler_arrays(s):return {'scaler_'+k:np.asarray(v) for k,v in s.items()}

def prepare(job,cfg,*,local_data_dir=None,download_data=False,dataset_cache=None):
 kwargs=dict(local_data_dir=local_data_dir,download_data=download_data,dataset_cache=dataset_cache)
 if job['family']!='cost_regression':
  import jax
  # Original simulation PRNG output precision must not inherit a user's inference setting.
  with jax.experimental.enable_x64(False):return _prepare(job,cfg,**kwargs)
 return _prepare(job,cfg,**kwargs)

def _prepare(job,cfg,*,local_data_dir=None,download_data=False,dataset_cache=None):
 family=job['family'];e=job['epsilon'];rep=job['rep'];base=cfg['costs']['seed'] if family=='cost_regression' else cfg['seed'];d=cfg['direct']
 if family=='density':
  from experiments.exp1_simulation.utils_exp1 import simulate_dataset
  seed=base+1000*rep+10*int(round(e*10));X,Z=simulate_dataset(d['n'],e,'A',seed)
  return dict(X=X,y=np.empty(0),Z=Z.astype(np.uint8),mu_true=np.zeros(X.shape[1]),sigma_true=np.ones(X.shape[1]),seed=np.array(seed),epsilon0=np.array(e))
 if family=='regression':
  from experiments.exp1_regression.utils_exp1_regression import simulate_regression_dataset,standardize_regression_data
  seed=base+1000*rep+10000*int(round(e*100))
  raw=simulate_regression_dataset(d['n'],d['n_test'],e,d['mu_out'],seed,x_dim=d['x_dim'],contamination_mode='response_homogeneous')
  X,y,Xtest,s=standardize_regression_data(raw['x_train'],raw['y_train'],raw['x_test'])
  return dict(X=np.asarray(X),y=np.asarray(y),X_test=np.asarray(Xtest),y_test=raw['y_test'],Z=raw['mask_outlier'].astype(np.uint8),
              beta_true=raw['beta_true'],sigma_true=np.array(raw['sigma_true']),seed=np.array(seed),epsilon0=np.array(e),**_scaler_arrays(s))
 if family=='benchmark':
  from experiments.exp1_regression.run_exp1_realdata_benchmark import load_benchmark_dataset
  from hb_experiments.native_run import validate_dataset
  from experiments.exp1_regression.utils_exp1_regression import (sample_train_test_with_split_cfg,kanamori_standardize_regression_data,
   make_realdata_split_seed,make_realdata_contamination_seed,make_realdata_fit_seed,contaminate_response)
  cache=Path(dataset_cache);cache.mkdir(parents=True,exist_ok=True);path=cache/(job['dataset']+'.npz')
  if path.exists():
   with np.load(path,allow_pickle=False) as z:xx,yy=validate_dataset(job['dataset'],z['X'],z['y'])
  else:
   if local_data_dir is None and not download_data:raise FileNotFoundError('Use --local-data-dir or --download-data for benchmark data')
   xx,yy=validate_dataset(job['dataset'],*load_benchmark_dataset(job['dataset'],local_data_dir=local_data_dir))
   save_npz(path,X=xx,y=yy)
  rawX,rawY,testX,testY=sample_train_test_with_split_cfg(xx,yy,make_realdata_split_seed(base,rep),split_sizes='ratio',train_ratio=.8,test_ratio=.2)
  X,yclean,Xtest,s=kanamori_standardize_regression_data(rawX,rawY,testX)
  shift=0. if e==0 else d['mu_out']
  if e:yc,Z=contaminate_response(np.asarray(yclean,float),e,shift,make_realdata_contamination_seed(base,rep,e,shift))
  else:yc=np.asarray(yclean,float);Z=np.zeros(len(yc),bool)
  # Match the original float32 likelihood coordinates after injection.
  return dict(X=np.asarray(X),y=yc.astype(np.float32),X_test=np.asarray(Xtest),y_test=testY,Z=Z.astype(np.uint8),
      seed=np.array(make_realdata_fit_seed(base,rep,e,shift,job['gamma'])),epsilon0=np.array(e),**_scaler_arrays(s))
 if family=='cost_regression':
  # Independent of gamma and shift; same covariates/noise/latent uniforms pair the conditions.
  n,p=job['n'],job['p'];seed=base+10_000*p+100*n+rep;rng=np.random.default_rng(seed)
  cov=.5**abs(np.arange(p)[:,None]-np.arange(p)[None,:]);x=rng.multivariate_normal(np.zeros(p),cov,size=n)
  leverage=rng.random(n)<.2;x*=np.where(leverage,3.,1.)[:,None]
  beta=np.ones(p)/np.sqrt(2.6*cov.sum());noise=rng.normal(size=n);u=rng.random(n);sign=rng.choice([-1.,1.],size=n)
  Z=u<e;raw_y=x@beta+noise+job['shift']*Z*sign
  xm=x.mean(axis=0);xs=x.std(axis=0);ym=raw_y.mean();ys=raw_y.std()
  X=np.c_[np.ones(n),(x-xm)/xs];y=(raw_y-ym)/ys
  return dict(X=X,y=y,Z=Z.astype(np.uint8),x_raw=x,y_raw=raw_y,beta_true=np.r_[0.,beta],sigma_true=np.array(1.),population_cov=2.6*cov,
       seed=np.array(seed),epsilon0=np.array(e),scaler_x_mean=xm,scaler_x_std=xs,scaler_y_mean=np.array(ym),scaler_y_std=np.array(ys))
 raise ValueError('Unknown dataset family')

def observed_hash(data):return {k:array_hash(data[k]) for k in ['X','y']}
