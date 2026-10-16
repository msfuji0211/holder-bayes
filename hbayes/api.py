"""Public interfaces for Hölder–Bayes fitting and observation auditing.

The target builders and sampler are shared with the paper implementation.
JAX is imported only when fitting; inspecting saved results needs NumPy/SciPy.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass,field
from pathlib import Path
from threading import RLock
import csv,os
import numpy as np
from scipy.special import expit

from .experiment_temperature import covariance_temperature,response_temperature

_LOCK=RLock()
_PHIS={"dpd":("DPD",1.),"exponential_saturated":("ExpSatDPD",.5),
       "rational_saturated":("RatSatDPD",.25)}

@dataclass(frozen=True)
class SamplerConfig:
    """Per-chain settings. Rejected MH states remain in the retained draws."""
    chains:int=4
    warmup:int=4000
    draws:int=4000
    seed:int=42
    target_accept:float=.99
    initial_step:float=.1

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in [self.chains,self.warmup,self.draws,self.seed]):
            raise ValueError("chains, warmup, draws, and seed must be integers")
        if self.chains<2 or min(self.warmup,self.draws)<20 or not 0<=self.seed<2**32:
            raise ValueError("Use at least two chains, at least 20 warmup/draws, and a uint32 seed")
        if not 0<self.target_accept<1 or not np.isfinite(self.initial_step) or self.initial_step<=0:
            raise ValueError("Invalid acceptance target or initial step")

def _matrix(x,name):
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x[:,None]
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError(name+" must be a finite matrix with at least two rows")
    return x

def _regression_inputs(X,y):
    X=_matrix(X,"X");y=np.asarray(y,dtype=float)
    if y.shape!=(len(X),) or not np.isfinite(y).all():raise ValueError("y must be a finite vector matching X")
    xm=X.mean(axis=0);xs=X.std(axis=0);xs=np.where(xs<1e-9,1.,xs)
    ym=float(y.mean());ys=float(y.std());ys=1. if ys<1e-9 else ys
    design=np.hstack([np.ones((len(X),1)),(X-xm)/xs]).astype(np.float32)
    return design,((y-ym)/ys).astype(np.float32),dict(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)

def audit_probabilities(samples,inputs,model,*,max_draws=400,seed=42):
    """Posterior frequency of bottom-K likelihood membership, K=round(n*epsilon).

    `samples` uses physical coordinates [..., location/coefficient, log_scale,
    rho]. Regression inputs are the design matrix and response passed to the
    likelihood, including the intercept. Scores are relative to this batch.
    """
    z=np.asarray(samples,dtype=float)
    if z.ndim not in [2,3] or not np.isfinite(z).all():raise ValueError("Invalid posterior array")
    z=z.reshape(-1,z.shape[-1]);X=_matrix(inputs[0],"likelihood inputs")
    if len(z)==0 or max_draws<1:raise ValueError("At least one scoring draw is required")
    if len(z)>max_draws:
        idx=np.sort(np.random.default_rng(seed).choice(len(z),size=max_draws,replace=False));z=z[idx]
    d=X.shape[1];n=len(X)
    if model=="density":
        if z.shape[1]!=2*d+1:raise ValueError("Density sample dimension differs from the data")
        ls=z[:,d:2*d];residual=(X[None]-z[:,:d,None].transpose(0,2,1))/np.exp(ls[:,None,:])
        ll=-.5*np.sum(np.log(2*np.pi)+2*ls[:,None,:]+residual**2,axis=2)
    elif model=="regression":
        y=np.asarray(inputs[1],dtype=float)
        if y.shape!=(n,) or not np.isfinite(y).all() or z.shape[1]!=d+2:raise ValueError("Regression inputs and samples differ")
        ls=z[:,d,None];residual=(y[None]-z[:,:d]@X.T)/np.exp(ls)
        ll=-.5*np.log(2*np.pi)-ls-.5*residual**2
    else:raise ValueError("model must be density or regression")
    k=np.clip(np.round(expit(-z[:,-1])*n).astype(int),0,n);counts=np.zeros(n,dtype=int)
    for values,budget in zip(ll,k):counts[np.argsort(values)[:budget]]+=1
    return counts/len(z)

@dataclass
class FitResult:
    """Full retained physical draws, diagnostics, and batch-relative audit scores."""
    model:str
    samples:np.ndarray
    outlier_probability:np.ndarray
    diagnostics:dict
    metadata:dict=field(default_factory=dict)
    scaler:dict=field(default_factory=dict)

    @property
    def epsilon_samples(self):return expit(-self.samples[...,-1])

    @property
    def location_samples(self):
        if self.model!="density":raise ValueError("Location draws apply to density fits")
        return self.samples[...,:self.metadata["dimension"]]

    @property
    def coefficient_samples(self):
        if self.model!="regression":raise ValueError("Coefficient draws apply to regression fits")
        d=self.metadata["dimension"];beta=self.samples[...,:d].copy()
        beta[...,1:]*=self.scaler["y_std"]/self.scaler["x_std"]
        beta[...,0]=self.scaler["y_mean"]+self.scaler["y_std"]*beta[...,0]-beta[...,1:]@self.scaler["x_mean"]
        return beta

    @property
    def scale_samples(self):
        d=self.metadata["dimension"]
        if self.model=="density":return np.exp(self.samples[...,d:2*d])
        return self.scaler["y_std"]*np.exp(self.samples[...,d])

    def predict(self,X):
        """Posterior mean response in original units, including the intercept."""
        if self.model!="regression":raise ValueError("Prediction applies to regression fits")
        X=np.asarray(X,dtype=float)
        if X.ndim==1:X=X[:,None]
        if X.ndim!=2 or X.shape[1]!=self.metadata["dimension"]-1 or not np.isfinite(X).all():
            raise ValueError("Prediction covariates have the wrong shape or contain non-finite values")
        beta=self.coefficient_samples.mean(axis=(0,1));return beta[0]+X@beta[1:]

    def summary(self):
        """Parameter diagnostics are retained even when their gate is not passed."""
        return dict(model=self.model,chains=self.samples.shape[0],draws_per_chain=self.samples.shape[1],
                    epsilon_mean=float(self.epsilon_samples.mean()),
                    epsilon_ci95=np.quantile(self.epsilon_samples,[.025,.975]).tolist(),
                    diagnostics=self.diagnostics,configuration=self.metadata)

    def save(self,directory):
        from .paper_sampling import atomic_json,atomic_npz
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        arrays=dict(samples_physical=self.samples,epsilon=self.epsilon_samples,scale=self.scale_samples,
                    outlier_probability=self.outlier_probability)
        if self.model=="density":arrays["location"]=self.location_samples
        else:arrays.update(coefficients=self.coefficient_samples,**{f"scaler_{k}":v for k,v in self.scaler.items()})
        atomic_npz(directory/"posterior_samples.npz",**arrays)
        atomic_json(directory/"summary.json",self.summary())
        with (directory/"audit_scores.csv").open("w",newline="") as f:
            writer=csv.writer(f);writer.writerow(["observation","outlier_probability"])
            writer.writerows(enumerate(self.outlier_probability))
        return directory

@contextmanager
def _sampler_context(cache,sampler):
    from .paper_sampling import job_context
    updates=dict(HBAYES_CACHE_ROOT=str(Path(cache).resolve()),HBAYES_MH_WARMUP=str(sampler.warmup),
                 HBAYES_MH_SAMPLES=str(sampler.draws),HBAYES_MH30_ROOT=None)
    with _LOCK:
        previous={k:os.environ.get(k) for k in updates}
        try:
            for k,v in updates.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
            with job_context():yield
        finally:
            for k,v in previous.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v

def _sample(model,inputs,*,gamma,eta,phi,kappa,sampler,cache_dir,priors):
    try:
        import jax
        import jax.numpy as jnp
        from .posteriors import (PriorsConfig,MVNPriorsConfig,GibbsConfig,MVNGibbsConfig,
            logprob_joint_mvn_from_vector,logprob_joint_regression_from_vector)
    except ModuleNotFoundError as exc:
        raise RuntimeError('Install inference dependencies: python -m pip install -e ".[inference]"') from exc
    from .paper_sampling import fit
    method=_PHIS[phi][0];d=inputs[0].shape[1]
    priors=priors if priors is not None else (MVNPriorsConfig() if model=="density" else PriorsConfig())
    names=([f"mu_{i+1}" for i in range(d)]+[f"log_scale_{i+1}" for i in range(d)] if model=="density"
           else [f"beta_{i}" for i in range(d)]+["log_scale"])+["rho"]
    with _sampler_context(cache_dir,sampler),jax.experimental.enable_x64():
        arrays=[jnp.asarray(a,dtype=jnp.float64) for a in inputs]
        cfg=(MVNGibbsConfig if model=="density" else GibbsConfig)(gamma=gamma,eta=eta,holder_phi=phi,holder_kappa=kappa,priors=priors)
        target=logprob_joint_mvn_from_vector(arrays[0],cfg) if model=="density" else logprob_joint_regression_from_vector(*arrays,cfg)
        return fit(target,inputs,method=method,model=model,seed=sampler.seed,chains=sampler.chains,
                   num_warmup=sampler.warmup,num_samples=sampler.draws,target_accept=sampler.target_accept,
                   init_step=sampler.initial_step,priors=priors,eta=eta,gamma=gamma,phi=phi,kappa=kappa,
                   floor=None if model=="density" else .05,names=names)

def _fit(model,inputs,*,gamma,phi,kappa,sampler,cache_dir,output_dir,priors,scaler=None):
    if phi not in _PHIS:raise ValueError("Unknown phi; choose "+", ".join(_PHIS))
    kappa=_PHIS[phi][1] if kappa is None else float(kappa)
    if not np.isfinite(kappa) or kappa<=0:raise ValueError("kappa must be positive and finite")
    sampler=SamplerConfig() if sampler is None else sampler
    if not isinstance(sampler,SamplerConfig):raise TypeError("sampler must be SamplerConfig")
    eta,_=(covariance_temperature(inputs[0],gamma) if model=="density" else response_temperature(inputs[1],gamma))
    cache_dir=Path(cache_dir) if cache_dir is not None else Path(output_dir)/"cache" if output_dir is not None else Path("results/cache")
    raw=_sample(model,inputs,gamma=gamma,eta=eta,phi=phi,kappa=kappa,sampler=sampler,cache_dir=cache_dir,priors=priors)
    draws=np.asarray(raw["samples_chains"],dtype=float);d=inputs[0].shape[1]
    expected=(sampler.chains,sampler.draws,2*d+1 if model=="density" else d+2)
    if draws.shape!=expected or not np.isfinite(draws).all():raise ValueError("Backend returned invalid retained draws")
    scores=audit_probabilities(draws,inputs,model,seed=sampler.seed)
    result=FitResult(model,draws,scores,{k:v for k,v in raw.items() if k not in ["samples","samples_chains"]},
        dict(gamma=float(gamma),eta=float(eta),phi=phi,kappa=kappa,dimension=d,n_observations=len(inputs[0]),
             scoring_draws=min(400,sampler.chains*sampler.draws),seed=sampler.seed),scaler or {})
    if output_dir is not None:result.save(output_dir)
    return result

def fit_density(x,*,gamma=.5,phi="dpd",kappa=None,sampler=None,cache_dir=None,output_dir=None,priors=None):
    """Fit the diagonal-Gaussian Hölder–Bayes model in original coordinates."""
    x=_matrix(x,"x")
    return _fit("density",[x],gamma=gamma,phi=phi,kappa=kappa,sampler=sampler,cache_dir=cache_dir,output_dir=output_dir,priors=priors)

def fit_regression(X,y,*,gamma=1.,phi="dpd",kappa=None,sampler=None,cache_dir=None,output_dir=None,priors=None):
    """Fit regression with an intercept and training-only standardisation.

    X contains covariates without an intercept column. Returned coefficients,
    scales and predictions use the original units; physical retained positions
    keep the likelihood coordinates. The paper's .05 response-scale floor is
    applied in standardised coordinates.
    """
    X,y,scaler=_regression_inputs(X,y)
    return _fit("regression",[X,y],gamma=gamma,phi=phi,kappa=kappa,sampler=sampler,cache_dir=cache_dir,output_dir=output_dir,priors=priors,scaler=scaler)
